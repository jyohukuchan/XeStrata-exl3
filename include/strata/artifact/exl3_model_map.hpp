// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Explicit Qwen EXL3 source-to-engine mapping metadata.
#pragma once

#include "strata/artifact/exl3.hpp"
#include "strata/artifact/exl3_raw.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::artifact::exl3 {

enum class ModelFamily { qwen35, qwen38, hyv3, lfm2moe };

enum class ProjectionKind {
    conv_in, conv_out, dense_gate, dense_up, dense_down,
    gdn_qkv, gdn_z, gdn_out, gdn_alpha, gdn_beta,
    qsa_q, qsa_k, qsa_v, qsa_out, qsa_indexer_q, qsa_indexer_k,
    router, shared_scalar, shared_gate, shared_up, shared_down,
    dense_gate, dense_up, dense_down,
    expert_gate, expert_up, expert_down,
    hc_norm, hc_down, hc_up, hc_inject,
    mtp_fc,
    embedding, lm_head, final_norm,
    auxiliary,
};

enum class MappingTransform {
    identity,
    raw_rows,
    output_slice,
};

struct ModelGeometry {
    int64_t hidden_size = 0;
    int64_t vocab_size = 0;
    int64_t layers = 0;
    int64_t mtp_layers = 0;
    bool dedicated_embedding = false;
    int64_t experts = 0;
    int64_t experts_per_token = 0;
    int64_t moe_intermediate = 0;
    int64_t shared_intermediate = 0;
    int64_t q_heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0;
    int64_t linear_k_heads = 0;
    int64_t linear_v_heads = 0;
    int64_t linear_k_dim = 0;
    int64_t linear_v_dim = 0;
    int64_t indexer_heads = 0;
    int64_t indexer_kv_heads = 0;
    int64_t indexer_head_dim = 0;
    int64_t full_attention_interval = 0;
    int64_t dense_lead = 0;          ///< hyv3: leading dense-MLP layers
    int64_t dense_intermediate = 0;  ///< hyv3 dense MLP / lfm2moe leading dense FFN width
    int64_t dense_layers = 0;         ///< lfm2moe: leading dense FFN layers
    int64_t shortconv_l_cache = 0;    ///< lfm2moe: the short-conv window length
    int64_t hc_count = 0;
    int64_t hc_rank = 0;
};

struct OutputSlice {
    int64_t source_offset = 0;
    int64_t target_offset = 0;
    int64_t elements = 0;
};

/// How one physical tensor axis is placed in the engine's canonical axis.
/// `permutation` is target-to-source.  For a block map it contains the full
/// element permutation while `block_permutation` gives the local block map;
/// `block_offset` locates that mapped region in the axis.  Elements inside a
/// block stay in their original order. `slices` describes disjoint
/// source/target ranges and is useful when one fused source output is split
/// into canonical projections.
enum class AxisTransform {
    identity,
    explicit_permutation,
    h128_block_permutation,
    slices,
    raw_rows,
};

struct AxisMapping {
    AxisTransform transform = AxisTransform::identity;
    int64_t source_size = 0;
    int64_t target_size = 0;
    int64_t block_size = 0;
    int64_t block_offset = 0;
    std::vector<int64_t> permutation;
    std::vector<int64_t> block_permutation;
    std::vector<OutputSlice> slices;
};

/// One source tensor projected into one engine canonical row. `source_*` are
/// HF/EXL3 matrix dimensions [input, output] after the converter's normal
/// Linear orientation; raw safetensors retain their physical shape in
/// `raw_shape`. Slices are output-column ranges, never a guessed
/// dequantize/requantize permutation.
struct ProjectionMapping {
    ProjectionKind kind = ProjectionKind::auxiliary;
    MappingTransform transform = MappingTransform::identity;
    ValueTransform value_transform = ValueTransform::identity;
    int64_t layer = -1;
    int64_t expert = -1;
    std::string source_name;
    std::string canonical_name;
    int64_t source_k = 0;
    int64_t source_n = 0;
    int64_t target_k = 0;
    int64_t target_n = 0;
    int64_t raw_rows = 0;
    int64_t raw_cols = 0;
    std::string raw_dtype;
    std::vector<int64_t> raw_shape;
    /// Physical raw axis -> canonical axis order.  Matrices and embeddings
    /// stored as [rows, columns] use {1, 0}; vectors use {0}.
    std::vector<int64_t> raw_to_canonical_axes;
    /// Matrix axes in canonical [input, output] order.  The raw tensor axes
    /// are also retained in `raw_axes` in their physical order, so a loader
    /// can represent raw alpha/beta row permutations and multidimensional
    /// convolution/A/dt tensors without guessing their layout.
    AxisMapping input_axis;
    AxisMapping output_axis;
    std::vector<AxisMapping> raw_axes;
    std::vector<OutputSlice> output_slices;
};

struct ModelMap {
    ModelFamily family = ModelFamily::qwen38;
    ModelGeometry geometry;
    std::string source_prefix;
    std::vector<ProjectionMapping> projections;
    std::vector<std::string> unsupported;

    bool supported() const { return unsupported.empty(); }
    bool require_supported(std::string& err) const;
};

/// Discover and validate the explicit Qwen3.5/Qwen3.8 mapping (an LFM2 MoE archive is recognized, its
/// geometry recorded, and the engine pieces it still lacks listed in `unsupported`) from the
/// archive's config, EXL3 descriptors, and raw tensor headers. This reads no
/// weight payloads and returns a useful partial map with `unsupported` reasons
/// for recognized but not yet engine-compatible variants.
bool build_model_map(const Archive& archive, ModelMap& result, std::string& err, bool include_mtp = false);

/// Build the single Qwen3.8 MTP decoder map from an already validated trunk
/// map.  The MTP layer keeps its packed/raw component identities and uses a
/// one-layer `mtp.layers` source prefix; the final combine-less mixer is
/// consumed by the existing source-prefix-derived WeightSet path.
bool build_qwen38_mtp_map(const Archive& archive, const ModelMap& trunk, ModelMap& result, std::string& err);

/// Build the pinned Qwen V-head reorder independently of an archive. The
/// returned element permutation is target-to-source and the block map is
/// target-block-to-source-block. This is also useful to validate a loader's
/// raw fallback path without reading model payloads.
AxisMapping make_v_head_axis(int64_t total, int64_t offset, int64_t key_heads, int64_t value_heads,
                             int64_t head_dim);

const char* projection_kind_name(ProjectionKind kind);
const char* model_family_name(ModelFamily family);
const char* value_transform_name(ValueTransform transform);

}  // namespace strata::artifact::exl3
