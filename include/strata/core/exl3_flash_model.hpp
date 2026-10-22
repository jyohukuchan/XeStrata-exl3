// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Explicit Qwen3.8 EXL3 Flash-model ownership and stage preparation.
#pragma once

#include "strata/artifact/exl3_model_map.hpp"
#include "strata/artifact/exl3_model_spec.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/weights.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace strata::artifact::exl3 {
class Archive;
}

namespace strata::kernels {
class PleTable;
}

namespace strata::core {

class Exl3Dense;
class Exl3ExpertRuntime;
class Exl3ExpertSource;
class Exl3Ple;
class Exl3WeightSet;
class ExpertPool;
class NativeEmbed;

/// Startup policy for the first EXL3 Flash coordinator.
///
/// This deliberately describes one autoregressive token at a time.  MTP,
/// vision, and batched/prefill execution need their own graph contracts and
/// are rejected by open() until their callers register those contracts.
struct Exl3FlashModelOptions {
    std::vector<int> stage_devices;       ///< explicit runtime ordinals, one per layer stage
    std::vector<int64_t> stage_layers;    ///< explicit layer counts per stage, e.g. 18,18,12; empty: the even split
    int pool_workers = 0;                 ///< workers per stage; 0 preserves the automatic physical-core default
    int64_t max_rows = 1;                 ///< persistent projection/expert row capacity
    int64_t max_cells = 0;                ///< 0: config.max_position_embeddings
    int64_t routed_k = 0;                 ///< 0: text_config.num_experts_per_tok
    bool include_mtp = false;             ///< currently rejected for Qwen3.8
    bool vision = false;                  ///< currently rejected
    bool allow_cpu_expert_miss = true;    ///< misses use Exl3ExpertSource + ExpertPool
    std::string embedding_name;            ///< empty: ModelMap's HF embedding prefix
    std::string ple_sidecar;              ///< empty: discover the specialized EXL3 sidecar
};

/// Non-owning view of one stage.  The coordinator owns every pointer for its
/// lifetime; a caller must destroy captured graphs before close().  `table`
/// contains only this stage's canonical rows, while `experts` can classify
/// all layers and has resident cache entries only for this stage's range.
struct Exl3FlashStageView {
    int device = -1;
    int64_t layer_begin = 0;
    int64_t layer_end = 0;
    WeightTable* table = nullptr;
    Exl3WeightSet* weights = nullptr;
    Exl3ExpertSource* expert_source = nullptr;
    Exl3ExpertRuntime* expert_runtime = nullptr;
};

/// The layer range one stage owns.  The layers are split as evenly as the count allows, so every
/// stage gets at least one layer as long as there are no more stages than layers; open() refuses
/// that case, and an empty range is what the helper answers for it.
struct Exl3FlashStageRange {
    int64_t layer_begin = 0;
    int64_t layer_end = 0;
};

/// The range of stage `stage_index` of `stage_count` over `layer_count` layers.  The split is the
/// same for every layer count and stage count, and it is the only place the split is written down.
Exl3FlashStageRange exl3_flash_stage_range(int64_t layer_count, int64_t stage_count, int64_t stage_index);

/// The stage ranges of one split: every stage's layers, contiguous and in order.
///
/// `counts` gives each stage its layer count (`--stage-layers 18,18,12`); empty takes the even
/// split of `exl3_flash_stage_range`.  A split where a stage gets no layer, or where the counts do
/// not cover the model's layers exactly, is refused with the reason in `err`, because the stage
/// that holds a layer decides where that layer's state lives and the last stage holds the head.
bool exl3_flash_stage_ranges(int64_t layer_count, int64_t stage_count, const std::vector<int64_t>& counts,
                             std::vector<Exl3FlashStageRange>& ranges, std::string& err);

/// The stage that runs the PLE layer: the PLE's HOME STAGE.
///
/// The PLE block belongs to one layer, and the layer split gives that layer to exactly one stage.
/// The home stage holds the PLE's device half - the key and value projections' owners, the
/// gathered row, the preprojection destinations, the scratch and the conv history - and it reads
/// the n-gram rows there, so the PLE never crosses a stage boundary.  The other stages do not
/// touch the PLE at all.  -1 when no stage holds the layer.
int64_t exl3_flash_ple_stage(const std::vector<Exl3FlashStageRange>& ranges);
/// The same for the even split of `exl3_flash_stage_range`.
int64_t exl3_flash_ple_stage(int64_t layer_count, int64_t stage_count);

/// Owns the explicit EXL3 archive-to-runtime bridge for Flash AR execution.  It validates typed geometry and canonical mapping, attaches
/// the exact EXL3/raw rows to one WeightTable per GPU stage, prepares the
/// embedding/PLE owners, and opens the mixed packed/raw expert runtime.
///
/// forward() records the existing block pre/post graphs, merges CPU expert
/// misses with resident GPU rows, and bridges the residual between stages.
/// The explicit expert-window API also supports callers with their own buffers.
/// All GPU allocations and teardown use the stage device's OnDevice scope.
class Exl3FlashModel {
public:
    Exl3FlashModel();
    ~Exl3FlashModel();
    Exl3FlashModel(const Exl3FlashModel&) = delete;
    Exl3FlashModel& operator=(const Exl3FlashModel&) = delete;

    bool open(const std::filesystem::path& directory, const Exl3FlashModelOptions& options, std::string& err);
    bool close(std::string& err) noexcept;
    void close() noexcept;

    /// Prepare every owner that has stream-local state before graph capture.
    /// The stream may be null for the current device's compute queue.
    bool prepare_stream(void* stream, std::string& err);

    /// Prepare one routed expert window, run CPU misses, and enqueue GPU hits.
    /// `cpu_output` is written for misses and `device_output` for resident
    /// hits; the caller performs the existing router-weighted merge.  The
    /// returned event belongs to the selected stage's runtime.
    bool prepare_expert_window(size_t stage, int64_t layer, const float* host_x, int64_t ldx,
                               const int32_t* routed_ids, int64_t n_tokens, float* cpu_output, int64_t ldy,
                               void* stream, const float* device_x, float* device_output, sycl::event& done,
                               std::string& err);
    bool wait_expert_window(size_t stage, sycl::event& done, std::string& err);

    /// Reset all stage-local recurrent/KV/PLE state.  The next forward call
    /// must use position 0 and supplies the first token's embedding.
    bool reset(std::string& err);
    /// Run one AR token through every stage, including the mapped router,
    /// exact expert merge, post graph, final EXL3 head, and host argmax.
    /// `host_logits` may be null; argmax is still computed internally.
    bool forward(int32_t token, int64_t pos, int32_t& next_token, float* host_logits, std::string& err);

    bool validate_token_request(int64_t n_tokens, int64_t batch, std::string& err) const;

    bool loaded() const;
    size_t stage_count() const;
    Exl3FlashStageView stage(size_t index) const;
    const strata::ModelSpec& spec() const;
    const ModelGeometry& geometry() const;
    const strata::artifact::exl3::ModelMap& model_map() const;
    NativeEmbed* embedding() const;
    Exl3Ple* ple() const;
    strata::kernels::PleTable* ple_table() const;
    const std::shared_ptr<const strata::artifact::exl3::Archive>& archive() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::core
