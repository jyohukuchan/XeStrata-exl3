// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/plain_model.cpp - an EXL3 MoE model with one residual stream (plain_model.hpp).
//
// The block follows llama.cpp's qwen35moe graph (src/models/qwen35moe.cpp at de7fa0a3) step for step:
//
//   GDN:        qkv = attn_qkv(h), z = attn_gate(h), beta = sigmoid(ssm_beta(h)), alpha = ssm_alpha(h)
//               conv over [history | qkv], SiLU on all of it; q, k l2-normalized (v head j reads q/k head j % h_k)
//               gate = softplus(alpha + dt) * ssm_a; the delta rule; y = rms_norm(o) * ssm_norm * silu(z)
//               out = ssm_out(y)
//   attention:  [q | gate] per head = attn_q(h), k = attn_k(h), v = attn_v(h); q, k rms-normed per head, then
//               rotated in their first 64 dims (text positions: every IMRoPE section at the same position)
//               softmax(q k / sqrt(d)) v over every cell, times sigmoid(gate); out = attn_output(.)
//   MoE:        softmax over the experts, top-k, the k weights renormalized; sum of w_j expert_j(h)
//               + sigmoid(ffn_gate_inp_shexp . h) * shared(h)
//
// and qwen3moe's (llm-jp-4.1, src/models/qwen3moe.cpp): every block a full attention without the gate, rotated over the
// whole head; the MoE without the shared expert; the second norm named ffn_norm.
#include "strata/core/plain_model.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq_mmq.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/platform/memory.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <type_traits>
#include <utility>

namespace strata::core {
namespace {

namespace k = strata::kernels;

constexpr int kBF16 = 30;   // a BF16 file's projections: bf16_gemv_fp32_mmvf_multi on the FP32 rows
constexpr int kF32 = 0;

bool mapped(size_t bytes, void** h, void** d) {
    if (!strata::gpu::alloc_host(h, bytes)) return false;
    std::memset(*h, 0, bytes);
    return strata::gpu::device_pointer(d, *h);
}

k::QsaShapes attn_shapes(const ModelSpec& s, int64_t n_head_kv) {
    k::QsaShapes q = k::qsa_real_shapes();
    q.n_head = s.n_head;
    q.n_head_kv = n_head_kv;
    q.head_dim = s.key_length;
    q.n_rot = s.rope_dims;
    return q;
}

}  // namespace

PlainModel::PlainModel() = default;

PlainModel::~PlainModel() {
    if (swap_cs_ != nullptr) strata::gpu::stream_sync(swap_cs_);
    if (cs_ != nullptr) strata::gpu::stream_sync(cs_);
    prompt_gemm_.reset();   // the products' state before the stream
    strata::gpu::free(prompt_host_);
    for (auto* g : exec_) strata::gpu::graph_destroy(g);
    for (auto& v : segs_)
        for (auto* g : v) strata::gpu::graph_destroy(g);
    for (auto* g : mexec_) strata::gpu::graph_destroy(g);
    for (auto& v : msegs_)
        for (auto* g : v) strata::gpu::graph_destroy(g);
    for (void* p : allocs_) strata::gpu::free(p);
    for (void* p : exl3_aux_allocs_) strata::gpu::free(p);
    // Exl3ExpertRuntime borrows the CPU pool and source. Destroy it while both
    // owners are still alive; member declaration order cannot express this
    // because the legacy pool/cache fields are interleaved with the EXL3 state.
    exl3_runtime_.reset();
    exl3_source_.reset();
    exl3_dense_.clear();
    for (void* p : {(void*) h_emb_, (void*) h_tokens_, (void*) h_step_, (void*) h_pos_, (void*) h_out_, (void*) fill_tmp_})
        strata::gpu::free(p);
    if (swap_ev_ != nullptr) strata::gpu::event_destroy(swap_ev_);
    if (swap_cs_ != nullptr) strata::gpu::stream_destroy(swap_cs_);
    if (cs_ != nullptr) strata::gpu::stream_destroy(cs_);
}

const strata::artifact::exl3::ProjectionMapping* PlainModel::exl3_projection(
    strata::artifact::exl3::ProjectionKind kind, int64_t layer, int64_t expert) const {
    for (const auto& mapping : exl3_map_.projections)
        if (mapping.kind == kind && mapping.layer == layer && mapping.expert == expert) return &mapping;
    return nullptr;
}

namespace {

bool exl3_axis_map(const strata::artifact::exl3::AxisMapping& axis, std::vector<int64_t>& result, std::string& err) {
    result.clear();
    if (!axis.permutation.empty()) { result = axis.permutation; return true; }
    if (axis.transform == strata::artifact::exl3::AxisTransform::identity ||
        axis.transform == strata::artifact::exl3::AxisTransform::raw_rows)
        return true;
    if (axis.transform == strata::artifact::exl3::AxisTransform::h128_block_permutation) {
        if (axis.block_size <= 0 || axis.block_offset < 0 || axis.target_size <= 0 ||
            axis.block_permutation.empty() || axis.target_size < axis.block_offset ||
            (axis.target_size - axis.block_offset) % axis.block_size != 0 ||
            axis.block_permutation.size() != static_cast<size_t>((axis.target_size - axis.block_offset) / axis.block_size)) {
            err = "EXL3 raw axis has an invalid H128 block map";
            return false;
        }
        result.resize(static_cast<size_t>(axis.target_size));
        for (int64_t i = 0; i < axis.target_size; ++i) result[(size_t) i] = i;
        for (size_t block = 0; block < axis.block_permutation.size(); ++block)
            for (int64_t d = 0; d < axis.block_size; ++d) {
                const int64_t target = axis.block_offset + (int64_t) block * axis.block_size + d;
                const int64_t source = axis.block_offset + axis.block_permutation[block] * axis.block_size + d;
                if (source < 0 || source >= axis.source_size) {
                    err = "EXL3 raw axis H128 map is outside the source";
                    return false;
                }
                result[(size_t) target] = source;
            }
        return true;
    }
    if (axis.transform == strata::artifact::exl3::AxisTransform::slices) {
        if (axis.target_size <= 0) { err = "EXL3 raw axis slice map has no target"; return false; }
        result.resize(static_cast<size_t>(axis.target_size), -1);
        for (const auto& slice : axis.slices)
            for (int64_t i = 0; i < slice.elements; ++i) {
                const int64_t target = slice.target_offset + i;
                const int64_t source = slice.source_offset + i;
                if (target < 0 || target >= axis.target_size || source < 0 || source >= axis.source_size ||
                    result[(size_t) target] != -1) {
                    err = "EXL3 raw axis slices overlap or are outside bounds";
                    return false;
                }
                result[(size_t) target] = source;
            }
        if (std::find(result.begin(), result.end(), -1) != result.end()) {
            err = "EXL3 raw axis slices do not cover the target";
            return false;
        }
        return true;
    }
    err = "EXL3 raw axis transform is unsupported";
    return false;
}

std::vector<int64_t> exl3_block_map(const strata::artifact::exl3::AxisMapping& axis) {
    // The map may permute only the V-head region after an unchanged Q/K
    // prefix. Packed remapping needs the complete axis in H128 units.
    std::vector<int64_t> elements;
    std::string error;
    if (!exl3_axis_map(axis, elements, error)) throw std::runtime_error(error);
    if (elements.empty()) return {};
    if (axis.source_size % 128 != 0 || elements.size() % 128 != 0)
        throw std::runtime_error("EXL3 packed axis map is not H128 aligned");
    std::vector<int64_t> blocks(elements.size() / 128);
    for (size_t block = 0; block < blocks.size(); ++block) {
        const int64_t first = elements[block * 128];
        if (first < 0 || first % 128 != 0)
            throw std::runtime_error("EXL3 packed axis starts inside an H128 block");
        for (int64_t lane = 0; lane < 128; ++lane)
            if (elements[block * 128 + static_cast<size_t>(lane)] != first + lane)
                throw std::runtime_error("EXL3 packed axis reorders values within an H128 block");
        blocks[block] = first / 128;
    }
    return blocks;
}

}  // namespace

bool PlainModel::raw_aux_f32(const strata::artifact::exl3::ProjectionMapping& mapping, const float*& out, uint64_t expected_count,
                             std::string& err) {
    const auto* tensor = exl3_archive_ == nullptr ? nullptr : exl3_archive_->find_tensor(mapping.source_name);
    if (tensor == nullptr) {
        err = "plain model: EXL3 raw tensor is missing: " + mapping.source_name;
        return false;
    }
    strata::artifact::exl3::RawTensor raw;
    std::vector<std::vector<int64_t>> axes;
    axes.reserve(mapping.raw_axes.size());
    for (const auto& axis : mapping.raw_axes) {
        std::vector<int64_t> map;
        if (!exl3_axis_map(axis, map, err)) return false;
        axes.push_back(std::move(map));
    }
    if (!strata::artifact::exl3::materialize_raw(*exl3_archive_, *tensor, axes, mapping.value_transform, raw, err))
        return false;
    std::vector<float> values;
    if (!strata::artifact::exl3::raw_to_f32(raw, values, err)) return false;
    if (values.empty() || values.size() != expected_count ||
        values.size() > std::numeric_limits<size_t>::max() / sizeof(float)) {
        err = "plain model: EXL3 raw auxiliary tensor has an invalid size: " + mapping.source_name;
        return false;
    }
    void* device = nullptr;
    if (!strata::gpu::alloc_device(&device, values.size() * sizeof(float)) ||
        !strata::gpu::copy(device, values.data(), values.size() * sizeof(float))) {
        err = "plain model: uploading EXL3 raw auxiliary tensor failed: " + mapping.source_name;
        if (device != nullptr) strata::gpu::free(device);
        return false;
    }
    exl3_aux_allocs_.push_back(device);
    out = static_cast<const float*>(device);
    return true;
}

bool PlainModel::run_projection(const W& w, const float* x, float* y, int rows, bool, strata::gpu::Stream cs,
                                std::string& err) {
    if (w.exl3 == nullptr || w.n_in <= 0 || w.n_out <= 0 || rows <= 0) {
        err = "plain model: EXL3 projection owner is not loaded";
        return false;
    }
    return w.exl3->run(x, w.n_in, y, w.n_out, rows, cs, err);
}

bool PlainModel::load_exl3(std::string& err) {
    err.clear();
    try {
        exl3_archive_ = std::make_shared<strata::artifact::exl3::Archive>(opt_.exl3_dir);
        const bool include_mtp = opt_.spec >= 2;
        if (!strata::artifact::exl3::build_model_map(*exl3_archive_, exl3_map_, err, include_mtp) ||
            !exl3_map_.require_supported(err))
            return false;
        if (exl3_map_.family != strata::artifact::exl3::ModelFamily::qwen35) {
            err = "plain model: this EXL3 integration currently supports Qwen3.5 only";
            return false;
        }
        const auto& g = exl3_map_.geometry;
        int64_t bootstrap_kv = 0;
        for (int64_t l = 0; l < spec_.n_layer; ++l)
            if (!spec_.recurrent[(size_t) l]) { bootstrap_kv = spec_.n_head_kv[(size_t) l]; break; }
        const bool geometry_ok = g.hidden_size == spec_.n_embd && g.layers == spec_.n_layer &&
            g.experts == spec_.n_expert && g.experts_per_token == spec_.n_expert_used &&
            g.moe_intermediate == spec_.n_ff_exp && g.shared_intermediate == spec_.n_ff_shexp &&
            g.vocab_size > 0 && g.vocab_size == exl3_map_.geometry.vocab_size &&
            g.q_heads == spec_.n_head && g.kv_heads == bootstrap_kv && g.head_dim == spec_.key_length &&
            g.linear_k_heads == spec_.ssm_n_group && g.linear_v_heads == spec_.ssm_dt_rank &&
            g.linear_k_dim == spec_.ssm_d_state && g.linear_v_dim == spec_.ssm_d_state &&
            g.full_attention_interval == (spec_.full_attention_interval > 0 ? spec_.full_attention_interval : 4) &&
            spec_.key_length == spec_.value_length;
        for (int64_t l = 0; geometry_ok && l < spec_.n_layer; ++l) {
            const bool mapped_recurrent = exl3_projection(strata::artifact::exl3::ProjectionKind::gdn_qkv, l) != nullptr;
            if (mapped_recurrent != spec_.recurrent[(size_t) l]) {
                err = "plain model: EXL3 layer attention pattern disagrees with metadata bootstrap";
                return false;
            }
        }
        if (!geometry_ok) {
            err = "plain model: EXL3 geometry disagrees with metadata bootstrap";
            return false;
        }
        if (include_mtp && (spec_.n_nextn != 1 || exl3_map_.geometry.mtp_layers != 1)) {
            err = "plain model: EXL3 MTP requires metadata block_count=trunk+1 and one nextn layer";
            return false;
        }
        const auto* embedding = exl3_projection(strata::artifact::exl3::ProjectionKind::embedding);
        if (embedding == nullptr || embedding->source_name.size() <= 7 ||
            embedding->source_name.compare(embedding->source_name.size() - 7, 7, ".weight") != 0) {
            err = "plain model: EXL3 embedding mapping is missing";
            return false;
        }
        const std::string embedding_name = embedding->source_name.substr(0, embedding->source_name.size() - 7);
        if (!exl3_embed_.load_exl3(*exl3_archive_, embedding_name, spec_.n_embd, g.vocab_size, err)) return false;
        n_vocab_ = g.vocab_size;
        auto load_dense_owner = [&](const strata::artifact::exl3::ProjectionMapping* mapping, W& target) {
            if (mapping == nullptr) { err = "plain model: EXL3 MTP projection mapping is missing"; return false; }
            auto owner = std::make_unique<Exl3Dense>();
            if (mapping->transform == strata::artifact::exl3::MappingTransform::raw_rows) {
                const auto* tensor = exl3_archive_->find_tensor(mapping->source_name);
                if (tensor == nullptr) { err = "plain model: EXL3 raw projection is missing"; return false; }
                strata::artifact::exl3::RawTensor raw;
                std::vector<std::vector<int64_t>> axes;
                for (const auto& axis : mapping->raw_axes) {
                    std::vector<int64_t> map;
                    if (!exl3_axis_map(axis, map, err)) return false;
                    axes.push_back(std::move(map));
                }
                strata::artifact::exl3::RawTensor bias;
                const strata::artifact::exl3::RawTensor* bias_ptr = nullptr;
                const std::string suffix = ".weight";
                const std::string base = mapping->source_name.size() > suffix.size() &&
                                                 mapping->source_name.compare(mapping->source_name.size() - suffix.size(),
                                                                              suffix.size(), suffix) == 0
                                             ? mapping->source_name.substr(0, mapping->source_name.size() - suffix.size())
                                             : mapping->source_name;
                if (const auto* bias_tensor = exl3_archive_->find_tensor(base + ".bias"); bias_tensor != nullptr) {
                    if (!strata::artifact::exl3::materialize_raw(*exl3_archive_, *bias_tensor, {},
                                                                 strata::artifact::exl3::ValueTransform::identity, bias,
                                                                 err))
                        return false;
                    bias_ptr = &bias;
                }
                if (!strata::artifact::exl3::materialize_raw(*exl3_archive_, *tensor, axes, mapping->value_transform, raw, err) ||
                    !owner->load_raw_detached(raw, bias_ptr, err, opt_.max_t)) return false;
            } else if (!owner->load_detached(*exl3_archive_, mapping->source_name, mapping->target_k, mapping->target_n,
                                             exl3_block_map(mapping->input_axis), exl3_block_map(mapping->output_axis), err,
                                             opt_.max_t)) return false;
            if (!owner->prepare_stream(cs_, err)) return false;
            target.n_in = (int) mapping->target_k;
            target.n_out = (int) mapping->target_n;
            target.exl3 = owner.get();
            exl3_dense_.push_back(std::move(owner));
            return true;
        };
        const auto* head = exl3_projection(strata::artifact::exl3::ProjectionKind::lm_head);
        const auto* norm = exl3_projection(strata::artifact::exl3::ProjectionKind::final_norm);
        if (head == nullptr || norm == nullptr) {
            err = "plain model: EXL3 head or final norm mapping is missing";
            return false;
        }
        auto dense = std::make_unique<Exl3Dense>();
        if (head->transform == strata::artifact::exl3::MappingTransform::raw_rows) {
            const auto* tensor = exl3_archive_->find_tensor(head->source_name);
            if (tensor == nullptr) { err = "plain model: EXL3 raw head is missing"; return false; }
            strata::artifact::exl3::RawTensor raw;
            std::vector<std::vector<int64_t>> axes;
            for (const auto& axis : head->raw_axes) {
                std::vector<int64_t> map;
                if (!exl3_axis_map(axis, map, err)) return false;
                axes.push_back(std::move(map));
            }
            if (!strata::artifact::exl3::materialize_raw(*exl3_archive_, *tensor, axes, head->value_transform, raw, err))
                return false;
            strata::artifact::exl3::RawTensor bias;
            const std::string suffix = ".weight";
            const std::string base = head->source_name.size() > suffix.size() &&
                                             head->source_name.compare(head->source_name.size() - suffix.size(), suffix.size(), suffix) == 0
                                         ? head->source_name.substr(0, head->source_name.size() - suffix.size())
                                         : head->source_name;
            const auto* bias_tensor = exl3_archive_->find_tensor(base + ".bias");
            const strata::artifact::exl3::RawTensor* bias_ptr = nullptr;
            if (bias_tensor != nullptr) {
                if (!strata::artifact::exl3::materialize_raw(*exl3_archive_, *bias_tensor, {},
                                                             strata::artifact::exl3::ValueTransform::identity, bias, err))
                    return false;
                bias_ptr = &bias;
            }
            if (!dense->load_raw_detached(raw, bias_ptr, err, opt_.max_t)) return false;
        } else if (!dense->load_detached(*exl3_archive_, head->source_name, spec_.n_embd, n_vocab_,
                                         exl3_block_map(head->input_axis), exl3_block_map(head->output_axis), err,
                                         opt_.max_t))
            return false;
        if (!dense->prepare_stream(cs_, err)) return false;
        head_.n_in = (int) spec_.n_embd;
        head_.n_out = (int) n_vocab_;
        head_.exl3 = dense.get();
        exl3_dense_.push_back(std::move(dense));
        if (!raw_aux_f32(*norm, out_norm_, static_cast<uint64_t>(spec_.n_embd), err)) return false;
        emb_type_ = -1;
        emb_ = nullptr;
        emb_row_ = 0;
        blocks_.resize((size_t) spec_.n_layer);
        int64_t n_gdn = 0, n_attn = 0;
        for (int64_t l = 0; l < spec_.n_layer; ++l) {
            Block& b = blocks_[(size_t) l];
            b.recurrent = spec_.recurrent[(size_t) l];
            if (b.recurrent) b.gdn = n_gdn++;
            else b.attn = n_attn++;
            if (!load_exl3_block(l, b, err)) return false;
        }
        has_mtp_ = include_mtp;
        if (has_mtp_) {
            mtp_.recurrent = false;
            mtp_.attn = n_attn++;
            if (!load_exl3_block(spec_.n_layer, mtp_, err) ||
                !load_dense_owner(exl3_projection(strata::artifact::exl3::ProjectionKind::mtp_fc, spec_.n_layer), mtp_eh_))
                return false;
            const auto find_canonical = [&](const std::string& canonical) -> const strata::artifact::exl3::ProjectionMapping* {
                for (const auto& mapping : exl3_map_.projections)
                    if (mapping.canonical_name == canonical) return &mapping;
                return nullptr;
            };
            const auto* enorm = find_canonical("blk." + std::to_string(spec_.n_layer) + ".nextn.enorm.weight");
            const auto* hnorm = find_canonical("blk." + std::to_string(spec_.n_layer) + ".nextn.hnorm.weight");
            const auto* head_norm = find_canonical("blk." + std::to_string(spec_.n_layer) + ".nextn.shared_head_norm.weight");
            if (enorm == nullptr || hnorm == nullptr || head_norm == nullptr) {
                err = "plain model: EXL3 MTP norm mappings are missing";
                return false;
            }
            if (!raw_aux_f32(*enorm,
                             mtp_enorm_, (uint64_t) spec_.n_embd, err) ||
                !raw_aux_f32(*hnorm,
                             mtp_hnorm_, (uint64_t) spec_.n_embd, err) ||
                !raw_aux_f32(*head_norm,
                             mtp_head_norm_, (uint64_t) spec_.n_embd, err)) return false;
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("plain model: EXL3 load: ") + e.what();
        return false;
    }
}

bool PlainModel::load_exl3_block(int64_t l, Block& b, std::string& err) {
    using PK = strata::artifact::exl3::ProjectionKind;
    auto dense = [&](PK kind, W& out, int64_t expert = -1) {
        const auto* mapping = exl3_projection(kind, l, expert);
        if (mapping == nullptr) { err = "plain model: EXL3 projection mapping is missing for layer " + std::to_string(l); return false; }
        auto owner = std::make_unique<Exl3Dense>();
        if (mapping->transform == strata::artifact::exl3::MappingTransform::raw_rows) {
            const auto* tensor = exl3_archive_->find_tensor(mapping->source_name);
            if (tensor == nullptr) { err = "plain model: EXL3 raw projection is missing: " + mapping->source_name; return false; }
            strata::artifact::exl3::RawTensor raw;
            std::vector<std::vector<int64_t>> axes;
            for (const auto& axis : mapping->raw_axes) {
                std::vector<int64_t> map;
                if (!exl3_axis_map(axis, map, err)) return false;
                axes.push_back(std::move(map));
            }
            if (!strata::artifact::exl3::materialize_raw(*exl3_archive_, *tensor, axes, mapping->value_transform, raw, err)) return false;
            strata::artifact::exl3::RawTensor bias;
            const std::string suffix = ".weight";
            const std::string base = mapping->source_name.size() > suffix.size() &&
                                             mapping->source_name.compare(mapping->source_name.size() - suffix.size(), suffix.size(), suffix) == 0
                                         ? mapping->source_name.substr(0, mapping->source_name.size() - suffix.size())
                                         : mapping->source_name;
            const auto* bias_tensor = exl3_archive_->find_tensor(base + ".bias");
            const auto* bias_ptr = static_cast<const strata::artifact::exl3::RawTensor*>(nullptr);
            if (bias_tensor != nullptr) {
                if (!strata::artifact::exl3::materialize_raw(*exl3_archive_, *bias_tensor, {},
                                                             strata::artifact::exl3::ValueTransform::identity, bias, err))
                    return false;
                bias_ptr = &bias;
            }
            if (!owner->load_raw_detached(raw, bias_ptr, err, opt_.max_t)) return false;
        } else if (!owner->load_detached(*exl3_archive_, mapping->source_name, mapping->target_k, mapping->target_n,
                                         exl3_block_map(mapping->input_axis), exl3_block_map(mapping->output_axis), err,
                                         opt_.max_t)) return false;
        if (!owner->prepare_stream(cs_, err)) return false;
        out.n_in = (int) mapping->target_k;
        out.n_out = (int) mapping->target_n;
        out.exl3 = owner.get();
        exl3_dense_.push_back(std::move(owner));
        return true;
    };
    const bool q35 = spec_.arch == ModelArch::Qwen35Moe;
    // The MTP block uses the ordinary canonical block prefix.  Only its FC and
    // pre/head norms carry the nextn namespace (resolved separately below).
    const std::string canonical_prefix = "blk." + std::to_string(l) + ".";
    // The map records canonical names for auxiliary tensors; find by canonical name because several auxiliaries share a kind.
    auto aux_name = [&](const std::string& canonical, const float*& out, uint64_t expected_count) {
        for (const auto& mapping : exl3_map_.projections)
            if (mapping.layer == l && mapping.canonical_name == canonical) return raw_aux_f32(mapping, out, expected_count, err);
        err = std::string("plain model: EXL3 auxiliary mapping is missing: ") + canonical;
        return false;
    };
    if (!aux_name(canonical_prefix + "attn_norm.weight", b.attn_norm, static_cast<uint64_t>(spec_.n_embd)) ||
        !aux_name(canonical_prefix + (q35 ? "post_attention_norm.weight" : "ffn_norm.weight"), b.post_norm,
                  static_cast<uint64_t>(spec_.n_embd)))
        return false;
    if (b.recurrent) {
        const int64_t C = 2 * spec_.ssm_n_group * spec_.ssm_d_state + spec_.ssm_d_inner;
        if (!dense(PK::gdn_qkv, b.qkv) || !dense(PK::gdn_z, b.z) || !dense(PK::gdn_out, b.ssm_out) ||
            !dense(PK::gdn_alpha, b.alpha) || !dense(PK::gdn_beta, b.beta) ||
            !aux_name(canonical_prefix + "ssm_norm.weight", b.ssm_norm, static_cast<uint64_t>(spec_.ssm_d_state)) ||
            !aux_name(canonical_prefix + "ssm_conv1d.weight", b.conv, static_cast<uint64_t>(C * spec_.ssm_d_conv)) ||
            !aux_name(canonical_prefix + "ssm_dt.bias", b.dt, static_cast<uint64_t>(spec_.ssm_dt_rank)) ||
            !aux_name(canonical_prefix + "ssm_a", b.a, static_cast<uint64_t>(spec_.ssm_dt_rank))) return false;
    } else {
        const int64_t nkv = spec_.n_head_kv[(size_t) l];
        if (nkv <= 0 || spec_.n_head % nkv != 0 || (nkv_ != 0 && nkv_ != nkv)) {
            err = "plain model: EXL3 attention K/V head geometry is inconsistent";
            return false;
        }
        nkv_ = nkv;
        if (!dense(PK::qsa_q, b.q) || !dense(PK::qsa_k, b.k) || !dense(PK::qsa_v, b.v) || !dense(PK::qsa_out, b.o) ||
            !aux_name(canonical_prefix + "attn_q_norm.weight", b.q_norm, static_cast<uint64_t>(spec_.key_length)) ||
            !aux_name(canonical_prefix + "attn_k_norm.weight", b.k_norm, static_cast<uint64_t>(spec_.key_length))) return false;
        if (b.q.n_out != (has_gate_ ? 2 : 1) * spec_.n_head * spec_.key_length ||
            b.k.n_out != nkv * spec_.key_length || b.v.n_out != nkv * spec_.value_length ||
            b.o.n_in != spec_.n_head * spec_.key_length) {
            err = "plain model: EXL3 attention projection shapes disagree with bootstrap geometry";
            return false;
        }
    }
    if (!aux_name(canonical_prefix + "ffn_gate_inp.weight", b.router,
                  static_cast<uint64_t>(spec_.n_embd * spec_.n_expert))) return false;
    if (has_shared_) {
        if (!aux_name(canonical_prefix + "ffn_gate_inp_shexp.weight", b.sh_gate_inp, static_cast<uint64_t>(spec_.n_embd)) ||
            !dense(PK::shared_gate, b.sh_gate) || !dense(PK::shared_up, b.sh_up) || !dense(PK::shared_down, b.sh_down))
            return false;
    }
    return true;
}

bool PlainModel::load(const PlainOptions& opt, std::string& err) {
    opt_ = opt;
    exl3_mode_ = !opt_.exl3_dir.empty();
    if (!exl3_mode_) {
        err = "plain model: an EXL3 checkpoint is required";
        return false;
    }
    if (opt_.max_t < 1 || opt_.max_t > k::cpu::MAXT || opt_.max_cells < 16) {
        err = "plain model: the window or the context is out of range";
        return false;
    }
    // not on the processor's own graphics through Level Zero yet: on the UHD 770 the driver aborted (its command stream
    // full) in the batched prompt and after some hundreds of segmented windows
    if (const sycl::device& dev = Runtime::get().device();
        integrated_gpu(dev) && dev.get_backend() == sycl::backend::ext_oneapi_level_zero) {
        err = "plain model: this model does not run on the processor's own graphics yet; choose another GPU (--gpu)";
        return false;
    }
    const std::string metadata_path = exl3_mode_ && !opt_.metadata_gguf.empty() ? opt_.metadata_gguf : opt_.gguf;
    if (metadata_path.empty()) {
        err = "plain model: EXL3 requires a metadata bootstrap GGUF";
        return false;
    }
    try {
        model_ = std::make_unique<GgufModel>(GgufModel::open(metadata_path));
    } catch (const std::exception& e) {
        err = std::string("plain model: ") + e.what();
        return false;
    }
    if (!read_model_spec(model_->meta(), spec_, err)) return false;
    const ModelSpec& s = spec_;
    if (s.arch != ModelArch::Qwen35Moe && s.arch != ModelArch::Qwen3Moe) {
        err = "plain model: " + s.arch_name + " is not run by this engine yet";
        return false;
    }
    const bool q35 = s.arch == ModelArch::Qwen35Moe;
    has_gate_ = q35;
    has_shared_ = q35;
    bool shape_ok = s.key_length == s.value_length && (s.key_length == 128 || s.key_length == 256) &&
                    (s.rope_dims == 64 || s.rope_dims == s.key_length) && s.n_expert_used <= 15 && s.n_dense_lead == 0;
    if (q35)
        shape_ok = shape_ok && s.ssm_d_state == 128 && s.ssm_d_conv == 4 && s.ssm_n_group > 0 &&
                   s.ssm_dt_rank % s.ssm_n_group == 0 && s.ssm_d_inner == s.ssm_dt_rank * s.ssm_d_state &&
                   s.n_ff_shexp > 0;
    if (!shape_ok) {
        err = "plain model: a shape this engine's kernels are not built for";
        return false;
    }
    cs_ = strata::gpu::stream_create();
    if (cs_ == nullptr) { err = "plain model: stream create failed"; return false; }
    // process constants, before any graph: the rotation (its base, and the scaling the options or the file ask for, as
    // generate.cpp resolves it), and the GDN output gate (SiLU in qwen35moe)
    {
        using RST = k::RopeScalingType;
        k::RopeScaling rs;
        rs.freq_base = s.rope_base;
        const std::string type = !opt_.rope_scaling.empty() ? opt_.rope_scaling : s.rope_scaling_type;
        rs.type = type == "yarn" ? RST::YaRN : type == "linear" ? RST::Linear : RST::None;
        if (rs.type != RST::None) {
            rs.factor = opt_.rope_scale > 0 ? opt_.rope_scale : s.rope_scaling_factor > 1.0 ? s.rope_scaling_factor : 1.0;
            rs.orig_ctx = opt_.yarn_orig_ctx > 0 ? opt_.yarn_orig_ctx
                          : s.rope_scaling_orig_ctx >= 1 ? s.rope_scaling_orig_ctx : (double) s.context_length;
            rs.ext_factor = rs.type == RST::YaRN ? 1.0 : 0.0;
        }
        if (const char* why = k::rope_scaling_invalid(rs)) {
            err = std::string("plain model: the rope scaling: ") + why;
            return false;
        }
        k::rope_scaling_set(rs);
    }
    k::gdn_set_silu_gate(q35);

    if (exl3_mode_ && !load_exl3(err)) return false;

    const uint64_t N = (uint64_t) s.n_embd, E = (uint64_t) s.n_expert;
    int64_t n_attn = 0;
    {
        for (int64_t l = 0; l < s.n_layer; ++l)
            if (!s.recurrent[(size_t) l]) {
                ++n_attn;
                if (nkv_ == 0) nkv_ = s.n_head_kv[(size_t) l];
            }
        if (has_mtp_) ++n_attn;
    }

    if (nkv_ == 0) { err = "plain model: the model has no attention block"; return false; }
    // the BF16 projections' two GEMV forms timed for each shape a window takes, before any graph is recorded; the
    // head is left to the default form (its timing would allocate a second copy of it)
    try {
        auto tune = [&](const Block& b) {
            for (const W* w : {&b.qkv, &b.z, &b.alpha, &b.beta, &b.ssm_out, &b.q, &b.k, &b.v, &b.o, &b.sh_gate, &b.sh_up,
                               &b.sh_down})
                if (w->type == kBF16) k::bf16_gemv_fp32_mmvf_multi_tune(w->n_in, w->n_out, cs_);
        };
        for (const Block& b : blocks_) tune(b);
        if (has_mtp_) {
            tune(mtp_);
            if (mtp_eh_.type == kBF16) k::bf16_gemv_fp32_mmvf_multi_tune(mtp_eh_.n_in, mtp_eh_.n_out, cs_);
        }
    } catch (const std::exception& e) {
        err = std::string("plain model: timing the BF16 products: ") + e.what();
        return false;
    }
    // per-sequence state
    const int64_t C = 2 * s.ssm_n_group * s.ssm_d_state + s.ssm_d_inner;
    const k::QsaShapes qs = attn_shapes(s, nkv_);
    for (const Block& b : blocks_) {
        if (b.recurrent) {
            float *st = nullptr, *cv = nullptr;
            if (!strata::gpu::alloc_device(&st, (size_t) (s.ssm_d_state * s.ssm_dt_rank * s.ssm_d_state) * 4) ||
                !strata::gpu::alloc_device(&cv, (size_t) (C * (s.ssm_d_conv - 1)) * 4)) {
                err = "plain model: no VRAM for the recurrent states";
                return false;
            }
            allocs_.push_back(st);
            allocs_.push_back(cv);
            gdn_state_.push_back(st);
            gdn_conv_.push_back(cv);
        }
    }
    // the states after a window's rows but its last, for going back to the last accepted draft: a window checks up to
    // spec - 1 drafts (at most the window kernel's snapshots)
    if (has_mtp_) {
        opt_.spec = std::min({opt_.spec, opt_.max_t, k::kNativeGdnMaxSnaps + 1});
        n_snap_ = opt_.spec - 1;
        gdn_snap_.assign(gdn_state_.size(), {});
        conv_snap_.assign(gdn_state_.size(), {});
        for (size_t g = 0; g < gdn_state_.size(); ++g)
            for (int t = 0; t < n_snap_; ++t) {
                float *st = nullptr, *cv = nullptr;
                if (!strata::gpu::alloc_device(&st, (size_t) (s.ssm_d_state * s.ssm_dt_rank * s.ssm_d_state) * 4) ||
                    !strata::gpu::alloc_device(&cv, (size_t) (C * (s.ssm_d_conv - 1)) * 4)) {
                    err = "plain model: no VRAM for the recurrent states' copies (--spec)";
                    return false;
                }
                allocs_.push_back(st);
                allocs_.push_back(cv);
                gdn_snap_[g].push_back(st);
                conv_snap_[g].push_back(cv);
            }
    }
    // the drafts' head over a token subset (Qwen3.8's draft head, mtp.cpp): on the RX 9060 XT the whole Q8_0 head of
    // Qwen3.6's 248,320 tokens took 1.7 ms a draft, at the card's bandwidth, 2026-10-15.  The engine's row layouts
    // (upload) rewrite each row in place, so a row keeps its GGUF type's size.
    if (has_mtp_ && exl3_mode_ && !opt_.draft_vocab.empty())
        std::fprintf(stderr, "strata plain: EXL3 MTP uses the full output head; --draft-vocab is ignored\n");
    n_pages_ = (opt_.max_cells + qs.page_size - 1) / qs.page_size;
    opt_.max_cells = n_pages_ * qs.page_size;
    {
        const size_t pool = (size_t) (n_pages_ * qs.n_head_kv * qs.page_size * qs.head_dim) * 2;
        for (int64_t i = 0; i < n_attn; ++i) {   // the trunk's attention blocks, then the MTP layer
            uint16_t *kp = nullptr, *vp = nullptr;
            if (!strata::gpu::alloc_device(&kp, pool) || !strata::gpu::alloc_device(&vp, pool)) {
                err = "plain model: no VRAM for the K/V cache (" + std::to_string(opt_.max_cells) + " cells)";
                return false;
            }
            allocs_.push_back(kp);
            allocs_.push_back(vp);
            k_pool_.push_back(kp);
            v_pool_.push_back(vp);
        }
        std::vector<int32_t> table((size_t) n_pages_);
        for (int64_t i = 0; i < n_pages_; ++i) table[(size_t) i] = (int32_t) i;
        if (!strata::gpu::alloc_device(&page_table_, table.size() * 4) ||
            !strata::gpu::copy(page_table_, table.data(), table.size() * 4)) {
            err = "plain model: the page table";
            return false;
        }
        allocs_.push_back(page_table_);
    }

    // the window's buffers
    const uint64_t T = (uint64_t) opt_.max_t, NH = (uint64_t) s.n_head, HD = (uint64_t) s.key_length;
    const uint64_t NKV = (uint64_t) qs.n_head_kv, HV = (uint64_t) s.ssm_dt_rank, ZV = (uint64_t) s.ssm_d_inner;
    const uint64_t K = (uint64_t) s.n_expert_used, FS = (uint64_t) s.n_ff_shexp;
    const bool mapped_tokens = exl3_mode_ ? mapped(T * 4, (void**) &h_tokens_, (void**) &m_tokens_)
                                          : mapped(T * N * 4, (void**) &h_emb_, (void**) &m_emb_);
    if (!mapped_tokens || !mapped(T * k::kStepCount * 4, (void**) &h_step_, (void**) &m_step_) ||
        !mapped(T * (NH + NKV) * 4, (void**) &h_pos_, (void**) &m_pos_) ||
        !mapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_)) {
        err = "plain model: mapped staging allocation failed";
        return false;
    }
    attn_scratch_floats_ = (int64_t) k::qsa_decode_attn_scratch_floats(opt_.max_cells, qs);
    const uint64_t max_in = std::max<uint64_t>({N, ZV, NH * HD, FS, has_mtp_ ? 2 * N : 0});
    auto dev = [&](auto** p, uint64_t bytes) {
        void* d = nullptr;
        if (!strata::gpu::alloc_device(&d, (size_t) std::max<uint64_t>(bytes, 256))) return false;
        allocs_.push_back(d);
        *p = (std::remove_reference_t<decltype(*p)>) d;
        return true;
    };
    if (!dev(&step_, T * k::kStepCount * 4) || !dev(&pos_, T * (NH + NKV) * 4) || !dev(&x_, T * N * 4) ||
        !dev(&hn_, T * N * 4) || !dev(&mix_, T * N * 4) || !dev(&xq_, k::native_q8_1_bytes((int) max_in, (int) T)) ||
        !dev(&qkv_, T * (uint64_t) C * 4) || !dev(&craw_, (uint64_t) C * 4) || !dev(&hc_, T * (uint64_t) C * 4) ||
        !dev(&z_, T * ZV * 4) || !dev(&alpha_, T * HV * 4) || !dev(&beta_, T * HV * 4) || !dev(&gate_, T * HV * 4) ||
        !dev(&o_, T * ZV * 4) || !dev(&y_, T * ZV * 4) || !dev(&qfull_, T * std::max(NH * 2 * HD, 2 * N) * 4) ||
        !dev(&qcur_, T * NH * HD * 4) || !dev(&kraw_, T * NKV * HD * 4) || !dev(&kcur_, T * NKV * HD * 4) ||
        !dev(&vcur_, T * NKV * HD * 4) || !dev(&attn_, T * NH * HD * 4) || !dev(&attn32_, T * NH * HD * 4) ||
        !dev(&attn_scratch_, T * (uint64_t) attn_scratch_floats_ * 4) || !dev(&rlog_, T * E * 4) ||
        !dev(&ids_, T * K * 4) || !dev(&w_, T * K * 4) || !dev(&shared_, T * N * 4) || !dev(&sh_g_, (T + 4) * 4) ||
        !dev(&sh_gate_, T * FS * 4) || !dev(&sh_up_, T * FS * 4) || !dev(&moe_out_, T * N * 4) ||
        !dev(&logits_, T * (uint64_t) n_vocab_ * 4) || (has_mtp_ && !dev(&mtp_h_, T * N * 4))) {
        err = "plain model: no VRAM for the window's buffers";
        return false;
    }
    if (!strata::gpu::stream_sync(cs_)) { err = "plain model: the uploads failed"; return false; }
    if (!open_experts(err)) return false;
    exec_.assign((size_t) opt_.max_t + 1, nullptr);
    segs_.assign((size_t) opt_.max_t + 1, {});
    mexec_.assign((size_t) opt_.max_t + 1, nullptr);
    msegs_.assign((size_t) opt_.max_t + 1, {});
    return reset(err);
}

bool PlainModel::prompt_int8() const {
    if (exl3_mode_) return false;
    const char* v = std::getenv("STRATA_PLAIN_MMQ");
    if (v != nullptr && std::strtol(v, nullptr, 10) == 0) return false;
    if (!prefill::mmq::chosen()) return false;
    // the products' outputs in 128s (2 n_ff and n_embd), their K in blocks of 32; gate's rows right before up's
    const int64_t N = spec_.n_embd, FF = spec_.n_ff_exp;
    if (N % 128 != 0 || (2 * FF) % 128 != 0 || FF % 32 != 0) return false;
    return true;
}

void PlainModel::prompt_carve(int64_t T, PromptBufs& b, uint8_t* dev, uint8_t* host, uint64_t& dev_bytes,
                              uint64_t& host_bytes) const {
    const ModelSpec& s = spec_;
    const auto& lay = k::cpu::expert_layout();
    const uint64_t N = (uint64_t) s.n_embd, NH = (uint64_t) s.n_head, HD = (uint64_t) s.key_length, NKV = (uint64_t) nkv_;
    const uint64_t HV = (uint64_t) s.ssm_dt_rank, ZV = (uint64_t) s.ssm_d_inner;
    const uint64_t C = (uint64_t) (2 * s.ssm_n_group * s.ssm_d_state) + ZV, E = (uint64_t) s.n_expert;
    const uint64_t K = (uint64_t) s.n_expert_used, FS = (uint64_t) s.n_ff_shexp, FF = (uint64_t) s.n_ff_exp;
    // the MTP layer's input rows ([embedding | trunk row], 2 N wide) go through qfull and x16
    const uint64_t U = (uint64_t) T, TK = U * K, QW = std::max<uint64_t>((has_gate_ ? 2u : 1u) * NH * HD, has_mtp_ ? 2 * N : 0);
    const uint64_t max_in = std::max<uint64_t>({N, ZV, NH * HD, FS, FF, has_mtp_ ? 2 * N : 0});
    b.T = T;
    b.gu_elems = (int64_t) (2 * FF * N);
    b.d_elems = (int64_t) (N * FF);
    b.int8 = prompt_int8();
    // the largest dense weight, dequantized at once (Gemm::native slices past 2^25 values)
    b.gemm_elems = (int64_t) mtp_eh_.n_in * mtp_eh_.n_out;   // 0 without the MTP layer
    std::vector<const Block*> all;
    for (const Block& bl : blocks_) all.push_back(&bl);
    if (has_mtp_) all.push_back(&mtp_);
    for (const Block* bl : all)
        for (const W* w : {&bl->qkv, &bl->z, &bl->alpha, &bl->beta, &bl->ssm_out, &bl->q, &bl->k, &bl->v, &bl->o,
                           &bl->sh_gate, &bl->sh_up, &bl->sh_down})
            b.gemm_elems = std::max<int64_t>(b.gemm_elems, (int64_t) w->n_in * w->n_out);
    b.gemm_elems = std::min<int64_t>(b.gemm_elems, (int64_t) 1 << 25);
    uint64_t at = 0;
    auto take = [&](auto*& ptr, uint64_t bytes) {
        ptr = dev ? (std::remove_reference_t<decltype(ptr)>) (dev + at) : nullptr;
        at += (std::max<uint64_t>(bytes, 1) + 255) & ~255ull;
    };
    take(b.x, U * N * 4); take(b.hn, U * N * 4); take(b.mix, U * N * 4); take(b.x16, U * max_in * 2);
    take(b.qkv, U * C * 4); take(b.z, U * ZV * 4); take(b.ab, U * 2 * HV * 4); take(b.gate, U * HV * 4);
    take(b.beta, U * HV * 4); take(b.h, U * C * 4); take(b.y, U * ZV * 4); take(b.y16, U * ZV * 2);
    take(b.qfull, U * QW * 4); take(b.qcur, U * NH * HD * 4); take(b.kraw, U * NKV * HD * 4);
    take(b.kcur, U * NKV * HD * 4); take(b.vcur, U * NKV * HD * 4); take(b.attn, U * NH * HD * 4);
    take(b.attn32, U * NH * HD * 4);
    take(b.steps, U * k::kStepCount * 4); take(b.pos_q, U * NH * 4); take(b.pos_k, U * NKV * 4);
    take(b.rlog, U * E * 4); take(b.ids, TK * 4); take(b.w, TK * 4); take(b.shared, U * N * 4);
    take(b.sh_g, (U + 4) * 4); take(b.sh_gate, U * FS * 4); take(b.sh_up, U * FS * 4);
    // the FP16 path's: the routed rows and the SwiGLU in FP16, kGroup experts' FP16 weights, the down product's rows
    // in grouped order (the int8 one writes them in routing order, straight into parts)
    const uint64_t f16 = b.int8 ? 0 : 1;
    take(b.xs16, f16 * TK * N * 2); take(b.gu, TK * 2 * FF * 4); take(b.hh, TK * FF * 4); take(b.h16, f16 * TK * FF * 2);
    take(b.dm, f16 * TK * N * 4); take(b.parts, TK * N * 4); take(b.moe_out, U * N * 4);
    take(b.src, TK * 4); take(b.slot, TK * 4); take(b.bounds, (E + 1) * 4); take(b.entry, TK * 4);
    take(b.dq_gu, f16 * (uint64_t) PromptBufs::kGroup * (uint64_t) b.gu_elems * 2);
    take(b.dq_d, f16 * (uint64_t) PromptBufs::kGroup * (uint64_t) b.d_elems * 2);
    take(b.stage, 2 * (uint64_t) PromptBufs::kGroup * lay.max_blob);   // two halves: one copied while one is read
    // the int8 activations: values, then a scale and a sum per 32 (iq_mmq_act_bytes); twice as many where a product
    // reads them with every block repeated (Q2_K, iq_mmq_type_dup), and a second copy of the routed rows for a layer
    // whose gate and up differ in that
    bool dup = false, dup_mixed = false;
    for (const Block* bl : all)
        for (const W* w : {&bl->qkv, &bl->z, &bl->ssm_out, &bl->q, &bl->k, &bl->v, &bl->o})
            dup = dup || k::iq_mmq_type_dup(w->type);
    dup = dup || (has_mtp_ && k::iq_mmq_type_dup(mtp_eh_.type));
    const uint64_t rep = dup ? 2 : 1;
    const uint64_t xq_cols = rep * std::max<uint64_t>(TK * N, U * max_in);   // the routed rows', or a dense input's
    take(b.xq, xq_cols + 2 * (xq_cols / 32) * 4); take(b.hq, rep * (TK * FF + 2 * TK * (FF / 32) * 4));
    take(b.xq2, dup_mixed && b.int8 ? rep * (TK * N + 2 * TK * (N / 32) * 4) : 0);
    if (!(dup_mixed && b.int8) && dev != nullptr) b.xq2 = nullptr;
    take(b.identity, TK * 4);
    take(b.wtab, E * sizeof(void*)); take(b.rbounds, (2 * E + 2) * 4); take(b.dense_bounds, 2 * sizeof(int32_t));
    // the CPU's share of the missed experts (plain_prompt.cpp): at most a window's tokens an expert, so 8 E rows
    b.cpu_cap = std::min<int64_t>((int64_t) TK, (int64_t) k::cpu::MAXT * (int64_t) E);
    take(b.cpu, (uint64_t) b.cpu_cap * N * 4); take(b.cpu_row, (uint64_t) b.cpu_cap * 4);
    take(b.gemm_scratch, (uint64_t) b.gemm_elems * 2);
    dev_bytes = at;
    at = 0;
    auto hold = [&](auto*& ptr, uint64_t bytes) {
        ptr = host ? (std::remove_reference_t<decltype(ptr)>) (host + at) : nullptr;
        at += (std::max<uint64_t>(bytes, 1) + 255) & ~255ull;
    };
    hold(b.emb, U * N * 4); hold(b.h_steps, U * k::kStepCount * 4); hold(b.h_pos, U * (NH + NKV) * 4);
    hold(b.h_ids, TK * 4); hold(b.h_src, TK * 4); hold(b.h_slot, TK * 4); hold(b.h_bounds, (E + 1) * 4);
    hold(b.h_entry, TK * 4);
    hold(b.h_wtab, E * sizeof(void*)); hold(b.h_rbounds, (2 * E + 2) * 4);
    hold(b.h_hn, U * N * 4); hold(b.h_cpu, (uint64_t) b.cpu_cap * N * 4); hold(b.h_cpu_row, (uint64_t) b.cpu_cap * 4);
    hold(b.h_dense_bounds, 2 * sizeof(int32_t));
    host_bytes = at;
}

bool PlainModel::open_experts(std::string& err) {
    const ModelSpec& s = spec_;
    n_moe_ = s.n_moe_layers() + (has_mtp_ ? 1 : 0);
    const int64_t L = n_moe_, E = s.n_expert;
    if (exl3_mode_) {
        Exl3ExpertSource::Names names((size_t) L * (size_t) E);
        for (const auto& mapping : exl3_map_.projections) {
            if (mapping.layer < 0 || mapping.layer >= L || mapping.expert < 0 || mapping.expert >= E) continue;
            auto& name = names[(size_t) mapping.layer * (size_t) E + (size_t) mapping.expert];
            if (mapping.kind == strata::artifact::exl3::ProjectionKind::expert_gate) name.gate = mapping.source_name;
            if (mapping.kind == strata::artifact::exl3::ProjectionKind::expert_up) name.up = mapping.source_name;
            if (mapping.kind == strata::artifact::exl3::ProjectionKind::expert_down) name.down = mapping.source_name;
        }
        exl3_source_ = std::make_unique<Exl3ExpertSource>();
        if (!exl3_source_->open(exl3_archive_, L, E, std::move(names), err)) return false;
        pool_ = std::make_unique<k::cpu::ExpertPool>(opt_.pool_workers);
        std::vector<uint64_t> bytes((size_t) L * (size_t) E);
        for (int64_t l = 0; l < L; ++l)
            for (int64_t e = 0; e < E; ++e) {
                const auto* layout = exl3_source_->exl3_layout(l, e);
                if (layout == nullptr) { err = "plain model: EXL3 expert layout is missing"; return false; }
                bytes[(size_t) l * (size_t) E + (size_t) e] = layout->total_bytes;
            }
        size_t free_b = 0, total_b = 0;
        if (!strata::gpu::mem_info(&free_b, &total_b)) { err = "plain model: device memory is unknown"; return false; }
        const uint64_t reserve = opt_.reserve_mib >= 0 ? (uint64_t) opt_.reserve_mib << 20
                                                       : std::clamp<uint64_t>(total_b / 32, 192ull << 20, 768ull << 20);
        const uint64_t entries = (uint64_t) opt_.max_t * (uint64_t) s.n_expert_used;
        const auto align256 = [](uint64_t bytes) { return (bytes + 255u) & ~uint64_t(255u); };
        const uint64_t plan_cap = 4u + (entries + 1u) + 2u * entries;
        const uint64_t plan_i32 = ((plan_cap + 1u) & ~uint64_t(1u)) + 4u * entries + (entries + 1u) + 1u;
        const uint64_t window_device =
            align256(2u * (plan_i32 + 16u) * sizeof(int32_t)) +
            align256(2u * entries * (uint64_t) s.n_embd * sizeof(float)) +
            align256(entries * (sizeof(uint64_t) + 5u * sizeof(int32_t)) +
                    (entries + 1u) * sizeof(int32_t) +
                    (uint64_t) L * (uint64_t) E * sizeof(Exl3ExpertFormatDesc)) +
            align256((uint64_t) L * (uint64_t) E * sizeof(uint64_t)) +
            align256((uint64_t) L * (uint64_t) E * sizeof(int32_t)) +
            align256(entries * (uint64_t) std::max(s.n_embd, s.n_ff_exp) * 6u * sizeof(float));
        uint64_t room = free_b > reserve + window_device ? free_b - reserve - window_device : 0;
        std::vector<Exl3ExpertRuntime::ResidentPair> resident;
        // Fill one resident pair per routed layer first, then round-robin by expert. The actual bytes of every pair
        // are charged; EXL3 formats may differ by layer and are not rounded to max_blob.
        for (int pass = 0; pass < 2; ++pass)
            for (int64_t e = 0; e < E; ++e)
                for (int64_t l = 0; l < L; ++l) {
                    if (pass == 0 && e != 0) continue;
                    if (pass == 1 && e == 0) continue;
                    const size_t at = (size_t) l * (size_t) E + (size_t) e;
                    const uint64_t aligned = (bytes[at] + 255u) & ~uint64_t(255u);
                    if (aligned > room) continue;
                    resident.emplace_back((int32_t) l, (int32_t) e);
                    room -= aligned;
                }
        exl3_runtime_ = std::make_unique<Exl3ExpertRuntime>();
        if (!exl3_runtime_->open(*exl3_source_, *pool_, opt_.max_t, s.n_expert_used, resident, err)) return false;
        res_.clear();
        free_.clear();
        n_slots_ = (int64_t) resident.size();
        opt_.adapt_every = 0;
        MoeWindowShape ws{ s.n_embd, E, s.n_expert_used, opt_.max_t, 1 };
        MoeWindowTier tier;
        tier.exl3_runtime = exl3_runtime_.get();
        if (!win_.init(ws, tier, err)) return false;
        disp_.pool = pool_.get();
        disp_.src = exl3_source_.get();
        disp_.exl3_runtime = exl3_runtime_.get();
        disp_.n_expert = E;
        disp_.plan = win_.sink();
        return true;
    }
    err = "plain model: this engine runs EXL3 archives (--exl3), not GGUF model files";
    return false;
}

int64_t PlainModel::resident() const {
    if (exl3_mode_) return n_slots_;
    int64_t n = 0;
    for (int32_t r : res_) n += r >= 0;
    return n;
}

bool PlainModel::reset(std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t C = 2 * s.ssm_n_group * s.ssm_d_state + s.ssm_d_inner;
    bool ok = true;
    for (float* p : gdn_state_)
        ok = ok && strata::gpu::memset(p, 0, (size_t) (s.ssm_d_state * s.ssm_dt_rank * s.ssm_d_state) * 4);
    for (float* p : gdn_conv_) ok = ok && strata::gpu::memset(p, 0, (size_t) (C * (s.ssm_d_conv - 1)) * 4);
    const k::QsaShapes qs = attn_shapes(s, nkv_);
    const size_t pool = (size_t) (n_pages_ * qs.n_head_kv * qs.page_size * qs.head_dim) * 2;
    for (size_t i = 0; i < k_pool_.size(); ++i)
        ok = ok && strata::gpu::memset(k_pool_[i], 0, pool) && strata::gpu::memset(v_pool_[i], 0, pool);
    cur_ = -1;
    drafts_.clear();
    if (!ok) err = "plain model: resetting the state failed";
    return ok;
}

void PlainModel::pool_fn(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                         int64_t layer) {
    PlainModel* m = (PlainModel*) user;
    const int64_t E = m->spec_.n_expert;
    for (int64_t i = 0; i < n_tok * k; ++i)
        if (!m->heat_.empty() && ids[i] >= 0 && ids[i] < E) m->heat_[(size_t) (layer * E + ids[i])] += 1.0;
    m->disp_.layers = layer;
    expert_pool_dispatch_multi(m->disp_, x_f, ids, n_tok, k, out);
}

bool PlainModel::record_block(const Block& b, int64_t ml, int T, uint32_t& ring, std::vector<strata::gpu::Graph*>& segs,
                              strata::gpu::Stream cs, std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, NH = s.n_head, HD = s.key_length, HV = s.ssm_dt_rank, HK = s.ssm_n_group;
    const int64_t ZV = s.ssm_d_inner, S = s.ssm_d_state, C = 2 * HK * S + ZV, E = s.n_expert, K = s.n_expert_used;
    const int64_t FS = s.n_ff_shexp;
    const float eps = (float) s.rms_eps;
    const k::QsaShapes qs = attn_shapes(s, nkv_);
    const int64_t NKV = qs.n_head_kv;
    const k::GdnShapes gs{S, HK, HV};
    const k::RopeScaling& rs = k::rope_scaling();
    const int32_t* pos_q = pos_;
    const int32_t* pos_k = pos_ + (size_t) opt_.max_t * NH;
    auto proj = [&](const W& w, const float* x, float* y, int ncols, bool quantize) {
        if (exl3_mode_) {
            if (!run_projection(w, x, y, ncols, quantize, cs, err)) throw std::runtime_error(err);
            return;
        }
        if (w.type == kBF16) {   // from the FP32 rows x (row stride n_in), as llama.cpp's float products
            k::bf16_gemv_fp32_mmvf_multi(x, w.n_in, static_cast<const uint16_t*>(w.data), y, w.n_out, w.n_in, w.n_out,
                                         ncols, cs);
            return;
        }
        if (w.type == kF32) {
            k::f32_gemv_fp32_mmvf_multi(x, w.n_in, static_cast<const float*>(w.data), y, w.n_out, w.n_in, w.n_out,
                                        ncols, cs);
            return;
        }
        if (quantize) k::native_quantize_q8_1(x, xq_, w.n_in, ncols, cs);
        k::native_mmvq(w.type, w.data, xq_, y, w.n_in, w.n_out, ncols, cs);
    };
    // the norm writes the q8_1 rows the projections read as well (one launch; after nextsycl's Qwen3.5 norm)
    k::native_qsa_rms_norm_weighted_q8_1(x_, b.attn_norm, hn_, xq_, (int) N, T, eps, cs);
    if (b.recurrent) {
        float* state = gdn_state_[(size_t) b.gdn];
        float* hist = gdn_conv_[(size_t) b.gdn];
        proj(b.qkv, hn_, qkv_, T, false);
        proj(b.z, hn_, z_, T, false);
        // alpha and beta, one shape and type: one launch where the pair kernel takes the type (nextsycl's
        // Qwen3.5 decode fuses the two products)
        if (exl3_mode_ || b.alpha.type == kBF16 || b.alpha.type == kF32 || b.alpha.type != b.beta.type ||
            b.alpha.n_in != b.beta.n_in || b.alpha.n_out != b.beta.n_out ||
            !k::native_mmvq_pair(b.alpha.type, b.alpha.data, b.beta.data, xq_, alpha_, beta_, b.alpha.n_in,
                                 b.alpha.n_out, T, cs)) {
            proj(b.alpha, hn_, alpha_, T, false);
            proj(b.beta, hn_, beta_, T, false);
        }
        const int n_snaps = std::min(T - 1, n_snap_);   // the states after rows 0 .. n_snaps - 1, for spec_step
        for (int t = 0; t < T; ++t) {   // each row's inputs; the convolution's history goes on row by row
            float* h = hc_ + (size_t) t * C;
            k::native_gdn_conv_silu(hist, qkv_ + (size_t) t * C, b.conv, craw_, h, C, s.ssm_d_conv, cs);
            k::native_gdn_l2_norm(h, HK, S, eps, cs);
            k::native_gdn_l2_norm(h + HK * S, HK, S, eps, cs);
            k::native_gdn_gate(alpha_ + (size_t) t * HV, b.dt, b.a, gate_ + (size_t) t * HV, HV, cs);
            if (t < n_snaps)
                k::copy_from_mapped(conv_snap_[(size_t) b.gdn][(size_t) t], hist, C * (s.ssm_d_conv - 1), cs);
        }
        k::native_gdn_beta_gate(beta_, HV * T, cs);
        // the recurrence over the window's rows in one launch (the state read and written once, the rows' states
        // to go back to written beside): one launch a row read and wrote the 2 MiB state of Qwen3.6's 32 heads each
        // time, 45 us a row on the RX 9060 XT (2026-10-15)
        float* snaps[k::kNativeGdnMaxSnaps] = {};
        for (int t = 0; t < n_snaps; ++t) snaps[t] = gdn_snap_[(size_t) b.gdn][(size_t) t];
        k::native_gdn_step_rows(state, hc_, C, gate_, beta_, o_, ZV, T, snaps, n_snaps, gs, cs);
        k::native_gdn_out_norm(o_, z_, b.ssm_norm, y_, HV * T, S, eps, cs);
        proj(b.ssm_out, y_, mix_, T, true);
    } else {
        const int64_t a = b.attn;
        proj(b.q, hn_, qfull_, T, false);
        proj(b.k, hn_, kraw_, T, false);
        proj(b.v, hn_, vcur_, T, false);
        k::native_qsa_rms_norm_rope(qfull_, (int) ((has_gate_ ? 2 : 1) * HD), b.q_norm, qcur_, (int) (T * NH), (int) HD,
                                    (int) s.rope_dims, eps, rs, pos_q, cs);
        k::native_qsa_rms_norm_rope(kraw_, (int) HD, b.k_norm, kcur_, (int) (T * NKV), (int) HD,
                                    (int) s.rope_dims, eps, rs, pos_k, cs);
        for (int t = 0; t < T; ++t)
            k::kv_append_step(k_pool_[(size_t) a], v_pool_[(size_t) a], page_table_,
                              step_ + (size_t) t * k::kStepCount, kcur_ + (size_t) t * NKV * HD,
                              vcur_ + (size_t) t * NKV * HD, qs, cs);
        k::QsaAttnPools pools;
        pools.k_pool = k_pool_[(size_t) a];
        pools.v_pool = v_pool_[(size_t) a];
        pools.page_table = page_table_;
        k::qsa_dense_attn_batch(qcur_, pools, step_, opt_.max_cells, qs, attn_scratch_, attn_, T, cs);
        if (has_gate_) {
            k::native_qsa_gate_apply(attn_, qfull_, attn32_, (int) (T * NH), (int) HD, cs);
            proj(b.o, attn32_, mix_, T, true);
        } else {
            proj(b.o, attn_, mix_, T, true);
        }
    }
    k::add_inplace(x_, mix_, (int64_t) T * N, cs);

    // the MoE: route, publish, the shared expert while the host serves, then the experts and the combine
    const bool sh_q8 = !exl3_mode_ && has_shared_ && b.sh_gate.type != kBF16 && b.sh_down.type != kBF16;
    if (sh_q8) k::native_qsa_rms_norm_weighted_q8_1(x_, b.post_norm, hn_, xq_, (int) N, T, eps, cs);
    else k::native_qsa_rms_norm_weighted(x_, b.post_norm, hn_, (int) N, T, eps, cs);
    k::f32_gemv_fp32_mmvf_multi(hn_, N, b.router, rlog_, E, N, E, T, cs);
    k::router_top10(rlog_, T, (int) E, (int) K, ids_, w_, cs);
    ++ring;
    win_.publish(ring, 0, 0, T, hn_, ids_, w_, cs);
    if (has_shared_ && !sh_q8) {   // in FP32 between the products
        proj(b.sh_gate, hn_, sh_gate_, T, true);
        proj(b.sh_up, hn_, sh_up_, T, false);
        k::native_swiglu_f32(sh_gate_, sh_up_, sh_gate_, (int) FS, T, cs);
        proj(b.sh_down, sh_gate_, shared_, T, true);
        k::f32_gemv_fp32_mmvf_multi(hn_, N, b.sh_gate_inp, sh_g_, 1, N, 1, T, cs);
    } else if (has_shared_) {
        if (!k::native_mmvq_pair(b.sh_gate.type, b.sh_gate.data, b.sh_up.data, xq_, sh_gate_, sh_up_, (int) N,
                                 (int) FS, T, cs)) {
            k::native_mmvq(b.sh_gate.type, b.sh_gate.data, xq_, sh_gate_, (int) N, (int) FS, T, cs);
            k::native_mmvq(b.sh_up.type, b.sh_up.data, xq_, sh_up_, (int) N, (int) FS, T, cs);
        }
        k::native_swiglu_q8_1(sh_gate_, sh_up_, xq_, (int) FS, T, cs);
        k::native_mmvq(b.sh_down.type, b.sh_down.data, xq_, shared_, (int) FS, (int) N, T, cs);
        k::f32_gemv_fp32_mmvf_multi(hn_, N, b.sh_gate_inp, sh_g_, 1, N, 1, T, cs);
    }
    if (win_.segmented()) {   // a graph ends where the host must answer the ring
        strata::gpu::Graph* g = nullptr;
        if (!strata::gpu::end_capture(cs, &g)) {
            err = std::string("plain model: segment capture: ") + strata::gpu::last_error();
            return false;
        }
        segs.push_back(g);
        if (!strata::gpu::begin_capture(cs)) { err = "plain model: begin capture failed"; return false; }
    }
    if (!win_.experts(ml, ring, 0, 0, T, cs, err)) return false;
    k::native_moe_combine_multi(win_.parts(), w_, has_shared_ ? shared_ : nullptr, moe_out_, N, K, T, cs,
                                has_shared_ ? sh_g_ : nullptr);
    k::add_inplace(x_, moe_out_, (int64_t) T * N, cs);
    return true;
}

bool PlainModel::record_window(int T, strata::gpu::Stream cs, std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, NH = s.n_head;
    const float eps = (float) s.rms_eps;
    const int64_t NKV = attn_shapes(s, nkv_).n_head_kv;
    if (exl3_mode_) {
        k::copy_i32_from_mapped(m_tokens_, h_tokens_, T, cs);
        exl3_embed_.gather_dev(m_tokens_, T, x_, cs);
    } else k::copy_from_mapped(x_, m_emb_, (int64_t) T * N, cs);
    k::copy_i32_from_mapped(step_, m_step_, (int64_t) T * k::kStepCount, cs);
    k::copy_i32_from_mapped(pos_, m_pos_, (int64_t) opt_.max_t * (NH + NKV), cs);
    uint32_t ring = 0;
    for (int64_t l = 0; l < s.n_layer; ++l)
        if (!record_block(blocks_[(size_t) l], s.moe_index(l), T, ring, segs_[(size_t) T], cs, err)) return false;
    k::native_qsa_rms_norm_weighted(x_, out_norm_, hn_, (int) N, T, eps, cs);
    head_logits(hn_, T, cs);   // the tokens are picked after the graph (forward), with the run's sampler
    return true;
}

void PlainModel::head_logits(const float* x, int T, strata::gpu::Stream cs) {
    if (exl3_mode_) {
        std::string err;
        if (!run_projection(head_, x, logits_, T, true, cs, err)) throw std::runtime_error(err);
        return;
    }
    if (head_.type == kBF16) {
        k::bf16_gemv_fp32_mmvf_multi(x, head_.n_in, static_cast<const uint16_t*>(head_.data), logits_, head_.n_out,
                                     head_.n_in, head_.n_out, T, cs);
        return;
    }
    k::native_quantize_q8_1(x, xq_, head_.n_in, T, cs);
    k::native_mmvq(head_.type, head_.data, xq_, logits_, head_.n_in, head_.n_out, T, cs);
}

bool PlainModel::record_mtp(int T, strata::gpu::Stream cs, std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, NH = s.n_head;
    const float eps = (float) s.rms_eps;
    const int64_t NKV = attn_shapes(s, nkv_).n_head_kv;
    if (exl3_mode_) {
        k::copy_i32_from_mapped(m_tokens_, h_tokens_, T, cs);
        exl3_embed_.gather_dev(m_tokens_, T, x_, cs);
    } else k::copy_from_mapped(x_, m_emb_, (int64_t) T * N, cs);
    k::copy_i32_from_mapped(step_, m_step_, (int64_t) T * k::kStepCount, cs);
    k::copy_i32_from_mapped(pos_, m_pos_, (int64_t) opt_.max_t * (NH + NKV), cs);
    // llama.cpp's graph_mtp: the block's input is eh_proj([rms_norm(e) enorm | rms_norm(h) hnorm])
    k::native_qsa_rms_norm_weighted(x_, mtp_enorm_, hn_, (int) N, T, eps, cs);
    k::native_qsa_rms_norm_weighted(mtp_h_, mtp_hnorm_, mix_, (int) N, T, eps, cs);
    k::concat_rows(qfull_, hn_, mix_, N, T, cs);
    if (exl3_mode_) {
        if (!run_projection(mtp_eh_, qfull_, x_, T, false, cs, err)) return false;
    } else if (mtp_eh_.type == kBF16) {
        k::bf16_gemv_fp32_mmvf_multi(qfull_, 2 * N, static_cast<const uint16_t*>(mtp_eh_.data), x_, N, 2 * N, N, T, cs);
    } else {
        k::native_quantize_q8_1(qfull_, xq_, (int) (2 * N), T, cs);
        k::native_mmvq(mtp_eh_.type, mtp_eh_.data, xq_, x_, (int) (2 * N), (int) N, T, cs);
    }
    uint32_t ring = 0;
    if (!record_block(mtp_, n_moe_ - 1, T, ring, msegs_[(size_t) T], cs, err)) return false;
    // the last row only: its norm is the next step's trunk row, and its logits pick the draft
    k::native_qsa_rms_norm_weighted(x_ + (size_t) (T - 1) * N, mtp_head_norm_, hn_, (int) N, 1, eps, cs);
    k::copy_from_mapped(mtp_h_, hn_, N, cs);
    if (dhead_ != nullptr) {   // the subset's rows: logits_[i] is token dvocab_[i]'s
        if (head_.type == kBF16) {
            k::bf16_gemv_fp32_mmvf_multi(hn_, head_.n_in, static_cast<const uint16_t*>(dhead_), logits_,
                                         (int) n_dvocab_, head_.n_in, (int) n_dvocab_, 1, cs);
        } else {
            k::native_quantize_q8_1(hn_, xq_, head_.n_in, 1, cs);
            k::native_mmvq(head_.type, dhead_, xq_, logits_, head_.n_in, (int) n_dvocab_, 1, cs);
        }
    } else {
        head_logits(hn_, 1, cs);
    }
    return true;
}

// The graph of a window of T rows, the trunk's or (`mtp`) the MTP layer's, recorded once.
bool PlainModel::record(int T, bool mtp, std::string& err) {
    strata::gpu::Graph*& exec = (mtp ? mexec_ : exec_)[(size_t) T];
    std::vector<strata::gpu::Graph*>& segs = (mtp ? msegs_ : segs_)[(size_t) T];
    if (exec != nullptr || !segs.empty()) return true;
    if (!strata::gpu::begin_capture(cs_)) {
        err = std::string("plain model: begin capture: ") + strata::gpu::last_error();
        return false;
    }
    bool ok = false;
    try {
        ok = mtp ? record_mtp(T, cs_, err) : record_window(T, cs_, err);
    } catch (const std::exception& e) {
        err = std::string("plain model: recording the window: ") + e.what();
    }
    if (!ok) {
        strata::gpu::abandon_capture(cs_);
        for (auto* g : segs) strata::gpu::graph_destroy(g);
        segs.clear();
        return false;
    }
    strata::gpu::Graph* g = nullptr;
    if (!strata::gpu::end_capture(cs_, &g)) {
        err = std::string("plain model: end capture: ") + strata::gpu::last_error();
        return false;
    }
    if (win_.segmented()) segs.push_back(g);
    else exec = g;
    return true;
}

// The window's inputs in the mapped staging: the tokens' embedding rows, their step records and rotary positions.
bool PlainModel::stage(const int32_t* tokens, int T, int64_t pos, std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t NH = s.n_head;
    const k::QsaShapes qs = attn_shapes(s, nkv_);
    for (int t = 0; t < T; ++t) {
        if (tokens[t] < 0 || tokens[t] >= n_vocab_) { err = "plain model: a token is out of the vocabulary"; return false; }
        h_tokens_[t] = tokens[t];
        k::qsa_step_fill(h_step_ + (size_t) t * k::kStepCount, pos + t, qs);
        for (int64_t h = 0; h < NH; ++h) h_pos_[(size_t) (t * NH + h)] = (int32_t) (pos + t);
        int32_t* pk = h_pos_ + (size_t) opt_.max_t * NH;
        for (int64_t h = 0; h < qs.n_head_kv; ++h) pk[(size_t) (t * qs.n_head_kv + h)] = (int32_t) (pos + t);
    }
    return true;
}

bool PlainModel::launch_serve(strata::gpu::Graph* g, std::vector<strata::gpu::Graph*>& segs, int64_t rings,
                              const int64_t* layers, int T, std::string& err) {
    win_.reset();
    const bool seg = win_.segmented();
    if (!strata::gpu::graph_launch(seg ? segs[0] : g, cs_)) {
        err = std::string("plain model: launch: ") + strata::gpu::last_error();
        return false;
    }
    (void) strata::gpu::stream_idle(cs_);
    std::vector<int> grps((size_t) rings, 0), tbs((size_t) rings, 0), ns((size_t) rings, T);
    if (seg) win_.set_segments(&segs);
    if (!win_.serve(rings, layers, grps.data(), tbs.data(), ns.data(), &PlainModel::pool_fn, this, cs_, err)) return false;
    if (win_.direct_resident())
        for (int64_t ring = 0; ring < rings; ++ring) disp_.cache_hits += (int64_t) ns[(size_t) ring] * spec_.n_expert_used;
    return true;
}

bool PlainModel::forward(const int32_t* tokens, int T, int64_t pos, int32_t* out, float* logits, std::string& err) {
    const ModelSpec& s = spec_;
    if (T < 1 || T > opt_.max_t || pos < 0 || pos + T > opt_.max_cells) {
        err = "plain model: the window is out of range";
        return false;
    }
    if (!record(T, false, err) || !settle_swaps(false, err) || !stage(tokens, T, pos, err)) return false;
    const int64_t L = s.n_moe_layers();
    std::vector<int64_t> layers((size_t) L);
    for (int64_t l = 0; l < L; ++l) layers[(size_t) l] = l;
    if (!launch_serve(exec_[(size_t) T], segs_[(size_t) T], L, layers.data(), T, err)) return false;
    k::SamplerParams sp = sampler_;
    sp.counter = (uint64_t) pos;   // row t's draw: position pos + t
    k::sample_tokens(logits_, T, (int) n_vocab_, nullptr, 0, sp, m_out_, cs_);
    if (!strata::gpu::stream_sync(cs_)) {
        err = std::string("plain model: the window: ") + strata::gpu::last_error();
        return false;
    }
    if (disp_.failed) {
        err = std::string("plain model: the CPU experts: ") + (disp_.fail ? disp_.fail : "failed");
        return false;
    }
    for (int t = 0; t < T; ++t) out[t] = ((volatile int32_t*) h_out_)[t];
    if (logits != nullptr && !strata::gpu::copy(logits, logits_, (size_t) T * (size_t) n_vocab_ * 4)) {
        err = "plain model: reading the logits failed";
        return false;
    }
    last_T_ = T;
    mtp_h_set_ = false;
    ++windows_;
    tokens_since_adapt_ += T;
    if (!disp_.usage.empty() && windows_ % opt_.adapt_every == 0 && !adapt(err)) return false;
    return true;
}

bool PlainModel::mtp_run(int T, const int32_t* tokens, int64_t pos, int32_t* draft, std::string& err) {
    if (!record(T, true, err) || !settle_swaps(false, err) || !stage(tokens, T, pos, err)) return false;
    const int64_t ml = n_moe_ - 1;
    if (!launch_serve(mexec_[(size_t) T], msegs_[(size_t) T], 1, &ml, T, err)) return false;
    k::SamplerParams sp;   // the drafts are greedy
    sp.greedy = true;
    sp.temperature = 0.0f;
    k::sample_tokens(logits_, 1, (int) (dhead_ != nullptr ? n_dvocab_ : n_vocab_), nullptr, 0, sp, m_out_, cs_);
    if (!strata::gpu::stream_sync(cs_)) {
        err = std::string("plain model: the MTP layer: ") + strata::gpu::last_error();
        return false;
    }
    if (disp_.failed) {
        err = std::string("plain model: the CPU experts: ") + (disp_.fail ? disp_.fail : "failed");
        return false;
    }
    *draft = ((volatile int32_t*) h_out_)[0];
    if (dhead_ != nullptr) *draft = *draft >= 0 && *draft < n_dvocab_ ? dvocab_[(size_t) *draft] : -1;
    return true;
}

bool PlainModel::mtp_chain(int want, int64_t pos, std::string& err) {
    // each step reads the draft before it at the next position, and the previous step's normed row (mtp_h_)
    while ((int) drafts_.size() < want) {
        int32_t d = -1;
        const int64_t p = pos + (int64_t) drafts_.size() - 1;
        if (p >= opt_.max_cells) return true;
        if (!mtp_run(1, &drafts_.back(), p, &d, err)) return false;
        drafts_.push_back(d);
    }
    return true;
}

bool PlainModel::spec_start(int32_t first, int64_t pos, std::string& err) {
    drafts_.clear();
    cur_ = first;
    cur_pos_ = pos;
    if (!has_mtp_ || n_snap_ < 1) return true;
    // the trunk's row before `first`: prompt() leaves it in mtp_h_, a window in hn_
    if (!mtp_h_set_ &&
        !strata::gpu::copy(mtp_h_, hn_ + (size_t) (last_T_ - 1) * (size_t) spec_.n_embd, (size_t) spec_.n_embd * 4)) {
        err = "plain model: the MTP layer's input";
        return false;
    }
    if (pos >= opt_.max_cells) return true;
    int32_t d = -1;
    if (!mtp_run(1, &first, pos, &d, err)) return false;
    drafts_.push_back(d);
    return mtp_chain(n_snap_, pos + 1, err);
}

bool PlainModel::spec_step(const std::vector<int32_t>& stop, int64_t limit, int32_t* emitted, int& n_emit, float* logits,
                           std::string& err) {
    n_emit = 0;
    if (cur_ < 0) { err = "plain model: spec_step before spec_start"; return false; }
    // the window: the pending token and the drafts that fit the context and the limit
    int nd = (int) std::min<int64_t>({(int64_t) drafts_.size(), limit - 1, opt_.max_cells - cur_pos_ - 1});
    nd = std::max(nd, 0);
    int32_t win[k::cpu::MAXT];
    int32_t out[k::cpu::MAXT];
    win[0] = cur_;
    for (int i = 0; i < nd; ++i) win[i + 1] = drafts_[(size_t) i];
    const int T = nd + 1;
    const auto t0 = std::chrono::steady_clock::now();
    if (!forward(win, T, cur_pos_, out, logits, err)) return false;
    const auto t1 = std::chrono::steady_clock::now();
    spec_ms_check_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
    // accepted: the drafts the window's own picks agree with, up to the first stop token among the emitted
    int n = 0;
    while (n < nd && out[n] == win[n + 1]) ++n;
    for (int j = 0; j < n; ++j)
        if (std::find(stop.begin(), stop.end(), out[j]) != stop.end()) { n = j; break; }
    ++spec_windows_;
    spec_drafted_ += nd;
    spec_accepted_ += n;
    for (int j = 0; j <= n; ++j) emitted[j] = out[j];
    n_emit = n + 1;
    // the recurrent states back to the last accepted row's (the attention's later cells are rewritten in turn)
    if (n < T - 1) {
        const ModelSpec& s = spec_;
        const size_t st = (size_t) (s.ssm_d_state * s.ssm_dt_rank * s.ssm_d_state) * 4;
        const size_t cv = (size_t) ((2 * s.ssm_n_group * s.ssm_d_state + s.ssm_d_inner) * (s.ssm_d_conv - 1)) * 4;
        for (size_t g = 0; g < gdn_state_.size(); ++g)
            if (!strata::gpu::copy_async(gdn_state_[g], gdn_snap_[g][(size_t) n], st, cs_) ||
                !strata::gpu::copy_async(gdn_conv_[g], conv_snap_[g][(size_t) n], cv, cs_)) {
                err = std::string("plain model: restoring the recurrent states: ") + strata::gpu::last_error();
                return false;
            }
    }
    // the MTP layer catches up over the accepted rows (row t: the window's trunk row t and the token after it), whose
    // last row drafts the token after the new pending one; then the chain
    const int64_t pos = cur_pos_;
    cur_ = out[n];
    cur_pos_ = pos + n + 1;
    drafts_.clear();
    if (cur_pos_ >= opt_.max_cells) return true;
    if (!strata::gpu::copy_async(mtp_h_, hn_, (size_t) (n + 1) * (size_t) spec_.n_embd * 4, cs_)) {
        err = std::string("plain model: the MTP layer's input: ") + strata::gpu::last_error();
        return false;
    }
    int32_t d = -1;
    if (!mtp_run(n + 1, out, pos + 1, &d, err)) return false;
    drafts_.push_back(d);
    const bool ok = mtp_chain(n_snap_, cur_pos_ + 1, err);
    spec_ms_draft_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    return ok;
}

// One of `layer`'s free slots takes `expert`, read from the expert source (blocking: between windows, or at load).
bool PlainModel::fill_slot(int64_t layer, int32_t expert, std::string& err) {
    auto& f = free_[(size_t) layer];
    if (f.empty()) { err = "plain model: no free slot in the layer"; return false; }
    const int32_t slot = f.back();
    if (!src_->copy_blob(layer, expert, fill_tmp_) ||
        !cache_.fill_slot_blocking(slot, fill_tmp_, err, (int64_t) k::cpu::expert_layout().blob_bytes(layer))) {
        if (err.empty()) err = "plain model: an expert could not be read for the VRAM tier";
        return false;
    }
    f.pop_back();
    res_[(size_t) (layer * spec_.n_expert + expert)] = slot;
    return true;
}

bool PlainModel::save_profile(const std::string& path, std::string& err) const {
    const int64_t L = n_moe_, E = spec_.n_expert;
    std::vector<uint8_t> resident((size_t) (L * E), 0);
    for (size_t i = 0; i < res_.size(); ++i) resident[i] = res_[i] >= 0;
    return write_expert_profile(path, L, E, rank_learned_profile(L, E, resident, heat_, prior_), err);
}

// XeStrata's adaptive tier (generate.cpp's `adapt`): per MoE layer, a missing expert routed at least twice (decayed
// counts) replaces the least-routed resident one when it leads by 1.5, or takes a free slot; at most adapt_swaps a
// round, the largest gains first; then the counts decay.  Runs between windows, so no window reads a slot it changes.
bool PlainModel::adapt(std::string& err, int64_t max_swaps) {
    const int64_t L = n_moe_, E = spec_.n_expert;
    if (!settle_swaps(max_swaps > 0, err)) return false;
    if (!pending_.empty()) return true;   // the last round's copies are still landing: no new round yet
    struct Swap { float gain; int32_t layer, in, out; };   // out < 0: a free slot
    std::vector<Swap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < L; ++l) {
        cand.clear();
        vict.clear();
        const float* u = disp_.usage.data() + l * E;
        const int32_t* r = res_.data() + l * E;
        for (int32_t e = 0; e < (int32_t) E; ++e) {
            if (r[e] < 0) { if (u[e] > 0.0f) cand.emplace_back(u[e], e); }
            else vict.emplace_back(u[e], e);
        }
        // a free slot takes any expert routed since (an empty slot only costs a copy); one held, a missing expert
        // routed at least twice
        const size_t n_free = free_[(size_t) l].size();
        for (size_t i = 0; i < n_free; ++i) vict.emplace_back(-1.0f, -1);
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        size_t keep = 0;
        while (keep < cand.size() && (keep < n_free || cand[keep].first >= 2.0f)) ++keep;
        cand.resize(keep);
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + 1.5f) break;
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
    // and at most adapt_mib_per_token of copies for each token the windows read since the last round: the copies share
    // the RAM's bandwidth with the missed experts the CPU and PCIe read
    const uint64_t blob = k::cpu::expert_layout().max_blob;
    const int64_t budget = blob > 0 && opt_.adapt_mib_per_token > 0
                               ? (int64_t) (((uint64_t) opt_.adapt_mib_per_token << 20) * (uint64_t) tokens_since_adapt_ / blob)
                               : opt_.adapt_swaps;
    tokens_since_adapt_ = 0;
    const int64_t cap = max_swaps > 0 ? max_swaps : std::min<int64_t>(opt_.adapt_swaps, std::max<int64_t>(budget, 1));
    if ((int64_t) swaps.size() > cap) swaps.resize((size_t) cap);
    for (const Swap& w : swaps) {
        if (w.out < 0) {
            if (free_[(size_t) w.layer].empty()) continue;
        } else {   // the evicted expert's slot becomes the layer's free one, which the fill takes
            const int32_t slot = res_[(size_t) (w.layer * E + w.out)];
            res_[(size_t) (w.layer * E + w.out)] = -1;
            free_[(size_t) w.layer].push_back(slot);
        }
        // from the pinned RAM copy on the copy stream, beside the next windows: the evicted expert is out of the plan
        // already, and the new one goes in once the copy is done (settle_swaps); else read into a slot and wait
        const uint8_t* hb = src_->transient(w.layer, w.in) ? nullptr : src_->blob(w.layer, w.in);
        if (hb != nullptr && strata::gpu::is_host_usm(hb)) {
            auto& f = free_[(size_t) w.layer];
            const int32_t slot = f.back();
            if (!strata::gpu::copy_async(cache_.device_slot(slot), hb, k::cpu::expert_layout().blob_bytes(w.layer),
                                         swap_cs_)) {
                err = std::string("plain model: the adaptive tier's copy: ") + strata::gpu::last_error();
                return false;
            }
            f.pop_back();
            pending_.push_back({w.layer, w.in, slot});
        } else if (!fill_slot(w.layer, w.in, err)) {
            return false;
        }
        ++swaps_total_;
    }
    if (!pending_.empty() && !strata::gpu::event_record(swap_ev_, swap_cs_)) {
        err = std::string("plain model: the adaptive tier's copies: ") + strata::gpu::last_error();
        return false;
    }
    for (float& v : disp_.usage) v *= opt_.adapt_decay;
    return true;
}

bool PlainModel::settle_swaps(bool wait, std::string& err) {
    if (pending_.empty()) return true;
    if (wait ? !strata::gpu::event_sync(swap_ev_) : !strata::gpu::event_query(swap_ev_)) {
        if (!wait) return true;
        err = std::string("plain model: the adaptive tier's copies: ") + strata::gpu::last_error();
        return false;
    }
    for (const PendingSwap& p : pending_) res_[(size_t) (p.layer * spec_.n_expert + p.expert)] = p.slot;
    pending_.clear();
    return true;
}

}  // namespace strata::core
