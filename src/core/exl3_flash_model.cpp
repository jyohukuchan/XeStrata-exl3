// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Explicit Qwen3.8 EXL3 Flash-model ownership and stage preparation.
#include "strata/core/exl3_flash_model.hpp"

#include "strata/core/exl3_expert_runtime.hpp"
#include "strata/core/exl3_expert_source.hpp"
#include "strata/core/exl3_ple.hpp"
#include "strata/core/exl3_weight_set.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
namespace {

using Archive = strata::artifact::exl3::Archive;
using ExpertTensorNames = strata::artifact::exl3::ExpertTensorNames;
using ModelMap = strata::artifact::exl3::ModelMap;
using ProjectionKind = strata::artifact::exl3::ProjectionKind;

constexpr const char* kPlePending = "Qwen3.8 PLE ngram sidecar binding is pending in ModelMap";

bool product_size(int64_t a, int64_t b, size_t& result) {
    if (a <= 0 || b <= 0 || static_cast<uint64_t>(a) > std::numeric_limits<size_t>::max() /
                                      static_cast<uint64_t>(b))
        return false;
    result = static_cast<size_t>(a) * static_cast<size_t>(b);
    return true;
}

bool make_expert_names(const ModelMap& map, int64_t layers, int64_t experts,
                       std::vector<ExpertTensorNames>& result, std::string& err) {
    size_t count = 0;
    if (!product_size(layers, experts, count)) {
        err = "EXL3 Flash: expert layer/name table dimensions overflow";
        return false;
    }
    result.assign(count, {});
    for (const auto& projection : map.projections) {
        if (projection.layer < 0 || projection.expert < 0 || projection.layer >= layers || projection.expert >= experts)
            continue;
        std::string* destination = nullptr;
        switch (projection.kind) {
        case ProjectionKind::expert_gate: destination = &result[static_cast<size_t>(projection.layer * experts + projection.expert)].gate; break;
        case ProjectionKind::expert_up: destination = &result[static_cast<size_t>(projection.layer * experts + projection.expert)].up; break;
        case ProjectionKind::expert_down: destination = &result[static_cast<size_t>(projection.layer * experts + projection.expert)].down; break;
        default: break;
        }
        if (destination == nullptr) continue;
        if (!destination->empty() && *destination != projection.source_name) {
            err = "EXL3 Flash: duplicate expert projection mapping for layer " + std::to_string(projection.layer) +
                  " expert " + std::to_string(projection.expert);
            return false;
        }
        *destination = projection.source_name;
    }
    for (int64_t layer = 0; layer < layers; ++layer) {
        for (int64_t expert = 0; expert < experts; ++expert) {
            const auto& names = result[static_cast<size_t>(layer * experts + expert)];
            if (names.gate.empty() || names.up.empty() || names.down.empty()) {
                err = "EXL3 Flash: expert mapping is incomplete at layer " + std::to_string(layer) + " expert " +
                      std::to_string(expert);
                return false;
            }
        }
    }
    return true;
}

std::filesystem::path find_ple_sidecar(const Archive& archive, const std::string& requested) {
    if (!requested.empty()) {
        const std::filesystem::path path(requested);
        return path.is_absolute() ? path : archive.directory() / path;
    }
    for (const auto& sidecar : archive.sidecars())
        if (sidecar.role == strata::artifact::exl3::SidecarRole::specialized_ngram_codec)
            return archive.directory() / sidecar.file;
    return {};
}

bool capacity_error(const std::string& error) {
    return error.find("cache") != std::string::npos || error.find("allocation") != std::string::npos ||
           error.find("resident") != std::string::npos || error.find("VRAM") != std::string::npos;
}

uint64_t gdn_state_floats(const ModelGeometry& geometry) {
    return static_cast<uint64_t>(geometry.ssm_state_size) * static_cast<uint64_t>(geometry.ssm_v_heads) *
               static_cast<uint64_t>(geometry.ssm_state_size) +
           static_cast<uint64_t>(geometry.ssm_conv_channels) * static_cast<uint64_t>(geometry.ssm_d_conv - 1);
}

void point_gdn(const ModelGeometry& geometry, int64_t layer, SessionState& session) {
    if (is_qsa_layer(geometry, layer)) return;
    int64_t ordinal = 0;
    for (int64_t l = 0; l < layer; ++l)
        if (!is_qsa_layer(geometry, l)) ++ordinal;
    session.gdn.state = session.gdn_state +
                        static_cast<size_t>(ordinal - session.gdn_ord0) * gdn_state_floats(geometry);
    session.gdn.conv_state = session.gdn.state +
                             static_cast<uint64_t>(geometry.ssm_state_size) * geometry.ssm_v_heads *
                                 geometry.ssm_state_size;
}

QsaState& qsa_for(const ModelGeometry& geometry, int64_t layer, SessionState& session) {
    int64_t ordinal = 0;
    for (int64_t l = 0; l < layer; ++l)
        if (is_qsa_layer(geometry, l)) ++ordinal;
    return session.qsa_states[ordinal];
}

void stage_token(const ModelGeometry& geometry, int64_t pos, int32_t pos_base, SessionState& session) {
    strata::kernels::QsaShapes shapes = strata::kernels::qsa_real_shapes();
    shapes.n_head = geometry.n_head;
    shapes.n_head_kv = geometry.n_head_kv;
    shapes.head_dim = geometry.head_dim;
    shapes.idx_n_head = geometry.idx_q_heads;
    shapes.idx_dim = geometry.idx_key_dim;
    for (int64_t j = 0; j < session.qsa_alloc; ++j) {
        QsaState& state = session.qsa_states[session.qsa_ord0 + j];
        strata::kernels::qsa_step_fill(state.host_step, pos, shapes);
        for (int64_t h = 0; h < geometry.n_head; ++h)
            state.host_pos[h] = static_cast<int32_t>(pos_base + pos);
    }
}

void destroy_graphs(std::vector<strata::gpu::Graph*>& graphs) noexcept {
    for (auto*& graph : graphs) {
        if (graph != nullptr) strata::gpu::graph_destroy(graph);
        graph = nullptr;
    }
    graphs.clear();
}

}  // namespace

struct Exl3FlashModel::Impl {
    struct Stage {
        int device = -1;
        int64_t layer_begin = 0;
        int64_t layer_end = 0;
        WeightTable table;
        std::unique_ptr<Exl3WeightSet> weights;
        std::unique_ptr<Exl3ExpertSource> source;
        std::unique_ptr<strata::kernels::cpu::ExpertPool> pool;
        std::unique_ptr<Exl3ExpertRuntime> runtime;
        SessionState session;
        void* session_arena = nullptr;
        Doorbell doorbell;
        void* parts_device = nullptr;
        std::vector<float> miss_output;
        std::vector<float> merge_output;
        float* embedding_device = nullptr;
        std::vector<strata::gpu::Graph*> pre_graphs;
        std::vector<strata::gpu::Graph*> post_graphs;
    };

    Exl3FlashModelOptions options;
    std::shared_ptr<const Archive> archive;
    strata::ModelSpec spec;
    ModelGeometry geometry;
    ModelMap map;
    std::vector<ExpertTensorNames> expert_names;
    std::vector<Stage> stages;
    std::unique_ptr<NativeEmbed> embedding;
    std::unique_ptr<Exl3Ple> ple;
    std::unique_ptr<strata::kernels::PleTable> ple_table;
    int embedding_device = -1;
    int ple_device = -1;
    int ple_stage_index = -1;
    void* logits_device = nullptr;
    std::vector<float> logits_host;
    std::vector<float> bridge_host;
    bool initialized = false;
    bool seeded = false;
    int64_t next_position = 0;
    bool previous_fused_gr = false;
    bool previous_shared_early = false;
    bool policy_saved = false;
    strata::kernels::RopeScaling previous_rope;
    bool previous_gdn_silu = false;
    bool model_globals_saved = false;
    bool opened = false;
};

Exl3FlashModel::Exl3FlashModel() : impl_(std::make_unique<Impl>()) {}

Exl3FlashModel::~Exl3FlashModel() { close(); }

bool Exl3FlashModel::open(const std::filesystem::path& directory, const Exl3FlashModelOptions& options,
                          std::string& err) {
    close();
    err.clear();
    struct OpenCleanup {
        Exl3FlashModel* owner;
        ~OpenCleanup() { if (owner != nullptr) owner->close(); }
    } cleanup{this};
    try {
        if (directory.empty()) { err = "EXL3 Flash: archive directory is required"; return false; }
        if (options.include_mtp) {
            err = "EXL3 Flash: Qwen3.8 MTP is not bound by the initial AR coordinator";
            return false;
        }
        if (options.vision) {
            err = "EXL3 Flash: vision input is not supported by the initial AR coordinator";
            return false;
        }
        if (options.max_rows != 1) {
            err = "EXL3 Flash: the initial coordinator accepts exactly one autoregressive row";
            return false;
        }
        if (options.stage_devices.empty()) {
            const int count = Runtime::count();
            if (count <= 0) { err = "EXL3 Flash: no runtime GPU is available for stage construction"; return false; }
            impl_->options = options;
            impl_->options.stage_devices.resize(static_cast<size_t>(count));
            for (int i = 0; i < count; ++i) impl_->options.stage_devices[static_cast<size_t>(i)] = i;
        } else {
            impl_->options = options;
        }
        if (impl_->options.stage_devices.empty()) {
            err = "EXL3 Flash: at least one stage device is required";
            return false;
        }
        std::set<int> unique_devices;
        for (const int device : impl_->options.stage_devices) {
            if (device < 0 || device >= Runtime::count() || !unique_devices.insert(device).second) {
                err = "EXL3 Flash: stage device ordinals must be valid and unique";
                return false;
            }
        }

        impl_->archive = std::make_shared<Archive>(directory);
        if (!strata::artifact::exl3::read_exl3_model_spec(*impl_->archive, impl_->spec, err, false)) return false;
        if (impl_->spec.arch != strata::ModelArch::Qwen4Exp) {
            err = "EXL3 Flash: the initial coordinator requires the Qwen3.8 Qwen4Exp architecture";
            return false;
        }
        if (!geometry_from_spec(impl_->spec, impl_->geometry, err)) {
            err = "EXL3 Flash: " + err;
            return false;
        }
        const int64_t context_cells = options.max_cells > 0 ? options.max_cells : impl_->spec.context_length;
        if (context_cells <= 0 || context_cells > impl_->spec.context_length) {
            err = "EXL3 Flash: max_cells must be within max_position_embeddings";
            return false;
        }
        impl_->options.max_cells = context_cells;
        {
            using RST = strata::kernels::RopeScalingType;
            strata::kernels::RopeScaling scaling;
            scaling.freq_base = impl_->spec.rope_base;
            const std::string& type = impl_->spec.rope_scaling_type;
            if (type.empty() || type == "none") scaling.type = RST::None;
            else if (type == "linear") scaling.type = RST::Linear;
            else if (type == "yarn") scaling.type = RST::YaRN;
            else {
                err = "EXL3 Flash: unsupported rope scaling type " + type;
                return false;
            }
            if (scaling.type != RST::None) {
                scaling.factor = impl_->spec.rope_scaling_factor > 1.0 ? impl_->spec.rope_scaling_factor : 1.0;
                scaling.orig_ctx = impl_->spec.rope_scaling_orig_ctx >= 1.0
                                       ? impl_->spec.rope_scaling_orig_ctx
                                       : static_cast<double>(impl_->spec.context_length);
                scaling.ext_factor = scaling.type == RST::YaRN ? 1.0 : 0.0;
            }
            if (const char* why = strata::kernels::rope_scaling_invalid(scaling)) {
                err = std::string("EXL3 Flash: rope scaling: ") + why;
                return false;
            }
            impl_->previous_rope = strata::kernels::rope_scaling();
            impl_->previous_gdn_silu = strata::kernels::gdn_silu_gate();
            impl_->model_globals_saved = true;
            strata::kernels::rope_scaling_set(scaling);
            // Qwen3.8/Qwen4Exp uses sigmoid(z) in the GDN closing gate;
            // capture bakes this choice into every GDN graph.
            strata::kernels::gdn_set_silu_gate(false);
        }
        if (!strata::artifact::exl3::build_model_map(*impl_->archive, impl_->map, err, false)) {
            err = "EXL3 Flash: " + err;
            return false;
        }
        for (const auto& unsupported : impl_->map.unsupported) {
            if (unsupported != kPlePending) {
                err = "EXL3 Flash: model map is not runnable: " + unsupported;
                return false;
            }
        }
        if (!make_expert_names(impl_->map, impl_->geometry.n_layers, impl_->geometry.n_expert,
                               impl_->expert_names, err)) return false;
        const int64_t routed_k = options.routed_k > 0 ? options.routed_k : impl_->map.geometry.experts_per_token;
        if (routed_k <= 0 || routed_k > impl_->geometry.n_expert) {
            err = "EXL3 Flash: routed expert count is outside model geometry";
            return false;
        }
        impl_->options.routed_k = routed_k;

        // The embedding is one shared raw table.  NativeEmbed creates a device
        // alias on this first stage and prepare_stream() creates other-stage
        // copies before a graph captures the gather.
        const auto layers_pos = impl_->map.source_prefix.rfind(".layers");
        if (layers_pos == std::string::npos) {
            err = "EXL3 Flash: ModelMap has no canonical model root for embedding";
            return false;
        }
        const std::string embedding_name = options.embedding_name.empty()
                                               ? impl_->map.source_prefix.substr(0, layers_pos) + ".embed_tokens"
                                               : options.embedding_name;
        impl_->embedding = std::make_unique<NativeEmbed>();
        {
            const OnDevice on(impl_->options.stage_devices.front());
            if (!impl_->embedding->load_exl3(*impl_->archive, embedding_name, impl_->geometry.n_embd,
                                             impl_->map.geometry.vocab_size, err)) {
                err = "EXL3 Flash: embedding: " + err;
                return false;
            }
        }
        impl_->embedding_device = impl_->options.stage_devices.front();

        const int64_t layer_count = impl_->geometry.n_layers;
        const int stage_count = static_cast<int>(impl_->options.stage_devices.size());
        if (stage_count > layer_count) {
            err = "EXL3 Flash: stage count cannot exceed the number of model layers";
            return false;
        }
        // Exl3WeightSet::table_ borrows the Stage's WeightTable.  Reserve the
        // complete vector before any Stage is constructed so later stages
        // cannot move an earlier table while it is already registered.
        impl_->stages.reserve(static_cast<size_t>(stage_count));
        for (int stage_index = 0; stage_index < stage_count; ++stage_index) {
            Impl::Stage stage;
            stage.device = impl_->options.stage_devices[static_cast<size_t>(stage_index)];
            stage.layer_begin = layer_count * stage_index / stage_count;
            stage.layer_end = layer_count * (stage_index + 1) / stage_count;
            stage.weights = std::make_unique<Exl3WeightSet>();
            stage.source = std::make_unique<Exl3ExpertSource>();
            stage.pool = std::make_unique<strata::kernels::cpu::ExpertPool>();
            stage.runtime = std::make_unique<Exl3ExpertRuntime>();
            impl_->stages.push_back(std::move(stage));
            Impl::Stage& current = impl_->stages.back();
            const OnDevice on(current.device);
            Exl3WeightSetOptions weight_options;
            weight_options.device = current.device;
            weight_options.layer_begin = current.layer_begin;
            weight_options.layer_end = current.layer_end;
            weight_options.include_head = current.layer_end == layer_count;
            weight_options.include_final_hc = current.layer_end == layer_count;
            weight_options.max_rows = options.max_rows;
            if (!current.weights->load(*impl_->archive, impl_->map, current.table, weight_options, err)) {
                err = "EXL3 Flash: stage " + std::to_string(stage_index) + " dense weights: " + err;
                return false;
            }
            if (!current.source->open(impl_->archive, layer_count, impl_->geometry.n_expert,
                                      impl_->expert_names, err)) {
                err = "EXL3 Flash: stage " + std::to_string(stage_index) + " expert source: " + err;
                return false;
            }
            std::vector<Exl3ExpertRuntime::ResidentPair> resident;
            for (int64_t layer = current.layer_begin; layer < current.layer_end; ++layer)
                for (int64_t expert = 0; expert < impl_->geometry.n_expert; ++expert)
                    resident.emplace_back(static_cast<int32_t>(layer), static_cast<int32_t>(expert));
            if (!current.runtime->open(*current.source, *current.pool, options.max_rows, routed_k, resident, err)) {
                const std::string first_error = err;
                if (!options.allow_cpu_expert_miss || !capacity_error(first_error)) {
                    err = "EXL3 Flash: stage " + std::to_string(stage_index) + " expert runtime: " + first_error;
                    return false;
                }
                current.runtime->close();
                resident.clear();
                if (!current.runtime->open(*current.source, *current.pool, options.max_rows, routed_k, resident, err)) {
                    err = "EXL3 Flash: stage " + std::to_string(stage_index) +
                          " expert cache does not fit and CPU-miss fallback failed: " + err;
                    return false;
                }
            }
        }

        // The Qwen3.8 PLE owner is stage-local at trunk layer 1, while the
        // sidecar row table remains one host reader shared by the session.
        const std::filesystem::path ple_path = find_ple_sidecar(*impl_->archive, options.ple_sidecar);
        if (ple_path.empty()) {
            err = "EXL3 Flash: the Qwen3.8 specialized n-gram PLE sidecar is required";
            return false;
        }
        int ple_owner_stage_index = -1;
        for (size_t i = 0; i < impl_->stages.size(); ++i) {
            if (impl_->stages[i].layer_begin <= 1 && 1 < impl_->stages[i].layer_end) {
                ple_owner_stage_index = static_cast<int>(i);
                break;
            }
        }
        if (ple_owner_stage_index < 0) {
            err = "EXL3 Flash: no stage owns the required Qwen3.8 PLE layer 1";
            return false;
        }
        impl_->ple_stage_index = ple_owner_stage_index;
        impl_->ple = std::make_unique<Exl3Ple>();
        impl_->ple_device = impl_->stages[static_cast<size_t>(ple_owner_stage_index)].device;
        {
            const OnDevice on(impl_->ple_device);
            if (!impl_->ple->load(*impl_->archive, impl_->map, impl_->ple_device, options.max_rows, err)) {
                err = "EXL3 Flash: PLE projection owner: " + err;
                return false;
            }
        }
        impl_->ple_table = std::make_unique<strata::kernels::PleTable>();
        if (!impl_->ple_table->open_exl3(ple_path.string(), err)) {
            err = "EXL3 Flash: PLE sidecar table: " + err;
            return false;
        }

        const size_t residual_elements = static_cast<size_t>(impl_->geometry.hc) *
                                         static_cast<size_t>(impl_->geometry.n_embd);
        const size_t parts_elements = static_cast<size_t>(routed_k) *
                                      static_cast<size_t>(impl_->geometry.n_embd);
        impl_->bridge_host.resize(residual_elements);
        impl_->logits_host.resize(static_cast<size_t>(impl_->map.geometry.vocab_size));
        for (size_t stage_index = 0; stage_index < impl_->stages.size(); ++stage_index) {
            auto& stage = impl_->stages[stage_index];
            const OnDevice on(stage.device);
            const uint64_t session_size = session_bytes(impl_->geometry, impl_->options.max_cells, routed_k,
                                                        stage.layer_begin, stage.layer_end);
            stage.session_arena = strata::gpu::alloc_device(static_cast<size_t>(session_size));
            if (stage.session_arena == nullptr || session_init(impl_->geometry, impl_->options.max_cells, routed_k,
                                                               stage.session_arena, stage.session, stage.layer_begin,
                                                               stage.layer_end) == 0) {
                err = "EXL3 Flash: stage " + std::to_string(stage_index) + " SessionState allocation failed";
                return false;
            }
            if (doorbell_init(impl_->geometry, routed_k, stage.doorbell) == 0) {
                err = "EXL3 Flash: stage " + std::to_string(stage_index) + " expert doorbell allocation failed";
                return false;
            }
            stage.session.db = &stage.doorbell;
            stage.parts_device = strata::gpu::alloc_device(parts_elements * sizeof(float));
            if (stage.parts_device == nullptr) {
                err = "EXL3 Flash: stage " + std::to_string(stage_index) + " expert parts allocation failed";
                return false;
            }
            stage.miss_output.assign(parts_elements, 0.0f);
            stage.merge_output.assign(parts_elements, 0.0f);
            if (stage_index == 0) {
                stage.embedding_device = static_cast<float*>(strata::gpu::alloc_device(
                    static_cast<size_t>(impl_->geometry.n_embd) * sizeof(float)));
                if (stage.embedding_device == nullptr) {
                    err = "EXL3 Flash: embedding activation allocation failed";
                    return false;
                }
            }
            session_zero(stage.session, impl_->geometry, nullptr, nullptr);
            if (stage_index == 0) {
                impl_->embedding->gather_one(0, stage.embedding_device, nullptr);
                session_zero(stage.session, impl_->geometry, stage.embedding_device, nullptr);
            }
        }
        {
            auto& ple_owner_stage = impl_->stages[static_cast<size_t>(ple_owner_stage_index)];
            const OnDevice on(ple_owner_stage.device);
            ple_owner_stage.session.ple.table = impl_->ple_table.get();
            ple_owner_stage.session.ple.w = impl_->ple->weights();
            ple_owner_stage.session.ple.key_exl3 = impl_->ple->key();
            ple_owner_stage.session.ple.value_exl3 = impl_->ple->value();
            ple_owner_stage.session.ple.pre_key_out = impl_->ple->pre_key();
            ple_owner_stage.session.ple.pre_value_out = impl_->ple->pre_value();
            ple_owner_stage.session.ple.token = &ple_owner_stage.session.ple_token;
            ple_owner_stage.session.ple.prev = ple_owner_stage.session.ple_prev;
            ple_owner_stage.session.ple.hist = ple_owner_stage.session.ple_hist;
            ple_owner_stage.session.ple.consts = strata::kernels::ple_artifact_consts();
            ple_owner_stage.session.ple.emb_dev = static_cast<float*>(strata::gpu::alloc_device(
                static_cast<size_t>(strata::kernels::NG_N_EMBD) * sizeof(float)));
            ple_owner_stage.session.ple.emb_host = static_cast<float*>(strata::gpu::alloc_host(
                static_cast<size_t>(strata::kernels::NG_N_EMBD) * sizeof(float)));
            ple_owner_stage.session.ple.scratch = static_cast<float*>(strata::gpu::alloc_device(
                static_cast<size_t>(strata::core::ple_run_scratch_bytes())));
            if (ple_owner_stage.session.ple.emb_dev == nullptr || ple_owner_stage.session.ple.emb_host == nullptr ||
                ple_owner_stage.session.ple.scratch == nullptr) {
                err = "EXL3 Flash: PLE token workspace allocation failed";
                return false;
            }
        }

        {
            const auto& last = impl_->stages.back();
            const OnDevice on(last.device);
            impl_->logits_device = strata::gpu::alloc_device(impl_->logits_host.size() * sizeof(float));
            if (impl_->logits_device == nullptr) {
                err = "EXL3 Flash: output logits allocation failed";
                return false;
            }
        }

        // The initial coordinator captures the unfused, non-early form.  The
        // selected values are restored after capture; graph nodes retain the
        // correct branch and later callers do not inherit this policy.
        impl_->previous_fused_gr = layer_fused_gr();
        impl_->previous_shared_early = layer_shared_early();
        impl_->policy_saved = true;
        layer_set_fused_gr(false);
        layer_set_shared_early(false);
        for (size_t stage_index = 0; stage_index < impl_->stages.size(); ++stage_index) {
            auto& stage = impl_->stages[stage_index];
            const OnDevice on(stage.device);
            stage.pre_graphs.resize(static_cast<size_t>(stage.layer_end - stage.layer_begin), nullptr);
            stage.post_graphs.resize(static_cast<size_t>(stage.layer_end - stage.layer_begin), nullptr);
            if (!stage.weights->prepare_stream(nullptr, err)) {
                err = "EXL3 Flash: preparing stage graph stream: " + err;
                return false;
            }
            stage_token(impl_->geometry, 0, 0, stage.session);
            if (stage.session.ple.table != nullptr) {
                stage.session.ple_token = 0;
                stage.session.ple_prev[0] = -1;
                stage.session.ple_prev[1] = -1;
            }
            strata::gpu::Stream capture_stream = strata::gpu::stream_create();
            if (capture_stream == nullptr) {
                err = "EXL3 Flash: graph capture stream creation failed";
                return false;
            }
            if (!stage.weights->prepare_stream(capture_stream, err) ||
                (stage.session.ple.table != nullptr && !impl_->ple->prepare_stream(capture_stream, err))) {
                strata::gpu::stream_destroy(capture_stream);
                err = "EXL3 Flash: preparing graph owners: " + err;
                return false;
            }
            if (stage.session.ple.table != nullptr) {
                if (!ple_stage_token(stage.session.ple, capture_stream, err) ||
                    !strata::gpu::stream_sync(capture_stream)) {
                    strata::gpu::stream_destroy(capture_stream);
                    err = "EXL3 Flash: preparing PLE capture token: " + err;
                    return false;
                }
            }
            for (int64_t layer = stage.layer_begin; layer < stage.layer_end; ++layer) {
                point_gdn(impl_->geometry, layer, stage.session);
                QsaState& qsa = qsa_for(impl_->geometry, layer, stage.session);
                const size_t local = static_cast<size_t>(layer - stage.layer_begin);
                if (!strata::gpu::begin_capture(capture_stream)) {
                    err = "EXL3 Flash: begin pre graph capture failed at layer " + std::to_string(layer);
                    break;
                }
                if (!block_layer_pre(stage.table, impl_->geometry, layer, 0, 0, stage.session.gdn, qsa,
                                     stage.session.qsa_bufs, stage.session.moe, routed_k, stage.session.block,
                                     capture_stream, err, &stage.doorbell,
                                     stage.session.ple.table != nullptr ? &stage.session.ple : nullptr)) {
                    strata::gpu::abandon_capture(capture_stream);
                    err = "EXL3 Flash: pre graph layer " + std::to_string(layer) + ": " + err;
                    break;
                }
                if (!strata::gpu::end_capture(capture_stream, &stage.pre_graphs[local])) {
                    err = "EXL3 Flash: end pre graph layer " + std::to_string(layer) + ": " + strata::gpu::last_error();
                    break;
                }
                if (!strata::gpu::begin_capture(capture_stream)) {
                    err = "EXL3 Flash: begin post graph capture failed at layer " + std::to_string(layer);
                    break;
                }
                if (!block_layer_post(stage.table, impl_->geometry, layer, routed_k, stage.session.moe,
                                      stage.session.block, static_cast<const float*>(stage.parts_device),
                                      capture_stream, err)) {
                    strata::gpu::abandon_capture(capture_stream);
                    err = "EXL3 Flash: post graph layer " + std::to_string(layer) + ": " + err;
                    break;
                }
                if (!strata::gpu::end_capture(capture_stream, &stage.post_graphs[local])) {
                    err = "EXL3 Flash: end post graph layer " + std::to_string(layer) + ": " + strata::gpu::last_error();
                    break;
                }
            }
            strata::gpu::stream_destroy(capture_stream);
            if (!err.empty()) return false;
        }
        layer_set_fused_gr(impl_->previous_fused_gr);
        layer_set_shared_early(impl_->previous_shared_early);
        impl_->policy_saved = false;
        impl_->opened = true;
        impl_->initialized = false;
        cleanup.owner = nullptr;
        return true;
    } catch (const std::exception& e) {
        err = std::string("EXL3 Flash: ") + e.what();
        return false;
    }
}

bool Exl3FlashModel::close(std::string& err) noexcept {
    err.clear();
    if (!impl_) return true;
    bool ok = true;
    try {
        for (auto it = impl_->stages.rbegin(); it != impl_->stages.rend(); ++it) {
            auto& stage = *it;
            const OnDevice on(stage.device);
            if (stage.runtime) Runtime::at(stage.device).finish();
            destroy_graphs(stage.pre_graphs);
            destroy_graphs(stage.post_graphs);
            session_release(stage.session);
            if (stage.session.qsa_states != nullptr) {
                for (int64_t ordinal = 0; ordinal < impl_->geometry.n_qsa_layers(); ++ordinal) {
                    if (stage.session.qsa_states[ordinal].host_step != nullptr)
                        strata::gpu::free(stage.session.qsa_states[ordinal].host_step);
                    if (stage.session.qsa_states[ordinal].host_pos != nullptr)
                        strata::gpu::free(stage.session.qsa_states[ordinal].host_pos);
                }
                delete[] stage.session.qsa_states;
                stage.session.qsa_states = nullptr;
            }
            if (stage.session.ple.emb_dev != nullptr) strata::gpu::free(stage.session.ple.emb_dev);
            if (stage.session.ple.emb_host != nullptr) strata::gpu::free(stage.session.ple.emb_host);
            if (stage.session.ple.scratch != nullptr) strata::gpu::free(stage.session.ple.scratch);
            stage.session.ple.emb_dev = nullptr;
            stage.session.ple.emb_host = nullptr;
            stage.session.ple.scratch = nullptr;
            if (stage.embedding_device != nullptr) strata::gpu::free(stage.embedding_device);
            if (stage.parts_device != nullptr) strata::gpu::free(stage.parts_device);
            if (stage.session_arena != nullptr) strata::gpu::free(stage.session_arena);
            stage.embedding_device = nullptr;
            stage.parts_device = nullptr;
            stage.session_arena = nullptr;
            doorbell_free(stage.doorbell);
            if (stage.runtime) stage.runtime->close();
            stage.runtime.reset();
            stage.source.reset();
            stage.pool.reset();
            if (stage.weights) stage.weights->close();
            stage.weights.reset();
        }
        impl_->stages.clear();
        if (impl_->logits_device != nullptr) {
            const OnDevice on(impl_->options.stage_devices.empty() ? -1 : impl_->options.stage_devices.back());
            strata::gpu::free(impl_->logits_device);
            impl_->logits_device = nullptr;
        }
        if (impl_->ple) {
            const OnDevice on(impl_->ple_device);
            impl_->ple->close();
            impl_->ple.reset();
        }
        impl_->ple_table.reset();
        if (impl_->embedding) {
            const OnDevice on(impl_->embedding_device);
            impl_->embedding.reset();
        }
        impl_->archive.reset();
        impl_->expert_names.clear();
        impl_->map = {};
        impl_->spec = {};
        impl_->geometry = {};
        impl_->embedding_device = impl_->ple_device = -1;
        impl_->ple_stage_index = -1;
        if (impl_->policy_saved) {
            layer_set_fused_gr(impl_->previous_fused_gr);
            layer_set_shared_early(impl_->previous_shared_early);
            impl_->policy_saved = false;
        }
        if (impl_->model_globals_saved) {
            strata::kernels::rope_scaling_set(impl_->previous_rope);
            strata::kernels::gdn_set_silu_gate(impl_->previous_gdn_silu);
            impl_->model_globals_saved = false;
        }
        impl_->opened = false;
        impl_->initialized = false;
        impl_->seeded = false;
        impl_->next_position = 0;
    } catch (const std::exception& e) {
        ok = false;
        err = std::string("EXL3 Flash: close: ") + e.what();
        if (impl_->policy_saved) {
            try {
                layer_set_fused_gr(impl_->previous_fused_gr);
                layer_set_shared_early(impl_->previous_shared_early);
                impl_->policy_saved = false;
            } catch (...) {
            }
        }
        if (impl_->model_globals_saved) {
            try {
                strata::kernels::rope_scaling_set(impl_->previous_rope);
                strata::kernels::gdn_set_silu_gate(impl_->previous_gdn_silu);
                impl_->model_globals_saved = false;
            } catch (...) {
            }
        }
    }
    return ok;
}

void Exl3FlashModel::close() noexcept {
    std::string ignored;
    (void) close(ignored);
}

bool Exl3FlashModel::prepare_stream(void* stream, std::string& err) {
    err.clear();
    if (!loaded()) { err = "EXL3 Flash: open is required before prepare_stream"; return false; }
    try {
        Runtime* owner = stream == nullptr ? &Runtime::at(current_device()) : Runtime::owner_of_stream(stream);
        if (owner == nullptr) { err = "EXL3 Flash: stream has no Xe runtime owner"; return false; }
        const int device = owner->ordinal();
        bool found = false;
        for (auto& stage : impl_->stages) {
            if (stage.device != device) continue;
            found = true;
            const OnDevice on(device);
            if (!stage.weights->prepare_stream(stream, err)) return false;
            if (impl_->ple && impl_->ple_device == device && !impl_->ple->prepare_stream(stream, err)) return false;
        }
        if (impl_->embedding && !impl_->embedding->prepare_stream(stream, err)) return false;
        if (!found) {
            err = "EXL3 Flash: stream belongs to a device with no model stage";
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("EXL3 Flash: preparing stream: ") + e.what();
        return false;
    }
}

bool Exl3FlashModel::prepare_expert_window(size_t stage_index, int64_t layer, const float* host_x, int64_t ldx,
                                           const int32_t* routed_ids, int64_t n_tokens, float* cpu_output, int64_t ldy,
                                           void* stream, const float* device_x, float* device_output, sycl::event& done,
                                           std::string& err) {
    err.clear();
    if (!loaded() || stage_index >= impl_->stages.size()) {
        err = "EXL3 Flash: invalid stage or unopened model";
        return false;
    }
    if (!validate_token_request(n_tokens, 1, err)) return false;
    auto& stage = impl_->stages[stage_index];
    if (layer < stage.layer_begin || layer >= stage.layer_end) {
        err = "EXL3 Flash: expert layer does not belong to the selected stage";
        return false;
    }
    try {
        Runtime& runtime = Runtime::at(stage.device);
        if (stream != nullptr && Runtime::owner_of_stream(stream) != &runtime) {
            err = "EXL3 Flash: expert stream belongs to another device stage";
            return false;
        }
        const OnDevice on(stage.device);
        sycl::queue& queue = runtime.stream(stream);
        if (!stage.runtime->prepare_window(layer, host_x, ldx, routed_ids, n_tokens, cpu_output, ldy, err) ||
            !stage.runtime->enqueue_hit_only(queue, device_x, ldx, device_output, ldy, done, {}, err)) {
            err = "EXL3 Flash: expert window: " + err;
            return false;
        }
        if (!stage.runtime->run_cpu(err)) {
            const std::string cpu_error = err;
            if (stage.runtime->pending_gpu()) {
                std::string drain_error;
                if (!stage.runtime->wait_gpu(done, drain_error) && !drain_error.empty())
                    err = cpu_error + "; GPU drain: " + drain_error;
                else
                    err = cpu_error;
            } else {
                err = cpu_error;
            }
            err = "EXL3 Flash: expert window: " + err;
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("EXL3 Flash: expert window: ") + e.what();
        return false;
    }
}

bool Exl3FlashModel::wait_expert_window(size_t stage_index, sycl::event& done, std::string& err) {
    err.clear();
    if (!loaded() || stage_index >= impl_->stages.size()) {
        err = "EXL3 Flash: invalid stage or unopened model";
        return false;
    }
    try {
        const OnDevice on(impl_->stages[stage_index].device);
        return impl_->stages[stage_index].runtime->wait_gpu(done, err);
    } catch (const std::exception& e) {
        err = std::string("EXL3 Flash: waiting for expert window: ") + e.what();
        return false;
    }
}

bool Exl3FlashModel::reset(std::string& err) {
    err.clear();
    if (!loaded()) { err = "EXL3 Flash: model is not open"; return false; }
    try {
        for (auto& stage : impl_->stages) {
            const OnDevice on(stage.device);
            session_zero(stage.session, impl_->geometry, nullptr, nullptr);
            if (!strata::gpu::stream_sync(nullptr)) {
                err = std::string("EXL3 Flash: resetting stage state: ") + strata::gpu::last_error();
                return false;
            }
            stage.session.ple_prev[0] = -1;
            stage.session.ple_prev[1] = -1;
            stage.session.ple_token = -1;
            std::fill(stage.miss_output.begin(), stage.miss_output.end(), 0.0f);
            std::fill(stage.merge_output.begin(), stage.merge_output.end(), 0.0f);
        }
        impl_->initialized = true;
        impl_->seeded = false;
        impl_->next_position = 0;
        return true;
    } catch (const std::exception& e) {
        err = std::string("EXL3 Flash: reset: ") + e.what();
        return false;
    }
}

bool Exl3FlashModel::forward(int32_t token, int64_t pos, int32_t& next_token, float* host_logits,
                             std::string& err) {
    err.clear();
    next_token = -1;
    if (!loaded()) { err = "EXL3 Flash: model is not open"; return false; }
    if (!impl_->initialized && !reset(err)) return false;
    if (pos != impl_->next_position) {
        err = "EXL3 Flash: position " + std::to_string(pos) + " does not follow reset/previous forward at " +
              std::to_string(impl_->next_position);
        return false;
    }
    if (pos < 0 || pos >= impl_->options.max_cells) {
        err = "EXL3 Flash: position is outside the configured max_cells";
        return false;
    }
    if (token < 0 || static_cast<int64_t>(token) >= impl_->map.geometry.vocab_size) {
        err = "EXL3 Flash: token is outside the embedding vocabulary";
        return false;
    }
    try {
        auto& first = impl_->stages.front();
        {
            const OnDevice on(first.device);
            impl_->embedding->gather_one(token, first.embedding_device, nullptr);
            for (int64_t stream = 0; stream < impl_->geometry.hc; ++stream) {
                if (!strata::gpu::copy_async(first.session.block.R +
                                                 static_cast<size_t>(stream) * impl_->geometry.n_embd,
                                             first.embedding_device,
                                             static_cast<size_t>(impl_->geometry.n_embd) * sizeof(float), nullptr)) {
                    err = std::string("EXL3 Flash: embedding residual broadcast failed: ") + strata::gpu::last_error();
                    return false;
                }
            }
            if (!strata::gpu::stream_sync(nullptr)) {
                err = std::string("EXL3 Flash: embedding upload failed: ") + strata::gpu::last_error();
                return false;
            }
            impl_->seeded = true;
        }

        for (auto& stage : impl_->stages) {
            stage_token(impl_->geometry, pos, 0, stage.session);
        }
        if (impl_->ple_stage_index < 0 || static_cast<size_t>(impl_->ple_stage_index) >= impl_->stages.size()) {
            err = "EXL3 Flash: PLE stage ownership is missing";
            return false;
        }
        auto& ple_stage = impl_->stages[static_cast<size_t>(impl_->ple_stage_index)];
        ple_stage.session.ple_token = token;
        if (ple_stage.session.ple.table != nullptr) {
            const OnDevice on(ple_stage.device);
            if (!ple_stage.session.ple.ready() ||
                !ple_stage.session.ple.table->is_open() ||
                !ple_stage.session.ple.hist ||
                !ple_stage.session.ple.emb_dev ||
                !ple_stage.session.ple.scratch ||
                !ple_stage.session.ple.prev) {
                err = "EXL3 Flash: PLE state is not ready before token staging";
                return false;
            }
            if (!ple_stage_token(ple_stage.session.ple, nullptr, err)) {
                err = "EXL3 Flash: PLE token staging: " + err;
                return false;
            }
        }

        for (size_t stage_index = 0; stage_index < impl_->stages.size(); ++stage_index) {
            auto& stage = impl_->stages[stage_index];
            const OnDevice on(stage.device);
            if (stage_index > 0) {
                if (!strata::gpu::copy(stage.session.block.R, impl_->bridge_host.data(),
                                       impl_->bridge_host.size() * sizeof(float))) {
                    err = std::string("EXL3 Flash: stage boundary residual upload failed: ") + strata::gpu::last_error();
                    return false;
                }
            }
            for (int64_t layer = stage.layer_begin; layer < stage.layer_end; ++layer) {
                const size_t local = static_cast<size_t>(layer - stage.layer_begin);
                doorbell_reset(stage.doorbell);
                if (!strata::gpu::graph_launch(stage.pre_graphs[local], nullptr) ||
                    !strata::gpu::stream_sync(nullptr)) {
                    err = "EXL3 Flash: pre graph launch layer " + std::to_string(layer) + ": " +
                          strata::gpu::last_error();
                    return false;
                }
                if (!strata::gpu::memset(stage.parts_device, 0,
                                         stage.merge_output.size() * sizeof(float))) {
                    err = std::string("EXL3 Flash: clearing expert output: ") + strata::gpu::last_error();
                    return false;
                }
                sycl::event expert_done;
                if (!prepare_expert_window(stage_index, layer, stage.doorbell.h_x_f, impl_->geometry.n_embd,
                                           stage.doorbell.h_ids, 1, stage.miss_output.data(), impl_->geometry.n_embd,
                                           nullptr, static_cast<const float*>(stage.session.block.mixed),
                                           static_cast<float*>(stage.parts_device), expert_done, err) ||
                    !wait_expert_window(stage_index, expert_done, err)) {
                    err = "EXL3 Flash: expert window layer " + std::to_string(layer) + ": " + err;
                    return false;
                }
                if (!strata::gpu::copy(stage.merge_output.data(), stage.parts_device,
                                       stage.merge_output.size() * sizeof(float))) {
                    err = std::string("EXL3 Flash: expert device-to-host merge copy failed: ") + strata::gpu::last_error();
                    return false;
                }
                for (size_t i = 0; i < stage.merge_output.size(); ++i)
                    stage.merge_output[i] += stage.miss_output[i];
                if (!strata::gpu::copy(stage.parts_device, stage.merge_output.data(),
                                       stage.merge_output.size() * sizeof(float))) {
                    err = std::string("EXL3 Flash: expert host-to-device merge copy failed: ") + strata::gpu::last_error();
                    return false;
                }
                if (!strata::gpu::graph_launch(stage.post_graphs[local], nullptr) ||
                    !strata::gpu::stream_sync(nullptr)) {
                    err = "EXL3 Flash: post graph launch layer " + std::to_string(layer) + ": " +
                          strata::gpu::last_error();
                    return false;
                }
            }
            if (stage_index + 1 < impl_->stages.size()) {
                if (!strata::gpu::copy(impl_->bridge_host.data(), stage.session.block.R,
                                       impl_->bridge_host.size() * sizeof(float))) {
                    err = std::string("EXL3 Flash: stage boundary residual download failed: ") + strata::gpu::last_error();
                    return false;
                }
            }
        }

        auto& last = impl_->stages.back();
        {
            const OnDevice on(last.device);
            if (!lm_head(last.table, impl_->geometry, last.session.block, static_cast<float*>(impl_->logits_device),
                         nullptr, err) || !strata::gpu::stream_sync(nullptr) ||
                !strata::gpu::copy(impl_->logits_host.data(), impl_->logits_device,
                                   impl_->logits_host.size() * sizeof(float))) {
                err = "EXL3 Flash: output head: " + (err.empty() ? strata::gpu::last_error() : err);
                return false;
            }
        }
        double best = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < impl_->logits_host.size(); ++i) {
            const double value = impl_->logits_host[i];
            if (std::isfinite(value) && value > best) {
                best = value;
                next_token = static_cast<int32_t>(i);
            }
        }
        if (next_token < 0) { err = "EXL3 Flash: output logits are all non-finite"; return false; }
        if (host_logits != nullptr)
            std::copy(impl_->logits_host.begin(), impl_->logits_host.end(), host_logits);
        if (ple_stage.session.ple.table != nullptr) {
            ple_stage.session.ple_prev[0] = ple_stage.session.ple_prev[1];
            ple_stage.session.ple_prev[1] = token;
        }
        ++impl_->next_position;
        return true;
    } catch (const std::exception& e) {
        err = std::string("EXL3 Flash: forward: ") + e.what();
        return false;
    }
}

bool Exl3FlashModel::validate_token_request(int64_t n_tokens, int64_t batch, std::string& err) const {
    if (!loaded()) { err = "EXL3 Flash: model is not open"; return false; }
    if (n_tokens != 1 || batch != 1) {
        err = "EXL3 Flash: initial coordinator supports only one autoregressive token (batch=1)";
        return false;
    }
    return true;
}

bool Exl3FlashModel::loaded() const { return impl_ != nullptr && impl_->opened; }
size_t Exl3FlashModel::stage_count() const { return impl_ == nullptr ? 0 : impl_->stages.size(); }

Exl3FlashStageView Exl3FlashModel::stage(size_t index) const {
    if (impl_ == nullptr || index >= impl_->stages.size()) return {};
    const auto& stage = impl_->stages[index];
    return {stage.device, stage.layer_begin, stage.layer_end, const_cast<WeightTable*>(&stage.table),
            stage.weights.get(), stage.source.get(), stage.runtime.get()};
}

const strata::ModelSpec& Exl3FlashModel::spec() const {
    static const strata::ModelSpec empty;
    return impl_ == nullptr ? empty : impl_->spec;
}

const ModelGeometry& Exl3FlashModel::geometry() const {
    static const ModelGeometry empty;
    return impl_ == nullptr ? empty : impl_->geometry;
}

const ModelMap& Exl3FlashModel::model_map() const {
    static const ModelMap empty;
    return impl_ == nullptr ? empty : impl_->map;
}

NativeEmbed* Exl3FlashModel::embedding() const { return impl_ == nullptr ? nullptr : impl_->embedding.get(); }
Exl3Ple* Exl3FlashModel::ple() const { return impl_ == nullptr ? nullptr : impl_->ple.get(); }
strata::kernels::PleTable* Exl3FlashModel::ple_table() const {
    return impl_ == nullptr ? nullptr : impl_->ple_table.get();
}

const std::shared_ptr<const strata::artifact::exl3::Archive>& Exl3FlashModel::archive() const {
    static const std::shared_ptr<const strata::artifact::exl3::Archive> empty;
    return impl_ == nullptr ? empty : impl_->archive;
}

}  // namespace strata::core
