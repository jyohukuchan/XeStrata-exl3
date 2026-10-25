// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Qwen EXL3 model-map discovery and header-only validation.

#include "strata/artifact/exl3_model_map.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <set>
#include <stdexcept>

namespace strata::artifact::exl3 {
namespace {

using Json = nlohmann::json;

std::string string_value(const Json& object, const char* key, const std::string& fallback = {}) {
    const auto it = object.find(key);
    return it != object.end() && it->is_string() ? it->get<std::string>() : fallback;
}

int64_t int_value(const Json& object, const char* key, int64_t fallback = 0) {
    const auto it = object.find(key);
    if (it == object.end() || !it->is_number_integer()) return fallback;
    return it->get<int64_t>();
}

bool bool_value(const Json& object, const char* key, bool fallback = false) {
    const auto it = object.find(key);
    if (it == object.end()) return fallback;
    if (it->is_boolean()) return it->get<bool>();
    if (it->is_number_integer()) return it->get<int64_t>() != 0;
    return fallback;
}

const Json* text_config(const Json& root) {
    const auto it = root.find("text_config");
    return it != root.end() && it->is_object() ? &*it : &root;
}

Json parse_config(const Archive& archive) {
    try {
        return Json::parse(archive.config_json());
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("model map: config.json is malformed: ") + e.what());
    }
}

void reason(ModelMap& map, const std::string& text) {
    if (std::find(map.unsupported.begin(), map.unsupported.end(), text) == map.unsupported.end())
        map.unsupported.push_back(text);
}

AxisMapping identity_axis(int64_t size, AxisTransform transform = AxisTransform::identity) {
    AxisMapping axis;
    axis.transform = transform;
    axis.source_size = size;
    axis.target_size = size;
    if (transform == AxisTransform::explicit_permutation || transform == AxisTransform::raw_rows) {
        axis.permutation.resize(size > 0 ? static_cast<size_t>(size) : 0);
        for (int64_t i = 0; i < size; ++i) axis.permutation[static_cast<size_t>(i)] = i;
    }
    return axis;
}

AxisMapping v_head_axis(int64_t total, int64_t offset, int64_t key_heads, int64_t value_heads,
                        int64_t head_dim) {
    AxisMapping axis;
    axis.transform = AxisTransform::h128_block_permutation;
    axis.source_size = total;
    axis.target_size = total;
    axis.block_size = head_dim;
    axis.block_offset = offset;
    axis.permutation.resize(total > 0 ? static_cast<size_t>(total) : 0);
    for (int64_t i = 0; i < total; ++i) axis.permutation[static_cast<size_t>(i)] = i;
    if (key_heads <= 0 || value_heads <= 0 || head_dim <= 0 || value_heads % key_heads != 0 ||
        offset < 0 || offset > total || value_heads * head_dim > total - offset)
        return axis;
    const int64_t values_per_key = value_heads / key_heads;
    axis.block_permutation.resize(static_cast<size_t>(value_heads));
    for (int64_t target_head = 0; target_head < value_heads; ++target_head) {
        const int64_t key_head = target_head % key_heads;
        const int64_t value_in_key = target_head / key_heads;
        const int64_t source_head = key_head * values_per_key + value_in_key;
        axis.block_permutation[static_cast<size_t>(target_head)] = source_head;
        for (int64_t d = 0; d < head_dim; ++d) {
            const int64_t target = offset + target_head * head_dim + d;
            const int64_t source = offset + source_head * head_dim + d;
            axis.permutation[static_cast<size_t>(target)] = source;
        }
    }
    return axis;
}

const LinearDescriptor* linear(const Archive& archive, const std::string& name) {
    return archive.find_linear(name);
}

const TensorDescriptor* raw(const Archive& archive, const std::string& name) {
    return archive.find_tensor(name + ".weight");
}

bool add_raw(ModelMap& map, const Archive& archive, ProjectionKind kind, int64_t layer, int64_t expert,
             const std::string& source, const std::string& canonical, int64_t target_k, int64_t target_n,
             const std::set<std::string>& allowed_dtypes = {"F16", "BF16", "F32"},
             AxisMapping output_axis = {}, AxisMapping input_axis = {});

bool add_linear(ModelMap& map, const Archive& archive, ProjectionKind kind, int64_t layer, int64_t expert,
                const std::string& source, const std::string& canonical, int64_t target_k, int64_t target_n,
                MappingTransform transform = MappingTransform::identity, std::vector<OutputSlice> slices = {},
                AxisMapping input_axis = {}, AxisMapping output_axis = {}) {
    const LinearDescriptor* descriptor = linear(archive, source);
    const TensorDescriptor* raw_descriptor = raw(archive, source);
    if (descriptor != nullptr && raw_descriptor != nullptr) {
        reason(map, source + ": both packed EXL3 and raw .weight tensors are present; mapping is ambiguous");
        return false;
    }
    if (descriptor == nullptr) {
        const bool raw_exception = kind == ProjectionKind::conv_in || kind == ProjectionKind::conv_out ||
                                   kind == ProjectionKind::gdn_qkv || kind == ProjectionKind::gdn_z ||
                                   kind == ProjectionKind::gdn_out || kind == ProjectionKind::qsa_q ||
                                   kind == ProjectionKind::qsa_k ||
                                   kind == ProjectionKind::qsa_v || kind == ProjectionKind::qsa_out ||
                                   kind == ProjectionKind::expert_gate || kind == ProjectionKind::expert_up ||
                                   kind == ProjectionKind::expert_down || kind == ProjectionKind::shared_gate ||
                                   kind == ProjectionKind::shared_up || kind == ProjectionKind::shared_down ||
                                   kind == ProjectionKind::dense_gate || kind == ProjectionKind::dense_up ||
                                   kind == ProjectionKind::dense_down ||
                                   kind == ProjectionKind::lm_head || kind == ProjectionKind::mtp_fc;
        if (raw_descriptor != nullptr && raw_exception)
            return add_raw(map, archive, kind, layer, expert, source, canonical, target_k, target_n,
                           {"F16", "BF16", "F32"}, output_axis, input_axis);
        reason(map, "missing EXL3 linear: " + source);
        return false;
    }
    ProjectionMapping mapping;
    mapping.kind = kind;
    mapping.transform = transform;
    mapping.layer = layer;
    mapping.expert = expert;
    mapping.source_name = source;
    mapping.canonical_name = canonical;
    mapping.source_k = descriptor->k;
    mapping.source_n = descriptor->n;
    mapping.target_k = target_k;
    mapping.target_n = target_n;
    mapping.input_axis = input_axis;
    if (mapping.input_axis.source_size == 0 && mapping.input_axis.target_size == 0)
        mapping.input_axis = identity_axis(descriptor->k);
    if (mapping.input_axis.source_size != descriptor->k || mapping.input_axis.target_size != target_k)
        reason(map, source + ": input-axis map does not match matrix dimensions");
    mapping.output_axis = output_axis;
    if (mapping.output_axis.source_size == 0 && mapping.output_axis.target_size == 0) {
        mapping.output_axis.source_size = descriptor->n;
        mapping.output_axis.target_size = target_n;
    }
    if (mapping.output_axis.source_size != descriptor->n || mapping.output_axis.target_size != target_n)
        reason(map, source + ": output-axis map does not match matrix dimensions");
    mapping.output_slices = std::move(slices);
    if (transform == MappingTransform::output_slice) {
        mapping.output_axis.transform = AxisTransform::slices;
        mapping.output_axis.slices = mapping.output_slices;
        int64_t covered_target = 0;
        for (const auto& slice : mapping.output_slices) {
            if (slice.source_offset < 0 || slice.target_offset < 0 || slice.elements <= 0 ||
                slice.source_offset > descriptor->n - slice.elements || slice.target_offset > target_n - slice.elements) {
                reason(map, source + ": output slice is outside source or target bounds");
                continue;
            }
            covered_target += slice.elements;
        }
        for (size_t i = 0; i < mapping.output_slices.size(); ++i) {
            const auto& a = mapping.output_slices[i];
            for (size_t j = i + 1; j < mapping.output_slices.size(); ++j) {
                const auto& b = mapping.output_slices[j];
                const bool source_overlap = a.source_offset < b.source_offset + b.elements &&
                                            b.source_offset < a.source_offset + a.elements;
                const bool target_overlap = a.target_offset < b.target_offset + b.elements &&
                                            b.target_offset < a.target_offset + a.elements;
                if (source_overlap || target_overlap)
                    reason(map, source + ": output slices overlap on source or target axis");
            }
        }
        if (covered_target != target_n)
            reason(map, source + ": output slices do not cover the canonical target output");
        if (descriptor->k != target_k) {
            reason(map, source + ": EXL3 input shape [" + std::to_string(descriptor->k) + "] does not match target [" +
                   std::to_string(target_k) + "]");
        }
    } else if (descriptor->k != target_k || descriptor->n != target_n) {
        reason(map, source + ": EXL3 shape [" + std::to_string(descriptor->k) + "," +
               std::to_string(descriptor->n) + "] does not match target [" + std::to_string(target_k) + "," +
               std::to_string(target_n) + "]");
    }
    if (transform != MappingTransform::output_slice && output_axis.source_size == 0 && output_axis.target_size == 0)
        mapping.output_axis.transform = AxisTransform::identity;
    map.projections.push_back(std::move(mapping));
    return true;
}

bool add_raw(ModelMap& map, const Archive& archive, ProjectionKind kind, int64_t layer, int64_t expert,
             const std::string& source, const std::string& canonical, int64_t target_k, int64_t target_n,
             const std::set<std::string>& allowed_dtypes, AxisMapping output_axis, AxisMapping input_axis) {
    const TensorDescriptor* descriptor = raw(archive, source);
    if (descriptor == nullptr) {
        reason(map, "missing raw tensor: " + source + ".weight");
        return false;
    }
    ProjectionMapping mapping;
    mapping.kind = kind;
    mapping.transform = MappingTransform::raw_rows;
    mapping.layer = layer;
    mapping.expert = expert;
    mapping.source_name = source + ".weight";
    mapping.canonical_name = canonical;
    mapping.target_k = target_k;
    mapping.target_n = target_n;
    mapping.raw_dtype = descriptor->dtype;
    mapping.raw_shape.reserve(descriptor->shape.size());
    for (const auto dim : descriptor->shape) mapping.raw_shape.push_back(static_cast<int64_t>(dim));
    if (descriptor->shape.size() == 2) {
        mapping.raw_rows = static_cast<int64_t>(descriptor->shape[0]);
        mapping.raw_cols = static_cast<int64_t>(descriptor->shape[1]);
        mapping.source_k = mapping.raw_cols;
        mapping.source_n = mapping.raw_rows;
        mapping.input_axis = input_axis.source_size != 0 || input_axis.target_size != 0
                                 ? std::move(input_axis)
                                 : identity_axis(mapping.raw_cols);
        if (mapping.input_axis.source_size != mapping.raw_cols || mapping.input_axis.target_size != target_k)
            reason(map, source + ": raw input-axis map does not match column count");
        mapping.output_axis = output_axis.source_size != 0 || output_axis.target_size != 0
                                  ? std::move(output_axis)
                                  : identity_axis(mapping.raw_rows, AxisTransform::raw_rows);
        if (mapping.output_axis.source_size != mapping.raw_rows || mapping.output_axis.target_size != mapping.raw_rows)
            reason(map, source + ": raw output-axis map does not match row count");
        mapping.raw_axes = {mapping.output_axis, mapping.input_axis};
        mapping.raw_to_canonical_axes = {1, 0};
    } else if (descriptor->shape.size() == 1) {
        mapping.raw_rows = static_cast<int64_t>(descriptor->shape[0]);
        mapping.source_k = mapping.raw_rows;
        mapping.source_n = 1;
        mapping.output_axis = identity_axis(mapping.raw_rows, AxisTransform::raw_rows);
        mapping.raw_axes = {mapping.output_axis};
        mapping.raw_to_canonical_axes = {0};
    } else {
        reason(map, source + ".weight: raw tensor must be one- or two-dimensional");
    }
    if (!allowed_dtypes.count(descriptor->dtype)) reason(map, source + ": unsupported raw dtype " + descriptor->dtype);
    if (descriptor->shape.size() == 2 &&
        (mapping.raw_cols != target_k || mapping.raw_rows != target_n))
        reason(map, source + ": raw shape does not match target [" + std::to_string(target_n) + "," +
               std::to_string(target_k) + "]");
    if (descriptor->shape.size() == 1 && (target_n != 1 || mapping.raw_rows != target_k))
        reason(map, source + ": raw vector shape does not match target [" + std::to_string(target_k) + "]");
    map.projections.push_back(std::move(mapping));
    return true;
}

bool add_aux(ModelMap& map, const Archive& archive, int64_t layer, const std::string& source,
             const std::string& canonical, int64_t expected_elements = -1,
             ValueTransform value_transform = ValueTransform::identity, AxisMapping output_axis = {},
             ProjectionKind kind = ProjectionKind::auxiliary, std::vector<int64_t> raw_to_canonical_axes = {}) {
    std::string tensor_name = source;
    const TensorDescriptor* descriptor = archive.find_tensor(tensor_name);
    const bool has_weight_suffix = tensor_name.size() >= 7 &&
                                    tensor_name.compare(tensor_name.size() - 7, 7, ".weight") == 0;
    if (descriptor == nullptr && !has_weight_suffix) {
        tensor_name += ".weight";
        descriptor = archive.find_tensor(tensor_name);
    }
    if (descriptor == nullptr) {
        reason(map, "missing auxiliary tensor: " + tensor_name);
        return false;
    }
    ProjectionMapping mapping;
    mapping.kind = kind;
    mapping.transform = MappingTransform::raw_rows;
    mapping.value_transform = value_transform;
    mapping.layer = layer;
    mapping.source_name = tensor_name;
    mapping.canonical_name = canonical;
    mapping.raw_dtype = descriptor->dtype;
    mapping.raw_shape.reserve(descriptor->shape.size());
    for (const auto dim : descriptor->shape) mapping.raw_shape.push_back(static_cast<int64_t>(dim));
    mapping.raw_to_canonical_axes = std::move(raw_to_canonical_axes);
    if (mapping.raw_to_canonical_axes.empty()) {
        mapping.raw_to_canonical_axes.resize(descriptor->shape.size());
        for (size_t i = 0; i < mapping.raw_to_canonical_axes.size(); ++i)
            mapping.raw_to_canonical_axes[i] = static_cast<int64_t>(i);
    }
    uint64_t elements = 1;
    for (const auto dim : descriptor->shape) elements *= dim;
    mapping.source_k = static_cast<int64_t>(elements);
    mapping.source_n = 1;
    mapping.raw_axes.reserve(descriptor->shape.size());
    for (const auto dim : descriptor->shape)
        mapping.raw_axes.push_back(identity_axis(static_cast<int64_t>(dim), AxisTransform::raw_rows));
    if (output_axis.source_size != 0 || output_axis.target_size != 0) {
        mapping.output_axis = std::move(output_axis);
        if (!mapping.raw_axes.empty()) mapping.raw_axes.front() = mapping.output_axis;
        if (mapping.raw_shape.empty() || mapping.output_axis.source_size != mapping.raw_shape.front())
            reason(map, tensor_name + ": auxiliary output-axis map does not match the physical first axis");
    } else if (!mapping.raw_axes.empty()) {
        mapping.output_axis = mapping.raw_axes.front();
    }
    if (expected_elements >= 0 && static_cast<int64_t>(elements) != expected_elements)
        reason(map, tensor_name + ": auxiliary element count does not match expected geometry");
    map.projections.push_back(std::move(mapping));
    return true;
}

}  // namespace

AxisMapping make_v_head_axis(int64_t total, int64_t offset, int64_t key_heads, int64_t value_heads,
                             int64_t head_dim) {
    return v_head_axis(total, offset, key_heads, value_heads, head_dim);
}

const char* projection_kind_name(ProjectionKind kind) {
    switch (kind) {
        case ProjectionKind::conv_in: return "conv_in";
        case ProjectionKind::conv_out: return "conv_out";
        case ProjectionKind::gdn_qkv: return "gdn_qkv";
        case ProjectionKind::gdn_z: return "gdn_z";
        case ProjectionKind::gdn_out: return "gdn_out";
        case ProjectionKind::gdn_alpha: return "gdn_alpha";
        case ProjectionKind::gdn_beta: return "gdn_beta";
        case ProjectionKind::qsa_q: return "qsa_q";
        case ProjectionKind::qsa_k: return "qsa_k";
        case ProjectionKind::qsa_v: return "qsa_v";
        case ProjectionKind::qsa_out: return "qsa_out";
        case ProjectionKind::qsa_indexer_q: return "qsa_indexer_q";
        case ProjectionKind::qsa_indexer_k: return "qsa_indexer_k";
        case ProjectionKind::router: return "router";
        case ProjectionKind::shared_scalar: return "shared_scalar";
        case ProjectionKind::shared_gate: return "shared_gate";
        case ProjectionKind::shared_up: return "shared_up";
        case ProjectionKind::shared_down: return "shared_down";
        case ProjectionKind::dense_gate: return "dense_gate";
        case ProjectionKind::dense_up: return "dense_up";
        case ProjectionKind::dense_down: return "dense_down";
        case ProjectionKind::expert_gate: return "expert_gate";
        case ProjectionKind::expert_up: return "expert_up";
        case ProjectionKind::expert_down: return "expert_down";
        case ProjectionKind::hc_norm: return "hc_norm";
        case ProjectionKind::hc_down: return "hc_down";
        case ProjectionKind::hc_up: return "hc_up";
        case ProjectionKind::hc_inject: return "hc_inject";
        case ProjectionKind::mtp_fc: return "mtp_fc";
        case ProjectionKind::embedding: return "embedding";
        case ProjectionKind::lm_head: return "lm_head";
        case ProjectionKind::final_norm: return "final_norm";
        case ProjectionKind::auxiliary: return "auxiliary";
    }
    return "unknown";
}

const char* model_family_name(ModelFamily family) {
    switch (family) {
        case ModelFamily::qwen35: return "qwen35";
        case ModelFamily::qwen38: return "qwen38";
        case ModelFamily::hyv3: return "hyv3";
        case ModelFamily::lfm2moe: return "lfm2moe";
    }
    return "qwen38";
}

const char* value_transform_name(ValueTransform transform) {
    switch (transform) {
        case ValueTransform::identity: return "identity";
        case ValueTransform::add_one: return "add_one";
        case ValueTransform::negative_exp: return "negative_exp";
    }
    return "unknown";
}

bool ModelMap::require_supported(std::string& err) const {
    if (unsupported.empty()) { err.clear(); return true; }
    err = "EXL3 model map for " + std::string(model_family_name(family)) + " is unsupported:";
    for (const auto& item : unsupported) err += "\n- " + item;
    return false;
}

bool build_model_map(const Archive& archive, ModelMap& result, std::string& err, bool include_mtp) {
    result = ModelMap{};
    err.clear();
    try {
        const Json config = parse_config(archive);
        const Json* text = text_config(config);
        const std::string model_type = string_value(config, "model_type", string_value(*text, "model_type"));
        if (model_type == "qwen4_exp" || model_type == "qwen4_exp_text") result.family = ModelFamily::qwen38;
        else if (model_type == "qwen3_5_moe" || model_type == "qwen3_5_moe_text" ||
                 model_type == "qwen3_5" || model_type == "qwen3_5_text") result.family = ModelFamily::qwen35;
        else if (model_type == "hy_v3") result.family = ModelFamily::hyv3;
        else if (model_type == "lfm2_moe") result.family = ModelFamily::lfm2moe;
        else {
            err = "EXL3 model map: unsupported model_type " + model_type;
            return false;
        }
        if (result.family == ModelFamily::hyv3) {
            // HYV3ForCausalLM: HF names under model.layers.*; norms are plain RMSNorm weights (no +1),
            // Q is not gated, the shared expert is always on, and the router bias is a plain buffer.
            auto& g = result.geometry;
            g.hidden_size = int_value(*text, "hidden_size");
            g.vocab_size = int_value(*text, "vocab_size", int_value(config, "vocab_size"));
            g.layers = int_value(*text, "num_hidden_layers");
            g.mtp_layers = int_value(*text, "num_nextn_predict_layers", int_value(config, "num_nextn_predict_layers"));
            g.experts = int_value(*text, "num_experts");
            g.experts_per_token = int_value(*text, "num_experts_per_tok");
            g.moe_intermediate = int_value(*text, "expert_hidden_dim", int_value(*text, "moe_intermediate_size"));
            const int64_t shared_count = int_value(*text, "num_shared_experts", 1);
            g.shared_intermediate = g.moe_intermediate * (shared_count > 0 ? shared_count : 1);
            g.q_heads = int_value(*text, "num_attention_heads");
            g.kv_heads = int_value(*text, "num_key_value_heads", g.q_heads);
            g.head_dim = int_value(*text, "head_dim", g.hidden_size / (g.q_heads ? g.q_heads : 1));
            g.dense_lead = int_value(*text, "first_k_dense_replace");
            g.dense_intermediate = int_value(*text, "intermediate_size");
            if (g.hidden_size <= 0 || g.layers <= 0 || g.experts <= 0 || g.moe_intermediate <= 0 ||
                g.dense_intermediate <= 0 || g.dense_lead < 0 || g.dense_lead >= g.layers)
                return (err = "EXL3 model map: config has incomplete HyV3 geometry", false);
            if (include_mtp && g.mtp_layers > 0)
                reason(result, "HyV3 NextN/MTP block is not run by this engine yet");
            result.source_prefix = "model.layers";
            if (archive.find_tensor("model.embed_tokens.weight") == nullptr &&
                archive.find_linear("model.embed_tokens") == nullptr) {
                err = "EXL3 model map: cannot find the HyV3 embedding under model.embed_tokens";
                return false;
            }
            if (g.vocab_size <= 0) {
                reason(result, "config has no positive vocab_size for embedding/head mapping");
            } else {
                add_aux(result, archive, -1, "model.embed_tokens", "token_embd.weight",
                        g.vocab_size * g.hidden_size, ValueTransform::identity, {}, ProjectionKind::embedding, {1, 0});
                add_linear(result, archive, ProjectionKind::lm_head, -1, -1, "lm_head", "output.weight",
                           g.hidden_size, g.vocab_size);
            }
            add_aux(result, archive, -1, "model.norm", "output_norm.weight", g.hidden_size,
                    ValueTransform::identity, {}, ProjectionKind::final_norm);
            for (int64_t layer = 0; layer < g.layers; ++layer) {
                const std::string p = result.source_prefix + "." + std::to_string(layer) + ".";
                const std::string cp = "blk." + std::to_string(layer) + ".";
                add_aux(result, archive, layer, p + "input_layernorm", cp + "attn_norm.weight", g.hidden_size,
                        ValueTransform::identity);
                add_aux(result, archive, layer, p + "post_attention_layernorm", cp + "ffn_norm.weight", g.hidden_size,
                        ValueTransform::identity);
                add_linear(result, archive, ProjectionKind::qsa_q, layer, -1, p + "self_attn.q_proj", cp + "attn_q.weight",
                           g.hidden_size, g.q_heads * g.head_dim);
                add_linear(result, archive, ProjectionKind::qsa_k, layer, -1, p + "self_attn.k_proj", cp + "attn_k.weight",
                           g.hidden_size, g.kv_heads * g.head_dim);
                add_linear(result, archive, ProjectionKind::qsa_v, layer, -1, p + "self_attn.v_proj", cp + "attn_v.weight",
                           g.hidden_size, g.kv_heads * g.head_dim);
                add_linear(result, archive, ProjectionKind::qsa_out, layer, -1, p + "self_attn.o_proj", cp + "attn_output.weight",
                           g.q_heads * g.head_dim, g.hidden_size);
                add_aux(result, archive, layer, p + "self_attn.q_norm", cp + "attn_q_norm.weight", g.head_dim,
                        ValueTransform::identity);
                add_aux(result, archive, layer, p + "self_attn.k_norm", cp + "attn_k_norm.weight", g.head_dim,
                        ValueTransform::identity);
                if (layer < g.dense_lead) {
                    add_linear(result, archive, ProjectionKind::dense_gate, layer, -1, p + "mlp.gate_proj", cp + "ffn_gate.weight",
                               g.hidden_size, g.dense_intermediate);
                    add_linear(result, archive, ProjectionKind::dense_up, layer, -1, p + "mlp.up_proj", cp + "ffn_up.weight",
                               g.hidden_size, g.dense_intermediate);
                    add_linear(result, archive, ProjectionKind::dense_down, layer, -1, p + "mlp.down_proj", cp + "ffn_down.weight",
                               g.dense_intermediate, g.hidden_size);
                    continue;
                }
                add_raw(result, archive, ProjectionKind::router, layer, -1, p + "mlp.router.gate", cp + "ffn_gate_inp.weight",
                        g.hidden_size, g.experts);
                if (archive.find_tensor(p + "mlp.expert_bias") != nullptr)
                    add_aux(result, archive, layer, p + "mlp.expert_bias", cp + "exp_probs_b",
                            g.experts, ValueTransform::identity);
                add_linear(result, archive, ProjectionKind::shared_gate, layer, -1, p + "mlp.shared_mlp.gate_proj",
                           cp + "ffn_gate_shexp.weight", g.hidden_size, g.shared_intermediate);
                add_linear(result, archive, ProjectionKind::shared_up, layer, -1, p + "mlp.shared_mlp.up_proj",
                           cp + "ffn_up_shexp.weight", g.hidden_size, g.shared_intermediate);
                add_linear(result, archive, ProjectionKind::shared_down, layer, -1, p + "mlp.shared_mlp.down_proj",
                           cp + "ffn_down_shexp.weight", g.shared_intermediate, g.hidden_size);
                for (int64_t expert = 0; expert < g.experts; ++expert) {
                    const std::string ep = p + "mlp.experts." + std::to_string(expert) + ".";
                    const std::string ec = cp + "experts." + std::to_string(expert) + ".";
                    add_linear(result, archive, ProjectionKind::expert_gate, layer, expert, ep + "gate_proj", ec + "gate_proj",
                               g.hidden_size, g.moe_intermediate);
                    add_linear(result, archive, ProjectionKind::expert_up, layer, expert, ep + "up_proj", ec + "up_proj",
                               g.hidden_size, g.moe_intermediate);
                    add_linear(result, archive, ProjectionKind::expert_down, layer, expert, ep + "down_proj", ec + "down_proj",
                               g.moe_intermediate, g.hidden_size);
                }
            }
            return true;
        }
        if (result.family == ModelFamily::lfm2moe) {
            // LFM2 MoE: every projection of the family is mapped here (short-conv, full attention, the
            // leading dense FFN, routed experts with their bias and router).  Which engine block runs the
            // map is the engine's call; the map itself is complete for a well-formed archive.
            auto& g = result.geometry;
            g.hidden_size = int_value(*text, "hidden_size");
            g.vocab_size = int_value(*text, "vocab_size", int_value(config, "vocab_size"));
            g.layers = int_value(*text, "num_hidden_layers");
            g.experts = int_value(*text, "num_experts");
            g.experts_per_token = int_value(*text, "num_experts_per_tok");
            g.moe_intermediate = int_value(*text, "moe_intermediate_size");
            g.q_heads = int_value(*text, "num_attention_heads");
            g.kv_heads = int_value(*text, "num_key_value_heads", g.q_heads);
            g.head_dim = int_value(*text, "head_dim", g.q_heads > 0 ? g.hidden_size / g.q_heads : 0);
            g.dense_intermediate = int_value(*text, "intermediate_size");
            g.dense_layers = int_value(*text, "num_dense_layers");
            g.shortconv_l_cache = int_value(*text, "conv_L_cache", 3);
            if (g.hidden_size <= 0 || g.layers <= 0 || g.experts <= 0 || g.moe_intermediate <= 0 ||
                g.shortconv_l_cache <= 1 || (g.dense_layers > 0 && g.dense_intermediate <= 0) ||
                (g.dense_layers < 0 || g.dense_layers >= g.layers))
                return (err = "EXL3 model map: config has incomplete LFM2 MoE geometry", false);
            if (include_mtp) reason(result, "lfm2moe: this family has no MTP layer");
            result.source_prefix = "model.layers";
            if (g.vocab_size <= 0)
                reason(result, "config has no positive vocab_size for embedding/head mapping");
            else {
                add_aux(result, archive, -1, "model.embed_tokens", "token_embd.weight",
                        g.vocab_size * g.hidden_size, ValueTransform::identity, {}, ProjectionKind::embedding, {1, 0});
                add_linear(result, archive, ProjectionKind::lm_head, -1, -1, "lm_head", "output.weight",
                           g.hidden_size, g.vocab_size);
            }
            add_aux(result, archive, -1, "model.embedding_norm", "output_norm.weight", g.hidden_size,
                    ValueTransform::identity, {}, ProjectionKind::final_norm);
            const Json* layer_types = nullptr;
            const auto lt = text->find("layer_types");
            if (lt != text->end() && lt->is_array()) layer_types = &*lt;
            for (int64_t layer = 0; layer < g.layers; ++layer) {
                const bool conv = layer_types != nullptr && layer < static_cast<int64_t>(layer_types->size())
                                      ? (*layer_types)[static_cast<size_t>(layer)].get<std::string>() == "conv"
                                      : (layer + 1) % 4 != 0;
                const std::string p = result.source_prefix + "." + std::to_string(layer) + ".";
                const std::string cp = "blk." + std::to_string(layer) + ".";
                add_aux(result, archive, layer, p + "operator_norm", cp + "attn_norm.weight", g.hidden_size);
                add_aux(result, archive, layer, p + "ffn_norm", cp + "ffn_norm.weight", g.hidden_size);
                if (conv) {
                    add_linear(result, archive, ProjectionKind::conv_in, layer, -1, p + "conv.in_proj",
                               cp + "shortconv_in_proj.weight", g.hidden_size, 3 * g.hidden_size);
                    add_aux(result, archive, layer, p + "conv.conv", cp + "shortconv_conv.weight",
                            g.hidden_size * g.shortconv_l_cache);
                    add_linear(result, archive, ProjectionKind::conv_out, layer, -1, p + "conv.out_proj",
                               cp + "shortconv_out_proj.weight", g.hidden_size, g.hidden_size);
                } else {
                    add_linear(result, archive, ProjectionKind::qsa_q, layer, -1, p + "self_attn.q_proj",
                               cp + "attn_q.weight", g.hidden_size, g.q_heads * g.head_dim);
                    add_linear(result, archive, ProjectionKind::qsa_k, layer, -1, p + "self_attn.k_proj",
                               cp + "attn_k.weight", g.hidden_size, g.kv_heads * g.head_dim);
                    add_linear(result, archive, ProjectionKind::qsa_v, layer, -1, p + "self_attn.v_proj",
                               cp + "attn_v.weight", g.hidden_size, g.kv_heads * g.head_dim);
                    add_linear(result, archive, ProjectionKind::qsa_out, layer, -1, p + "self_attn.out_proj",
                               cp + "attn_output.weight", g.q_heads * g.head_dim, g.hidden_size);
                    add_aux(result, archive, layer, p + "self_attn.q_layernorm", cp + "attn_q_norm.weight", g.head_dim);
                    add_aux(result, archive, layer, p + "self_attn.k_layernorm", cp + "attn_k_norm.weight", g.head_dim);
                }
                if (layer < g.dense_layers) {
                    add_linear(result, archive, ProjectionKind::dense_gate, layer, -1, p + "feed_forward.w1",
                               cp + "ffn_gate.weight", g.hidden_size, g.dense_intermediate);
                    add_linear(result, archive, ProjectionKind::dense_up, layer, -1, p + "feed_forward.w3",
                               cp + "ffn_up.weight", g.hidden_size, g.dense_intermediate);
                    add_linear(result, archive, ProjectionKind::dense_down, layer, -1, p + "feed_forward.w2",
                               cp + "ffn_down.weight", g.dense_intermediate, g.hidden_size);
                } else {
                    add_raw(result, archive, ProjectionKind::router, layer, -1, p + "feed_forward.gate",
                            cp + "ffn_gate_inp.weight", g.hidden_size, g.experts);
                    add_aux(result, archive, layer, p + "feed_forward.expert_bias", cp + "ffn_exp_probs_b.bias",
                            g.experts);
                    for (int64_t expert = 0; expert < g.experts; ++expert) {
                        const std::string ep = p + "feed_forward.experts." + std::to_string(expert) + ".";
                        const std::string ec = cp + "experts." + std::to_string(expert) + ".";
                        add_linear(result, archive, ProjectionKind::expert_gate, layer, expert, ep + "w1",
                                   ec + "gate_proj", g.hidden_size, g.moe_intermediate);
                        add_linear(result, archive, ProjectionKind::expert_up, layer, expert, ep + "w3",
                                   ec + "up_proj", g.hidden_size, g.moe_intermediate);
                        add_linear(result, archive, ProjectionKind::expert_down, layer, expert, ep + "w2",
                                   ec + "down_proj", g.moe_intermediate, g.hidden_size);
                    }
                }
            }
            return true;
        }
        auto& g = result.geometry;
        g.hidden_size = int_value(*text, "hidden_size");
        g.vocab_size = int_value(*text, "vocab_size", int_value(config, "vocab_size"));
        g.layers = int_value(*text, "num_hidden_layers");
        g.mtp_layers = int_value(*text, "mtp_num_hidden_layers", int_value(config, "mtp_num_hidden_layers"));
        g.dedicated_embedding = bool_value(*text, "mtp_use_dedicated_embeddings",
                                           bool_value(config, "mtp_use_dedicated_embeddings"));
        g.experts = int_value(*text, "num_experts");
        g.experts_per_token = int_value(*text, "num_experts_per_tok");
        g.moe_intermediate = int_value(*text, "moe_intermediate_size");
        g.shared_intermediate = int_value(*text, "shared_expert_intermediate_size", g.moe_intermediate);
        g.q_heads = int_value(*text, "num_attention_heads");
        g.kv_heads = int_value(*text, "num_key_value_heads", g.q_heads);
        g.head_dim = int_value(*text, "head_dim", g.hidden_size / (g.q_heads ? g.q_heads : 1));
        g.linear_k_heads = int_value(*text, "linear_num_key_heads");
        g.linear_v_heads = int_value(*text, "linear_num_value_heads");
        g.linear_k_dim = int_value(*text, "linear_key_head_dim");
        g.linear_v_dim = int_value(*text, "linear_value_head_dim");
        g.indexer_heads = int_value(*text, "indexer_n_heads");
        g.indexer_kv_heads = int_value(*text, "indexer_kv_heads");
        g.indexer_head_dim = int_value(*text, "indexer_head_dim");
        g.full_attention_interval = int_value(*text, "full_attention_interval", 4);
        g.hc_count = int_value(*text, "hc_count");
        g.hc_rank = int_value(*text, "hc_rank", int_value(*text, "hc_lr", int_value(*text, "hc_lowrank")));
        if (g.hidden_size <= 0 || g.layers <= 0 || g.experts <= 0 || g.moe_intermediate <= 0)
            return (err = "EXL3 model map: config has incomplete Qwen geometry", false);
        result.source_prefix = archive.find_linear("model.language_model.layers.0.linear_attn.in_proj_qkv") != nullptr ||
                                       archive.find_tensor("model.language_model.layers.0.linear_attn.in_proj_qkv.weight") != nullptr ||
                                       archive.find_tensor("model.language_model.layers.0.linear_attn.in_proj_a.weight") != nullptr
                                   ? "model.language_model.layers"
                                   : "model.layers";
        const std::string first = result.source_prefix + ".0.linear_attn.in_proj_qkv";
        if (archive.find_linear(first) == nullptr && archive.find_tensor(first + ".weight") == nullptr) {
            err = "EXL3 model map: cannot find the first Qwen linear projection under " + result.source_prefix;
            return false;
        }
        const std::string model_root = result.source_prefix.substr(0, result.source_prefix.rfind(".layers"));
        if (g.vocab_size <= 0)
            reason(result, "config has no positive vocab_size for embedding/head mapping");
        else {
            add_aux(result, archive, -1, model_root + ".embed_tokens", "token_embd.weight",
                    g.vocab_size * g.hidden_size, ValueTransform::identity, {}, ProjectionKind::embedding, {1, 0});
            add_linear(result, archive, ProjectionKind::lm_head, -1, -1, "lm_head", "output.weight",
                       g.hidden_size, g.vocab_size);
        }
        const int64_t qkv_n = 2 * g.linear_k_heads * g.linear_k_dim + g.linear_v_heads * g.linear_v_dim;
        const int64_t z_n = g.linear_v_heads * g.linear_v_dim;
        const int64_t out_k = g.linear_v_heads * g.linear_v_dim;
        if (g.linear_k_heads <= 0 || g.linear_v_heads <= 0 || g.linear_k_dim <= 0 || g.linear_v_dim <= 0) {
            reason(result, "linear-attention head geometry is incomplete");
        } else if (g.linear_v_heads % g.linear_k_heads != 0) {
            reason(result, "linear_num_value_heads must be divisible by linear_num_key_heads for V-head reorder");
        }
        if (g.linear_v_dim != 128 || g.linear_k_dim != 128)
            reason(result, "the Qwen V-reorder map currently requires 128-wide linear head blocks");
        const AxisMapping qkv_v_axis = v_head_axis(qkv_n, 2 * g.linear_k_heads * g.linear_k_dim,
                                                     g.linear_k_heads, g.linear_v_heads, g.linear_v_dim);
        const AxisMapping z_v_axis = v_head_axis(z_n, 0, g.linear_k_heads, g.linear_v_heads, g.linear_v_dim);
        const AxisMapping out_v_axis = v_head_axis(out_k, 0, g.linear_k_heads, g.linear_v_heads, g.linear_v_dim);
        const AxisMapping scalar_v_axis = v_head_axis(g.linear_v_heads, 0, g.linear_k_heads, g.linear_v_heads, 1);
        const Json* layer_types = nullptr;
        const auto lt = text->find("layer_types");
        if (lt != text->end() && lt->is_array()) layer_types = &*lt;
        for (int64_t layer = 0; layer < g.layers; ++layer) {
            const bool qsa = layer_types != nullptr && layer < static_cast<int64_t>(layer_types->size())
                                 ? (*layer_types)[static_cast<size_t>(layer)].get<std::string>() == "full_attention"
                                 : (g.full_attention_interval > 0 && (layer + 1) % g.full_attention_interval == 0);
            const std::string p = result.source_prefix + "." + std::to_string(layer) + ".";
            const std::string cp = "blk." + std::to_string(layer) + ".";
            if (!qsa) {
                add_linear(result, archive, ProjectionKind::gdn_qkv, layer, -1, p + "linear_attn.in_proj_qkv", cp + "attn_qkv.weight",
                           g.hidden_size, qkv_n, MappingTransform::identity, {}, {}, qkv_v_axis);
                add_linear(result, archive, ProjectionKind::gdn_z, layer, -1, p + "linear_attn.in_proj_z", cp + "attn_gate.weight",
                           g.hidden_size, z_n, MappingTransform::identity, {}, {}, z_v_axis);
                add_linear(result, archive, ProjectionKind::gdn_out, layer, -1, p + "linear_attn.out_proj", cp + "ssm_out.weight",
                           out_k, g.hidden_size, MappingTransform::identity, {}, out_v_axis);
                add_raw(result, archive, ProjectionKind::gdn_alpha, layer, -1, p + "linear_attn.in_proj_a", cp + "ssm_alpha.weight",
                        g.hidden_size, g.linear_v_heads, {"F16", "BF16", "F32"}, scalar_v_axis);
                add_raw(result, archive, ProjectionKind::gdn_beta, layer, -1, p + "linear_attn.in_proj_b", cp + "ssm_beta.weight",
                        g.hidden_size, g.linear_v_heads, {"F16", "BF16", "F32"}, scalar_v_axis);
                add_aux(result, archive, layer, p + "linear_attn.norm", cp + "ssm_norm.weight");
                add_aux(result, archive, layer, p + "linear_attn.A_log", cp + "ssm_a", g.linear_v_heads,
                        ValueTransform::negative_exp, scalar_v_axis);
                add_aux(result, archive, layer, p + "linear_attn.dt_bias", cp + "ssm_dt.bias", g.linear_v_heads,
                        ValueTransform::identity, scalar_v_axis);
                add_aux(result, archive, layer, p + "linear_attn.conv1d", cp + "ssm_conv1d.weight", -1,
                        ValueTransform::identity, qkv_v_axis);
            }
            if (result.family == ModelFamily::qwen35) {
                add_aux(result, archive, layer, p + "input_layernorm", cp + "attn_norm.weight", g.hidden_size,
                        ValueTransform::add_one);
                add_aux(result, archive, layer, p + "post_attention_layernorm", cp + "post_attention_norm.weight", g.hidden_size,
                        ValueTransform::add_one);
            }
            add_raw(result, archive, ProjectionKind::router, layer, -1, p + "mlp.gate", cp + "ffn_gate_inp.weight",
                    g.hidden_size, g.experts);
            add_raw(result, archive, ProjectionKind::shared_scalar, layer, -1, p + "mlp.shared_expert_gate", cp + "ffn_gate_inp_shexp.weight",
                    g.hidden_size, 1);
            add_linear(result, archive, ProjectionKind::shared_gate, layer, -1, p + "mlp.shared_expert.gate_proj", cp + "ffn_gate_shexp.weight",
                       g.hidden_size, g.shared_intermediate);
            add_linear(result, archive, ProjectionKind::shared_up, layer, -1, p + "mlp.shared_expert.up_proj", cp + "ffn_up_shexp.weight",
                       g.hidden_size, g.shared_intermediate);
            add_linear(result, archive, ProjectionKind::shared_down, layer, -1, p + "mlp.shared_expert.down_proj", cp + "ffn_down_shexp.weight",
                       g.shared_intermediate, g.hidden_size);
            if (result.family == ModelFamily::qwen38) {
                auto add_hc = [&](const char* source_prefix, const char* canonical_prefix) {
                    add_aux(result, archive, layer, p + source_prefix + std::string(".hc_norm"),
                            cp + canonical_prefix + std::string("_norm.weight"), g.hidden_size * g.hc_count, ValueTransform::add_one);
                    add_raw(result, archive, ProjectionKind::hc_down, layer, -1,
                            p + source_prefix + std::string(".input_mix_weight_down"),
                            cp + canonical_prefix + std::string("_down.weight"), g.hidden_size * g.hc_count, g.hc_rank);
                    add_raw(result, archive, ProjectionKind::hc_up, layer, -1,
                            p + source_prefix + std::string(".input_mix_weight_up"),
                            cp + canonical_prefix + std::string("_up.weight"), g.hc_rank, g.hidden_size * g.hc_count);
                    add_raw(result, archive, ProjectionKind::hc_inject, layer, -1,
                            p + source_prefix + std::string(".block_inject_weight"),
                            cp + canonical_prefix + std::string("_inject.weight"), g.hidden_size * g.hc_count, g.hc_count);
                };
                add_hc("attn_hyper_connection", "hc_attn");
                add_hc("mlp_hyper_connection", "hc_ffn");
            }
            if (qsa) {
                add_linear(result, archive, ProjectionKind::qsa_q, layer, -1, p + "self_attn.q_proj", cp + "attn_q.weight",
                           g.hidden_size, 2 * g.q_heads * g.head_dim);
                add_linear(result, archive, ProjectionKind::qsa_k, layer, -1, p + "self_attn.k_proj", cp + "attn_k.weight",
                           g.hidden_size, g.kv_heads * g.head_dim);
                add_linear(result, archive, ProjectionKind::qsa_v, layer, -1, p + "self_attn.v_proj", cp + "attn_v.weight",
                           g.hidden_size, g.kv_heads * g.head_dim);
                add_linear(result, archive, ProjectionKind::qsa_out, layer, -1, p + "self_attn.o_proj", cp + "attn_output.weight",
                           g.q_heads * g.head_dim, g.hidden_size);
                add_aux(result, archive, layer, p + "self_attn.q_norm", cp + "attn_q_norm.weight", g.head_dim,
                        ValueTransform::add_one);
                add_aux(result, archive, layer, p + "self_attn.k_norm", cp + "attn_k_norm.weight", g.head_dim,
                        ValueTransform::add_one);
                if (result.family == ModelFamily::qwen38) {
                    const int64_t qdim = g.indexer_heads * g.indexer_head_dim;
                    const int64_t kdim = g.indexer_kv_heads * g.indexer_head_dim;
                    const std::string source = p + "self_attn.indexer.index_qk_proj";
                    std::vector<OutputSlice> qs{{0, 0, qdim}};
                    std::vector<OutputSlice> ks{{qdim, 0, kdim}};
                    add_linear(result, archive, ProjectionKind::qsa_indexer_q, layer, -1, source, cp + "indexer.q_proj.weight",
                               g.hidden_size, qdim, MappingTransform::output_slice, qs);
                    add_linear(result, archive, ProjectionKind::qsa_indexer_k, layer, -1, source, cp + "indexer.k_proj.weight",
                               g.hidden_size, kdim, MappingTransform::output_slice, ks);
                    add_aux(result, archive, layer, p + "self_attn.indexer.q_layernorm", cp + "indexer.q_norm.weight",
                            g.indexer_head_dim, ValueTransform::add_one);
                    add_aux(result, archive, layer, p + "self_attn.indexer.k_layernorm", cp + "indexer.k_norm.weight",
                            g.indexer_head_dim, ValueTransform::add_one);
                }
            }
            for (int64_t expert = 0; expert < g.experts; ++expert) {
                const std::string ep = p + "mlp.experts." + std::to_string(expert) + ".";
                const std::string ec = cp + "experts." + std::to_string(expert) + ".";
                add_linear(result, archive, ProjectionKind::expert_gate, layer, expert, ep + "gate_proj", ec + "gate_proj",
                           g.hidden_size, g.moe_intermediate);
                add_linear(result, archive, ProjectionKind::expert_up, layer, expert, ep + "up_proj", ec + "up_proj",
                           g.hidden_size, g.moe_intermediate);
                add_linear(result, archive, ProjectionKind::expert_down, layer, expert, ep + "down_proj", ec + "down_proj",
                           g.moe_intermediate, g.hidden_size);
            }
        }
        if (include_mtp) {
            const bool mtp_fc_present = archive.find_linear("mtp.fc") != nullptr ||
                                        archive.find_tensor("mtp.fc.weight") != nullptr;
            if (result.family != ModelFamily::qwen35)
                reason(result, "MTP mapping is currently supported only for Qwen3.5");
            if (g.mtp_layers != 1)
                reason(result, "MTP mapping requires text_config.mtp_num_hidden_layers == 1");
            if (g.dedicated_embedding)
                reason(result, "MTP dedicated embeddings are not supported; the MTP map requires shared embedding");
            if (g.mtp_layers == 0 && mtp_fc_present)
                reason(result, "MTP tensors are present but text_config.mtp_num_hidden_layers is zero");
            if (result.family == ModelFamily::qwen35 && g.mtp_layers == 1 && !g.dedicated_embedding) {
                const int64_t layer = g.layers;
                const std::string p = "mtp.layers.0.";
                const std::string cp = "blk." + std::to_string(layer) + ".";
                add_linear(result, archive, ProjectionKind::mtp_fc, layer, -1, "mtp.fc",
                           cp + "nextn.eh_proj.weight", 2 * g.hidden_size, g.hidden_size);
                add_aux(result, archive, -1, "mtp.pre_fc_norm_embedding", cp + "nextn.enorm.weight",
                        g.hidden_size, ValueTransform::add_one);
                add_aux(result, archive, -1, "mtp.pre_fc_norm_hidden", cp + "nextn.hnorm.weight",
                        g.hidden_size, ValueTransform::add_one);
                add_aux(result, archive, layer, p + "input_layernorm", cp + "attn_norm.weight",
                        g.hidden_size, ValueTransform::add_one);
                add_aux(result, archive, layer, p + "post_attention_layernorm", cp + "post_attention_norm.weight",
                        g.hidden_size, ValueTransform::add_one);
                add_linear(result, archive, ProjectionKind::qsa_q, layer, -1, p + "self_attn.q_proj",
                           cp + "attn_q.weight", g.hidden_size, 2 * g.q_heads * g.head_dim);
                add_linear(result, archive, ProjectionKind::qsa_k, layer, -1, p + "self_attn.k_proj",
                           cp + "attn_k.weight", g.hidden_size, g.kv_heads * g.head_dim);
                add_linear(result, archive, ProjectionKind::qsa_v, layer, -1, p + "self_attn.v_proj",
                           cp + "attn_v.weight", g.hidden_size, g.kv_heads * g.head_dim);
                add_linear(result, archive, ProjectionKind::qsa_out, layer, -1, p + "self_attn.o_proj",
                           cp + "attn_output.weight", g.q_heads * g.head_dim, g.hidden_size);
                add_aux(result, archive, layer, p + "self_attn.q_norm", cp + "attn_q_norm.weight",
                        g.head_dim, ValueTransform::add_one);
                add_aux(result, archive, layer, p + "self_attn.k_norm", cp + "attn_k_norm.weight",
                        g.head_dim, ValueTransform::add_one);
                add_raw(result, archive, ProjectionKind::router, layer, -1, p + "mlp.gate",
                        cp + "ffn_gate_inp.weight", g.hidden_size, g.experts);
                add_raw(result, archive, ProjectionKind::shared_scalar, layer, -1, p + "mlp.shared_expert_gate",
                        cp + "ffn_gate_inp_shexp.weight", g.hidden_size, 1);
                add_linear(result, archive, ProjectionKind::shared_gate, layer, -1,
                           p + "mlp.shared_expert.gate_proj", cp + "ffn_gate_shexp.weight",
                           g.hidden_size, g.shared_intermediate);
                add_linear(result, archive, ProjectionKind::shared_up, layer, -1,
                           p + "mlp.shared_expert.up_proj", cp + "ffn_up_shexp.weight",
                           g.hidden_size, g.shared_intermediate);
                add_linear(result, archive, ProjectionKind::shared_down, layer, -1,
                           p + "mlp.shared_expert.down_proj", cp + "ffn_down_shexp.weight",
                           g.shared_intermediate, g.hidden_size);
                for (int64_t expert = 0; expert < g.experts; ++expert) {
                    const std::string ep = p + "mlp.experts." + std::to_string(expert) + ".";
                    const std::string ec = cp + "experts." + std::to_string(expert) + ".";
                    add_linear(result, archive, ProjectionKind::expert_gate, layer, expert, ep + "gate_proj",
                               ec + "gate_proj", g.hidden_size, g.moe_intermediate);
                    add_linear(result, archive, ProjectionKind::expert_up, layer, expert, ep + "up_proj",
                               ec + "up_proj", g.hidden_size, g.moe_intermediate);
                    add_linear(result, archive, ProjectionKind::expert_down, layer, expert, ep + "down_proj",
                               ec + "down_proj", g.moe_intermediate, g.hidden_size);
                }
                add_aux(result, archive, -1, "mtp.norm", cp + "nextn.shared_head_norm.weight",
                        g.hidden_size, ValueTransform::add_one);
            }
        }
        if (result.family == ModelFamily::qwen35) {
            add_aux(result, archive, -1, model_root + ".norm", "output_norm.weight", g.hidden_size,
                    ValueTransform::add_one, {}, ProjectionKind::final_norm);
        } else {
            for (const auto& sidecar : archive.sidecars()) {
                if (sidecar.role == SidecarRole::specialized_ngram_codec) {
                    reason(result, "Qwen3.8 PLE ngram sidecar binding is pending in ModelMap");
                    break;
                }
            }
        }
        if (result.family == ModelFamily::qwen38 && g.hc_count <= 0)
            reason(result, "Qwen3.8 config has no positive hc_count");
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}


bool build_qwen38_mtp_map(const Archive& archive, const ModelMap& trunk, ModelMap& result, std::string& err) {
    result = ModelMap{};
    err.clear();
    if (trunk.family != ModelFamily::qwen38) {
        err = "EXL3 MTP map: trunk map is not Qwen3.8";
        return false;
    }
    if (trunk.geometry.layers <= 0 || trunk.geometry.hidden_size <= 0 || trunk.geometry.experts <= 0 ||
        trunk.geometry.experts_per_token <= 0 || trunk.geometry.hc_count <= 0 || trunk.geometry.hc_rank <= 0) {
        err = "EXL3 MTP map: trunk geometry is incomplete";
        return false;
    }
    const auto g = trunk.geometry;
    if (g.mtp_layers != 1 || trunk.geometry.dedicated_embedding) {
        err = "EXL3 MTP map: exactly one shared-embedding MTP layer is required";
        return false;
    }
    result.family = ModelFamily::qwen38;
    result.geometry = g;
    result.geometry.layers = 1;
    result.geometry.mtp_layers = 0;
    result.source_prefix = "mtp.layers";
    const int64_t N = g.hidden_size, HC = g.hc_count, LR = g.hc_rank;
    const int64_t Q = g.q_heads * g.head_dim, KV = g.kv_heads * g.head_dim;
    const int64_t IDXQ = g.indexer_heads * g.indexer_head_dim;
    const int64_t IDXK = g.indexer_kv_heads * g.indexer_head_dim;
    const std::string p = "mtp.layers.0.";
    const std::string cp = "blk.0.";
    add_linear(result, archive, ProjectionKind::mtp_fc, -1, -1, "mtp.fc_embedding", "mtp.fc_embedding.weight", N, N);
    add_linear(result, archive, ProjectionKind::mtp_fc, -1, -1, "mtp.fc_hidden", "mtp.fc_hidden.weight", N, N);
    add_aux(result, archive, -1, "mtp.pre_fc_norm_embedding", "mtp.pre_fc_norm_embedding.weight", N,
            ValueTransform::add_one);
    add_aux(result, archive, -1, "mtp.pre_fc_norm_hidden", "mtp.pre_fc_norm_hidden.weight", N * HC,
            ValueTransform::add_one);

    auto add_hc = [&](const char* source_prefix, const char* canonical_prefix) {
        add_aux(result, archive, 0, p + source_prefix + std::string(".hc_norm"),
                cp + canonical_prefix + std::string("_norm.weight"), N * HC, ValueTransform::add_one,
                {}, ProjectionKind::hc_norm);
        add_raw(result, archive, ProjectionKind::hc_down, 0, -1,
                p + source_prefix + std::string(".input_mix_weight_down"),
                cp + canonical_prefix + std::string("_down.weight"), N * HC, LR);
        add_raw(result, archive, ProjectionKind::hc_up, 0, -1,
                p + source_prefix + std::string(".input_mix_weight_up"),
                cp + canonical_prefix + std::string("_up.weight"), LR, N * HC);
        add_raw(result, archive, ProjectionKind::hc_inject, 0, -1,
                p + source_prefix + std::string(".block_inject_weight"),
                cp + canonical_prefix + std::string("_inject.weight"), N * HC, HC);
    };
    add_hc("attn_hyper_connection", "hc_attn");
    add_hc("mlp_hyper_connection", "hc_ffn");

    const std::string attn = p + "self_attn.";
    add_linear(result, archive, ProjectionKind::qsa_q, 0, -1, attn + "q_proj", cp + "attn_q.weight", N, 2 * Q);
    add_linear(result, archive, ProjectionKind::qsa_k, 0, -1, attn + "k_proj", cp + "attn_k.weight", N, KV);
    add_linear(result, archive, ProjectionKind::qsa_v, 0, -1, attn + "v_proj", cp + "attn_v.weight", N, KV);
    add_linear(result, archive, ProjectionKind::qsa_out, 0, -1, attn + "o_proj", cp + "attn_output.weight", Q, N);
    add_aux(result, archive, 0, attn + "q_norm", cp + "attn_q_norm.weight", g.head_dim, ValueTransform::add_one);
    add_aux(result, archive, 0, attn + "k_norm", cp + "attn_k_norm.weight", g.head_dim, ValueTransform::add_one);
    const std::string indexer = attn + "indexer.index_qk_proj";
    add_linear(result, archive, ProjectionKind::qsa_indexer_q, 0, -1, indexer, cp + "indexer.q_proj.weight", N,
               IDXQ, MappingTransform::output_slice, {{0, 0, IDXQ}});
    add_linear(result, archive, ProjectionKind::qsa_indexer_k, 0, -1, indexer, cp + "indexer.k_proj.weight", N,
               IDXK, MappingTransform::output_slice, {{IDXQ, 0, IDXK}});
    add_aux(result, archive, 0, attn + "indexer.q_layernorm", cp + "indexer.q_norm.weight",
            g.indexer_head_dim, ValueTransform::add_one);
    add_aux(result, archive, 0, attn + "indexer.k_layernorm", cp + "indexer.k_norm.weight",
            g.indexer_head_dim, ValueTransform::add_one);

    const std::string mlp = p + "mlp.";
    add_raw(result, archive, ProjectionKind::router, 0, -1, mlp + "gate", cp + "ffn_gate_inp.weight", N, g.experts);
    add_raw(result, archive, ProjectionKind::shared_scalar, 0, -1, mlp + "shared_expert_gate",
            cp + "ffn_gate_inp_shexp.weight", N, 1);
    add_linear(result, archive, ProjectionKind::shared_gate, 0, -1, mlp + "shared_expert.gate_proj",
               cp + "ffn_gate_shexp.weight", N, g.shared_intermediate);
    add_linear(result, archive, ProjectionKind::shared_up, 0, -1, mlp + "shared_expert.up_proj",
               cp + "ffn_up_shexp.weight", N, g.shared_intermediate);
    add_linear(result, archive, ProjectionKind::shared_down, 0, -1, mlp + "shared_expert.down_proj",
               cp + "ffn_down_shexp.weight", g.shared_intermediate, N);
    for (int64_t expert = 0; expert < g.experts; ++expert) {
        const std::string ep = mlp + "experts." + std::to_string(expert) + ".";
        const std::string ec = cp + "experts." + std::to_string(expert) + ".";
        add_linear(result, archive, ProjectionKind::expert_gate, 0, expert, ep + "gate_proj", ec + "gate_proj", N, g.moe_intermediate);
        add_linear(result, archive, ProjectionKind::expert_up, 0, expert, ep + "up_proj", ec + "up_proj", N, g.moe_intermediate);
        add_linear(result, archive, ProjectionKind::expert_down, 0, expert, ep + "down_proj", ec + "down_proj", g.moe_intermediate, N);
    }
    return true;
}

}  // namespace strata::artifact::exl3
