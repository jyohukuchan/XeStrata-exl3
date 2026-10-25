// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Typed model geometry read directly from an EXL3 archive config.
#include "strata/artifact/exl3_model_spec.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace strata::artifact::exl3 {
namespace {

using Json = nlohmann::json;

const Json* object_field(const Json& object, const char* key) {
    if (!object.is_object()) return nullptr;
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &*it;
}

bool required_int(const Json& object, const char* key, int64_t& result, std::string& err) {
    const Json* value = object_field(object, key);
    if (value == nullptr || (!value->is_number_integer() && !value->is_number_unsigned())) {
        err = std::string("EXL3 model spec: missing or non-integer config field ") + key;
        return false;
    }
    if (value->is_number_unsigned()) {
        const uint64_t unsigned_value = value->get<uint64_t>();
        if (unsigned_value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            err = std::string("EXL3 model spec: integer config field is outside int64 range ") + key;
            return false;
        }
        result = static_cast<int64_t>(unsigned_value);
    } else {
        result = value->get<int64_t>();
    }
    return true;
}

bool optional_int(const Json& object, const char* key, int64_t& result, std::string& err) {
    const Json* value = object_field(object, key);
    if (value == nullptr) return true;
    if (!value->is_number_integer() && !value->is_number_unsigned()) {
        err = std::string("EXL3 model spec: non-integer config field ") + key;
        return false;
    }
    if (value->is_number_unsigned()) {
        const uint64_t unsigned_value = value->get<uint64_t>();
        if (unsigned_value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            err = std::string("EXL3 model spec: integer config field is outside int64 range ") + key;
            return false;
        }
        result = static_cast<int64_t>(unsigned_value);
    } else {
        result = value->get<int64_t>();
    }
    return true;
}

bool required_num(const Json& object, const char* key, double& result, std::string& err) {
    const Json* value = object_field(object, key);
    if (value == nullptr || !value->is_number()) {
        err = std::string("EXL3 model spec: missing or non-numeric config field ") + key;
        return false;
    }
    result = value->get<double>();
    if (!std::isfinite(result)) {
        err = std::string("EXL3 model spec: non-finite config field ") + key;
        return false;
    }
    return true;
}

bool optional_num(const Json& object, const char* key, double& result, std::string& err) {
    const Json* value = object_field(object, key);
    if (value == nullptr) return true;
    if (!value->is_number()) {
        err = std::string("EXL3 model spec: non-numeric config field ") + key;
        return false;
    }
    result = value->get<double>();
    if (!std::isfinite(result)) {
        err = std::string("EXL3 model spec: non-finite config field ") + key;
        return false;
    }
    return true;
}

bool optional_bool(const Json& object, const char* key, bool& result, std::string& err) {
    const Json* value = object_field(object, key);
    if (value == nullptr) return true;
    if (!value->is_boolean()) {
        err = std::string("EXL3 model spec: non-boolean config field ") + key;
        return false;
    }
    result = value->get<bool>();
    return true;
}

bool optional_per_layer_num(const Json& object, const char* key, int64_t n, std::vector<double>& result,
                            std::string& err) {
    const Json* value = object_field(object, key);
    if (value == nullptr) return true;
    if (value->is_number()) {
        const double one = value->get<double>();
        if (!std::isfinite(one)) { err = std::string("EXL3 model spec: non-finite config field ") + key; return false; }
        result.assign(static_cast<size_t>(n), one);
        return true;
    }
    if (!value->is_array() || static_cast<int64_t>(value->size()) != n) {
        err = std::string("EXL3 model spec: config field ") + key + " is not one value per layer";
        return false;
    }
    result.clear();
    result.reserve(value->size());
    for (const auto& item : *value) {
        if (!item.is_number() || !std::isfinite(item.get<double>())) {
            err = std::string("EXL3 model spec: config field ") + key + " contains a non-finite number";
            return false;
        }
        result.push_back(item.get<double>());
    }
    return true;
}

bool required_array(const Json& object, const char* key, const Json*& result, std::string& err) {
    result = object_field(object, key);
    if (result == nullptr || !result->is_array()) {
        err = std::string("EXL3 model spec: missing or non-array config field ") + key;
        return false;
    }
    return true;
}

bool positive(int64_t value, const char* key, std::string& err) {
    if (value > 0) return true;
    err = std::string("EXL3 model spec: config field ") + key + " must be positive";
    return false;
}

bool product(int64_t a, int64_t b, int64_t& result, const char* label, std::string& err) {
    if (a <= 0 || b <= 0 || a > std::numeric_limits<int64_t>::max() / b) {
        err = std::string("EXL3 model spec: geometry product overflows for ") + label;
        return false;
    }
    result = a * b;
    return true;
}

bool qwen_mtp_complete(const Archive& archive) {
    bool layer_tensor = false;
    for (const auto& tensor : archive.tensors())
        if (tensor.name.rfind("mtp.layers.0.", 0) == 0) { layer_tensor = true; break; }
    return archive.find_linear("mtp.fc") != nullptr &&
           archive.find_tensor("mtp.pre_fc_norm_embedding.weight") != nullptr &&
           archive.find_tensor("mtp.pre_fc_norm_hidden.weight") != nullptr && layer_tensor;
}

}  // namespace

bool read_exl3_model_spec(const Archive& archive, strata::ModelSpec& spec, std::string& err,
                          bool include_mtp) {
    err.clear();
    spec = strata::ModelSpec{};
    try {
        const Json root = Json::parse(archive.config_json());
        const Json* text = object_field(root, "text_config");
        if (text == nullptr) text = &root;
        if (!text->is_object()) { err = "EXL3 model spec: text_config must be an object"; return false; }

        const Json* model_type_value = object_field(*text, "model_type");
        if (model_type_value == nullptr) model_type_value = object_field(root, "model_type");
        if (model_type_value == nullptr || !model_type_value->is_string()) {
            err = "EXL3 model spec: missing model_type";
            return false;
        }
        const std::string model_type = model_type_value->get<std::string>();
        const bool qwen35 = model_type == "qwen3_5_moe_text" || model_type == "qwen3_5_moe";
        const bool qwen38 = model_type == "qwen4_exp_text" || model_type == "qwen4_exp";
        const bool lfm2 = model_type == "lfm2_moe";
        if (!qwen35 && !qwen38 && !lfm2) {
            err = "EXL3 model spec: unsupported model_type " + model_type;
            return false;
        }
        if (lfm2) {
            // LFM2 MoE (HF Lfm2MoeForCausalLM): short-conv and full-attention layers, leading dense FFN
            // layers, routed experts with an expert bias and no shared expert.  The geometry lands in the
            // same ModelSpec the GGUF reader fills; which engine block could run it is the model map's call.
            if (include_mtp) {
                err = "EXL3 model spec: LFM2 MoE has no MTP layer this engine reads";
                return false;
            }
            spec.arch = strata::ModelArch::Lfm2Moe;
            spec.arch_name = "lfm2moe";
            spec.n_nextn = 0;
            int64_t vocab_lfm = 0, kv_heads_lfm = 0;
            if (!required_int(*text, "num_hidden_layers", spec.n_layer, err) ||
                !required_int(*text, "hidden_size", spec.n_embd, err) ||
                !required_int(*text, "vocab_size", vocab_lfm, err) ||
                !required_int(*text, "num_attention_heads", spec.n_head, err) ||
                !required_int(*text, "num_key_value_heads", kv_heads_lfm, err) ||
                !required_int(*text, "max_position_embeddings", spec.context_length, err))
                return false;
            const Json* eps_field = object_field(*text, "norm_eps");
            if (eps_field == nullptr) eps_field = object_field(*text, "rms_norm_eps");
            if (eps_field == nullptr || !eps_field->is_number() || eps_field->get<double>() <= 0.0) {
                err = "EXL3 model spec: missing or non-positive config field norm_eps";
                return false;
            }
            spec.rms_eps = eps_field->get<double>();
            spec.key_length = spec.n_head > 0 ? spec.n_embd / spec.n_head : 0;
            if (!optional_int(*text, "head_dim", spec.key_length, err)) return false;
            spec.value_length = spec.key_length;
            if (!positive(spec.n_layer, "num_hidden_layers", err) || !positive(spec.n_embd, "hidden_size", err) ||
                !positive(vocab_lfm, "vocab_size", err) || !positive(spec.n_head, "num_attention_heads", err) ||
                !positive(kv_heads_lfm, "num_key_value_heads", err) || !positive(spec.key_length, "head_dim", err) ||
                !positive(spec.context_length, "max_position_embeddings", err))
                return false;
            const Json* layer_types_lfm = nullptr;
            if (!required_array(*text, "layer_types", layer_types_lfm, err) ||
                static_cast<int64_t>(layer_types_lfm->size()) != spec.n_layer) {
                err = "EXL3 model spec: layer_types must contain exactly num_hidden_layers entries";
                return false;
            }
            spec.n_head_kv.assign(static_cast<size_t>(spec.n_layer), 0);
            spec.recurrent.assign(static_cast<size_t>(spec.n_layer), false);
            for (int64_t layer = 0; layer < spec.n_layer; ++layer) {
                const auto& item = (*layer_types_lfm)[static_cast<size_t>(layer)];
                if (!item.is_string() || (item.get<std::string>() != "conv" && item.get<std::string>() != "full_attention")) {
                    err = "EXL3 model spec: layer_types contains an unsupported layer type";
                    return false;
                }
                const bool conv = item.get<std::string>() == "conv";
                spec.recurrent[static_cast<size_t>(layer)] = conv;
                spec.n_head_kv[static_cast<size_t>(layer)] = conv ? 0 : kv_heads_lfm;
            }
            if (!required_int(*text, "conv_L_cache", spec.shortconv_l_cache, err) ||
                spec.shortconv_l_cache <= 1) {
                if (err.empty()) err = "EXL3 model spec: conv_L_cache must be above 1";
                return false;
            }
            if (!optional_int(*text, "num_dense_layers", spec.n_dense_lead, err) ||
                !optional_int(*text, "intermediate_size", spec.n_ff, err))
                return false;
            if (spec.n_dense_lead < 0 || spec.n_dense_lead >= spec.n_layer ||
                (spec.n_dense_lead > 0 && !positive(spec.n_ff, "intermediate_size", err)))
                return false;
            if (!required_int(*text, "num_experts", spec.n_expert, err) ||
                !required_int(*text, "num_experts_per_tok", spec.n_expert_used, err) ||
                !required_int(*text, "moe_intermediate_size", spec.n_ff_exp, err))
                return false;
            if (!positive(spec.n_expert, "num_experts", err) ||
                !positive(spec.n_expert_used, "num_experts_per_tok", err) ||
                spec.n_expert_used > spec.n_expert || !positive(spec.n_ff_exp, "moe_intermediate_size", err))
                return false;
            spec.n_ff_shexp = 0;
            spec.n_expert_shared = 0;
            spec.gating = strata::ExpertGating::Sigmoid;
            if (!optional_bool(*text, "use_expert_bias", spec.expert_bias, err) ||
                !optional_bool(*text, "norm_topk_prob", spec.expert_weights_norm, err) ||
                !optional_num(*text, "routed_scaling_factor", spec.expert_weights_scale, err))
                return false;
            if (object_field(*text, "norm_topk_prob") == nullptr) spec.expert_weights_norm = true;
            if (object_field(*text, "routed_scaling_factor") == nullptr) spec.expert_weights_scale = 1.0;
            const Json* rope_lfm = object_field(*text, "rope_parameters");
            if (rope_lfm == nullptr || !rope_lfm->is_object() ||
                !required_num(*rope_lfm, "rope_theta", spec.rope_base, err) || spec.rope_base <= 0.0) {
                if (err.empty()) err = "EXL3 model spec: rope_parameters.rope_theta is missing";
                return false;
            }
            spec.rope_dims = spec.key_length;   // LFM2 rotates the whole head
            return true;
        }
        spec.arch = qwen35 ? strata::ModelArch::Qwen35Moe : strata::ModelArch::Qwen4Exp;
        spec.arch_name = qwen35 ? "qwen35moe" : "qwen4exp";

        int64_t configured_mtp = 0;
        if (!optional_int(*text, "mtp_num_hidden_layers", configured_mtp, err)) return false;
        if (configured_mtp < 0) { err = "EXL3 model spec: mtp_num_hidden_layers is negative"; return false; }
        if (include_mtp && configured_mtp <= 0) {
            err = "EXL3 model spec: MTP was requested but mtp_num_hidden_layers is not positive";
            return false;
        }
        if (include_mtp && qwen38) {
            err = "EXL3 model spec: Qwen3.8 MTP requires the external official raw MTP source and override binding; "
                  "main archive MTP-looking tensors are not treated as complete";
            return false;
        }
        if (include_mtp && qwen35 && !qwen_mtp_complete(archive)) {
            err = "EXL3 model spec: Qwen3.5 MTP was requested but its mapped mtp.fc, norms, or block tensors are missing";
            return false;
        }
        spec.n_nextn = include_mtp ? configured_mtp : 0;

        int64_t vocab_size = 0, num_kv_heads = 0;
        if (!required_int(*text, "num_hidden_layers", spec.n_layer, err) ||
            !required_int(*text, "hidden_size", spec.n_embd, err) ||
            !required_int(*text, "vocab_size", vocab_size, err) ||
            !required_int(*text, "num_attention_heads", spec.n_head, err) ||
            !required_int(*text, "num_key_value_heads", num_kv_heads, err) ||
            !required_int(*text, "head_dim", spec.key_length, err) ||
            !required_int(*text, "max_position_embeddings", spec.context_length, err) ||
            !required_num(*text, "rms_norm_eps", spec.rms_eps, err)) return false;
        const int64_t vocab = vocab_size;
        spec.value_length = spec.key_length;
        if (!positive(spec.n_layer, "num_hidden_layers", err) || !positive(spec.n_embd, "hidden_size", err) ||
            !positive(vocab, "vocab_size", err) || !positive(spec.n_head, "num_attention_heads", err) ||
            !positive(num_kv_heads, "num_key_value_heads", err) || !positive(spec.key_length, "head_dim", err) ||
            !positive(spec.context_length, "max_position_embeddings", err) || spec.rms_eps <= 0.0) {
            if (err.empty()) err = "EXL3 model spec: rms_norm_eps must be positive";
            return false;
        }

        int64_t interval = 0;
        if (!required_int(*text, "full_attention_interval", interval, err) || !positive(interval, "full_attention_interval", err)) return false;
        spec.full_attention_interval = interval;
        const Json* layer_types = nullptr;
        if (!required_array(*text, "layer_types", layer_types, err) ||
            static_cast<int64_t>(layer_types->size()) != spec.n_layer) {
            err = "EXL3 model spec: layer_types must contain exactly num_hidden_layers entries";
            return false;
        }
        std::vector<std::string> layers;
        layers.reserve(layer_types->size());
        for (const auto& item : *layer_types) {
            if (!item.is_string() || (item.get<std::string>() != "linear_attention" && item.get<std::string>() != "full_attention")) {
                err = "EXL3 model spec: layer_types contains an unsupported layer type";
                return false;
            }
            layers.push_back(item.get<std::string>());
        }
        if (spec.n_layer > std::numeric_limits<int64_t>::max() - spec.n_nextn) {
            err = "EXL3 model spec: num_hidden_layers plus MTP layers overflows int64";
            return false;
        }
        const int64_t n_all = spec.n_layer + spec.n_nextn;
        spec.n_head_kv.assign(static_cast<size_t>(n_all), num_kv_heads);
        spec.recurrent.assign(static_cast<size_t>(n_all), false);
        for (int64_t layer = 0; layer < spec.n_layer; ++layer)
            spec.recurrent[static_cast<size_t>(layer)] = layers[static_cast<size_t>(layer)] == "linear_attention";

        const Json* rope = object_field(*text, "rope_parameters");
        if (rope == nullptr) rope = object_field(*text, "rope_scaling");
        if (rope == nullptr || !rope->is_object()) {
            err = "EXL3 model spec: rope_parameters is missing";
            return false;
        }
        double rope_theta = 0.0, partial = 0.0;
        if (!required_num(*rope, "rope_theta", rope_theta, err) || !required_num(*rope, "partial_rotary_factor", partial, err) ||
            rope_theta <= 0.0 || partial <= 0.0 || partial > 1.0) {
            if (err.empty()) err = "EXL3 model spec: invalid rope_parameters";
            return false;
        }
        const double rope_dim = static_cast<double>(spec.key_length) * partial;
        if (std::floor(rope_dim) != rope_dim || rope_dim <= 0.0) {
            err = "EXL3 model spec: partial_rotary_factor does not produce an integral rope dimension";
            return false;
        }
        spec.rope_base = rope_theta;
        spec.rope_dims = static_cast<int64_t>(rope_dim);
        if (const Json* type = object_field(*rope, "rope_type"); type != nullptr) {
            if (!type->is_string()) { err = "EXL3 model spec: rope_type is not a string"; return false; }
            spec.rope_scaling_type = type->get<std::string>();
            // HF's unscaled/default spelling maps to the engine's explicit
            // no-scaling spelling. generate.cpp accepts none|linear|yarn.
            if (spec.rope_scaling_type == "default") spec.rope_scaling_type = "none";
        }
        if (!optional_num(*rope, "factor", spec.rope_scaling_factor, err) ||
            !optional_num(*rope, "original_max_position_embeddings", spec.rope_scaling_orig_ctx, err)) return false;
        if (const Json* sections = object_field(*rope, "mrope_section"); sections != nullptr) {
            if (!sections->is_array()) { err = "EXL3 model spec: mrope_section is not an array"; return false; }
            for (const auto& item : *sections) {
                if ((!item.is_number_integer() && !item.is_number_unsigned()) || item.get<int64_t>() <= 0) {
                    err = "EXL3 model spec: mrope_section contains a non-positive integer";
                    return false;
                }
                spec.rope_sections.push_back(item.get<int64_t>());
            }
        }

        int64_t k_heads = 0, v_heads = 0, k_dim = 0, v_dim = 0, conv = 0;
        if (!required_int(*text, "linear_num_key_heads", k_heads, err) ||
            !required_int(*text, "linear_num_value_heads", v_heads, err) ||
            !required_int(*text, "linear_key_head_dim", k_dim, err) ||
            !required_int(*text, "linear_value_head_dim", v_dim, err) ||
            !required_int(*text, "linear_conv_kernel_dim", conv, err)) return false;
        int64_t inner_a = 0, inner_b = 0;
        if (!product(k_heads, k_dim, inner_a, "linear key channels", err) ||
            !product(v_heads, v_dim, inner_b, "linear value channels", err)) {
            if (err.empty()) err = "EXL3 model spec: linear attention geometry overflows";
            return false;
        }
        // llama.cpp's ModelSpec `ssm.inner_size` is the value stream width
        // for both Qwen variants. The complete GDN convolution width is a
        // separate engine-derived quantity: 2*key_channels + value_channels.
        spec.ssm_d_inner = inner_b;
        spec.ssm_d_conv = conv;
        spec.ssm_d_state = k_dim;
        spec.ssm_dt_rank = v_heads;
        spec.ssm_n_group = k_heads;
        if (!positive(spec.ssm_d_conv, "linear_conv_kernel_dim", err) || !positive(spec.ssm_d_state, "linear_key_head_dim", err)) return false;

        if (!required_int(*text, "num_experts", spec.n_expert, err) ||
            !required_int(*text, "num_experts_per_tok", spec.n_expert_used, err) ||
            !required_int(*text, "moe_intermediate_size", spec.n_ff_exp, err) ||
            !required_int(*text, "shared_expert_intermediate_size", spec.n_ff_shexp, err)) return false;
        if (!positive(spec.n_expert, "num_experts", err) || !positive(spec.n_expert_used, "num_experts_per_tok", err) ||
            spec.n_expert_used > spec.n_expert || !positive(spec.n_ff_exp, "moe_intermediate_size", err) ||
            !positive(spec.n_ff_shexp, "shared_expert_intermediate_size", err)) return false;
        spec.n_expert_shared = 1;
        spec.n_dense_lead = 0;
        spec.n_ff = 0;
        spec.gating = strata::ExpertGating::Softmax;
        // Qwen's HF implementation uses normalized softmax top-k routing and
        // has no GGUF expert-weight correction or SwiGLU clamp fields. Keep
        // those ModelSpec fields at their explicit architecture defaults, but
        // honor them if a future Qwen config writes the fields directly.
        if (!optional_bool(*text, "expert_weights_norm", spec.expert_weights_norm, err) ||
            !optional_num(*text, "expert_weights_scale", spec.expert_weights_scale, err) ||
            !optional_per_layer_num(*text, "swiglu_clamp_exp", n_all, spec.swiglu_clamp_exp, err) ||
            !optional_per_layer_num(*text, "swiglu_clamp_shexp", n_all, spec.swiglu_clamp_shexp, err)) return false;

        if (qwen38) {
            int64_t indexer_kv_heads = 0;
            if (!required_int(*text, "hc_count", spec.hc, err) ||
                !required_int(*text, "hc_lowrank", spec.hc_low_rank, err) ||
                !required_int(*text, "indexer_n_heads", spec.idx_n_head, err) ||
                !required_int(*text, "indexer_kv_heads", indexer_kv_heads, err) ||
                !required_int(*text, "indexer_head_dim", spec.idx_key_length, err) ||
                !required_int(*text, "indexer_budget", spec.idx_top_k, err)) return false;
            spec.idx_kpool = 0;
            spec.idx_kpool_select_tail = false;
            if (!positive(spec.hc, "hc_count", err) || !positive(spec.hc_low_rank, "hc_lowrank", err) ||
                !positive(spec.idx_n_head, "indexer_n_heads", err) || !positive(indexer_kv_heads, "indexer_kv_heads", err) ||
                !positive(spec.idx_key_length, "indexer_head_dim", err) || !positive(spec.idx_top_k, "indexer_budget", err)) return false;
        }
        (void) vocab;  // retained in the typed fields only by the caller's tokenizer metadata
        return true;
    } catch (const std::exception& e) {
        err = std::string("EXL3 model spec: malformed config: ") + e.what();
        return false;
    }
}

}  // namespace strata::artifact::exl3
