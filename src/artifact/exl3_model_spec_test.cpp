// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// CPU-only EXL3 model-spec config reader tests.
#include "strata/artifact/exl3_model_spec.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;

void write_archive(const std::filesystem::path& root, Json config) {
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json") << config.dump();
    Json header = Json{{"dummy", Json{{"dtype", "F32"}, {"shape", {1}}, {"data_offsets", {0, 4}}}}};
    const std::string text = header.dump();
    std::ofstream data(root / "model.safetensors", std::ios::binary);
    const uint64_t size = text.size();
    for (int i = 0; i < 8; ++i) data.put(static_cast<char>(size >> (8 * i)));
    data.write(text.data(), static_cast<std::streamsize>(text.size()));
    const uint32_t zero = 0;
    data.write(reinterpret_cast<const char*>(&zero), sizeof(zero));
    std::ofstream(root / "model.safetensors.index.json")
        << Json{{"weight_map", Json{{"dummy", "model.safetensors"}}}}.dump();
}

Json base_config(const std::string& type) {
    Json text = {
        {"model_type", type}, {"hidden_size", 2560}, {"vocab_size", 248320}, {"num_hidden_layers", 4},
        {"num_attention_heads", 20}, {"num_key_value_heads", 2}, {"head_dim", 128},
        {"max_position_embeddings", 32768}, {"rms_norm_eps", 1e-6}, {"full_attention_interval", 4},
        {"layer_types", {"linear_attention", "linear_attention", "linear_attention", "full_attention"}},
        {"linear_num_key_heads", 16}, {"linear_num_value_heads", 32}, {"linear_key_head_dim", 128},
        {"linear_value_head_dim", 128}, {"linear_conv_kernel_dim", 4}, {"num_experts", 256},
        {"num_experts_per_tok", 8}, {"moe_intermediate_size", 512}, {"shared_expert_intermediate_size", 512},
        {"mtp_num_hidden_layers", 1},
        {"rope_parameters", Json{{"rope_theta", 10000000.0}, {"partial_rotary_factor", 0.25},
                                  {"rope_type", "default"}, {"mrope_section", {11, 11, 10}}}}
    };
    if (type == "qwen4_exp_text") {
        text["hc_count"] = 4;
        text["hc_lowrank"] = 320;
        text["indexer_n_heads"] = 4;
        text["indexer_kv_heads"] = 1;
        text["indexer_head_dim"] = 128;
        text["indexer_budget"] = 2048;
    }
    return Json{{"model_type", type}, {"text_config", text}};
}

bool check(bool value, const char* label) {
    std::cout << label << ": " << (value ? "OK" : "FAIL") << '\n';
    return value;
}
}  // namespace

int main() {
    const auto root = std::filesystem::temp_directory_path() /
                      ("xestrata_exl3_model_spec_" + std::to_string(
                          std::chrono::steady_clock::now().time_since_epoch().count()));
    int failures = 0;
    try {
        write_archive(root / "q38", base_config("qwen4_exp_text"));
        strata::artifact::exl3::Archive q38_archive(root / "q38");
        strata::ModelSpec q38; std::string err;
        failures += !check(strata::artifact::exl3::read_exl3_model_spec(q38_archive, q38, err), "Qwen3.8 trunk config");
        failures += !check(q38.arch == strata::ModelArch::Qwen4Exp && q38.n_layer == 4 && q38.n_nextn == 0 &&
                               q38.n_head_kv.size() == 4 && q38.n_head_kv[0] == 2 && q38.ssm_d_inner == 4096 &&
                               q38.idx_top_k == 2048 && q38.recurrent.size() == 4,
                           "Qwen3.8 typed geometry");
        failures += !check(!strata::artifact::exl3::read_exl3_model_spec(q38_archive, q38, err, true) &&
                               err.find("external official raw MTP") != std::string::npos,
                           "Qwen3.8 unbound MTP rejection");
        write_archive(root / "q35", base_config("qwen3_5_moe_text"));
        strata::artifact::exl3::Archive q35_archive(root / "q35");
        strata::ModelSpec q35;
        failures += !check(strata::artifact::exl3::read_exl3_model_spec(q35_archive, q35, err), "Qwen3.5 MoE trunk config");
        failures += !check(q35.arch == strata::ModelArch::Qwen35Moe && q35.arch_name == "qwen35moe" &&
                               q35.n_layer == 4 && q35.n_nextn == 0 && q35.n_head_kv.size() == 4 &&
                               q35.recurrent.size() == 4 && q35.recurrent[0] && q35.recurrent[1] && q35.recurrent[2] &&
                               !q35.recurrent[3] && q35.ssm_d_inner == 4096 && q35.ssm_d_state == 128 &&
                               q35.ssm_dt_rank == 32 && q35.ssm_n_group == 16 && q35.ssm_d_conv == 4 &&
                               q35.n_expert == 256 && q35.n_expert_used == 8 && q35.n_ff_shexp == 512 &&
                               q35.n_expert_shared == 1 && q35.rope_dims == 32,
                           "Qwen3.5 MoE typed geometry");
        failures += !check(!strata::artifact::exl3::read_exl3_model_spec(q35_archive, q35, err, true) &&
                               err.find("Qwen3.5 MTP was requested") != std::string::npos,
                           "Qwen3.5 unbound MTP rejection");
        Json lfm_text = {
            {"model_type", "lfm2_moe"}, {"hidden_size", 2048}, {"vocab_size", 128000}, {"num_hidden_layers", 4},
            {"num_attention_heads", 32}, {"num_key_value_heads", 8}, {"max_position_embeddings", 131072},
            {"norm_eps", 1e-5}, {"conv_L_cache", 3}, {"num_dense_layers", 1}, {"intermediate_size", 11776},
            {"layer_types", {"conv", "conv", "full_attention", "conv"}},
            {"num_experts", 64}, {"num_experts_per_tok", 4}, {"moe_intermediate_size", 1536},
            {"use_expert_bias", true}, {"norm_topk_prob", true}, {"routed_scaling_factor", 1.0},
            {"rope_parameters", Json{{"rope_theta", 10000000.0}, {"rope_type", "default"}}}};
        write_archive(root / "lfm2", Json{{"model_type", "lfm2_moe"}, {"text_config", lfm_text}});
        strata::artifact::exl3::Archive lfm_archive(root / "lfm2");
        strata::ModelSpec lfm;
        failures += !check(strata::artifact::exl3::read_exl3_model_spec(lfm_archive, lfm, err), "LFM2 MoE config");
        failures += !check(lfm.arch == strata::ModelArch::Lfm2Moe && lfm.n_layer == 4 && lfm.key_length == 64 &&
                               lfm.recurrent[0] && !lfm.recurrent[2] && lfm.n_head_kv[2] == 8 && lfm.n_head_kv[0] == 0 &&
                               lfm.shortconv_l_cache == 3 && lfm.n_dense_lead == 1 && lfm.n_ff == 11776 &&
                               lfm.expert_bias && lfm.expert_weights_norm && lfm.gating == strata::ExpertGating::Sigmoid,
                           "LFM2 MoE typed geometry");
        Json malformed = base_config("qwen3_5_moe_text");
        malformed["text_config"].erase("linear_key_head_dim");
        write_archive(root / "bad", malformed);
        strata::artifact::exl3::Archive bad_archive(root / "bad");
        failures += !check(!strata::artifact::exl3::read_exl3_model_spec(bad_archive, q38, err) &&
                               err.find("linear_key_head_dim") != std::string::npos,
                           "missing required geometry rejection");
    } catch (const std::exception& e) {
        std::cerr << "model spec fixture failed: " << e.what() << '\n';
        failures = 1;
    }
    std::filesystem::remove_all(root);
    return failures;
}
