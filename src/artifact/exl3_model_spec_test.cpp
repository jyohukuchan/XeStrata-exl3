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
