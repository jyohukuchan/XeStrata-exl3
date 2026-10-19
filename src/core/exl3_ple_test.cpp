// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Bounded validation for the EXL3 PLE owner. A real archive path is optional;
// the default checks are CPU-only and do not allocate model weights.

#include "strata/core/exl3_ple.hpp"
#include "strata/core/exl3_dense.hpp"
#include "strata/core/gpu.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <chrono>
#include <cstring>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

using strata::artifact::exl3::Archive;
using strata::artifact::exl3::ModelMap;

std::filesystem::path fixture(const std::string &config) {
    const auto root =
        std::filesystem::temp_directory_path() /
        ("xestrata_exl3_ple_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json") << config;
    const std::string header = "{}";
    std::ofstream model(root / "model.safetensors", std::ios::binary);
    const uint64_t bytes = header.size();
    for (int i = 0; i < 8; ++i)
        model.put(static_cast<char>(bytes >> (8 * i)));
    model.write(header.data(), static_cast<std::streamsize>(header.size()));
    return root;
}

bool rejects_bad_geometry() {
    const auto root = fixture(
        R"({"model_type":"qwen4_exp","text_config":{"ple_embed_dim":1,"ple_conv_kernel_size":4,"ngram_size":3,"heads_per_ngram":8,"ple_layer_ids":[2]}})");
    Archive archive(root);
    ModelMap map;
    map.source_prefix = "model.language_model.layers";
    strata::core::Exl3Ple owner;
    std::string err;
    const bool loaded = owner.load(archive, map, 0, 1, err);
    std::filesystem::remove_all(root);
    if (loaded || err.find("ple_embed_dim") == std::string::npos) {
        std::cerr << "bad PLE geometry was accepted: " << err << '\n';
        return false;
    }
    return true;
}

bool rejects_missing_projection() {
    const auto root = fixture(
        R"({"model_type":"qwen4_exp","text_config":{"ple_embed_dim":2560,"ple_conv_kernel_size":4,"ngram_size":3,"heads_per_ngram":8,"ple_layer_ids":[2]}})");
    Archive archive(root);
    ModelMap map;
    map.source_prefix = "model.language_model.layers";
    map.geometry.layers = 48;
    strata::core::Exl3Ple owner;
    std::string err;
    const bool loaded = owner.load(archive, map, 0, 1, err);
    std::filesystem::remove_all(root);
    if (loaded || err.find("key_proj") == std::string::npos) {
        std::cerr << "missing PLE projection did not fail clearly: " << err << '\n';
        return false;
    }
    return true;
}

bool inspect_real_archive(const std::filesystem::path &path) {
    try {
        Archive archive(path);
        strata::core::Exl3Ple owner;
        std::string err;
        ModelMap map;
        if (!strata::artifact::exl3::build_model_map(archive, map, err) || !owner.load(archive, map, 0, 1, err)) {
            std::cerr << "real PLE archive load failed: " << err << '\n';
            return false;
        }
        const auto &weights = owner.weights();
        if (owner.key() == nullptr || owner.value() == nullptr || weights.norm_key == nullptr ||
            weights.norm_query == nullptr || weights.norm_conv == nullptr || weights.conv1d_f16 == nullptr ||
            owner.pre_key() == nullptr || owner.pre_value() == nullptr) {
            std::cerr << "real PLE archive left an incomplete owner\n";
            return false;
        }
        const std::string prefix = map.source_prefix + ".1.ple.";
        const auto check_norm = [&](const char* role, const float* device) {
            const auto* tensor = archive.find_tensor(prefix + role + ".weight");
            if (tensor == nullptr) return false;
            std::vector<uint8_t> source;
            archive.read_tensor(*tensor, source);
            std::vector<float> result(static_cast<size_t>(tensor->shape.at(0)));
            if (!strata::gpu::copy(result.data(), device, result.size() * sizeof(float))) return false;
            for (size_t i = 0; i < result.size(); ++i) {
                float value = 0.0f;
                if (tensor->dtype == "F32") std::memcpy(&value, source.data() + i * 4, 4);
                else {
                    const uint16_t bits = static_cast<uint16_t>(source[i * 2]) |
                                          static_cast<uint16_t>(source[i * 2 + 1] << 8);
                    value = tensor->dtype == "BF16" ? strata::kernels::f32_from_bf16(bits)
                                                     : strata::kernels::f32_from_f16(bits);
                }
                if (result[i] != 1.0f + value) return false;
            }
            return true;
        };
        if (!check_norm("norm_key", weights.norm_key) || !check_norm("norm_query", weights.norm_query) ||
            !check_norm("norm_conv", weights.norm_conv)) {
            std::cerr << "real PLE norm folding did not preserve source+1\n";
            return false;
        }
        const auto* conv = archive.find_tensor(prefix + "conv1d.weight");
        if (conv == nullptr) return false;
        std::vector<uint8_t> conv_source;
        archive.read_tensor(*conv, conv_source);
        std::vector<uint8_t> conv_device(conv_source.size());
        if (!strata::gpu::copy(conv_device.data(), weights.conv1d_f16, conv_device.size()) ||
            conv_device != conv_source) {
            std::cerr << "real PLE convolution bytes changed\n";
            return false;
        }
        if (!owner.close(err)) { std::cerr << err << '\n'; return false; }
        return true;
    } catch (const std::exception &e) {
        std::cerr << "real PLE archive exception: " << e.what() << '\n';
        return false;
    }
}

} // namespace

int main(int argc, char **argv) {
    if (!rejects_bad_geometry() || !rejects_missing_projection())
        return 1;
    if (argc > 1 && !inspect_real_archive(argv[1]))
        return 1;
    std::cout << "EXL3 PLE owner checks: OK\n";
    return 0;
}
