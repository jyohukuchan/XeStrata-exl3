// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Focused EXL3 external WeightTable registration test.
#include "strata/core/exl3_weight_set.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace {

using Json = nlohmann::json;
using Tensor = std::tuple<std::string, std::string, std::vector<uint64_t>, std::vector<uint8_t>>;

std::vector<uint8_t> words(size_t count, uint16_t seed) {
    std::vector<uint8_t> result(count * 2);
    for (size_t i = 0; i < count; ++i) {
        const uint16_t value = static_cast<uint16_t>(seed + i * 13);
        result[2 * i] = static_cast<uint8_t>(value);
        result[2 * i + 1] = static_cast<uint8_t>(value >> 8);
    }
    return result;
}

std::vector<uint8_t> marker(uint32_t value) {
    return {static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8), static_cast<uint8_t>(value >> 16),
            static_cast<uint8_t>(value >> 24)};
}

void write_archive(const std::filesystem::path& root) {
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json") << R"({"model_type":"synthetic-exl3-weight-set"})";
    std::vector<Tensor> tensors;
    tensors.emplace_back("packed.trellis", "I16", std::vector<uint64_t>{8, 8, 32}, words(8 * 8 * 32, 7));
    tensors.emplace_back("packed.suh", "F16", std::vector<uint64_t>{128}, words(128, 0x3c00));
    tensors.emplace_back("packed.svh", "F16", std::vector<uint64_t>{128}, words(128, 0x3c00));
    tensors.emplace_back("packed.mul1", "I32", std::vector<uint64_t>{1}, marker(0x83DCD12Du));
    tensors.emplace_back("blk.0.ssm_norm.weight", "F16", std::vector<uint64_t>{4}, words(4, 0x3c00));
    tensors.emplace_back("blk.0.router.weight", "F16", std::vector<uint64_t>{2, 128}, words(2 * 128, 9));
    Json header = Json::object();
    uint64_t offset = 0;
    for (const auto& [name, dtype, shape, data] : tensors) {
        header[name] = Json{{"dtype", dtype}, {"shape", shape}, {"data_offsets", {offset, offset + data.size()}}};
        offset += data.size();
    }
    const std::string text = header.dump();
    std::ofstream output(root / "model.safetensors", std::ios::binary);
    for (int i = 0; i < 8; ++i) output.put(static_cast<char>(text.size() >> (8 * i)));
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    for (const auto& [name, dtype, shape, data] : tensors) {
        (void) name; (void) dtype; (void) shape;
        output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
    Json index;
    index["weight_map"] = Json::object();
    for (const auto& [name, dtype, shape, data] : tensors) {
        (void) dtype; (void) shape; (void) data;
        index["weight_map"][name] = "model.safetensors";
    }
    std::ofstream(root / "model.safetensors.index.json") << index.dump();
}

strata::artifact::exl3::ModelMap model_map() {
    using namespace strata::artifact::exl3;
    ModelMap map;
    map.family = ModelFamily::qwen38;
    map.geometry.hidden_size = 128;
    map.geometry.layers = 1;
    map.geometry.experts = 2;
    ProjectionMapping packed;
    packed.kind = ProjectionKind::qsa_q;
    packed.layer = 0;
    packed.source_name = "packed";
    packed.canonical_name = "blk.0.attn_q.weight";
    packed.transform = MappingTransform::identity;
    packed.target_k = 128;
    packed.target_n = 128;
    map.projections.push_back(packed);
    ProjectionMapping norm;
    norm.kind = ProjectionKind::auxiliary;
    norm.layer = 0;
    norm.source_name = "blk.0.ssm_norm.weight";
    norm.canonical_name = "blk.0.ssm_norm.weight";
    norm.transform = MappingTransform::raw_rows;
    norm.value_transform = ValueTransform::add_one;
    map.projections.push_back(norm);
    ProjectionMapping router;
    router.kind = ProjectionKind::router;
    router.layer = 0;
    router.source_name = "blk.0.router.weight";
    router.canonical_name = "blk.0.ffn_gate_inp.weight";
    router.transform = MappingTransform::raw_rows;
    router.target_k = 128;
    router.target_n = 2;
    map.projections.push_back(router);
    return map;
}

}  // namespace

int main() {
    const auto root = std::filesystem::temp_directory_path() /
        ("xestrata_exl3_weight_set_fixture_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (!std::filesystem::create_directory(root)) return 1;
    int failures = 0;
    try {
        write_archive(root);
        if (strata::core::Runtime::count() == 0) {
            std::filesystem::remove_all(root);
            return 77;
        }
        strata::artifact::exl3::Archive archive(root);
        const auto map = model_map();
        strata::core::WeightTable table;
        strata::core::Exl3WeightSetOptions options;
        options.layer_end = 1;
        strata::core::Exl3WeightSet set;
        std::string err;
        auto unsupported = map;
        unsupported.unsupported.push_back("multiple projections target a canonical row");
        strata::core::Exl3WeightSet rejected;
        const bool rejected_invalid = !rejected.load(archive, unsupported, table, options, err) && table.all().empty();
        failures += !rejected_invalid;
        rejected.close();
        if (!set.load(archive, map, table, options, err)) {
            std::cerr << "weight set load failed: " << err << '\n';
            std::filesystem::remove_all(root);
            return 1;
        }
        const auto* packed = table.find("blk.0.attn_q.weight");
        const auto* norm = table.find("blk.0.ssm_norm.weight");
        const auto* router = table.find("blk.0.ffn_gate_inp.weight");
        const bool metadata_ok = packed != nullptr && packed->exl3 != nullptr && packed->ne0 == 128 &&
                                 packed->ne1 == 128 && packed->bytes == 4608 && norm != nullptr &&
                                 norm->data != nullptr && norm->kind == strata::core::WeightKind::F32 &&
                                 norm->bytes == 16 && router != nullptr && router->exl3 != nullptr &&
                                 router->ne0 == 128 && router->ne1 == 2 && router->bytes == 512;
        std::cout << "packed/raw auxiliary/raw router registration: " << (metadata_ok ? "OK" : "FAIL") << '\n';
        failures += !metadata_ok;
        float first_norm = 0.0f;
        const bool transformed = norm != nullptr && strata::gpu::copy(&first_norm, norm->data, sizeof(float)) &&
                                 first_norm == 2.0f;
        failures += !transformed;
        strata::core::Exl3WeightSet duplicate;
        failures += duplicate.load(archive, map, table, options, err);
        duplicate.close();
        failures += table.find("blk.0.attn_q.weight") == nullptr ||
                    table.find("blk.0.attn_q.weight")->exl3 != packed->exl3;
        if (!set.prepare_stream(nullptr, err)) {
            std::cerr << "weight set stream preparation failed: " << err << '\n';
            ++failures;
        }
        set.close();
        const bool detached = table.find("blk.0.attn_q.weight") == nullptr &&
                              table.find("blk.0.ssm_norm.weight") == nullptr &&
                              table.find("blk.0.ffn_gate_inp.weight") == nullptr;
        std::cout << "bundle close removes only its rows: " << (detached ? "OK" : "FAIL") << '\n';
        failures += !detached;
    } catch (const std::exception& e) {
        std::cerr << "weight set fixture failed: " << e.what() << '\n';
        failures = 1;
    }
    std::filesystem::remove_all(root);
    return failures;
}
