// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Raw F16/BF16/F32 EXL3-head owner parity on synthetic safetensors.
#include "strata/artifact/exl3.hpp"
#include "strata/core/exl3_dense.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;

struct RawCase {
    const char* weight_dtype;
    const char* bias_dtype;
};

std::vector<uint8_t> bytes_f16(const std::vector<float>& values) {
    std::vector<uint8_t> result(values.size() * 2);
    for (size_t i = 0; i < values.size(); ++i) {
        const uint16_t bits = strata::kernels::f16_from_f32(values[i]);
        result[2 * i] = static_cast<uint8_t>(bits);
        result[2 * i + 1] = static_cast<uint8_t>(bits >> 8);
    }
    return result;
}

std::vector<uint8_t> bytes_bf16(const std::vector<float>& values) {
    std::vector<uint8_t> result(values.size() * 2);
    for (size_t i = 0; i < values.size(); ++i) {
        const uint16_t bits = strata::kernels::bf16_from_f32(values[i]);
        result[2 * i] = static_cast<uint8_t>(bits);
        result[2 * i + 1] = static_cast<uint8_t>(bits >> 8);
    }
    return result;
}

std::vector<uint8_t> bytes_f32(const std::vector<float>& values) {
    std::vector<uint8_t> result(values.size() * 4);
    std::memcpy(result.data(), values.data(), result.size());
    return result;
}

void write_safetensors(const std::filesystem::path& path, const std::vector<std::tuple<std::string, std::string,
                                                                                        std::vector<uint64_t>, std::vector<uint8_t>>>& tensors) {
    Json header = Json::object();
    uint64_t offset = 0;
    for (const auto& [name, dtype, shape, data] : tensors) {
        header[name] = Json{{"dtype", dtype}, {"shape", shape}, {"data_offsets", {offset, offset + data.size()}}};
        offset += data.size();
    }
    const std::string text = header.dump();
    std::ofstream output(path, std::ios::binary);
    const uint64_t size = text.size();
    for (int i = 0; i < 8; ++i) output.put(static_cast<char>(size >> (8 * i)));
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    for (const auto& [name, dtype, shape, data] : tensors) {
        (void) name; (void) dtype; (void) shape;
        output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
}

std::filesystem::path make_archive(const std::filesystem::path& root, const RawCase& raw_case, int64_t k, int64_t n,
                                   std::vector<float>& weights, std::vector<float>& bias, bool malformed_bias = false) {
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json") << R"({"model_type":"synthetic-raw-head"})";
    weights.resize((size_t) n * (size_t) k);
    bias.resize((size_t) n);
    for (int64_t row = 0; row < n; ++row) {
        bias[(size_t) row] = 0.01f * (float) (row - 7);
        for (int64_t col = 0; col < k; ++col)
            weights[(size_t) row * (size_t) k + (size_t) col] = 0.002f * (float) ((row * 11 + col * 3) % 37 - 18);
    }
    std::vector<uint8_t> weight_bytes;
    if (std::string(raw_case.weight_dtype) == "F16") weight_bytes = bytes_f16(weights);
    else if (std::string(raw_case.weight_dtype) == "BF16") weight_bytes = bytes_bf16(weights);
    else weight_bytes = bytes_f32(weights);
    std::vector<uint8_t> bias_bytes;
    if (std::string(raw_case.bias_dtype) == "F16") bias_bytes = bytes_f16(bias);
    else if (std::string(raw_case.bias_dtype) == "BF16") bias_bytes = bytes_bf16(bias);
    else bias_bytes = bytes_f32(bias);
    const size_t bias_count = malformed_bias ? static_cast<size_t>(n - 1) : static_cast<size_t>(n);
    bias_bytes.resize(bias_count * (std::string(raw_case.bias_dtype) == "F32" ? 4 : 2));
    write_safetensors(root / "model.safetensors",
                      {{"lm_head.weight", raw_case.weight_dtype, {(uint64_t) n, (uint64_t) k}, weight_bytes},
                       {"lm_head.bias", raw_case.bias_dtype, {(uint64_t) bias_count}, bias_bytes}});
    std::ofstream dense(root / "dense.bin", std::ios::binary);
    const std::vector<uint8_t> canonical((size_t) k * (size_t) n * 4, 0);
    dense.write(reinterpret_cast<const char*>(canonical.data()), static_cast<std::streamsize>(canonical.size()));
    std::ofstream index(root / "index.txt");
    index << "# strata synthetic raw head\n# align 256 pool " << canonical.size() << " tensors 1\n";
    index << "output.weight 0 2 0 " << canonical.size() << " 0 " << canonical.size() << ' ' << k << ' ' << n
          << " 0 0 0 0 0 0 0 0 0 0\n";
    return root;
}

bool close(float got, float expected) {
    return std::isfinite(got) && std::isfinite(expected) &&
           std::fabs(got - expected) <= 2e-4f * std::max(1.0f, std::fabs(expected));
}

int run_case(const std::filesystem::path& root, const RawCase& raw_case) {
    constexpr int64_t k = 128, n = 128, rows = 4, ldx = 131, ldy = 133;
    std::vector<float> weights, bias;
    make_archive(root, raw_case, k, n, weights, bias);
    strata::core::WeightTable table;
    std::string err;
    uint64_t pool_bytes = 0;
    if (!strata::core::WeightTable::pool_bytes(root.string(), pool_bytes, err)) {
        std::cerr << err << '\n'; return 1;
    }
    void* arena = strata::gpu::alloc_device(pool_bytes);
    if (arena == nullptr || !table.load(root.string(), arena, pool_bytes, err)) {
        std::cerr << "raw parity table: " << (err.empty() ? strata::gpu::last_error() : err) << '\n';
        if (arena != nullptr) strata::gpu::free(arena);
        return 1;
    }
    strata::core::Exl3Dense owner;
    if (!owner.load(root, "lm_head", table, err, rows)) {
        std::cerr << "raw parity load: " << err << '\n';
        strata::gpu::free(arena);
        return 1;
    }
    auto stream = strata::gpu::stream_create();
    if (stream == nullptr || !owner.prepare_stream(stream, err)) {
        std::cerr << "raw parity stream: " << (err.empty() ? strata::gpu::last_error() : err) << '\n';
        if (stream != nullptr) strata::gpu::stream_destroy(stream);
        strata::gpu::free(arena);
        return 1;
    }
    std::vector<float> input((size_t) rows * (size_t) ldx), expected((size_t) rows * (size_t) ldy, 9917.0f), actual = expected;
    for (int64_t row = 0; row < rows; ++row)
        for (int64_t col = 0; col < k; ++col)
            input[(size_t) row * (size_t) ldx + (size_t) col] = 0.01f * (float) ((row * 7 + col * 5) % 43 - 21);
    for (int64_t row = 0; row < rows; ++row)
        for (int64_t out = 0; out < n; ++out) {
            float rounded_bias = bias[(size_t) out];
            if (std::string(raw_case.bias_dtype) == "F16") rounded_bias = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(rounded_bias));
            if (std::string(raw_case.bias_dtype) == "BF16") rounded_bias = strata::kernels::f32_from_bf16(strata::kernels::bf16_from_f32(rounded_bias));
            double sum = rounded_bias;
            for (int64_t in = 0; in < k; ++in) {
                float weight = weights[(size_t) out * (size_t) k + (size_t) in];
                if (std::string(raw_case.weight_dtype) == "F16") weight = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(weight));
                if (std::string(raw_case.weight_dtype) == "BF16") weight = strata::kernels::f32_from_bf16(strata::kernels::bf16_from_f32(weight));
                sum += (double) input[(size_t) row * (size_t) ldx + (size_t) in] * weight;
            }
            expected[(size_t) row * (size_t) ldy + (size_t) out] = (float) sum;
        }
    float* d_input = static_cast<float*>(strata::gpu::alloc_device(input.size() * sizeof(float)));
    float* d_output = static_cast<float*>(strata::gpu::alloc_device(actual.size() * sizeof(float)));
    strata::gpu::copy(d_input, input.data(), input.size() * sizeof(float));
    strata::gpu::copy(d_output, actual.data(), actual.size() * sizeof(float));
    if (!owner.run(d_input, ldx, d_output, ldy, rows, stream, err) || !strata::gpu::stream_sync(stream)) {
        std::cerr << "raw parity run: " << (err.empty() ? strata::gpu::last_error() : err) << '\n';
        return 1;
    }
    strata::gpu::copy(actual.data(), d_output, actual.size() * sizeof(float));
    bool ok = true;
    for (int64_t row = 0; row < rows && ok; ++row) {
        for (int64_t out = 0; out < n; ++out)
            ok = close(actual[(size_t) row * (size_t) ldy + (size_t) out], expected[(size_t) row * (size_t) ldy + (size_t) out]);
        for (int64_t out = n; out < ldy; ++out) ok = ok && actual[(size_t) row * (size_t) ldy + (size_t) out] == 9917.0f;
    }
    std::fill(actual.begin(), actual.end(), 9917.0f);
    strata::gpu::copy(d_output, actual.data(), actual.size() * sizeof(float));
    strata::gpu::Graph* graph = nullptr;
    if (!strata::gpu::begin_capture(stream) || !owner.run(d_input, ldx, d_output, ldy, 1, stream, err) ||
        !strata::gpu::end_capture(stream, &graph) || !strata::gpu::graph_launch(graph, stream) ||
        !strata::gpu::stream_sync(stream)) {
        std::cerr << "raw parity graph: " << (err.empty() ? strata::gpu::last_error() : err) << '\n';
        if (graph != nullptr) strata::gpu::graph_destroy(graph);
        strata::gpu::free(d_input); strata::gpu::free(d_output);
        strata::gpu::stream_destroy(stream); strata::gpu::free(arena);
        return 1;
    }
    strata::gpu::copy(actual.data(), d_output, actual.size() * sizeof(float));
    for (int64_t out = 0; out < n; ++out)
        ok = ok && close(actual[(size_t) out], expected[(size_t) out]);
    for (int64_t out = n; out < ldy; ++out) ok = ok && actual[(size_t) out] == 9917.0f;
    strata::gpu::graph_destroy(graph);
    strata::gpu::free(d_input); strata::gpu::free(d_output);
    strata::gpu::stream_destroy(stream); strata::gpu::free(arena);
    return ok ? 0 : 1;
}

int run_malformed_bias_case(const std::filesystem::path& root) {
    constexpr int64_t k = 128, n = 128;
    RawCase raw_case{"F32", "F32"};
    std::vector<float> weights, bias;
    make_archive(root, raw_case, k, n, weights, bias, true);
    strata::core::WeightTable table;
    std::string err;
    uint64_t pool_bytes = 0;
    if (!strata::core::WeightTable::pool_bytes(root.string(), pool_bytes, err)) return 1;
    void* arena = strata::gpu::alloc_device(pool_bytes);
    if (arena == nullptr || !table.load(root.string(), arena, pool_bytes, err)) {
        if (arena != nullptr) strata::gpu::free(arena);
        return 1;
    }
    strata::core::Exl3Dense owner;
    const bool rejected = !owner.load(root, "lm_head", table, err, 1);
    const auto* ref = table.find("output.weight");
    const bool unattached = ref != nullptr && ref->exl3 == nullptr;
    strata::gpu::free(arena);
    std::cout << "malformed raw bias [n-1]: " << (rejected && unattached ? "OK" : "FAIL") << '\n';
    return rejected && unattached ? 0 : 1;
}

}  // namespace

int main() {
    const auto root = std::filesystem::temp_directory_path() / "xestrata_exl3_dense_raw_parity";
    std::filesystem::remove_all(root);
    int failures = 0;
    failures += run_case(root / "f16", {"F16", "F16"});
    failures += run_case(root / "bf16", {"BF16", "BF16"});
    failures += run_case(root / "f32", {"F32", "F32"});
    failures += run_malformed_bias_case(root / "malformed-bias");
    std::filesystem::remove_all(root);
    std::cout << "exl3_dense_raw_parity: " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
