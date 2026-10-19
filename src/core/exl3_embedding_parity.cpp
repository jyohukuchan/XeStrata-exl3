// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// GPU parity for the raw EXL3 token embedding owner.
#include "strata/artifact/exl3.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/raw_linear.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#define STRATA_EXL3_EMBED_PID _getpid()
#else
#include <unistd.h>
#define STRATA_EXL3_EMBED_PID getpid()
#endif

namespace {

using Json = nlohmann::json;
using Tensor = std::tuple<std::string, std::string, std::vector<uint64_t>, std::vector<uint8_t>>;

std::vector<uint8_t> bytes_for(const std::vector<float>& values, const std::string& dtype) {
    const size_t width = dtype == "F32" ? 4 : 2;
    std::vector<uint8_t> bytes(values.size() * width);
    for (size_t i = 0; i < values.size(); ++i) {
        if (dtype == "F32") {
            std::memcpy(bytes.data() + i * 4, &values[i], 4);
        } else {
            const uint16_t bits = dtype == "BF16" ? strata::kernels::bf16_from_f32(values[i])
                                                   : strata::kernels::f16_from_f32(values[i]);
            bytes[i * 2] = static_cast<uint8_t>(bits);
            bytes[i * 2 + 1] = static_cast<uint8_t>(bits >> 8);
        }
    }
    return bytes;
}

void write_safetensors(const std::filesystem::path& path, const std::vector<Tensor>& tensors) {
    Json header = Json::object();
    uint64_t offset = 0;
    for (const auto& [name, dtype, shape, data] : tensors) {
        header[name] = Json{{"dtype", dtype}, {"shape", shape}, {"data_offsets", {offset, offset + data.size()}}};
        offset += data.size();
    }
    const std::string text = header.dump();
    std::ofstream out(path, std::ios::binary);
    const uint64_t size = text.size();
    for (int i = 0; i < 8; ++i) out.put(static_cast<char>(size >> (i * 8)));
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    for (const auto& [name, dtype, shape, data] : tensors) {
        (void) name; (void) dtype; (void) shape;
        out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
}

struct Fixture {
    std::filesystem::path root;
    int64_t n_embd = 0;
    int64_t n_vocab = 0;
    std::vector<float> values;
};

Fixture make_fixture(const std::filesystem::path& root, const std::string& dtype, int64_t n_embd = 256,
                     int64_t n_vocab = 7) {
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json") << R"({"model_type":"synthetic-exl3-raw-embedding"})";
    std::vector<float> values(static_cast<size_t>(n_vocab) * static_cast<size_t>(n_embd));
    for (int64_t token = 0; token < n_vocab; ++token)
        for (int64_t col = 0; col < n_embd; ++col)
            values[static_cast<size_t>(token) * static_cast<size_t>(n_embd) + static_cast<size_t>(col)] =
                0.01f * static_cast<float>((token * 13 + col * 7) % 97 - 48);
    write_safetensors(root / "model.safetensors",
                      {{"token_embd.weight", dtype, {(uint64_t) n_vocab, (uint64_t) n_embd}, bytes_for(values, dtype)}});
    return {root, n_embd, n_vocab, std::move(values)};
}

bool make_embedding_metadata_pack(const std::filesystem::path& root, int64_t n_embd, int64_t n_vocab,
                                  strata::core::WeightTable& table, void*& arena, std::string& error) {
    std::filesystem::create_directories(root);
    const uint64_t bytes = static_cast<uint64_t>(n_embd) * static_cast<uint64_t>(n_vocab) * sizeof(float);
    std::vector<uint8_t> zeros(static_cast<size_t>(bytes), 0);
    std::ofstream(root / "dense.bin", std::ios::binary).write(reinterpret_cast<const char*>(zeros.data()),
                                                                static_cast<std::streamsize>(zeros.size()));
    std::ofstream index(root / "index.txt");
    index << "# strata synthetic embedding metadata\n# align 256 pool " << bytes << " tensors 1\n"
          << "token_embd.weight 0 2 0 " << bytes << " 0 " << bytes << ' ' << n_embd << ' ' << n_vocab
          << " 0 0 0 0 0 0 0 0 0 0\n";
    index.close();
    uint64_t pool = 0;
    if (!strata::core::WeightTable::pool_bytes(root.string(), pool, error)) return false;
    arena = strata::gpu::alloc_device(static_cast<size_t>(pool));
    return arena != nullptr && table.load(root.string(), arena, pool, error);
}

bool close(float a, float b, const std::string& dtype) {
    if (dtype == "F32") return a == b;
    const float tolerance = dtype == "BF16" ? 1e-2f : 2e-3f;
    return std::abs(a - b) <= tolerance * std::max(1.0f, std::abs(b));
}

int check_gather(strata::core::NativeEmbed& embed, const Fixture& fixture, const std::string& dtype, void* stream) {
    const int32_t ids[] = {2, 0, 2, 99};
    constexpr int64_t rows = 4;
    const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(fixture.n_embd);
    std::vector<float> host(count, -9917.0f);
    auto* d_ids = static_cast<int32_t*>(strata::gpu::alloc_device(sizeof(ids)));
    auto* d_out = static_cast<float*>(strata::gpu::alloc_device(count * sizeof(float)));
    if (!d_ids || !d_out) return 1;
    strata::gpu::copy(d_ids, ids, sizeof(ids));
    strata::gpu::copy(d_out, host.data(), count * sizeof(float));
    embed.gather_dev(d_ids, rows, d_out, stream);
    if (!strata::gpu::stream_sync(stream)) return 2;
    strata::gpu::copy(host.data(), d_out, count * sizeof(float));
    int failures = 0;
    for (int64_t row = 0; row < rows; ++row)
        for (int64_t col = 0; col < fixture.n_embd; ++col) {
            const float expected = ids[row] < 0 || ids[row] >= fixture.n_vocab
                                       ? 0.0f
                                       : fixture.values[static_cast<size_t>(ids[row]) * fixture.n_embd + col];
            if (!close(host[static_cast<size_t>(row) * fixture.n_embd + col], expected, dtype)) ++failures;
        }
    strata::gpu::free(d_ids);
    strata::gpu::free(d_out);
    return failures;
}

int check_one_and_capture(strata::core::NativeEmbed& embed, const Fixture& fixture, void* stream) {
    auto* d_out = static_cast<float*>(strata::gpu::alloc_device(static_cast<size_t>(fixture.n_embd) * sizeof(float)));
    if (!d_out) return 1;
    embed.gather_one(3, d_out, stream);
    if (!strata::gpu::stream_sync(stream)) return 2;
    std::vector<float> host(static_cast<size_t>(fixture.n_embd));
    strata::gpu::copy(host.data(), d_out, host.size() * sizeof(float));
    int failures = 0;
    for (int64_t col = 0; col < fixture.n_embd; ++col)
        if (!close(host[static_cast<size_t>(col)], fixture.values[3 * fixture.n_embd + col], "F32")) ++failures;

    std::fill(host.begin(), host.end(), -9917.0f);
    strata::gpu::copy(d_out, host.data(), host.size() * sizeof(float));
    strata::gpu::begin_capture(stream);
    embed.gather_one(1, d_out, stream);
    strata::gpu::Graph* graph = nullptr;
    if (!strata::gpu::end_capture(stream, &graph) || graph == nullptr || !strata::gpu::graph_launch(graph, stream) ||
        !strata::gpu::stream_sync(stream)) {
        if (graph) strata::gpu::graph_destroy(graph);
        strata::gpu::free(d_out);
        return failures + 1;
    }
    strata::gpu::copy(host.data(), d_out, host.size() * sizeof(float));
    for (int64_t col = 0; col < fixture.n_embd; ++col)
        if (!close(host[static_cast<size_t>(col)], fixture.values[fixture.n_embd + col], "F32")) ++failures;
    strata::gpu::graph_destroy(graph);
    strata::gpu::free(d_out);
    return failures;
}

}  // namespace

int main() {
    int failures = 0;
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto base = std::filesystem::temp_directory_path() /
                      ("xestrata_exl3_embedding_parity_" + std::to_string(STRATA_EXL3_EMBED_PID) +
                       "_" + std::to_string(stamp));
    std::filesystem::remove_all(base);
    auto stream = strata::gpu::stream_create();
    if (!stream) return 2;
    const strata::core::ModelGeometry geometry{};
    for (const std::string dtype : {"F16", "BF16", "F32"}) {
        const Fixture fixture = make_fixture(base / dtype, dtype);
        try {
            strata::artifact::exl3::Archive archive(fixture.root);
            strata::core::NativeEmbed embed;
            std::string error;
            if (!embed.load_exl3(archive, "token_embd", fixture.n_embd, fixture.n_vocab, error)) {
                std::cerr << dtype << " load: " << error << '\n'; ++failures; continue;
            }
            if (!embed.prepare_stream(stream, error)) {
                std::cerr << dtype << " stream prepare: " << error << '\n'; ++failures; continue;
            }
            const int expected_type = dtype == "F16" ? 1 : dtype == "BF16" ? 30 : 0;
            if (embed.type() != expected_type) { std::cerr << dtype << " type mismatch\n"; ++failures; }
            const int gather_failures = check_gather(embed, fixture, dtype, stream);
            if (gather_failures) std::cerr << dtype << " gather failures: " << gather_failures << '\n';
            failures += gather_failures;
            if (dtype == "F32") {
                const int one_failures = check_one_and_capture(embed, fixture, stream);
                if (one_failures) std::cerr << "F32 single/capture failures: " << one_failures << '\n';
                failures += one_failures;
                strata::core::set_native_embed(&embed);
                auto* d_out = static_cast<float*>(strata::gpu::alloc_device(static_cast<size_t>(fixture.n_embd) * sizeof(float)));
                if (!d_out) ++failures;
                else {
                    std::string err;
                    strata::core::WeightTable metadata_table;
                    void* metadata_arena = nullptr;
                    if (!make_embedding_metadata_pack(base / "metadata-pack", fixture.n_embd, fixture.n_vocab,
                                                      metadata_table, metadata_arena, err) ||
                        !strata::core::embed_row(metadata_table, geometry, 2, d_out, stream, err)) {
                        std::cerr << "public embed_row failed: " << err << '\n'; ++failures;
                    }
                    if (metadata_arena) strata::gpu::free(metadata_arena);
                    strata::gpu::free(d_out);
                }
                strata::core::set_native_embed(nullptr);
            }
        } catch (const std::exception& e) {
            std::cerr << dtype << " exception: " << e.what() << '\n'; ++failures;
        }
    }
    try {
        int32_t id = 0;
        float out = 0.0f;
        strata::kernels::raw::embedding_gather(reinterpret_cast<const void*>(0x1000),
                                                strata::kernels::raw::DType::F16, 255, &id, 1, 1, 128, &out, 128,
                                                stream);
        ++failures;
    } catch (const std::invalid_argument&) {
        std::cout << "invalid raw embedding stride rejected before enqueue\n";
    }
    // A malformed tensor must leave the owner reusable for a valid load.
    try {
        const Fixture bad = make_fixture(base / "bad", "F32", 128, 7);
        strata::artifact::exl3::Archive archive(bad.root);
        strata::core::NativeEmbed embed;
        std::string error;
        if (embed.load_exl3(archive, "token_embd", 256, 7, error)) ++failures;
        const Fixture good = make_fixture(base / "good", "F32");
        strata::artifact::exl3::Archive good_archive(good.root);
        if (!embed.load_exl3(good_archive, "token_embd", good.n_embd, good.n_vocab, error)) ++failures;
    } catch (...) { ++failures; }
    strata::gpu::stream_destroy(stream);
    std::filesystem::remove_all(base);
    std::cout << "exl3_embedding_parity: " << failures << " failures\n";
    return failures ? 1 : 0;
}
