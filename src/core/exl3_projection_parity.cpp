// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Public layer-entry parity for a raw-F16 EXL3 router owner.

#include "strata/artifact/exl3.hpp"
#include "strata/core/exl3_dense.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#define STRATA_EXL3_PROJECTION_PID _getpid()
#else
#include <unistd.h>
#define STRATA_EXL3_PROJECTION_PID getpid()
#endif

namespace {

using Json = nlohmann::json;
using Tensor = std::tuple<std::string, std::string, std::vector<uint64_t>, std::vector<uint8_t>>;

std::vector<uint8_t> f16_bytes(const std::vector<float>& values) {
    std::vector<uint8_t> bytes(values.size() * 2);
    for (size_t i = 0; i < values.size(); ++i) {
        const uint16_t bits = strata::kernels::f16_from_f32(values[i]);
        bytes[2 * i] = static_cast<uint8_t>(bits);
        bytes[2 * i + 1] = static_cast<uint8_t>(bits >> 8);
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
    std::ofstream output(path, std::ios::binary);
    const uint64_t size = text.size();
    for (int i = 0; i < 8; ++i) output.put(static_cast<char>(size >> (8 * i)));
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    for (const auto& [name, dtype, shape, data] : tensors) {
        (void) name; (void) dtype; (void) shape;
        output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
}

std::filesystem::path make_router_archive(const std::filesystem::path& root, std::vector<float>& weights,
                                          std::vector<float>& input) {
    constexpr int64_t n_in = 128, n_out = 2;
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json") << R"({"model_type":"synthetic-exl3-router"})";
    weights.resize(static_cast<size_t>(n_in * n_out));
    input.resize(static_cast<size_t>(n_in));
    for (int64_t col = 0; col < n_in; ++col) {
        input[static_cast<size_t>(col)] = 0.019f * static_cast<float>((col * 7) % 31 - 15);
        weights[static_cast<size_t>(col)] = 0.011f * static_cast<float>((col % 13) - 6);
        weights[static_cast<size_t>(n_in + col)] = -0.007f * static_cast<float>((col % 11) - 5);
    }
    write_safetensors(root / "model.safetensors",
                      {{"blk.0.ffn_gate_inp.weight", "F16", {n_out, n_in}, f16_bytes(weights)}});

    // The canonical row is deliberately skipped: the public route must use
    // the attached raw owner and must never interpret null canonical bytes as
    // an S-form or BF16 tensor.
    constexpr uint64_t canonical_bytes = n_in * n_out * sizeof(float);
    std::ofstream(root / "dense.bin", std::ios::binary).write("", 0);
    std::ofstream index(root / "index.txt");
    index << "# strata synthetic raw router\n# align 256 pool " << canonical_bytes << " tensors 1\n";
    index << "blk.0.ffn_gate_inp.weight 0 2 0 " << canonical_bytes << " 0 " << canonical_bytes << ' '
          << n_in << ' ' << n_out << " 0 0 0 0 0 0 0 0 0 0\n";
    return root;
}

std::filesystem::path make_shared_archive(const std::filesystem::path& root,
                                          std::vector<std::vector<float>>& weights, std::vector<float>& input) {
    constexpr int64_t n = 128;
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json") << R"({"model_type":"synthetic-exl3-shared"})";
    input.resize(n);
    for (int64_t col = 0; col < n; ++col) input[static_cast<size_t>(col)] = 0.013f * static_cast<float>((col * 5) % 29 - 14);
    weights.resize(4);
    const char* names[] = {"blk.0.ffn_gate_inp_shexp.weight", "blk.0.ffn_gate_shexp.weight",
                           "blk.0.ffn_up_shexp.weight", "blk.0.ffn_down_shexp.weight"};
    const int64_t rows[] = {1, n, n, n};
    const int64_t cols[] = {n, n, n, n};
    std::vector<Tensor> tensors;
    for (int role = 0; role < 4; ++role) {
        weights[static_cast<size_t>(role)].resize(static_cast<size_t>(rows[role] * cols[role]));
        for (int64_t row = 0; row < rows[role]; ++row)
            for (int64_t col = 0; col < cols[role]; ++col)
                weights[static_cast<size_t>(role)][static_cast<size_t>(row * cols[role] + col)] =
                    0.0031f * static_cast<float>(((row + 3) * (col + 5) + role * 11) % 31 - 15);
        tensors.emplace_back(names[role], "F16", std::vector<uint64_t>{static_cast<uint64_t>(rows[role]),
                                                                          static_cast<uint64_t>(cols[role])},
                              f16_bytes(weights[static_cast<size_t>(role)]));
    }
    write_safetensors(root / "model.safetensors", tensors);
    const uint64_t bytes[] = {n * sizeof(float), n * n * sizeof(float), n * n * sizeof(float), n * n * sizeof(float)};
    uint64_t pool = 0;
    for (uint64_t value : bytes) pool += value;
    std::ofstream(root / "dense.bin", std::ios::binary).write("", 0);
    std::ofstream index(root / "index.txt");
    index << "# strata synthetic raw shared expert\n# align 256 pool " << pool << " tensors 4\n";
    uint64_t offset = 0;
    for (int role = 0; role < 4; ++role) {
        index << names[role] << " 0 2 " << offset << ' ' << bytes[role] << ' ' << offset << ' ' << bytes[role] << ' '
              << cols[role] << ' ' << rows[role] << " 0 0 0 0 0 0 0 0 0 0\n";
        offset += bytes[role];
    }
    return root;
}

bool close(float got, float expected) {
    return std::isfinite(got) && std::isfinite(expected) &&
           std::fabs(got - expected) <= 3e-3f * std::max(1.0f, std::fabs(expected));
}

int run_shared_owner_case(const std::filesystem::path& root) {
    constexpr int64_t n = 128;
    std::vector<std::vector<float>> weights;
    std::vector<float> input;
    make_shared_archive(root, weights, input);
    strata::core::WeightTable table;
    std::string err;
    const std::set<std::string> skip{"blk.0.ffn_gate_inp_shexp.weight", "blk.0.ffn_gate_shexp.weight",
                                     "blk.0.ffn_up_shexp.weight", "blk.0.ffn_down_shexp.weight"};
    uint64_t pool_bytes = 0;
    if (!strata::core::WeightTable::pool_bytes(root.string(), pool_bytes, err, &skip) ||
        !table.load(root.string(), nullptr, 0, err, &skip)) {
        std::cerr << "shared table setup: " << err << '\n';
        return 1;
    }
    strata::core::Exl3Dense owners[4];
    const char* bases[] = {"blk.0.ffn_gate_inp_shexp", "blk.0.ffn_gate_shexp", "blk.0.ffn_up_shexp",
                           "blk.0.ffn_down_shexp"};
    const char* canonical[] = {"blk.0.ffn_gate_inp_shexp.weight", "blk.0.ffn_gate_shexp.weight",
                               "blk.0.ffn_up_shexp.weight", "blk.0.ffn_down_shexp.weight"};
    for (int i = 0; i < 4; ++i) {
        if (!owners[i].load(root, bases[i], canonical[i], table, err, 1)) {
            std::cerr << "shared owner " << i << ": " << err << '\n';
            return 1;
        }
    }
    strata::core::ModelGeometry geometry;
    geometry.n_embd = n;
    geometry.n_ff = n;
    geometry.n_expert = 2;
    geometry.n_layers = 1;
    const uint64_t bytes = strata::core::moe_buffers_bytes(geometry, 1);
    void* arena = strata::gpu::alloc_device(bytes);
    float* d_input = static_cast<float*>(strata::gpu::alloc_device(input.size() * sizeof(float)));
    auto stream = strata::gpu::stream_create();
    if (arena == nullptr || d_input == nullptr || stream == nullptr || !strata::gpu::copy(d_input, input.data(), input.size() * sizeof(float))) {
        std::cerr << "shared buffers: " << strata::gpu::last_error() << '\n';
        if (stream != nullptr) strata::gpu::stream_destroy(stream);
        if (d_input != nullptr) strata::gpu::free(d_input);
        if (arena != nullptr) strata::gpu::free(arena);
        return 1;
    }
    for (auto& owner : owners)
        if (!owner.prepare_stream(stream, err)) {
            std::cerr << "shared prepare stream: " << err << '\n';
            strata::gpu::stream_destroy(stream); strata::gpu::free(d_input); strata::gpu::free(arena);
            return 1;
        }
    strata::core::MoEBuffers buffers;
    strata::core::moe_buffers_init(geometry, 1, arena, buffers);
    std::vector<float> expected(n);
    std::vector<float> gate(n), up(n), hidden(n);
    auto rounded = [](float value) {
        return strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(value));
    };
    for (int64_t row = 0; row < n; ++row) {
        double gate_sum = 0.0, up_sum = 0.0;
        for (int64_t col = 0; col < n; ++col) {
            gate_sum += static_cast<double>(input[static_cast<size_t>(col)]) *
                        rounded(weights[1][static_cast<size_t>(row * n + col)]);
            up_sum += static_cast<double>(input[static_cast<size_t>(col)]) *
                      rounded(weights[2][static_cast<size_t>(row * n + col)]);
        }
        gate[static_cast<size_t>(row)] = static_cast<float>(gate_sum);
        up[static_cast<size_t>(row)] = static_cast<float>(up_sum);
        hidden[static_cast<size_t>(row)] = gate[static_cast<size_t>(row)] /
                                            (1.0f + std::exp(-gate[static_cast<size_t>(row)])) *
                                            up[static_cast<size_t>(row)];
    }
    double scalar = 0.0;
    for (int64_t col = 0; col < n; ++col)
        scalar += static_cast<double>(input[static_cast<size_t>(col)]) * rounded(weights[0][static_cast<size_t>(col)]);
    const float scalar_scale = 1.0f / (1.0f + std::exp(-static_cast<float>(scalar)));
    for (int64_t row = 0; row < n; ++row) {
        double down_sum = 0.0;
        for (int64_t col = 0; col < n; ++col)
            down_sum += static_cast<double>(hidden[static_cast<size_t>(col)]) *
                        rounded(weights[3][static_cast<size_t>(row * n + col)]);
        expected[static_cast<size_t>(row)] = static_cast<float>(down_sum) * scalar_scale;
    }
    int failures = 0;
    auto check = [&](const char* label) {
        std::vector<float> actual(n);
        if (!strata::gpu::copy(actual.data(), buffers.shared, actual.size() * sizeof(float))) return false;
        bool ok = true;
        for (size_t i = 0; i < actual.size(); ++i) ok = ok && close(actual[i], expected[i]);
        std::cout << label << " public moe_shared owner parity: " << (ok ? "OK" : "FAIL") << '\n';
        return ok;
    };
    if (!strata::core::moe_shared(table, geometry, 0, buffers, d_input, stream, err) ||
        !strata::gpu::stream_sync(stream)) {
        std::cerr << "direct moe_shared: " << (err.empty() ? strata::gpu::last_error() : err) << '\n';
        ++failures;
    } else if (!check("direct")) ++failures;
    strata::gpu::Graph* graph = nullptr;
    if (!strata::gpu::begin_capture(stream) || !strata::core::moe_shared(table, geometry, 0, buffers, d_input, stream, err) ||
        !strata::gpu::end_capture(stream, &graph) || !strata::gpu::graph_launch(graph, stream) ||
        !strata::gpu::stream_sync(stream)) {
        std::cerr << "graph moe_shared: " << (err.empty() ? strata::gpu::last_error() : err) << '\n';
        if (strata::gpu::is_capturing(stream)) strata::gpu::abandon_capture(stream);
        ++failures;
    } else if (!check("graph")) ++failures;
    if (graph != nullptr) strata::gpu::graph_destroy(graph);
    strata::gpu::stream_destroy(stream);
    strata::gpu::free(d_input);
    strata::gpu::free(arena);
    return failures;
}

std::filesystem::path make_hc_archive(const std::filesystem::path& root, std::vector<float>& norm,
                                      std::vector<float>& down, std::vector<float>& up,
                                      std::vector<float>& inject, std::vector<float>& residual) {
    constexpr int64_t n = 128, hc = 2, lr = 32, dim = n * hc;
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json") << R"({"model_type":"synthetic-exl3-hc"})";
    norm.resize(static_cast<size_t>(dim));
    down.resize(static_cast<size_t>(lr * dim));
    up.resize(static_cast<size_t>(dim * lr));
    inject.resize(static_cast<size_t>(hc * dim));
    residual.resize(static_cast<size_t>(dim));
    for (int64_t i = 0; i < dim; ++i) {
        norm[static_cast<size_t>(i)] = 0.91f + 0.003f * static_cast<float>(i % 17);
        residual[static_cast<size_t>(i)] = 0.021f * static_cast<float>((i * 3) % 41 - 20);
    }
    for (int64_t row = 0; row < lr; ++row)
        for (int64_t col = 0; col < dim; ++col)
            down[static_cast<size_t>(row * dim + col)] = 0.0021f * static_cast<float>((row * 5 + col * 7) % 29 - 14);
    for (int64_t row = 0; row < dim; ++row)
        for (int64_t col = 0; col < lr; ++col)
            up[static_cast<size_t>(row * lr + col)] = 0.0027f * static_cast<float>((row * 11 + col * 3) % 31 - 15);
    for (int64_t row = 0; row < hc; ++row)
        for (int64_t col = 0; col < dim; ++col)
            inject[static_cast<size_t>(row * dim + col)] = 0.0019f * static_cast<float>((row * 13 + col * 5) % 23 - 11);
    write_safetensors(root / "model.safetensors",
                      {{"hc_norm.weight", "F16", {static_cast<uint64_t>(dim)}, f16_bytes(norm)},
                       {"hc_down.weight", "F16", {static_cast<uint64_t>(lr), static_cast<uint64_t>(dim)}, f16_bytes(down)},
                       {"hc_up.weight", "F16", {static_cast<uint64_t>(dim), static_cast<uint64_t>(lr)}, f16_bytes(up)},
                       {"hc_inject.weight", "F16", {static_cast<uint64_t>(hc), static_cast<uint64_t>(dim)}, f16_bytes(inject)}});
    const uint64_t norm_bytes = static_cast<uint64_t>(dim) * sizeof(float);
    const uint64_t down_bytes = static_cast<uint64_t>(lr * dim) * sizeof(float);
    const uint64_t up_bytes = static_cast<uint64_t>(dim * lr) * sizeof(float);
    const uint64_t inject_bytes = static_cast<uint64_t>(hc * dim) * sizeof(float);
    const uint64_t total = norm_bytes + down_bytes + up_bytes + inject_bytes;
    std::ofstream dense(root / "dense.bin", std::ios::binary);
    dense.write(reinterpret_cast<const char*>(norm.data()), static_cast<std::streamsize>(norm_bytes));
    std::vector<uint8_t> padding(static_cast<size_t>(total - norm_bytes), 0);
    dense.write(reinterpret_cast<const char*>(padding.data()), static_cast<std::streamsize>(padding.size()));
    std::ofstream index(root / "index.txt");
    index << "# strata synthetic EXL3 HC\n# align 256 pool " << total << " tensors 4\n";
    const char* names[] = {"hc.norm", "hc.down", "hc.up", "hc.inject"};
    const uint64_t bytes[] = {norm_bytes, down_bytes, up_bytes, inject_bytes};
    const int64_t ne0[] = {dim, dim, lr, dim};
    const int64_t ne1[] = {1, lr, dim, hc};
    uint64_t offset = 0;
    for (int i = 0; i < 4; ++i) {
        index << names[i] << " 0 2 " << offset << ' ' << bytes[i] << ' ' << offset << ' ' << bytes[i] << ' '
              << ne0[i] << ' ' << ne1[i] << " 0 0 0 0 0 0 0 0 0 0\n";
        offset += bytes[i];
    }
    return root;
}

int run_hc_owner_case(const std::filesystem::path& root) {
    constexpr int64_t n = 128, hc = 2, lr = 32, dim = n * hc;
    std::vector<float> norm, down, up, inject, residual;
    make_hc_archive(root, norm, down, up, inject, residual);
    strata::core::WeightTable table;
    std::string err;
    const std::set<std::string> skip{"hc.down", "hc.up", "hc.inject"};
    uint64_t pool_bytes = 0;
    if (!strata::core::WeightTable::pool_bytes(root.string(), pool_bytes, err, &skip) || pool_bytes == 0) {
        std::cerr << "HC table setup: " << err << '\n';
        return 1;
    }
    void* norm_arena = strata::gpu::alloc_device(pool_bytes);
    if (norm_arena == nullptr || !table.load(root.string(), norm_arena, pool_bytes, err, &skip)) {
        std::cerr << "HC table setup: " << (err.empty() ? strata::gpu::last_error() : err) << '\n';
        if (norm_arena != nullptr) strata::gpu::free(norm_arena);
        return 1;
    }
    strata::core::Exl3Dense down_owner, up_owner, inject_owner;
    if (!down_owner.load(root, "hc_down", "hc.down", table, err, 1) ||
        !up_owner.load(root, "hc_up", "hc.up", table, err, 1) ||
        !inject_owner.load(root, "hc_inject", "hc.inject", table, err, 1)) {
        std::cerr << "HC owner load: " << err << '\n';
        strata::gpu::free(norm_arena);
        return 1;
    }
    const auto* norm_ref = table.find("hc.norm");
    const auto* down_ref = table.find("hc.down");
    const auto* up_ref = table.find("hc.up");
    const auto* inject_ref = table.find("hc.inject");
    if (norm_ref == nullptr || down_ref == nullptr || up_ref == nullptr || inject_ref == nullptr ||
        down_ref->data != nullptr || up_ref->data != nullptr || inject_ref->data != nullptr) {
        strata::gpu::free(norm_arena);
        return 1;
    }
    auto stream = strata::gpu::stream_create();
    void* residual_dev = strata::gpu::alloc_device(residual.size() * sizeof(float));
    void* scratch_dev = strata::gpu::alloc_device(strata::kernels::gr_workspace_bytes({n, hc, lr}));
    float* mixed_dev = static_cast<float*>(strata::gpu::alloc_device(n * sizeof(float)));
    float* inject_dev = static_cast<float*>(strata::gpu::alloc_device(hc * sizeof(float)));
    if (stream == nullptr || residual_dev == nullptr || scratch_dev == nullptr || mixed_dev == nullptr || inject_dev == nullptr ||
        !strata::gpu::copy(residual_dev, residual.data(), residual.size() * sizeof(float))) {
        if (stream != nullptr) strata::gpu::stream_destroy(stream);
        if (residual_dev != nullptr) strata::gpu::free(residual_dev);
        if (scratch_dev != nullptr) strata::gpu::free(scratch_dev);
        if (mixed_dev != nullptr) strata::gpu::free(mixed_dev);
        if (inject_dev != nullptr) strata::gpu::free(inject_dev);
        strata::gpu::free(norm_arena);
        return 1;
    }
    if (!down_owner.prepare_stream(stream, err) || !up_owner.prepare_stream(stream, err) ||
        !inject_owner.prepare_stream(stream, err)) {
        std::cerr << "HC stream: " << err << '\n';
        strata::gpu::free(norm_arena);
        return 1;
    }
    strata::kernels::GrShapes shapes{n, hc, lr};
    strata::kernels::GrWorkspace workspace;
    strata::kernels::gr_workspace_init(shapes, scratch_dev, workspace);
    const float* d_norm = static_cast<const float*>(norm_ref->data);
    const float* d_down = static_cast<const float*>(residual_dev); // overwritten below only for type clarity
    (void) d_norm; (void) d_down;
    auto expected = [&](bool with_inject, std::vector<float>& mixed, std::vector<float>& inj) {
        std::vector<float> xn(static_cast<size_t>(dim)), lo(static_cast<size_t>(lr));
        for (int64_t c = 0; c < hc; ++c) {
            double sum = 0.0;
            for (int64_t d = 0; d < n; ++d) {
                const float x = residual[static_cast<size_t>(c * n + d)];
                sum += static_cast<double>(x) * x;
            }
            const float scale = 1.0f / std::sqrt(static_cast<float>(sum / n) + 1e-6f);
            for (int64_t d = 0; d < n; ++d)
                xn[static_cast<size_t>(c * n + d)] = scale * residual[static_cast<size_t>(c * n + d)] * norm[static_cast<size_t>(c * n + d)];
        }
        for (int64_t row = 0; row < lr; ++row) {
            double sum = 0.0;
            for (int64_t col = 0; col < dim; ++col)
                sum += static_cast<double>(xn[static_cast<size_t>(col)]) *
                       strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(down[static_cast<size_t>(row * dim + col)]));
            const float x = static_cast<float>(sum / hc);
            lo[static_cast<size_t>(row)] = x / (1.0f + std::exp(-x));
        }
        std::vector<float> gated(static_cast<size_t>(dim));
        for (int64_t row = 0; row < dim; ++row) {
            double sum = 0.0;
            for (int64_t col = 0; col < lr; ++col)
                sum += static_cast<double>(lo[static_cast<size_t>(col)]) *
                       strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(up[static_cast<size_t>(row * lr + col)]));
            gated[static_cast<size_t>(row)] = xn[static_cast<size_t>(row)] /
                                               (1.0f + std::exp(-static_cast<float>(sum)));
        }
        mixed.assign(static_cast<size_t>(n), 0.0f);
        for (int64_t d = 0; d < n; ++d)
            mixed[static_cast<size_t>(d)] = (gated[static_cast<size_t>(d)] + gated[static_cast<size_t>(n + d)]) / hc;
        inj.assign(static_cast<size_t>(hc), 0.0f);
        if (with_inject)
            for (int64_t row = 0; row < hc; ++row)
                for (int64_t col = 0; col < dim; ++col)
                    inj[static_cast<size_t>(row)] += xn[static_cast<size_t>(col)] *
                                                       strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(inject[static_cast<size_t>(row * dim + col)]));
    };
    std::vector<float> expected_mixed, expected_inject;
    expected(true, expected_mixed, expected_inject);
    int failures = 0;
    auto check = [&](const char* label, bool with_inject) {
        std::vector<float> mixed(n), got_inject(hc);
        if (!strata::gpu::copy(mixed.data(), mixed_dev, mixed.size() * sizeof(float))) return false;
        if (with_inject && !strata::gpu::copy(got_inject.data(), inject_dev, got_inject.size() * sizeof(float))) return false;
        bool ok = true;
        for (size_t i = 0; i < mixed.size(); ++i) ok = ok && close(mixed[i], expected_mixed[i]);
        if (with_inject) for (size_t i = 0; i < got_inject.size(); ++i) ok = ok && close(got_inject[i], expected_inject[i]);
        std::cout << label << " public HC owner parity: " << (ok ? "OK" : "FAIL") << '\n';
        return ok;
    };
    if (!strata::core::gr_read_exl3(*norm_ref, *down_ref, *up_ref, inject_ref, static_cast<const float*>(residual_dev),
                                   shapes, workspace, mixed_dev, inject_dev, stream, err) ||
        !strata::gpu::stream_sync(stream)) {
        std::cerr << "HC direct: " << (err.empty() ? strata::gpu::last_error() : err) << '\n'; ++failures;
    } else if (!check("direct", true)) ++failures;
    strata::gpu::Graph* graph = nullptr;
    if (!strata::gpu::begin_capture(stream) ||
        !strata::core::gr_read_exl3(*norm_ref, *down_ref, *up_ref, nullptr, static_cast<const float*>(residual_dev),
                                    shapes, workspace, mixed_dev, nullptr, stream, err) ||
        !strata::gpu::end_capture(stream, &graph) || !strata::gpu::graph_launch(graph, stream) ||
        !strata::gpu::stream_sync(stream)) {
        std::cerr << "HC graph: " << (err.empty() ? strata::gpu::last_error() : err) << '\n';
        if (strata::gpu::is_capturing(stream)) strata::gpu::abandon_capture(stream); ++failures;
    } else {
        expected(false, expected_mixed, expected_inject);
        if (!check("graph", false)) ++failures;
    }
    if (graph != nullptr) strata::gpu::graph_destroy(graph);
    strata::gpu::stream_destroy(stream); strata::gpu::free(residual_dev); strata::gpu::free(scratch_dev);
    strata::gpu::free(mixed_dev); strata::gpu::free(inject_dev);
    strata::gpu::free(norm_arena);
    return failures;
}

}  // namespace

int main() {
    constexpr int64_t n_in = 128, n_expert = 2, top_k = 1;
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = std::filesystem::temp_directory_path() /
                      ("xestrata_exl3_projection_parity_" + std::to_string(STRATA_EXL3_PROJECTION_PID) +
                       "_" + std::to_string(stamp));
    std::filesystem::remove_all(root);
    std::vector<float> weights, input;
    make_router_archive(root, weights, input);

    strata::core::WeightTable table;
    std::string err;
    const std::set<std::string> skip{"blk.0.ffn_gate_inp.weight"};
    uint64_t pool_bytes = 0;
    if (!strata::core::WeightTable::pool_bytes(root.string(), pool_bytes, err, &skip) ||
        !table.load(root.string(), nullptr, 0, err, &skip)) {
        std::cerr << "table setup: " << err << '\n';
        std::filesystem::remove_all(root);
        return 1;
    }
    const auto* canonical = table.find("blk.0.ffn_gate_inp.weight");
    if (canonical == nullptr || canonical->data != nullptr) {
        std::cerr << "canonical row was not kept as null metadata\n";
        std::filesystem::remove_all(root);
        return 1;
    }

    strata::core::Exl3Dense owner;
    if (!owner.load(root, "blk.0.ffn_gate_inp", "blk.0.ffn_gate_inp.weight", table, err, 1)) {
        std::cerr << "raw router owner: " << err << '\n';
        std::filesystem::remove_all(root);
        return 1;
    }
    strata::core::ModelGeometry geometry;
    geometry.n_embd = n_in;
    geometry.n_expert = n_expert;
    geometry.n_layers = 1;
    auto stream = strata::gpu::stream_create();
    if (stream == nullptr || !owner.prepare_stream(stream, err)) {
        std::cerr << "router stream: " << (err.empty() ? strata::gpu::last_error() : err) << '\n';
        if (stream != nullptr) strata::gpu::stream_destroy(stream);
        std::filesystem::remove_all(root);
        return 1;
    }

    const uint64_t bytes = strata::core::moe_buffers_bytes(geometry, top_k);
    void* arena = strata::gpu::alloc_device(bytes);
    float* d_input = static_cast<float*>(strata::gpu::alloc_device(input.size() * sizeof(float)));
    if (arena == nullptr || d_input == nullptr || !strata::gpu::copy(d_input, input.data(), input.size() * sizeof(float))) {
        std::cerr << "router buffers: " << strata::gpu::last_error() << '\n';
        if (d_input != nullptr) strata::gpu::free(d_input);
        if (arena != nullptr) strata::gpu::free(arena);
        strata::gpu::stream_destroy(stream);
        std::filesystem::remove_all(root);
        return 1;
    }
    strata::core::MoEBuffers buffers;
    strata::core::moe_buffers_init(geometry, top_k, arena, buffers);

    std::vector<float> expected_logits(n_expert, 0.0f);
    for (int64_t row = 0; row < n_expert; ++row) {
        double sum = 0.0;
        for (int64_t col = 0; col < n_in; ++col) {
            const float w = strata::kernels::f32_from_f16(
                strata::kernels::f16_from_f32(weights[static_cast<size_t>(row * n_in + col)]));
            sum += static_cast<double>(input[static_cast<size_t>(col)]) * w;
        }
        expected_logits[static_cast<size_t>(row)] = static_cast<float>(sum);
    }
    const int expected_id = expected_logits[0] >= expected_logits[1] ? 0 : 1;
    int failures = 0;
    auto check_route = [&](const char* label) {
        std::vector<float> got_logits(n_expert);
        int got_id = -1;
        float got_weight = 0.0f;
        if (!strata::gpu::copy(got_logits.data(), buffers.logits, got_logits.size() * sizeof(float)) ||
            !strata::gpu::copy(&got_id, buffers.ids, sizeof got_id) ||
            !strata::gpu::copy(&got_weight, buffers.weights, sizeof got_weight)) {
            std::cerr << label << " readback: " << strata::gpu::last_error() << '\n';
            return false;
        }
        bool ok = got_id == expected_id && close(got_weight, 1.0f);
        for (size_t i = 0; i < got_logits.size(); ++i) ok = ok && close(got_logits[i], expected_logits[i]);
        std::cout << label << " public moe_route owner parity: " << (ok ? "OK" : "FAIL") << '\n';
        return ok;
    };

    if (!strata::core::moe_route(table, geometry, 0, top_k, buffers, d_input, stream, err, nullptr) ||
        !strata::gpu::stream_sync(stream)) {
        std::cerr << "direct moe_route: " << (err.empty() ? strata::gpu::last_error() : err) << '\n';
        ++failures;
    } else if (!check_route("direct")) {
        ++failures;
    }

    strata::gpu::Graph* graph = nullptr;
    if (!strata::gpu::begin_capture(stream) ||
        !strata::core::moe_route(table, geometry, 0, top_k, buffers, d_input, stream, err, nullptr) ||
        !strata::gpu::end_capture(stream, &graph) || !strata::gpu::graph_launch(graph, stream) ||
        !strata::gpu::stream_sync(stream)) {
        std::cerr << "graph moe_route: " << (err.empty() ? strata::gpu::last_error() : err) << '\n';
        if (strata::gpu::is_capturing(stream)) strata::gpu::abandon_capture(stream);
        ++failures;
    } else if (!check_route("graph")) {
        ++failures;
    }
    if (graph != nullptr) strata::gpu::graph_destroy(graph);
    strata::gpu::free(d_input);
    strata::gpu::free(arena);
    strata::gpu::stream_destroy(stream);
    failures += run_shared_owner_case(root / "shared");
    failures += run_hc_owner_case(root / "hc");
    std::filesystem::remove_all(root);
    std::cout << "exl3_projection_parity: " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
