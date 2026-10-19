// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Public EXL3 ExpertDispatch/MoeWindow graph-capture and replay selftest.
#include "strata/artifact/exl3_expert.hpp"
#include "strata/core/exl3_expert_runtime.hpp"
#include "strata/core/moe_window.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/exl3_expert.hpp"
#include "../kernels/parity_device.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#if defined(_WIN32)
#include <process.h>
#define STRATA_EXL3_DISPATCH_PID _getpid()
#else
#include <unistd.h>
#define STRATA_EXL3_DISPATCH_PID getpid()
#endif

namespace {
using Json = nlohmann::json;
using Tensor = std::tuple<std::string, std::string, std::vector<uint64_t>, std::vector<uint8_t>>;

std::vector<uint8_t> words(size_t n, uint16_t seed) {
    std::vector<uint8_t> out(n * 2);
    for (size_t i = 0; i < n; ++i) {
        const uint16_t v = static_cast<uint16_t>(seed + i * 7);
        out[2 * i] = static_cast<uint8_t>(v);
        out[2 * i + 1] = static_cast<uint8_t>(v >> 8);
    }
    return out;
}
std::vector<uint8_t> u32(uint32_t v) {
    return {static_cast<uint8_t>(v), static_cast<uint8_t>(v >> 8), static_cast<uint8_t>(v >> 16),
            static_cast<uint8_t>(v >> 24)};
}
void add_linear(std::vector<Tensor>& ts, const std::string& name, int bits, int cb) {
    ts.emplace_back(name + ".trellis", "I16", std::vector<uint64_t>{8, 8, static_cast<uint64_t>(16 * bits)},
                    words(static_cast<size_t>(8 * 8 * 16 * bits), static_cast<uint16_t>(bits * 19)));
    ts.emplace_back(name + ".suh", "F16", std::vector<uint64_t>{128}, words(128, 0x3c00));
    ts.emplace_back(name + ".svh", "F16", std::vector<uint64_t>{128}, words(128, 0x3c00));
    if (cb == 1) ts.emplace_back(name + ".mcg", "U32", std::vector<uint64_t>{}, u32(0xCBAC1FEDu));
    if (cb == 2) ts.emplace_back(name + ".mul1", "I32", std::vector<uint64_t>{1}, u32(0x83DCD12Du));
}
void write_archive(const std::filesystem::path& root, std::vector<strata::artifact::exl3::ExpertTensorNames>& names) {
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json") << R"({"model_type":"synthetic-exl3-dispatch"})";
    std::vector<std::vector<Tensor>> shards(3);
    for (int layer = 0; layer < 2; ++layer)
        for (int expert = 0; expert < 2; ++expert) {
            const std::string base = "layers." + std::to_string(layer) + ".experts." + std::to_string(expert);
            names.push_back({base + ".gate", base + ".up", base + ".down"});
            const int bits = 2 + ((layer * 2 + expert) % 4);
            auto& shard = shards[expert == 0 ? layer : 2];
            add_linear(shard, base + ".gate", bits, layer % 3);
            add_linear(shard, base + ".up", 2 + ((bits + 1) % 4), (layer + expert + 1) % 3);
            add_linear(shard, base + ".down", 2 + ((bits + 2) % 4), (layer + expert + 2) % 3);
        }
    Json index; index["weight_map"] = Json::object();
    for (size_t si = 0; si < shards.size(); ++si) {
        const std::string file = "model-0000" + std::to_string(si + 1) + "-of-00003.safetensors";
        Json header = Json::object(); uint64_t off = 0;
        for (const auto& [name, dtype, shape, data] : shards[si]) {
            header[name] = Json{{"dtype", dtype}, {"shape", shape}, {"data_offsets", {off, off + data.size()}}};
            index["weight_map"][name] = file; off += data.size();
        }
        const std::string text = header.dump(); std::ofstream out(root / file, std::ios::binary);
        for (int i = 0; i < 8; ++i) out.put(static_cast<char>(text.size() >> (8 * i)));
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        for (const auto& [name, dtype, shape, data] : shards[si]) {
            (void) name; (void) dtype; (void) shape;
            out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        }
    }
    std::ofstream(root / "model.safetensors.index.json") << index.dump();
}

struct DispatchContext { strata::core::ExpertDispatch dispatch; };
void pool_fn(void* user, const float* x, const int32_t* ids, int64_t nt, int64_t k, float* out, int64_t layer) {
    auto& c = *static_cast<DispatchContext*>(user); c.dispatch.layers = layer;
    strata::core::expert_pool_dispatch_multi(c.dispatch, x, ids, nt, k, out);
}
bool close_enough(float a, float b) { return std::isfinite(a) && std::isfinite(b) && std::fabs(a - b) <= 2e-3f * std::max(1.0f, std::fabs(b)); }
bool compare_parts(const std::vector<float>& a, const std::vector<float>& b) {
    for (size_t i = 0; i < a.size(); ++i) if (!close_enough(a[i], b[i])) return false;
    return true;
}
bool capture(strata::gpu::Stream s, const std::function<void()>& body, strata::gpu::Graph*& graph, std::string& err) {
    if (!strata::gpu::begin_capture(s)) { err = strata::gpu::last_error(); return false; }
    body();
    if (!strata::gpu::end_capture(s, &graph) || graph == nullptr) { err = strata::gpu::last_error(); return false; }
    return true;
}

}  // namespace

int main() {
    const auto root = std::filesystem::temp_directory_path() /
                      ("xestrata_exl3_dispatch_test_" + std::to_string(static_cast<unsigned long long>(STRATA_EXL3_DISPATCH_PID)));
    std::filesystem::remove_all(root);
    int failures = 0;
    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> moved;
    try {
        std::vector<strata::artifact::exl3::ExpertTensorNames> names; write_archive(root, names);
        strata::parity::Device device;
        auto archive = std::make_shared<strata::artifact::exl3::Archive>(root);
        strata::core::Exl3ExpertSource source(archive, 2, 2, names);
        strata::kernels::cpu::ExpertPool pool(1, false, false);
        strata::core::Exl3ExpertRuntime runtime; std::string err;
        if (!runtime.open(source, pool, 2, 2, {{0, 0}, {1, 0}}, err)) throw std::runtime_error(err);
        strata::core::MoeWindowShape shape{128, 2, 2, 2, 1};
        strata::core::MoeWindowTier tier; tier.exl3_runtime = &runtime;
        strata::core::MoeWindow window;
#if defined(_WIN32)
        _putenv_s("STRATA_VERIFY_SEGMENTED", "1");
#else
        setenv("STRATA_VERIFY_SEGMENTED", "1", 1);
#endif
        if (!window.init(shape, tier, err)) throw std::runtime_error(err);
        std::vector<float> x(2 * 128, -777.0f), weights(4, 1.0f);
        for (int r = 0; r < 2; ++r) for (int j = 0; j < 128; ++j)
            x[static_cast<size_t>(r * 128 + j)] = 0.013f * static_cast<float>(((r + 3) * 17 + j * 11) % 101 - 50);
        std::vector<int32_t> ids{0, 1, 1, 0};
        float* dx = device.upload(x); int32_t* dids = device.upload(ids); float* dw = device.upload(weights);
        std::vector<std::vector<uint8_t>> blobs(4); std::vector<strata::kernels::exl3::ExpertView> views;
        for (int layer = 0; layer < 2; ++layer) for (int e = 0; e < 2; ++e) {
            const size_t at = static_cast<size_t>(layer * 2 + e);
            const auto* layout = source.exl3_layout(layer, e); blobs[at].resize(layout->total_bytes);
            if (!source.copy_blob(layer, e, blobs[at].data())) throw std::runtime_error("copy blob");
            views.push_back(strata::kernels::exl3::expert_view(*layout, blobs[at].data()));
        }
        for (const char* file : {"model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors"}) {
            auto original = root / file, hidden = root / (std::string(file) + ".resident-offline");
            std::filesystem::rename(original, hidden); moved.emplace_back(original, hidden);
        }
        auto stream = strata::gpu::stream_create(); if (stream == nullptr) throw std::runtime_error("stream");
        float* first_parts = device.alloc<float>(4 * 128);
        strata::gpu::Graph* gs[3] = {nullptr, nullptr, nullptr};
        if (!capture(stream, [&] { window.publish(1, 0, 0, 2, dx, dids, dw, stream); }, gs[0], err) ||
            !capture(stream, [&] {
                if (!window.experts(0, 1, 0, 0, 2, stream, err) ||
                    !strata::gpu::copy_async(first_parts, window.parts(), 4 * 128 * sizeof(float), stream))
                    throw std::runtime_error(err);
                window.publish(2, 0, 0, 2, dx, dids, dw, stream);
            }, gs[1], err) ||
            !capture(stream, [&] { if (!window.experts(1, 2, 0, 0, 2, stream, err)) throw std::runtime_error(err); }, gs[2], err))
            throw std::runtime_error(err);
        std::vector<strata::gpu::Graph*> segments{gs[0], gs[1], gs[2]}; window.set_segments(&segments);
        DispatchContext context; context.dispatch.pool = &pool; context.dispatch.src = &source;
        context.dispatch.exl3_runtime = &runtime; context.dispatch.n_expert = 2; context.dispatch.plan = window.sink();
        const int64_t layers[] = {0, 1}; const int groups[] = {0, 0}; const int tbs[] = {0, 0}; const int ns[] = {2, 2};
        for (int replay = 0; replay < 2; ++replay) {
            if (replay == 1) ids = {1, 1, 0, 1};
            strata::parity::copy_in(dids, ids.data(), ids.size());
            window.reset();
            if (!strata::gpu::graph_launch(gs[0], stream) ||
                !window.serve(2, layers, groups, tbs, ns, &pool_fn, &context, stream, err))
                throw std::runtime_error(err.empty() ? strata::gpu::last_error() : err);
            if (context.dispatch.failed) throw std::runtime_error(context.dispatch.fail);
            for (int layer = 0; layer < 2; ++layer) {
                std::vector<float> got(4 * 128), expected(got.size(), 24680.5f);
                strata::parity::copy_out(got.data(), layer == 0 ? first_parts : window.parts(), got.size());
                for (int i = 0; i < 4; ++i)
                    if (strata::kernels::cpu::exl3::expert(views[static_cast<size_t>(layer * 2 + ids[i])],
                                                           x.data() + (i / 2) * 128, 128,
                                                           expected.data() + i * 128, 128, 1) !=
                        strata::kernels::cpu::exl3::Status::ok)
                        throw std::runtime_error("expected expert");
                if (!compare_parts(got, expected)) {
                    std::fprintf(stderr, "route rows differ at replay %d layer %d\n", replay, layer);
                    ++failures;
                }
            }
        }
        if (context.dispatch.cache_hits != 6 || context.dispatch.cache_refused != 10 ||
            context.dispatch.multi_entries != 16 || context.dispatch.multi_misses != 4) {
            std::fprintf(stderr, "incorrect hit/miss counters\n");
            ++failures;
        }
        for (auto* graph : segments) if (graph != nullptr) strata::gpu::graph_destroy(graph);
        strata::gpu::stream_destroy(stream);
        for (auto it = moved.rbegin(); it != moved.rend(); ++it) {
            std::error_code ec; if (std::filesystem::exists(it->second)) std::filesystem::rename(it->second, it->first, ec);
        }
        moved.clear();
        DispatchContext bad; bad.dispatch.pool = &pool; bad.dispatch.src = &source; bad.dispatch.exl3_runtime = &runtime;
        bad.dispatch.n_expert = 2; std::vector<float> stale(4 * 128, 12345.0f);
        strata::core::expert_pool_dispatch_multi(bad.dispatch, x.data(), ids.data(), 2, 2, stale.data());
        if (!bad.dispatch.failed || stale[0] != 12345.0f) ++failures;
    } catch (const std::exception& e) { std::fprintf(stderr, "exl3_expert_dispatch_test: %s\n", e.what()); ++failures; }
    for (auto it = moved.rbegin(); it != moved.rend(); ++it) {
        std::error_code ec; if (std::filesystem::exists(it->second)) std::filesystem::rename(it->second, it->first, ec);
    }
    std::filesystem::remove_all(root);
    std::printf("exl3_expert_dispatch_test: %s\n", failures == 0 ? "all passed" : "failures");
    return failures == 0 ? 0 : 1;
}
