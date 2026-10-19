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

struct DispatchContext {
    strata::core::ExpertDispatch dispatch;
    int calls = 0;
    int fail_call = -1;
    bool throw_after_publish = false;
};
void pool_fn(void* user, const float* x, const int32_t* ids, int64_t nt, int64_t k, float* out, int64_t layer) {
    auto& c = *static_cast<DispatchContext*>(user); c.dispatch.layers = layer;
    if (c.calls++ == c.fail_call) {
        // Fail after publishing metadata, with dirty CPU output and prepared_ still set: this is the ordering a
        // failed CPU job has. The graph must see an empty plan, discard all rows and answer the remaining rings.
        std::string err;
        auto& runtime = *c.dispatch.exl3_runtime;
        if (!runtime.prepare_window(layer, x, 128, ids, nt, out, 128, err) ||
            !runtime.publish_window_plan(*c.dispatch.plan, err))
            throw std::runtime_error(err);
        std::fill(out, out + nt * k * 128, 98765.0f);
        if (c.throw_after_publish) throw std::runtime_error("injected CPU callback failure after plan publish");
        c.dispatch.failed = true;
        c.dispatch.fail = "injected CPU failure after plan publish";
        return;
    }
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

struct EnvSetting {
    std::string name, old;
    bool had_old;
    static void set(const char* name, const char* value) {
#if defined(_WIN32)
        _putenv_s(name, value == nullptr ? "" : value);
#else
        if (value == nullptr) unsetenv(name); else setenv(name, value, 1);
#endif
    }
    EnvSetting(const char* key, const char* value) : name(key), had_old(std::getenv(key) != nullptr) {
        if (had_old) old = std::getenv(key);
        set(name.c_str(), value);
    }
    ~EnvSetting() { set(name.c_str(), had_old ? old.c_str() : nullptr); }
};

struct CapturedWindow {
    strata::gpu::Stream stream = strata::gpu::stream_create();
    std::vector<strata::gpu::Graph*> graphs;
    ~CapturedWindow() {
        for (auto* graph : graphs) if (graph != nullptr) strata::gpu::graph_destroy(graph);
        if (stream != nullptr) strata::gpu::stream_destroy(stream);
    }
};

void device_plan_case(strata::parity::Device& device, strata::core::Exl3ExpertSource& source,
                      strata::kernels::cpu::ExpertPool& pool,
                      const std::vector<strata::kernels::exl3::ExpertView>& views,
                      int capacity, int nt, bool failures) {
    EnvSetting opt_in("STRATA_EXL3_DEVICE_HIT_PLAN", "1");
    EnvSetting segments("STRATA_VERIFY_SEGMENTED", nullptr);
    strata::core::Exl3ExpertRuntime runtime;
    std::string err;
    if (!runtime.open(source, pool, capacity, 2, {{0, 0}, {1, 0}}, err)) throw std::runtime_error(err);
    strata::core::MoeWindow window;
    strata::core::MoeWindowTier tier; tier.exl3_runtime = &runtime;
    if (!window.init({128, 2, 2, capacity, 1}, tier, err)) throw std::runtime_error(err);
    if (window.direct_resident() || (window.device_hit_plan() == window.segmented()))
        throw std::runtime_error("partial cache selected an inconsistent window mode");
    if (window.device_hit_plan() && window.sink()->before_cpu != nullptr)
        throw std::runtime_error("device hit plan would enqueue from the CPU callback");
    std::printf("device hit plan capacity=%d nt=%d: device_hit_plan=%d segmented=%d\n", capacity, nt,
                 static_cast<int>(window.device_hit_plan()), static_cast<int>(window.segmented()));

    std::vector<float> x(static_cast<size_t>(capacity * 128), -777.0f);
    std::vector<float> weights(static_cast<size_t>(capacity * 2), 0.25f);
    for (int t = 0; t < nt; ++t) for (int j = 0; j < 128; ++j)
        x[static_cast<size_t>(t * 128 + j)] = 0.013f * static_cast<float>(((t + 3) * 17 + j * 11) % 101 - 50);
    std::vector<int32_t> ids(static_cast<size_t>(capacity * 2), 0);
    float* dx = device.upload(x); int32_t* dids = device.upload(ids); float* dw = device.upload(weights);
    const std::vector<int64_t> layers{0, 1, 0};
    const std::vector<int> groups(layers.size(), 0), tbs(layers.size(), 0), ns(layers.size(), nt);
    const size_t row_elements = static_cast<size_t>(nt * 2 * 128);
    const size_t combined_elements = static_cast<size_t>(nt * 128);
    float* history = device.alloc<float>(layers.size() * row_elements);
    float* combined = device.alloc<float>(layers.size() * combined_elements);
    CapturedWindow captured;
    if (captured.stream == nullptr) throw std::runtime_error("stream");
    auto append = [&](const std::function<void()>& body) {
        strata::gpu::Graph* graph = nullptr;
        if (!capture(captured.stream, body, graph, err)) throw std::runtime_error(err);
        captured.graphs.push_back(graph);
    };
    auto publish = [&](size_t ring) {
        window.publish(static_cast<uint32_t>(ring + 1), 0, 0, nt, dx, dids, dw, captured.stream);
    };
    auto consume = [&](size_t ring) {
        if (!window.experts(layers[ring], static_cast<uint32_t>(ring + 1), 0, 0, nt, captured.stream, err) ||
            !strata::gpu::copy_async(history + ring * row_elements, window.parts(), row_elements * sizeof(float),
                                     captured.stream))
            throw std::runtime_error(err);
        // An actual parts reader after the merge and before the next doorbell exercises the combine lifetime.
        const float* parts = window.parts();
        float* sum = combined + ring * combined_elements;
        strata::core::Runtime::get().stream(captured.stream).parallel_for(
            sycl::range<1>(combined_elements), [=](sycl::id<1> index) {
                const size_t i = index[0], t = i / 128, col = i % 128;
                sum[i] = parts[(t * 2) * 128 + col] * dw[t * 2] +
                         parts[(t * 2 + 1) * 128 + col] * dw[t * 2 + 1];
            });
    };
    if (window.segmented()) {
        append([&] { publish(0); });
        for (size_t ring = 0; ring < layers.size(); ++ring)
            append([&] { consume(ring); if (ring + 1 < layers.size()) publish(ring + 1); });
        window.set_segments(&captured.graphs);
    } else {
        append([&] { for (size_t ring = 0; ring < layers.size(); ++ring) { publish(ring); consume(ring); } });
    }
    auto replay = [&](int route_pattern, int fail_call, bool throwing) {
        for (int i = 0; i < nt * 2; ++i)
            ids[static_cast<size_t>(i)] = route_pattern == 0 ? (i % 3 == 0 ? 0 : 1) :
                                          route_pattern == 1 ? 1 : (i == 0 ? 1 : 0);
        strata::parity::copy_in(dids, ids.data(), ids.size());
        DispatchContext context;
        context.dispatch.pool = &pool; context.dispatch.src = &source; context.dispatch.exl3_runtime = &runtime;
        context.dispatch.n_expert = 2; context.dispatch.plan = window.sink();
        context.fail_call = fail_call; context.throw_after_publish = throwing;
        window.reset();
        if (!strata::gpu::graph_launch(captured.graphs[0], captured.stream))
            throw std::runtime_error(strata::gpu::last_error());
        const bool served = window.serve(static_cast<int64_t>(layers.size()), layers.data(), groups.data(), tbs.data(),
                                         ns.data(), &pool_fn, &context, captured.stream, err);
        if (served != (fail_call < 0) || (!served && err.empty()) || (fail_call < 0 && context.dispatch.failed) ||
            context.calls != (fail_call < 0 ? static_cast<int>(layers.size()) : fail_call + 1) ||
            runtime.prepared() || runtime.cpu_ready() || runtime.pending_gpu())
            throw std::runtime_error("incorrect graph completion or CPU failure drain state: " + err);
        std::vector<float> got(layers.size() * row_elements), sums(layers.size() * combined_elements);
        strata::parity::copy_out(got.data(), history, got.size());
        strata::parity::copy_out(sums.data(), combined, sums.size());
        for (size_t ring = 0; ring < layers.size(); ++ring) {
            std::vector<float> expected(row_elements, 0.0f);
            if (fail_call < 0 || ring < static_cast<size_t>(fail_call))
                for (int i = 0; i < nt * 2; ++i)
                    if (strata::kernels::cpu::exl3::expert(views[static_cast<size_t>(layers[ring] * 2 + ids[i])],
                                                           x.data() + (i / 2) * 128, 128,
                                                           expected.data() + i * 128, 128, 1) !=
                        strata::kernels::cpu::exl3::Status::ok)
                        throw std::runtime_error("expected expert");
            for (size_t i = 0; i < row_elements; ++i)
                if (!close_enough(got[ring * row_elements + i], expected[i]))
                    throw std::runtime_error("device hit plan route rows differ");
            for (size_t i = 0; i < combined_elements; ++i) {
                const size_t t = i / 128, col = i % 128;
                const float expected_sum = (expected[t * 2 * 128 + col] + expected[(t * 2 + 1) * 128 + col]) * 0.25f;
                if (!close_enough(sums[ring * combined_elements + i], expected_sum))
                    throw std::runtime_error("device hit plan combine differs");
            }
        }
    };
    replay(0, -1, false);
    replay(1, -1, false); // Miss-only replay must not read stale GPU hit rows from the previous route pattern.
    if (failures && window.device_hit_plan()) {
        replay(0, 0, false);
        replay(2, -1, false); // An error must leave the same graph and persistent images reusable.
        replay(0, 1, true);
        replay(2, -1, false);
    }
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
        EnvSetting opt_in("STRATA_EXL3_DEVICE_HIT_PLAN", "1");
#if defined(_WIN32)
        _putenv_s("STRATA_VERIFY_SEGMENTED", "1");
#else
        setenv("STRATA_VERIFY_SEGMENTED", "1", 1);
#endif
        if (!window.init(shape, tier, err)) throw std::runtime_error(err);
        if (!window.segmented() || window.device_hit_plan() || window.direct_resident())
            throw std::runtime_error("explicit segmentation did not preserve the partial-cache fallback");
        std::printf("explicit segmented EXL3: device_hit_plan=%d segmented=%d\n",
                     static_cast<int>(window.device_hit_plan()), static_cast<int>(window.segmented()));
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
        DispatchContext segmented_failure;
        segmented_failure.dispatch.pool = &pool; segmented_failure.dispatch.src = &source;
        segmented_failure.dispatch.exl3_runtime = &runtime; segmented_failure.dispatch.n_expert = 2;
        segmented_failure.dispatch.plan = window.sink();
        segmented_failure.fail_call = 0; segmented_failure.throw_after_publish = true;
        window.reset();
        if (!strata::gpu::graph_launch(gs[0], stream) ||
            window.serve(2, layers, groups, tbs, ns, &pool_fn, &segmented_failure, stream, err) || err.empty() ||
            segmented_failure.calls != 1 || runtime.pending_gpu() || runtime.prepared() || runtime.cpu_ready())
            throw std::runtime_error("segmented CPU exception did not drain its pending GPU hit plan");
        context.calls = 0;
        window.reset();
        if (!strata::gpu::graph_launch(gs[0], stream) ||
            !window.serve(2, layers, groups, tbs, ns, &pool_fn, &context, stream, err) || context.dispatch.failed ||
            context.calls != 2 || runtime.pending_gpu() || runtime.prepared() || runtime.cpu_ready())
            throw std::runtime_error("segmented graph did not recover after a CPU exception");
        for (auto* graph : segments) if (graph != nullptr) strata::gpu::graph_destroy(graph);
        strata::gpu::stream_destroy(stream);
        for (auto it = moved.rbegin(); it != moved.rend(); ++it) {
            std::error_code ec; if (std::filesystem::exists(it->second)) std::filesystem::rename(it->second, it->first, ec);
        }
        moved.clear();
        device_plan_case(device, source, pool, views, 1, 1, false); // AR capacity and geometry.
        device_plan_case(device, source, pool, views, 2, 1, false); // AR using an MTP-sized allocation.
        device_plan_case(device, source, pool, views, 2, 2, true);  // MTP, first/middle failure and recovery.
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
