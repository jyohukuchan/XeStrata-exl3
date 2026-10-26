// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Bounded EXL3 runtime source/cache/CPU/GPU integration parity.
#include "strata/core/exl3_expert_runtime.hpp"

#include "strata/core/gpu.hpp"
#include "strata/kernels/cpu/exl3_expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "../kernels/parity_device.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#define STRATA_EXL3_RUNTIME_PID _getpid()
#else
#include <unistd.h>
#define STRATA_EXL3_RUNTIME_PID getpid()
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

void add_linear(std::vector<Tensor>& tensors, const std::string& name, int bits, int codebook, uint64_t tiles_k = 8,
                uint64_t tiles_n = 8) {
    tensors.emplace_back(name + ".trellis", "I16",
                         std::vector<uint64_t>{tiles_k, tiles_n, static_cast<uint64_t>(16 * bits)},
                         words(static_cast<size_t>(tiles_k * tiles_n * 16 * bits), static_cast<uint16_t>(bits * 19)));
    tensors.emplace_back(name + ".suh", "F16", std::vector<uint64_t>{tiles_k * 16}, words(tiles_k * 16, 0x3c00));
    tensors.emplace_back(name + ".svh", "F16", std::vector<uint64_t>{tiles_n * 16}, words(tiles_n * 16, 0x3c00));
    if (codebook == 1) tensors.emplace_back(name + ".mcg", "U32", std::vector<uint64_t>{}, u32(0xCBAC1FEDu));
    if (codebook == 2) tensors.emplace_back(name + ".mul1", "I32", std::vector<uint64_t>{1}, u32(0x83DCD12Du));
}

void write_archive(const std::filesystem::path& root,
                   std::vector<strata::artifact::exl3::ExpertTensorNames>& names) {
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json") << R"({"model_type":"synthetic-exl3-runtime"})";
    std::vector<std::vector<Tensor>> shards(3);
    for (int layer = 0; layer < 2; ++layer)
        for (int expert = 0; expert < 2; ++expert) {
            const std::string base = "layers." + std::to_string(layer) + ".experts." + std::to_string(expert);
            names.push_back({base + ".gate", base + ".up", base + ".down"});
            const int bits = 2 + ((layer * 2 + expert) % 4);
            // Each selected resident expert has its own shard; both CPU-miss experts share the third shard.
            auto& shard = shards[expert == 0 ? layer : 2];
            // The resident experts and one CPU-miss expert are MUL1 in all three projections, and their output
            // width covers two 128-column blocks: the host cache keeps the CPU block tile order for them, and
            // the device keeps the format's own order.  The last expert keeps the mixed codebooks.
            const bool mul1 = expert == 0 || (layer == 0 && expert == 1);
            add_linear(shard, base + ".gate", bits, mul1 ? 2 : layer % 3, 8, mul1 ? 16 : 8);
            add_linear(shard, base + ".up", 2 + ((bits + 1) % 4), mul1 ? 2 : (layer + expert + 1) % 3, 8, mul1 ? 16 : 8);
            add_linear(shard, base + ".down", 2 + ((bits + 2) % 4), mul1 ? 2 : (layer + expert + 2) % 3, mul1 ? 16 : 8,
                       8);
        }
    Json index;
    index["weight_map"] = Json::object();
    for (size_t shard_index = 0; shard_index < shards.size(); ++shard_index) {
        const std::string filename = "model-0000" + std::to_string(shard_index + 1) + "-of-00003.safetensors";
        Json shard_header = Json::object();
        uint64_t offset = 0;
        for (const auto& [name, dtype, shape, data] : shards[shard_index]) {
            shard_header[name] = Json{{"dtype", dtype}, {"shape", shape},
                                      {"data_offsets", {offset, offset + data.size()}}};
            index["weight_map"][name] = filename;
            offset += data.size();
        }
        const std::string text = shard_header.dump();
        std::ofstream output(root / filename, std::ios::binary);
        for (int i = 0; i < 8; ++i) output.put(static_cast<char>(text.size() >> (8 * i)));
        output.write(text.data(), static_cast<std::streamsize>(text.size()));
        for (const auto& [name, dtype, shape, data] : shards[shard_index]) {
            (void) name; (void) dtype; (void) shape;
            output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        }
    }
    std::ofstream(root / "model.safetensors.index.json") << index.dump();
}

bool close(float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return a == b;
    return std::fabs(a - b) <= 2e-3f * std::max(1.0f, std::fabs(b));
}

bool compare_merged(const std::vector<float>& cpu, const std::vector<float>& gpu, const std::vector<float>& expected,
                   int64_t rows, int64_t ldy, int64_t hidden, const char* label) {
    size_t bad = 0;
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t j = 0; j < hidden; ++j) {
            const size_t at = static_cast<size_t>(r) * static_cast<size_t>(ldy) + static_cast<size_t>(j);
            const float merged = std::isfinite(gpu[at]) && gpu[at] != 24680.5f ? gpu[at] : cpu[at];
            if (!close(merged, expected[at])) ++bad;
        }
    std::printf("%s: %s (%zu bad)\n", label, bad == 0 ? "OK" : "FAIL", bad);
    if (bad != 0)
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t j = 0; j < hidden; ++j) {
                const size_t at = static_cast<size_t>(r) * static_cast<size_t>(ldy) + static_cast<size_t>(j);
                const float merged = std::isfinite(gpu[at]) && gpu[at] != 24680.5f ? gpu[at] : cpu[at];
                if (!close(merged, expected[at])) {
                    std::printf("  first mismatch row=%lld col=%lld cpu=%.9g gpu=%.9g expected=%.9g\n",
                                (long long) r, (long long) j, (double) cpu[at], (double) gpu[at],
                                (double) expected[at]);
                    r = rows; break;
                }
            }
    return bad == 0;
}

}  // namespace

int main() {
    // plan_split_gpu is a pure function of its arguments: exact decisions on synthetic costs,
    // with no device needed.
    {
        using Runtime = strata::core::Exl3ExpertRuntime;
        using Candidate = Runtime::SplitCandidate;
        int split_math_failures = 0;
        const auto expect_split = [&](int64_t got, int64_t want, const char* label) {
            if (got != want) {
                std::fprintf(stderr, "plan_split_gpu %s: got %lld want %lld\n", label, (long long) got,
                             (long long) want);
                ++split_math_failures;
            }
        };
        const Candidate any[2] = {{0, 1, 1000}, {1, 1, 1000}};
        expect_split(Runtime::plan_split_gpu(nullptr, 2, 100.0, 10e9, 8), 0, "null");
        expect_split(Runtime::plan_split_gpu(any, 0, 100.0, 10e9, 8), 0, "empty");
        expect_split(Runtime::plan_split_gpu(any, 2, 100.0, 10e9, 0), 0, "no capacity");
        expect_split(Runtime::plan_split_gpu(any, 2, 0.0, 10e9, 8), 0, "no cpu sample");
        expect_split(Runtime::plan_split_gpu(any, 2, -1.0, 10e9, 8), 0, "negative cpu");
        expect_split(Runtime::plan_split_gpu(any, 2, 100.0, 0.0, 8), 0, "no bandwidth");
        // One entry, 1 MB over 10 GB/s: the GPU's 100 us ties the CPU's 100 us, and a tie
        // keeps the smaller k (the CPU default).
        const Candidate tie[1] = {{0, 1, 1000000}};
        expect_split(Runtime::plan_split_gpu(tie, 1, 100.0, 10e9, 8), 0, "tie stays cpu");
        // Four entries at 100 us each on the CPU against 1 MB over 10 GB/s on the GPU:
        // k=0 costs 400, k=1 costs max(0, 100) = 100.
        const Candidate one[1] = {{0, 4, 1000000}};
        expect_split(Runtime::plan_split_gpu(one, 1, 100.0, 10e9, 8), 1, "gpu wins");
        // Two experts at 300 us of CPU each against 600 KB over 10 GB/s (60 us) each:
        // k=0 costs 600, k=1 costs 300, k=2 costs 120.
        const Candidate two[2] = {{0, 1, 600000}, {1, 1, 600000}};
        expect_split(Runtime::plan_split_gpu(two, 2, 300.0, 10e9, 8), 2, "both to gpu");
        // The same pair over 0.1 GB/s (6000 us each): k=1 costs 6000 against k=0's 600.
        expect_split(Runtime::plan_split_gpu(two, 2, 300.0, 0.1e9, 8), 0, "slow bus stays cpu");
        // The capacity caps the GPU share even where the bus would take more.
        expect_split(Runtime::plan_split_gpu(two, 2, 300.0, 10e9, 1), 1, "capacity caps");
        if (split_math_failures == 0) std::printf("split plan math: OK\n");
        if (split_math_failures != 0) return 1;
    }
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = std::filesystem::temp_directory_path() /
                      ("xestrata_exl3_runtime_test_" + std::to_string(STRATA_EXL3_RUNTIME_PID) +
                       "_" + std::to_string(stamp));
    std::filesystem::remove_all(root);
    std::vector<strata::artifact::exl3::ExpertTensorNames> names;
    write_archive(root, names);
    std::vector<std::pair<std::filesystem::path, std::filesystem::path>> moved_resident_shards;
    const auto restore_resident_shards = [&] {
        for (auto it = moved_resident_shards.rbegin(); it != moved_resident_shards.rend(); ++it) {
            std::error_code restore_error;
            if (std::filesystem::exists(it->second)) std::filesystem::rename(it->second, it->first, restore_error);
        }
        moved_resident_shards.clear();
    };
    int failures = 0;
    try {
        auto archive = std::make_shared<strata::artifact::exl3::Archive>(root);
        strata::core::Exl3ExpertSource source(archive, 2, 2, names);
        strata::kernels::cpu::ExpertPool pool(1, false, false);
        strata::core::Exl3ExpertRuntime runtime;
        std::string err;
        if (!runtime.open(source, pool, 2, 2, {{0, 0}, {1, 0}}, err)) {
            std::fprintf(stderr, "runtime open failed: %s\n", err.c_str());
            std::filesystem::remove_all(root);
            return 1;
        }
        if (runtime.cache().slot_format(runtime.cache().slot_of(0, 0)) == 0 || runtime.cache().resident() != 2) {
            std::fprintf(stderr, "cache format buckets did not preload two selected pairs\n");
            ++failures;
        }
        constexpr int64_t hidden = 128, max_tokens = 2, k = 2, ldx = 131, ldy = 133;
        const int32_t ids[] = {0, 1, 1, 0};
        std::vector<float> x(static_cast<size_t>(max_tokens * ldx), -777.0f);
        for (int64_t r = 0; r < max_tokens; ++r)
            for (int64_t i = 0; i < hidden; ++i) x[static_cast<size_t>(r * ldx + i)] =
                0.013f * static_cast<float>(((r + 3) * 17 + i * 11) % 101 - 50);
        std::vector<std::vector<uint8_t>> blobs(2);
        std::vector<strata::kernels::exl3::ExpertView> views;
        for (int expert = 0; expert < 2; ++expert) {
            const auto* layout = source.exl3_layout(0, expert);
            blobs[static_cast<size_t>(expert)].resize(static_cast<size_t>(layout->total_bytes));
            if (!source.copy_blob(0, expert, blobs[static_cast<size_t>(expert)].data())) ++failures;
            views.push_back(strata::kernels::exl3::expert_view(*layout, blobs[static_cast<size_t>(expert)].data()));
        }
        // The two selected resident pairs are now fully cached. Remove only their shards; the CPU-miss shard remains
        // readable. A successful mixed window after this mutation proves cache hits do not materialize from source.
        for (const std::string& filename : {"model-00001-of-00003.safetensors",
                                             "model-00002-of-00003.safetensors"}) {
            const auto original = root / filename;
            const auto hidden = root / (filename + ".resident-offline");
            std::filesystem::rename(original, hidden);
            moved_resident_shards.emplace_back(original, hidden);
        }
        std::vector<float> expected(static_cast<size_t>(max_tokens * k * ldy), 24680.5f);
        for (int64_t entry = 0; entry < max_tokens * k; ++entry) {
            if (strata::kernels::cpu::exl3::expert(views[static_cast<size_t>(ids[entry])],
                                                   x.data() + (entry / k) * ldx, ldx,
                                                   expected.data() + entry * ldy, ldy, 1) !=
                strata::kernels::cpu::exl3::Status::ok)
                ++failures;
        }
        std::vector<float> cpuout(expected.size(), 24680.5f);
        if (!runtime.prepare_window(0, x.data(), ldx, ids, max_tokens, cpuout.data(), ldy, err) ||
            !runtime.run_cpu(err)) {
            std::fprintf(stderr, "runtime CPU preparation failed: %s\n", err.c_str());
            ++failures;
        }
        strata::parity::Device device;
        float* d_x = device.upload(x);
        std::vector<float> gpuout(expected.size(), 24680.5f);
        float* d_gpuout = device.upload(gpuout);
        sycl::event done;
        auto& queue = strata::core::Runtime::get().compute();
        if (!runtime.enqueue(queue, d_x, ldx, d_gpuout, ldy, done, {}, err) || !runtime.wait_gpu(done, err)) {
            std::fprintf(stderr, "runtime GPU enqueue failed: %s\n", err.c_str());
            ++failures;
        }
        strata::parity::copy_out(gpuout.data(), d_gpuout, gpuout.size());
        if (!compare_merged(cpuout, gpuout, expected, max_tokens * k, ldy, hidden, "direct CPU/GPU merge")) ++failures;

        // The bounded hybrid contract launches resident hits before the CPU miss pool. The same output must remain
        // unchanged after the event is drained, while the legacy CPU-first enqueue above remains covered separately.
        std::vector<float> overlap_cpu(expected.size(), 24680.5f), overlap_gpu(expected.size(), 24680.5f);
        strata::parity::copy_in(d_gpuout, overlap_gpu.data(), overlap_gpu.size());
        sycl::event overlap_done;
        if (!runtime.prepare_window(0, x.data(), ldx, ids, max_tokens, overlap_cpu.data(), ldy, err) ||
            !runtime.enqueue_hit_only(queue, d_x, ldx, d_gpuout, ldy, overlap_done, {}, err) ||
            !runtime.run_cpu(err) || !runtime.wait_gpu(overlap_done, err)) {
            std::fprintf(stderr, "runtime hit-before-CPU overlap failed: %s\n", err.c_str());
            ++failures;
        } else {
            strata::parity::copy_out(overlap_gpu.data(), d_gpuout, overlap_gpu.size());
            if (!compare_merged(overlap_cpu, overlap_gpu, expected, max_tokens * k, ldy, hidden,
                                "hit-before-CPU merge"))
                ++failures;
        }

        // The GPU-side merge: the misses upload from pinned host memory and are added into the hit
        // rows' buffer on the compute queue, without wait_gpu.  The device-merged rows must equal the
        // host loop bit for bit, and the next window must prepare cleanly after mark_gpu_consumed.
        {
            const size_t tight_count = static_cast<size_t>(max_tokens * k * hidden);
            std::vector<float> tight_x(static_cast<size_t>(max_tokens * hidden));
            for (int64_t row = 0; row < max_tokens; ++row)
                std::copy_n(x.data() + row * ldx, hidden, tight_x.data() + row * hidden);
            std::vector<float> tight_expected(tight_count, 24680.5f);
            for (int64_t entry = 0; entry < max_tokens * k; ++entry)
                if (strata::kernels::cpu::exl3::expert(views[static_cast<size_t>(ids[entry])],
                                                       tight_x.data() + (entry / k) * hidden, hidden,
                                                       tight_expected.data() + entry * hidden, hidden,
                                                       1) != strata::kernels::cpu::exl3::Status::ok)
                    ++failures;
            float* d_tight_x = device.upload(tight_x);
            std::vector<float> tight_zero(tight_count, 0.0f);
            float* d_parts = device.upload(tight_zero);  // cleared parts, like the model's memset
            float* d_pre = device.alloc<float>(tight_count);
            float* d_staging = device.alloc<float>(tight_count);
            float* pinned =
                static_cast<float*>(strata::gpu::alloc_host(tight_count * sizeof(float)));
            strata::gpu::Event* merge_mark = nullptr;
            bool merge_ok = pinned != nullptr && d_pre != nullptr && d_staging != nullptr &&
                            strata::gpu::event_create(&merge_mark, false);
            sycl::event window_done;
            if (merge_ok &&
                (!runtime.prepare_window(0, tight_x.data(), hidden, ids, max_tokens, pinned, hidden, err) ||
                 runtime.miss_entries() != 2 || runtime.hit_entries() != 2 ||
                 !runtime.enqueue_hit_only(queue, d_tight_x, hidden, d_parts, hidden, window_done, {}, err) ||
                 !runtime.run_cpu(err))) {
                std::fprintf(stderr, "runtime GPU-merge window failed: %s\n", err.c_str());
                merge_ok = false;
            }
            if (merge_ok) {
                // Miss rows of the cleared parts must still read +0.0 (the hit kernels plan hits only),
                // and hit rows of the CPU buffer must read +0.0: the halves merged below are disjoint.
                std::vector<float> pre_merge(tight_count, 24680.5f);
                if (!strata::gpu::copy_async(d_pre, d_parts, tight_count * sizeof(float), nullptr) ||
                    !strata::gpu::copy_async(d_staging, pinned, tight_count * sizeof(float), nullptr)) {
                    std::fprintf(stderr, "runtime GPU-merge queue failed: %s\n", strata::gpu::last_error());
                    merge_ok = false;
                } else {
                    strata::kernels::add_inplace(d_parts, d_staging, static_cast<int64_t>(tight_count),
                                                 strata::parity::stream());
                    merge_ok = strata::gpu::event_record(merge_mark, nullptr) &&
                               strata::gpu::stream_sync(nullptr);
                    if (!merge_ok)
                        std::fprintf(stderr, "runtime GPU-merge mark failed: %s\n", strata::gpu::last_error());
                }
                if (merge_ok) {
                    if (strata::gpu::event_status(merge_mark) != strata::gpu::Status::Done) {
                        std::fprintf(stderr, "runtime GPU-merge mark is not done after the queue drained\n");
                        merge_ok = false;
                    }
                    strata::parity::copy_out(pre_merge.data(), d_pre, tight_count);
                    std::vector<float> device_merged(tight_count, 24680.5f);
                    strata::parity::copy_out(device_merged.data(), d_parts, tight_count);
                    size_t bad = 0;
                    for (int64_t entry = 0; entry < max_tokens * k && bad == 0; ++entry) {
                        const bool hit = ids[entry] == 0;
                        for (int64_t col = 0; col < hidden; ++col) {
                            const size_t at = static_cast<size_t>(entry * hidden + col);
                            const float want_pre = hit ? pre_merge[at] : 0.0f;
                            const float want_cpu = hit ? 0.0f : pinned[at];
                            if (pre_merge[at] != want_pre || pinned[at] != want_cpu ||
                                device_merged[at] != pre_merge[at] + pinned[at]) {
                                std::fprintf(stderr,
                                             "runtime GPU-merge mismatch entry=%lld col=%lld pre=%.9g cpu=%.9g "
                                             "device=%.9g\n",
                                             (long long) entry, (long long) col, (double) pre_merge[at],
                                             (double) pinned[at], (double) device_merged[at]);
                                bad = 1;
                                break;
                            }
                        }
                    }
                    if (bad != 0) merge_ok = false;
                }
                // The pending plan is consumed without blocking: the next window must prepare cleanly.
                runtime.mark_gpu_consumed();
                const int32_t ids_all_hit[] = {0, 0, 0, 0};
                std::vector<float> second_expected(tight_count, 24680.5f);
                for (int64_t entry = 0; entry < max_tokens * k; ++entry)
                    if (strata::kernels::cpu::exl3::expert(views[0], tight_x.data() + (entry / k) * hidden,
                                                           hidden, second_expected.data() + entry * hidden,
                                                           hidden, 1) != strata::kernels::cpu::exl3::Status::ok)
                        merge_ok = false;
                std::vector<float> second_cpu(tight_count, 24680.5f);
                sycl::event second_done;
                if (!runtime.prepare_window(0, tight_x.data(), hidden, ids_all_hit, max_tokens, second_cpu.data(),
                                            hidden, err) ||
                    runtime.miss_entries() != 0 ||
                    !runtime.enqueue_hit_only(queue, d_tight_x, hidden, d_parts, hidden, second_done, {}, err) ||
                    !runtime.run_cpu(err) || !runtime.wait_gpu(second_done, err)) {
                    std::fprintf(stderr, "runtime window after GPU merge failed: %s\n", err.c_str());
                    merge_ok = false;
                } else {
                    std::vector<float> second_gpu(tight_count, 24680.5f);
                    strata::parity::copy_out(second_gpu.data(), d_parts, tight_count);
                    if (!compare_merged(second_cpu, second_gpu, second_expected, max_tokens * k, hidden, hidden,
                                        "window after GPU-merge consume"))
                        merge_ok = false;
                }
            }
            if (merge_ok) {
                std::printf("GPU-side merge without wait_gpu: OK\n");
            } else {
                ++failures;
            }
            if (pinned != nullptr) strata::gpu::free(pinned);
            if (merge_mark != nullptr) strata::gpu::event_destroy(merge_mark);
        }

        // Capture the plan-copy nodes and grouped graph, then replay after prepare_window updates only the persistent
        // host plan image.  The graph is intentionally not recaptured.
        std::vector<float> graph_gpu(expected.size(), 24680.5f);
        float* d_graph_gpu = device.upload(graph_gpu);
        strata::gpu::Stream stream = strata::gpu::stream_create();
        strata::gpu::Graph* graph = nullptr;
        bool graph_ok = stream != nullptr;
        if (graph_ok) {
            auto& capture_queue = strata::core::Runtime::get().stream(stream);
            if (!runtime.prepare_window(0, x.data(), ldx, ids, max_tokens, cpuout.data(), ldy, err) ||
                !runtime.run_cpu(err) || !strata::gpu::begin_capture(stream)) graph_ok = false;
            sycl::event captured_done;
            if (graph_ok && (!runtime.enqueue(capture_queue, d_x, ldx, d_graph_gpu, ldy, captured_done, {}, err) ||
                             !strata::gpu::end_capture(stream, &graph) || graph == nullptr ||
                             !strata::gpu::graph_launch(graph, stream) || !strata::gpu::stream_sync(stream)))
                graph_ok = false;
            if (graph_ok) runtime.mark_gpu_consumed();
            if (graph_ok) {
                strata::parity::copy_out(graph_gpu.data(), d_graph_gpu, graph_gpu.size());
                if (!compare_merged(cpuout, graph_gpu, expected, max_tokens * k, ldy, hidden, "captured CPU/GPU merge"))
                    graph_ok = false;
            }
            const int32_t ids_replay[] = {1, 0, 1, 0};
            std::vector<float> expected_replay(expected.size(), 24680.5f);
            for (int64_t entry = 0; entry < max_tokens * k; ++entry)
                if (strata::kernels::cpu::exl3::expert(views[static_cast<size_t>(ids_replay[entry])],
                                                       x.data() + (entry / k) * ldx, ldx,
                                                       expected_replay.data() + entry * ldy, ldy, 1) !=
                    strata::kernels::cpu::exl3::Status::ok)
                    graph_ok = false;
            std::fill(graph_gpu.begin(), graph_gpu.end(), 24680.5f);
            strata::parity::copy_in(d_graph_gpu, graph_gpu.data(), graph_gpu.size());
            if (graph_ok && (!runtime.prepare_window(0, x.data(), ldx, ids_replay, max_tokens, cpuout.data(), ldy, err) ||
                             !runtime.run_cpu(err) || !strata::gpu::graph_launch(graph, stream) ||
                             !strata::gpu::stream_sync(stream)))
                graph_ok = false;
            if (graph_ok) {
                strata::parity::copy_out(graph_gpu.data(), d_graph_gpu, graph_gpu.size());
                if (!compare_merged(cpuout, graph_gpu, expected_replay, max_tokens * k, ldy, hidden,
                                    "replayed plan merge"))
                    graph_ok = false;
            }
        }
        if (!graph_ok) {
            std::fprintf(stderr, "runtime graph capture/replay failed: %s\n", err.empty() ? strata::gpu::last_error() : err.c_str());
            ++failures;
        } else {
            std::printf("runtime graph capture/replay: OK\n");
        }
        if (graph != nullptr) strata::gpu::graph_destroy(graph);
        if (stream != nullptr) strata::gpu::stream_destroy(stream);
        restore_resident_shards();

        // Direct miss mode: the GPU reads the non-resident expert from pinned host memory and computes
        // it itself, so the merged rows match the CPU-miss run and no entry is left to the CPU pool.
        // Heat recording counts every routed entry for the profile a later run ranks with.
        {
            strata::core::Exl3ExpertRuntime direct;
            direct.set_miss_mode(strata::core::Exl3ExpertRuntime::MissMode::Direct);
            direct.set_record_heat(true);
            std::vector<float> direct_cpu(expected.size(), 24680.5f), direct_gpu(expected.size(), 24680.5f);
            float* d_direct_gpu = device.upload(direct_gpu);
            sycl::event direct_done;
            if (!direct.open(source, pool, max_tokens, k, {{0, 0}, {1, 0}}, err) ||
                !direct.prepare_window(0, x.data(), ldx, ids, max_tokens, direct_cpu.data(), ldy, err) ||
                !direct.enqueue_hit_only(queue, d_x, ldx, d_direct_gpu, ldy, direct_done, {}, err) ||
                !direct.run_cpu(err) || !direct.wait_gpu(direct_done, err)) {
                std::fprintf(stderr, "direct miss runtime failed: %s\n", err.c_str());
                ++failures;
            } else {
                strata::parity::copy_out(direct_gpu.data(), d_direct_gpu, direct_gpu.size());
                if (!compare_merged(direct_cpu, direct_gpu, expected, max_tokens * k, ldy, hidden,
                                    "direct-read miss merge"))
                    ++failures;
                if (!direct.direct_ready() || direct.total_hit_entries() != 2 ||
                    direct.total_cpu_miss_entries() != 0 || direct.total_direct_miss_entries() != 2 ||
                    direct.heat().size() != 4 || direct.heat()[0] != 2 || direct.heat()[1] != 2) {
                    std::fprintf(stderr, "direct miss counters/heat are wrong\n");
                    ++failures;
                }
            }
        }

        // Split miss mode: a window's misses divide between the GPU (pinned staging, the same
        // grouped launch as the hits) and the CPU pool.  Layer 0 routes one distinct miss
        // (expert 1, two entries), so the two queued windows below each decide one expert.
        // The first window has no CPU cost sample yet and stays all-CPU; the second divides
        // from the measured costs.  Both merge to the CPU reference: the hit, CPU and GPU
        // rows are disjoint by construction.
        {
            strata::core::Exl3ExpertRuntime split;
            split.set_miss_mode(strata::core::Exl3ExpertRuntime::MissMode::Split);
            std::vector<float> split_cpu(expected.size(), 24680.5f), split_gpu(expected.size(), 24680.5f);
            float* d_split_gpu = device.upload(split_gpu);
            sycl::event split_done;
            bool split_ok =
                split.open(source, pool, max_tokens, k, {{0, 0}, {1, 0}}, err) && split.direct_ready() &&
                split.prepare_window(0, x.data(), ldx, ids, max_tokens, split_cpu.data(), ldy, err) &&
                split.enqueue_hit_only(queue, d_x, ldx, d_split_gpu, ldy, split_done, {}, err) &&
                split.run_cpu(err) && split.wait_gpu(split_done, err);
            if (!split_ok) {
                std::fprintf(stderr, "split miss runtime failed: %s\n", err.c_str());
                ++failures;
            } else {
                strata::parity::copy_out(split_gpu.data(), d_split_gpu, split_gpu.size());
                if (!compare_merged(split_cpu, split_gpu, expected, max_tokens * k, ldy, hidden,
                                    "split-read miss merge"))
                    ++failures;
                if (split.split_windows() != 1 || split.split_last_gpu_experts() != 0 ||
                    split.split_last_cpu_experts() != 1) {
                    std::fprintf(stderr, "split first window did not stay all-CPU: windows=%lld gpu=%lld cpu=%lld\n",
                                 (long long) split.split_windows(), (long long) split.split_last_gpu_experts(),
                                 (long long) split.split_last_cpu_experts());
                    ++failures;
                }
            }
            // The second window divides from the first window's CPU sample and the open-time
            // bandwidth.  The direction is the machine's own (tiny synthetic experts may land
            // either way), but the decision is recorded and the halves still merge exactly.
            std::vector<float> split_cpu2(expected.size(), 24680.5f), split_gpu2(expected.size(), 24680.5f);
            strata::parity::copy_in(d_split_gpu, split_gpu2.data(), split_gpu2.size());
            if (split_ok &&
                (!split.prepare_window(0, x.data(), ldx, ids, max_tokens, split_cpu2.data(), ldy, err) ||
                 !split.enqueue_hit_only(queue, d_x, ldx, d_split_gpu, ldy, split_done, {}, err) ||
                 !split.run_cpu(err) || !split.wait_gpu(split_done, err))) {
                std::fprintf(stderr, "split miss second window failed: %s\n", err.c_str());
                ++failures;
            } else if (split_ok) {
                strata::parity::copy_out(split_gpu2.data(), d_split_gpu, split_gpu2.size());
                if (!compare_merged(split_cpu2, split_gpu2, expected, max_tokens * k, ldy, hidden,
                                    "split-read second merge"))
                    ++failures;
                if (split.split_windows() != 2 ||
                    split.split_last_gpu_experts() + split.split_last_cpu_experts() != 1) {
                    std::fprintf(stderr, "split second window decision is wrong: windows=%lld gpu=%lld cpu=%lld\n",
                                 (long long) split.split_windows(), (long long) split.split_last_gpu_experts(),
                                 (long long) split.split_last_cpu_experts());
                    ++failures;
                }
                if (split.split_all_miss_entries() !=
                    split.total_cpu_miss_entries() + split.total_direct_miss_entries()) {
                    std::fprintf(stderr, "split entry totals do not match the cpu/direct counters\n");
                    ++failures;
                }
                if (!(split.split_cpu_us_per_entry() > 0.0) || !(split.split_pcie_bytes_per_sec() > 0.0)) {
                    std::fprintf(stderr, "split controller has no measured costs: cpu=%g us/entry pcie=%g B/s\n",
                                 split.split_cpu_us_per_entry(), split.split_pcie_bytes_per_sec());
                    ++failures;
                }
            }
            // A third identical window must reuse the staged expert from the pinned cache:
            // one more lookup, this time a hit, with the same merged rows.
            std::vector<float> split_cpu3(expected.size(), 24680.5f), split_gpu3(expected.size(), 24680.5f);
            strata::parity::copy_in(d_split_gpu, split_gpu3.data(), split_gpu3.size());
            if (split_ok &&
                (!split.prepare_window(0, x.data(), ldx, ids, max_tokens, split_cpu3.data(), ldy, err) ||
                 !split.enqueue_hit_only(queue, d_x, ldx, d_split_gpu, ldy, split_done, {}, err) ||
                 !split.run_cpu(err) || !split.wait_gpu(split_done, err))) {
                std::fprintf(stderr, "split miss third window failed: %s\n", err.c_str());
                ++failures;
            } else if (split_ok) {
                strata::parity::copy_out(split_gpu3.data(), d_split_gpu, split_gpu3.size());
                if (!compare_merged(split_cpu3, split_gpu3, expected, max_tokens * k, ldy, hidden,
                                    "split-read third merge"))
                    ++failures;
                if (split.split_windows() != 3 ||
                    split.split_last_gpu_experts() + split.split_last_cpu_experts() != 1) {
                    std::fprintf(stderr, "split third window decision is wrong: windows=%lld gpu=%lld cpu=%lld\n",
                                 (long long) split.split_windows(), (long long) split.split_last_gpu_experts(),
                                 (long long) split.split_last_cpu_experts());
                    ++failures;
                }
                if (split.split_cache_slots() == 0 || split.split_cache_lookups() != 2 ||
                    split.split_cache_hits() != 1) {
                    std::fprintf(stderr, "split pinned cache did not serve the repeat: slots=%lld lookups=%lld hits=%lld\n",
                                 (long long) split.split_cache_slots(), (long long) split.split_cache_lookups(),
                                 (long long) split.split_cache_hits());
                    ++failures;
                }
            }
            if (split_ok) std::printf("split-read miss merge: OK (windows=%lld share=%.2f cpu=%.1f us/entry cache=%lld/%lld)\n",
                                      (long long) split.split_windows(), split.split_gpu_share(),
                                      split.split_cpu_us_per_entry(), (long long) split.split_cache_hits(),
                                      (long long) split.split_cache_lookups());
        }

        // The all-resident graph reads device IDs on every replay, without a host plan update.
        strata::core::Exl3ExpertRuntime resident;
        if (!resident.open(source, pool, max_tokens, k, {{0, 0}, {0, 1}, {1, 0}, {1, 1}}, err) ||
            !resident.fully_resident() || runtime.fully_resident()) {
            std::fprintf(stderr, "all-resident runtime open failed: %s\n", err.c_str());
            ++failures;
        } else {
            std::vector<float> tight_x(static_cast<size_t>(max_tokens * hidden));
            for (int64_t row = 0; row < max_tokens; ++row)
                std::copy_n(x.data() + row * ldx, hidden, tight_x.data() + row * hidden);
            float* d_tight_x = device.upload(tight_x);
            int32_t* d_ids = device.upload(std::vector<int32_t>(ids, ids + max_tokens * k));
            std::vector<float> direct_output(expected.size(), 24680.5f);
            float* d_direct_output = device.upload(direct_output);
            auto resident_stream = strata::gpu::stream_create();
            strata::gpu::Graph* resident_graph = nullptr;
            bool resident_ok = resident_stream != nullptr;
            if (resident_ok) {
                auto& resident_queue = strata::core::Runtime::get().stream(resident_stream);
                resident_ok = strata::gpu::begin_capture(resident_stream) &&
                    resident.record_resident_enqueue(resident_queue, 0, d_tight_x, d_ids, max_tokens,
                                                       d_direct_output, ldy, err) &&
                    strata::gpu::end_capture(resident_stream, &resident_graph) && resident_graph != nullptr;
            }
            const std::vector<std::vector<int32_t>> replay_ids{{0, 1, 1, 0}, {1, 1, 0, 0}, {0, -1, 2, 0}};
            for (size_t replay = 0; resident_ok && replay < replay_ids.size(); ++replay) {
                std::fill(direct_output.begin(), direct_output.end(), 24680.5f);
                strata::parity::copy_in(d_ids, replay_ids[replay].data(), replay_ids[replay].size());
                strata::parity::copy_in(d_direct_output, direct_output.data(), direct_output.size());
                resident_ok = strata::gpu::graph_launch(resident_graph, resident_stream) &&
                              strata::gpu::stream_sync(resident_stream);
                if (!resident_ok) break;
                strata::parity::copy_out(direct_output.data(), d_direct_output, direct_output.size());
                for (int64_t entry = 0; entry < max_tokens * k; ++entry) {
                    const int32_t expert = replay_ids[replay][static_cast<size_t>(entry)];
                    const bool valid = expert >= 0 && expert < 2;
                    if (valid && strata::kernels::cpu::exl3::expert(views[static_cast<size_t>(expert)],
                            tight_x.data() + (entry / k) * hidden, hidden,
                            expected.data() + entry * ldy, ldy, 1) != strata::kernels::cpu::exl3::Status::ok)
                        resident_ok = false;
                    for (int64_t col = 0; col < ldy; ++col) {
                        const size_t at = static_cast<size_t>(entry * ldy + col);
                        const bool matches = col >= hidden ? direct_output[at] == 24680.5f :
                            valid ? std::isfinite(direct_output[at]) && close(direct_output[at], expected[at]) :
                                    std::isnan(direct_output[at]);
                        if (!matches) {
                            std::fprintf(stderr, "resident replay %zu row %lld col %lld: got %.9g\n",
                                         replay, (long long) entry, (long long) col, (double) direct_output[at]);
                            resident_ok = false;
                            break;
                        }
                    }
                }
            }
            if (!resident_ok) {
                std::fprintf(stderr, "all-resident device-plan replay failed: %s\n", err.c_str());
                ++failures;
            } else {
                std::printf("all-resident device-plan replay/duplicates/invalid guards: OK\n");
            }
            if (resident_graph != nullptr) strata::gpu::graph_destroy(resident_graph);
            if (resident_stream != nullptr) strata::gpu::stream_destroy(resident_stream);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exl3_expert_runtime_test: %s\n", e.what());
        ++failures;
        restore_resident_shards();
    }
    std::filesystem::remove_all(root);
    std::printf("exl3_expert_runtime_test: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
