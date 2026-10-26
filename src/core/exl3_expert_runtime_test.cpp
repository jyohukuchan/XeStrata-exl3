// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Bounded EXL3 runtime source/cache/CPU/GPU integration parity.
#include "strata/core/exl3_expert_runtime.hpp"

#include "strata/core/gpu.hpp"
#include "strata/kernels/cpu/exl3_expert.hpp"
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

        // DMA miss mode: the GPU copies the non-resident expert from pinned host memory into VRAM
        // and computes it there, so the merged rows match the CPU-miss run and no entry is left to
        // the CPU pool.
        {
            strata::core::Exl3ExpertRuntime dma;
            dma.set_miss_mode(strata::core::Exl3ExpertRuntime::MissMode::Dma);
            std::vector<float> dma_cpu(expected.size(), 24680.5f), dma_gpu(expected.size(), 24680.5f);
            float* d_dma_gpu = device.upload(dma_gpu);
            sycl::event dma_done;
            if (!dma.open(source, pool, max_tokens, k, {{0, 0}, {1, 0}}, err) ||
                !dma.prepare_window(0, x.data(), ldx, ids, max_tokens, dma_cpu.data(), ldy, err) ||
                !dma.enqueue_hit_only(queue, d_x, ldx, d_dma_gpu, ldy, dma_done, {}, err) ||
                !dma.run_cpu(err) || !dma.wait_gpu(dma_done, err)) {
                std::fprintf(stderr, "dma miss runtime failed: %s\n", err.c_str());
                ++failures;
            } else {
                strata::parity::copy_out(dma_gpu.data(), d_dma_gpu, dma_gpu.size());
                if (!compare_merged(dma_cpu, dma_gpu, expected, max_tokens * k, ldy, hidden,
                                    "dma miss merge"))
                    ++failures;
                if (!dma.dma_ready() || dma.total_hit_entries() != 2 ||
                    dma.total_cpu_miss_entries() != 0 || dma.total_dma_miss_entries() != 2) {
                    std::fprintf(stderr, "dma miss counters are wrong\n");
                    ++failures;
                }
            }
        }

        // DMA staging LRU: a second identical window reuses the staged expert (no new
        // transfer), and a one-slot cache evicts across layers so each window transfers again.
        {
            strata::core::Exl3ExpertRuntime lru;
            lru.set_miss_mode(strata::core::Exl3ExpertRuntime::MissMode::Dma);
            lru.set_dma_staging_slots(4);
            std::vector<float> lru_cpu(expected.size(), 24680.5f), lru_gpu(expected.size(), 24680.5f);
            float* d_lru_gpu = device.upload(lru_gpu);
            bool lru_ok = lru.open(source, pool, max_tokens, k, {{0, 0}, {1, 0}}, err);
            for (int rep = 0; lru_ok && rep < 3; ++rep) {
                std::fill(lru_cpu.begin(), lru_cpu.end(), 24680.5f);
                std::fill(lru_gpu.begin(), lru_gpu.end(), 24680.5f);
                strata::parity::copy_in(d_lru_gpu, lru_gpu.data(), lru_gpu.size());
                sycl::event lru_done;
                lru_ok = lru.prepare_window(0, x.data(), ldx, ids, max_tokens, lru_cpu.data(), ldy, err) &&
                         lru.enqueue_hit_only(queue, d_x, ldx, d_lru_gpu, ldy, lru_done, {}, err) &&
                         lru.run_cpu(err) && lru.wait_gpu(lru_done, err);
                if (lru_ok) {
                    strata::parity::copy_out(lru_gpu.data(), d_lru_gpu, lru_gpu.size());
                    if (!compare_merged(lru_cpu, lru_gpu, expected, max_tokens * k, ldy, hidden,
                                        "dma lru reuse merge"))
                        lru_ok = false;
                }
            }
            if (!lru_ok) {
                std::fprintf(stderr, "dma lru reuse failed: %s\n", err.c_str());
                ++failures;
            } else if (lru.dma_staging_capacity() != 4 || lru.total_dma_staging_transfers() != 1 ||
                       lru.total_dma_staging_hits() != 2 || lru.total_dma_staging_evictions() != 0) {
                std::fprintf(stderr, "dma lru counters are wrong: cap=%lld xfer=%lld hits=%lld evict=%lld\n",
                             (long long) lru.dma_staging_capacity(),
                             (long long) lru.total_dma_staging_transfers(),
                             (long long) lru.total_dma_staging_hits(),
                             (long long) lru.total_dma_staging_evictions());
                ++failures;
            } else {
                std::printf("dma lru reuse: OK\n");
            }
            // One slot: layer 0 and layer 1 misses (different pairs) evict each other.
            strata::core::Exl3ExpertRuntime tiny;
            tiny.set_miss_mode(strata::core::Exl3ExpertRuntime::MissMode::Dma);
            tiny.set_dma_staging_slots(1);
            bool tiny_ok = tiny.open(source, pool, max_tokens, k, {{0, 0}, {1, 0}}, err);
            const int32_t ids_l1[] = {0, 1, 1, 0};
            for (int rep = 0; tiny_ok && rep < 2; ++rep) {
                for (int64_t layer = 0; layer < 2; ++layer) {
                    std::vector<float> t_cpu(expected.size(), 24680.5f), t_gpu(expected.size(), 24680.5f);
                    float* d_t_gpu = device.upload(t_gpu);
                    sycl::event t_done;
                    tiny_ok = tiny.prepare_window(layer, x.data(), ldx, ids_l1, max_tokens, t_cpu.data(), ldy, err) &&
                              tiny.enqueue_hit_only(queue, d_x, ldx, d_t_gpu, ldy, t_done, {}, err) &&
                              tiny.run_cpu(err) && tiny.wait_gpu(t_done, err);
                    if (!tiny_ok) break;
                }
            }
            if (!tiny_ok) {
                std::fprintf(stderr, "dma lru eviction failed: %s\n", err.c_str());
                ++failures;
            } else if (tiny.total_dma_staging_transfers() != 4 || tiny.total_dma_staging_hits() != 0 ||
                       tiny.total_dma_staging_evictions() != 3) {
                std::fprintf(stderr, "dma lru eviction counters are wrong: xfer=%lld hits=%lld evict=%lld\n",
                             (long long) tiny.total_dma_staging_transfers(),
                             (long long) tiny.total_dma_staging_hits(),
                             (long long) tiny.total_dma_staging_evictions());
                ++failures;
            } else {
                std::printf("dma lru eviction: OK\n");
            }
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
