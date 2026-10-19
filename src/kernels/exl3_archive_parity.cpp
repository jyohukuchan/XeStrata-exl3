// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Bounded real-checkpoint probe: archive -> packed CPU/GPU linear operations.
#include "strata/artifact/exl3.hpp"
#include "strata/core/exl3_dense.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/exl3.hpp"
#include "strata/kernels/exl3.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace artifact = strata::artifact::exl3;
namespace cpu = strata::kernels::cpu::exl3;
namespace device_exl3 = strata::kernels::exl3;

namespace {
struct TemporaryPack {
    std::filesystem::path path;
    TemporaryPack(int64_t k, int64_t n) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto candidate = std::filesystem::temp_directory_path() / ("strata-exl3-probe-" + std::to_string(stamp));
        if (!std::filesystem::create_directory(candidate)) throw std::runtime_error("cannot create probe directory");
        path = candidate;
        std::ofstream index(path / "index.txt");
        // A metadata-only head row, as in a native pack.  No canonical weight
        // bytes are read: this tests the EXL3 attachment to the engine table.
        index << "# strata pack index v3\n# align 64 pool 64 tensors 1\n"
              << "output.weight 0 0 0 0 0 0 " << k << ' ' << n << " 8 0 32 0 0 0 0 0 0 0\n";
        if (!index) throw std::runtime_error("cannot write probe index");
    }
    ~TemporaryPack() {
        std::error_code error;
        if (!path.empty()) std::filesystem::remove_all(path, error);
    }
};
struct CapturedStream {
    strata::gpu::Stream stream = strata::gpu::stream_create();
    strata::gpu::Graph* graph = nullptr;
    ~CapturedStream() {
        if (graph != nullptr) strata::gpu::graph_destroy(graph);
        if (stream != nullptr) strata::gpu::stream_destroy(stream);
    }
};
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: exl3_archive_parity MODEL_DIR LINEAR_NAME\n");
        return 2;
    }
    try {
        const artifact::Archive archive(argv[1]);
        const auto it = std::find_if(archive.linears().begin(), archive.linears().end(),
                                     [&](const auto& item) { return item.name == argv[2]; });
        if (it == archive.linears().end()) throw std::runtime_error("requested EXL3 linear does not exist");
        artifact::MaterializedMatrix host;
        archive.materialize(*it, host);
        const auto host_view = host.view();
        if (cpu::validate(host_view) != cpu::Status::ok)
            throw std::runtime_error("materialized EXL3 descriptor failed validation");
        const size_t k = (size_t) host.k, n = (size_t) host.n;
        TemporaryPack pack(host.k, host.n);
        strata::core::WeightTable table;
        std::string error;
        const std::set<std::string> skip{"output.weight"};
        if (!table.load(pack.path.string(), nullptr, 0, error, &skip)) throw std::runtime_error(error);
        strata::core::Exl3Dense owner;
        if (!owner.load(archive, it->name, table, error, 4)) throw std::runtime_error(error);
        if (table.find("output.weight")->exl3 != &owner) throw std::runtime_error("EXL3 owner was not attached");
        // This diagnostic deliberately reconstructs only this selected matrix.
        // The runtime linear operations below keep all weights packed.
        std::vector<uint16_t> expected_w(k * n);
        if (cpu::reconstruct(host_view, expected_w.data(), expected_w.size()) != cpu::Status::ok)
            throw std::runtime_error("CPU reconstruction failed");

        strata::parity::Device memory;
        device_exl3::MatrixView view = host_view;
        view.trellis = memory.upload(host.trellis);
        view.suh = memory.upload(host.suh);
        view.svh = memory.upload(host.svh);
        view.bias = host.bias.empty() ? nullptr : memory.upload(host.bias);
        auto& queue = strata::core::Runtime::get().compute();
        uint16_t* decoded = memory.alloc<uint16_t>(expected_w.size());
        device_exl3::exl3_reconstruct(queue, view, decoded).wait_and_throw();
        std::vector<uint16_t> got_w(expected_w.size());
        strata::parity::copy_out(got_w.data(), decoded, got_w.size());
        size_t weight_bad = 0;
        for (size_t i = 0; i < got_w.size(); ++i) weight_bad += got_w[i] != expected_w[i];
        std::printf("archive: %zu indexed tensors, %zu linears, %zu sidecars\n",
                    archive.indexed_tensors().size(), archive.linears().size(), archive.sidecars().size());
        std::printf("linear: %s [%lld,%lld], bits %d codebook %d, reconstruct %zu bad\n",
                    it->name.c_str(), (long long) host.k, (long long) host.n, host.bits, host.codebook, weight_bad);
        size_t failures = weight_bad != 0;
        for (const int64_t rows : {int64_t(1), int64_t(4)}) {
            const size_t ldx = k + 7, ldy = n + 11;
            std::vector<float> x((size_t) rows * ldx, -99.0f);
            std::vector<float> expected((size_t) rows * ldy, 99.0f);
            for (int64_t r = 0; r < rows; ++r) {
                for (size_t i = 0; i < k; ++i)
                    x[(size_t) r * ldx + i] = (float) ((int) ((i * 17 + (size_t) r * 31) % 257) - 128) / 1024.0f;
                const auto status = cpu::linear(host_view, x.data() + (size_t) r * ldx,
                                                expected.data() + (size_t) r * ldy);
                if (status != cpu::Status::ok) throw std::runtime_error(cpu::status_string(status));
            }
            float* input = memory.upload(x);
            std::vector<float> initial(expected.size(), 99.0f);
            float* output = memory.upload(initial);
            const size_t scratch_count = device_exl3::exl3_gemm_scratch_elements(view, rows);
            float* scratch = memory.alloc<float>(scratch_count);
            device_exl3::exl3_gemm_fp32(queue, view, input, (int64_t) ldx, output, (int64_t) ldy,
                                       rows, scratch, scratch_count).wait_and_throw();
            std::vector<float> got(expected.size());
            strata::parity::copy_out(got.data(), output, got.size());
            size_t bad = 0, padding_bad = 0;
            float max_abs = 0;
            for (int64_t r = 0; r < rows; ++r) {
                for (size_t j = 0; j < n; ++j) {
                    const size_t at = (size_t) r * ldy + j;
                    const float error = std::fabs(got[at] - expected[at]);
                    max_abs = std::max(max_abs, error);
                    if (!std::isfinite(got[at]) || !std::isfinite(expected[at]) ||
                        error > 5e-5f * std::max(1.0f, std::fabs(expected[at]))) ++bad;
                }
                for (size_t j = n; j < ldy; ++j) padding_bad += got[(size_t) r * ldy + j] != 99.0f;
            }
            std::printf("rows %lld: max_abs %.9g, %zu value and %zu padding bad\n",
                        (long long) rows, (double) max_abs, bad, padding_bad);
            failures += bad != 0 || padding_bad != 0;
            // The engine owner uploads this same selected archive matrix and
            // supplies persistent, graph-stable scratch through WeightRef.
            float* owned_output = memory.upload(initial);
            if (!owner.run(input, (int64_t) ldx, owned_output, (int64_t) ldy, rows, nullptr, error))
                throw std::runtime_error(error);
            strata::parity::copy_out(got.data(), owned_output, got.size());
            size_t owner_bad = 0;
            for (size_t at = 0; at < expected.size(); ++at) {
                const bool padding = at % ldy >= n;
                if (!std::isfinite(got[at]) || (padding ? got[at] != 99.0f :
                    std::fabs(got[at] - expected[at]) > 5e-5f * std::max(1.0f, std::fabs(expected[at])))) ++owner_bad;
            }
            std::printf("rows %lld engine owner: %zu bad\n", (long long) rows, owner_bad);
            failures += owner_bad != 0;

            CapturedStream captured;
            float* graph_output = memory.upload(initial);
            if (captured.stream == nullptr || !owner.prepare_stream(captured.stream, error))
                throw std::runtime_error("preparing captured EXL3 stream: " + error);
            if (!strata::gpu::begin_capture(captured.stream)) throw std::runtime_error(strata::gpu::last_error());
            if (!owner.run(input, (int64_t) ldx, graph_output, (int64_t) ldy, rows, captured.stream, error)) {
                strata::gpu::abandon_capture(captured.stream);
                throw std::runtime_error(error);
            }
            if (!strata::gpu::end_capture(captured.stream, &captured.graph) ||
                !strata::gpu::graph_launch(captured.graph, captured.stream) ||
                !strata::gpu::stream_sync(captured.stream)) throw std::runtime_error(strata::gpu::last_error());
            strata::parity::copy_out(got.data(), graph_output, got.size());
            size_t graph_bad = 0;
            for (size_t at = 0; at < expected.size(); ++at) {
                const bool padding = at % ldy >= n;
                if (!std::isfinite(got[at]) || (padding ? got[at] != 99.0f :
                    std::fabs(got[at] - expected[at]) > 5e-5f * std::max(1.0f, std::fabs(expected[at])))) ++graph_bad;
            }
            std::printf("rows %lld captured engine owner: %zu bad\n", (long long) rows, graph_bad);
            failures += graph_bad != 0;
        }
        return failures != 0 ? 1 : 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "exl3_archive_parity: %s\n", error.what());
        return 2;
    }
}
