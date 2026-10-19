// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// EXL3 packed GEMM stage profile. This reports stage costs; it is not a speed verdict.
#include "strata/artifact/exl3.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/exl3.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::filesystem::path archive;
    std::string head = "lm_head";
    std::string gate = "model.language_model.layers.0.mlp.experts.0.gate_proj";
    std::string up = "model.language_model.layers.0.mlp.experts.0.up_proj";
    std::string down = "model.language_model.layers.0.mlp.experts.0.down_proj";
    int warmup = 2;
    int repeats = 5;
    int rows = 1;
};

void usage() {
    std::fprintf(stderr,
                 "usage: exl3_bench ARCHIVE [--head NAME] [--gate NAME] [--up NAME] [--down NAME] "
                 "[--warmup N] [--repeats N] [--rows 1|4]\n");
}

bool parse_int(const char* text, int& out) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 1000000) return false;
    out = static_cast<int>(value);
    return true;
}

bool parse_options(int argc, char** argv, Options& out) {
    if (argc < 2) return false;
    out.archive = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string key = argv[i];
        if ((key == "--head" || key == "--gate" || key == "--up" || key == "--down") && i + 1 < argc) {
            const std::string value = argv[++i];
            if (key == "--head") out.head = value;
            else if (key == "--gate") out.gate = value;
            else if (key == "--up") out.up = value;
            else out.down = value;
        } else if ((key == "--warmup" || key == "--repeats" || key == "--rows") && i + 1 < argc) {
            int value = 0;
            if (!parse_int(argv[++i], value)) return false;
            if (key == "--warmup") out.warmup = value;
            else if (key == "--repeats") out.repeats = value;
            else out.rows = value;
        } else {
            return false;
        }
    }
    return !out.archive.empty() && out.repeats > 0 && (out.rows == 1 || out.rows == 4);
}

double event_ms(const sycl::event& event) {
    try {
        const uint64_t begin = event.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
        return static_cast<double>(end - begin) * 1e-6;
    } catch (const sycl::exception&) {
        return -1.0;
    }
}

struct DeviceMatrix {
    strata::kernels::exl3::MatrixView view{};
    uint16_t* trellis = nullptr;
    uint16_t* suh = nullptr;
    uint16_t* svh = nullptr;
    uint16_t* bias = nullptr;
    float* input = nullptr;
    float* output = nullptr;
    float* scratch = nullptr;

    void release() {
        if (trellis != nullptr) strata::gpu::free(trellis);
        if (suh != nullptr) strata::gpu::free(suh);
        if (svh != nullptr) strata::gpu::free(svh);
        if (bias != nullptr) strata::gpu::free(bias);
        if (input != nullptr) strata::gpu::free(input);
        if (output != nullptr) strata::gpu::free(output);
        if (scratch != nullptr) strata::gpu::free(scratch);
        trellis = suh = svh = bias = nullptr;
        input = output = scratch = nullptr;
    }
    ~DeviceMatrix() { release(); }
};

bool upload_matrix(const strata::artifact::exl3::MaterializedMatrix& host, int rows, DeviceMatrix& device,
                   std::string& error) {
    using strata::kernels::exl3::exl3_gemm_scratch_elements;
    device.view = host.view();
    auto alloc_copy = [&](const std::vector<uint16_t>& source, uint16_t*& destination, const char* label) {
        if (source.empty()) { destination = nullptr; return true; }
        destination = static_cast<uint16_t*>(strata::gpu::alloc_device(source.size() * sizeof(uint16_t)));
        if (destination == nullptr || !strata::gpu::copy(destination, source.data(), source.size() * sizeof(uint16_t))) {
            error = std::string("alloc/copy ") + label + ": " + strata::gpu::last_error();
            return false;
        }
        return true;
    };
    if (!alloc_copy(host.trellis, device.trellis, "trellis") || !alloc_copy(host.suh, device.suh, "suh") ||
        !alloc_copy(host.svh, device.svh, "svh") || !alloc_copy(host.bias, device.bias, "bias")) return false;
    device.view.trellis = device.trellis;
    device.view.suh = device.suh;
    device.view.svh = device.svh;
    device.view.bias = device.bias;
    const int64_t ldx = device.view.k + 7, ldy = device.view.n + 11;
    const size_t input_count = static_cast<size_t>(rows) * static_cast<size_t>(ldx);
    const size_t output_count = static_cast<size_t>(rows) * static_cast<size_t>(ldy);
    const size_t scratch_count = exl3_gemm_scratch_elements(device.view, rows);
    device.input = static_cast<float*>(strata::gpu::alloc_device(input_count * sizeof(float)));
    device.output = static_cast<float*>(strata::gpu::alloc_device(output_count * sizeof(float)));
    device.scratch = static_cast<float*>(strata::gpu::alloc_device(scratch_count * sizeof(float)));
    if (device.input == nullptr || device.output == nullptr || device.scratch == nullptr) {
        error = "FP32 benchmark buffer allocation failed";
        return false;
    }
    std::vector<float> input(input_count, 0.0f);
    for (int64_t row = 0; row < rows; ++row)
        for (int64_t i = 0; i < device.view.k; ++i)
            input[static_cast<size_t>(row * ldx + i)] =
                0.013f * static_cast<float>(((row + 5) * 19 + i * 7) % 101 - 50);
    if (!strata::gpu::copy(device.input, input.data(), input.size() * sizeof(float)) ||
        !strata::gpu::memset(device.output, 0, output_count * sizeof(float)) ||
        !strata::gpu::memset(device.scratch, 0, scratch_count * sizeof(float))) {
        error = std::string("FP32 benchmark buffer initialization failed: ") + strata::gpu::last_error();
        return false;
    }
    return true;
}

bool profile_matrix(const std::string& label, const strata::artifact::exl3::MaterializedMatrix& host, const Options& options,
                    sycl::queue& queue) {
    DeviceMatrix device;
    std::string error;
    if (!upload_matrix(host, options.rows, device, error)) {
        std::fprintf(stderr, "%s: %s\n", label.c_str(), error.c_str());
        return false;
    }
    const int64_t ldx = device.view.k + 7, ldy = device.view.n + 11;
    strata::kernels::exl3::GemvOptions gemv_options;
    for (int i = 0; i < options.warmup; ++i) {
        auto event = strata::kernels::exl3::exl3_gemm_fp32(
            queue, device.view, device.input, ldx, device.output, ldy, options.rows, device.scratch,
            strata::kernels::exl3::exl3_gemm_scratch_elements(device.view, options.rows), gemv_options);
        event.wait_and_throw();
    }
    double input_ms = 0.0, dot_ms = 0.0, output_ms = 0.0, total_ms = 0.0, host_wait_ms = 0.0;
    int profiled = 0;
    for (int i = 0; i < options.repeats; ++i) {
        strata::kernels::exl3::GemvProfile profile;
        const auto host_begin = std::chrono::steady_clock::now();
        sycl::event event = strata::kernels::exl3::exl3_gemm_fp32(
            queue, device.view, device.input, ldx, device.output, ldy, options.rows, device.scratch,
            strata::kernels::exl3::exl3_gemm_scratch_elements(device.view, options.rows), gemv_options, {}, &profile);
        event.wait_and_throw();
        host_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - host_begin).count();
        const double in = event_ms(profile.input_hadamard), dot = event_ms(profile.dot), out = event_ms(profile.output_hadamard);
        if (in >= 0.0 && dot >= 0.0 && out >= 0.0) {
            input_ms += in; dot_ms += dot; output_ms += out; total_ms += in + dot + out; ++profiled;
        }
    }
    std::printf("%s k=%lld n=%lld bits=%d cb=%d rows=%d ldx=%lld ldy=%lld warmup=%d repeats=%d\n",
                label.c_str(), (long long) device.view.k, (long long) device.view.n, device.view.bits,
                device.view.codebook, options.rows, (long long) ldx, (long long) ldy, options.warmup, options.repeats);
    if (profiled > 0)
        std::printf("  GPU event ms/call: Hpre %.6f dot %.6f Hpost %.6f total-stages %.6f (profiled %d)\n",
                    input_ms / profiled, dot_ms / profiled, output_ms / profiled, total_ms / profiled, profiled);
    std::printf("  host submit+wait ms/call: %.6f\n", host_wait_ms / options.repeats);
    return profiled == options.repeats;
}

bool run_one(const strata::artifact::exl3::Archive& archive, const std::string& name, const Options& options,
             sycl::queue& queue) {
    const auto* descriptor = archive.find_linear(name);
    if (descriptor == nullptr) {
        std::fprintf(stderr, "missing linear: %s\n", name.c_str());
        return false;
    }
    strata::artifact::exl3::MaterializedMatrix materialized;
    try {
        archive.materialize(*descriptor, materialized);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "materialize %s: %s\n", name.c_str(), e.what());
        return false;
    }
    return profile_matrix(name, materialized, options, queue);
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        usage();
        return 2;
    }
    try {
        const strata::artifact::exl3::Archive archive(options.archive);
        auto& runtime = strata::core::Runtime::get();
        sycl::queue queue(runtime.context(), runtime.device(),
                          sycl::property_list{sycl::property::queue::in_order{},
                                              sycl::property::queue::enable_profiling{}});
        bool ok = true;
        ok = run_one(archive, options.head, options, queue) && ok;
        ok = run_one(archive, options.gate, options, queue) && ok;
        ok = run_one(archive, options.up, options, queue) && ok;
        ok = run_one(archive, options.down, options, queue) && ok;
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exl3_bench: %s\n", e.what());
        return 1;
    }
}
