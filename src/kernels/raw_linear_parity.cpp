// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Parity for the neutral raw F16/BF16/F32 linear fallback.
#include "strata/kernels/raw_linear.hpp"
#include "strata/kernels/cpu/raw_linear.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

using strata::kernels::raw::DType;
using strata::kernels::raw::LinearView;
using strata::kernels::f32_from_bf16;
using strata::kernels::f32_from_f16;
using strata::kernels::f16_from_f32;
using strata::kernels::bf16_from_f32;

float host_weight(const LinearView& view, size_t index) {
    if (view.dtype == DType::F16) return f32_from_f16(static_cast<const uint16_t*>(view.data)[index]);
    if (view.dtype == DType::BF16) return f32_from_bf16(static_cast<const uint16_t*>(view.data)[index]);
    return static_cast<const float*>(view.data)[index];
}

void oracle(const LinearView& view, const std::vector<float>& x, int64_t ldx, std::vector<float>& y, int64_t ldy,
            int64_t rows) {
    for (int64_t row = 0; row < rows; ++row)
        for (int64_t out = 0; out < view.n; ++out) {
            double acc = 0.0;
            for (int64_t in = 0; in < view.k; ++in)
                acc += (double) x[(size_t) row * ldx + in] *
                       (double) host_weight(view, (size_t) out * (size_t) view.k + in);
            float value = (float) acc;
            if (view.bias != nullptr) value += view.bias[out];
            y[(size_t) row * ldy + out] = value;
        }
}

bool close(float got, float want) {
    if (std::isnan(got) || std::isnan(want)) return std::isnan(got) && std::isnan(want);
    if (std::isinf(got) || std::isinf(want)) return got == want;
    return std::fabs(got - want) <= 2e-4f * std::max(1.0f, std::fabs(want));
}

const char* name(DType dtype) {
    return dtype == DType::F16 ? "F16" : dtype == DType::BF16 ? "BF16" : "F32";
}

}  // namespace

int main() {
    constexpr int64_t k = 129, n = 137;
    constexpr int64_t ldx = k + 7, ldy = n + 11;
    constexpr float padding = -777.25f;
    const DType dtypes[] = {DType::F16, DType::BF16, DType::F32};
    auto& queue = strata::core::Runtime::get().compute();
    int failures = 0;

    for (const DType dtype : dtypes) {
        std::vector<uint16_t> w16(dtype == DType::F32 ? 0 : (size_t) n * k);
        std::vector<float> wf32(dtype == DType::F32 ? (size_t) n * k : 0);
        for (int64_t out = 0; out < n; ++out)
            for (int64_t in = 0; in < k; ++in) {
                const float value = 0.01f * (float) (((out * 17 + in * 5) % 101) - 50);
                const size_t index = (size_t) out * k + in;
                if (dtype == DType::F16) w16[index] = f16_from_f32(value);
                else if (dtype == DType::BF16) w16[index] = bf16_from_f32(value);
                else wf32[index] = value;
            }
        std::vector<float> bias(n);
        for (int64_t out = 0; out < n; ++out) bias[out] = 0.003f * (float) ((out % 17) - 8);
        LinearView host_view{dtype == DType::F32 ? static_cast<const void*>(wf32.data())
                                                 : static_cast<const void*>(w16.data()),
                             bias.data(), k, n, dtype};
        for (const int64_t rows : {int64_t(1), int64_t(4)}) {
            std::vector<float> x((size_t) rows * ldx, padding);
            for (int64_t row = 0; row < rows; ++row)
                for (int64_t in = 0; in < k; ++in)
                    x[(size_t) row * ldx + in] = 0.01f * (float) (((row * 19 + in * 7) % 79) - 39);
            std::vector<float> expected((size_t) rows * ldy, padding);
            std::vector<float> cpu_out = expected;
            oracle(host_view, x, ldx, expected, ldy, rows);
            const auto cpu_status = strata::kernels::cpu::raw::linear_fp32(
                host_view, x.data(), ldx, cpu_out.data(), ldy, rows);
            strata::parity::Device device;
            float* d_x = device.upload(x);
            float* d_bias = device.upload(bias);
            uint16_t* d_w16 = nullptr;
            float* d_wf32 = nullptr;
            if (dtype == DType::F32) d_wf32 = device.upload(wf32);
            else d_w16 = device.upload(w16);
            float* d_y = device.upload(std::vector<float>((size_t) rows * ldy, padding));
            LinearView device_view{dtype == DType::F32 ? static_cast<const void*>(d_wf32)
                                                       : static_cast<const void*>(d_w16),
                                   d_bias, k, n, dtype};
            sycl::event ready = queue.ext_oneapi_submit_barrier();
            sycl::event event = strata::kernels::raw::linear_fp32(
                queue, device_view, d_x, ldx, d_y, ldy, rows, {ready});
            event.wait_and_throw();
            std::vector<float> gpu_out((size_t) rows * ldy);
            strata::parity::copy_out(gpu_out.data(), d_y, gpu_out.size());
            size_t bad_cpu = cpu_status == strata::kernels::cpu::exl3::Status::ok ? 0 : 1;
            size_t bad_gpu = 0, bad_padding = 0;
            for (int64_t row = 0; row < rows; ++row) {
                for (int64_t out = 0; out < n; ++out) {
                    if (!close(cpu_out[(size_t) row * ldy + out], expected[(size_t) row * ldy + out])) ++bad_cpu;
                    if (!close(gpu_out[(size_t) row * ldy + out], expected[(size_t) row * ldy + out])) ++bad_gpu;
                }
                for (int64_t out = n; out < ldy; ++out)
                    if (gpu_out[(size_t) row * ldy + out] != padding) ++bad_padding;
            }
            std::printf("%s rows %lld [%lld,%lld]: CPU %s (%zu bad), GPU %s (%zu bad), padding %s (%zu bad)\n",
                        name(dtype), (long long) rows, (long long) k, (long long) n,
                        bad_cpu == 0 ? "OK" : "FAIL", bad_cpu, bad_gpu == 0 ? "OK" : "FAIL", bad_gpu,
                        bad_padding == 0 ? "OK" : "FAIL", bad_padding);
            if (bad_cpu != 0 || bad_gpu != 0 || bad_padding != 0) ++failures;
        }

        if (dtype == DType::F16) {
            // Include normal/subnormal boundaries, signed zeros and nonfinite weights in one GPU case.
            const uint16_t edges[] = {0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xbc00,
                                      0x7bff, 0x7c00, 0xfc00, 0x7e00, 0x7c01};
            constexpr size_t edge_count = sizeof(edges) / sizeof(edges[0]);
            std::vector<uint16_t> weights(edge_count * static_cast<size_t>(k), 0);
            std::vector<float> input(static_cast<size_t>(k), 0.0f), output(edge_count), expected(edge_count);
            input[0] = 1.0f;
            for (size_t i = 0; i < edge_count; ++i) { weights[i * static_cast<size_t>(k)] = edges[i]; expected[i] = f32_from_f16(edges[i]); }
            strata::parity::Device device;
            const LinearView v{device.upload(weights), nullptr, k, static_cast<int64_t>(edge_count), DType::F16};
            auto* dy = device.alloc<float>(edge_count);
            strata::kernels::raw::linear_fp32(queue, v, device.upload(input), k, dy,
                                             static_cast<int64_t>(edge_count), 1).wait_and_throw();
            strata::parity::copy_out(output.data(), dy, edge_count);
            size_t bad = 0;
            for (size_t i = 0; i < edge_count; ++i) {
                const bool same = std::isnan(expected[i]) ? std::isnan(output[i]) :
                    expected[i] == 0.0f ? output[i] == 0.0f :
                    std::memcmp(&output[i], &expected[i], sizeof(float)) == 0;
                if (!same) ++bad;
            }
            std::printf("F16 GPU edge weights: %s (%zu bad)\n", bad ? "FAIL" : "OK", bad);
            if (bad) ++failures;
        }

        if (dtype == DType::F16) {
            // The plain path hands every half to the device's conversion and the exact path
            // builds the value from its bits.  On a tensor of normal halves the two must agree
            // bit for bit, and the scan must call such a tensor plain.
            strata::parity::Device device;
            const size_t count = static_cast<size_t>(n) * static_cast<size_t>(k);
            uint16_t* d_w16 = device.upload(w16);
            std::vector<float> plain_input(static_cast<size_t>(k), 0.25f);
            float* d_plain = device.alloc<float>(static_cast<size_t>(n));
            float* d_exact = device.alloc<float>(static_cast<size_t>(n));
            LinearView plain_view{d_w16, nullptr, k, n, DType::F16};
            plain_view.plain_halves = true;
            LinearView exact_view{d_w16, nullptr, k, n, DType::F16};
            const bool scanned_plain = strata::kernels::raw::plain_f16_halves(queue, d_w16, count * sizeof(uint16_t));
            strata::kernels::raw::linear_fp32(queue, plain_view, device.upload(plain_input), k, d_plain, n, 1)
                .wait_and_throw();
            strata::kernels::raw::linear_fp32(queue, exact_view, device.upload(plain_input), k, d_exact, n, 1)
                .wait_and_throw();
            std::vector<float> plain_out(static_cast<size_t>(n)), exact_out(static_cast<size_t>(n));
            strata::parity::copy_out(plain_out.data(), d_plain, plain_out.size());
            strata::parity::copy_out(exact_out.data(), d_exact, exact_out.size());
            size_t bad = 0;
            for (size_t i = 0; i < plain_out.size(); ++i)
                if (std::memcmp(&plain_out[i], &exact_out[i], sizeof(float)) != 0) ++bad;
            const bool plain_ok = scanned_plain && bad == 0;
            std::printf("F16 plain path: %s (scanned plain %d, %zu differing)\n", plain_ok ? "OK" : "FAIL",
                        (int) scanned_plain, bad);
            if (!plain_ok) ++failures;

            // A subnormal half is not plain: the scan must say so, and the exact path must still
            // return its value.
            std::vector<uint16_t> subnormal = w16;
            subnormal[0] = 0x0001;
            uint16_t* d_sub = device.upload(subnormal);
            const bool scanned_sub = strata::kernels::raw::plain_f16_halves(queue, d_sub, count * sizeof(uint16_t));
            std::vector<float> sub_input(static_cast<size_t>(k), 0.0f), sub_out(static_cast<size_t>(n));
            sub_input[0] = 1.0f;
            LinearView sub_view{d_sub, nullptr, k, n, DType::F16};
            strata::kernels::raw::linear_fp32(queue, sub_view, device.upload(sub_input), k, d_exact, n, 1).wait_and_throw();
            strata::parity::copy_out(sub_out.data(), d_exact, sub_out.size());
            const bool sub_ok = !scanned_sub && sub_out[0] == f32_from_f16(0x0001);
            std::printf("F16 subnormal weight: %s (scanned plain %d)\n", sub_ok ? "OK" : "FAIL", (int) scanned_sub);
            if (!sub_ok) ++failures;
        }

        // NaN/Inf are ordinary floating-point weights: no special clamping or filtering is allowed.
        if (dtype == DType::F16) {
            w16[0] = 0x7E00;
            w16[(size_t) k] = 0x7C00;
        } else if (dtype == DType::BF16) {
            w16[0] = 0x7FC1;
            w16[(size_t) k] = 0x7F80;
        } else {
            wf32[0] = std::numeric_limits<float>::quiet_NaN();
            wf32[(size_t) k] = std::numeric_limits<float>::infinity();
        }
        LinearView special{dtype == DType::F32 ? static_cast<const void*>(wf32.data())
                                               : static_cast<const void*>(w16.data()),
                           bias.data(), k, n, dtype};
        std::vector<float> sx((size_t) ldx, 0.0f);
        sx[0] = 1.0f;
        std::vector<float> sy((size_t) ldy, padding), se = sy;
        oracle(special, sx, ldx, se, ldy, 1);
        strata::kernels::cpu::raw::linear_fp32(special, sx.data(), ldx, sy.data(), ldy, 1);
        const bool special_ok = std::isnan(sy[0]) && std::isinf(sy[1]) &&
                                std::isnan(se[0]) && std::isinf(se[1]);
        std::printf("%s NaN/Inf weights: %s\n", name(dtype), special_ok ? "OK" : "FAIL");
        if (!special_ok) ++failures;
    }

    std::printf("raw_linear_parity: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
