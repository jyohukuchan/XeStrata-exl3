// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Compare wide compressed-word products against the preserved scalar order.
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "parity_device.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <stdexcept>

namespace {
void select_path(bool enabled) {
#if defined(_WIN32)
    if (_putenv_s("STRATA_EXL3_WIDE_WORD_TILE", enabled ? "1" : "0") != 0)
#else
    if (setenv("STRATA_EXL3_WIDE_WORD_TILE", enabled ? "1" : "0", 1) != 0)
#endif
        throw std::runtime_error("cannot select wide word path");
}
}

int main() {
    try {
        auto& q = strata::core::Runtime::get().compute();
        const auto device = q.get_device();
        const auto sizes = device.get_info<sycl::info::device::sub_group_sizes>();
        if (std::find(sizes.begin(), sizes.end(), size_t(32)) == sizes.end()) return 77;
        const size_t waves = std::max<size_t>(1, std::min(
            device.get_info<sycl::info::device::max_work_group_size>() / 32,
            device.get_info<sycl::info::device::local_mem_size>() / (16 * sizeof(float))));
        constexpr int64_t k = 128, rows = 3, ldx = k + 7;
        const int64_t n = static_cast<int64_t>((k / 16 + 1) * waves * 16), ldy = n + 11;
        constexpr float sentinel = 27182.75f;
        size_t failures = 0;
        for (int bits : {2, 6, 8}) {
            const size_t count = static_cast<size_t>(k / 16) * static_cast<size_t>(n / 16) * (16 * bits);
            std::vector<uint16_t> packed(count + 2, 0xa55a), suh(k), svh(n), bias(n);
            for (size_t i = 0; i < count; ++i) packed[i + 1] = static_cast<uint16_t>((i * 747796405u) ^ (i >> 3));
            for (int64_t i = 0; i < k; ++i) suh[i] = strata::kernels::f16_from_f32(0.75f + float(i % 13) * 0.015f);
            for (int64_t i = 0; i < n; ++i) {
                svh[i] = strata::kernels::f16_from_f32(0.85f + float(i % 11) * 0.011f);
                bias[i] = strata::kernels::f16_from_f32(float(i % 17 - 8) * 0.00031f);
            }
            std::vector<float> input(2 + rows * ldx, -31337.25f);
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t i = 0; i < k; ++i) input[1 + r * ldx + i] = float((i * 19 + r * 7) % 131 - 65) / 1024;
            strata::parity::Device memory;
            const auto* dp = memory.upload(packed);
            const strata::kernels::exl3::MatrixView view{dp + 1, memory.upload(suh), memory.upload(svh),
                                                       memory.upload(bias), k, n, bits, 2};
            const auto* dx = memory.upload(input);
            for (int mode : {0, 3}) {
                const strata::kernels::exl3::GemvOptions opt{(mode & 1) != 0, (mode & 2) != 0, true};
                const size_t needed = strata::kernels::exl3::exl3_gemm_scratch_elements(view, rows, opt);
                std::vector<float> reference;
                for (bool enabled : {false, true}) {
                    select_path(enabled);
                    std::vector<float> output(2 + rows * ldy, sentinel), scratch(needed + 9, sentinel);
                    auto* dy = memory.upload(output); auto* ds = memory.upload(scratch);
                    strata::kernels::exl3::exl3_gemm_fp32(q, view, dx + 1, ldx, dy + 1, ldy, rows, ds, needed, opt).wait_and_throw();
                    strata::parity::copy_out(output.data(), dy, output.size());
                    strata::parity::copy_out(scratch.data(), ds, scratch.size());
                    for (size_t i = needed; i < scratch.size(); ++i) if (scratch[i] != sentinel) ++failures;
                    if (output.front() != sentinel || output.back() != sentinel) ++failures;
                    for (int64_t r = 0; r < rows; ++r) {
                        for (int64_t c = 0; c < n; ++c) if (!std::isfinite(output[1 + r * ldy + c])) ++failures;
                        for (int64_t c = n; c < ldy; ++c) if (output[1 + r * ldy + c] != sentinel) ++failures;
                    }
                    if (!enabled) reference = output;
                    else if (std::memcmp(reference.data(), output.data(), output.size() * sizeof(float)) != 0) ++failures;
                }
            }
        }
        std::printf("exl3_wide_parity: %zu failures\n", failures);
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "exl3_wide_parity: %s\n", e.what());
        return 1;
    }
}
