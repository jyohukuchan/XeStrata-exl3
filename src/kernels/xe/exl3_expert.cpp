// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 packed tiles follow exllamav3's MIT-licensed trellis storage rules.
// src/kernels/xe/exl3_expert.cpp - correctness-first packed expert composition.
#include "strata/kernels/exl3_expert.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_exl3_expert
#include "strata/kernels/device_arch_tag.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels::exl3 {
namespace {

uint64_t max_float_elements() {
    return (uint64_t) std::numeric_limits<ptrdiff_t>::max() / sizeof(float);
}

void check_projection(const MatrixView& view, const char* name) {
    if (view.trellis == nullptr) throw std::invalid_argument(std::string(name) + ": trellis is null");
    if (view.k <= 0 || view.n <= 0 || view.k % 16 != 0 || view.n % 16 != 0)
        throw std::invalid_argument(std::string(name) + ": dimensions must be positive multiples of 16");
    if (view.bits < 1 || view.bits > 8)
        throw std::invalid_argument(std::string(name) + ": bits must be in 1..8");
    if (view.codebook < 0 || view.codebook > 2)
        throw std::invalid_argument(std::string(name) + ": codebook must be 0, 1 or 2");
}

void check_expert(const ExpertView& view, const float* x, float* y, int64_t ldx, int64_t ldy, int64_t rows,
                  const float* scratch, size_t scratch_elements, ExpertOptions options) {
    check_projection(view.gate, "exl3_expert_fp32 gate");
    check_projection(view.up, "exl3_expert_fp32 up");
    check_projection(view.down, "exl3_expert_fp32 down");
    if (view.gate.k != view.up.k || view.gate.n != view.up.n || view.down.k != view.gate.n ||
        view.down.n != view.gate.k)
        throw std::invalid_argument("exl3_expert_fp32: gate/up/down shapes do not form hidden-ff-hidden");
    if (x == nullptr || y == nullptr) throw std::invalid_argument("exl3_expert_fp32: x and y are required");
    if (rows <= 0) throw std::invalid_argument("exl3_expert_fp32: rows must be positive");
    if (ldx < view.gate.k || ldy < view.gate.k)
        throw std::invalid_argument("exl3_expert_fp32: leading dimensions are smaller than hidden size");
    const uint64_t max_elements = max_float_elements();
    if ((uint64_t) ldx > max_elements / (uint64_t) rows ||
        (uint64_t) ldy > max_elements / (uint64_t) rows)
        throw std::invalid_argument("exl3_expert_fp32: input/output span exceeds host address range");
    const auto check_transform = [](const MatrixView& projection, GemvOptions projection_options, const char* name) {
        if (projection_options.input_hadamard && projection.suh != nullptr && projection.k % 128 != 0)
            throw std::invalid_argument(std::string(name) + ": k must be divisible by 128 for input H128");
        if (projection_options.output_hadamard && projection.svh != nullptr && projection.n % 128 != 0)
            throw std::invalid_argument(std::string(name) + ": n must be divisible by 128 for output H128");
    };
    check_transform(view.gate, options.gate, "exl3_expert_fp32 gate");
    check_transform(view.up, options.up, "exl3_expert_fp32 up");
    check_transform(view.down, options.down, "exl3_expert_fp32 down");
    const size_t needed = exl3_expert_scratch_elements(view, rows, options);
    if (scratch == nullptr || scratch_elements < needed)
        throw std::invalid_argument("exl3_expert_fp32: caller scratch is too small");
}

sycl::event submit_silu_mul(sycl::queue& queue, float* gate, const float* up, int64_t rows, int64_t ff,
                            const std::vector<sycl::event>& deps) {
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::range<1>((size_t) rows * (size_t) ff), [=](sycl::id<1> id) {
            const size_t i = (size_t) id[0];
            const float g = gate[i];
            gate[i] = (g / (1.0f + sycl::exp(-g))) * up[i];
        });
    });
}

}  // namespace

size_t exl3_expert_scratch_elements(const ExpertView& view, int64_t rows, ExpertOptions options) {
    if (rows <= 0) throw std::invalid_argument("exl3_expert_fp32: rows must be positive");
    check_projection(view.gate, "exl3_expert_fp32 gate");
    check_projection(view.up, "exl3_expert_fp32 up");
    check_projection(view.down, "exl3_expert_fp32 down");
    if (view.gate.k != view.up.k || view.gate.n != view.up.n || view.down.k != view.gate.n ||
        view.down.n != view.gate.k)
        throw std::invalid_argument("exl3_expert_fp32: gate/up/down shapes do not form hidden-ff-hidden");
    const size_t gate = exl3_gemm_scratch_elements(view.gate, rows, options.gate);
    const size_t up = exl3_gemm_scratch_elements(view.up, rows, options.up);
    const size_t down = exl3_gemm_scratch_elements(view.down, rows, options.down);
    const uint64_t ff = (uint64_t) view.gate.n;
    const uint64_t row_count = (uint64_t) rows;
    if (ff > std::numeric_limits<uint64_t>::max() / row_count ||
        row_count * ff > std::numeric_limits<uint64_t>::max() / 2)
        throw std::invalid_argument("exl3_expert_fp32: scratch size overflows uint64");
    const uint64_t intermediates = row_count * ff * 2;
    uint64_t total = (uint64_t) gate;
    if ((uint64_t) up > std::numeric_limits<uint64_t>::max() - total)
        throw std::invalid_argument("exl3_expert_fp32: scratch size overflows uint64");
    total += (uint64_t) up;
    if ((uint64_t) down > std::numeric_limits<uint64_t>::max() - total)
        throw std::invalid_argument("exl3_expert_fp32: scratch size overflows uint64");
    total += (uint64_t) down;
    if (intermediates > std::numeric_limits<uint64_t>::max() - total)
        throw std::invalid_argument("exl3_expert_fp32: scratch size overflows uint64");
    total += intermediates;
    if (total > (uint64_t) std::numeric_limits<size_t>::max())
        throw std::invalid_argument("exl3_expert_fp32: scratch size overflows size_t");
    if (total > max_float_elements())
        throw std::invalid_argument("exl3_expert_fp32: scratch size exceeds host address range");
    return (size_t) total;
}

sycl::event exl3_expert_fp32(sycl::queue& queue, const ExpertView& view, const float* x, int64_t ldx, float* y,
                             int64_t ldy, int64_t rows, float* scratch, size_t scratch_elements,
                             ExpertOptions options, const std::vector<sycl::event>& deps) {
    check_expert(view, x, y, ldx, ldy, rows, scratch, scratch_elements, options);

    const size_t gate_scratch_size = exl3_gemm_scratch_elements(view.gate, rows, options.gate);
    const size_t up_scratch_size = exl3_gemm_scratch_elements(view.up, rows, options.up);
    const size_t down_scratch_size = exl3_gemm_scratch_elements(view.down, rows, options.down);
    size_t offset = 0;
    float* gate_scratch = scratch + offset;
    offset += gate_scratch_size;
    float* up_scratch = scratch + offset;
    offset += up_scratch_size;
    float* down_scratch = scratch + offset;
    offset += down_scratch_size;
    float* gate_out = scratch + offset;
    offset += (size_t) rows * (size_t) view.gate.n;
    float* up_out = scratch + offset;

    const sycl::event gate_event = exl3_gemm_fp32(queue, view.gate, x, ldx, gate_out, view.gate.n, rows,
                                                  gate_scratch, gate_scratch_size, options.gate, deps);
    const sycl::event up_event = exl3_gemm_fp32(queue, view.up, x, ldx, up_out, view.up.n, rows, up_scratch,
                                                up_scratch_size, options.up, deps);
    const sycl::event activation_event = submit_silu_mul(queue, gate_out, up_out, rows, view.gate.n,
                                                         {gate_event, up_event});
    return exl3_gemm_fp32(queue, view.down, gate_out, view.gate.n, y, ldy, rows, down_scratch, down_scratch_size,
                          options.down, {activation_event});
}

}  // namespace strata::kernels::exl3
