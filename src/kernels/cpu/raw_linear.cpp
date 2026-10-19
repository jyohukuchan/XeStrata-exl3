// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Scalar raw F16/BF16/F32 linear reference path.
#include "strata/kernels/cpu/raw_linear.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace strata::kernels::cpu::raw {
namespace {

using strata::kernels::raw::DType;
using strata::kernels::raw::LinearView;
using strata::kernels::cpu::exl3::Status;

Status validate(const LinearView& view, const float* x, float* y, int64_t ldx, int64_t ldy, int64_t rows) {
    if (view.data == nullptr || x == nullptr || y == nullptr) return Status::missing_data;
    if (view.k <= 0 || view.n <= 0 || rows <= 0 || ldx < view.k || ldy < view.n)
        return Status::invalid_argument;
    if (view.dtype != DType::F16 && view.dtype != DType::BF16 && view.dtype != DType::F32)
        return Status::unsupported_layout;
    const uint64_t max_bytes = (uint64_t) std::numeric_limits<ptrdiff_t>::max();
    if ((uint64_t) view.n > std::numeric_limits<uint64_t>::max() / (uint64_t) view.k)
        return Status::size_overflow;
    const uint64_t weight_count = (uint64_t) view.k * (uint64_t) view.n;
    if (weight_count > max_bytes / (uint64_t) strata::kernels::raw::dtype_bytes(view.dtype) ||
        (uint64_t) rows > max_bytes / sizeof(float) / (uint64_t) ldx ||
        (uint64_t) rows > max_bytes / sizeof(float) / (uint64_t) ldy ||
        (view.bias != nullptr && (uint64_t) view.n > max_bytes / sizeof(float)))
        return Status::size_overflow;
    return Status::ok;
}

float weight_at(const LinearView& view, size_t index) {
    if (view.dtype == DType::F16)
        return strata::kernels::f32_from_f16(static_cast<const uint16_t*>(view.data)[index]);
    if (view.dtype == DType::BF16)
        return strata::kernels::f32_from_bf16(static_cast<const uint16_t*>(view.data)[index]);
    return static_cast<const float*>(view.data)[index];
}

}  // namespace

Status linear_fp32(const strata::kernels::raw::LinearView& view, const float* x, int64_t ldx, float* y, int64_t ldy,
                   int64_t rows) {
    const Status valid = validate(view, x, y, ldx, ldy, rows);
    if (valid != Status::ok) return valid;
    for (int64_t row = 0; row < rows; ++row) {
        const float* xr = x + row * ldx;
        float* yr = y + row * ldy;
        for (int64_t out = 0; out < view.n; ++out) {
            double acc = 0.0;
            for (int64_t in = 0; in < view.k; ++in)
                acc += (double) xr[in] * (double) weight_at(view, (size_t) out * (size_t) view.k + (size_t) in);
            float value = (float) acc;
            if (view.bias != nullptr) value += view.bias[out];
            yr[out] = value;
        }
    }
    return Status::ok;
}

}  // namespace strata::kernels::cpu::raw
