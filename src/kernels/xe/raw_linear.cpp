// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Scalar SYCL correctness fallback for raw F16/BF16/F32 linear weights.
#include "strata/kernels/raw_linear.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_raw_linear
#include "strata/kernels/device_arch_tag.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels::raw {
namespace {

void validate(const LinearView& view, const float* x, float* y, int64_t ldx, int64_t ldy, int64_t rows) {
    if (view.data == nullptr || x == nullptr || y == nullptr)
        throw std::invalid_argument("raw::linear_fp32: data, x and y are required");
    if (view.k <= 0 || view.n <= 0 || rows <= 0 || ldx < view.k || ldy < view.n)
        throw std::invalid_argument("raw::linear_fp32: invalid dimensions or leading dimensions");
    if (view.dtype != DType::F16 && view.dtype != DType::BF16 && view.dtype != DType::F32)
        throw std::invalid_argument("raw::linear_fp32: unsupported dtype");
    const uint64_t max_bytes = (uint64_t) std::numeric_limits<ptrdiff_t>::max();
    const uint64_t max_weight_count = max_bytes / (uint64_t) dtype_bytes(view.dtype);
    if ((uint64_t) view.n > std::numeric_limits<uint64_t>::max() / (uint64_t) view.k ||
        (uint64_t) view.k * (uint64_t) view.n > max_weight_count ||
        (uint64_t) rows > max_bytes / sizeof(float) / (uint64_t) ldx ||
        (uint64_t) rows > max_bytes / sizeof(float) / (uint64_t) ldy ||
        (view.bias != nullptr && (uint64_t) view.n > max_bytes / sizeof(float)))
        throw std::invalid_argument("raw::linear_fp32: caller span exceeds host address range");
}

}  // namespace

sycl::event linear_fp32(sycl::queue& queue, const LinearView& view, const float* x, int64_t ldx, float* y,
                        int64_t ldy, int64_t rows, const std::vector<sycl::event>& deps) {
    validate(view, x, y, ldx, ldy, rows);
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::range<1>((size_t) rows * (size_t) view.n), [=](sycl::id<1> id) {
            const size_t flat = (size_t) id[0];
            const int64_t row = (int64_t) (flat / (size_t) view.n);
            const int64_t out = (int64_t) (flat % (size_t) view.n);
            const float* xr = x + row * ldx;
            float value = 0.0f;
            const size_t base = (size_t) out * (size_t) view.k;
            for (int64_t in = 0; in < view.k; ++in) {
                float weight;
                if (view.dtype == DType::F16)
                    weight = f32_from_f16(static_cast<const uint16_t*>(view.data)[base + (size_t) in]);
                else if (view.dtype == DType::BF16)
                    weight = f32_from_bf16(static_cast<const uint16_t*>(view.data)[base + (size_t) in]);
                else
                    weight = static_cast<const float*>(view.data)[base + (size_t) in];
                value += xr[in] * weight;
            }
            if (view.bias != nullptr) value += view.bias[out];
            y[row * ldy + out] = value;
        });
    });
}

}  // namespace strata::kernels::raw
