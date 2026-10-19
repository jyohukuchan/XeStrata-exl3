// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "strata/kernels/raw_linear_format.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

namespace strata::kernels::raw {

/// Scalar correctness fallback for FP32 x FP16/BF16/F32 row-major linear weights.
/// The source weight dtype is read directly; no expanded copy is made.
sycl::event linear_fp32(sycl::queue& queue, const LinearView& view, const float* x, int64_t ldx, float* y,
                        int64_t ldy, int64_t rows, const std::vector<sycl::event>& deps = {});

}  // namespace strata::kernels::raw
