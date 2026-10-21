// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Raw F16/BF16/F32 linear weights are kept in their source dtype.
#pragma once

#include <cstdint>

namespace strata::kernels::raw {

enum class DType : uint8_t {
    F16 = 0,
    BF16 = 1,
    F32 = 2,
};

/// Non-quantized row-major weight view.  data has n rows of k values: data[o*k+i].
/// Optional bias is always FP32 and has n values.
struct LinearView {
    const void* data = nullptr;
    const float* bias = nullptr;
    int64_t k = 0;
    int64_t n = 0;
    DType dtype = DType::F32;
    /// The F16 weights hold normal halves and signed zeros only, so the device's half to float
    /// conversion is exact for every one of them.  `plain_f16_halves` establishes it.
    bool plain_halves = false;
};

constexpr int dtype_bytes(DType dtype) {
    return dtype == DType::F32 ? 4 : (dtype == DType::F16 || dtype == DType::BF16 ? 2 : 0);
}

}  // namespace strata::kernels::raw
