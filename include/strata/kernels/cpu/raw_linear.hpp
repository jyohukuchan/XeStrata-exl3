// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "strata/kernels/cpu/exl3.hpp"
#include "strata/kernels/raw_linear_format.hpp"

#include <cstdint>

namespace strata::kernels::cpu::raw {

strata::kernels::cpu::exl3::Status linear_fp32(const strata::kernels::raw::LinearView& view, const float* x,
                                                int64_t ldx, float* y, int64_t ldy, int64_t rows);

}  // namespace strata::kernels::cpu::raw
