// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Bounded experimental CPU MUL1 integer dot helpers, derived from exllamav3.
#pragma once

#include "strata/kernels/cpu/exl3.hpp"

namespace strata::kernels::cpu::exl3 {

struct Mul1Input {
    const int32_t* duplicated = nullptr;  // int8 activation in both signed i16 slots
    size_t count = 0;
    float q = 1.0f;
    int32_t sum = 0;
    bool ready = false;
};

// Explicit prototype opt-in plus the existing CPUID/XCR0 capability check.
bool mul1_intdot_enabled();

// Quantize an already transformed FP32 input using one Q8 scale per K vector.
// A false result leaves `prepared` disabled; the caller keeps the exact path.
bool prepare_mul1_input(const strata::kernels::exl3::MatrixView& matrix, const float* input_basis,
                        int32_t* duplicated, size_t capacity, Mul1Input& prepared);

// Only dot-space output is written. Hpost, svh, bias remain in the existing
// caller. Decode uses register permutes and integer multiply-add, no full W.
bool linear_mul1_dot_range(const strata::kernels::exl3::MatrixView& matrix, const Mul1Input& prepared,
                           size_t first_tile, size_t tile_count, float* dot);

}  // namespace strata::kernels::cpu::exl3
