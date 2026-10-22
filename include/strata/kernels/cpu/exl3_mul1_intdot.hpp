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
    // The integer partial sums of a K split, when the dot of this tile range
    // was computed by `linear_mul1_accumulate_range` over K ranges instead of
    // in one walk over the trellis.  `partials` points at the first range's
    // sums; the range `c` starts `partial_stride` values further on, and each
    // range holds `partial_tiles * 16` values (one per output of the range).
    // The sums are exact integers, so the combined dot does not depend on how
    // many ranges the K was cut into.
    const int32_t* partials = nullptr;
    int partial_count = 0;
    size_t partial_stride = 0;
    size_t partial_first_tile = 0;
    size_t partial_tiles = 0;
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

// The same dot, cut along K: the sums of the K range [k_tile_begin,
// k_tile_begin + k_tile_count) of the tile range [first_tile, first_tile +
// tile_count) are written to `acc` as exact integers, one per output.  The
// caller adds the ranges up and converts them once, which gives the same
// value as the unsplit dot: the sums are integers, so the order of the
// addition does not change the result.
Status linear_mul1_accumulate_range(const strata::kernels::exl3::MatrixView& matrix, const Mul1Input& prepared,
                                    size_t first_tile, size_t tile_count, size_t k_tile_begin,
                                    size_t k_tile_count, int32_t* acc);

// Whether the integer dot path can read this projection: the codebook, the
// bit width and the geometry the path accepts, and the explicit opt-in.
bool linear_mul1_supported(const strata::kernels::exl3::MatrixView& matrix);

}  // namespace strata::kernels::cpu::exl3
