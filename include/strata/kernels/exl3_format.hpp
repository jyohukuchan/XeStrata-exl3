// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 packed tiles follow exllamav3's MIT-licensed trellis storage rules.
#pragma once

#include <cstdint>

namespace strata::kernels::exl3 {

/// All half-valued buffers contain IEEE binary16 bits. The pointers may refer to
/// host memory (CPU routines) or device memory (SYCL routines), respectively.
/// Packed storage is [k / 16, n / 16, 16 * bits], in uint16_t elements.
struct MatrixView {
    const uint16_t* trellis = nullptr;
    const uint16_t* suh = nullptr;
    const uint16_t* svh = nullptr;
    const uint16_t* bias = nullptr;
    int64_t k = 0;
    int64_t n = 0;
    int bits = 0;
    int codebook = 0;  // 0: legacy/default, 1: mcg, 2: mul1
};

/// Matrix offset inside a 16x16 tile for one of its 256 trellis states.
constexpr int tile_element_for_state(int state) {
    const int lane = state / 8;
    const int part = state % 8;
    const int r = (lane % 4) * 2 + (part % 4) / 2 * 8 + part % 2;
    const int c = lane / 4 + part / 4 * 8;
    return r * 16 + c;
}

constexpr int tile_state_for_element(int element) {
    const int r = element / 16;
    const int c = element % 16;
    return ((c % 8) * 4 + (r % 8) / 2) * 8 + c / 8 * 4 + r / 8 * 2 + r % 2;
}

}  // namespace strata::kernels::exl3
