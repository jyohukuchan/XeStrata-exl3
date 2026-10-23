// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The 128-point normalized Hadamard transforms around an EXL3 dot, shared by the grouped expert
// pipeline and the fused transform tails.  A 128-wide transform uses one SG32 and four registers
// per lane: the first five butterfly stages stay in the SG, the two remaining bits are butterflies
// between the four registers.  The input form scales each element by its `suh` entry before the
// butterfly, the output form applies `scale * svh` and the bias after it - the order the
// standalone transform kernels (`exl3.cpp`'s `submit_hadamard_coop`) use, so a fused tail and the
// kernels it replaces produce the same values.  Keep the two copies in step.
#pragma once

#include "strata/kernels/exl3_format.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <sycl/sycl.hpp>

namespace strata::kernels::exl3::had {

constexpr float kScale = 0.08838834764831845f;  // 1 / sqrt(128), as in exllamav3's reconstruct path

inline void fwht128_registers(const sycl::sub_group& sg, int lane, float& a0, float& a1, float& a2, float& a3) {
    for (int step = 1; step < 32; step <<= 1) {
        const float p0 = sycl::permute_group_by_xor(sg, a0, step);
        const float p1 = sycl::permute_group_by_xor(sg, a1, step);
        const float p2 = sycl::permute_group_by_xor(sg, a2, step);
        const float p3 = sycl::permute_group_by_xor(sg, a3, step);
        a0 = (lane & step) ? p0 - a0 : a0 + p0;
        a1 = (lane & step) ? p1 - a1 : a1 + p1;
        a2 = (lane & step) ? p2 - a2 : a2 + p2;
        a3 = (lane & step) ? p3 - a3 : a3 + p3;
    }
    const float b0 = a0 + a1, b1 = a0 - a1, b2 = a2 + a3, b3 = a2 - a3;
    a0 = b0 + b2; a1 = b1 + b3; a2 = b0 - b2; a3 = b1 - b3;
}

/// The input-basis image of one 128-wide block of `input`, elements scaled by `suh` first.
inline void fwht128_input(const sycl::sub_group& sg, int lane, const float* input, const uint16_t* suh,
                          int64_t block, float& a0, float& a1, float& a2, float& a3) {
    a0 = input[block + lane] * f32_from_f16(suh[block + lane]);
    a1 = input[block + lane + 32] * f32_from_f16(suh[block + lane + 32]);
    a2 = input[block + lane + 64] * f32_from_f16(suh[block + lane + 64]);
    a3 = input[block + lane + 96] * f32_from_f16(suh[block + lane + 96]);
    fwht128_registers(sg, lane, a0, a1, a2, a3);
    a0 *= kScale; a1 *= kScale; a2 *= kScale; a3 *= kScale;
}

/// The input-basis image of a block a fused tail already holds in registers, in the same
/// element order fwht128_input loads: slot k holds the element at `block + lane + k * 32`.
inline void fwht128_input_registers(const sycl::sub_group& sg, int lane, const uint16_t* suh,
                                    int64_t block, float& a0, float& a1, float& a2, float& a3) {
    a0 *= f32_from_f16(suh[block + lane]);
    a1 *= f32_from_f16(suh[block + lane + 32]);
    a2 *= f32_from_f16(suh[block + lane + 64]);
    a3 *= f32_from_f16(suh[block + lane + 96]);
    fwht128_registers(sg, lane, a0, a1, a2, a3);
    a0 *= kScale; a1 *= kScale; a2 *= kScale; a3 *= kScale;
}

/// The output-basis image of one 128-wide block of `dot`: the butterfly, then `kScale * svh`
/// and the optional bias, exactly as the standalone output transform.
inline void fwht128_output(const sycl::sub_group& sg, int lane, const float* dot, const uint16_t* svh,
                           const uint16_t* bias, int64_t block, bool add_bias, float& a0, float& a1, float& a2,
                           float& a3) {
    a0 = dot[block + lane]; a1 = dot[block + lane + 32];
    a2 = dot[block + lane + 64]; a3 = dot[block + lane + 96];
    fwht128_registers(sg, lane, a0, a1, a2, a3);
    a0 *= kScale * f32_from_f16(svh[block + lane]);
    a1 *= kScale * f32_from_f16(svh[block + lane + 32]);
    a2 *= kScale * f32_from_f16(svh[block + lane + 64]);
    a3 *= kScale * f32_from_f16(svh[block + lane + 96]);
    if (add_bias && bias != nullptr) {
        a0 += f32_from_f16(bias[block + lane]); a1 += f32_from_f16(bias[block + lane + 32]);
        a2 += f32_from_f16(bias[block + lane + 64]); a3 += f32_from_f16(bias[block + lane + 96]);
    }
}

}  // namespace strata::kernels::exl3::had
