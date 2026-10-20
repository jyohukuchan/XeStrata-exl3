// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 codebook rules follow exllamav3's MIT-licensed implementation.
// src/kernels/xe/exl3_cb2.hpp - the codebook 2 (mul1) decode, in the portable form and in half precision.
//
// The value of a byte sum B is (887 * B - 452608) * 2^-17.  The reference decoder (exllamav3's
// codebook_rdna.hip.h, `decode_3inst` for codebook 2) computes it with one fused multiply add in half
// precision:
//
//     hfma16(F16(1024 + B), 0x1EEE, 0xC931)
//
// 0x1EEE is 887 * 2^-17 and 0xC931 is -10.3828125, both exact in F16, and 0x6400 + B is the F16 bit
// pattern of 1024 + B: B is at most 1020, so 2044 stays below 2048 and is exact in F16.  The product
// and the difference are therefore the exact number (887 * B - 452608) * 2^-17, a multiple of 2^-17
// with an absolute value below 2^19, so it is exact in F32 as well.
//
// The two forms are the same number, not an approximation of it.  The half form rounds the exact value
// once to F16; the portable form computes the same exact value in F32 and rounds it once to F16.  One
// rounding of one exact number gives the same F16 bits, so the two forms agree bit for bit for every
// byte sum.  The portable form is therefore the reference and stays available on every device; the half
// form is used where the device reports fp16 arithmetic.
//
// Nothing else changes with the form: the F16 RNE boundary of the codebook values, the FP32
// accumulation order, the Hadamard stages, the bias, the weight storage and the bpw are the same.
#pragma once

#include "strata/kernels/f16_bits.hpp"
#include "cuda_intrinsics.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::exl3 {

// The unsigned byte sum of the mul1 product.  dp4a is a signed byte dot product, so the sign bit is
// flipped and 512 is added: this is exactly sum(u_i - 128) + 512 = sum(u_i).
inline int cb2_byte_sum(uint32_t x) {
    return strata::kernels::xe::dp4a(sycl::bit_cast<int32_t>(x ^ 0x80808080u), 0x01010101, 512);
}

// The F16 bit pattern of 1024 + B.
inline uint16_t cb2_half_bits(int byte_sum) {
    return (uint16_t) (0x6400u + (unsigned) byte_sum);
}

// The exact value of a codebook 2 entry, in F32.  887 * 2^-17 and 452608 * 2^-17 are exact in F32, and
// for every byte sum in [0, 1020] the product and the difference are exact as well, so the fused
// multiply add returns the exact number whether it is contracted or not.
inline float cb2_exact_f32(int byte_sum) {
    return sycl::fma(static_cast<float>(byte_sum), 887.0f * 0x1.0p-17f, -452608.0f * 0x1.0p-17f);
}

// The portable form: the exact value rounded once to F16, widened back to F32.
inline float cb2_portable_f32(int byte_sum) {
    const float value = cb2_exact_f32(byte_sum);
    uint32_t u = 0;
    std::memcpy(&u, &value, sizeof(u));
    // All byte sums 0..1020 have abs(value) >= 238 * 2^-17, inside the finite F16 normal range.
    // The closed form is pre-rounded: preserve the exact F16 RNE boundary before widening to F32.
    u = (u + 0xfffu + ((u >> 13) & 1u)) & 0xffffe000u;
    float rounded = 0.0f;
    std::memcpy(&rounded, &u, sizeof(rounded));
    return rounded;
}

// The portable form's F16 bits.
inline uint16_t cb2_portable_bits(int byte_sum) {
    return f16_from_f32(cb2_exact_f32(byte_sum));
}

// The half form: one fused multiply add in F16, as the reference decoder does it.  The FMA rounds the
// exact product and difference once, so it returns the same F16 bits as the portable form.
inline sycl::half cb2_half_f16(int byte_sum) {
    const sycl::half operand = sycl::bit_cast<sycl::half>(cb2_half_bits(byte_sum));
    const sycl::half inverse = sycl::bit_cast<sycl::half>(uint16_t(0x1EEE));
    const sycl::half bias = sycl::bit_cast<sycl::half>(uint16_t(0xC931));
    return sycl::fma(operand, inverse, bias);
}

// The half form, widened to F32.  The widening is exact.
inline float cb2_half_f32(int byte_sum) {
    return static_cast<float>(cb2_half_f16(byte_sum));
}

// The half form's F16 bits.
inline uint16_t cb2_half_value_bits(int byte_sum) {
    return sycl::bit_cast<uint16_t>(cb2_half_f16(byte_sum));
}

// The value of a codebook 2 entry, in the chosen form.  Both forms return the same number.
template <bool HalfFma>
inline float cb2_value_f32(int byte_sum) {
    if constexpr (HalfFma) return cb2_half_f32(byte_sum);
    else return cb2_portable_f32(byte_sum);
}

// The F16 bits of a codebook 2 entry, in the chosen form.
template <bool HalfFma>
inline uint16_t cb2_value_bits(int byte_sum) {
    if constexpr (HalfFma) return cb2_half_value_bits(byte_sum);
    else return cb2_portable_bits(byte_sum);
}

// Whether the device runs the half form of the codebook 2 decode.  The device reports whether it has
// fp16 arithmetic; a device without it keeps the portable form.  STRATA_EXL3_CB2_HALF_FMA=0 forces
// the portable form on every device, so both forms can be compared with the same binary.
inline bool cb2_half_fma_enabled(const sycl::queue& queue) {
    const char* text = std::getenv("STRATA_EXL3_CB2_HALF_FMA");
    if (text != nullptr && text[0] == '0' && text[1] == '\0') return false;
    try {
        return queue.get_device().has(sycl::aspect::fp16);
    } catch (const sycl::exception&) {
        return false;
    }
}

}  // namespace strata::kernels::exl3
