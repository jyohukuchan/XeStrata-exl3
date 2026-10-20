// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 codebook rules follow exllamav3's MIT-licensed implementation.
// src/kernels/xe/exl3_cb2.hpp - the codebook 2 (mul1) decode, in the portable form and in half precision.
//
// The byte sum B of the mul1 product is the sum of its four bytes, 0..1020.  The reference decoder computes it
// with an unsigned dot product and no sign flip: its __dp4a is the unsigned dot4 (exllamav3's hip_compat.hip.h
// maps it to __builtin_amdgcn_udot4).  The targets that have an unsigned dot product use it here; the ones
// that have only a signed one keep the sign bits flipped and 512 added, which gives the same sum, because
// sum(u_i - 128) + 512 = sum(u_i).  The two forms are compared with each other, and with the host's
// reference, for every byte sum and for every 16-bit state (src/kernels/exl3_cb2_parity.cpp).
//
// The choice is per device image, not per device: the AMD, the NVIDIA and the SPIR-V images are compiled
// separately (CMakeLists.txt's -fsycl-targets) and the SYCL runtime hands a device the image that was built
// for its architecture, so the code a device runs is the code that was compiled for that image.  A device
// that runs the SPIR-V image (Intel's GPUs, and the AMD GPUs that have no image of their own) gets the
// signed form: the SPIR-V unsigned dot product has no expansion for the targets that lack the instruction,
// and that image is built for every device.  The AMD and the NVIDIA images get the unsigned one.
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

// The unsigned byte dot product: the sum of the four products of the byte pairs, both operands unsigned,
// with the accumulator c added to the sum.  Where the target has an instruction for it, one instruction does
// the whole sum; where it has only the signed one, the sign bits are flipped and the accumulator carries the
// 512 back, which gives the same number, because sum(u_i - 128) + 512 = sum(u_i).  The byte pairs are the
// bytes of b taken as signed, so b is 0x01010101 for a byte sum.
inline uint32_t udot4_bytes(uint32_t a, uint32_t b, uint32_t c) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__) && \
    (defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__))
    // v_dot4_u32_u8, the unsigned byte dot product of gfx10 and later (the dot7-insts feature).  gfx1010,
    // which lacks it, is no target here either: the shared dp4a wrapper needs its dot1-insts.
    return __builtin_amdgcn_udot4(a, b, c, false);
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) && STRATA_NV_ARCH >= 610
    // dp4a.u32.u32, the unsigned form of the instruction the shared wrapper uses in its signed form.  It
    // needs the same architecture (sm_61 and later) and does not saturate either.
    uint32_t r;
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
#else
    // The signed dot product with the sign bits flipped: every byte counts as u_i - 128, so the sum is the
    // unsigned sum less 512, and the accumulator adds the 512 back.
    return (uint32_t) strata::kernels::xe::dp4a(sycl::bit_cast<int32_t>(a ^ 0x80808080u),
                                                sycl::bit_cast<int32_t>(b), (int) (c + 512u));
#endif
}

// Whether the byte sum of this device image uses the unsigned dot product.  The choice is made when the
// image is compiled (the macros above), and the runtime hands a device the image that was built for it, so
// the device's own image decides, not the device's name or model.
inline bool cb2_unsigned_dot_native() {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__) && \
    (defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__))
    return true;
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) && STRATA_NV_ARCH >= 610
    return true;
#else
    return false;
#endif
}

// The unsigned byte sum of the mul1 product: the sum of its four bytes, 0..1020.
inline int cb2_byte_sum(uint32_t x) {
    return (int) udot4_bytes(x, 0x01010101u, 0u);
}

// The same sum, with the signed dot product and the sign bits flipped: this is the form the targets without
// an unsigned dot product use, and the one the parity test compares the unsigned form with.
inline int cb2_byte_sum_signed(uint32_t x) {
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
