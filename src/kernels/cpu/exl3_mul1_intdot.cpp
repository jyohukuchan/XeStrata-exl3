// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Experimental CB2 Q8 integer dot path. The register decode follows the
// MIT-licensed exllamav3 cpu/moe_mul1.cpp, without its Torch or pool dependencies.
#include "strata/kernels/cpu/exl3_mul1_intdot.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__clang__) || defined(__GNUC__))
#include "strata/kernels/cpu/expert_layout.hpp"
#include <immintrin.h>
#define EXL3_INTDOT_AVAILABLE 1
#define EXL3_INTDOT_AVX2 __attribute__((target("avx2")))
#else
#define EXL3_INTDOT_AVAILABLE 0
#define EXL3_INTDOT_AVX2
#endif

namespace strata::kernels::cpu::exl3 {
namespace {
using MatrixView = strata::kernels::exl3::MatrixView;
// Public MUL1 codebook constant, never a credential.
// gitleaks:allow -- public EXL3 codebook multiplier.
#if EXL3_INTDOT_AVAILABLE
constexpr uint32_t kMul1 = 0x83DCD12Du;
constexpr float kInv = 887.0f / 131072.0f;  // fp16(0x1eee)
// GPU/F16 codebook before final RNE is (byte_sum-510)*kInv+kAffineDelta.
// The original CPU kernel omitted this delta as well as final binary16 RNE.
constexpr float kAffineDelta = -238.0f / 131072.0f;
#endif
constexpr int64_t kMaxK = (std::numeric_limits<int32_t>::max() / (1020 * 127) / 128) * 128;

bool eligible(const MatrixView& matrix) {
    return matrix.codebook == 2 && matrix.bits >= 1 && matrix.bits <= 8 && matrix.k > 0 &&
           matrix.k <= kMaxK && matrix.k % 128 == 0 && matrix.n > 0 && matrix.n % 128 == 0 &&
           matrix.trellis != nullptr && matrix.suh != nullptr && matrix.svh != nullptr;
}

#if EXL3_INTDOT_AVAILABLE
template <int bits, int row, bool second_word>
constexpr std::array<int32_t, 16> make_row_indices() {
    std::array<int32_t, 16> indices{};
    for (int col = 0; col < 16; ++col) {
        const int state = strata::kernels::exl3::tile_state_for_element(row * 16 + col);
        const int b0 = state * bits + bits - 16 + 256 * bits;
        const int b1 = b0 + 16;
        indices[col] = (second_word ? (b1 - 1) / 32 : b0 / 32) % (bits * 8);
    }
    return indices;
}

template <int bits, int row>
constexpr int row_shift(int col) {
    const int state = strata::kernels::exl3::tile_state_for_element(row * 16 + col);
    const int b1 = state * bits + bits + 256 * bits;
    return ((b1 - 1) / 32 + 1) * 32 - b1;
}

template <int bits, int row, bool second_word, int half, int Reg>
constexpr uint8_t avx2_reg_mask()
{
    constexpr auto idx16 = make_row_indices<bits, row, second_word>();
    uint8_t mask = 0;
    for (int i = 0; i < 8; ++i)
        if (idx16[half * 8 + i] / 8 == Reg) mask |= uint8_t(1) << i;
    return mask;
}

template <int bits, int row, bool second_word, int half>
constexpr std::array<int32_t, 8> avx2_lane_idx()
{
    constexpr auto idx16 = make_row_indices<bits, row, second_word>();
    std::array<int32_t, 8> out{};
    for (int i = 0; i < 8; ++i) out[i] = idx16[half * 8 + i] % 8;
    return out;
}

// Permutes+blends together only the registers that actually contribute a lane to this half, in
// increasing Reg order (skipped candidates cost nothing -- if constexpr eliminates them, so low
// bitrates collapse to a single unconditional permute, same as VNNI's cheapest case)
template <int bits, int row, bool second_word, int half, int Reg = 0>
EXL3_INTDOT_AVX2
inline __m256i avx2_gather_half(const __m256i (&preg)[bits])
{
    // All-compile-time-constant arguments: the compiler folds this to a single constant load,
    // same as a hand-written lookup table. No lambda (this function is target-attributed, and
    // GCC does not propagate the target to a lambda's closure -- see file header note).
    constexpr auto li = avx2_lane_idx<bits, row, second_word, half>();
    const __m256i lane_idx_v = _mm256_setr_epi32(li[0], li[1], li[2], li[3], li[4], li[5], li[6], li[7]);
    if constexpr (Reg + 1 >= bits)
    {
        // last candidate: every column not already claimed must come from here
        return _mm256_permutevar8x32_epi32(preg[Reg], lane_idx_v);
    }
    else
    {
        constexpr uint8_t mask = avx2_reg_mask<bits, row, second_word, half, Reg>();
        if constexpr (mask == 0)
        {
            return avx2_gather_half<bits, row, second_word, half, Reg + 1>(preg);
        }
        else
        {
            const __m256i cur = _mm256_permutevar8x32_epi32(preg[Reg], lane_idx_v);
            const __m256i rest = avx2_gather_half<bits, row, second_word, half, Reg + 1>(preg);
            return _mm256_blend_epi32(rest, cur, mask);
        }
    }
}

// Decodes row `row`'s 16 states into two 8-wide registers (cols 0-7, cols 8-15) from the k-tile's
// pre-loaded registers; no scalar decode_state_scalar calls. b0/b1 and the funnel shift follow
// the same bit layout as decode_state_scalar; the shift is shared across each 8-column half (by
// construction of the tile's tensor-core permutation, the same invariant the VNNI path relies on
// for its single per-half s0/s1).
template <int bits, int row>
EXL3_INTDOT_AVX2
inline void avx2_row_codes(const __m256i (&preg)[bits], __m256i& codes_lo, __m256i& codes_hi)
{
    const __m256i a_lo = avx2_gather_half<bits, row, false, 0>(preg);
    const __m256i b_lo = avx2_gather_half<bits, row, true, 0>(preg);
    const __m256i a_hi = avx2_gather_half<bits, row, false, 1>(preg);
    const __m256i b_hi = avx2_gather_half<bits, row, true, 1>(preg);
    constexpr int s0 = row_shift<bits, row>(0);
    constexpr int s1 = row_shift<bits, row>(8);
    const __m256i mask16 = _mm256_set1_epi32(0xffff);
    codes_lo = _mm256_and_si256(_mm256_or_si256(
        _mm256_srli_epi32(b_lo, s0), _mm256_slli_epi32(a_lo, 32 - s0)), mask16);
    codes_hi = _mm256_and_si256(_mm256_or_si256(
        _mm256_srli_epi32(b_hi, s1), _mm256_slli_epi32(a_hi, 32 - s1)), mask16);
}


// Pair product bytes BEFORE including the activation: each unsigned pair is
// <=510, so vpmaddubsw cannot saturate. Then vpmaddwd applies the duplicated
// signed activation and produces one exact i32 byte-sum product per column.
template <int Bits, int Row = 0>
EXL3_INTDOT_AVX2 inline void accumulate_rows(const __m256i (&packed)[Bits], const int32_t* x,
                                            __m256i& acc_lo, __m256i& acc_hi) {
    if constexpr (Row < 16) {
        __m256i states_lo, states_hi;
        avx2_row_codes<Bits, Row>(packed, states_lo, states_hi);
        const __m256i mult = _mm256_set1_epi32(static_cast<int32_t>(kMul1));
        const __m256i ones = _mm256_set1_epi8(1);
        const __m256i pairs_lo = _mm256_maddubs_epi16(_mm256_mullo_epi32(states_lo, mult), ones);
        const __m256i pairs_hi = _mm256_maddubs_epi16(_mm256_mullo_epi32(states_hi, mult), ones);
        const __m256i activation = _mm256_set1_epi32(x[Row]);
        acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(pairs_lo, activation));
        acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(pairs_hi, activation));
        accumulate_rows<Bits, Row + 1>(packed, x, acc_lo, acc_hi);
    }
}

EXL3_INTDOT_AVX2 void quantize_input(const float* input, size_t k, int32_t* duplicated,
                                     float q, int32_t& sum) {
    const __m256 reciprocal = _mm256_set1_ps(1.0f / q);
    const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
    const __m256i mask = _mm256_set1_epi32(0xffff);
    __m256i total = _mm256_setzero_si256();
    for (size_t i = 0; i < k; i += 8) {
        const __m256 scaled = _mm256_mul_ps(_mm256_loadu_ps(input + i), reciprocal);
        // Explicit ties-to-even, independent of ambient MXCSR rounding mode.
        const __m256 rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const __m256i quantized = _mm256_min_epi32(hi, _mm256_max_epi32(lo, _mm256_cvttps_epi32(rounded)));
        total = _mm256_add_epi32(total, quantized);
        const __m256i lower = _mm256_and_si256(quantized, mask);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(duplicated + i),
                            _mm256_or_si256(lower, _mm256_slli_epi32(lower, 16)));
    }
    alignas(32) int32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), total);
    sum = 0;
    for (int32_t lane : lanes) sum += lane;
}

template <int Bits>
EXL3_INTDOT_AVX2 void dot_range(const MatrixView& matrix, const Mul1Input& input,
                                size_t first_tile, size_t tile_count, float* dot) {
    const size_t tiles_k = static_cast<size_t>(matrix.k / 16);
    const size_t tiles_n = static_cast<size_t>(matrix.n / 16);
    const size_t packed_words = static_cast<size_t>(16 * Bits);
    const __m256i correction = _mm256_set1_epi32(510 * input.sum);
    const __m256 scale = _mm256_set1_ps(kInv * input.q);
    // Constant correction restores the exact pre-RNE affine codebook center.
    const __m256 delta = _mm256_set1_ps((kAffineDelta * input.q) * static_cast<float>(input.sum));
    for (size_t tile_n = first_tile; tile_n < first_tile + tile_count; ++tile_n) {
        __m256i acc_lo = _mm256_setzero_si256(), acc_hi = _mm256_setzero_si256();
        for (size_t tile_k = 0; tile_k < tiles_k; ++tile_k) {
            const uint16_t* tile = matrix.trellis + (tile_k * tiles_n + tile_n) * packed_words;
            __m256i registers[Bits];
            for (int i = 0; i < Bits; ++i)
                registers[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tile + i * 16));
            accumulate_rows<Bits>(registers, input.duplicated + tile_k * 16, acc_lo, acc_hi);
        }
        // Center in exact i32 before conversion, avoiding cancellation between
        // two large FP32 terms. No FP64 products or binary16 decode remains.
        const __m256 z_lo = _mm256_add_ps(_mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_sub_epi32(acc_lo, correction)), scale), delta);
        const __m256 z_hi = _mm256_add_ps(_mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_sub_epi32(acc_hi, correction)), scale), delta);
        float* out = dot + (tile_n - first_tile) * 16;
        _mm256_storeu_ps(out, z_lo);
        _mm256_storeu_ps(out + 8, z_hi);
    }
}
#endif
}  // namespace

bool mul1_intdot_enabled() {
#if EXL3_INTDOT_AVAILABLE
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_EXL3_CPU_INTDOT");
        const char* force_generic = std::getenv("STRATA_NO_EXL3_AVX2");
        return value != nullptr && value[0] == '1' &&
               !(force_generic != nullptr && force_generic[0] == '1') && strata::kernels::cpu::cpu_avx2_ok();
    }();
    return enabled;
#else
    return false;
#endif
}

bool prepare_mul1_input(const MatrixView& matrix, const float* input_basis, int32_t* duplicated,
                        size_t capacity, Mul1Input& prepared) {
    prepared = {};
    if (!mul1_intdot_enabled() || !eligible(matrix) || input_basis == nullptr || duplicated == nullptr ||
        capacity < static_cast<size_t>(matrix.k))
        return false;
#if EXL3_INTDOT_AVAILABLE
    // Positive IEEE bits are ordered like their finite values. This scan also
    // sees tiny/non-finite inputs when ambient DAZ/FTZ is enabled.
    uint32_t amax_bits = 0;
    for (size_t i = 0; i < static_cast<size_t>(matrix.k); ++i) {
        uint32_t value_bits;
        std::memcpy(&value_bits, input_basis + i, sizeof(value_bits));
        value_bits &= 0x7fffffffu;
        if (value_bits >= 0x7f800000u) return false;
        amax_bits = (std::max)(amax_bits, value_bits);
    }
    float amax;
    std::memcpy(&amax, &amax_bits, sizeof(amax));
    const float q = amax_bits != 0 ? amax / 127.0f : 1.0f;
    if (!(q > 0.0f) || !std::isfinite(1.0f / q)) return false;
    int32_t sum = 0;
    quantize_input(input_basis, static_cast<size_t>(matrix.k), duplicated, q, sum);
    prepared = {duplicated, static_cast<size_t>(matrix.k), q, sum, true};
    return true;
#else
    return false;
#endif
}

bool linear_mul1_dot_range(const MatrixView& matrix, const Mul1Input& prepared,
                           size_t first_tile, size_t tile_count, float* dot) {
    if (!mul1_intdot_enabled() || !eligible(matrix) || !prepared.ready || prepared.duplicated == nullptr ||
        prepared.count < static_cast<size_t>(matrix.k) || dot == nullptr || tile_count == 0 ||
        tile_count > static_cast<size_t>(matrix.n / 16) || first_tile > static_cast<size_t>(matrix.n / 16) - tile_count)
        return false;
#if EXL3_INTDOT_AVAILABLE
    switch (matrix.bits) {
        case 1: dot_range<1>(matrix, prepared, first_tile, tile_count, dot); break;
        case 2: dot_range<2>(matrix, prepared, first_tile, tile_count, dot); break;
        case 3: dot_range<3>(matrix, prepared, first_tile, tile_count, dot); break;
        case 4: dot_range<4>(matrix, prepared, first_tile, tile_count, dot); break;
        case 5: dot_range<5>(matrix, prepared, first_tile, tile_count, dot); break;
        case 6: dot_range<6>(matrix, prepared, first_tile, tile_count, dot); break;
        case 7: dot_range<7>(matrix, prepared, first_tile, tile_count, dot); break;
        default: dot_range<8>(matrix, prepared, first_tile, tile_count, dot); break;
    }
    return true;
#else
    return false;
#endif
}

}  // namespace strata::kernels::cpu::exl3
