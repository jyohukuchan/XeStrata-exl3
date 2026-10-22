// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU EXL3 trellis extraction, codebooks and the batch-1 correctness path.
// The packed-tile and codebook rules are derived from exllamav3's MIT code;
// the implementation here remains portable C++ with an AVX2-safe baseline.
#include "strata/kernels/cpu/exl3.hpp"
#include "strata/kernels/cpu/exl3_mul1_intdot.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

// Clang also supports per-function targets for the MSVC ABI. Other compilers
// retain the generic implementation rather than emitting unguarded AVX2.
// The half conversion and the fused multiply-add need F16C and FMA; the CPU
// check in `cpu_avx2_ok` asks for both, so a kernel that uses them is only
// reached on a CPU that has them.
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__clang__) || defined(__GNUC__))
#include "strata/kernels/cpu/expert_layout.hpp"
#include <immintrin.h>
#define EXL3_HAVE_TARGET_AVX2 1
#define EXL3_TARGET_AVX2 __attribute__((target("avx2")))
#define EXL3_TARGET_AVX2_F16C __attribute__((target("avx2,f16c,fma")))
#else
#define EXL3_HAVE_TARGET_AVX2 0
#define EXL3_TARGET_AVX2
#define EXL3_TARGET_AVX2_F16C
#endif

namespace strata::kernels::cpu::exl3 {
namespace {

using MatrixView = strata::kernels::exl3::MatrixView;

constexpr int kTileSide = 16;
constexpr int kTileValues = 256;
constexpr int kHadamardSide = 128;
constexpr float kHadamardScale = 0.088388347648f;

// The three constants are the public EXL3 codebook multipliers.  They are
// data, rather than secrets; keep the gitleaks marker next to them.
// gitleaks:allow -- public EXL3 codebook multipliers, never credentials.
constexpr uint32_t kCb0Multiplier = 89226354u;
constexpr uint32_t kCb0Addend = 64248484u;
constexpr uint32_t kCb1Multiplier = 0xCBAC1FEDu;
constexpr uint32_t kCb2Multiplier = 0x83DCD12Du;

bool checked_mul(size_t a, size_t b, size_t& out) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) return false;
    out = a * b;
    return true;
}

bool checked_matrix_words(const MatrixView& matrix, size_t& words) {
    if (matrix.k <= 0 || matrix.n <= 0) return false;
    const uint64_t tk = static_cast<uint64_t>(matrix.k / kTileSide);
    const uint64_t tn = static_cast<uint64_t>(matrix.n / kTileSide);
    size_t tiles;
    if (tk > std::numeric_limits<size_t>::max() || tn > std::numeric_limits<size_t>::max()) return false;
    if (!checked_mul(static_cast<size_t>(tk), static_cast<size_t>(tn), tiles)) return false;
    size_t per_tile;
    if (!checked_mul(static_cast<size_t>(16), static_cast<size_t>(matrix.bits), per_tile)) return false;
    return checked_mul(tiles, per_tile, words);
}

uint32_t load_u32(const uint16_t* p, size_t word) {
    // Building the value from uint16 elements makes the packed format's
    // little-endian bit order explicit even on a big-endian host.
    return static_cast<uint32_t>(p[word * 2]) | (static_cast<uint32_t>(p[word * 2 + 1]) << 16);
}

uint16_t f32_to_f16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    uint32_t mantissa = bits & 0x7fffffu;

    if (exponent == 0xffu) {
        if (mantissa == 0) return static_cast<uint16_t>(sign | 0x7c00u);
        // Preserve a quiet NaN and some payload bits, as the hardware half
        // conversion does.  EXL3 codebooks normally produce finite values.
        return static_cast<uint16_t>(sign | 0x7c00u | (mantissa >> 13) | 1u);
    }

    int half_exp = static_cast<int>(exponent) - 127 + 15;
    if (half_exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);

    if (half_exp <= 0) {
        if (half_exp < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x800000u;
        const int shift = 14 - half_exp;
        uint32_t half_mantissa = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u))) ++half_mantissa;
        return static_cast<uint16_t>(sign | half_mantissa);
    }

    uint32_t half_mantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_mantissa & 1u))) ++half_mantissa;
    if (half_mantissa == 0x400u) {
        half_mantissa = 0;
        ++half_exp;
        if (half_exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(half_exp) << 10) | half_mantissa);
}

// The half to float conversion is the shared one (`f32_from_f16` in
// f16_bits.hpp), so that the CPU and the device read a half the same way and
// the parity test can cover all 65,536 of them in one place.  It is exact:
// every half, the subnormals, the zeros, the infinities and the NaNs
// included, maps to a float that holds it exactly.

float half_add(uint16_t a, uint16_t b) {
    // The volatile temporary prevents a compiler from contracting the
    // operation into an unrelated wider expression before half rounding.
    volatile float sum = f32_from_f16(a) + f32_from_f16(b);
    return sum;
}

uint16_t half_add_bits(uint16_t a, uint16_t b) {
    return f32_to_f16(half_add(a, b));
}

uint32_t lop3_6a(uint32_t value) {
    return ((~value & 0x3B603B60u) | (value & (0x8FFF8FFFu ^ 0x3B603B60u)));
}

uint16_t decode_state_half(uint16_t state, int codebook) {
    uint32_t value;
    if (codebook == 0) {
        value = static_cast<uint32_t>(state) * kCb0Multiplier + kCb0Addend;
        value = lop3_6a(value);
        return half_add_bits(static_cast<uint16_t>(value), static_cast<uint16_t>(value >> 16));
    }
    if (codebook == 1) {
        value = static_cast<uint32_t>(state) * kCb1Multiplier;
        value = lop3_6a(value);
        return half_add_bits(static_cast<uint16_t>(value), static_cast<uint16_t>(value >> 16));
    }

    value = static_cast<uint32_t>(state) * kCb2Multiplier;
    const int byte_sum = static_cast<int>(value & 0xffu) + static_cast<int>((value >> 8) & 0xffu) +
                         static_cast<int>((value >> 16) & 0xffu) + static_cast<int>(value >> 24);
    const uint16_t h = static_cast<uint16_t>(0x6400u + byte_sum);
    const float inv = f32_from_f16(0x1eeeu);
    const float bias = f32_from_f16(0xc931u);
    volatile float product = f32_from_f16(h) * inv;
    volatile float result = product + bias;
    return f32_to_f16(result);
}

using CodebookLut = std::array<std::array<uint16_t, 65536>, 3>;

// The LUT is initialized once with the existing exact decoder.  Every lookup is
// therefore bit-preserving relative to decode_state_half, including half-add and
// MUL1 half-FMA rounding boundaries.  It stores codebook values only, never W.
const CodebookLut& codebook_lut() {
    static const CodebookLut table = [] {
        CodebookLut result{};
        for (int codebook = 0; codebook < 3; ++codebook)
            for (uint32_t state = 0; state < 65536; ++state)
                result[static_cast<size_t>(codebook)][state] =
                    decode_state_half(static_cast<uint16_t>(state), codebook);
        return result;
    }();
    return table;
}

inline uint16_t decode_state_half_lut(uint16_t state, int codebook) {
    return codebook_lut()[static_cast<size_t>(codebook)][state];
}

using CodebookF32Lut = std::array<std::array<float, 65536>, 3>;

// F32 values are derived from the exact F16 LUT once.  The pointer for the
// selected codebook is hoisted by linear_with_scratch; no function/static guard
// remains in the per-weight loop.
const CodebookF32Lut& codebook_lut_f32() {
    static const CodebookF32Lut table = [] {
        CodebookF32Lut result{};
        const auto& half_table = codebook_lut();
        for (int codebook = 0; codebook < 3; ++codebook)
            for (uint32_t state = 0; state < 65536; ++state)
                result[static_cast<size_t>(codebook)][state] =
                    f32_from_f16(half_table[static_cast<size_t>(codebook)][state]);
        return result;
    }();
    return table;
}

#if EXL3_HAVE_TARGET_AVX2
// The MUL1 codebook's byte sum is the sum of four bytes, so it lies between
// 0 and 1020, and the AVX2 path computes the value from the sum in arithmetic
// (see `decode_cb2_8_f32_avx2`), so no table of the sums is needed here.
#endif

// ---------------------------------------------------------------- the tables
//
// A trellis window's place in its tile does not depend on the data: the window
// of a state sits at a fixed bit offset, so the words it spans and the shift
// it needs are a property of the bit width alone.  They are therefore built
// once, when the program starts, and the per-window arithmetic leaves the
// loop that decodes a tile.

/// The words and the shift of every trellis window of a 16x16 tile.
template <int Bits>
struct TileWindows {
    uint16_t word0[256];  ///< the first word of the window
    uint16_t word1[256];  ///< the last word of the window
    uint8_t shift[256];   ///< the window's shift inside the two words
};

template <int Bits>
constexpr TileWindows<Bits> make_tile_windows() {
    TileWindows<Bits> out{};
    const size_t words = static_cast<size_t>(Bits) * 8;
    for (int state = 0; state < 256; ++state) {
        const int b0 = state * Bits + Bits - 16 + 256 * Bits;
        const int b1 = b0 + 16;
        const size_t i0 = static_cast<size_t>(b0 >> 5);
        const size_t i1 = static_cast<size_t>((b1 - 1) >> 5);
        out.word0[state] = static_cast<uint16_t>(i0 >= words ? i0 - words : i0);
        out.word1[state] = static_cast<uint16_t>(i1 >= words ? i1 - words : i1);
        out.shift[state] = static_cast<uint8_t>((i1 + 1) * 32 - b1);
    }
    return out;
}

template <int Bits>
const TileWindows<Bits>& tile_windows() {
    static constexpr TileWindows<Bits> table = make_tile_windows<Bits>();
    return table;
}

/// The words and the shifts of the eight states of a group, relative to the
/// group's first word: the group is read with one sixteen-byte load, and the
/// table says which of the four words each lane takes.
template <int Bits>
struct TileGroups {
    int32_t relative0[32][8];
    int32_t relative1[32][8];
    int32_t shift[32][8];
    int32_t left_shift[32][8];
    int32_t word_base[32];
};

template <int Bits>
constexpr TileGroups<Bits> make_tile_groups() {
    TileGroups<Bits> out{};
    const size_t words = static_cast<size_t>(Bits) * 8;
    for (int group = 0; group < 32; ++group) {
        const int state_base = group * 8;
        const int first_b0 = state_base * Bits + Bits - 16 + 256 * Bits;
        const size_t first_i0 = static_cast<size_t>(first_b0 >> 5);
        out.word_base[group] = static_cast<int>(first_i0 >= words ? first_i0 - words : first_i0);
        for (int lane = 0; lane < 8; ++lane) {
            const int b0 = (state_base + lane) * Bits + Bits - 16 + 256 * Bits;
            const int b1 = b0 + 16;
            const size_t i0 = static_cast<size_t>(b0 >> 5);
            const size_t i1 = static_cast<size_t>((b1 - 1) >> 5);
            out.relative0[group][lane] = static_cast<int32_t>(i0 - first_i0);
            out.relative1[group][lane] = static_cast<int32_t>(i1 - first_i0);
            out.shift[group][lane] = static_cast<int32_t>((i1 + 1) * 32 - b1);
            out.left_shift[group][lane] = static_cast<int32_t>(32 - ((i1 + 1) * 32 - b1));
        }
    }
    return out;
}

template <int Bits>
const TileGroups<Bits>& tile_groups() {
    static constexpr TileGroups<Bits> table = make_tile_groups<Bits>();
    return table;
}

// Hot linear path: validation and the codebook LUT pointer are already hoisted.
// This writes only one bounded 16x16 tile in F32; it never materializes W.
template <int Bits>
void decode_tile_f32_hot_bits(const MatrixView& matrix, int64_t tile_k, int64_t tile_n, const float* lut,
                              float* values) {
    const int64_t tiles_n = matrix.n / kTileSide;
    const size_t packed_size = static_cast<size_t>(16 * Bits);
    const size_t tile_index = strata::kernels::exl3::trellis_tile_index(matrix.tile_order, tile_k, tile_n,
                                                                         matrix.k / kTileSide, tiles_n);
    const uint16_t* packed = matrix.trellis + tile_index * packed_size;
    const TileWindows<Bits>& windows = tile_windows<Bits>();
    for (int state = 0; state < kTileValues; ++state) {
        const uint64_t merged = (static_cast<uint64_t>(load_u32(packed, windows.word0[state])) << 32) |
                                load_u32(packed, windows.word1[state]);
        const uint16_t packed_state = static_cast<uint16_t>((merged >> windows.shift[state]) & 0xffffu);
        values[strata::kernels::exl3::tile_element_for_state(state)] = lut[packed_state];
    }
}

template <int Bits>
void linear_dot_range_f32(const MatrixView& matrix, const float* input_basis, size_t first_tile,
                          size_t tile_count, float* dot, const float* lut) {
    const size_t k = static_cast<size_t>(matrix.k);
    const size_t tiles_k = k / kTileSide;
    std::array<float, kTileValues> tile_f32{};
    for (size_t tile_n = first_tile; tile_n < first_tile + tile_count; ++tile_n) {
        double acc[kTileSide] = {};
        for (size_t tile_k = 0; tile_k < tiles_k; ++tile_k) {
            decode_tile_f32_hot_bits<Bits>(matrix, static_cast<int64_t>(tile_k), static_cast<int64_t>(tile_n), lut,
                                           tile_f32.data());
            const size_t row_base = tile_k * kTileSide;
            for (int row = 0; row < kTileSide; ++row) {
                const double input = static_cast<double>(input_basis[row_base + static_cast<size_t>(row)]);
                for (int col = 0; col < kTileSide; ++col)
                    acc[col] += static_cast<double>(tile_f32[static_cast<size_t>(row * kTileSide + col)]) * input;
            }
        }
        for (int col = 0; col < kTileSide; ++col)
            dot[(tile_n - first_tile) * kTileSide + static_cast<size_t>(col)] = static_cast<float>(acc[col]);
    }
}

#if EXL3_HAVE_TARGET_AVX2

// Each tile is staged once as dwords, followed by three wrapped dwords.
// An eight-state window spans at most four dwords; one unaligned 16-byte
// load plus a broadcast/vpermd therefore replaces the two packed-word
// vpgatherdd operations used by the gather prototype.
//
// The words and the shifts of the eight windows come from a table that the bit
// width alone fixes, so a group costs one load and the shifts, and no
// arithmetic per window.
template <int Bits>
EXL3_TARGET_AVX2_F16C void decode_cb2_8_f32_avx2(const uint32_t* staged, const TileGroups<Bits>& groups, int group,
                                                  float* values) {
    const __m256i vrelative0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(groups.relative0[group]));
    const __m256i vrelative1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(groups.relative1[group]));
    const __m256i vshift = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(groups.shift[group]));
    const __m256i vleft_shift = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(groups.left_shift[group]));
    const __m128i four_words = _mm_loadu_si128(reinterpret_cast<const __m128i*>(staged + groups.word_base[group]));
    const __m256i broadcast_words = _mm256_broadcastsi128_si256(four_words);
    const __m256i word0 = _mm256_permutevar8x32_epi32(broadcast_words, vrelative0);
    const __m256i word1 = _mm256_permutevar8x32_epi32(broadcast_words, vrelative1);
    const __m256i merged = _mm256_or_si256(_mm256_sllv_epi32(word0, vleft_shift),
                                           _mm256_srlv_epi32(word1, vshift));
    const __m256i state = _mm256_and_si256(merged, _mm256_set1_epi32(0xffff));

    const __m256i multiplied = _mm256_mullo_epi32(state, _mm256_set1_epi32(static_cast<int>(kCb2Multiplier)));
    const __m256i byte_pairs = _mm256_maddubs_epi16(multiplied, _mm256_set1_epi8(1));
    const __m256i byte_sum = _mm256_madd_epi16(byte_pairs, _mm256_set1_epi16(1));

    // The MUL1 codebook in arithmetic.  The byte sum of the multiplied state
    // is the sum of four bytes, so it lies between 0 and 1020, and it selects
    // the value
    //     f16_rne((1024 + sum) * inv + bias)
    // with `inv` the half 0x1eee and `bias` the half 0xc931.  The sum is an
    // integer of at most 1020, so (1024 + sum) * inv is exact in f32 (the
    // product needs 24 bits at most) and the fused multiply-add therefore
    // gives the same bits as the scalar decoder's separate multiply and add.
    // The hardware's half rounding is the same round-to-nearest-even as
    // `f32_to_f16`, and every value of the codebook is a normal half.
    const __m256 t = _mm256_add_ps(_mm256_cvtepi32_ps(byte_sum), _mm256_set1_ps(1024.0f));
    const __m256 q = _mm256_fmadd_ps(t, _mm256_set1_ps(f32_from_f16(0x1eeeu)), _mm256_set1_ps(f32_from_f16(0xc931u)));
    const __m128i half = _mm256_cvtps_ph(q, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm256_storeu_ps(values, _mm256_cvtph_ps(half));
}

template <int Bits>
EXL3_TARGET_AVX2_F16C void decode_tile_f32_hot_bits_avx2_cb2(const MatrixView& matrix, int64_t tile_k,
                                                              int64_t tile_n, float* values) {
    const int64_t tiles_n = matrix.n / kTileSide;
    const size_t packed_size = static_cast<size_t>(16 * Bits);
    const size_t tile_index = strata::kernels::exl3::trellis_tile_index(matrix.tile_order, tile_k, tile_n,
                                                                         matrix.k / kTileSide, tiles_n);
    const uint16_t* packed = matrix.trellis + tile_index * packed_size;
    const size_t words32_count = static_cast<size_t>(Bits) * 8;
    const TileGroups<Bits>& groups = tile_groups<Bits>();
    alignas(32) uint32_t staged[64 + 3];
    std::memcpy(staged, packed, words32_count * sizeof(uint32_t));
    std::memcpy(staged + words32_count, staged, 3 * sizeof(uint32_t));
    alignas(32) float decoded[8];
    for (int group = 0; group < kTileValues / 8; ++group) {
        decode_cb2_8_f32_avx2<Bits>(staged, groups, group, decoded);
        for (int lane = 0; lane < 8; ++lane)
            values[strata::kernels::exl3::tile_element_for_state(group * 8 + lane)] = decoded[lane];
    }
}

/// The row-major accumulation of one tile into the sixteen output columns.
///
/// The product of the decoded weight and the input is exact in f64 (both are
/// halves, so their product needs 48 bits at most), so the fused multiply-add
/// gives the same bits as the separate multiply and add.
EXL3_TARGET_AVX2_F16C void accumulate_tile_f32_avx2(const float* tile, const float* input, double* acc) {
    __m256d a0 = _mm256_loadu_pd(acc);
    __m256d a1 = _mm256_loadu_pd(acc + 4);
    __m256d a2 = _mm256_loadu_pd(acc + 8);
    __m256d a3 = _mm256_loadu_pd(acc + 12);
    for (int row = 0; row < kTileSide; ++row) {
        const __m256d in = _mm256_set1_pd(static_cast<double>(input[row]));
        a0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(tile + row * kTileSide)), in, a0);
        a1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(tile + row * kTileSide + 4)), in, a1);
        a2 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(tile + row * kTileSide + 8)), in, a2);
        a3 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(tile + row * kTileSide + 12)), in, a3);
    }
    _mm256_storeu_pd(acc, a0);
    _mm256_storeu_pd(acc + 4, a1);
    _mm256_storeu_pd(acc + 8, a2);
    _mm256_storeu_pd(acc + 12, a3);
}

template <int Bits>
EXL3_TARGET_AVX2_F16C void linear_dot_range_f32_avx2(const MatrixView& matrix, const float* input_basis, size_t first_tile,
                                                       size_t tile_count, float* dot, const float* lut) {
    const size_t tiles_k = matrix.k / kTileSide;
    std::array<float, kTileValues> tile_f32{};
    for (size_t tile_n = first_tile; tile_n < first_tile + tile_count; ++tile_n) {
        double acc[kTileSide] = {};
        for (size_t tile_k = 0; tile_k < tiles_k; ++tile_k) {
            if (matrix.codebook == 2)
                decode_tile_f32_hot_bits_avx2_cb2<Bits>(matrix, static_cast<int64_t>(tile_k),
                                                        static_cast<int64_t>(tile_n), tile_f32.data());
            else
                decode_tile_f32_hot_bits<Bits>(matrix, static_cast<int64_t>(tile_k), static_cast<int64_t>(tile_n),
                                               lut, tile_f32.data());
            const size_t row_base = tile_k * kTileSide;
            accumulate_tile_f32_avx2(tile_f32.data(), input_basis + row_base, acc);
        }
        for (int col = 0; col < kTileSide; ++col)
            dot[(tile_n - first_tile) * kTileSide + static_cast<size_t>(col)] = static_cast<float>(acc[col]);
    }
}

#endif  // EXL3_HAVE_TARGET_AVX2

void linear_dot_range_dispatch(const MatrixView& matrix, const float* input_basis, size_t first_tile,
                               size_t tile_count, float* dot, const Mul1Input* mul1 = nullptr) {
    if (mul1 != nullptr && linear_mul1_dot_range(matrix, *mul1, first_tile, tile_count, dot)) return;
    const float* const lut = codebook_lut_f32()[static_cast<size_t>(matrix.codebook)].data();
#if EXL3_HAVE_TARGET_AVX2
    static const bool use_avx2 = [] {
        const char* const force_generic = std::getenv("STRATA_NO_EXL3_AVX2");
        return !(force_generic != nullptr && force_generic[0] == '1') && strata::kernels::cpu::cpu_avx2_ok();
    }();
    if (use_avx2) {
        switch (matrix.bits) {
            case 1: linear_dot_range_f32_avx2<1>(matrix, input_basis, first_tile, tile_count, dot, lut); break;
            case 2: linear_dot_range_f32_avx2<2>(matrix, input_basis, first_tile, tile_count, dot, lut); break;
            case 3: linear_dot_range_f32_avx2<3>(matrix, input_basis, first_tile, tile_count, dot, lut); break;
            case 4: linear_dot_range_f32_avx2<4>(matrix, input_basis, first_tile, tile_count, dot, lut); break;
            case 5: linear_dot_range_f32_avx2<5>(matrix, input_basis, first_tile, tile_count, dot, lut); break;
            case 6: linear_dot_range_f32_avx2<6>(matrix, input_basis, first_tile, tile_count, dot, lut); break;
            case 7: linear_dot_range_f32_avx2<7>(matrix, input_basis, first_tile, tile_count, dot, lut); break;
            default: linear_dot_range_f32_avx2<8>(matrix, input_basis, first_tile, tile_count, dot, lut); break;
        }
        return;
    }
#endif
    switch (matrix.bits) {
        case 1: linear_dot_range_f32<1>(matrix, input_basis, first_tile, tile_count, dot, lut); break;
        case 2: linear_dot_range_f32<2>(matrix, input_basis, first_tile, tile_count, dot, lut); break;
        case 3: linear_dot_range_f32<3>(matrix, input_basis, first_tile, tile_count, dot, lut); break;
        case 4: linear_dot_range_f32<4>(matrix, input_basis, first_tile, tile_count, dot, lut); break;
        case 5: linear_dot_range_f32<5>(matrix, input_basis, first_tile, tile_count, dot, lut); break;
        case 6: linear_dot_range_f32<6>(matrix, input_basis, first_tile, tile_count, dot, lut); break;
        case 7: linear_dot_range_f32<7>(matrix, input_basis, first_tile, tile_count, dot, lut); break;
        default: linear_dot_range_f32<8>(matrix, input_basis, first_tile, tile_count, dot, lut); break;
    }
}

void hadamard_normalized(float* values) {
    for (int width = 1; width < kHadamardSide; width *= 2) {
        for (int base = 0; base < kHadamardSide; base += 2 * width) {
            for (int i = 0; i < width; ++i) {
                const float left = values[base + i];
                const float right = values[base + width + i];
                values[base + i] = left + right;
                values[base + width + i] = left - right;
            }
        }
    }
    for (int i = 0; i < kHadamardSide; ++i) {
        volatile float scaled = values[i] * kHadamardScale;
        values[i] = scaled;
    }
}

Status validate_linear(const MatrixView& matrix, const float* x, float* y) {
    const Status status = validate(matrix);
    if (status != Status::ok) return status;
    if (x == nullptr || y == nullptr) return Status::missing_data;
    // EXL3's suh/svh are H128 transform vectors.  A descriptor lacking one is
    // malformed for this operation; silently treating it as identity changes
    // the model, so report it to the caller.
    if (matrix.suh == nullptr || matrix.svh == nullptr) return Status::missing_data;
    if (matrix.k % kHadamardSide != 0 || matrix.n % kHadamardSide != 0) return Status::unsupported_layout;
    return Status::ok;
}

Status linear_with_scratch(const MatrixView& matrix, const float* x, float* y, LinearScratch& scratch) {
    const Status status = validate_linear(matrix, x, y);
    if (status != Status::ok) return status;
    const size_t k = static_cast<size_t>(matrix.k);
    const size_t n = static_cast<size_t>(matrix.n);
    if (scratch.input_basis == nullptr || scratch.input_count < k || scratch.tile == nullptr || scratch.tile_count < 256)
        return Status::scratch_too_small;

    for (size_t base = 0; base < k; base += kHadamardSide) {
        for (size_t i = 0; i < kHadamardSide; ++i) {
            const float scale = f32_from_f16(matrix.suh[base + i]);
            volatile float product = x[base + i] * scale;
            scratch.input_basis[base + i] = product;
        }
        hadamard_normalized(scratch.input_basis + base);
    }

    Mul1Input mul1;
    prepare_mul1_input(matrix, scratch.input_basis, scratch.mul1_duplicated, scratch.mul1_count, mul1);
    linear_dot_range_dispatch(matrix, scratch.input_basis, 0, n / kTileSide, y, &mul1);

    for (size_t base = 0; base < n; base += kHadamardSide) {
        hadamard_normalized(y + base);
        for (size_t i = 0; i < kHadamardSide; ++i) {
            volatile float scaled = y[base + i] * f32_from_f16(matrix.svh[base + i]);
            y[base + i] = scaled;
            if (matrix.bias != nullptr) {
                volatile float with_bias = y[base + i] + f32_from_f16(matrix.bias[base + i]);
                y[base + i] = with_bias;
            }
        }
    }
    return Status::ok;
}

Status prepare_input_basis_impl(const MatrixView& matrix, const float* x, float* basis) {
    const Status status = validate_linear(matrix, x, basis);
    if (status != Status::ok) return status;
    for (size_t base = 0; base < static_cast<size_t>(matrix.k); base += kHadamardSide) {
        for (size_t i = 0; i < kHadamardSide; ++i) {
            const float scale = f32_from_f16(matrix.suh[base + i]);
            volatile float product = x[base + i] * scale;
            basis[base + i] = product;
        }
        hadamard_normalized(basis + base);
    }
    return Status::ok;
}

Status linear_output_block_impl(const MatrixView& matrix, const float* input_basis, int64_t col0, float* y,
                                LinearScratch& scratch) {
    const Status status = validate_linear(matrix, input_basis, y);
    if (status != Status::ok) return status;
    if (col0 < 0 || col0 % kHadamardSide != 0 || col0 > matrix.n - kHadamardSide ||
        scratch.input_basis == nullptr || scratch.input_count < kHadamardSide || scratch.tile == nullptr ||
        scratch.tile_count < kTileValues)
        return Status::scratch_too_small;
    linear_dot_range_dispatch(matrix, input_basis, static_cast<size_t>(col0 / kTileSide),
                              kHadamardSide / kTileSide, scratch.input_basis, scratch.mul1_prepared);
    hadamard_normalized(scratch.input_basis);
    for (size_t i = 0; i < kHadamardSide; ++i) {
        volatile float scaled = scratch.input_basis[i] * f32_from_f16(matrix.svh[col0 + static_cast<int64_t>(i)]);
        y[col0 + static_cast<int64_t>(i)] = scaled;
        if (matrix.bias != nullptr) {
            volatile float with_bias = y[col0 + static_cast<int64_t>(i)] +
                                       f32_from_f16(matrix.bias[col0 + static_cast<int64_t>(i)]);
            y[col0 + static_cast<int64_t>(i)] = with_bias;
        }
    }
    return Status::ok;
}

void silu_mul_block_impl(const float* gate, const float* up, float* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const float value = gate[i];
        out[i] = (value / (1.0f + std::exp(-value))) * up[i];
    }
}

}  // namespace

const char* status_string(Status status) {
    switch (status) {
        case Status::ok: return "ok";
        case Status::invalid_argument: return "invalid argument";
        case Status::unsupported_bits: return "unsupported EXL3 bit width";
        case Status::unsupported_codebook: return "unsupported EXL3 codebook";
        case Status::unsupported_layout: return "unsupported EXL3 layout";
        case Status::missing_data: return "missing EXL3 data";
        case Status::output_too_small: return "output buffer too small";
        case Status::scratch_too_small: return "scratch buffer too small";
        case Status::size_overflow: return "EXL3 size overflow";
    }
    return "unknown EXL3 status";
}

Status prepare_input_basis(const MatrixView& matrix, const float* x, float* basis) {
    return prepare_input_basis_impl(matrix, x, basis);
}

Status linear_output_block(const MatrixView& matrix, const float* input_basis, int64_t col0, float* y,
                           LinearScratch& scratch) {
    return linear_output_block_impl(matrix, input_basis, col0, y, scratch);
}

void silu_mul_block(const float* gate, const float* up, float* out, size_t count) {
    silu_mul_block_impl(gate, up, out, count);
}

Status validate(const MatrixView& matrix) {
    if (matrix.k <= 0 || matrix.n <= 0 || matrix.k % kTileSide != 0 || matrix.n % kTileSide != 0)
        return Status::unsupported_layout;
    if (matrix.bits < 1 || matrix.bits > 8) return Status::unsupported_bits;
    if (matrix.codebook < 0 || matrix.codebook > 2) return Status::unsupported_codebook;
    if (matrix.trellis == nullptr) return Status::missing_data;
    size_t words;
    if (!checked_matrix_words(matrix, words)) return Status::size_overflow;
    (void) words;
    return Status::ok;
}

Status unpack_trellis_states(const uint16_t* packed, int bits, uint16_t* states, size_t state_count) {
    if (packed == nullptr || states == nullptr) return Status::missing_data;
    if (bits < 1 || bits > 8) return Status::unsupported_bits;
    if (state_count < kTileValues) return Status::output_too_small;
    const size_t words32_count = static_cast<size_t>(bits) * 8;
    for (int state = 0; state < kTileValues; ++state) {
        const int b0 = state * bits + bits - 16 + 256 * bits;
        const int b1 = b0 + 16;
        const size_t i0 = static_cast<size_t>(b0 / 32) % words32_count;
        const size_t i1 = static_cast<size_t>((b1 - 1) / 32) % words32_count;
        const int shift = ((b1 - 1) / 32 + 1) * 32 - b1;
        const uint64_t merged = (static_cast<uint64_t>(load_u32(packed, i0)) << 32) | load_u32(packed, i1);
        states[state] = static_cast<uint16_t>((merged >> shift) & 0xffffu);
    }
    return Status::ok;
}

Status decode_codebook(const uint16_t* states, size_t state_count, int codebook, uint16_t* values,
                       size_t value_count) {
    if (states == nullptr || values == nullptr) return Status::missing_data;
    if (codebook < 0 || codebook > 2) return Status::unsupported_codebook;
    if (value_count < state_count) return Status::output_too_small;
    for (size_t i = 0; i < state_count; ++i) values[i] = decode_state_half_lut(states[i], codebook);
    return Status::ok;
}

Status decode_tile(const MatrixView& matrix, int64_t tile_k, int64_t tile_n, uint16_t* values, size_t value_count) {
    const Status status = validate(matrix);
    if (status != Status::ok) return status;
    if (values == nullptr) return Status::missing_data;
    if (value_count < kTileValues) return Status::output_too_small;
    const int64_t tiles_k = matrix.k / kTileSide;
    const int64_t tiles_n = matrix.n / kTileSide;
    if (tile_k < 0 || tile_k >= tiles_k || tile_n < 0 || tile_n >= tiles_n) return Status::invalid_argument;
    const size_t packed_size = static_cast<size_t>(16 * matrix.bits);
    const size_t tile_index = strata::kernels::exl3::trellis_tile_index(matrix.tile_order, tile_k, tile_n, tiles_k,
                                                                         tiles_n);
    const uint16_t* packed = matrix.trellis + tile_index * packed_size;
    std::array<uint16_t, kTileValues> states{};
    const Status state_status = unpack_trellis_states(packed, matrix.bits, states.data(), states.size());
    if (state_status != Status::ok) return state_status;
    std::array<uint16_t, kTileValues> decoded{};
    const Status decode_status = decode_codebook(states.data(), states.size(), matrix.codebook, decoded.data(), decoded.size());
    if (decode_status != Status::ok) return decode_status;
    for (int state = 0; state < kTileValues; ++state) {
        values[strata::kernels::exl3::tile_element_for_state(state)] = decoded[static_cast<size_t>(state)];
    }
    return Status::ok;
}

Status reconstruct(const MatrixView& matrix, uint16_t* values, size_t value_count) {
    const Status status = validate(matrix);
    if (status != Status::ok) return status;
    if (values == nullptr) return Status::missing_data;
    size_t count;
    if (!checked_mul(static_cast<size_t>(matrix.k), static_cast<size_t>(matrix.n), count)) return Status::size_overflow;
    if (value_count < count) return Status::output_too_small;
    const int64_t tiles_k = matrix.k / kTileSide;
    const int64_t tiles_n = matrix.n / kTileSide;
    std::array<uint16_t, kTileValues> tile{};
    for (int64_t tile_k = 0; tile_k < tiles_k; ++tile_k) {
        for (int64_t tile_n = 0; tile_n < tiles_n; ++tile_n) {
            const Status tile_status = decode_tile(matrix, tile_k, tile_n, tile.data(), tile.size());
            if (tile_status != Status::ok) return tile_status;
            for (int row = 0; row < kTileSide; ++row) {
                const size_t dst = (static_cast<size_t>(tile_k) * kTileSide + static_cast<size_t>(row)) *
                                   static_cast<size_t>(matrix.n) + static_cast<size_t>(tile_n) * kTileSide;
                for (int col = 0; col < kTileSide; ++col) values[dst + static_cast<size_t>(col)] = tile[row * kTileSide + col];
            }
        }
    }
    return Status::ok;
}

size_t scratch_bytes(const MatrixView& matrix) {
    if (validate(matrix) != Status::ok) return 0;
    size_t input_bytes;
    if (!checked_mul(static_cast<size_t>(matrix.k), sizeof(float), input_bytes)) return 0;
    size_t result = input_bytes;
    if (result > std::numeric_limits<size_t>::max() - 256 * sizeof(uint16_t)) return 0;
    return result + 256 * sizeof(uint16_t);
}

Status linear(const MatrixView& matrix, const float* x, float* y, LinearScratch& scratch) {
    return linear_with_scratch(matrix, x, y, scratch);
}

Status linear(const MatrixView& matrix, const float* x, float* y) {
    const Status status = validate_linear(matrix, x, y);
    if (status != Status::ok) return status;
    std::vector<float> input(static_cast<size_t>(matrix.k));
    std::array<uint16_t, kTileValues> tile{};
    LinearScratch scratch{input.data(), input.size(), tile.data(), tile.size()};
    return linear_with_scratch(matrix, x, y, scratch);
}

}  // namespace strata::kernels::cpu::exl3
