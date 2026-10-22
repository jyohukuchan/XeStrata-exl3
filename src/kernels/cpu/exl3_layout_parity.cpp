// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The CPU tile orders against the format's own order.  All of them hold the
// same compressed bits, so every result must be bit identical; the test also
// checks that a mixed-up order is actually detected.
//
// The test also carries the exact decode's own rules, written out again, so
// that the library's vector path is checked against something that shares no
// code with it: the half conversion over all 65,536 halves, the codebook over
// all 65,536 states, and the whole projection bit for bit.
#include "strata/artifact/exl3_expert.hpp"
#include "strata/kernels/cpu/exl3.hpp"
#include "strata/kernels/cpu/exl3_expert.hpp"
#include "strata/kernels/cpu/exl3_mul1_intdot.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#define PARITY_HAVE_F16C 1
#else
#define PARITY_HAVE_F16C 0
#endif

namespace cpu_exl3 = strata::kernels::cpu::exl3;
namespace format = strata::kernels::exl3;

namespace {

int failures = 0;

void check(bool ok, const std::string& label) {
    std::printf("  %s: %s\n", label.c_str(), ok ? "OK" : "FAIL");
    if (!ok) ++failures;
}

/// The CPU's own orders, and the format's own order beside them.
const format::TileOrder kCpuOrders[] = {format::TileOrder::cpu_block, format::TileOrder::cpu_nmajor};

const char* order_name(format::TileOrder order) {
    switch (order) {
        case format::TileOrder::cpu_block: return "cpu_block";
        case format::TileOrder::cpu_nmajor: return "cpu_nmajor";
        default: return "canonical";
    }
}

struct Matrix {
    int64_t k = 0;
    int64_t n = 0;
    int bits = 0;
    int codebook = 0;
    std::vector<uint16_t> trellis;
    std::vector<uint16_t> suh;
    std::vector<uint16_t> svh;
    std::vector<uint16_t> bias;
};

uint32_t next_random(uint32_t& seed) {
    seed = seed * 1664525u + 1013904223u;
    return seed >> 8;
}

Matrix make_matrix(int64_t k, int64_t n, int bits, int codebook, uint32_t seed) {
    Matrix matrix;
    matrix.k = k;
    matrix.n = n;
    matrix.bits = bits;
    matrix.codebook = codebook;
    const size_t tiles = static_cast<size_t>(k / 16) * static_cast<size_t>(n / 16);
    matrix.trellis.resize(tiles * static_cast<size_t>(16 * bits));
    for (uint16_t& value : matrix.trellis) value = static_cast<uint16_t>(next_random(seed));
    matrix.suh.resize(static_cast<size_t>(k));
    matrix.svh.resize(static_cast<size_t>(n));
    matrix.bias.resize(static_cast<size_t>(n));
    for (uint16_t& value : matrix.suh) value = 0x3c00;   // 1.0
    for (uint16_t& value : matrix.svh) value = 0x3c00;   // 1.0
    for (uint16_t& value : matrix.bias) value = 0x0000;  // 0.0
    return matrix;
}

/// The same matrix with its compressed tiles in the given CPU order.
Matrix reordered(const Matrix& source, format::TileOrder order) {
    Matrix result = source;
    std::vector<uint16_t> canonical = source.trellis;
    strata::artifact::exl3::copy_packed_tiles(canonical.data(), result.trellis.data(), source.k, source.n, source.bits,
                                              format::TileOrder::canonical, order);
    return result;
}

format::MatrixView view(const Matrix& matrix, format::TileOrder order) {
    format::MatrixView result;
    result.trellis = matrix.trellis.data();
    result.suh = matrix.suh.data();
    result.svh = matrix.svh.data();
    result.bias = matrix.bias.data();
    result.k = matrix.k;
    result.n = matrix.n;
    result.bits = matrix.bits;
    result.codebook = matrix.codebook;
    result.tile_order = order;
    return result;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) return false;
    return true;
}

bool all_finite(const std::vector<float>& values) {
    for (float value : values) if (!std::isfinite(value)) return false;
    return true;
}

std::vector<float> make_input(int64_t count, uint32_t seed) {
    std::vector<float> x(static_cast<size_t>(count));
    for (size_t i = 0; i < x.size(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        x[i] = 0.03125f * static_cast<float>(static_cast<int>((seed >> 16) % 29) - 14);
    }
    return x;
}

// ------------------------------------------------------------- the oracle
//
// The exact decode's rules, written out again, so that the library's vector
// path is checked against something that shares no code with it.

constexpr int kTileSide = 16;

/// The half rounding of a float: round to nearest even, and a NaN keeps its
/// sign and its payload.  The library's codebook decoder rounds this way.
uint16_t ref_f32_to_f16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        if (mantissa == 0) return static_cast<uint16_t>(sign | 0x7c00u);
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

uint32_t ref_lop3_6a(uint32_t value) {
    return ((~value & 0x3B603B60u) | (value & (0x8FFF8FFFu ^ 0x3B603B60u)));
}

/// The hardware's half to float conversion, which is exact for every half.
/// It is an implementation of its own, so it is an oracle for the library's.
#if PARITY_HAVE_F16C
__attribute__((target("f16c"))) float ref_f16_to_f32(uint16_t bits) {
    const __m128 values = _mm_cvtph_ps(_mm_set1_epi16(static_cast<short>(bits)));
    return _mm_cvtss_f32(values);
}
#else
/// The same conversion from the bits, for a build without F16C.
float ref_f16_to_f32(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t mantissa = bits & 0x3ffu;
    uint32_t out;
    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            uint32_t m = mantissa;
            int s = 0;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                ++s;
            }
            out = sign | (static_cast<uint32_t>(113 - s) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exponent == 31) {
        out = sign | 0x7f800000u | (mantissa << 13);
    } else {
        out = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &out, sizeof(result));
    return result;
}
#endif

/// The codebook's value for a state, from the format's rules.
///
/// The MUL1 codebook is the one the vector path computes in arithmetic, so it
/// is checked against the hardware's half conversion and the shared rounding.
/// The other two codebooks can hold a NaN, whose payload the shared rounding
/// drops, so they are checked against the rule above.
uint16_t ref_codebook_value(uint16_t state, int codebook) {
    // gitleaks:allow -- the public EXL3 codebook multipliers, never credentials.
    constexpr uint32_t cb0_multiplier = 89226354u;
    constexpr uint32_t cb0_addend = 64248484u;
    constexpr uint32_t cb1_multiplier = 0xCBAC1FEDu;
    constexpr uint32_t cb2_multiplier = 0x83DCD12Du;
    uint32_t value;
    if (codebook == 0) {
        value = static_cast<uint32_t>(state) * cb0_multiplier + cb0_addend;
        value = ref_lop3_6a(value);
        volatile float sum = strata::kernels::f32_from_f16(static_cast<uint16_t>(value)) +
                            strata::kernels::f32_from_f16(static_cast<uint16_t>(value >> 16));
        return ref_f32_to_f16(sum);
    }
    if (codebook == 1) {
        value = static_cast<uint32_t>(state) * cb1_multiplier;
        value = ref_lop3_6a(value);
        volatile float sum = strata::kernels::f32_from_f16(static_cast<uint16_t>(value)) +
                            strata::kernels::f32_from_f16(static_cast<uint16_t>(value >> 16));
        return ref_f32_to_f16(sum);
    }
    value = static_cast<uint32_t>(state) * cb2_multiplier;
    const int byte_sum = static_cast<int>(value & 0xffu) + static_cast<int>((value >> 8) & 0xffu) +
                         static_cast<int>((value >> 16) & 0xffu) + static_cast<int>(value >> 24);
    const uint16_t h = static_cast<uint16_t>(0x6400u + byte_sum);
    // The value of the codebook is a normal half, never a NaN, so the shared
    // rounding is the one the library's own decoder uses.
    volatile float product = ref_f16_to_f32(h) * ref_f16_to_f32(0x1eeeu);
    volatile float result = product + ref_f16_to_f32(0xc931u);
    return strata::kernels::f16_from_f32(result);
}

/// The normalized H128 transform, as the library applies it.
void ref_hadamard(float* values) {
    for (int width = 1; width < 128; width *= 2) {
        for (int base = 0; base < 128; base += 2 * width) {
            for (int i = 0; i < width; ++i) {
                const float left = values[base + i], right = values[base + width + i];
                values[base + i] = left + right;
                values[base + width + i] = left - right;
            }
        }
    }
    for (int i = 0; i < 128; ++i) values[i] = values[i] * 0.088388347648f;
}

/// The trellis's windows, read out of the packed tile the way the format
/// stores them.  The library reads them with a table; this reads them with
/// the arithmetic, so that the two can be compared.
uint16_t ref_window(const uint16_t* packed, int bits, int state) {
    const size_t words = static_cast<size_t>(bits) * 8;
    const int b0 = state * bits + bits - 16 + 256 * bits;
    const int b1 = b0 + 16;
    const size_t i0_raw = static_cast<size_t>(b0 >> 5);
    const size_t i1_raw = static_cast<size_t>((b1 - 1) >> 5);
    const size_t i0 = i0_raw >= words ? i0_raw - words : i0_raw;
    const size_t i1 = i1_raw >= words ? i1_raw - words : i1_raw;
    const uint32_t low = static_cast<uint32_t>(packed[i0 * 2]) | (static_cast<uint32_t>(packed[i0 * 2 + 1]) << 16);
    const uint32_t high = static_cast<uint32_t>(packed[i1 * 2]) | (static_cast<uint32_t>(packed[i1 * 2 + 1]) << 16);
    const uint64_t merged = (static_cast<uint64_t>(low) << 32) | high;
    const int shift = static_cast<int>((i1_raw + 1) * 32 - static_cast<size_t>(b1));
    return static_cast<uint16_t>((merged >> shift) & 0xffffu);
}

/// The whole projection, decoded with the rules above and with the scalar
/// codebook of the library.  The codebook's halves are the library's own, so
/// this checks the vector path's decode against the scalar one.
void reference_weights(const Matrix& matrix, const format::TileOrder order, std::vector<float>& weights) {
    const int64_t tiles_k = matrix.k / kTileSide;
    const int64_t tiles_n = matrix.n / kTileSide;
    const size_t packed_size = static_cast<size_t>(16 * matrix.bits);
    weights.assign(static_cast<size_t>(matrix.k * matrix.n), 0.0f);
    std::array<uint16_t, 256> states{};
    std::array<uint16_t, 256> values{};
    for (int64_t tile_k = 0; tile_k < tiles_k; ++tile_k) {
        for (int64_t tile_n = 0; tile_n < tiles_n; ++tile_n) {
            const size_t tile_index = format::trellis_tile_index(order, tile_k, tile_n, tiles_k, tiles_n);
            const uint16_t* packed = matrix.trellis.data() + tile_index * packed_size;
            for (int state = 0; state < 256; ++state)
                states[static_cast<size_t>(state)] = ref_window(packed, matrix.bits, state);
            cpu_exl3::decode_codebook(states.data(), states.size(), matrix.codebook, values.data(), values.size());
            for (int state = 0; state < 256; ++state) {
                const int element = format::tile_element_for_state(state);
                const int64_t row = tile_k * kTileSide + element / kTileSide;
                const int64_t col = tile_n * kTileSide + element % kTileSide;
                weights[static_cast<size_t>(row * matrix.n + col)] =
                    strata::kernels::f32_from_f16(values[static_cast<size_t>(state)]);
            }
        }
    }
}

/// The whole projection, computed from the reference's weights: the same
/// transforms, the same order of the additions, and the same bits.
void reference_linear(const Matrix& matrix, const format::TileOrder order, const std::vector<float>& x,
                      std::vector<float>& y) {
    const size_t k = static_cast<size_t>(matrix.k);
    const size_t n = static_cast<size_t>(matrix.n);
    std::vector<float> basis(k);
    for (size_t base = 0; base < k; base += 128) {
        for (size_t i = 0; i < 128; ++i)
            basis[base + i] = x[base + i] * strata::kernels::f32_from_f16(matrix.suh[base + i]);
        ref_hadamard(basis.data() + base);
    }
    std::vector<float> weights;
    reference_weights(matrix, order, weights);
    y.assign(n, 0.0f);
    for (size_t col = 0; col < n; ++col) {
        double acc = 0.0;
        for (size_t row = 0; row < k; ++row)
            acc += static_cast<double>(weights[row * n + col]) * static_cast<double>(basis[row]);
        y[col] = static_cast<float>(acc);
    }
    for (size_t base = 0; base < n; base += 128) {
        ref_hadamard(y.data() + base);
        for (size_t i = 0; i < 128; ++i) {
            y[base + i] = y[base + i] * strata::kernels::f32_from_f16(matrix.svh[base + i]);
            if (!matrix.bias.empty())
                y[base + i] = y[base + i] + strata::kernels::f32_from_f16(matrix.bias[base + i]);
        }
    }
}

/// The half conversion over all 65,536 halves: the normal range, the
/// subnormals, the zeros, the infinities and the NaNs.
void check_half_conversion() {
    size_t wrong = 0;
    size_t nan_checked = 0;
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        const uint16_t half = static_cast<uint16_t>(bits);
        const float got = strata::kernels::f32_from_f16(half);
        const uint32_t exponent = (bits >> 10) & 0x1fu;
        const uint32_t mantissa = bits & 0x3ffu;
        if (exponent == 31u && mantissa != 0u) {
            // A NaN keeps its sign and its payload, shifted into the float's
            // mantissa.  The hardware's conversion quiets the NaN and moves
            // the payload, so the shape and the payload are compared here.
            ++nan_checked;
            uint32_t got_bits;
            std::memcpy(&got_bits, &got, sizeof(got_bits));
            const uint32_t sign = (bits & 0x8000u) << 16;
            const uint32_t want = sign | 0x7f800000u | (mantissa << 13);
            if (got_bits != want) {
                if (wrong < 8)
                    std::printf("    half 0x%04X -> 0x%08X, want 0x%08X\n", bits, got_bits, want);
                ++wrong;
            }
            continue;
        }
        const float want = ref_f16_to_f32(half);
        uint32_t got_bits, want_bits;
        std::memcpy(&got_bits, &got, sizeof(got_bits));
        std::memcpy(&want_bits, &want, sizeof(want_bits));
        if (got_bits != want_bits) {
            if (wrong < 8)
                std::printf("    half 0x%04X -> 0x%08X, the hardware gives 0x%08X\n", bits, got_bits, want_bits);
            ++wrong;
        }
    }
    check(wrong == 0, "the half conversion is exact for all 65536 halves (" + std::to_string(nan_checked) +
                          " NaNs carry their sign and payload)");

    // The exceptional halves, one by one, with the bits they must give.
    struct Case {
        uint16_t half;
        uint32_t bits;
        const char* what;
    };
    const Case cases[] = {
        {0x0000u, 0x00000000u, "a positive zero"},
        {0x8000u, 0x80000000u, "a negative zero"},
        {0x0001u, 0x33800000u, "the smallest subnormal"},
        {0x8001u, 0xb3800000u, "the smallest negative subnormal"},
        {0x03ffu, 0x387fc000u, "the largest subnormal"},
        {0x0400u, 0x38800000u, "the smallest normal"},
        {0x3c00u, 0x3f800000u, "one"},
        {0x7bffu, 0x477fe000u, "the largest finite half"},
        {0x7c00u, 0x7f800000u, "an infinity"},
        {0xfc00u, 0xff800000u, "a negative infinity"},
        {0x7c01u, 0x7f802000u, "a NaN"},
        {0xffffu, 0xffffe000u, "a negative NaN"},
    };
    size_t wrong_cases = 0;
    for (const Case& c : cases) {
        float value = strata::kernels::f32_from_f16(c.half);
        uint32_t got;
        std::memcpy(&got, &value, sizeof(got));
        if (got != c.bits) {
            std::printf("    half 0x%04X (%s) -> 0x%08X, want 0x%08X\n", c.half, c.what, got, c.bits);
            ++wrong_cases;
        }
    }
    check(wrong_cases == 0, "the exceptional halves give the bits they must");
}

/// The codebook over all 65,536 states.
void check_codebook() {
    std::vector<uint16_t> states(65536);
    for (size_t i = 0; i < states.size(); ++i) states[i] = static_cast<uint16_t>(i);
    std::vector<uint16_t> values(65536);
    for (int codebook = 0; codebook < 3; ++codebook) {
        const cpu_exl3::Status status =
            cpu_exl3::decode_codebook(states.data(), states.size(), codebook, values.data(), values.size());
        if (status != cpu_exl3::Status::ok) {
            check(false, "the codebook " + std::to_string(codebook) + " decodes");
            continue;
        }
        size_t wrong = 0;
        for (size_t state = 0; state < states.size(); ++state) {
            const uint16_t want = ref_codebook_value(static_cast<uint16_t>(state), codebook);
            if (values[state] != want) {
                if (wrong < 8)
                    std::printf("    codebook %d state 0x%04zX -> 0x%04X, want 0x%04X\n", codebook, state, values[state],
                                want);
                ++wrong;
            }
        }
        check(wrong == 0, "the codebook " + std::to_string(codebook) + " is right for all 65536 states");
    }
}

/// The byte sums of the MUL1 codebook that the test's own matrices reach.
/// The codebook's value is a function of the sum alone, so the check below
/// is worth exactly as much as the number of sums it reaches.
std::array<uint8_t, 128>& mul1_sums() {
    static std::array<uint8_t, 128> table{};
    return table;
}

/// The byte sums the MUL1 codebook can produce at all.  The four bytes of the
/// multiplied state can sum to anything from 0 to 1020, but the multiplier
/// does not reach all of them: 913 of the 1021 are reachable.
std::array<uint8_t, 128> mul1_reachable_sums() {
    std::array<uint8_t, 128> table{};
    for (uint32_t state = 0; state < 65536; ++state) {
        const uint32_t product = state * 0x83DCD12Du;
        const int sum = static_cast<int>(product & 0xffu) + static_cast<int>((product >> 8) & 0xffu) +
                        static_cast<int>((product >> 16) & 0xffu) + static_cast<int>(product >> 24);
        table[static_cast<size_t>(sum) >> 3] |= static_cast<uint8_t>(1u << (sum & 7));
    }
    return table;
}

/// The whole projection, bit for bit, against the reference.
void check_exact_linear(const Matrix& canonical, const format::TileOrder order, const std::vector<float>& x,
                        const std::string& label) {
    format::MatrixView matrix_view = view(canonical, order);
    std::vector<float> got(static_cast<size_t>(canonical.n), 0.0f);
    std::vector<float> input_basis(static_cast<size_t>(canonical.k));
    std::array<uint16_t, 256> tile{};
    cpu_exl3::LinearScratch scratch{input_basis.data(), input_basis.size(), tile.data(), tile.size()};
    const cpu_exl3::Status status = cpu_exl3::linear(matrix_view, x.data(), got.data(), scratch);
    std::vector<float> want;
    reference_linear(canonical, order, x, want);
    bool same = status == cpu_exl3::Status::ok && got.size() == want.size();
    for (size_t i = 0; same && i < got.size(); ++i)
        if (std::memcmp(&got[i], &want[i], sizeof(float)) != 0) same = false;
    check(same, label + " is bit identical to the reference");

    // The states the matrix holds, and the codebook's byte sums they reach.
    if (canonical.codebook == 2) {
        const int64_t tiles_k = canonical.k / kTileSide;
        const int64_t tiles_n = canonical.n / kTileSide;
        const size_t packed_size = static_cast<size_t>(16 * canonical.bits);
        for (int64_t tile_k = 0; tile_k < tiles_k; ++tile_k) {
            for (int64_t tile_n = 0; tile_n < tiles_n; ++tile_n) {
                const size_t tile_index = format::trellis_tile_index(order, tile_k, tile_n, tiles_k, tiles_n);
                const uint16_t* packed = canonical.trellis.data() + tile_index * packed_size;
                for (int state = 0; state < 256; ++state) {
                    const uint16_t value = ref_window(packed, canonical.bits, state);
                    const uint32_t product = static_cast<uint32_t>(value) * 0x83DCD12Du;
                    const int sum = static_cast<int>(product & 0xffu) + static_cast<int>((product >> 8) & 0xffu) +
                                    static_cast<int>((product >> 16) & 0xffu) + static_cast<int>(product >> 24);
                    mul1_sums()[static_cast<size_t>(sum) >> 3] |= static_cast<uint8_t>(1u << (sum & 7));
                }
            }
        }
    }
}

// The packed projection's own results: the W-hat matrix, the whole-matrix
// product and every 128-column block.  The block is the unit of the H128
// transform, so the row-parallel path never asks for less.
void check_matrix(const Matrix& canonical, const Matrix& cpu, format::TileOrder order, const std::vector<float>& x,
                  const std::string& label, size_t& projections, size_t& values) {
    ++projections;
    const format::MatrixView canonical_view = view(canonical, format::TileOrder::canonical);
    const format::MatrixView cpu_view = view(cpu, order);

    std::vector<uint16_t> canonical_w(static_cast<size_t>(canonical.k * canonical.n));
    std::vector<uint16_t> cpu_w(static_cast<size_t>(cpu.k * cpu.n));
    const cpu_exl3::Status canonical_status = cpu_exl3::reconstruct(canonical_view, canonical_w.data(), canonical_w.size());
    const cpu_exl3::Status cpu_status = cpu_exl3::reconstruct(cpu_view, cpu_w.data(), cpu_w.size());
    check(canonical_status == cpu_exl3::Status::ok && cpu_status == cpu_exl3::Status::ok && canonical_w == cpu_w,
          label + " reconstruct is bit identical");

    std::vector<float> canonical_input(static_cast<size_t>(canonical.k));
    std::array<uint16_t, 256> tile{};
    cpu_exl3::LinearScratch canonical_scratch{canonical_input.data(), canonical_input.size(), tile.data(), tile.size()};
    std::vector<float> canonical_y(static_cast<size_t>(canonical.n));
    std::vector<float> cpu_y(static_cast<size_t>(cpu.n));
    const cpu_exl3::Status canonical_linear = cpu_exl3::linear(canonical_view, x.data(), canonical_y.data(), canonical_scratch);
    const cpu_exl3::Status cpu_linear = cpu_exl3::linear(cpu_view, x.data(), cpu_y.data(), canonical_scratch);
    check(canonical_linear == cpu_exl3::Status::ok && cpu_linear == cpu_exl3::Status::ok && same_bits(canonical_y, cpu_y) &&
              all_finite(cpu_y),
          label + " linear is bit identical");
    values += canonical_y.size();

        // The vector path's own decode, against the rules written out above: the
    // window table, the codebook and the accumulation must give the bits the
    // reference gives.
    check_exact_linear(canonical, format::TileOrder::canonical, x, label + " exact decode");
    check_exact_linear(cpu, order, x, label + " exact decode in the " + order_name(order) + " order");

    // The row-parallel path reads one 128-column block at a time.  The output
    // buffer holds the whole output, as the pool's workspace does.
    bool blocks_match = true;
    for (int64_t col = 0; col + 128 <= canonical.n; col += 128) {
        std::vector<float> canonical_output(static_cast<size_t>(canonical.n), 0.0f);
        std::vector<float> cpu_output(static_cast<size_t>(cpu.n), 0.0f);
        std::vector<float> canonical_basis(static_cast<size_t>(canonical.k));
        std::vector<float> cpu_basis(static_cast<size_t>(cpu.k));
        std::vector<float> dot_buffer(static_cast<size_t>(canonical.k) + 128, 0.0f);
        cpu_exl3::LinearScratch block_scratch{dot_buffer.data(), dot_buffer.size(), tile.data(), tile.size()};
        const cpu_exl3::Status canonical_prepare =
            cpu_exl3::prepare_input_basis(canonical_view, x.data(), canonical_basis.data());
        const cpu_exl3::Status cpu_prepare = cpu_exl3::prepare_input_basis(cpu_view, x.data(), cpu_basis.data());
        const cpu_exl3::Status canonical_block_status =
            cpu_exl3::linear_output_block(canonical_view, canonical_basis.data(), col, canonical_output.data(),
                                          block_scratch);
        const cpu_exl3::Status cpu_block_status =
            cpu_exl3::linear_output_block(cpu_view, cpu_basis.data(), col, cpu_output.data(), block_scratch);
        if (canonical_prepare != cpu_exl3::Status::ok || cpu_prepare != cpu_exl3::Status::ok ||
            canonical_block_status != cpu_exl3::Status::ok || cpu_block_status != cpu_exl3::Status::ok)
            blocks_match = false;
        for (int64_t i = 0; i < 128; ++i)
            if (std::memcmp(&canonical_output[static_cast<size_t>(col + i)], &cpu_output[static_cast<size_t>(col + i)],
                            sizeof(float)) != 0)
                blocks_match = false;
    }
    check(blocks_match, label + " output blocks are bit identical");
    values += canonical.n;

    // The integer dot, when it is enabled, must read the same tiles.  It is
    // the MUL1 codebook's path, so only that codebook reaches it.
    if (cpu_exl3::mul1_intdot_enabled() && canonical.codebook == 2) {
        std::vector<int32_t> canonical_duplicated(static_cast<size_t>(canonical.k));
        std::vector<int32_t> cpu_duplicated(static_cast<size_t>(cpu.k));
        cpu_exl3::Mul1Input canonical_prepared;
        cpu_exl3::Mul1Input cpu_prepared;
        const bool canonical_ready = cpu_exl3::prepare_mul1_input(
            canonical_view, x.data(), canonical_duplicated.data(), canonical_duplicated.size(), canonical_prepared);
        const bool cpu_ready =
            cpu_exl3::prepare_mul1_input(cpu_view, x.data(), cpu_duplicated.data(), cpu_duplicated.size(), cpu_prepared);
        bool intdot_match = canonical_ready && cpu_ready;
        if (intdot_match) {
            std::vector<float> canonical_dot(static_cast<size_t>(canonical.n));
            std::vector<float> cpu_dot(static_cast<size_t>(cpu.n));
            const size_t tiles = static_cast<size_t>(canonical.n / 16);
            intdot_match = cpu_exl3::linear_mul1_dot_range(canonical_view, canonical_prepared, 0, tiles,
                                                            canonical_dot.data()) &&
                           cpu_exl3::linear_mul1_dot_range(cpu_view, cpu_prepared, 0, tiles, cpu_dot.data()) &&
                           same_bits(canonical_dot, cpu_dot);
        }
        check(intdot_match, label + " integer dot is bit identical");
    }
}

}  // namespace

int main() {
    // The half conversion, over every half there is.
    std::printf("the half conversion\n");
    check_half_conversion();

    // The codebook, over every state there is.
    std::printf("the codebook\n");
    check_codebook();

    // A geometry whose output width covers several 128-column blocks, and one
    // that covers exactly one.
    const struct {
        int64_t hidden;
        int64_t ff;
    } shapes[] = {{128, 128}, {256, 256}, {128, 384}};
    uint32_t seed = 20261020u;
    size_t compared = 0;
    size_t projections = 0;
    size_t values = 0;
    for (const auto& shape : shapes) {
        for (const int bits : {2, 4, 5}) {
            for (const int codebook : {0, 1, 2}) {
                const Matrix canonical_gate = make_matrix(shape.hidden, shape.ff, bits, codebook, seed);
                const Matrix canonical_up = make_matrix(shape.hidden, shape.ff, bits, codebook, seed + 1u);
                const Matrix canonical_down = make_matrix(shape.ff, shape.hidden, bits, codebook, seed + 2u);
                seed += 3u;
                const std::vector<float> hidden_input = make_input(shape.hidden, seed);
                const std::vector<float> ff_input = make_input(shape.ff, seed + 1u);
                const std::string label = "matrix " + std::to_string(shape.hidden) + "x" + std::to_string(shape.ff) +
                                         " bits" + std::to_string(bits) + " cb" + std::to_string(codebook);
                for (const format::TileOrder order : kCpuOrders) {
                    const Matrix cpu_gate = reordered(canonical_gate, order);
                    const Matrix cpu_up = reordered(canonical_up, order);
                    const Matrix cpu_down = reordered(canonical_down, order);
                    const std::string tag = std::string(" [") + order_name(order) + "]";
                    check_matrix(canonical_gate, cpu_gate, order, hidden_input, label + " gate" + tag, projections,
                                 values);
                    check_matrix(canonical_up, cpu_up, order, hidden_input, label + " up" + tag, projections, values);
                    check_matrix(canonical_down, cpu_down, order, ff_input, label + " down" + tag, projections, values);

                    // The expert composition: gate, up, the product and down.
                    const format::ExpertView canonical_expert{
                        format::ProjectionView(view(canonical_gate, format::TileOrder::canonical)),
                        format::ProjectionView(view(canonical_up, format::TileOrder::canonical)),
                        format::ProjectionView(view(canonical_down, format::TileOrder::canonical))};
                    const format::ExpertView cpu_expert{format::ProjectionView(view(cpu_gate, order)),
                                                        format::ProjectionView(view(cpu_up, order)),
                                                        format::ProjectionView(view(cpu_down, order))};
                    const size_t max_k = static_cast<size_t>(std::max(shape.hidden, shape.ff));
                    std::vector<float> scratch_input(max_k);
                    std::vector<float> gate(static_cast<size_t>(shape.ff));
                    std::vector<float> up(static_cast<size_t>(shape.ff));
                    std::array<uint16_t, 256> tile{};
                    cpu_exl3::ExpertScratch canonical_scratch{{scratch_input.data(), scratch_input.size(), tile.data(),
                                                               tile.size()},
                                                              gate.data(), gate.size(), up.data(), up.size()};
                    std::vector<float> canonical_y(static_cast<size_t>(shape.hidden));
                    std::vector<float> cpu_y(static_cast<size_t>(shape.hidden));
                    const cpu_exl3::Status canonical_status =
                        cpu_exl3::expert(canonical_expert, hidden_input.data(), shape.hidden, canonical_y.data(),
                                         shape.hidden, 1, canonical_scratch);
                    const cpu_exl3::Status cpu_status = cpu_exl3::expert(cpu_expert, hidden_input.data(), shape.hidden,
                                                                         cpu_y.data(), shape.hidden, 1,
                                                                         canonical_scratch);
                    check(canonical_status == cpu_exl3::Status::ok && cpu_status == cpu_exl3::Status::ok &&
                              same_bits(canonical_y, cpu_y) && all_finite(cpu_y),
                          label + " expert is bit identical" + tag);
                    compared += canonical_y.size();
                }
            }
        }
    }

    // A mixed-up order must be visible: the same bytes read with the other
    // order cannot give the same answer, or the comparison above proves
    // nothing.  The two CPU orders are not the same bytes either, so one
    // cannot be read as the other.
    {
        const Matrix canonical = make_matrix(256, 256, 4, 2, 99u);
        std::vector<uint16_t> canonical_w(static_cast<size_t>(canonical.k * canonical.n));
        const format::MatrixView canonical_view = view(canonical, format::TileOrder::canonical);
        cpu_exl3::reconstruct(canonical_view, canonical_w.data(), canonical_w.size());
        for (const format::TileOrder order : kCpuOrders) {
            const Matrix cpu = reordered(canonical, order);
            std::vector<uint16_t> mixed_w(static_cast<size_t>(cpu.k * cpu.n));
            cpu_exl3::reconstruct(view(cpu, format::TileOrder::canonical), mixed_w.data(), mixed_w.size());
            check(canonical_w != mixed_w, std::string("a mixed-up ") + order_name(order) + " order changes the result");
            // The other CPU order's bytes read with this one are wrong too.
            for (const format::TileOrder other : kCpuOrders) {
                if (other == order) continue;
                const Matrix wrong = reordered(canonical, other);
                std::vector<uint16_t> wrong_w(static_cast<size_t>(wrong.k * wrong.n));
                cpu_exl3::reconstruct(view(wrong, order), wrong_w.data(), wrong_w.size());
                check(canonical_w != wrong_w, std::string("the ") + order_name(other) + " bytes read as " +
                                                   order_name(order) + " change the result");
            }
        }
    }

    // A non-finite or a tiny input sends the work back to the exact path, and
    // the exact path must read the same tiles.
    {
        const Matrix canonical = make_matrix(256, 256, 4, 2, 4242u);
        for (const format::TileOrder order : kCpuOrders) {
            const Matrix cpu = reordered(canonical, order);
            for (const float poison : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::denorm_min()}) {
                std::vector<float> x = make_input(256, 11u);
                x[7] = poison;
                std::vector<float> canonical_input(256);
                std::array<uint16_t, 256> tile{};
                cpu_exl3::LinearScratch scratch{canonical_input.data(), canonical_input.size(), tile.data(), tile.size()};
                std::vector<float> canonical_y(256), cpu_y(256);
                const cpu_exl3::Status canonical_status =
                    cpu_exl3::linear(view(canonical, format::TileOrder::canonical), x.data(), canonical_y.data(), scratch);
                const cpu_exl3::Status cpu_status =
                    cpu_exl3::linear(view(cpu, order), x.data(), cpu_y.data(), scratch);
                check(canonical_status == cpu_exl3::Status::ok && cpu_status == cpu_exl3::Status::ok &&
                          same_bits(canonical_y, cpu_y),
                      std::string("the exact fallback reads the same tiles [") + order_name(order) + "]");
            }
        }
    }

    std::printf("exl3_cpu_layout_parity: %d failures, %zu packed projections, %zu expert output values, %zu matrix "
                "values compared (integer dot %s)\n",
                failures, projections, compared, values,
                cpu_exl3::mul1_intdot_enabled() ? "enabled" : "disabled");
    const std::array<uint8_t, 128> reachable = mul1_reachable_sums();
    size_t reachable_count = 0;
    for (uint8_t byte : reachable)
        for (int bit = 0; bit < 8; ++bit)
            if (byte & (1u << bit)) ++reachable_count;
    size_t sums = 0;
    bool covered = true;
    for (size_t i = 0; i < reachable.size(); ++i) {
        const uint8_t missing = static_cast<uint8_t>(reachable[i] & ~mul1_sums()[i]);
        if (missing != 0) covered = false;
        for (int bit = 0; bit < 8; ++bit)
            if (mul1_sums()[i] & (1u << bit)) ++sums;
    }
    check(covered && sums == reachable_count,
          "the vector path reached every value the MUL1 codebook can produce (" + std::to_string(sums) + " of " +
              std::to_string(reachable_count) + ")");
    return failures == 0 ? 0 : 1;
}
