// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU EXL3 trellis extraction, codebooks and the batch-1 correctness path.
// The packed-tile and codebook rules are derived from exllamav3's MIT code;
// the implementation here remains portable C++ with an AVX2-safe baseline.
#include "strata/kernels/cpu/exl3.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

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

float f16_to_f32(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x3ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3ffu;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float half_add(uint16_t a, uint16_t b) {
    // The volatile temporary prevents a compiler from contracting the
    // operation into an unrelated wider expression before half rounding.
    volatile float sum = f16_to_f32(a) + f16_to_f32(b);
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
    const float inv = f16_to_f32(0x1eeeu);
    const float bias = f16_to_f32(0xc931u);
    volatile float product = f16_to_f32(h) * inv;
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
                    f16_to_f32(half_table[static_cast<size_t>(codebook)][state]);
        return result;
    }();
    return table;
}

// Hot linear path: validation and the codebook LUT pointer are already hoisted.
// This writes only one bounded 16x16 tile in F32; it never materializes W.
template <int Bits>
void decode_tile_f32_hot_bits(const MatrixView& matrix, int64_t tile_k, int64_t tile_n, const float* lut,
                              float* values) {
    constexpr size_t words32_count = static_cast<size_t>(Bits) * 8;
    const int64_t tiles_n = matrix.n / kTileSide;
    const size_t packed_size = static_cast<size_t>(16 * Bits);
    const size_t tile_index = static_cast<size_t>(tile_k) * static_cast<size_t>(tiles_n) +
                              static_cast<size_t>(tile_n);
    const uint16_t* packed = matrix.trellis + tile_index * packed_size;
    for (int state = 0; state < kTileValues; ++state) {
        const int b0 = state * Bits + Bits - 16 + 256 * Bits;
        const int b1 = b0 + 16;
        const size_t i0_raw = static_cast<size_t>(b0 >> 5);
        const size_t i1_raw = static_cast<size_t>((b1 - 1) >> 5);
        const size_t i0 = i0_raw >= words32_count ? i0_raw - words32_count : i0_raw;
        const size_t i1 = i1_raw >= words32_count ? i1_raw - words32_count : i1_raw;
        const int shift = (i1_raw + 1) * 32 - b1;
        const uint64_t merged = (static_cast<uint64_t>(load_u32(packed, i0)) << 32) |
                                load_u32(packed, i1);
        const uint16_t packed_state = static_cast<uint16_t>((merged >> shift) & 0xffffu);
        values[strata::kernels::exl3::tile_element_for_state(state)] = lut[packed_state];
    }
}

template <int Bits>
void linear_dot_tiles_f32(const MatrixView& matrix, const float* input_basis, float* y, const float* lut) {
    const size_t k = static_cast<size_t>(matrix.k);
    const size_t n = static_cast<size_t>(matrix.n);
    const size_t tiles_k = k / kTileSide;
    const size_t tiles_n = n / kTileSide;
    std::array<float, kTileValues> tile_f32{};
    for (size_t tile_n = 0; tile_n < tiles_n; ++tile_n) {
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
            y[tile_n * kTileSide + static_cast<size_t>(col)] = static_cast<float>(acc[col]);
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
            const float scale = f16_to_f32(matrix.suh[base + i]);
            volatile float product = x[base + i] * scale;
            scratch.input_basis[base + i] = product;
        }
        hadamard_normalized(scratch.input_basis + base);
    }

    const float* const f32_lut = codebook_lut_f32()[static_cast<size_t>(matrix.codebook)].data();
    switch (matrix.bits) {
        case 1: linear_dot_tiles_f32<1>(matrix, scratch.input_basis, y, f32_lut); break;
        case 2: linear_dot_tiles_f32<2>(matrix, scratch.input_basis, y, f32_lut); break;
        case 3: linear_dot_tiles_f32<3>(matrix, scratch.input_basis, y, f32_lut); break;
        case 4: linear_dot_tiles_f32<4>(matrix, scratch.input_basis, y, f32_lut); break;
        case 5: linear_dot_tiles_f32<5>(matrix, scratch.input_basis, y, f32_lut); break;
        case 6: linear_dot_tiles_f32<6>(matrix, scratch.input_basis, y, f32_lut); break;
        case 7: linear_dot_tiles_f32<7>(matrix, scratch.input_basis, y, f32_lut); break;
        default: linear_dot_tiles_f32<8>(matrix, scratch.input_basis, y, f32_lut); break;
    }

    for (size_t base = 0; base < n; base += kHadamardSide) {
        hadamard_normalized(y + base);
        for (size_t i = 0; i < kHadamardSide; ++i) {
            volatile float scaled = y[base + i] * f16_to_f32(matrix.svh[base + i]);
            y[base + i] = scaled;
            if (matrix.bias != nullptr) {
                volatile float with_bias = y[base + i] + f16_to_f32(matrix.bias[base + i]);
                y[base + i] = with_bias;
            }
        }
    }
    return Status::ok;
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
    size_t tile_index = static_cast<size_t>(tile_k) * static_cast<size_t>(tiles_n) + static_cast<size_t>(tile_n);
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
