// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 trellis and codebook rules follow exllamav3's MIT-licensed implementation.
//
// The scalar launch remains a portable correctness fallback.  Devices with a suitable subgroup use the packed dot
// path below, which keeps the trellis in its original form and preserves the descriptor and event contract.
#include "strata/kernels/exl3.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "cuda_intrinsics.hpp"
#include "exl3_word_tile.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_exl3
#include "strata/kernels/device_arch_tag.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels::exl3 {
namespace {

constexpr float kHadamardScale = 0.08838834764831845f;  // 1 / sqrt(128), as in exllamav3's reconstruct path
// Codebook 2 uses the unsigned byte sum of state * 0x83DCD12D.  dp4a is signed-byte, so flip the sign bit
// and add 512; this is exactly sum(u_i - 128) + 512 = sum(u_i), with no change to the following F16-RNE formula.
inline int cb2_byte_sum_dp4a(uint32_t x) {
    return strata::kernels::xe::dp4a(sycl::bit_cast<int32_t>(x ^ 0x80808080u), 0x01010101, 512);
}
// gitleaks:allow -- public EXL3 codebook multipliers and bit masks, never credentials.

inline void check_view(const MatrixView& view, const char* who) {
    if (view.trellis == nullptr) throw std::invalid_argument(std::string(who) + ": trellis is null");
    // The device reads the format's own tile order only.  The CPU block order
    // is a host-side arrangement, so a view that carries it must never reach
    // a device kernel.
    if (view.tile_order != TileOrder::canonical)
        throw std::invalid_argument(std::string(who) + ": the device reads the canonical tile order only");
    if (view.k <= 0 || view.n <= 0) throw std::invalid_argument(std::string(who) + ": dimensions must be positive");
    if (view.k % 16 != 0 || view.n % 16 != 0)
        throw std::invalid_argument(std::string(who) + ": k and n must be multiples of 16");
    if (view.bits < 1 || view.bits > 8)
        throw std::invalid_argument(std::string(who) + ": bits must be in 1..8");
    if (view.codebook < 0 || view.codebook > 2)
        throw std::invalid_argument(std::string(who) + ": codebook must be 0, 1 or 2");
    // Check before multiplying: a wrapped element count would pass a later
    // comparison on a 64-bit host and then reach device pointer arithmetic.
    const uint64_t max_elements = (uint64_t) std::numeric_limits<ptrdiff_t>::max() / sizeof(uint16_t);
    if ((uint64_t) view.k > max_elements / (uint64_t) view.n)
        throw std::invalid_argument(std::string(who) + ": matrix is too large");
}

inline void check_gemv(const MatrixView& view, const float* x, float* y, const float* scratch,
                       size_t scratch_elements, GemvOptions options) {
    check_view(view, "exl3_gemv_fp32");
    if (x == nullptr || y == nullptr) throw std::invalid_argument("exl3_gemv_fp32: x and y are required");
    if (options.input_hadamard && view.suh != nullptr && view.k % 128 != 0)
        throw std::invalid_argument("exl3_gemv_fp32: k must be divisible by 128 for input H128");
    if (options.output_hadamard && view.svh != nullptr && view.n % 128 != 0)
        throw std::invalid_argument("exl3_gemv_fp32: n must be divisible by 128 for output H128");
    const size_t needed = exl3_gemv_scratch_elements(view, options);
    if (needed != 0 && (scratch == nullptr || scratch_elements < needed))
        throw std::invalid_argument("exl3_gemv_fp32: caller scratch is too small");
}

inline void check_gemm(const MatrixView& view, const float* x, float* y, int64_t ldx, int64_t ldy, int64_t rows,
                       const float* scratch, size_t scratch_elements, GemvOptions options) {
    check_view(view, "exl3_gemm_fp32");
    if (x == nullptr || y == nullptr) throw std::invalid_argument("exl3_gemm_fp32: x and y are required");
    if (rows <= 0) throw std::invalid_argument("exl3_gemm_fp32: rows must be positive");
    if (ldx < view.k || ldy < view.n)
        throw std::invalid_argument("exl3_gemm_fp32: leading dimensions must cover the logical vectors");
    const uint64_t max_elements = (uint64_t) std::numeric_limits<ptrdiff_t>::max() / sizeof(float);
    if ((uint64_t) rows > max_elements / (uint64_t) ldx ||
        (uint64_t) rows > max_elements / (uint64_t) ldy)
        throw std::invalid_argument("exl3_gemm_fp32: input or output span is too large");
    if (options.input_hadamard && view.suh != nullptr && view.k % 128 != 0)
        throw std::invalid_argument("exl3_gemm_fp32: k must be divisible by 128 for input H128");
    if (options.output_hadamard && view.svh != nullptr && view.n % 128 != 0)
        throw std::invalid_argument("exl3_gemm_fp32: n must be divisible by 128 for output H128");
    const size_t needed = exl3_gemm_scratch_elements(view, rows, options);
    if (needed != 0 && (scratch == nullptr || scratch_elements < needed))
        throw std::invalid_argument("exl3_gemm_fp32: caller scratch is too small");
}

// The device code loads uint16 words explicitly rather than casting the packed pointer to uint32_t.  This keeps the
// contract valid for a 2-byte-aligned allocation and documents EXL3's little-endian packed-word order.
inline uint32_t load_word32(const uint16_t* packed, int word, int word_count) {
    // extract_state visits [word_count - 1, 2 * word_count - 1], so one ring wrap suffices.
    const int w = word >= word_count ? word - word_count : word;
    return (uint32_t) packed[2 * w] | ((uint32_t) packed[2 * w + 1] << 16);
}

inline uint16_t extract_state(const uint16_t* packed, int bits, int state) {
    const int word_count = bits * 8;  // (256 * bits) / 32
    const int b0 = state * bits + bits - 16 + 256 * bits;
    const int b1 = b0 + 16;
    const int i0 = b0 / 32;
    const int i1 = (b1 - 1) / 32;
    const int shift = (i1 + 1) * 32 - b1;
    const uint32_t hi = load_word32(packed, i0, word_count);
    const uint32_t lo = load_word32(packed, i1, word_count);
    return static_cast<uint16_t>(strata::kernels::xe::funnel_right_u32(hi, lo, static_cast<unsigned>(shift)));
}

inline uint32_t lop3_6a(uint32_t value) {
    return ((~value & 0x3B603B60u) | (value & (0x8FFF8FFFu ^ 0x3B603B60u)));
}

inline uint16_t decode_state(uint16_t state, int codebook) {
    uint32_t x = state;
    if (codebook == 0) {
        x = x * 89226354u + 64248484u;
        x = lop3_6a(x);
        return f16_from_f32(f32_from_f16((uint16_t) x) + f32_from_f16((uint16_t) (x >> 16)));
    }
    if (codebook == 1) {
        x = x * 0xCBAC1FEDu;
        x = lop3_6a(x);
        return f16_from_f32(f32_from_f16((uint16_t) x) + f32_from_f16((uint16_t) (x >> 16)));
    }
    x = x * 0x83DCD12Du;
    const int byte_sum = cb2_byte_sum_dp4a(x);
    const uint16_t h = (uint16_t) (0x6400u + byte_sum);
    // The native mul1 path performs a half FMA.  The operands are half-valued and the portable path deliberately
    // rounds once at the same boundary, matching the scalar EXL3 reference used by the parity fixture.
    return f16_from_f32(f32_from_f16(h) * f32_from_f16(0x1EEEu) + f32_from_f16(0xC931u));
}

inline float decoded_weight(const MatrixView& view, int64_t row, int64_t col) {
    const int tile_row = (int) (row >> 4);
    const int tile_col = (int) (col >> 4);
    const size_t tile_index = (size_t) tile_row * (size_t) (view.n >> 4) + (size_t) tile_col;
    const uint16_t* tile = view.trellis + tile_index * (size_t) (16 * view.bits);
    const int state = tile_state_for_element((int) ((row & 15) * 16 + (col & 15)));
    return f32_from_f16(decode_state(extract_state(tile, view.bits, state), view.codebook));
}

// The packed-dot path assigns one output tile to a subgroup. Bits and codebook are template parameters at the launch
// site, so the compiler can fold the packed-word geometry and remove codec branches. The scalar path above remains
// the capability/layout fallback and is retained as the correctness reference.
template <int Codebook>
inline uint16_t packed_decode_codebook(uint16_t state) {
    uint32_t x = state;
    if constexpr (Codebook == 0) {
        x = x * 89226354u + 64248484u;
        x = lop3_6a(x);
        return f16_from_f32(f32_from_f16((uint16_t) x) + f32_from_f16((uint16_t) (x >> 16)));
    } else if constexpr (Codebook == 1) {
        x = x * 0xCBAC1FEDu;
        x = lop3_6a(x);
        return f16_from_f32(f32_from_f16((uint16_t) x) + f32_from_f16((uint16_t) (x >> 16)));
    } else {
        x = x * 0x83DCD12Du;
        const int byte_sum = cb2_byte_sum_dp4a(x);
        const int32_t numerator = 887 * static_cast<int32_t>(byte_sum) - 452608;
        return f16_from_f32(static_cast<float>(numerator) * 0x1.0p-17f);
    }
}

template <int Bits>
inline uint16_t packed_extract_state(const uint16_t* tile, int state) {
    constexpr int word_count = Bits * 8;
    const int b0 = state * Bits + Bits - 16 + 256 * Bits;
    const int b1 = b0 + 16;
    const int i0 = b0 / 32;
    const int i1 = (b1 - 1) / 32;
    const int shift = (i1 + 1) * 32 - b1;
    const uint32_t hi = load_word32(tile, i0, word_count);
    const uint32_t lo = load_word32(tile, i1, word_count);
    return static_cast<uint16_t>(strata::kernels::xe::funnel_right_u32(hi, lo, static_cast<unsigned>(shift)));
}

// Scalar fallback specialization: it keeps one output column per work item and avoids the subgroup requirement.
// Codebook 2 has a proven normal-range closed form, so it never enters the software half conversion path.
template <int Codebook>
inline float scalar_decode_codebook_f32(uint16_t state) {
    if constexpr (Codebook == 2) {
        uint32_t x = state * 0x83DCD12Du;
        const int byte_sum = cb2_byte_sum_dp4a(x);
        const int32_t numerator = 887 * static_cast<int32_t>(byte_sum) - 452608;
        const float value = static_cast<float>(numerator) * 0x1.0p-17f;
        uint32_t u = 0;
        std::memcpy(&u, &value, sizeof(u));
        // All byte sums 0..1020 have abs(value) >= 238 * 2^-17, inside the finite F16 normal range.
        // The closed form is pre-rounded: preserve the exact F16 RNE boundary before widening to F32.
        u = (u + 0xfffu + ((u >> 13) & 1u)) & 0xffffe000u;
        float rounded = 0.0f;
        std::memcpy(&rounded, &u, sizeof(rounded));
        return rounded;
    } else {
        return f32_from_f16(packed_decode_codebook<Codebook>(state));
    }
}

template <int Bits, int Codebook>
inline float scalar_decoded_weight(const MatrixView& view, int64_t row, int64_t col) {
    const int64_t n_tiles = view.n / 16;
    const size_t tile_index = static_cast<size_t>(row / 16) * static_cast<size_t>(n_tiles) +
                              static_cast<size_t>(col / 16);
    const uint16_t* tile = view.trellis + tile_index * static_cast<size_t>(16 * Bits);
    const int state = tile_state_for_element(static_cast<int>((row & 15) * 16 + (col & 15)));
    constexpr int word_count = Bits * 8;
    const int b0 = state * Bits + Bits - 16 + 256 * Bits;
    const int b1 = b0 + 16;
    const int i0 = b0 / 32;
    const int i1 = (b1 - 1) / 32;
    const int shift = (i1 + 1) * 32 - b1;
    const uint32_t hi = load_word32(tile, i0, word_count);
    const uint32_t lo = load_word32(tile, i1, word_count);
    const uint16_t packed_state =
        static_cast<uint16_t>(strata::kernels::xe::funnel_right_u32(hi, lo, static_cast<unsigned>(shift)));
    return scalar_decode_codebook_f32<Codebook>(packed_state);
}

template <int Bits, int Codebook>
sycl::event submit_scalar_dot_template(sycl::queue& queue, const MatrixView& view, const float* x, int64_t ldx,
                                       float* y, int64_t ldy, int64_t rows,
                                       const std::vector<sycl::event>& deps, bool add_bias) {
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::range<1>(static_cast<size_t>(rows) * static_cast<size_t>(view.n)), [=](sycl::id<1> id) {
            const size_t flat = static_cast<size_t>(id[0]);
            const int64_t row_out = static_cast<int64_t>(flat / static_cast<size_t>(view.n));
            const int64_t col = static_cast<int64_t>(flat % static_cast<size_t>(view.n));
            const float* input = x + row_out * ldx;
            float* output = y + row_out * ldy;
            float sum = 0.0f;
            for (int64_t row = 0; row < view.k; ++row)
                sum += input[row] * scalar_decoded_weight<Bits, Codebook>(view, row, col);
            if (add_bias && view.bias != nullptr) sum += f32_from_f16(view.bias[col]);
            output[col] = sum;
        });
    });
}

bool submit_scalar_dot_dispatch(sycl::queue& queue, const MatrixView& view, const float* x, int64_t ldx, float* y,
                                int64_t ldy, int64_t rows, const std::vector<sycl::event>& deps, bool add_bias,
                                sycl::event& result) {
    #define EXL3_SCALAR_DOT(B, C) \
        do { \
            result = submit_scalar_dot_template<B, C>(queue, view, x, ldx, y, ldy, rows, deps, add_bias); \
            return true; \
        } while (false)
    switch (view.bits) {
        case 1:
            if (view.codebook == 0) EXL3_SCALAR_DOT(1, 0);
            if (view.codebook == 1) EXL3_SCALAR_DOT(1, 1);
            EXL3_SCALAR_DOT(1, 2);
        case 2:
            if (view.codebook == 0) EXL3_SCALAR_DOT(2, 0);
            if (view.codebook == 1) EXL3_SCALAR_DOT(2, 1);
            EXL3_SCALAR_DOT(2, 2);
        case 3:
            if (view.codebook == 0) EXL3_SCALAR_DOT(3, 0);
            if (view.codebook == 1) EXL3_SCALAR_DOT(3, 1);
            EXL3_SCALAR_DOT(3, 2);
        case 4:
            if (view.codebook == 0) EXL3_SCALAR_DOT(4, 0);
            if (view.codebook == 1) EXL3_SCALAR_DOT(4, 1);
            EXL3_SCALAR_DOT(4, 2);
        case 5:
            if (view.codebook == 0) EXL3_SCALAR_DOT(5, 0);
            if (view.codebook == 1) EXL3_SCALAR_DOT(5, 1);
            EXL3_SCALAR_DOT(5, 2);
        case 6:
            if (view.codebook == 0) EXL3_SCALAR_DOT(6, 0);
            if (view.codebook == 1) EXL3_SCALAR_DOT(6, 1);
            EXL3_SCALAR_DOT(6, 2);
        case 7:
            if (view.codebook == 0) EXL3_SCALAR_DOT(7, 0);
            if (view.codebook == 1) EXL3_SCALAR_DOT(7, 1);
            EXL3_SCALAR_DOT(7, 2);
        default:
            if (view.codebook == 0) EXL3_SCALAR_DOT(8, 0);
            if (view.codebook == 1) EXL3_SCALAR_DOT(8, 1);
            EXL3_SCALAR_DOT(8, 2);
    }
    #undef EXL3_SCALAR_DOT
}

template <int Bits, int Codebook>
sycl::event submit_packed_dot(sycl::queue& queue, const MatrixView& view, const float* x, int64_t ldx, float* y,
                              int64_t ldy, int64_t rows, const std::vector<sycl::event>& deps, bool add_bias,
                              int split_k, int subgroup_size, int waves_per_group) {
    const int64_t n_tiles = view.n / 16;
    const int64_t k_tiles = view.k / 16;
    const size_t groups = static_cast<size_t>(rows) * static_cast<size_t>(n_tiles);
    const size_t local_size = static_cast<size_t>(waves_per_group * subgroup_size);
    const size_t local_reduce_words = static_cast<size_t>(waves_per_group) * 16;
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        sycl::local_accessor<float, 1> partial(sycl::range<1>(local_reduce_words), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * local_size), sycl::range<1>(local_size)),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            const sycl::sub_group subgroup = item.get_sub_group();
            const int lane = static_cast<int>(subgroup.get_local_id()[0]);
            const int wave = static_cast<int>(subgroup.get_group_id()[0]);
            const size_t group = static_cast<size_t>(item.get_group_linear_id());
            const int64_t row = static_cast<int64_t>(group / static_cast<size_t>(n_tiles));
            const int64_t tile_n = static_cast<int64_t>(group % static_cast<size_t>(n_tiles));
            const int64_t kb_begin = (k_tiles * wave) / split_k;
            const int64_t kb_end = (k_tiles * (wave + 1)) / split_k;
            float accum = 0.0f;
            const float* input = x + row * ldx;
            const int col_in_tile = lane & 15;
            const int row_parity = lane >> 4;
            for (int64_t k_tile = kb_begin; k_tile < kb_end; ++k_tile) {
                const uint16_t* source = view.trellis +
                    (static_cast<size_t>(k_tile) * static_cast<size_t>(n_tiles) + static_cast<size_t>(tile_n)) *
                    static_cast<size_t>(16 * Bits);
                float sum = 0.0f;
                for (int rr = 0; rr < 8; ++rr) {
                    const int r = 2 * rr + row_parity;
                    const int state = tile_state_for_element(r * 16 + col_in_tile);
                    const uint16_t packed_state = packed_extract_state<Bits>(source, state);
                    sum += input[k_tile * 16 + r] * scalar_decode_codebook_f32<Codebook>(packed_state);
                }
                accum += sum + sycl::select_from_group(subgroup, sum, lane ^ 16);
            }
            if (lane < 16) partial[static_cast<size_t>(wave) * 16 + static_cast<size_t>(lane)] = accum;
            item.barrier();
            if (wave == 0 && lane < 16) {
                float sum = 0.0f;
                for (int w = 0; w < split_k; ++w)
                    sum += partial[static_cast<size_t>(w) * 16 + static_cast<size_t>(lane)];
                const int64_t col = tile_n * 16 + lane;
                if (add_bias && view.bias != nullptr) sum += f32_from_f16(view.bias[col]);
                y[row * ldy + col] = sum;
            }
        });
    });
}


bool word_tile_preload_enabled() {
    const char* text = std::getenv("STRATA_EXL3_WORD_TILE_PRELOAD");
    return text != nullptr && text[0] == '1' && text[1] == '\0';
}

// RR is compile-time so bank masks remain constants, not an additional runtime mask table lookup.
// Recursion retains exactly the baseline rr0..rr7 FP32 addition order under strict FP.
template <int Bits, int RR = 0>
inline void accumulate_word_tile(const word_cache::DistributedTile<Bits>& words, const sycl::sub_group& subgroup,
                                  int lane, const float* input, int64_t k_tile, float& sum) {
    static_assert(RR >= 0 && RR < 8);
    const int r = 2 * RR + (lane >> 4);
    const uint16_t packed_state = words.template state<RR>(subgroup, lane);
    sum += input[k_tile * 16 + r] * scalar_decode_codebook_f32<2>(packed_state);
    if constexpr (RR + 1 < 8)
        accumulate_word_tile<Bits, RR + 1>(words, subgroup, lane, input, k_tile, sum);
}

template <int Bits>
sycl::event submit_packed_dot_word_cache(sycl::queue& queue, const MatrixView& view, const float* x, int64_t ldx,
                                         float* y, int64_t ldy, int64_t rows,
                                         const std::vector<sycl::event>& deps, bool add_bias,
                                         int split_k, int subgroup_size, int waves_per_group) {
    const int64_t n_tiles = view.n / 16, k_tiles = view.k / 16;
    const size_t groups = static_cast<size_t>(rows) * static_cast<size_t>(n_tiles);
    const size_t local_size = static_cast<size_t>(waves_per_group * subgroup_size);
    const size_t local_reduce_words = static_cast<size_t>(split_k) * 16;
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        sycl::local_accessor<float, 1> partial(sycl::range<1>(local_reduce_words), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * local_size), sycl::range<1>(local_size)),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            const auto subgroup = item.get_sub_group();
            // reqd_sub_group_size(32) makes this mask exact and exposes the index domain to the compiler.
            const int lane = static_cast<int>(subgroup.get_local_id()[0]) & 31;
            const int wave = static_cast<int>(subgroup.get_group_id()[0]);
            const size_t group = static_cast<size_t>(item.get_group_linear_id());
            const int64_t row = static_cast<int64_t>(group / static_cast<size_t>(n_tiles));
            const int64_t tile_n = static_cast<int64_t>(group % static_cast<size_t>(n_tiles));
            const float* input = x + row * ldx;
            for (int logical = wave; logical < split_k; logical += waves_per_group) {
                const int64_t kb_begin = (k_tiles * logical) / split_k, kb_end = (k_tiles * (logical + 1)) / split_k;
                float accum = 0.0f;
                for (int64_t k_tile = kb_begin; k_tile < kb_end; ++k_tile) {
                    const uint16_t* source = view.trellis +
                        (static_cast<size_t>(k_tile) * static_cast<size_t>(n_tiles) + static_cast<size_t>(tile_n)) *
                        static_cast<size_t>(16 * Bits);
                    // Assemble logical words with the existing uint16 loads: no uint32 pointer cast or halo.
                    word_cache::DistributedTile<Bits> words{0, 0};
                    if (lane < Bits * 8) words.bank0 = load_word32(source, lane, Bits * 8);
                    if constexpr (Bits > 4)
                        if (lane + 32 < Bits * 8) words.bank1 = load_word32(source, lane + 32, Bits * 8);
                    float sum = 0.0f;
                    accumulate_word_tile<Bits>(words, subgroup, lane, input, k_tile, sum);
                    accum += sum + sycl::select_from_group(subgroup, sum, lane ^ 16);
                }
                if (lane < 16) partial[static_cast<size_t>(logical) * 16 + static_cast<size_t>(lane)] = accum;
            }
            item.barrier();
            if (wave == 0 && lane < 16) {
                float sum = 0.0f;
                for (int w = 0; w < split_k; ++w)
                    sum += partial[static_cast<size_t>(w) * 16 + static_cast<size_t>(lane)];
                const int64_t col = tile_n * 16 + lane;
                if (add_bias && view.bias != nullptr) sum += f32_from_f16(view.bias[col]);
                y[row * ldy + col] = sum;
            }
        });
    });
}

// Wide outputs have enough independent tiles to use a single subgroup per tile. Exchange each
// adjacent product pair so the lower sixteen lanes retain scalar K=0..k-1 FP32 addition order.
template <int Bits, int RR = 0>
inline void accumulate_wide_tile(const word_cache::DistributedTile<Bits>& words,
                                 const sycl::sub_group& subgroup, int lane, const float* input,
                                 int64_t k_tile, float& sum) {
    const int r = 2 * RR + (lane >> 4);
    const float product = input[k_tile * 16 + r] * scalar_decode_codebook_f32<2>(words.template state<RR>(subgroup, lane));
    const float other = sycl::select_from_group(subgroup, product, lane ^ 16);
    if (lane < 16) { sum += product; sum += other; }
    if constexpr (RR < 7) accumulate_wide_tile<Bits, RR + 1>(words, subgroup, lane, input, k_tile, sum);
}

template <int Bits>
sycl::event submit_wide_word_dot(sycl::queue& queue, const MatrixView& view, const float* x, int64_t ldx,
                                 float* y, int64_t ldy, int64_t rows,
                                 const std::vector<sycl::event>& deps, bool add_bias) {
    const int64_t n_tiles = view.n / 16, k_tiles = view.k / 16;
    const size_t groups = static_cast<size_t>(rows) * static_cast<size_t>(n_tiles);
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(groups * 32, 32), [=](sycl::nd_item<1> item)
                       [[sycl::reqd_sub_group_size(32)]] {
            const auto subgroup = item.get_sub_group();
            const int lane = static_cast<int>(subgroup.get_local_id()[0]) & 31;
            const size_t group = item.get_group_linear_id();
            const int64_t row = static_cast<int64_t>(group / static_cast<size_t>(n_tiles));
            const int64_t tile_n = static_cast<int64_t>(group % static_cast<size_t>(n_tiles));
            const float* input = x + row * ldx;
            float sum = 0.0f;
            for (int64_t kt = 0; kt < k_tiles; ++kt) {
                const uint16_t* source = view.trellis +
                    (static_cast<size_t>(kt) * static_cast<size_t>(n_tiles) + static_cast<size_t>(tile_n)) *
                    static_cast<size_t>(16 * Bits);
                word_cache::DistributedTile<Bits> words{0, 0};
                if (lane < Bits * 8) words.bank0 = load_word32(source, lane, Bits * 8);
                if constexpr (Bits > 4)
                    if (lane + 32 < Bits * 8) words.bank1 = load_word32(source, lane + 32, Bits * 8);
                accumulate_wide_tile<Bits>(words, subgroup, lane, input, kt, sum);
            }
            if (lane < 16) {
                const int64_t col = tile_n * 16 + lane;
                if (add_bias && view.bias != nullptr) sum += f32_from_f16(view.bias[col]);
                y[row * ldy + col] = sum;
            }
        });
    });
}

bool wide_word_enabled() {
    const char* text = std::getenv("STRATA_EXL3_WIDE_WORD_TILE");
    return text == nullptr || !(text[0] == '0' && text[1] == '\0');
}

bool packed_dot_capability(const sycl::queue& queue, const MatrixView& view, int64_t rows,
                           int& subgroup_size, int& split_k, int& waves_per_group) {
    if (view.k % 16 != 0 || view.n % 16 != 0 || rows <= 0) return false;
    const auto subgroup_sizes = queue.get_device().get_info<sycl::info::device::sub_group_sizes>();
    if (std::find(subgroup_sizes.begin(), subgroup_sizes.end(), size_t(32)) == subgroup_sizes.end()) return false;
    const size_t max_work_group = queue.get_device().get_info<sycl::info::device::max_work_group_size>();
    const size_t local_mem = queue.get_device().get_info<sycl::info::device::local_mem_size>();
    constexpr size_t bytes_per_wave = 16 * sizeof(float);
    const int memory_waves = static_cast<int>(local_mem / bytes_per_wave);
    const int workgroup_waves = static_cast<int>(max_work_group / 32);
    const int max_waves = std::max(1, std::min(memory_waves, workgroup_waves));
    if (memory_waves < 1 || workgroup_waves < 1) return false;
    const int64_t n_tiles64 = view.n / 16;
    const int64_t k_tiles64 = view.k / 16;
    if (n_tiles64 > std::numeric_limits<int>::max() || k_tiles64 > std::numeric_limits<int>::max()) return false;
    const int n_tiles = static_cast<int>(n_tiles64);
    const int k_tiles = static_cast<int>(k_tiles64);
    // When there are already many more independent output tiles than the device can keep occupied across the
    // available K waves, the scalar one-work-item-per-output fallback has better parallel granularity.  Keep the
    // packed path for shapes where a wave can amortize its packed decode over a useful K span; this is derived from
    // the descriptor geometry and work-group capability rather than a model or vendor threshold.
    const int64_t required_k_tiles =
        static_cast<int64_t>(n_tiles) / max_waves + (static_cast<int64_t>(n_tiles) % max_waves != 0 ? 1 : 0);
    if (static_cast<int64_t>(k_tiles) < required_k_tiles) return false;
    // Use the available K waves so a narrow output matrix can hide decode/load latency.
    // Both the wave capacity and the useful K span come from device/descriptor limits.
    split_k = std::min(k_tiles, max_waves);
    subgroup_size = 32;
    waves_per_group = split_k;
    return split_k >= 1;
}

bool submit_packed_dot_dispatch(sycl::queue& queue, const MatrixView& view, const float* x, int64_t ldx, float* y,
                                int64_t ldy, int64_t rows, const std::vector<sycl::event>& deps, bool add_bias,
                                sycl::event& result) {
    int subgroup_size = 0, split_k = 0, waves_per_group = 0;
    if (!packed_dot_capability(queue, view, rows, subgroup_size, split_k, waves_per_group)) {
        if (view.codebook != 2 || !wide_word_enabled() || view.k % 16 != 0 || view.n % 16 != 0 || rows <= 0)
            return false;
        const auto sizes = queue.get_device().get_info<sycl::info::device::sub_group_sizes>();
        if (std::find(sizes.begin(), sizes.end(), size_t(32)) == sizes.end() ||
            queue.get_device().get_info<sycl::info::device::max_work_group_size>() < 32) return false;
        switch (view.bits) {
#define EXL3_WIDE_WORD_CASE(B) \
            case B: result = submit_wide_word_dot<B>(queue, view, x, ldx, y, ldy, rows, deps, add_bias); return true
            EXL3_WIDE_WORD_CASE(1);
            EXL3_WIDE_WORD_CASE(2);
            EXL3_WIDE_WORD_CASE(3);
            EXL3_WIDE_WORD_CASE(4);
            EXL3_WIDE_WORD_CASE(5);
            EXL3_WIDE_WORD_CASE(6);
            EXL3_WIDE_WORD_CASE(7);
            EXL3_WIDE_WORD_CASE(8);
#undef EXL3_WIDE_WORD_CASE
            default: return false;
        }
    }
    // Preserve all logical K partitions and their reduction order, but use fewer physical waves
    // when independent output tiles already cover the device's reported work-group wave capacity.
    const char* schedule = std::getenv("STRATA_EXL3_DENSE_SCHEDULE");
    const bool reschedule = schedule == nullptr || !(schedule[0] == '0' && schedule[1] == '\0');
    int physical_waves = waves_per_group;
    if (reschedule && view.codebook == 2 && word_tile_preload_enabled()) {
        const uint64_t capacity = static_cast<uint64_t>(queue.get_device().get_info<sycl::info::device::max_compute_units>()) *
                                  static_cast<uint64_t>(waves_per_group);
        const uint64_t groups = static_cast<uint64_t>(rows) * static_cast<uint64_t>(view.n / 16);
        const uint64_t waves = capacity / groups + (capacity % groups != 0 ? 1 : 0);
        physical_waves = static_cast<int>(std::max<uint64_t>(1, std::min<uint64_t>(waves_per_group, waves)));
    }
    #define EXL3_PACKED_DOT(B, C) \
        do { \
            if constexpr (C == 2) { \
                if (word_tile_preload_enabled()) \
                    result = submit_packed_dot_word_cache<B>(queue, view, x, ldx, y, ldy, rows, deps, add_bias, split_k, subgroup_size, physical_waves); \
                else \
                    result = submit_packed_dot<B, C>(queue, view, x, ldx, y, ldy, rows, deps, add_bias, split_k, subgroup_size, waves_per_group); \
            } else \
                result = submit_packed_dot<B, C>(queue, view, x, ldx, y, ldy, rows, deps, add_bias, split_k, subgroup_size, waves_per_group); \
            return true; \
        } while (false)
    switch (view.bits) {
        case 1:
            if (view.codebook == 0) EXL3_PACKED_DOT(1, 0);
            if (view.codebook == 1) EXL3_PACKED_DOT(1, 1);
            EXL3_PACKED_DOT(1, 2);
        case 2:
            if (view.codebook == 0) EXL3_PACKED_DOT(2, 0);
            if (view.codebook == 1) EXL3_PACKED_DOT(2, 1);
            EXL3_PACKED_DOT(2, 2);
        case 3:
            if (view.codebook == 0) EXL3_PACKED_DOT(3, 0);
            if (view.codebook == 1) EXL3_PACKED_DOT(3, 1);
            EXL3_PACKED_DOT(3, 2);
        case 4:
            if (view.codebook == 0) EXL3_PACKED_DOT(4, 0);
            if (view.codebook == 1) EXL3_PACKED_DOT(4, 1);
            EXL3_PACKED_DOT(4, 2);
        case 5:
            if (view.codebook == 0) EXL3_PACKED_DOT(5, 0);
            if (view.codebook == 1) EXL3_PACKED_DOT(5, 1);
            EXL3_PACKED_DOT(5, 2);
        case 6:
            if (view.codebook == 0) EXL3_PACKED_DOT(6, 0);
            if (view.codebook == 1) EXL3_PACKED_DOT(6, 1);
            EXL3_PACKED_DOT(6, 2);
        case 7:
            if (view.codebook == 0) EXL3_PACKED_DOT(7, 0);
            if (view.codebook == 1) EXL3_PACKED_DOT(7, 1);
            EXL3_PACKED_DOT(7, 2);
        default:
            if (view.codebook == 0) EXL3_PACKED_DOT(8, 0);
            if (view.codebook == 1) EXL3_PACKED_DOT(8, 1);
            EXL3_PACKED_DOT(8, 2);
    }
    #undef EXL3_PACKED_DOT
}

inline int hadamard_sign(int row, int col) {
    unsigned bits = (unsigned) (row & col);
    int parity = 0;
    while (bits != 0) {
        parity ^= 1;
        bits &= bits - 1;
    }
    return parity == 0 ? 1 : -1;
}

sycl::event submit_input_hadamard_scalar(sycl::queue& queue, const MatrixView& view, const float* x,
                                         float* transformed, const std::vector<sycl::event>& deps) {
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::range<1>((size_t) view.k), [=](sycl::id<1> id) {
            const int64_t i = (int64_t) id[0];
            const int64_t block = i & ~int64_t(127);
            const int local = (int) (i & 127);
            float sum = 0.0f;
            for (int j = 0; j < 128; ++j) {
                const float scale = f32_from_f16(view.suh[block + j]);
                sum += (float) hadamard_sign(local, j) * x[block + j] * scale;
            }
            transformed[i] = sum * kHadamardScale;
        });
    });
}

sycl::event submit_dot(sycl::queue& queue, const MatrixView& view, const float* x, float* y,
                       const std::vector<sycl::event>& deps, bool add_bias) {
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::range<1>((size_t) view.n), [=](sycl::id<1> id) {
            const int64_t col = (int64_t) id[0];
            float sum = 0.0f;
            for (int64_t row = 0; row < view.k; ++row) sum += x[row] * decoded_weight(view, row, col);
            if (add_bias && view.bias != nullptr) sum += f32_from_f16(view.bias[col]);
            y[col] = sum;
        });
    });
}

sycl::event submit_output_hadamard_scalar(sycl::queue& queue, const MatrixView& view, const float* dot, float* y,
                                          const std::vector<sycl::event>& deps, bool add_bias) {
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::range<1>((size_t) view.n), [=](sycl::id<1> id) {
            const int64_t j = (int64_t) id[0];
            const int64_t block = j & ~int64_t(127);
            const int local = (int) (j & 127);
            float sum = 0.0f;
            for (int i = 0; i < 128; ++i) sum += (float) hadamard_sign(local, i) * dot[block + i];
            sum *= kHadamardScale * f32_from_f16(view.svh[j]);
            if (add_bias && view.bias != nullptr) sum += f32_from_f16(view.bias[j]);
            y[j] = sum;
        });
    });
}

sycl::event submit_input_hadamard_rows_scalar(sycl::queue& queue, const MatrixView& view, const float* x, int64_t ldx,
                                              float* transformed, int64_t rows,
                                              const std::vector<sycl::event>& deps) {
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::range<1>((size_t) rows * (size_t) view.k), [=](sycl::id<1> id) {
            const size_t flat = (size_t) id[0];
            const int64_t r = (int64_t) (flat / (size_t) view.k);
            const int64_t i = (int64_t) (flat % (size_t) view.k);
            const int64_t block = i & ~int64_t(127);
            const int local = (int) (i & 127);
            const float* xr = x + r * ldx;
            float* dst = transformed + r * view.k;
            float sum = 0.0f;
            for (int j = 0; j < 128; ++j) {
                const float scale = f32_from_f16(view.suh[block + j]);
                sum += (float) hadamard_sign(local, j) * xr[block + j] * scale;
            }
            dst[i] = sum * kHadamardScale;
        });
    });
}

sycl::event submit_dot_rows(sycl::queue& queue, const MatrixView& view, const float* x, int64_t ldx, float* y,
                            int64_t ldy, int64_t rows, const std::vector<sycl::event>& deps, bool add_bias) {
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::range<1>((size_t) rows * (size_t) view.n), [=](sycl::id<1> id) {
            const size_t flat = (size_t) id[0];
            const int64_t r = (int64_t) (flat / (size_t) view.n);
            const int64_t col = (int64_t) (flat % (size_t) view.n);
            const float* xr = x + r * ldx;
            float* yr = y + r * ldy;
            float sum = 0.0f;
            for (int64_t row = 0; row < view.k; ++row) sum += xr[row] * decoded_weight(view, row, col);
            if (add_bias && view.bias != nullptr) sum += f32_from_f16(view.bias[col]);
            yr[col] = sum;
        });
    });
}

sycl::event submit_output_hadamard_rows_scalar(sycl::queue& queue, const MatrixView& view, const float* dot,
                                               float* y, int64_t ldy, int64_t rows,
                                               const std::vector<sycl::event>& deps, bool add_bias) {
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::range<1>((size_t) rows * (size_t) view.n), [=](sycl::id<1> id) {
            const size_t flat = (size_t) id[0];
            const int64_t r = (int64_t) (flat / (size_t) view.n);
            const int64_t j = (int64_t) (flat % (size_t) view.n);
            const int64_t block = j & ~int64_t(127);
            const int local = (int) (j & 127);
            const float* dr = dot + r * view.n;
            float* yr = y + r * ldy;
            float sum = 0.0f;
            for (int i = 0; i < 128; ++i) sum += (float) hadamard_sign(local, i) * dr[block + i];
            sum *= kHadamardScale * f32_from_f16(view.svh[j]);
            if (add_bias && view.bias != nullptr) sum += f32_from_f16(view.bias[j]);
            yr[j] = sum;
        });
    });
}

inline bool hadamard_coop_capability(const sycl::queue& queue) {
    try {
        const auto subgroup_sizes = queue.get_device().get_info<sycl::info::device::sub_group_sizes>();
        if (std::find(subgroup_sizes.begin(), subgroup_sizes.end(), size_t(32)) == subgroup_sizes.end()) return false;
        return queue.get_device().get_info<sycl::info::device::max_work_group_size>() >= 32;
    } catch (...) {
        return false;
    }
}

// A 128-wide H transform uses one SG32 and four registers per lane.  The first
// five butterfly stages stay in the SG; the two remaining bits are butterflies
// between the four registers.  This is shared by GEMV and GEMM and by input
// and output transforms so their strides and scale/bias order cannot diverge.
template <bool Output, bool Rows>
sycl::event submit_hadamard_coop(sycl::queue& queue, const MatrixView& view, const float* source,
                                 int64_t source_ld, float* destination, int64_t destination_ld, int64_t rows,
                                 const std::vector<sycl::event>& deps, bool add_bias) {
    const int64_t width = Output ? view.n : view.k;
    const size_t blocks = static_cast<size_t>(width / 128);
    const size_t groups = Rows ? static_cast<size_t>(rows) * blocks : blocks;
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * 32), sycl::range<1>(32)),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            const sycl::sub_group subgroup = item.get_sub_group();
            const int lane = static_cast<int>(subgroup.get_local_id()[0]);
            const size_t group = static_cast<size_t>(item.get_group_linear_id());
            const int64_t row = Rows ? static_cast<int64_t>(group / blocks) : 0;
            const int64_t block = static_cast<int64_t>(group % blocks) * 128;
            const float* src = Rows ? source + row * source_ld + block : source + block;
            float* dst = Rows ? destination + row * destination_ld + block : destination + block;

            float a0 = src[lane + 0 * 32];
            float a1 = src[lane + 1 * 32];
            float a2 = src[lane + 2 * 32];
            float a3 = src[lane + 3 * 32];
            if constexpr (!Output) {
                a0 *= f32_from_f16(view.suh[block + lane + 0 * 32]);
                a1 *= f32_from_f16(view.suh[block + lane + 1 * 32]);
                a2 *= f32_from_f16(view.suh[block + lane + 2 * 32]);
                a3 *= f32_from_f16(view.suh[block + lane + 3 * 32]);
            }
            for (int step = 1; step < 32; step <<= 1) {
                const float p0 = sycl::permute_group_by_xor(subgroup, a0, step);
                const float p1 = sycl::permute_group_by_xor(subgroup, a1, step);
                const float p2 = sycl::permute_group_by_xor(subgroup, a2, step);
                const float p3 = sycl::permute_group_by_xor(subgroup, a3, step);
                a0 = (lane & step) != 0 ? p0 - a0 : a0 + p0;
                a1 = (lane & step) != 0 ? p1 - a1 : a1 + p1;
                a2 = (lane & step) != 0 ? p2 - a2 : a2 + p2;
                a3 = (lane & step) != 0 ? p3 - a3 : a3 + p3;
            }
            const float b0 = a0 + a1;
            const float b1 = a0 - a1;
            const float b2 = a2 + a3;
            const float b3 = a2 - a3;
            a0 = b0 + b2;
            a1 = b1 + b3;
            a2 = b0 - b2;
            a3 = b1 - b3;
            if constexpr (Output) {
                const float scale = kHadamardScale;
                const int64_t i0 = block + lane + 0 * 32;
                const int64_t i1 = block + lane + 1 * 32;
                const int64_t i2 = block + lane + 2 * 32;
                const int64_t i3 = block + lane + 3 * 32;
                a0 *= scale * f32_from_f16(view.svh[i0]);
                a1 *= scale * f32_from_f16(view.svh[i1]);
                a2 *= scale * f32_from_f16(view.svh[i2]);
                a3 *= scale * f32_from_f16(view.svh[i3]);
                if (add_bias && view.bias != nullptr) {
                    a0 += f32_from_f16(view.bias[i0]);
                    a1 += f32_from_f16(view.bias[i1]);
                    a2 += f32_from_f16(view.bias[i2]);
                    a3 += f32_from_f16(view.bias[i3]);
                }
            } else {
                a0 *= kHadamardScale;
                a1 *= kHadamardScale;
                a2 *= kHadamardScale;
                a3 *= kHadamardScale;
            }
            dst[lane + 0 * 32] = a0;
            dst[lane + 1 * 32] = a1;
            dst[lane + 2 * 32] = a2;
            dst[lane + 3 * 32] = a3;
        });
    });
}

sycl::event submit_input_hadamard(sycl::queue& queue, const MatrixView& view, const float* x, float* transformed,
                                  const std::vector<sycl::event>& deps) {
    if (hadamard_coop_capability(queue))
        return submit_hadamard_coop<false, false>(queue, view, x, 0, transformed, 0, 1, deps, false);
    return submit_input_hadamard_scalar(queue, view, x, transformed, deps);
}

sycl::event submit_output_hadamard(sycl::queue& queue, const MatrixView& view, const float* dot, float* y,
                                   const std::vector<sycl::event>& deps, bool add_bias) {
    if (hadamard_coop_capability(queue))
        return submit_hadamard_coop<true, false>(queue, view, dot, 0, y, 0, 1, deps, add_bias);
    return submit_output_hadamard_scalar(queue, view, dot, y, deps, add_bias);
}

sycl::event submit_input_hadamard_rows(sycl::queue& queue, const MatrixView& view, const float* x, int64_t ldx,
                                       float* transformed, int64_t rows, const std::vector<sycl::event>& deps) {
    if (hadamard_coop_capability(queue))
        return submit_hadamard_coop<false, true>(queue, view, x, ldx, transformed, view.k, rows, deps, false);
    return submit_input_hadamard_rows_scalar(queue, view, x, ldx, transformed, rows, deps);
}

sycl::event submit_output_hadamard_rows(sycl::queue& queue, const MatrixView& view, const float* dot, float* y,
                                        int64_t ldy, int64_t rows, const std::vector<sycl::event>& deps,
                                        bool add_bias) {
    if (hadamard_coop_capability(queue))
        return submit_hadamard_coop<true, true>(queue, view, dot, view.n, y, ldy, rows, deps, add_bias);
    return submit_output_hadamard_rows_scalar(queue, view, dot, y, ldy, rows, deps, add_bias);
}

}  // namespace

size_t exl3_gemv_scratch_elements(const MatrixView& view, GemvOptions options) {
    check_view(view, "exl3_gemv_scratch_elements");
    size_t result = 0;
    if (options.input_hadamard && view.suh != nullptr) result += (size_t) view.k;
    if (options.output_hadamard && view.svh != nullptr) {
        const size_t max_elements = (size_t) std::numeric_limits<ptrdiff_t>::max() / sizeof(float);
        if ((uint64_t) view.n > max_elements || result > max_elements - (size_t) view.n)
            throw std::invalid_argument("exl3_gemv_fp32: scratch size overflows host address range");
        result += (size_t) view.n;
    }
    if (result > (size_t) std::numeric_limits<ptrdiff_t>::max() / sizeof(float))
        throw std::invalid_argument("exl3_gemv_fp32: scratch size overflows host address range");
    return result;
}

size_t exl3_gemm_scratch_elements(const MatrixView& view, int64_t rows, GemvOptions options) {
    if (rows <= 0) throw std::invalid_argument("exl3_gemm_fp32: rows must be positive");
    const size_t per_row = exl3_gemv_scratch_elements(view, options);
    const size_t max_elements = (size_t) std::numeric_limits<ptrdiff_t>::max() / sizeof(float);
    if (per_row > max_elements / (size_t) rows)
        throw std::invalid_argument("exl3_gemm_fp32: scratch size overflows size_t");
    return per_row * (size_t) rows;
}

sycl::event exl3_reconstruct(sycl::queue& queue, const MatrixView& view, uint16_t* out,
                             const std::vector<sycl::event>& deps) {
    check_view(view, "exl3_reconstruct");
    if (out == nullptr) throw std::invalid_argument("exl3_reconstruct: out is null");
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::range<1>((size_t) view.k * (size_t) view.n), [=](sycl::id<1> id) {
            const size_t flat = (size_t) id[0];
            const int64_t row = (int64_t) (flat / (size_t) view.n);
            const int64_t col = (int64_t) (flat % (size_t) view.n);
            const int tile_row = (int) (row >> 4);
            const int tile_col = (int) (col >> 4);
            const size_t tile_index = (size_t) tile_row * (size_t) (view.n >> 4) + (size_t) tile_col;
            const uint16_t* tile = view.trellis + tile_index * (size_t) (16 * view.bits);
            const int state = tile_state_for_element((int) ((row & 15) * 16 + (col & 15)));
            out[flat] = decode_state(extract_state(tile, view.bits, state), view.codebook);
        });
    });
}

sycl::event exl3_gemv_fp32(sycl::queue& queue, const MatrixView& view, const float* x, float* y, float* scratch,
                           size_t scratch_elements, GemvOptions options, const std::vector<sycl::event>& deps,
                           GemvProfile* profile) {
    check_gemv(view, x, y, scratch, scratch_elements, options);
    const bool use_input = options.input_hadamard && view.suh != nullptr;
    const bool use_output = options.output_hadamard && view.svh != nullptr;
    const size_t input_offset = 0;
    const size_t output_offset = use_input ? (size_t) view.k : 0;
    const float* dot_input = use_input ? scratch + input_offset : x;
    float* dot_output = use_output ? scratch + output_offset : y;

    std::vector<sycl::event> dot_deps = deps;
    if (use_input) {
        const sycl::event input_event = submit_input_hadamard(queue, view, x, scratch + input_offset, deps);
        if (profile != nullptr) profile->input_hadamard = input_event;
        dot_deps = {input_event};
    }
    const bool dot_add_bias = options.add_bias && !use_output;
    sycl::event dot_event;
    if (!submit_packed_dot_dispatch(queue, view, dot_input, view.k, dot_output, view.n, 1, dot_deps, dot_add_bias,
                                    dot_event))
        if (!submit_scalar_dot_dispatch(queue, view, dot_input, view.k, dot_output, view.n, 1, dot_deps,
                                        dot_add_bias, dot_event))
            dot_event = submit_dot(queue, view, dot_input, dot_output, dot_deps, dot_add_bias);
    if (profile != nullptr) profile->dot = dot_event;
    if (use_output) {
        const sycl::event output_event = submit_output_hadamard(queue, view, dot_output, y, {dot_event}, options.add_bias);
        if (profile != nullptr) profile->output_hadamard = output_event;
        return output_event;
    }
    return dot_event;
}

sycl::event exl3_gemm_fp32(sycl::queue& queue, const MatrixView& view, const float* x, int64_t ldx, float* y,
                           int64_t ldy, int64_t rows, float* scratch, size_t scratch_elements, GemvOptions options,
                           const std::vector<sycl::event>& deps, GemvProfile* profile) {
    check_gemm(view, x, y, ldx, ldy, rows, scratch, scratch_elements, options);
    const bool use_input = options.input_hadamard && view.suh != nullptr;
    const bool use_output = options.output_hadamard && view.svh != nullptr;
    const size_t input_elements = use_input ? (size_t) rows * (size_t) view.k : 0;
    const size_t output_offset = input_elements;
    const float* dot_input = use_input ? scratch : x;
    const int64_t dot_ldx = use_input ? view.k : ldx;
    float* dot_output = use_output ? scratch + output_offset : y;
    const int64_t dot_ldy = use_output ? view.n : ldy;

    std::vector<sycl::event> dot_deps = deps;
    if (use_input) {
        const sycl::event input_event = submit_input_hadamard_rows(queue, view, x, ldx, scratch, rows, deps);
        if (profile != nullptr) profile->input_hadamard = input_event;
        dot_deps = {input_event};
    }
    const bool dot_add_bias = options.add_bias && !use_output;
    sycl::event dot_event;
    if (!submit_packed_dot_dispatch(queue, view, dot_input, dot_ldx, dot_output, dot_ldy, rows, dot_deps, dot_add_bias,
                                    dot_event))
        if (!submit_scalar_dot_dispatch(queue, view, dot_input, dot_ldx, dot_output, dot_ldy, rows, dot_deps,
                                        dot_add_bias, dot_event))
            dot_event = submit_dot_rows(queue, view, dot_input, dot_ldx, dot_output, dot_ldy, rows, dot_deps, dot_add_bias);
    if (profile != nullptr) profile->dot = dot_event;
    if (use_output) {
        const sycl::event output_event =
            submit_output_hadamard_rows(queue, view, dot_output, y, ldy, rows, {dot_event}, options.add_bias);
        if (profile != nullptr) profile->output_hadamard = output_event;
        return output_event;
    }
    return dot_event;
}

}  // namespace strata::kernels::exl3
