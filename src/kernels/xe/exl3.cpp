// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 trellis and codebook rules follow exllamav3's MIT-licensed implementation.
//
// This is deliberately a scalar correctness path.  It keeps the packed trellis in its original form and decodes a
// state only when a GEMV term needs it.  A later optimized path can replace the launch bodies without changing the
// descriptor or event contract established here.
#include "strata/kernels/exl3.hpp"

#include "strata/kernels/f16_bits.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_exl3
#include "strata/kernels/device_arch_tag.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels::exl3 {
namespace {

constexpr float kHadamardScale = 0.08838834764831845f;  // 1 / sqrt(128), as in exllamav3's reconstruct path
// gitleaks:allow -- public EXL3 codebook multipliers and bit masks, never credentials.

inline void check_view(const MatrixView& view, const char* who) {
    if (view.trellis == nullptr) throw std::invalid_argument(std::string(who) + ": trellis is null");
    if (view.k <= 0 || view.n <= 0) throw std::invalid_argument(std::string(who) + ": dimensions must be positive");
    if (view.k % 16 != 0 || view.n % 16 != 0)
        throw std::invalid_argument(std::string(who) + ": k and n must be multiples of 16");
    if (view.bits < 1 || view.bits > 8)
        throw std::invalid_argument(std::string(who) + ": bits must be in 1..8");
    if (view.codebook < 0 || view.codebook > 2)
        throw std::invalid_argument(std::string(who) + ": codebook must be 0, 1 or 2");
    const uint64_t elements = (uint64_t) view.k * (uint64_t) view.n;
    if (elements > (uint64_t) std::numeric_limits<size_t>::max())
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
    const int w = word % word_count;
    return (uint32_t) packed[2 * w] | ((uint32_t) packed[2 * w + 1] << 16);
}

inline uint16_t extract_state(const uint16_t* packed, int bits, int state) {
    const int word_count = bits * 8;  // (256 * bits) / 32
    const int b0 = state * bits + bits - 16 + 256 * bits;
    const int b1 = b0 + 16;
    const int i0 = b0 / 32;
    const int i1 = (b1 - 1) / 32;
    const int shift = (i1 + 1) * 32 - b1;
    const uint64_t merged = ((uint64_t) load_word32(packed, i0, word_count) << 32) |
                            (uint64_t) load_word32(packed, i1, word_count);
    return (uint16_t) ((merged >> shift) & 0xFFFFu);
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
    const uint32_t byte_sum = (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
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

inline int hadamard_sign(int row, int col) {
    unsigned bits = (unsigned) (row & col);
    int parity = 0;
    while (bits != 0) {
        parity ^= 1;
        bits &= bits - 1;
    }
    return parity == 0 ? 1 : -1;
}

sycl::event submit_input_hadamard(sycl::queue& queue, const MatrixView& view, const float* x, float* transformed,
                                  const std::vector<sycl::event>& deps) {
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

sycl::event submit_output_hadamard(sycl::queue& queue, const MatrixView& view, const float* dot, float* y,
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

sycl::event submit_input_hadamard_rows(sycl::queue& queue, const MatrixView& view, const float* x, int64_t ldx,
                                       float* transformed, int64_t rows, const std::vector<sycl::event>& deps) {
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

sycl::event submit_output_hadamard_rows(sycl::queue& queue, const MatrixView& view, const float* dot, float* y,
                                        int64_t ldy, int64_t rows, const std::vector<sycl::event>& deps,
                                        bool add_bias) {
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

}  // namespace

size_t exl3_gemv_scratch_elements(const MatrixView& view, GemvOptions options) {
    size_t result = 0;
    if (options.input_hadamard && view.suh != nullptr) result += (size_t) view.k;
    if (options.output_hadamard && view.svh != nullptr) result += (size_t) view.n;
    return result;
}

size_t exl3_gemm_scratch_elements(const MatrixView& view, int64_t rows, GemvOptions options) {
    if (rows <= 0) return 0;
    const size_t per_row = exl3_gemv_scratch_elements(view, options);
    if (per_row > std::numeric_limits<size_t>::max() / (size_t) rows)
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
                           size_t scratch_elements, GemvOptions options, const std::vector<sycl::event>& deps) {
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
        dot_deps = {input_event};
    }
    const bool dot_add_bias = options.add_bias && !use_output;
    const sycl::event dot_event = submit_dot(queue, view, dot_input, dot_output, dot_deps, dot_add_bias);
    if (use_output)
        return submit_output_hadamard(queue, view, dot_output, y, {dot_event}, options.add_bias);
    return dot_event;
}

sycl::event exl3_gemm_fp32(sycl::queue& queue, const MatrixView& view, const float* x, int64_t ldx, float* y,
                           int64_t ldy, int64_t rows, float* scratch, size_t scratch_elements, GemvOptions options,
                           const std::vector<sycl::event>& deps) {
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
        dot_deps = {input_event};
    }
    const bool dot_add_bias = options.add_bias && !use_output;
    const sycl::event dot_event =
        submit_dot_rows(queue, view, dot_input, dot_ldx, dot_output, dot_ldy, rows, dot_deps, dot_add_bias);
    if (use_output)
        return submit_output_hadamard_rows(queue, view, dot_output, y, ldy, rows, {dot_event}, options.add_bias);
    return dot_event;
}

}  // namespace strata::kernels::exl3
