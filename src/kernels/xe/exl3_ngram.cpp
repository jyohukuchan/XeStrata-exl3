// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 n-gram ring rows follow exllamav3's MIT-licensed specialized codec.
#include "strata/kernels/exl3_ngram.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "exl3_cb2.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_exl3_ngram
#include "strata/kernels/device_arch_tag.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels::exl3_ngram {
namespace {

constexpr int kNgramRowDim = strata::kernels::exl3_ngram::kRowDim;
constexpr uint32_t kMul1 = 0x83DCD12Du;
// gitleaks:allow -- public EXL3 mul1 constants, never credentials.

void check_view(const View& view, const int32_t* explicit_head_ids, float* out, int64_t ldy) {
    if (view.packed_rows == nullptr || explicit_head_ids == nullptr || view.head_bias == nullptr || out == nullptr)
        throw std::invalid_argument("exl3_ngram_dequant: packed, heads, bias and output are required");
    if (view.row_count <= 0) throw std::invalid_argument("exl3_ngram_dequant: rows must be positive");
    if (view.bits < 1 || view.bits > 8) throw std::invalid_argument("exl3_ngram_dequant: bits must be in 1..8");
    const int64_t words = 1 + (kNgramRowDim * (int64_t) view.bits) / 16;
    if (view.words_per_row_stride < words)
        throw std::invalid_argument("exl3_ngram_dequant: row stride is smaller than the packed row");
    if (view.num_heads <= 0) throw std::invalid_argument("exl3_ngram_dequant: num_heads must be positive");
    if (ldy < kNgramRowDim) throw std::invalid_argument("exl3_ngram_dequant: ldy must cover 160 values");
    const uint64_t max_u16_elements = (uint64_t) std::numeric_limits<ptrdiff_t>::max() / sizeof(uint16_t);
    const uint64_t max_i32_elements = (uint64_t) std::numeric_limits<ptrdiff_t>::max() / sizeof(int32_t);
    if ((uint64_t) view.row_count > max_u16_elements / (uint64_t) view.words_per_row_stride ||
        (uint64_t) view.row_count > max_i32_elements ||
        (uint64_t) view.num_heads > max_u16_elements / kNgramRowDim)
        throw std::invalid_argument("exl3_ngram_dequant: caller span exceeds host address range");
    if ((uint64_t) view.row_count > (uint64_t) std::numeric_limits<ptrdiff_t>::max() / sizeof(float) /
                                  (uint64_t) ldy)
        throw std::invalid_argument("exl3_ngram_dequant: output span exceeds host address range");
}

template <bool HalfFma>
sycl::event submit_ngram_dequant(sycl::queue& queue, const View& view, const int32_t* explicit_head_ids, float* out,
                                  int64_t ldy,
                                  const std::vector<sycl::event>& deps) {
    return queue.submit([&](sycl::handler& h) {
        if (!deps.empty()) h.depends_on(deps);
        h.parallel_for(sycl::range<1>((size_t) view.row_count * kNgramRowDim), [=](sycl::id<1> id) {
            const size_t flat = (size_t) id[0];
            const int64_t row = (int64_t) (flat / kNgramRowDim);
            const int i = (int) (flat % kNgramRowDim);
            const uint16_t* packed = view.packed_rows + row * view.words_per_row_stride;
            const int head = explicit_head_ids[row];
            if (head < 0 || head >= view.num_heads) {
                out[row * ldy + i] = sycl::nan((uint32_t) 0);
                return;
            }
            uint32_t state = 0;
            for (int m = 0; m < 16; ++m) {
                int position = i - m / view.bits;
                if (position < 0) position += kNgramRowDim;
                const int stream_bit = position * view.bits + m % view.bits;
                const uint32_t bit = (packed[1 + (stream_bit >> 4)] >> (stream_bit & 15)) & 1u;
                state |= bit << m;
            }
            const uint32_t product = state * kMul1;
            const uint32_t byte_sum = (product & 0xFFu) + ((product >> 8) & 0xFFu) +
                                      ((product >> 16) & 0xFFu) + ((product >> 24) & 0xFFu);
            // The codebook 2 value, in the same form the other EXL3 kernels use: the half fused multiply add
            // where the device reports fp16 arithmetic, the portable form otherwise.  Both forms return the
            // same F16-rounded value.
            const float codebook = strata::kernels::exl3::cb2_value_f32<HalfFma>((int) byte_sum);
            const float scale = f32_from_f16(packed[0]);
            const float bias = f32_from_f16(view.head_bias[(size_t) head * kNgramRowDim + i]);
            out[row * ldy + i] = codebook * scale + bias;
        });
    });
}

}  // namespace

sycl::event exl3_ngram_dequant(sycl::queue& queue, const View& view, const int32_t* explicit_head_ids, float* out,
                               int64_t ldy,
                               const std::vector<sycl::event>& deps) {
    check_view(view, explicit_head_ids, out, ldy);
    if (strata::kernels::exl3::cb2_half_fma_enabled(queue))
        return submit_ngram_dequant<true>(queue, view, explicit_head_ids, out, ldy, deps);
    return submit_ngram_dequant<false>(queue, view, explicit_head_ids, out, ldy, deps);
}

}  // namespace strata::kernels::exl3_ngram
