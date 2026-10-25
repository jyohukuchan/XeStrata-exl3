// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/shortconv.cpp - LFM2's short convolution: one work-item a channel, the window's
// rows in order in registers (l_cache <= 8; LFM2 writes 3).  Plain FP32 arithmetic, no fast math.
#include "strata/kernels/shortconv.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_shortconv
#include "strata/kernels/device_arch_tag.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int64_t kMaxCache = 8;

}  // namespace

void shortconv_apply(float* state, const float* bcx, const float* weight, float* y, int64_t channels,
                     int64_t l_cache, int64_t rows, void* stream) {
    if (!stream) throw std::invalid_argument("shortconv requires an explicit stream");
    if (!state || !bcx || !weight || !y || channels <= 0 || rows <= 0 || l_cache < 2 || l_cache > kMaxCache)
        throw std::invalid_argument("shortconv: bad geometry or null buffer");
    auto& q = core::Runtime::get().stream(stream);
    const int ch = (int) channels, lc = (int) l_cache, nr = (int) rows;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) ch), [=](sycl::id<1> id) {
            const size_t c = id[0];
            float hist[kMaxCache];   // oldest first; hist[lc - 1] is the row being read
            for (int j = 0; j < lc - 1; ++j) hist[j] = state[(size_t) j * ch + c];
            const float* w = weight + c * (size_t) lc;
            for (int t = 0; t < nr; ++t) {
                const float* row = bcx + (size_t) t * 3 * ch;
                const float bx = row[c] * row[2 * (size_t) ch + c];
                hist[lc - 1] = bx;
                float acc = 0.0f;
                for (int j = 0; j < lc; ++j) acc += w[j] * hist[j];
                y[(size_t) t * ch + c] = row[(size_t) ch + c] * acc;
                for (int j = 0; j + 1 < lc; ++j) hist[j] = hist[j + 1];
            }
            for (int j = 0; j < lc - 1; ++j) state[(size_t) j * ch + c] = hist[j];
        });
    });
}

}  // namespace strata::kernels
