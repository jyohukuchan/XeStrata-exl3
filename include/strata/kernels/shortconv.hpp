// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/shortconv.hpp - LFM2's short convolution (the conv layers of lfm2moe).
//
// Per row and channel: bcx is the in-projection's [B | C | X] split (three channel-wide parts),
// bx = B * X, a depthwise causal convolution of window `l_cache` over bx (no activation), and
// y = C * conv(bx).  `state` holds the previous l_cache - 1 bx values per channel, oldest first
// (state[j * channels + c], j = 0 .. l_cache - 2), and is left holding the last l_cache - 1 bx
// values of the window's rows.  Weights are [channels, l_cache], oldest tap first.  Rows run in
// order; the state update is part of the call, so a graph replay advances the sequence.
#pragma once

#include <cstdint>

namespace strata::kernels {

void shortconv_apply(float* state, const float* bcx, const float* weight, float* y, int64_t channels,
                     int64_t l_cache, int64_t rows, void* stream);

}  // namespace strata::kernels
