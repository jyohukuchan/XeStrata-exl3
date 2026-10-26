// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The EXL3 adaptive tier's swap choice: which non-resident experts take resident experts' device slots.
#pragma once

#include <cstdint>
#include <vector>

namespace strata::core {

// One adaptive-tier move: expert `in` of `layer` takes the device slot expert `out` holds.
struct Exl3AdaptSwap {
    double gain = 0.0;
    int32_t layer = -1;
    int32_t in = -1;
    int32_t out = -1;
};

/// The adaptive tier's swap choice, as a pure function of the decayed routing counts: for each MoE layer in
/// [layer_begin, layer_end), a routed-but-not-resident expert (`usage` above 0, `resident` 0) pairs with the
/// least-routed resident one, hottest candidate first; a pair is kept only when the candidate was routed at
/// least `min_count` times and leads the victim by at least `min_gain`; the surviving pairs of every layer are
/// then ordered by gain, largest first, and cut to `cap`.  `usage`/`resident` hold n_layers*n_expert entries in
/// layer-major order (resident: 0 or 1).  `cap` <= 0 takes every surviving pair.  Layers outside the range are
/// never touched, so a stage adapts only the layers it owns.
std::vector<Exl3AdaptSwap> plan_adaptive_swaps(int64_t n_layers, int64_t n_expert, int64_t layer_begin,
                                               int64_t layer_end, const double* usage, const uint8_t* resident,
                                               int64_t cap, double min_count, double min_gain);

}  // namespace strata::core
