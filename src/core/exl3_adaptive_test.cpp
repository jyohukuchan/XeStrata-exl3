// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The adaptive tier's swap choice, as a pure function: no model, no device.
#include "strata/core/exl3_adaptive.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "exl3_adaptive_test: %s\n", what);
    }
}

bool has_swap(const std::vector<strata::core::Exl3AdaptSwap>& swaps, int32_t layer, int32_t in, int32_t out) {
    for (const auto& swap : swaps)
        if (swap.layer == layer && swap.in == in && swap.out == out) return true;
    return false;
}

void basic_hot_takes_cold_slot() {
    // One layer of four: experts 0 and 1 resident, 2 and 3 routed but not.  The hottest missing
    // expert pairs with the coldest resident one.
    const double usage[] = {1.0, 0.5, 5.0, 4.0};
    const uint8_t resident[] = {1, 1, 0, 0};
    const auto swaps = strata::core::plan_adaptive_swaps(1, 4, 0, 1, usage, resident, 0, 2.0, 1.5);
    check(swaps.size() == 2, "basic: two swaps");
    if (swaps.size() == 2) {
        check(has_swap(swaps, 0, 2, 1), "basic: expert 2 takes expert 1's slot");
        check(has_swap(swaps, 0, 3, 0), "basic: expert 3 takes expert 0's slot");
        check(swaps[0].in == 2, "basic: the largest gain comes first");
    }
}

void cold_candidate_stays_out() {
    // A missing expert routed fewer than min_count times never moves in, however cold the victim.
    const double usage[] = {0.0, 0.0, 1.5, 0.0};
    const uint8_t resident[] = {1, 1, 0, 1};
    const auto swaps = strata::core::plan_adaptive_swaps(1, 4, 0, 1, usage, resident, 0, 2.0, 1.5);
    check(swaps.empty(), "min_count: a once-routed expert stays out");
}

void small_gain_stays_out() {
    // A candidate leading the coldest victim by less than min_gain never moves in.
    const double usage[] = {2.0, 2.0, 3.0, 2.0};
    const uint8_t resident[] = {1, 1, 0, 1};
    const auto swaps = strata::core::plan_adaptive_swaps(1, 4, 0, 1, usage, resident, 0, 2.0, 1.5);
    check(swaps.empty(), "min_gain: a 3.0 against a 2.0 stays out");
}

void cap_keeps_largest_gains() {
    // Three layers with one swap each; the cap keeps the largest gain.
    const double usage[] = {0.5, 9.0, 0.5, 4.0, 0.5, 6.0};
    const uint8_t resident[] = {1, 0, 1, 0, 1, 0};
    const auto swaps = strata::core::plan_adaptive_swaps(3, 2, 0, 3, usage, resident, 1, 2.0, 1.5);
    check(swaps.size() == 1, "cap: one swap");
    if (!swaps.empty()) check(swaps[0].layer == 0 && swaps[0].in == 1, "cap: the largest gain wins");
}

void range_leaves_other_layers() {
    // A stage adapts only the layers it owns.
    const double usage[] = {0.5, 9.0, 0.5, 9.0};
    const uint8_t resident[] = {1, 0, 1, 0};
    const auto swaps = strata::core::plan_adaptive_swaps(2, 2, 1, 2, usage, resident, 0, 2.0, 1.5);
    check(swaps.size() == 1, "range: one swap");
    if (!swaps.empty()) check(swaps[0].layer == 1, "range: outside layers are untouched");
}

void settled_cache_moves_nothing() {
    // Every routed expert resident: nothing to move.  Nothing routed at all: nothing to choose.
    const double usage[] = {3.0, 4.0, 5.0, 6.0};
    const uint8_t resident[] = {1, 1, 1, 1};
    check(strata::core::plan_adaptive_swaps(1, 4, 0, 1, usage, resident, 0, 2.0, 1.5).empty(),
          "all-resident: no swaps");
    const double cold[] = {0.0, 0.0, 0.0, 0.0};
    const uint8_t split[] = {1, 1, 0, 0};
    check(strata::core::plan_adaptive_swaps(1, 4, 0, 1, cold, split, 0, 2.0, 1.5).empty(),
          "unrouted: no swaps");
}

void empty_range_moves_nothing() {
    const double usage[] = {0.5, 9.0};
    const uint8_t resident[] = {1, 0};
    check(strata::core::plan_adaptive_swaps(1, 2, 1, 1, usage, resident, 0, 2.0, 1.5).empty(),
          "empty range: no swaps");
    check(strata::core::plan_adaptive_swaps(0, 0, 0, 0, nullptr, nullptr, 0, 2.0, 1.5).empty(),
          "empty model: no swaps");
}

}  // namespace

int main() {
    basic_hot_takes_cold_slot();
    cold_candidate_stays_out();
    small_gain_stays_out();
    cap_keeps_largest_gains();
    range_leaves_other_layers();
    settled_cache_moves_nothing();
    empty_range_moves_nothing();
    if (failures == 0) std::printf("exl3_adaptive_test: OK\n");
    return failures == 0 ? 0 : 1;
}
