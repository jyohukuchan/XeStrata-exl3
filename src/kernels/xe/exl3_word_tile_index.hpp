// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Private compressed EXL3 tile lookup; the format follows exllamav3's MIT-licensed trellis rules.
#pragma once

#include "strata/kernels/exl3_format.hpp"
#include <cstdint>

namespace strata::kernels::exl3::word_cache {

struct StateWords {
    int high;
    int low;
    unsigned shift;
};

constexpr StateWords state_words(int bits, int lane, int rr) {
    const int r = 2 * rr + (lane >> 4), c = lane & 15;
    const int state = tile_state_for_element(r * 16 + c);
    const int word_count = bits * 8;
    const int first = state * bits + bits - 16 + 256 * bits;
    const int end = first + 16;
    const int i0 = first / 32, i1 = (end - 1) / 32;
    return {i0 >= word_count ? i0 - word_count : i0,
            i1 >= word_count ? i1 - word_count : i1,
            static_cast<unsigned>((i1 + 1) * 32 - end)};
}

constexpr int request_word(int bits, int lane, int rr, int which) {
    const auto words = state_words(bits, lane, rr);
    return which == 0 ? words.high : words.low;
}

constexpr uint32_t owner_bank_mask(int bits, int rr, int which, int bank) {
    uint32_t mask = 0;
    for (int destination = 0; destination < 32; ++destination) {
        const int word = request_word(bits, destination, rr, which);
        if ((word >> 5) == bank) mask |= uint32_t(1) << (word & 31);
    }
    return mask;
}

constexpr bool owner_bank_unambiguous(int bits, int rr, int which) {
    return (owner_bank_mask(bits, rr, which, 0) & owner_bank_mask(bits, rr, which, 1)) == 0;
}

constexpr bool all_lookups_unambiguous(int bits) {
    for (int rr = 0; rr < 8; ++rr)
        for (int which = 0; which < 2; ++which)
            if (!owner_bank_unambiguous(bits, rr, which)) return false;
    return true;
}

static_assert(all_lookups_unambiguous(1) && all_lookups_unambiguous(2) && all_lookups_unambiguous(3));
static_assert(all_lookups_unambiguous(4) && all_lookups_unambiguous(5) && all_lookups_unambiguous(6));
static_assert(all_lookups_unambiguous(7) && !all_lookups_unambiguous(8));

}  // namespace strata::kernels::exl3::word_cache
