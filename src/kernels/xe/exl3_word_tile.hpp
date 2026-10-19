// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#pragma once

#include "exl3_word_tile_index.hpp"
#include "exl3_funnel.hpp"
#include <sycl/sycl.hpp>

namespace strata::kernels::exl3::word_cache {

template <int Bits>
struct DistributedTile {
    static_assert(Bits >= 1 && Bits <= 8);
    uint32_t bank0;
    uint32_t bank1;

    template <int RR, int Which>
    inline uint32_t lookup(const sycl::sub_group& group, int lane, int word) const {
        const int owner = word & 31;
        if constexpr (Bits <= 4) {
            return sycl::select_from_group(group, bank0, owner);
        } else if constexpr (owner_bank_unambiguous(Bits, RR, Which)) {
            constexpr uint32_t mask = owner_bank_mask(Bits, RR, Which, 1);
            // Select for this SOURCE lane, from a proven owner mask; not from its destination request.
            const uint32_t source = ((mask >> lane) & 1u) != 0 ? bank1 : bank0;
            return sycl::select_from_group(group, source, owner);
        } else {
            // Both collectives execute for every lane. Choose the destination bank only after receiving.
            const uint32_t low = sycl::select_from_group(group, bank0, owner);
            const uint32_t high = sycl::select_from_group(group, bank1, owner);
            return word < 32 ? low : high;
        }
    }

    template <int RR>
    inline uint16_t state(const sycl::sub_group& group, int lane) const {
        static_assert(RR >= 0 && RR < 8);
        const auto words = state_words(Bits, lane, RR);
        const uint32_t high = lookup<RR, 0>(group, lane, words.high);
        const uint32_t low = lookup<RR, 1>(group, lane, words.low);
        return static_cast<uint16_t>(strata::kernels::xe::funnel_right_u32(high, low, words.shift));
    }
};

}  // namespace strata::kernels::exl3::word_cache
