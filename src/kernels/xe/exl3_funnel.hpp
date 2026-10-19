// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstdint>

namespace strata::kernels::xe {

// Low word of (hi:lo) >> (shift & 31). The zero case returns lo in every target.
// Keep the portable expression free of a shift by 32; EXL3 extracts use shifts in 0..31.
inline uint32_t funnel_right_u32(uint32_t hi, uint32_t lo, unsigned shift) {
    const unsigned s = shift & 31u;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__) && __has_builtin(__builtin_amdgcn_alignbit)
    return __builtin_amdgcn_alignbit(hi, lo, s);
#else
    return s == 0 ? lo : (lo >> s) | (hi << (32u - s));
#endif
}

}  // namespace strata::kernels::xe
