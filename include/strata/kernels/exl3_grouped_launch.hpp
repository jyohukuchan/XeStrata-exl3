// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The grouped gate/up dot's K split, chosen from what the device reports.
//
// The dot runs one work-group of `split_k` sub-groups of 32 work-items.  A split the device
// cannot launch is worse than a smaller one: on a device whose register file holds 65536
// registers a work-group, a kernel using 105 registers a work-item takes at most 624
// work-items, so the 1024 work-items of a split of 32 never launch.  Every limit below is a
// value the device or the compiled kernel reports at run time (the SYCL device queries and
// the CodePlay kernel query in exl3_grouped.cpp); no value here is measured on one card.
// A limit of 0 means the device did not report it, and that limit then does not bind: the
// remaining reported limits still do, which keeps the choice working where the register
// query is absent (the Intel and AMD paths today).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata::kernels::exl3 {

/// The reported limits the grouped gate/up dot's K split is chosen from.
struct GateUpSplitLimits {
    size_t max_work_group_size = 0;              ///< device: work-items a work-group (0: unreported)
    size_t max_num_sub_groups = 0;               ///< device: sub-groups a work-group (0: unreported)
    size_t local_mem_size = 0;                   ///< device: local memory, bytes (0: unreported)
    uint32_t max_registers_per_work_group = 0;   ///< device: registers a work-group (0: unreported)
    uint32_t registers_per_work_item = 0;        ///< compiled kernel: registers a work-item (0: unreported)
    size_t kernel_max_work_group_size = 0;       ///< compiled kernel: work-items a work-group (0: unreported)
    int64_t hidden = 0;                          ///< the dot's K extent
    long env_override = 0;                       ///< STRATA_EXL3_GU_SPLIT_K when set and positive (0: unset)
};

/// The K split for the grouped gate/up dot: the largest split that every reported limit
/// admits, at least 1.  The local-memory term counts the reduction's two padded 16-column
/// planes an eight-partial wave (2 * 16 floats a sub-group).  The register term floors to
/// whole sub-groups, as a work-group is a whole number of them; the kernel's own reported
/// work-group maximum is taken beside it, as the driver already folds the register
/// allocation's granularity into that number where the plain product would not.
inline int gate_up_split_k(const GateUpSplitLimits& limits) noexcept {
    if (limits.hidden <= 0 || (limits.hidden & 3) != 0) return 1;
    size_t cap = static_cast<size_t>(limits.hidden / 4);
    if (limits.kernel_max_work_group_size > 0) cap = std::min(cap, limits.kernel_max_work_group_size / 32);
    if (limits.max_work_group_size > 0) cap = std::min(cap, limits.max_work_group_size / 32);
    if (limits.max_num_sub_groups > 0) cap = std::min(cap, limits.max_num_sub_groups);
    if (limits.local_mem_size > 0) cap = std::min(cap, limits.local_mem_size / (2 * 16 * sizeof(float)));
    if (limits.max_registers_per_work_group > 0 && limits.registers_per_work_item > 0) {
        const size_t max_work_items = limits.max_registers_per_work_group / limits.registers_per_work_item;
        cap = std::min(cap, max_work_items / 32);
    }
    if (limits.env_override > 0) cap = std::min(cap, static_cast<size_t>(limits.env_override));
    return static_cast<int>(std::max<size_t>(1, cap));
}

}  // namespace strata::kernels::exl3
