// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The grouped gate/up dot's K split from reported limits.  CPU only: no device.
//
// The case that failed on the A40 (sm_86) is the first one: 105 registers a work-item
// against 65536 registers a work-group admits at most 19 whole sub-groups, so a split of
// 32 (1024 work-items) never launches and 16 is not needed by hand.

#include "strata/kernels/exl3_grouped_launch.hpp"

#include <cstdio>

namespace {

using strata::kernels::exl3::GateUpSplitLimits;
using strata::kernels::exl3::gate_up_split_k;

int failures = 0;

void check(const char* name, const GateUpSplitLimits& limits, int expected) {
    const int got = gate_up_split_k(limits);
    if (got != expected) {
        std::fprintf(stderr, "exl3_grouped_launch_test: %s: got %d, want %d\n", name, got, expected);
        ++failures;
    }
}

// A device with every limit reported generously: the hidden extent binds.
GateUpSplitLimits generous() {
    GateUpSplitLimits limits;
    limits.max_work_group_size = 1024;
    limits.max_num_sub_groups = 32;
    limits.local_mem_size = 48 * 1024;
    limits.max_registers_per_work_group = 65536;
    limits.registers_per_work_item = 32;
    limits.hidden = 2560;
    return limits;
}

}  // namespace

int main() {
    // The reported failure: 105 regs x 1024 work-items > 65536.  65536/105 = 624
    // work-items, 19 whole sub-groups.
    {
        GateUpSplitLimits limits = generous();
        limits.registers_per_work_item = 105;
        check("register file binds (105 regs, 65536 a work-group)", limits, 19);
    }
    // The hand workaround's split also fits, and a tighter register count needs less.
    {
        GateUpSplitLimits limits = generous();
        limits.registers_per_work_item = 128;
        check("128 regs", limits, 16);
        limits.registers_per_work_item = 255;
        check("255 regs", limits, 8);
    }
    // No register report (Intel/AMD today): the other reported limits bind as before.
    {
        GateUpSplitLimits limits = generous();
        limits.max_registers_per_work_group = 0;
        limits.registers_per_work_item = 0;
        check("no register report", limits, 32);
        limits.registers_per_work_item = 105;
        check("kernel regs without a device limit", limits, 32);
        limits.max_registers_per_work_group = 65536;
        limits.registers_per_work_item = 0;
        check("device limit without kernel regs", limits, 32);
    }
    // The kernel's own reported maximum (the driver folds the register allocation's
    // granularity into it) binds below the plain register product's 19.
    {
        GateUpSplitLimits limits = generous();
        limits.registers_per_work_item = 105;
        limits.kernel_max_work_group_size = 512;
        check("kernel work-group maximum binds", limits, 16);
    }
    // Each older limit still binds on its own.
    {
        GateUpSplitLimits limits = generous();
        limits.registers_per_work_item = 32;
        limits.max_work_group_size = 256;
        check("work-group size binds", limits, 8);
        limits = generous();
        limits.max_num_sub_groups = 4;
        check("sub-group count binds", limits, 4);
        limits = generous();
        limits.local_mem_size = 1024;
        check("local memory binds", limits, 8);
    }
    // The hidden extent and the manual override.
    {
        GateUpSplitLimits limits = generous();
        limits.hidden = 128;
        check("hidden extent binds", limits, 32);
        limits.hidden = 64;
        check("small hidden", limits, 16);
        limits = generous();
        limits.env_override = 16;
        check("env override lowers", limits, 16);
        limits.env_override = 64;
        check("env override never raises", limits, 32);
    }
    // Degenerate geometry keeps the launchable minimum.
    check("no hidden", GateUpSplitLimits{}, 1);
    {
        GateUpSplitLimits limits = generous();
        limits.hidden = 2562;
        check("hidden not a multiple of 4", limits, 1);
        limits.hidden = 2560;
        limits.registers_per_work_item = 4096;
        check("regs above the whole file still launch one sub-group", limits, 1);
    }
    if (failures != 0) {
        std::fprintf(stderr, "exl3_grouped_launch_test: %d failures\n", failures);
        return 1;
    }
    std::printf("exl3_grouped_launch_test OK\n");
    return 0;
}
