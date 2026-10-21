// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 packed tiles follow exllamav3's MIT-licensed trellis storage rules.
// Graph-capturable grouped routed EXL3 expert API.
#pragma once

#include "strata/kernels/exl3.hpp"
#include "strata/kernels/exl3_expert_blob.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::kernels::exl3 {

/// Device-resident resolution of one routed entry.
///
/// The format table is device memory and may be replaced between launches, so the plan's checks run on the device:
/// the pipeline's first kernel validates every entry and publishes the result here.  The projection kernels that
/// follow read it instead of repeating the checks for every work item.
struct GroupedEntryPlan {
    uint64_t base = 0;    ///< device address of the entry's expert blob
    int32_t format = -1;  ///< index into the plan's format table, or -1 when the entry has no usable format
    uint32_t flags = 0;   ///< bit 0: the entry is active; bit 1: the entry's format passed the checks
};

/// Device-resident routing plan for grouped unweighted experts.
///
/// The pointer fields point to device allocations, including the two scalar counts.  The capacity and geometry
/// fields are host-side immutable bounds supplied by the caller and are captured by a graph.  Group g owns entries
/// [start[g], start[g+1]); each entry reads x[tok[e]] and writes y[dst[e]].  Destination rows must be unique for a
/// call, so the operation has no inter-work-item accumulation and remains graph-capturable.
struct GroupedPlan {
    const uint64_t* blob_base = nullptr;             ///< [max_groups], device addresses of packed expert blobs
    const int32_t* format_index = nullptr;           ///< [max_groups], device indices into format_table
    const ExpertBlobLayout* format_table = nullptr;  ///< [max_formats], device POD layout table
    const int32_t* start = nullptr;                  ///< [max_groups + 1], device entry boundaries
    const int32_t* tok = nullptr;                    ///< [max_entries], device input token row IDs
    const int32_t* dst = nullptr;                    ///< [max_entries], device output row IDs
    const int32_t* ngroups = nullptr;                ///< device scalar, 0..max_groups
    const int32_t* nentries = nullptr;               ///< device scalar, 0..max_entries
    int64_t max_groups = 0;
    int64_t max_entries = 0;
    int64_t max_formats = 0;
    int64_t n_tokens = 0;     ///< immutable input row bound
    int64_t output_rows = 0;  ///< immutable output row bound
    int64_t hidden = 0;       ///< constant hidden width shared by every format
    int64_t max_ff = 0;       ///< maximum format feed-forward width
    GroupedEntryPlan* entry_plans = nullptr;  ///< [max_entries], device resolution published by the pipeline
};

/// Transform and bias behavior for each projection.  A missing suh/svh descriptor leaves that transform as identity,
/// matching the standalone EXL3 GEMM contract.
struct GroupedOptions {
    GemvOptions gate;
    GemvOptions up;
    GemvOptions down;
};

/// Caller-owned scratch elements.  Two planes of [max_entries, max(max_ff, hidden)] hold intermediate projection
/// dots and gate/up outputs. H128 input and output stages are evaluated in the projection kernels, so no full weight
/// or dynamic device allocation is made.
size_t exl3_grouped_scratch_elements(const GroupedPlan& plan);

/// Execute all currently active routed entries in a device plan.
///
/// The launch is bounded by the immutable capacities and reads ngroups/nentries on device, which permits graph
/// capture followed by count updates.  Inactive entries do not touch output or scratch.  A valid destination whose
/// active group has an invalid format/blob/shape receives quiet NaN; invalid token or destination IDs are skipped to
/// preserve memory safety.  The host should validate layouts with the exact-format registry before launching.
sycl::event exl3_grouped_fp32(sycl::queue& queue, const GroupedPlan& plan, const float* x, int64_t ldx, float* y,
                              int64_t ldy, float* scratch, size_t scratch_elements, GroupedOptions options = {},
                              const std::vector<sycl::event>& deps = {});

}  // namespace strata::kernels::exl3
