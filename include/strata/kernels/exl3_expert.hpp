// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 packed tiles follow exllamav3's MIT-licensed trellis storage rules.
#pragma once

#include "strata/kernels/exl3.hpp"
#include "strata/kernels/exl3_expert_format.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::kernels::exl3 {

/// Transform options for each projection of an expert.
struct ExpertOptions {
    GemvOptions gate;
    GemvOptions up;
    GemvOptions down;
};

/// Number of caller-owned float scratch elements required by exl3_expert_fp32.
///
/// The projection workspaces are followed by contiguous gate and up outputs.  The gate output is overwritten by
/// silu(gate) * up before the down projection reads it.
size_t exl3_expert_scratch_elements(const ExpertView& view, int64_t rows, ExpertOptions options = {});

/// Batched packed EXL3 MLP for an unweighted expert.
///
/// x[r*ldx] contains a hidden-wide input vector and y[r*ldy] receives a hidden-wide output vector.  The operation
/// is silu(gate(x)) * up(x), followed by down(...).  Leading dimensions may include padding.  Every projection
/// keeps its packed trellis and its own codebook/bit width; no full weight matrix is expanded.  All device pointers
/// and caller scratch stay live until the returned event completes.
sycl::event exl3_expert_fp32(sycl::queue& queue, const ExpertView& view, const float* x, int64_t ldx, float* y,
                             int64_t ldy, int64_t rows, float* scratch, size_t scratch_elements,
                             ExpertOptions options = {}, const std::vector<sycl::event>& deps = {});

}  // namespace strata::kernels::exl3
