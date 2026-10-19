// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU routed EXL3 composition; packed matrix rules derive from exllamav3's MIT implementation.
#pragma once

#include "strata/kernels/cpu/exl3.hpp"
#include "strata/kernels/exl3_expert_format.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu::exl3 {

/// Caller-owned bounded workspace for one unweighted routed expert.  The
/// gate/up arrays hold rows*ff values; `linear` is reused by all projections.
struct ExpertScratch {
    LinearScratch linear;
    float* gate = nullptr;
    size_t gate_count = 0;
    float* up = nullptr;
    size_t up_count = 0;
};

/// Bytes for a conventional layout containing one LinearScratch input vector,
/// one 256-state tile, and the gate/up intermediate arrays.  Zero means the
/// descriptor or the row count is invalid/overflowing.
size_t expert_scratch_bytes(const strata::kernels::exl3::ExpertView& view, int64_t rows);

/// Routed expert composition for FP32 rows:
/// `y = down(silu(gate(x)) * up(x))`.  Each projection keeps its own bits and
/// codebook.  No full weight matrix is materialized.
Status expert(const strata::kernels::exl3::ExpertView& view, const float* x, int64_t ldx, float* y, int64_t ldy,
              int64_t rows, ExpertScratch& scratch);

/// Convenience wrapper for bounded, non-hot-path checks.  It allocates only
/// activation/intermediate scratch and still never expands a weight matrix.
Status expert(const strata::kernels::exl3::ExpertView& view, const float* x, int64_t ldx, float* y, int64_t ldy,
              int64_t rows);

}  // namespace strata::kernels::cpu::exl3
