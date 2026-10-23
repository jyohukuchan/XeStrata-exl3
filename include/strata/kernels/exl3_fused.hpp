// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Fused transform tails around EXL3 dots: the stages that sit between two dots and pass their
// values on - an output-basis transform, the elementwise stage that consumes it, and the next
// projection's input-basis transform - as one launch, and the two input transforms of one
// activation where two projections of a shared expert read the same input.  Every stage computes
// exactly what the separate kernels compute, in the same order, so the values do not change; the
// fusion only removes the launches between them.  A caller runs the surrounding dots with the
// stages these own disabled (`GemvOptions`), which leaves their own bias to the tail in the same
// position.
#pragma once

#include "strata/kernels/exl3_format.hpp"

#include <sycl/sycl.hpp>

#include <vector>

namespace strata::kernels::exl3 {

/// Whether the fused tails run on this device: they stage one 128-wide block per SG32
/// work-group.  A device without that subgroup form keeps the separate kernels.
bool exl3_fused_tail_supported(void* stream);

/// Whether the fused tails are wanted at all (`STRATA_EXL3_FUSED_TAILS=0` keeps the separate
/// kernels for a like-for-like comparison).  Enabled by default.
bool exl3_fused_tail_enabled();

/// The shared expert's middle in one launch: the gate/up dots' output-basis images (`svh` + bias,
/// or the raw values with the bias the dots left out), the SwiGLU of the two, and the down
/// projection's input-basis image of the result.  `gate`/`up` name the dots' output transforms
/// and `down` the input transform the result feeds; `gate_values` and `up_values` hold the raw
/// dot outputs and `out` receives the down projection's transformed input.  `out` may alias
/// `gate_values`, which is what the separate kernels' SwiGLU writes.  Widths must be multiples
/// of 128.
void exl3_fused_swiglu_tail_fp32(const MatrixView& gate, const MatrixView& up, const MatrixView& down,
                                 float* gate_values, float* up_values, float* out, int64_t rows, void* stream);

/// The GDN z dot's tail in one launch: the gate projection's output-basis image of the raw dot
/// result in `z_values` (`svh` + bias, rewritten in place), the GDN output norm over it with the
/// step's output `o` and `ssm_norm`, and the ssm_out projection's input-basis image of the normed
/// value into `y`.  `ssm_norm` is the state-size vector every head shares, indexed within the
/// head like `gdn_out_norm` does.  Each 128-element block is one GDN head, so this requires a
/// state size of 128 and a value width that is a multiple of 128.  `eps` is the norm's epsilon;
/// like `gdn_out_norm`, the call waits when `stream` is the default one.
void exl3_fused_gdn_tail_fp32(const MatrixView& gate, const MatrixView& out, const float* o, float* z_values,
                              const float* ssm_norm, float* y, int64_t rows, bool silu, float eps, void* stream);

/// The two input-basis transforms of one activation in one launch (exllamav3's `had_r_128_dual`
/// shape): the shared expert's gate and up projections read the same `x`, so one work-group per
/// 128-wide block runs both transforms - each against its own view's `suh`, exactly the values
/// `exl3_gemv_fp32`'s input transform writes for it.  `a_out` and `b_out` are the two transformed
/// images (rows x `a.k` and rows x `b.k`); `a.k` must equal `b.k`, a multiple of 128.  The dots
/// that follow run on their own image with `GemvOptions::input_hadamard` off.
void exl3_input_hadamard_dual_fp32(const MatrixView& a, const MatrixView& b, const float* x, int64_t ldx,
                                   float* a_out, float* b_out, int64_t rows, void* stream);

}  // namespace strata::kernels::exl3
