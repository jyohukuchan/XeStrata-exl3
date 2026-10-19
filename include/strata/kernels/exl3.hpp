// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 packed tiles follow exllamav3's MIT-licensed trellis storage rules.
#pragma once

#include "strata/kernels/exl3_format.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::kernels::exl3 {

/// Options for the correctness-first single-token FP32 path.
///
/// When enabled and the corresponding descriptor array is present, input and output use the EXL3 128-point
/// normalized Hadamard basis.  A caller that wants to inspect the transformed-basis GEMV can disable either stage.
struct GemvOptions {
    bool input_hadamard = true;
    bool output_hadamard = true;
    bool add_bias = true;
};

/// Number of float elements required by `exl3_gemv_fp32`'s caller-owned scratch buffer.
///
/// The input transform occupies the first `k` elements and the output transform occupies the next `n` elements.
/// A transform that is disabled or has no descriptor pointer contributes no storage.
size_t exl3_gemv_scratch_elements(const MatrixView& view, GemvOptions options = {});

/// Decode the packed trellis into a contiguous row-major [k, n] matrix of IEEE binary16 bits.
///
/// The output is in EXL3's transformed basis.  `view.trellis`, `out`, and all optional arrays are device pointers
/// associated with `queue`; the call launches asynchronously and returns the final event.  `deps` are attached to
/// the launch explicitly so callers can build a copy/decode graph without an implicit synchronization.
sycl::event exl3_reconstruct(sycl::queue& queue, const MatrixView& view, uint16_t* out,
                             const std::vector<sycl::event>& deps = {});

/// Single-token FP32 GEMV with dequantization fused into the dot product.
///
/// The logical weight is [k, n] row-major and the result is `y[j] = sum_i x[i] * W[i,j]`.  If the descriptor has
/// `suh`/`svh`, the enabled options run normalized H128 before and after the dot.  `bias` is added after the output
/// transform.  No expanded weight matrix is produced.  A non-null scratch buffer of at least
/// `exl3_gemv_scratch_elements(view, options)` floats is required when a transform is active; it is owned by the
/// caller and remains live until the returned event completes.  Input/output and scratch ranges must not overlap.
sycl::event exl3_gemv_fp32(sycl::queue& queue, const MatrixView& view, const float* x, float* y, float* scratch,
                           size_t scratch_elements, GemvOptions options = {},
                           const std::vector<sycl::event>& deps = {});

/// Number of float elements required by exl3_gemm_fp32 for rows input rows.
/// Scratch is laid out as rows*k input-transform values followed by rows*n dot values.
size_t exl3_gemm_scratch_elements(const MatrixView& view, int64_t rows, GemvOptions options = {});

/// Batched FP32 linear operation for prefill and MTP.
///
/// Inputs are rows vectors at x[r*ldx], outputs are rows vectors at y[r*ldy].  The leading dimensions may exceed
/// k and n, respectively, so callers can keep padded row strides.  The weight remains packed and is decoded inside
/// each dot; no expanded matrix is allocated.  Scratch is caller-owned and has the size returned by
/// exl3_gemm_scratch_elements when a Hadamard stage is active.
sycl::event exl3_gemm_fp32(sycl::queue& queue, const MatrixView& view, const float* x, int64_t ldx, float* y,
                           int64_t ldy, int64_t rows, float* scratch, size_t scratch_elements,
                           GemvOptions options = {}, const std::vector<sycl::event>& deps = {});

}  // namespace strata::kernels::exl3
