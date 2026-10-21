// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "strata/kernels/raw_linear_format.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

namespace strata::kernels::raw {

/// Scalar correctness fallback for FP32 x FP16/BF16/F32 row-major linear weights.
/// The source weight dtype is read directly; no expanded copy is made.
sycl::event linear_fp32(sycl::queue& queue, const LinearView& view, const float* x, int64_t ldx, float* y,
                        int64_t ldy, int64_t rows, const std::vector<sycl::event>& deps = {});

/// Whether every F16 value in `data` is a normal half or a signed zero.
///
/// The device's half to float conversion is exact for those, so the kernel may use it without
/// checking each value.  The answer is false when a subnormal half, an infinity or a NaN is
/// present, and then the kernel converts every value exactly instead.
bool plain_f16_halves(sycl::queue& queue, const void* data, size_t bytes);

/// Gather raw F16/BF16/F32 embedding rows. Device ids are kept in their
/// caller order, repeated ids are allowed, and out-of-range ids produce zero
/// rows. The table is row-major with `row_bytes` bytes per vocabulary row.
void embedding_gather(const void* table, DType dtype, size_t row_bytes, const int32_t* tokens, int64_t n_tok,
                      int64_t n_vocab, int64_t n_embd, float* out, int64_t out_stride, void* stream);

/// Host-token variant used by the single-token path. The caller validates the
/// token range before entering the device launch.
void embedding_gather_one(const void* table, DType dtype, size_t row_bytes, int64_t token, int64_t n_vocab,
                          int64_t n_embd, float* out, void* stream);

}  // namespace strata::kernels::raw
