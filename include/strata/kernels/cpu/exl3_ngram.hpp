// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU EXL3 n-gram row decoder.  The packed row and mul1 rules originate in
// exllamav3's MIT implementation.
#pragma once

#include "strata/kernels/cpu/exl3.hpp"
#include "strata/kernels/exl3_ngram_format.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu::exl3_ngram {

using Status = strata::kernels::cpu::exl3::Status;

/// Validate a caller-owned n-gram view.  K=1..8 is accepted; the codec is
/// always MUL1 and has a fixed 160-value row dimension.
Status validate(const strata::kernels::exl3_ngram::View& view);

/// Decode selected packed rows into FP32 output.  `row_ids` indexes the
/// caller-provided row window and may contain duplicates or arbitrary order.
/// If it is null, rows 0..count-1 are used.  `explicit_head_ids` is a host
/// array with one valid head id per requested row; it may be null only when
/// num_heads == 1.  Each output row occupies `output_stride` floats, with the
/// first 160 values written and padding untouched.
///
/// The operation uses bounded stack scratch for one row (160 states and 160
/// decoded half values) and never allocates or materializes the table.
Status decode_rows(const strata::kernels::exl3_ngram::View& view, const int64_t* row_ids,
                   const int32_t* explicit_head_ids, size_t count, float* output, size_t output_stride);

}  // namespace strata::kernels::cpu::exl3_ngram
