// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 n-gram ring rows follow exllamav3's MIT-licensed specialized codec.
#pragma once

#include "strata/kernels/exl3_ngram_format.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

namespace strata::kernels::exl3_ngram {

/// Decode specialized 160-value mul1 n-gram rows into FP32.
///
/// The packed rows, explicit head ids and F16 head-bias table are caller-owned device buffers.  row_stride_words
/// includes any caller padding after the first 1+10*bits words.  The output is row-major with ldy >= 160; no
/// table-wide allocation or implicit row gather is performed.  Valid explicit head ids are in [0, num_heads).
/// An invalid device head writes quiet NaNs to its 160 output values and never reads head_bias; output padding is
/// untouched.  The call is asynchronous and attaches deps to its event.
sycl::event exl3_ngram_dequant(sycl::queue& queue, const View& view, const int32_t* explicit_head_ids, float* out,
                               int64_t ldy, const std::vector<sycl::event>& deps = {});

}  // namespace strata::kernels::exl3_ngram
