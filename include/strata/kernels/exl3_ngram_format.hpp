// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Neutral EXL3 n-gram row format.  This is separate from the 16x16 linear
// trellis MatrixView because n-gram rows have 160 values and a row scale.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::exl3_ngram {

inline constexpr int kRowDim = 160;

/// Number of uint16 words in one packed n-gram row, including its FP16 row
/// scale at word zero.  The remaining words hold the 160 * bits ring bits.
constexpr size_t words_per_row(int bits) {
    return bits >= 1 && bits <= 8 ? 1u + static_cast<size_t>(10 * bits) : 0u;
}

/// A view over caller-owned, already-requested packed rows.  `packed_rows`
/// contains `row_count` rows separated by `words_per_row_stride` uint16 words;
/// it does not describe the complete n-gram table or safetensors shards.
/// `head_bias` contains `num_heads * 160` binary16 values in head-major order.
/// `explicit_head_ids` is deliberately a host pointer in the CPU API below.
struct View {
    const uint16_t* packed_rows = nullptr;
    int64_t row_count = 0;
    int64_t words_per_row_stride = 0;
    int bits = 0;
    const uint16_t* head_bias = nullptr;
    int64_t num_heads = 0;
};

}  // namespace strata::kernels::exl3_ngram
