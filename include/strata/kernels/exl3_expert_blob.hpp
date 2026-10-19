// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 expert blob offsets follow exllamav3's MIT-licensed packed representation.
#pragma once

#include "strata/kernels/exl3_expert_format.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace strata::kernels::exl3 {

struct ExpertComponentLayout {
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

/// POD offsets and metadata for one packed gate/up/down expert.  The layout
/// contains no owning pointer, so it remains valid when a blob is copied or
/// moved to another base address.
struct ExpertProjectionLayout {
    int64_t k = 0;
    int64_t n = 0;
    int bits = 0;
    int codebook = 0;
    ExpertComponentLayout trellis;
    ExpertComponentLayout suh;  // codespell:ignore suh
    ExpertComponentLayout svh;
    ExpertComponentLayout bias;
};

struct ExpertBlobLayout {
    ExpertProjectionLayout gate;
    ExpertProjectionLayout up;
    ExpertProjectionLayout down;
    uint64_t total_bytes = 0;
    uint32_t alignment = 2;
};

static_assert(std::is_standard_layout_v<ExpertComponentLayout>);
static_assert(std::is_standard_layout_v<ExpertProjectionLayout>);
static_assert(std::is_standard_layout_v<ExpertBlobLayout>);

inline MatrixView expert_matrix_view(const ExpertProjectionLayout& projection, const uint8_t* base) {
    MatrixView result;
    result.trellis = projection.trellis.bytes == 0 ? nullptr : reinterpret_cast<const uint16_t*>(base + projection.trellis.offset);
    result.suh = projection.suh.bytes == 0 ? nullptr : reinterpret_cast<const uint16_t*>(base + projection.suh.offset);  // codespell:ignore suh
    result.svh = projection.svh.bytes == 0 ? nullptr : reinterpret_cast<const uint16_t*>(base + projection.svh.offset);
    result.bias = projection.bias.bytes == 0 ? nullptr : reinterpret_cast<const uint16_t*>(base + projection.bias.offset);
    result.k = projection.k;
    result.n = projection.n;
    result.bits = projection.bits;
    result.codebook = projection.codebook;
    return result;
}

inline ExpertView expert_view(const ExpertBlobLayout& layout, const uint8_t* base) {
    return {expert_matrix_view(layout.gate, base), expert_matrix_view(layout.up, base),
            expert_matrix_view(layout.down, base)};
}

}  // namespace strata::kernels::exl3
