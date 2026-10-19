// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 packed offsets follow exllamav3's MIT-licensed representation; raw
// projection spans carry their source dtype explicitly.
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

/// POD offsets and metadata for one packed/raw gate/up/down expert.  The layout
/// contains no owning pointer, so it remains valid when a blob is copied or
/// moved to another base address.
struct ExpertProjectionLayout {
    ProjectionStorage storage = ProjectionStorage::packed_trellis;
    int64_t k = 0;
    int64_t n = 0;
    int bits = 0;
    int codebook = 0;
    ExpertComponentLayout trellis;
    ExpertComponentLayout suh;  // codespell:ignore suh
    ExpertComponentLayout svh;
    ExpertComponentLayout bias;
    raw::DType raw_dtype = raw::DType::F32;
    raw::DType raw_bias_dtype = raw::DType::F32;
    ExpertComponentLayout raw_weights;
    ExpertComponentLayout raw_bias;
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
    const auto projection_view = [base](const ExpertProjectionLayout& projection) {
        if (projection.storage == ProjectionStorage::packed_trellis)
            return ProjectionView(expert_matrix_view(projection, base));
        RawProjectionView raw_view;
        raw_view.data = projection.raw_weights.bytes == 0 ? nullptr : base + projection.raw_weights.offset;
        raw_view.bias = projection.raw_bias.bytes == 0 ? nullptr : base + projection.raw_bias.offset;
        raw_view.k = projection.k;
        raw_view.n = projection.n;
        raw_view.dtype = projection.raw_dtype;
        raw_view.bias_dtype = projection.raw_bias_dtype;
        return ProjectionView::from_raw(raw_view);
    };
    return {projection_view(layout.gate), projection_view(layout.up), projection_view(layout.down)};
}

}  // namespace strata::kernels::exl3
