// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 packed tiles follow exllamav3's MIT-licensed trellis storage rules.
#pragma once

#include "strata/kernels/exl3_format.hpp"
#include "strata/kernels/raw_linear_format.hpp"

#include <cstdint>
#include <type_traits>

namespace strata::kernels::exl3 {

/// The storage discriminator is part of every projection view.  Raw F16/BF16/F32
/// weights are never represented as a packed matrix with a sentinel bit width.
enum class ProjectionStorage : uint8_t {
    packed_trellis = 0,
    raw = 1,
};

struct RawProjectionView {
    const void* data = nullptr;
    const void* bias = nullptr;
    int64_t k = 0;
    int64_t n = 0;
    strata::kernels::raw::DType dtype = strata::kernels::raw::DType::F32;
    strata::kernels::raw::DType bias_dtype = strata::kernels::raw::DType::F32;
};

/// A packed or source-dtype-preserving raw projection.  The packed members are
/// intentionally kept at the top level so existing packed callers can continue
/// to inspect `gate.k`, `gate.bits`, and the component pointers.  They are valid
/// only when `storage == packed_trellis`; raw callers must use `raw`.
struct ProjectionView {
    ProjectionStorage storage = ProjectionStorage::packed_trellis;

    const uint16_t* trellis = nullptr;
    const uint16_t* suh = nullptr;  // codespell:ignore suh
    const uint16_t* svh = nullptr;
    const uint16_t* bias = nullptr;
    int64_t k = 0;
    int64_t n = 0;
    int bits = 0;
    int codebook = 0;
    /// The order of the compressed tiles in `trellis`.  The device kernels
    /// accept the canonical order only.
    TileOrder tile_order = TileOrder::canonical;
    RawProjectionView raw{};

    ProjectionView() = default;

    explicit ProjectionView(const MatrixView& matrix)
        : storage(ProjectionStorage::packed_trellis), trellis(matrix.trellis), suh(matrix.suh), svh(matrix.svh),
          bias(matrix.bias), k(matrix.k), n(matrix.n), bits(matrix.bits), codebook(matrix.codebook),
          tile_order(matrix.tile_order), raw{} {}

    ProjectionView& operator=(const MatrixView& matrix) {
        *this = ProjectionView(matrix);
        return *this;
    }

    static ProjectionView from_raw(const RawProjectionView& matrix) {
        ProjectionView result;
        result.storage = ProjectionStorage::raw;
        result.k = matrix.k;
        result.n = matrix.n;
        result.raw = matrix;
        return result;
    }

    bool is_packed() const { return storage == ProjectionStorage::packed_trellis; }
    bool is_raw() const { return storage == ProjectionStorage::raw; }

    /// Explicitly recover the legacy packed descriptor.  A raw projection
    /// returns an empty descriptor rather than pretending to be a trellis.
    MatrixView packed_matrix() const {
        if (!is_packed()) return {};
        return {trellis, suh, svh, bias, k, n, bits, codebook, tile_order};
    }
};

/// Neutral descriptor for one gate/up/down expert, usable by CPU and SYCL callers.
struct ExpertView {
    ProjectionView gate{MatrixView{}};
    ProjectionView up{MatrixView{}};
    ProjectionView down{MatrixView{}};
};

static_assert(std::is_trivially_copyable_v<RawProjectionView>);
static_assert(std::is_trivially_copyable_v<ProjectionView>);
static_assert(std::is_trivially_copyable_v<ExpertView>);

}  // namespace strata::kernels::exl3
