// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU routed EXL3 composition; packed matrix rules derive from exllamav3's MIT implementation.
#include "strata/kernels/cpu/exl3_expert.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace strata::kernels::cpu::exl3 {
namespace {

using ExpertView = strata::kernels::exl3::ExpertView;
using MatrixView = strata::kernels::exl3::MatrixView;
using ProjectionView = strata::kernels::exl3::ProjectionView;
using Storage = strata::kernels::exl3::ProjectionStorage;
using DType = strata::kernels::raw::DType;

bool checked_mul(size_t a, size_t b, size_t& out) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) return false;
    out = a * b;
    return true;
}

bool valid_dtype(DType dtype) {
    return dtype == DType::F16 || dtype == DType::BF16 || dtype == DType::F32;
}

Status validate_projection(const ProjectionView& projection) {
    if (projection.k <= 0 || projection.n <= 0 ||
        (projection.raw.k != 0 && projection.raw.k != projection.k) ||
        (projection.raw.n != 0 && projection.raw.n != projection.n))
        return Status::unsupported_layout;
    if (projection.storage == Storage::packed_trellis) {
        const MatrixView matrix = projection.packed_matrix();
        const Status status = validate(matrix);
        if (status != Status::ok) return status;
        if (matrix.suh == nullptr || matrix.svh == nullptr) return Status::missing_data;
        if (matrix.k % 128 != 0 || matrix.n % 128 != 0) return Status::unsupported_layout;
        return Status::ok;
    }
    if (projection.storage != Storage::raw || projection.raw.data == nullptr || !valid_dtype(projection.raw.dtype) ||
        !valid_dtype(projection.raw.bias_dtype) || projection.raw.k != projection.k || projection.raw.n != projection.n)
        return Status::unsupported_layout;
    if (projection.raw.bias != nullptr && projection.raw.n <= 0) return Status::invalid_argument;
    const int weight_bytes = strata::kernels::raw::dtype_bytes(projection.raw.dtype);
    if (weight_bytes <= 0 || (projection.raw.dtype == DType::F32 &&
                              reinterpret_cast<uintptr_t>(projection.raw.data) % alignof(float) != 0) ||
        (projection.raw.dtype != DType::F32 && reinterpret_cast<uintptr_t>(projection.raw.data) % alignof(uint16_t) != 0))
        return Status::unsupported_layout;
    if (projection.raw.bias != nullptr &&
        reinterpret_cast<uintptr_t>(projection.raw.bias) % static_cast<size_t>(strata::kernels::raw::dtype_bytes(projection.raw.bias_dtype)) != 0)
        return Status::unsupported_layout;
    if ((uint64_t) projection.k > std::numeric_limits<uint64_t>::max() / (uint64_t) projection.n)
        return Status::size_overflow;
    const uint64_t elements = (uint64_t) projection.k * (uint64_t) projection.n;
    const uint64_t max_bytes = (uint64_t) std::numeric_limits<ptrdiff_t>::max();
    const uint64_t bias_bytes = (uint64_t) strata::kernels::raw::dtype_bytes(projection.raw.bias_dtype);
    if (elements > max_bytes / (uint64_t) weight_bytes ||
        (projection.raw.bias != nullptr && (uint64_t) projection.n > max_bytes / bias_bytes))
        return Status::size_overflow;
    return Status::ok;
}

Status validate_expert_impl(const ExpertView& view, int64_t rows, const float* x, int64_t ldx, float* y, int64_t ldy) {
    if (rows <= 0 || x == nullptr || y == nullptr) return Status::invalid_argument;
    const ProjectionView* projections[] = {&view.gate, &view.up, &view.down};
    for (const ProjectionView* projection : projections) {
        const Status status = validate_projection(*projection);
        if (status != Status::ok) return status;
    }
    const int64_t hidden = view.gate.k, ff = view.gate.n;
    if (view.up.k != hidden || view.up.n != ff || view.down.k != ff || view.down.n != hidden)
        return Status::unsupported_layout;
    if (ldx < hidden || ldy < hidden) return Status::invalid_argument;
    const uint64_t max_float_elements = (uint64_t) std::numeric_limits<ptrdiff_t>::max() / sizeof(float);
    if ((uint64_t) ldx > max_float_elements / (uint64_t) rows ||
        (uint64_t) ldy > max_float_elements / (uint64_t) rows)
        return Status::size_overflow;
    return Status::ok;
}

float raw_value(const strata::kernels::exl3::RawProjectionView& view, size_t index) {
    if (view.dtype == DType::F16) return strata::kernels::f32_from_f16(static_cast<const uint16_t*>(view.data)[index]);
    if (view.dtype == DType::BF16) return strata::kernels::f32_from_bf16(static_cast<const uint16_t*>(view.data)[index]);
    return static_cast<const float*>(view.data)[index];
}

float raw_bias_value(const strata::kernels::exl3::RawProjectionView& view, size_t index) {
    if (view.bias_dtype == DType::F16) return strata::kernels::f32_from_f16(static_cast<const uint16_t*>(view.bias)[index]);
    if (view.bias_dtype == DType::BF16) return strata::kernels::f32_from_bf16(static_cast<const uint16_t*>(view.bias)[index]);
    return static_cast<const float*>(view.bias)[index];
}

Status raw_linear_one(const strata::kernels::exl3::RawProjectionView& view, const float* x, float* y) {
    if (view.data == nullptr || x == nullptr || y == nullptr) return Status::missing_data;
    for (int64_t out = 0; out < view.n; ++out) {
        double acc = 0.0;
        for (int64_t in = 0; in < view.k; ++in)
            acc += static_cast<double>(x[in]) * static_cast<double>(raw_value(view, (size_t) out * (size_t) view.k + (size_t) in));
        y[out] = static_cast<float>(acc);
        if (view.bias != nullptr) y[out] += raw_bias_value(view, static_cast<size_t>(out));
    }
    return Status::ok;
}

Status projection_one(const ProjectionView& projection, const float* x, float* y, LinearScratch& scratch) {
    if (projection.storage == Storage::packed_trellis) return linear(projection.packed_matrix(), x, y, scratch);
    if (projection.storage == Storage::raw) return raw_linear_one(projection.raw, x, y);
    return Status::unsupported_layout;
}

}  // namespace

Status validate_expert(const ExpertView& view, int64_t rows, const float* x, int64_t ldx, float* y, int64_t ldy) {
    return validate_expert_impl(view, rows, x, ldx, y, ldy);
}

size_t expert_scratch_bytes(const ExpertView& view, int64_t rows) {
    float dummy = 0.0f;
    if (validate_expert_impl(view, rows, &dummy, view.gate.k, &dummy, view.gate.k) != Status::ok)
        return 0;
    const size_t max_k = static_cast<size_t>(std::max({view.gate.k, view.up.k, view.down.k}));
    const size_t ff = static_cast<size_t>(view.gate.n);
    size_t input_bytes = 0, intermediate_values = 0, intermediate_bytes = 0;
    if (!checked_mul(max_k, sizeof(float), input_bytes) ||
        !checked_mul(static_cast<size_t>(rows), ff, intermediate_values) ||
        !checked_mul(intermediate_values, 2 * sizeof(float), intermediate_bytes))
        return 0;
    const size_t max_scratch_bytes = static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max());
    if (input_bytes > max_scratch_bytes - 256 * sizeof(uint16_t)) return 0;
    size_t result = input_bytes + 256 * sizeof(uint16_t);
    if (intermediate_bytes > max_scratch_bytes - result) return 0;
    return result + intermediate_bytes;
}

Status expert(const ExpertView& view, const float* x, int64_t ldx, float* y, int64_t ldy, int64_t rows,
              ExpertScratch& scratch) {
    const Status valid = validate_expert_impl(view, rows, x, ldx, y, ldy);
    if (valid != Status::ok) return valid;
    const size_t hidden = static_cast<size_t>(view.gate.k);
    const size_t ff = static_cast<size_t>(view.gate.n);
    size_t intermediate_count;
    if (!checked_mul(static_cast<size_t>(rows), ff, intermediate_count)) return Status::size_overflow;
    if (scratch.linear.input_basis == nullptr || scratch.linear.input_count < hidden ||
        scratch.linear.input_count < static_cast<size_t>(view.down.k) || scratch.linear.tile == nullptr ||
        scratch.linear.tile_count < 256 || scratch.gate == nullptr || scratch.up == nullptr ||
        scratch.gate_count < intermediate_count || scratch.up_count < intermediate_count)
        return Status::scratch_too_small;
    if (scratch.gate == y || scratch.up == y || scratch.gate == scratch.up) return Status::invalid_argument;

    for (int64_t row = 0; row < rows; ++row) {
        const size_t offset = static_cast<size_t>(row) * ff;
        Status status = projection_one(view.gate, x + static_cast<size_t>(row) * static_cast<size_t>(ldx),
                                       scratch.gate + offset, scratch.linear);
        if (status != Status::ok) return status;
        status = projection_one(view.up, x + static_cast<size_t>(row) * static_cast<size_t>(ldx), scratch.up + offset,
                                scratch.linear);
        if (status != Status::ok) return status;
        for (size_t col = 0; col < ff; ++col) {
            const float gate = scratch.gate[offset + col];
            scratch.gate[offset + col] = (gate / (1.0f + std::exp(-gate))) * scratch.up[offset + col];
        }
        status = projection_one(view.down, scratch.gate + offset, y + static_cast<size_t>(row) * static_cast<size_t>(ldy),
                                scratch.linear);
        if (status != Status::ok) return status;
    }
    (void) hidden;
    return Status::ok;
}

Status expert(const ExpertView& view, const float* x, int64_t ldx, float* y, int64_t ldy, int64_t rows) {
    const Status valid = validate_expert_impl(view, rows, x, ldx, y, ldy);
    if (valid != Status::ok) return valid;
    const size_t hidden = static_cast<size_t>(view.gate.k);
    const size_t ff = static_cast<size_t>(view.gate.n);
    size_t intermediate_count;
    if (!checked_mul(static_cast<size_t>(rows), ff, intermediate_count)) return Status::size_overflow;
    const size_t max_input = std::max(hidden, ff);
    std::vector<float> input(max_input);
    std::vector<float> gate(intermediate_count);
    std::vector<float> up(intermediate_count);
    std::array<uint16_t, 256> tile{};
    ExpertScratch scratch{{input.data(), input.size(), tile.data(), tile.size()}, gate.data(), gate.size(), up.data(),
                           up.size()};
    return expert(view, x, ldx, y, ldy, rows, scratch);
}

}  // namespace strata::kernels::cpu::exl3
