// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU routed EXL3 composition; packed matrix rules derive from exllamav3's MIT implementation.
#include "strata/kernels/cpu/exl3_expert.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace strata::kernels::cpu::exl3 {
namespace {

using ExpertView = strata::kernels::exl3::ExpertView;
using MatrixView = strata::kernels::exl3::MatrixView;

bool checked_mul(size_t a, size_t b, size_t& out) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) return false;
    out = a * b;
    return true;
}

Status validate_expert(const ExpertView& view, int64_t rows, const float* x, int64_t ldx, float* y, int64_t ldy) {
    if (rows <= 0 || x == nullptr || y == nullptr) return Status::invalid_argument;
    const MatrixView* matrices[] = {&view.gate, &view.up, &view.down};
    for (const MatrixView* matrix : matrices) {
        const Status status = validate(*matrix);
        if (status != Status::ok) return status;
        if (matrix->suh == nullptr || matrix->svh == nullptr) return Status::missing_data;
        if (matrix->k % 128 != 0 || matrix->n % 128 != 0) return Status::unsupported_layout;
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

}  // namespace

size_t expert_scratch_bytes(const ExpertView& view, int64_t rows) {
    float dummy = 0.0f;
    if (validate_expert(view, rows, &dummy, view.gate.k, &dummy, view.gate.k) != Status::ok)
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
    const Status valid = validate_expert(view, rows, x, ldx, y, ldy);
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
        Status status = linear(view.gate, x + static_cast<size_t>(row) * static_cast<size_t>(ldx),
                               scratch.gate + offset, scratch.linear);
        if (status != Status::ok) return status;
        status = linear(view.up, x + static_cast<size_t>(row) * static_cast<size_t>(ldx), scratch.up + offset,
                        scratch.linear);
        if (status != Status::ok) return status;
        for (size_t col = 0; col < ff; ++col) {
            const float gate = scratch.gate[offset + col];
            scratch.gate[offset + col] = (gate / (1.0f + std::exp(-gate))) * scratch.up[offset + col];
        }
        status = linear(view.down, scratch.gate + offset, y + static_cast<size_t>(row) * static_cast<size_t>(ldy),
                        scratch.linear);
        if (status != Status::ok) return status;
    }
    (void) hidden;
    return Status::ok;
}

Status expert(const ExpertView& view, const float* x, int64_t ldx, float* y, int64_t ldy, int64_t rows) {
    const Status valid = validate_expert(view, rows, x, ldx, y, ldy);
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
