// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU EXL3 n-gram row decoder.  This is a bounded row path, not a table loader.
#include "strata/kernels/cpu/exl3_ngram.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <array>
#include <cstddef>
#include <limits>

namespace strata::kernels::cpu::exl3_ngram {
namespace {

using View = strata::kernels::exl3_ngram::View;
using CpuStatus = strata::kernels::cpu::exl3::Status;
constexpr int kRowDim = strata::kernels::exl3_ngram::kRowDim;

}  // namespace

CpuStatus validate(const View& view) {
    if (view.bits < 1 || view.bits > 8) return CpuStatus::unsupported_bits;
    if (view.row_count < 0 || view.words_per_row_stride < 0 || view.num_heads <= 0)
        return CpuStatus::unsupported_layout;
    const size_t words = strata::kernels::exl3_ngram::words_per_row(view.bits);
    if (words == 0 || view.words_per_row_stride < static_cast<int64_t>(words))
        return CpuStatus::unsupported_layout;
    if (static_cast<uint64_t>(view.row_count) > std::numeric_limits<size_t>::max() ||
        static_cast<uint64_t>(view.words_per_row_stride) > std::numeric_limits<size_t>::max())
        return CpuStatus::size_overflow;
    const size_t ptrdiff_u16 = static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(uint16_t);
    const size_t ptrdiff_f32 = static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
    if (view.row_count > 0 && view.packed_rows == nullptr) return CpuStatus::missing_data;
    if (view.head_bias == nullptr) return CpuStatus::missing_data;
    if (view.row_count > 0) {
        const size_t rows = static_cast<size_t>(view.row_count);
        const size_t stride = static_cast<size_t>(view.words_per_row_stride);
        if ((rows > 1 && stride > ptrdiff_u16) || rows - 1 > (ptrdiff_u16 - (words - 1)) / stride)
            return CpuStatus::size_overflow;
    }
    if (static_cast<uint64_t>(view.num_heads) > ptrdiff_u16 / static_cast<size_t>(kRowDim))
        return CpuStatus::size_overflow;
    if (static_cast<uint64_t>(view.num_heads) > std::numeric_limits<size_t>::max() / static_cast<size_t>(kRowDim))
        return CpuStatus::size_overflow;
    (void) ptrdiff_f32;
    return CpuStatus::ok;
}

CpuStatus decode_rows(const View& view, const int64_t* row_ids, const int32_t* explicit_head_ids, size_t count,
                      float* output, size_t output_stride) {
    const CpuStatus valid = validate(view);
    if (valid != CpuStatus::ok) return valid;
    if (count == 0) return CpuStatus::ok;
    if (output == nullptr) return CpuStatus::missing_data;
    if (output_stride < static_cast<size_t>(kRowDim)) return CpuStatus::output_too_small;
    const size_t ptrdiff_i64 = static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(int64_t);
    const size_t ptrdiff_i32 = static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(int32_t);
    const size_t ptrdiff_f32 = static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
    if (count > ptrdiff_i64 || (explicit_head_ids != nullptr && count > ptrdiff_i32))
        return CpuStatus::size_overflow;
    if (count > 0 && output_stride > ptrdiff_f32)
        return CpuStatus::size_overflow;
    if (count > 0 && count - 1 > (ptrdiff_f32 - (kRowDim - 1)) / output_stride)
        return CpuStatus::size_overflow;
    if (row_ids == nullptr && count > static_cast<size_t>(view.row_count)) return CpuStatus::invalid_argument;
    if (explicit_head_ids == nullptr && view.num_heads != 1) return CpuStatus::missing_data;

    // Validate every request before writing any destination row.
    for (size_t requested = 0; requested < count; ++requested) {
        const int64_t row_id = row_ids == nullptr ? static_cast<int64_t>(requested) : row_ids[requested];
        if (row_id < 0 || row_id >= view.row_count) return CpuStatus::invalid_argument;
        const int64_t head_id = explicit_head_ids == nullptr ? 0 : explicit_head_ids[requested];
        if (head_id < 0 || head_id >= view.num_heads) return CpuStatus::invalid_argument;
    }

    const size_t row_stride = static_cast<size_t>(view.words_per_row_stride);
    std::array<uint16_t, kRowDim> states{};
    std::array<uint16_t, kRowDim> decoded{};
    for (size_t requested = 0; requested < count; ++requested) {
        const int64_t row_id = row_ids == nullptr ? static_cast<int64_t>(requested) : row_ids[requested];
        const int64_t head_id = explicit_head_ids == nullptr ? 0 : explicit_head_ids[requested];
        const uint16_t* packed = view.packed_rows + static_cast<size_t>(row_id) * row_stride;

        // Word zero is an FP16 row scale.  The remaining words form a 160*K
        // little-endian bit ring.  Each 16-bit state consumes bits from a
        // tail-biting window, exactly as the Python reference's _unpack_states.
        for (int i = 0; i < kRowDim; ++i) {
            uint16_t state = 0;
            for (int m = 0; m < 16; ++m) {
                const int position = (i - m / view.bits + kRowDim) % kRowDim;
                const int stream_bit = position * view.bits + m % view.bits;
                const uint16_t word = packed[1 + stream_bit / 16];
                state |= static_cast<uint16_t>(((word >> (stream_bit & 15)) & 1u) << m);
            }
            states[static_cast<size_t>(i)] = state;
        }
        const CpuStatus decoded_status = strata::kernels::cpu::exl3::decode_codebook(
            states.data(), states.size(), 2, decoded.data(), decoded.size());
        if (decoded_status != CpuStatus::ok) return decoded_status;

        const float scale = strata::kernels::f32_from_f16(packed[0]);
        const uint16_t* bias = view.head_bias + static_cast<size_t>(head_id) * kRowDim;
        float* destination = output + requested * output_stride;
        for (int i = 0; i < kRowDim; ++i) {
            const float code = strata::kernels::f32_from_f16(decoded[static_cast<size_t>(i)]);
            const float head_value = strata::kernels::f32_from_f16(bias[i]);
            volatile float scaled = code * scale;
            volatile float result = scaled + head_value;
            destination[i] = result;
        }
    }
    return CpuStatus::ok;
}

}  // namespace strata::kernels::cpu::exl3_ngram
