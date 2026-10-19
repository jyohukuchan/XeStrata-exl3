// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU EXL3 n-gram row parity.  Synthetic rows cover every supported K without
// loading a sidecar or materializing the complete n-gram table.
#include "strata/kernels/cpu/exl3_ngram.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace ngram = strata::kernels::cpu::exl3_ngram;
namespace format = strata::kernels::exl3_ngram;

namespace {

constexpr uint32_t kMul1 = 0x83DCD12Du;
constexpr int kRows = 5;
constexpr int kHeads = 2;
int failures = 0;

void check(bool condition, const char* label) {
    std::cout << "  " << label << ": " << (condition ? "OK" : "FAIL") << '\n';
    if (!condition) ++failures;
}

uint16_t expected_code(uint16_t state) {
    const uint32_t product = static_cast<uint32_t>(state) * kMul1;
    const int byte_sum = static_cast<int>(product & 0xffu) + static_cast<int>((product >> 8) & 0xffu) +
                         static_cast<int>((product >> 16) & 0xffu) + static_cast<int>(product >> 24);
    const uint16_t h = static_cast<uint16_t>(0x6400u + byte_sum);
    volatile float product_f = strata::kernels::f32_from_f16(h) * strata::kernels::f32_from_f16(0x1eeeu);
    volatile float result_f = product_f + strata::kernels::f32_from_f16(0xc931u);
    return strata::kernels::f16_from_f32(result_f);
}

uint16_t state_from_row(const uint16_t* packed, int bits, int i) {
    std::array<uint8_t, format::kRowDim * 8> stream{};
    for (int stream_bit = 0; stream_bit < format::kRowDim * bits; ++stream_bit)
        stream[static_cast<size_t>(stream_bit)] = static_cast<uint8_t>((packed[1 + stream_bit / 16] >> (stream_bit & 15)) & 1u);
    uint16_t state = 0;
    for (int m = 0; m < 16; ++m) {
        const int position = (i - m / bits + format::kRowDim) % format::kRowDim;
        const int stream_bit = position * bits + m % bits;
        state |= static_cast<uint16_t>(stream[static_cast<size_t>(stream_bit)] << m);
    }
    return state;
}

std::vector<uint16_t> make_row(int bits, int row, uint16_t scale) {
    const size_t words = format::words_per_row(bits);
    std::vector<uint16_t> packed(words, 0);
    packed[0] = scale;
    for (int stream_bit = 0; stream_bit < format::kRowDim * bits; ++stream_bit) {
        const int bit = ((stream_bit * 17 + row * 13) ^ (stream_bit >> 2)) & 1;
        packed[1 + stream_bit / 16] |= static_cast<uint16_t>(bit << (stream_bit & 15));
    }
    return packed;
}

void run_k(int bits) {
    const size_t words = format::words_per_row(bits);
    constexpr size_t stride_padding = 3;
    const size_t stride = words + stride_padding;
    std::vector<uint16_t> packed(static_cast<size_t>(kRows) * stride, 0x7bad);
    for (int row = 0; row < kRows; ++row) {
        const std::vector<uint16_t> source = make_row(bits, row, static_cast<uint16_t>(0x3c00 + row * 0x100));
        std::memcpy(packed.data() + static_cast<size_t>(row) * stride, source.data(), words * sizeof(uint16_t));
    }
    std::array<uint16_t, kHeads * format::kRowDim> head_bias{};
    for (int i = 0; i < format::kRowDim; ++i) head_bias[format::kRowDim + i] = 0x3c00;  // +1.0
    format::View view{packed.data(), kRows, static_cast<int64_t>(stride), bits, head_bias.data(), kHeads};
    const int64_t row_ids[] = {2, 3, 0, 4};
    const int32_t head_ids[] = {0, 1, 0, 1};
    constexpr size_t output_stride = format::kRowDim + 7;
    std::vector<float> output(4 * output_stride, -77.0f);
    const auto status = ngram::decode_rows(view, row_ids, head_ids, 4, output.data(), output_stride);
    check(status == ngram::Status::ok, "decode K");
    for (int requested = 0; requested < 4; ++requested) {
        const int row = static_cast<int>(row_ids[requested]);
        const uint16_t* source = packed.data() + static_cast<size_t>(row) * stride;
        const float scale = strata::kernels::f32_from_f16(source[0]);
        const float bias = head_ids[requested] == 0 ? 0.0f : 1.0f;
        for (int i = 0; i < format::kRowDim; ++i) {
            volatile float scaled = strata::kernels::f32_from_f16(expected_code(state_from_row(source, bits, i))) * scale;
            float expected = scaled + bias;
            uint32_t got_bits, expected_bits;
            std::memcpy(&got_bits, &output[static_cast<size_t>(requested) * output_stride + i], sizeof(got_bits));
            std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
            if (got_bits != expected_bits) {
                ++failures;
                break;
            }
        }
        check(output[static_cast<size_t>(requested) * output_stride + format::kRowDim] == -77.0f,
              "padding untouched");
    }
}

}  // namespace

int main() {
    std::cout << "exl3_ngram_parity\n";
    for (int bits = 1; bits <= 8; ++bits) run_k(bits);
    format::View invalid{nullptr, 1, 1, 9, nullptr, 1};
    check(ngram::validate(invalid) == ngram::Status::unsupported_bits, "bits 9 rejected");
    format::View too_short{nullptr, 1, 1, 4, nullptr, 1};
    check(ngram::validate(too_short) == ngram::Status::unsupported_layout, "short row stride rejected");
    const size_t ptrdiff_u16 = static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(uint16_t);
    format::View row_overflow{reinterpret_cast<const uint16_t*>(1), 2, static_cast<int64_t>(ptrdiff_u16), 4,
                              reinterpret_cast<const uint16_t*>(1), 1};
    check(ngram::validate(row_overflow) == ngram::Status::size_overflow, "row pointer span overflow rejected");
    format::View head_overflow{reinterpret_cast<const uint16_t*>(1), 1, format::words_per_row(4), 4,
                               reinterpret_cast<const uint16_t*>(1),
                               static_cast<int64_t>(ptrdiff_u16 / format::kRowDim + 1)};
    check(ngram::validate(head_overflow) == ngram::Status::size_overflow, "head bias span overflow rejected");
    format::View output_view{reinterpret_cast<const uint16_t*>(1), 1, format::words_per_row(4), 4,
                             reinterpret_cast<const uint16_t*>(1), 1};
    const size_t ptrdiff_f32 = static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
    check(ngram::decode_rows(output_view, nullptr, nullptr, 2, reinterpret_cast<float*>(1), ptrdiff_f32) ==
              ngram::Status::size_overflow,
          "output pointer span overflow rejected");
    const uint16_t packed[] = {0x3c00, 0};
    const uint16_t bias[] = {0};
    format::View prescan_view{packed, 1, format::words_per_row(1), 1, bias, 1};
    const int64_t bad_rows[] = {0, 1};
    float untouched[format::kRowDim * 2];
    for (float& value : untouched) value = -9.0f;
    check(ngram::decode_rows(prescan_view, bad_rows, nullptr, 2, untouched, format::kRowDim) ==
              ngram::Status::invalid_argument && untouched[0] == -9.0f,
          "invalid requests are prescanned before output");
    std::cout << (failures == 0 ? "exl3_ngram_parity: all passed\n" : "exl3_ngram_parity: failures\n");
    return failures == 0 ? 0 : 1;
}
