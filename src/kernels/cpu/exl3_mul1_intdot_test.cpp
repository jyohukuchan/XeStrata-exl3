// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The experimental MUL1 SIMD dot against a bounded scalar integer reference.
#include "strata/kernels/cpu/exl3_mul1_intdot.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace exl3 = strata::kernels::cpu::exl3;

int check_bits(int bits) {
    constexpr size_t k = 128, n = 128, tiles = 8;
    std::vector<uint16_t> packed(tiles * tiles * 16 * bits), scale(128, 0x3c00);
    uint32_t seed = static_cast<uint32_t>(bits);
    for (auto& value : packed) {
        seed = seed * 1664525u + 1013904223u;
        value = static_cast<uint16_t>(seed >> 8);
    }
    const strata::kernels::exl3::MatrixView matrix{packed.data(), scale.data(), scale.data(), nullptr,
                                                 k, n, bits, 2};
    std::array<float, k> input{};
    for (size_t i = 0; i < k; ++i) input[i] = static_cast<float>(static_cast<int>((i * 13 + 7) % 127) - 63) / 32.0f;
    std::array<int32_t, k> duplicated{};
    exl3::Mul1Input prepared;
    if (!exl3::prepare_mul1_input(matrix, input.data(), duplicated.data(), duplicated.size(), prepared)) return 1;
    std::array<float, n> actual{};
    if (!exl3::linear_mul1_dot_range(matrix, prepared, 0, tiles, actual.data())) return 2;
    std::array<int64_t, n> reference{};
    std::array<uint16_t, 256> states{};
    for (size_t tk = 0; tk < tiles; ++tk)
        for (size_t tn = 0; tn < tiles; ++tn) {
            const uint16_t* tile = packed.data() + (tk * tiles + tn) * 16 * bits;
            if (exl3::unpack_trellis_states(tile, bits, states.data(), states.size()) != exl3::Status::ok) return 3;
            for (int state = 0; state < 256; ++state) {
                const int element = strata::kernels::exl3::tile_element_for_state(state);
                const uint32_t product = static_cast<uint32_t>(states[state]) * 0x83DCD12Du;
                const int byte_sum = static_cast<int>((product & 255u) + ((product >> 8) & 255u) +
                                                      ((product >> 16) & 255u) + (product >> 24));
                const int x = static_cast<int16_t>(duplicated[tk * 16 + element / 16] & 0xffff);
                reference[tn * 16 + element % 16] += static_cast<int64_t>(byte_sum) * x;
            }
        }
    for (size_t col = 0; col < n; ++col) {
        volatile float factor = (887.0f / 131072.0f) * prepared.q;
        volatile float delta = ((-238.0f / 131072.0f) * prepared.q) * static_cast<float>(prepared.sum);
        volatile float product = static_cast<float>(reference[col] - 510ll * prepared.sum) * factor;
        const float expected = product + delta;
        if (!std::isfinite(actual[col]) || std::memcmp(&expected, &actual[col], sizeof(float)) != 0) return 4;
    }
    input.fill(0.0f);
    if (!exl3::prepare_mul1_input(matrix, input.data(), duplicated.data(), duplicated.size(), prepared) ||
        prepared.q != 1.0f || prepared.sum != 0) return 5;
    input[0] = std::numeric_limits<float>::denorm_min();
    if (exl3::prepare_mul1_input(matrix, input.data(), duplicated.data(), duplicated.size(), prepared)) return 6;
    auto over = matrix;
    over.k = 16640;
    if (exl3::prepare_mul1_input(over, input.data(), duplicated.data(), 16640, prepared)) return 7;
    return 0;
}

int main() {
    if (!exl3::mul1_intdot_enabled()) return 77;
    const int b2 = check_bits(2), b5 = check_bits(5);
    std::printf("exl3_mul1_intdot_test: bits2=%d bits5=%d\n", b2, b5);
    return b2 == 0 && b5 == 0 ? 0 : 1;
}
