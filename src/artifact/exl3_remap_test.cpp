// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/artifact/exl3_remap.hpp"
#include "strata/kernels/cpu/exl3.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    namespace artifact = strata::artifact::exl3;
    namespace cpu = strata::kernels::cpu::exl3;
    artifact::MaterializedMatrix source;
    source.k = 256;
    source.n = 384;
    source.bits = 3;
    source.codebook = 2;
    source.trellis.resize((size_t) source.k * (size_t) source.n * source.bits / 16);
    for (size_t i = 0; i < source.trellis.size(); ++i) source.trellis[i] = (uint16_t) (i * 37 + 103);
    source.suh.assign((size_t) source.k, 0x3c00);
    source.svh.assign((size_t) source.n, 0x3c00);
    source.bias.assign((size_t) source.n, 0x3000);
    source.suh[128] = 0x3800;
    source.svh[256] = 0x4000;
    source.bias[256] = 0x3400;
    std::string error;
    artifact::MaterializedMatrix remapped;
    if (!artifact::remap_matrix_blocks(source, {1, 0}, {2, 0}, remapped, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    int failures = 0;
    if (remapped.k != 256 || remapped.n != 256 || remapped.bits != 3 || remapped.codebook != 2) ++failures;
    std::vector<uint16_t> original_hat((size_t) source.k * source.n), mapped_hat(256 * 256);
    if (cpu::reconstruct(source.view(), original_hat.data(), original_hat.size()) != cpu::Status::ok ||
        cpu::reconstruct(remapped.view(), mapped_hat.data(), mapped_hat.size()) != cpu::Status::ok) return 2;
    for (size_t row = 0; row < 256; ++row)
        for (size_t col = 0; col < 256; ++col) {
            const size_t sr = (row + 128) % 256, sc = col < 128 ? col + 256 : col - 128;
            if (mapped_hat[row * 256 + col] != original_hat[sr * 384 + sc]) ++failures;
        }
    std::vector<float> input(256), original_input(256), expected(384), actual(256);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = (float) ((int) (i % 19) - 9) / 128.0f;
        original_input[(i + 128) % 256] = input[i];
    }
    if (cpu::linear(source.view(), original_input.data(), expected.data()) != cpu::Status::ok ||
        cpu::linear(remapped.view(), input.data(), actual.data()) != cpu::Status::ok) return 3;
    for (size_t col = 0; col < actual.size(); ++col) {
        const size_t sc = col < 128 ? col + 256 : col - 128;
        if (std::fabs(actual[col] - expected[sc]) > 1e-5f * std::max(1.0f, std::fabs(expected[sc]))) ++failures;
    }
    artifact::MaterializedMatrix identity;
    if (!artifact::remap_matrix_blocks(source, {}, {}, identity, error) || identity.trellis != source.trellis ||
        identity.suh != source.suh || identity.svh != source.svh || identity.bias != source.bias) ++failures;
    const auto before = remapped.trellis;
    if (artifact::remap_matrix_blocks(source, {0, 0}, {}, remapped, error) || remapped.trellis != before) ++failures;
    std::printf("exl3_remap_test: %d failures\n", failures);
    return failures ? 1 : 0;
}
