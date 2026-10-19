// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// CPU routed EXL3 ExpertPool parity: mixed projection widths, expert shapes,
// rows 1/4 and host-drain modes against cpu::exl3::expert.
#include "strata/kernels/cpu/pool.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace exl3 = strata::kernels::cpu::exl3;

namespace {

struct Matrix {
    std::vector<uint16_t> trellis, suh, svh, bias;
    strata::kernels::exl3::MatrixView view{};

    Matrix(int64_t k, int64_t n, int bits, uint16_t seed) {
        trellis.resize(static_cast<size_t>(k / 16) * static_cast<size_t>(n / 16) * static_cast<size_t>(16 * bits));
        for (size_t i = 0; i < trellis.size(); ++i) trellis[i] = static_cast<uint16_t>(seed + i * 37);
        suh.assign(static_cast<size_t>(k), 0x3c00);
        svh.assign(static_cast<size_t>(n), 0x3c00);
        bias.assign(static_cast<size_t>(n), 0);
        view = {trellis.data(), suh.data(), svh.data(), bias.data(), k, n, bits, 0};
    }
};

struct Expert {
    Matrix gate, up, down;
    cpu::ExpertView view;

    Expert(int64_t hidden, int64_t ff, int gate_bits, int up_bits, int down_bits, uint16_t seed)
        : gate(hidden, ff, gate_bits, seed), up(hidden, ff, up_bits, static_cast<uint16_t>(seed + 1000)),
          down(ff, hidden, down_bits, static_cast<uint16_t>(seed + 2000)), view{gate.view, up.view, down.view} {}
};

void fill_input(std::vector<float>& values, int seed) {
    for (size_t i = 0; i < values.size(); ++i) values[i] = 0.1f * static_cast<float>((int(i) + seed * 7) % 19 - 9);
}

}  // namespace

int main() {
    std::printf("exl3_pool_test\n");
    // The two experts deliberately reverse the hidden/intermediate geometry:
    // one has FF > H and the other H > FF. Each projection has its own K.
    Expert wide(128, 256, 2, 3, 5, 11);
    Expert narrow(256, 128, 5, 2, 3, 101);
    Expert* experts[] = {&wide, &narrow};
    const int64_t rows[] = {1, 4};
    constexpr int64_t max_hidden = 256, max_ff = 256, ldx = 261, ldy = 267;
    std::vector<float> inputs[2], expected[2], actual[2];
    cpu::ExpertJobExl3 jobs[2]{};
    for (size_t i = 0; i < 2; ++i) {
        inputs[i].resize(static_cast<size_t>(rows[i]) * static_cast<size_t>(ldx));
        expected[i].assign(static_cast<size_t>(rows[i]) * static_cast<size_t>(ldy), -1234.0f);
        actual[i] = expected[i];
        fill_input(inputs[i], static_cast<int>(i));
        if (exl3::expert(experts[i]->view, inputs[i].data(), ldx, expected[i].data(), ldy, rows[i]) != exl3::Status::ok)
            return 1;
        jobs[i] = {experts[i]->view, rows[i], inputs[i].data(), ldx, actual[i].data(), ldy, static_cast<int>(i)};
    }

    int failures = 0;
    for (bool host_works : {false, true}) {
        for (int workers : {1, 2}) {
            cpu::ExpertPool pool(workers, false, host_works);
            if (pool.prepare_exl3(4, max_hidden, max_ff) != exl3::Status::ok) return 2;
            for (auto& output : actual) std::fill(output.begin(), output.end(), -1234.0f);
            if (pool.run_exl3(jobs, 2) != exl3::Status::ok) return 3;
            for (size_t i = 0; i < 2; ++i)
                if (std::memcmp(expected[i].data(), actual[i].data(), expected[i].size() * sizeof(float)) != 0) ++failures;

            cpu::ExpertJobExl3 invalid = jobs[0];
            invalid.view.gate.bits = 9;
            std::fill(actual[0].begin(), actual[0].end(), -1234.0f);
            if (pool.run_exl3(&invalid, 1) != exl3::Status::unsupported_bits || actual[0][0] != -1234.0f) ++failures;

            cpu::ExpertJobExl3 overflow = jobs[0];
            overflow.rows = 4;
            overflow.ldx = std::numeric_limits<ptrdiff_t>::max() / sizeof(float);
            overflow.ldy = overflow.ldx;
            if (pool.run_exl3(&overflow, 1) != exl3::Status::size_overflow || actual[0][0] != -1234.0f) ++failures;
            if (pool.prepare_exl3(std::numeric_limits<int64_t>::max(), max_hidden, max_ff) != exl3::Status::size_overflow)
                ++failures;
            if (pool.run_exl3(jobs, cpu::ExpertPool::kMaxTasks + 1) != exl3::Status::invalid_argument) ++failures;
        }
    }
    std::printf("  mixed expert K=2/3/5, H/FF 128x256 + 256x128, rows 1/4, host drain on/off: %s\n",
                failures ? "FAIL" : "OK");
    std::printf("exl3_pool_test: %s\n", failures ? "failures" : "all passed");
    return failures ? 1 : 0;
}
