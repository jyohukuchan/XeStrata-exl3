// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU EXL3 routed-expert parity against an independent FP64 dense oracle.
#include "strata/kernels/cpu/exl3_expert.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using ExpertView = strata::kernels::exl3::ExpertView;
using MatrixView = strata::kernels::exl3::MatrixView;
using Status = strata::kernels::cpu::exl3::Status;
using ExpertScratch = strata::kernels::cpu::exl3::ExpertScratch;
using strata::kernels::cpu::exl3::expert;

struct FixtureCase {
    uint32_t bits = 0, codebook = 0, k = 0, n = 0;
    std::vector<uint16_t> trellis, suh, svh, bias, wgold;
    std::vector<float> x, ygold;
};

struct MatrixData {
    uint32_t bits = 0, codebook = 0, k = 0, n = 0;
    std::vector<uint16_t> trellis, suh, svh, bias, wgold;
};

bool read_bytes(std::istream& input, void* dst, size_t bytes) {
    return bytes == 0 || input.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
}

bool read_u32(std::istream& input, uint32_t& value) {
    uint8_t b[4];
    if (!read_bytes(input, b, sizeof b)) return false;
    value = (uint32_t) b[0] | ((uint32_t) b[1] << 8) | ((uint32_t) b[2] << 16) | ((uint32_t) b[3] << 24);
    return true;
}

template<class T> bool read_array(std::istream& input, std::vector<T>& values, size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) return false;
    values.resize(count);
    return read_bytes(input, values.data(), count * sizeof(T));
}

bool load_fixture(const std::string& path, std::vector<FixtureCase>& cases, std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { error = "cannot open fixture: " + path; return false; }
    char magic[8];
    uint32_t version = 0, count = 0;
    if (!read_bytes(input, magic, sizeof magic) || std::memcmp(magic, "EXL3FX01", 8) != 0 ||
        !read_u32(input, version) || !read_u32(input, count) || version != 1 || count > 10000) {
        error = "invalid EXL3FX01 header";
        return false;
    }
    cases.clear();
    cases.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
        FixtureCase c;
        if (!read_u32(input, c.bits) || !read_u32(input, c.codebook) || !read_u32(input, c.k) || !read_u32(input, c.n) ||
            c.bits < 1 || c.bits > 8 || c.codebook > 2 || c.k == 0 || c.n == 0 || c.k % 16 != 0 || c.n % 16 != 0) {
            error = "invalid case metadata at " + std::to_string(index);
            return false;
        }
        const size_t tiles = (size_t) (c.k / 16) * (size_t) (c.n / 16);
        const size_t words = tiles * (size_t) (16 * c.bits);
        const size_t values = (size_t) c.k * (size_t) c.n;
        if (!read_array(input, c.trellis, words) || !read_array(input, c.suh, c.k) || !read_array(input, c.svh, c.n) ||
            !read_array(input, c.bias, c.n) || !read_array(input, c.x, c.k) || !read_array(input, c.wgold, values) ||
            !read_array(input, c.ygold, c.n)) {
            error = "truncated case payload at " + std::to_string(index);
            return false;
        }
        cases.push_back(std::move(c));
    }
    return true;
}

MatrixData materialize(const FixtureCase& source, uint32_t k, uint32_t n) {
    MatrixData out;
    out.bits = source.bits;
    out.codebook = source.codebook;
    out.k = k;
    out.n = n;
    const uint32_t source_tr = source.k / 16, source_tc = source.n / 16;
    const uint32_t target_tr = k / 16, target_tc = n / 16;
    const size_t words = (size_t) 16 * source.bits;
    out.trellis.resize((size_t) target_tr * target_tc * words);
    out.wgold.resize((size_t) k * n);
    for (uint32_t tr = 0; tr < target_tr; ++tr) {
        for (uint32_t tc = 0; tc < target_tc; ++tc) {
            const uint32_t sr = tr % source_tr, sc = tc % source_tc;
            const size_t src_tile = ((size_t) sr * source_tc + sc) * words;
            const size_t dst_tile = ((size_t) tr * target_tc + tc) * words;
            std::copy_n(source.trellis.data() + src_tile, words, out.trellis.data() + dst_tile);
            for (uint32_t r = 0; r < 16; ++r)
                for (uint32_t c = 0; c < 16; ++c)
                    out.wgold[((size_t) tr * 16 + r) * n + (size_t) tc * 16 + c] =
                        source.wgold[((size_t) sr * 16 + r) * source.n + (size_t) sc * 16 + c];
        }
    }
    out.suh.resize(k);
    out.svh.resize(n);
    out.bias.resize(n);
    for (uint32_t i = 0; i < k; ++i) out.suh[i] = source.suh[i % source.k];
    for (uint32_t j = 0; j < n; ++j) {
        out.svh[j] = source.svh[j % source.n];
        out.bias[j] = source.bias[j % source.n];
    }
    return out;
}

MatrixView view(const MatrixData& matrix) {
    return {matrix.trellis.data(), matrix.suh.data(), matrix.svh.data(), matrix.bias.data(), (int64_t) matrix.k,
            (int64_t) matrix.n, (int) matrix.bits, (int) matrix.codebook};
}

float ref_f16_to_f32(uint16_t value) {
    const uint32_t sign = (uint32_t) (value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x3ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) bits = sign;
        else {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400u) == 0) { mantissa <<= 1; --exponent; }
            mantissa &= 0x3ffu;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof result);
    return result;
}

void ref_hadamard(float* values) {
    for (int width = 1; width < 128; width *= 2)
        for (int base = 0; base < 128; base += width * 2)
            for (int i = 0; i < width; ++i) {
                const float left = values[base + i], right = values[base + width + i];
                values[base + i] = left + right;
                values[base + width + i] = left - right;
            }
    for (int i = 0; i < 128; ++i) values[i] *= 0.088388347648f;
}

void linear_oracle(const MatrixData& matrix, const float* x, int64_t ldx, float* y, int64_t ldy, int64_t rows) {
    std::vector<float> input(matrix.k), dot(matrix.n);
    for (int64_t row = 0; row < rows; ++row) {
        for (uint32_t base = 0; base < matrix.k; base += 128) {
            for (uint32_t i = 0; i < 128; ++i)
                input[base + i] = x[(size_t) row * (size_t) ldx + base + i] * ref_f16_to_f32(matrix.suh[base + i]);
            ref_hadamard(input.data() + base);
        }
        for (uint32_t col = 0; col < matrix.n; ++col) {
            double sum = 0.0;
            for (uint32_t i = 0; i < matrix.k; ++i)
                sum += (double) ref_f16_to_f32(matrix.wgold[(size_t) i * matrix.n + col]) * input[i];
            dot[col] = (float) sum;
        }
        for (uint32_t base = 0; base < matrix.n; base += 128) {
            ref_hadamard(dot.data() + base);
            for (uint32_t i = 0; i < 128; ++i)
                y[(size_t) row * (size_t) ldy + base + i] =
                    dot[base + i] * ref_f16_to_f32(matrix.svh[base + i]) + ref_f16_to_f32(matrix.bias[base + i]);
        }
    }
}

bool close(float got, float expected) {
    return std::isfinite(got) && std::isfinite(expected) &&
           std::fabs(got - expected) <= 5e-4f * std::max(1.0f, std::fabs(expected));
}

const FixtureCase* find_case(const std::vector<FixtureCase>& cases, uint32_t bits, uint32_t codebook, uint32_t k,
                             uint32_t n) {
    for (const FixtureCase& c : cases)
        if (c.bits == bits && c.codebook == codebook && c.k == k && c.n == n) return &c;
    return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    std::string path = "exl3_fixture.bin";
    if (argc == 2 && std::string(argv[1]) == "--selftest") selftest = true;
    else if (argc == 2) path = argv[1];
    else if (argc == 3 && std::string(argv[1]) == "--selftest") { selftest = true; path = argv[2]; }
    else if (argc > 1) {
        std::cerr << "usage: exl3_expert_cpu_parity [--selftest] [EXL3FX01 fixture]\n";
        return 2;
    }

    std::vector<FixtureCase> cases;
    std::string error;
    if (!load_fixture(path, cases, error)) { std::cerr << "exl3_expert_cpu_parity: " << error << '\n'; return 2; }
    const FixtureCase* gate_source = find_case(cases, 2, 0, 256, 128);
    const FixtureCase* up_source = find_case(cases, 3, 1, 128, 256);
    const FixtureCase* down_source = find_case(cases, 5, 2, 128, 256);
    if (gate_source == nullptr || up_source == nullptr || down_source == nullptr) {
        std::cerr << "exl3_expert_cpu_parity: fixture lacks mixed 2/3/5-bit cases\n";
        return 2;
    }

    const MatrixData gate = materialize(*gate_source, 256, 128);
    const MatrixData up = materialize(*up_source, 256, 128);
    const MatrixData down = materialize(*down_source, 128, 256);
    const ExpertView expert_view{view(gate), view(up), view(down)};
    const int64_t hidden = 256, ff = 128;
    int failures = 0;
    for (const int64_t rows : {int64_t(1), int64_t(4)}) {
        const int64_t ldx = hidden + 5, ldy = hidden + 7;
        const float padding = 24680.5f;
        std::vector<float> x((size_t) rows * (size_t) ldx, -12345.25f);
        for (int64_t row = 0; row < rows; ++row)
            for (int64_t col = 0; col < hidden; ++col)
                x[(size_t) row * (size_t) ldx + col] = 0.01f * (float) ((col * 13 + row * 7) % 97 - 48);
        std::vector<float> gate_values((size_t) rows * ff), up_values((size_t) rows * ff);
        linear_oracle(gate, x.data(), ldx, gate_values.data(), ff, rows);
        linear_oracle(up, x.data(), ldx, up_values.data(), ff, rows);
        for (size_t i = 0; i < gate_values.size(); ++i) {
            const float gate_value = gate_values[i];
            gate_values[i] = (gate_value / (1.0f + std::exp(-gate_value))) * up_values[i];
        }
        std::vector<float> expected((size_t) rows * (size_t) ldy, padding);
        linear_oracle(down, gate_values.data(), ff, expected.data(), ldy, rows);
        std::vector<float> actual((size_t) rows * (size_t) ldy, padding);
        std::vector<float> input_scratch(hidden);
        std::array<uint16_t, 256> tile{};
        std::vector<float> gate_scratch((size_t) rows * ff), up_scratch((size_t) rows * ff);
        ExpertScratch scratch{{input_scratch.data(), input_scratch.size(), tile.data(), tile.size()}, gate_scratch.data(),
                              gate_scratch.size(), up_scratch.data(), up_scratch.size()};
        const Status status = expert(expert_view, x.data(), ldx, actual.data(), ldy, rows, scratch);
        bool ok = status == Status::ok;
        for (int64_t row = 0; ok && row < rows; ++row) {
            for (int64_t col = 0; col < hidden; ++col)
                ok = close(actual[(size_t) row * (size_t) ldy + col], expected[(size_t) row * (size_t) ldy + col]);
            for (int64_t col = hidden; ok && col < ldy; ++col)
                ok = actual[(size_t) row * (size_t) ldy + col] == padding;
        }
        if (!ok) ++failures;
        std::cout << "rows=" << rows << " mixed 2/cb0 3/cb1 5/cb2: " << (ok ? "OK" : "FAIL") << '\n';
    }

    std::vector<float> bad_output(hidden, 7.0f);
    ExpertView bad_shape = expert_view;
    bad_shape.down.n = 128;
    if (expert(bad_shape, bad_output.data(), hidden, bad_output.data(), hidden, 1) != Status::unsupported_layout)
        ++failures;
    std::vector<float> short_input(hidden - 1), short_gate(ff), short_up(ff);
    std::array<uint16_t, 256> short_tile{};
    ExpertScratch short_scratch{{short_input.data(), short_input.size(), short_tile.data(), short_tile.size()},
                                short_gate.data(), short_gate.size(), short_up.data(), short_up.size()};
    if (expert(expert_view, bad_output.data(), hidden, bad_output.data(), hidden, 1, short_scratch) !=
        Status::scratch_too_small)
        ++failures;

    // Focused wrapper case: ff > hidden exercises the down projection's
    // larger input workspace, which used to be accidentally sized only for
    // the hidden width.
    const MatrixData wide_gate = materialize(*gate_source, 128, 256);
    const MatrixData wide_up = materialize(*up_source, 128, 256);
    const MatrixData wide_down = materialize(*down_source, 256, 128);
    const ExpertView wide_view{view(wide_gate), view(wide_up), view(wide_down)};
    std::vector<float> wide_x(128), wide_expected(128), wide_gate_values(256), wide_up_values(256);
    // Zero input keeps this shape-focused case finite even for the fixture's
    // deliberately wide random codebook values; biases still exercise every
    // projection and the down workspace remains ff-sized.
    std::fill(wide_x.begin(), wide_x.end(), 0.0f);
    linear_oracle(wide_gate, wide_x.data(), 128, wide_gate_values.data(), 256, 1);
    linear_oracle(wide_up, wide_x.data(), 128, wide_up_values.data(), 256, 1);
    for (size_t i = 0; i < wide_gate_values.size(); ++i) {
        const float gate_value = wide_gate_values[i];
        wide_gate_values[i] = (gate_value / (1.0f + std::exp(-gate_value))) * wide_up_values[i];
    }
    linear_oracle(wide_down, wide_gate_values.data(), 256, wide_expected.data(), 128, 1);
    std::vector<float> wide_actual(128);
    const Status wide_status = expert(wide_view, wide_x.data(), 128, wide_actual.data(), 128, 1);
    bool wide_ok = wide_status == Status::ok;
    for (size_t i = 0; wide_ok && i < wide_actual.size(); ++i) wide_ok = close(wide_actual[i], wide_expected[i]);
    if (!wide_ok) ++failures;
    float wide_error = 0.0f;
    for (size_t i = 0; i < wide_actual.size(); ++i) wide_error = std::max(wide_error, std::fabs(wide_actual[i] - wide_expected[i]));
    std::cout << "wrapper ff>hidden (128->256->128): " << (wide_ok ? "OK" : "FAIL") << " status="
              << strata::kernels::cpu::exl3::status_string(wide_status) << " maxerr=" << wide_error
              << " got0=" << wide_actual[0] << " want0=" << wide_expected[0] << '\n';

    if (selftest) std::cout << "exl3_expert_cpu_parity selftest\n";
    std::cout << "exl3_expert_cpu_parity: " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
