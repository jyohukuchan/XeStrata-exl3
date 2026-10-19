// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU EXL3 parity executable.  It consumes the independent Python-generated
// EXL3FX01 fixture and checks exact W-hat bits plus FP32/FP64 linear results.
#include "strata/kernels/cpu/exl3.hpp"

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

namespace cpu_exl3 = strata::kernels::cpu::exl3;
namespace format = strata::kernels::exl3;

namespace {

struct FixtureCase {
    uint32_t bits = 0;
    uint32_t codebook = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    std::vector<uint16_t> trellis;
    std::vector<uint16_t> suh;
    std::vector<uint16_t> svh;
    std::vector<uint16_t> bias;
    std::vector<float> x;
    std::vector<uint16_t> wgold;
    std::vector<float> ygold;
};

bool read_u16(std::istream& input, uint16_t& value) {
    unsigned char bytes[2];
    if (!input.read(reinterpret_cast<char*>(bytes), sizeof(bytes))) return false;
    value = static_cast<uint16_t>(bytes[0]) | static_cast<uint16_t>(bytes[1] << 8);
    return true;
}

bool read_u32(std::istream& input, uint32_t& value) {
    unsigned char bytes[4];
    if (!input.read(reinterpret_cast<char*>(bytes), sizeof(bytes))) return false;
    value = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
            (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
    return true;
}

bool read_f32(std::istream& input, float& value) {
    uint32_t bits;
    if (!read_u32(input, bits)) return false;
    std::memcpy(&value, &bits, sizeof(value));
    return true;
}

bool read_u16_array(std::istream& input, std::vector<uint16_t>& values, size_t count) {
    values.resize(count);
    for (size_t i = 0; i < count; ++i)
        if (!read_u16(input, values[i])) return false;
    return true;
}

bool read_f32_array(std::istream& input, std::vector<float>& values, size_t count) {
    values.resize(count);
    for (size_t i = 0; i < count; ++i)
        if (!read_f32(input, values[i])) return false;
    return true;
}

bool checked_product(size_t a, size_t b, size_t& result) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) return false;
    result = a * b;
    return true;
}

bool read_fixture(const std::string& path, std::vector<FixtureCase>& cases, std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot open fixture: " + path;
        return false;
    }
    char magic[8];
    if (!input.read(magic, sizeof(magic)) || std::memcmp(magic, "EXL3FX01", 8) != 0) {
        error = "fixture magic is not EXL3FX01";
        return false;
    }
    uint32_t version, count;
    if (!read_u32(input, version) || !read_u32(input, count) || version != 1) {
        error = "fixture version is not 1";
        return false;
    }
    if (count > 10000) {
        error = "fixture case count is unreasonable";
        return false;
    }
    cases.clear();
    cases.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
        FixtureCase c;
        if (!read_u32(input, c.bits) || !read_u32(input, c.codebook) || !read_u32(input, c.k) || !read_u32(input, c.n)) {
            error = "truncated case header at case " + std::to_string(index);
            return false;
        }
        if (c.k == 0 || c.n == 0 || c.k % 16 != 0 || c.n % 16 != 0 || c.bits < 1 || c.bits > 8 || c.codebook > 2) {
            error = "invalid case geometry at case " + std::to_string(index);
            return false;
        }
        size_t tiles, packed_per_tile, trellis_words, matrix_values;
        if (!checked_product(c.k / 16, c.n / 16, tiles) || !checked_product(16, c.bits, packed_per_tile) ||
            !checked_product(tiles, packed_per_tile, trellis_words) || !checked_product(c.k, c.n, matrix_values) ||
            trellis_words > 100000000 || matrix_values > 100000000) {
            error = "case size overflow at case " + std::to_string(index);
            return false;
        }
        if (!read_u16_array(input, c.trellis, trellis_words) || !read_u16_array(input, c.suh, c.k) ||
            !read_u16_array(input, c.svh, c.n) || !read_u16_array(input, c.bias, c.n) ||
            !read_f32_array(input, c.x, c.k) || !read_u16_array(input, c.wgold, matrix_values) ||
            !read_f32_array(input, c.ygold, c.n)) {
            error = "truncated case payload at case " + std::to_string(index);
            return false;
        }
        cases.push_back(std::move(c));
    }
    return true;
}

// Independent half conversion used only by this checker.  Keeping this copy
// separate from exl3.cpp makes the FP64 mathematical oracle useful against a
// wrong production half conversion as well.
float ref_f16_to_f32(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
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
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void ref_hadamard(float* values) {
    for (int width = 1; width < 128; width *= 2) {
        for (int base = 0; base < 128; base += width * 2) {
            for (int i = 0; i < width; ++i) {
                const float left = values[base + i], right = values[base + width + i];
                values[base + i] = left + right;
                values[base + width + i] = left - right;
            }
        }
    }
    for (int i = 0; i < 128; ++i) values[i] *= 0.088388347648f;
}

void ref_y(const FixtureCase& c, std::vector<float>& output) {
    std::vector<float> input(c.k);
    for (uint32_t base = 0; base < c.k; base += 128) {
        for (uint32_t i = 0; i < 128; ++i) input[base + i] = c.x[base + i] * ref_f16_to_f32(c.suh[base + i]);
        ref_hadamard(input.data() + base);
    }
    output.assign(c.n, 0.0f);
    for (uint32_t col = 0; col < c.n; ++col) {
        double acc = 0.0;
        for (uint32_t row = 0; row < c.k; ++row)
            acc += static_cast<double>(ref_f16_to_f32(c.wgold[static_cast<size_t>(row) * c.n + col])) * input[row];
        output[col] = static_cast<float>(acc);
    }
    for (uint32_t base = 0; base < c.n; base += 128) {
        ref_hadamard(output.data() + base);
        for (uint32_t i = 0; i < 128; ++i)
            output[base + i] = output[base + i] * ref_f16_to_f32(c.svh[base + i]) + ref_f16_to_f32(c.bias[base + i]);
    }
}

bool close_enough(float got, float expected) {
    const float difference = std::fabs(got - expected);
    return difference <= 5e-5f + 5e-5f * std::fabs(expected);
}

bool finite_case(const FixtureCase& c) {
    for (float value : c.x) if (!std::isfinite(value)) return false;
    for (float value : c.ygold) if (!std::isfinite(value)) return false;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::cerr << "usage: exl3_parity [EXL3FX01 fixture]\n";
        return 2;
    }
    const std::string path = argc == 2 ? argv[1] : "exl3_fixture.bin";
    std::vector<FixtureCase> cases;
    std::string error;
    if (!read_fixture(path, cases, error)) {
        std::cerr << "exl3_parity: " << error << '\n';
        return 2;
    }

    size_t failures = 0;
    for (size_t index = 0; index < cases.size(); ++index) {
        const FixtureCase& c = cases[index];
        format::MatrixView matrix;
        matrix.trellis = c.trellis.data();
        matrix.suh = c.suh.data();
        matrix.svh = c.svh.data();
        matrix.bias = c.bias.data();
        matrix.k = c.k;
        matrix.n = c.n;
        matrix.bits = static_cast<int>(c.bits);
        matrix.codebook = static_cast<int>(c.codebook);

        std::vector<uint16_t> reconstructed(c.wgold.size());
        const cpu_exl3::Status w_status = cpu_exl3::reconstruct(matrix, reconstructed.data(), reconstructed.size());
        bool w_ok = w_status == cpu_exl3::Status::ok && reconstructed == c.wgold;

        std::vector<float> output(c.n, 0.0f);
        std::vector<float> input_basis(c.k);
        std::array<uint16_t, 256> tile{};
        cpu_exl3::LinearScratch scratch{input_basis.data(), input_basis.size(), tile.data(), tile.size()};
        const cpu_exl3::Status y_status = cpu_exl3::linear(matrix, c.x.data(), output.data(), scratch);
        std::vector<float> independent;
        ref_y(c, independent);
        bool y_ok = y_status == cpu_exl3::Status::ok;
        bool oracle_ok = y_ok;
        float worst_golden = 0.0f, worst_independent = 0.0f;
        if (y_ok) {
            for (size_t i = 0; i < output.size(); ++i) {
                worst_golden = std::max(worst_golden, std::fabs(output[i] - c.ygold[i]));
                worst_independent = std::max(worst_independent, std::fabs(output[i] - independent[i]));
                if (!close_enough(output[i], c.ygold[i]) || !close_enough(output[i], independent[i])) oracle_ok = false;
            }
        }
        if (!w_ok || !oracle_ok || !finite_case(c)) ++failures;
        std::cout << "case " << index << " bits=" << c.bits << " cb=" << c.codebook << " " << c.k << 'x' << c.n
                  << ": W=" << (w_ok ? "OK" : "FAIL") << " y=" << (y_ok ? "OK" : "FAIL")
                  << " oracle=" << (oracle_ok ? "OK" : "FAIL") << " maxerr=" << worst_golden
                  << "/" << worst_independent << '\n';
        if (w_status != cpu_exl3::Status::ok)
            std::cout << "  reconstruct: " << cpu_exl3::status_string(w_status) << '\n';
        if (y_status != cpu_exl3::Status::ok)
            std::cout << "  linear: " << cpu_exl3::status_string(y_status) << '\n';
    }
    std::cout << "exl3_parity: " << failures << " failures / " << cases.size() << " cases\n";
    return failures == 0 ? 0 : 1;
}
