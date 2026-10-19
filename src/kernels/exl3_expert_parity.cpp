// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 fixture layout follows exllamav3's MIT-licensed packed representation.
// src/kernels/exl3_expert_parity.cpp - packed expert composition parity.
#include "strata/kernels/exl3_expert.hpp"
#include "strata/kernels/f16_bits.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

namespace {

using strata::kernels::exl3::ExpertView;
using strata::kernels::exl3::MatrixView;
using strata::kernels::f32_from_f16;

constexpr float kHadamardScale = 0.08838834764831845f;

struct FixtureCase {
    uint32_t bits = 0, codebook = 0, k = 0, n = 0;
    std::vector<uint16_t> trellis, suh, svh, bias, wgold;
};

struct HostMatrix {
    uint32_t bits = 0, codebook = 0, k = 0, n = 0;
    std::vector<uint16_t> trellis, suh, svh, bias, wgold;
};

bool read_bytes(const std::vector<uint8_t>& bytes, size_t& at, void* dst, size_t count) {
    if (at > bytes.size() || count > bytes.size() - at) return false;
    std::memcpy(dst, bytes.data() + at, count);
    at += count;
    return true;
}

bool read_u32(const std::vector<uint8_t>& bytes, size_t& at, uint32_t& value) {
    uint8_t raw[4];
    if (!read_bytes(bytes, at, raw, sizeof(raw))) return false;
    value = (uint32_t) raw[0] | ((uint32_t) raw[1] << 8) | ((uint32_t) raw[2] << 16) | ((uint32_t) raw[3] << 24);
    return true;
}

template<class T> bool read_vector(const std::vector<uint8_t>& bytes, size_t& at, std::vector<T>& out, size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) return false;
    out.resize(count);
    return read_bytes(bytes, at, out.data(), count * sizeof(T));
}

bool load_fixture(const std::string& path, std::vector<FixtureCase>& cases, std::string& error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "cannot open fixture: " + path;
        return false;
    }
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    size_t at = 0;
    uint8_t magic[8];
    uint32_t version = 0, count = 0;
    if (!read_bytes(bytes, at, magic, sizeof(magic)) || std::memcmp(magic, "EXL3FX01", 8) != 0 ||
        !read_u32(bytes, at, version) || !read_u32(bytes, at, count) || version != 1 || count == 0 || count > 10000) {
        error = "fixture header is invalid";
        return false;
    }
    cases.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        FixtureCase c;
        if (!read_u32(bytes, at, c.bits) || !read_u32(bytes, at, c.codebook) || !read_u32(bytes, at, c.k) ||
            !read_u32(bytes, at, c.n) || c.bits < 1 || c.bits > 8 || c.codebook > 2 || c.k % 16 != 0 ||
            c.n % 16 != 0 || c.k == 0 || c.n == 0) {
            error = "invalid metadata at case " + std::to_string(i);
            return false;
        }
        const size_t tiles = (size_t) (c.k / 16) * (size_t) (c.n / 16);
        const size_t words = tiles * (size_t) (16 * c.bits);
        const size_t matrix = (size_t) c.k * (size_t) c.n;
        if (!read_vector(bytes, at, c.trellis, words) || !read_vector(bytes, at, c.suh, c.k) ||
            !read_vector(bytes, at, c.svh, c.n) || !read_vector(bytes, at, c.bias, c.n)) {
            error = "truncated fixture at case " + std::to_string(i);
            return false;
        }
        std::vector<float> unused_x, unused_y;
        if (!read_vector(bytes, at, unused_x, c.k) || !read_vector(bytes, at, c.wgold, matrix) ||
            !read_vector(bytes, at, unused_y, c.n)) {
            error = "truncated fixture vectors at case " + std::to_string(i);
            return false;
        }
        cases.push_back(std::move(c));
    }
    if (at != bytes.size()) {
        error = "fixture has trailing bytes";
        return false;
    }
    return true;
}

HostMatrix materialize(const FixtureCase& source, uint32_t k, uint32_t n) {
    HostMatrix out;
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

int hadamard_sign(int row, int col) {
    unsigned bits = (unsigned) (row & col);
    int parity = 0;
    while (bits != 0) {
        parity ^= 1;
        bits &= bits - 1;
    }
    return parity == 0 ? 1 : -1;
}

void linear_reference(const HostMatrix& m, const std::vector<float>& x, int64_t ldx, std::vector<float>& y,
                      int64_t ldy, int64_t rows) {
    // Match the fixture's mathematical contract independently: H128 stages use FP32 accumulation, while the dense
    // matrix product uses a FP64 accumulator and rounds to FP32 before the output H128 stage.
    for (int64_t r = 0; r < rows; ++r) {
        std::vector<float> input(m.k);
        for (uint32_t i = 0; i < m.k; ++i) {
            double value = x[(size_t) r * ldx + i];
            const uint32_t block = i & ~uint32_t(127);
            const int local = (int) (i & 127);
            float sum = 0.0f;
            for (int j = 0; j < 128; ++j)
                sum += (float) hadamard_sign(local, j) * x[(size_t) r * ldx + block + (uint32_t) j] *
                       f32_from_f16(m.suh[block + j]);
            value = (double) (sum * kHadamardScale);
            input[i] = (float) value;
        }
        std::vector<float> dot(m.n);
        for (uint32_t j = 0; j < m.n; ++j) {
            double sum = 0.0;
            for (uint32_t i = 0; i < m.k; ++i)
                sum += (double) input[i] * (double) f32_from_f16(m.wgold[(size_t) i * m.n + j]);
            dot[j] = (float) sum;
        }
        for (uint32_t j = 0; j < m.n; ++j) {
            const uint32_t block = j & ~uint32_t(127);
            const int local = (int) (j & 127);
            float sum = 0.0f;
            for (int i = 0; i < 128; ++i)
                sum += (float) hadamard_sign(local, i) * dot[block + (uint32_t) i];
            double value = (double) (sum * kHadamardScale * f32_from_f16(m.svh[j]));
            value += (double) f32_from_f16(m.bias[j]);
            y[(size_t) r * ldy + j] = (float) value;
        }
    }
}

bool close(float got, float want) {
    if (!std::isfinite(got) || !std::isfinite(want)) return got == want;
    return std::fabs(got - want) <= 5e-4f * std::max(1.0f, std::fabs(want));
}

MatrixView device_view(const HostMatrix& host, uint16_t* trellis, uint16_t* suh, uint16_t* svh, uint16_t* bias) {
    MatrixView view;
    view.trellis = trellis;
    view.suh = suh;
    view.svh = svh;
    view.bias = bias;
    view.k = host.k;
    view.n = host.n;
    view.bits = (int) host.bits;
    view.codebook = (int) host.codebook;
    return view;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    std::string fixture_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--selftest")
            selftest = true;
        else if (arg == "--fixture" && i + 1 < argc)
            fixture_path = argv[++i];
        else if (fixture_path.empty() && arg != "--fixture")
            fixture_path = arg;
        else {
            std::fprintf(stderr, "usage: exl3_expert_parity [--selftest] [--fixture PATH]\n");
            return 2;
        }
    }
    if (fixture_path.empty()) {
        std::fprintf(stderr, "usage: exl3_expert_parity [--selftest] [--fixture PATH]\n");
        return 2;
    }

    std::vector<FixtureCase> cases;
    std::string error;
    if (!load_fixture(fixture_path, cases, error)) {
        std::fprintf(stderr, "exl3_expert_parity: %s\n", error.c_str());
        return 2;
    }
    const FixtureCase* gate_source = nullptr;
    const FixtureCase* up_source = nullptr;
    const FixtureCase* down_source = nullptr;
    for (const FixtureCase& c : cases) {
        if (c.bits == 2 && c.codebook == 0 && c.k == 256 && c.n == 128) gate_source = &c;
        if (c.bits == 3 && c.codebook == 1 && c.k == 128 && c.n == 256) up_source = &c;
        if (c.bits == 5 && c.codebook == 2 && c.k == 128 && c.n == 256) down_source = &c;
    }
    if (gate_source == nullptr || up_source == nullptr || down_source == nullptr) {
        std::fprintf(stderr, "exl3_expert_parity: fixture lacks mixed 2/3/5-bit source cases\n");
        return 2;
    }

    // hidden=256, ff=128: gate 2b/cb0, up 3b/cb1, down 5b/cb2.  The up source is tiled into the transposed
    // target shape so the three projections have genuinely independent packed descriptors.
    const HostMatrix gate = materialize(*gate_source, 256, 128);
    const HostMatrix up = materialize(*up_source, 256, 128);
    const HostMatrix down = materialize(*down_source, 128, 256);
    auto& queue = strata::core::Runtime::get().compute();
    strata::parity::Device device;
    ExpertView view;
    view.gate = device_view(gate, device.upload(gate.trellis), device.upload(gate.suh), device.upload(gate.svh),
                            device.upload(gate.bias));
    view.up = device_view(up, device.upload(up.trellis), device.upload(up.suh), device.upload(up.svh),
                          device.upload(up.bias));
    view.down = device_view(down, device.upload(down.trellis), device.upload(down.suh), device.upload(down.svh),
                            device.upload(down.bias));

    int failures = 0;
    for (const int64_t rows : {int64_t(1), int64_t(4)}) {
        const int64_t hidden = 256, ff = 128, ldx = hidden + 5, ldy = hidden + 7;
        const float out_padding = 24680.5f;
        std::vector<float> x((size_t) rows * ldx, -12345.25f);
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t i = 0; i < hidden; ++i)
                x[(size_t) r * ldx + i] = 0.01f * (float) ((i * 13 + r * 7) % 97 - 48);
        std::vector<float> gate_values((size_t) rows * ff), up_values((size_t) rows * ff);
        linear_reference(gate, x, ldx, gate_values, ff, rows);
        linear_reference(up, x, ldx, up_values, ff, rows);
        for (size_t i = 0; i < gate_values.size(); ++i) {
            const float g = gate_values[i];
            gate_values[i] = (g / (1.0f + std::exp(-g))) * up_values[i];
        }
        std::vector<float> expected((size_t) rows * ldy, out_padding);
        linear_reference(down, gate_values, ff, expected, ldy, rows);
        std::vector<float> y((size_t) rows * ldy, out_padding);
        float* d_x = device.upload(x);
        float* d_y = device.upload(y);
        const size_t scratch_elements = strata::kernels::exl3::exl3_expert_scratch_elements(view, rows);
        float* d_scratch = device.alloc<float>(scratch_elements);
        const sycl::event ready = queue.ext_oneapi_submit_barrier();
        sycl::event event = strata::kernels::exl3::exl3_expert_fp32(
            queue, view, d_x, ldx, d_y, ldy, rows, d_scratch, scratch_elements, {}, {ready});
        event.wait_and_throw();
        strata::parity::copy_out(y.data(), d_y, y.size());
        size_t bad_values = 0, bad_padding = 0;
        size_t first_bad = 0;
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t j = 0; j < hidden; ++j)
                if (!close(y[(size_t) r * ldy + j], expected[(size_t) r * ldy + j])) {
                    if (bad_values == 0) first_bad = (size_t) r * (size_t) ldy + j;
                    ++bad_values;
                }
            for (int64_t j = hidden; j < ldy; ++j)
                if (y[(size_t) r * ldy + j] != out_padding) ++bad_padding;
        }
        std::printf("expert mixed 2b/cb0 + 3b/cb1 + 5b/cb2 rows %lld [%lld,%lld] values: %s (%zu bad), padding: %s (%zu bad)\n",
                    (long long) rows, (long long) hidden, (long long) ff,
                    bad_values == 0 ? "OK" : "FAIL", bad_values, bad_padding == 0 ? "OK" : "FAIL", bad_padding);
        if (bad_values != 0)
            std::printf("  first value mismatch at row %zu col %zu: got %.9g expected %.9g\n",
                        first_bad / (size_t) ldy, first_bad % (size_t) ldy, (double) y[first_bad],
                        (double) expected[first_bad]);
        if (bad_values != 0 || bad_padding != 0) ++failures;
    }

    const int64_t malformed_rows = 2;
    float* malformed_x = device.alloc<float>(1);
    float* malformed_y = device.alloc<float>(1);
    const size_t malformed_scratch_size =
        strata::kernels::exl3::exl3_expert_scratch_elements(view, malformed_rows);
    float* malformed_scratch = device.alloc<float>(malformed_scratch_size);
    bool malformed_span_rejected = false;
    try {
        (void) strata::kernels::exl3::exl3_expert_fp32(
            queue, view, malformed_x, std::numeric_limits<int64_t>::max(), malformed_y, view.gate.k,
            malformed_rows, malformed_scratch, malformed_scratch_size);
    } catch (const std::invalid_argument&) {
        malformed_span_rejected = true;
    }
    std::printf("malformed input span rejected before launch: %s\n", malformed_span_rejected ? "yes" : "NO");
    if (!malformed_span_rejected) ++failures;

    std::printf("exl3_expert_parity: %d failures\n", failures);
    if (failures != 0) return 1;
    if (selftest) std::printf("exl3_expert_parity OK\n");
    return 0;
}
