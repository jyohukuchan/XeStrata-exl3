// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 fixture layout follows exllamav3's MIT-licensed packed representation.
// src/kernels/exl3_parity.cpp - device parity for the standalone EXL3 correctness primitives.
#include "strata/kernels/exl3.hpp"
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

using strata::kernels::exl3::MatrixView;

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

bool read_bytes(const std::vector<uint8_t>& bytes, size_t& at, void* dst, size_t count) {
    if (count > bytes.size() - at) return false;
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
    if (!read_bytes(bytes, at, magic, sizeof(magic)) || std::memcmp(magic, "EXL3FX01", 8) != 0) {
        error = "fixture magic is not EXL3FX01";
        return false;
    }
    uint32_t version = 0, count = 0;
    if (!read_u32(bytes, at, version) || !read_u32(bytes, at, count) || version != 1 || count == 0 || count > 10000) {
        error = "fixture header is invalid";
        return false;
    }
    cases.reserve(count);
    for (uint32_t ci = 0; ci < count; ++ci) {
        FixtureCase c;
        if (!read_u32(bytes, at, c.bits) || !read_u32(bytes, at, c.codebook) || !read_u32(bytes, at, c.k) ||
            !read_u32(bytes, at, c.n) || c.bits < 1 || c.bits > 8 || c.codebook > 2 || c.k == 0 || c.n == 0 ||
            c.k % 16 != 0 || c.n % 16 != 0) {
            error = "fixture case metadata is invalid at case " + std::to_string(ci);
            return false;
        }
        const uint64_t tiles = (uint64_t) (c.k / 16) * (uint64_t) (c.n / 16);
        const uint64_t trellis_words = tiles * (uint64_t) (16 * c.bits);
        const uint64_t matrix_elements = (uint64_t) c.k * (uint64_t) c.n;
        if (trellis_words > std::numeric_limits<size_t>::max() || matrix_elements > std::numeric_limits<size_t>::max()) {
            error = "fixture case is too large at case " + std::to_string(ci);
            return false;
        }
        if (!read_vector(bytes, at, c.trellis, (size_t) trellis_words) ||
            !read_vector(bytes, at, c.suh, c.k) || !read_vector(bytes, at, c.svh, c.n) ||
            !read_vector(bytes, at, c.bias, c.n) || !read_vector(bytes, at, c.x, c.k) ||
            !read_vector(bytes, at, c.wgold, (size_t) matrix_elements) || !read_vector(bytes, at, c.ygold, c.n)) {
            error = "fixture is truncated at case " + std::to_string(ci);
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

bool close(float got, float want) {
    if (!std::isfinite(got) || !std::isfinite(want)) return got == want;
    const float tolerance = 5e-5f * std::max(1.0f, std::fabs(want));
    return std::fabs(got - want) <= tolerance;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    std::string fixture_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--selftest") {
            selftest = true;
        } else if (arg == "--fixture" && i + 1 < argc) {
            fixture_path = argv[++i];
        } else if (fixture_path.empty() && arg != "--fixture") {
            fixture_path = arg;
        } else {
            std::fprintf(stderr, "usage: exl3_parity [--selftest] [--fixture PATH]\n");
            return 2;
        }
    }
    if (fixture_path.empty()) {
        std::fprintf(stderr, "usage: exl3_parity [--selftest] [--fixture PATH]\n");
        return 2;
    }

    std::vector<FixtureCase> cases;
    std::string error;
    if (!load_fixture(fixture_path, cases, error)) {
        std::fprintf(stderr, "exl3_parity: %s\n", error.c_str());
        return 2;
    }

    auto& queue = strata::core::Runtime::get().compute();
    int failures = 0;
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const FixtureCase& c = cases[ci];
        strata::parity::Device device;
        uint16_t* d_trellis = device.upload(c.trellis);
        uint16_t* d_suh = device.upload(c.suh);
        uint16_t* d_svh = device.upload(c.svh);
        uint16_t* d_bias = device.upload(c.bias);
        float* d_x = device.upload(c.x);
        uint16_t* d_w = device.alloc<uint16_t>(c.wgold.size());
        float* d_y = device.alloc<float>(c.ygold.size());
        float* d_scratch = device.alloc<float>((size_t) c.k + (size_t) c.n);

        MatrixView view;
        view.trellis = d_trellis;
        view.suh = d_suh;
        view.svh = d_svh;
        view.bias = d_bias;
        view.k = c.k;
        view.n = c.n;
        view.bits = (int) c.bits;
        view.codebook = (int) c.codebook;

        // Use an explicit barrier as a dependency so this fixture exercises the same queue graph contract as the
        // eventual loader (the parity_device uploads have already completed before this point).
        const sycl::event ready = queue.ext_oneapi_submit_barrier();
        sycl::event reconstruct_event = strata::kernels::exl3::exl3_reconstruct(queue, view, d_w, {ready});
        reconstruct_event.wait_and_throw();
        std::vector<uint16_t> got_w(c.wgold.size());
        strata::parity::copy_out(got_w.data(), d_w, got_w.size());
        size_t bad_w = 0;
        for (size_t i = 0; i < got_w.size(); ++i)
            if (got_w[i] != c.wgold[i]) ++bad_w;

        const size_t scratch_elements = strata::kernels::exl3::exl3_gemv_scratch_elements(view);
        const sycl::event gemv_ready = queue.ext_oneapi_submit_barrier();
        sycl::event gemv_event = strata::kernels::exl3::exl3_gemv_fp32(
            queue, view, d_x, d_y, d_scratch, scratch_elements, {}, {gemv_ready});
        gemv_event.wait_and_throw();
        std::vector<float> got_y(c.ygold.size());
        strata::parity::copy_out(got_y.data(), d_y, got_y.size());
        size_t bad_y = 0;
        for (size_t i = 0; i < got_y.size(); ++i)
            if (!close(got_y[i], c.ygold[i])) ++bad_y;

        std::printf("case %2zu: bits %u cb %u [%ux%u] reconstruct %s (%zu bad), gemv %s (%zu bad)\n", ci,
                    c.bits, c.codebook, c.k, c.n, bad_w == 0 ? "OK" : "FAIL", bad_w,
                    bad_y == 0 ? "OK" : "FAIL", bad_y);
        if (bad_w != 0 || bad_y != 0) ++failures;
    }

    std::printf("exl3_parity: %zu cases, %d failures\n", cases.size(), failures);
    if (failures != 0) return 1;
    if (selftest) std::printf("exl3_parity OK\n");
    return 0;
}
