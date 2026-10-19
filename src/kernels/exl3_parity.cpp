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
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using strata::kernels::exl3::MatrixView;
using strata::kernels::f32_from_f16;

constexpr float kHadamardScale = 0.08838834764831845f;

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

int hadamard_sign(int row, int col) {
    unsigned bits = (unsigned) (row & col);
    int parity = 0;
    while (bits != 0) {
        parity ^= 1;
        bits &= bits - 1;
    }
    return parity == 0 ? 1 : -1;
}

// Independent batch oracle: it starts with the fixture's reconstructed Wgold and uses double accumulation for both
// matrix products.  The device test therefore checks the batched packed-weight path against a host calculation,
// rather than checking two copies of the packed decode implementation against each other.
void batch_reference(const FixtureCase& c, const std::vector<float>& x, int64_t ldx, std::vector<float>& y, int64_t ldy,
                     int64_t rows) {
    for (int64_t r = 0; r < rows; ++r) {
        std::vector<float> input((size_t) c.k);
        for (int64_t i = 0; i < c.k; ++i) {
            double value = x[(size_t) r * (size_t) ldx + (size_t) i];
            if (!c.suh.empty()) {
                const int64_t block = i & ~int64_t(127);
                const int local = (int) (i & 127);
                double sum = 0.0;
                for (int j = 0; j < 128; ++j)
                    sum += (double) hadamard_sign(local, j) * x[(size_t) r * (size_t) ldx + (size_t) block + j] *
                           (double) f32_from_f16(c.suh[(size_t) block + j]);
                value = (double) (float) (sum * (double) kHadamardScale);
            }
            input[(size_t) i] = (float) value;
        }
        std::vector<float> dot((size_t) c.n);
        for (uint32_t col = 0; col < c.n; ++col) {
            double sum = 0.0;
            for (uint32_t row = 0; row < c.k; ++row)
                sum += (double) input[row] * (double) f32_from_f16(c.wgold[(size_t) row * c.n + col]);
            dot[col] = (float) sum;
        }
        for (uint32_t j = 0; j < c.n; ++j) {
            double value = dot[j];
            if (!c.svh.empty()) {
                const int local = (int) (j & 127);
                const int64_t block = (int64_t) j & ~int64_t(127);
                double sum = 0.0;
                for (int i = 0; i < 128; ++i) sum += (double) hadamard_sign(local, i) * dot[(size_t) block + i];
                value = (double) (float) (sum * (double) kHadamardScale *
                                          (double) f32_from_f16(c.svh[j]));
            }
            if (!c.bias.empty()) value += (double) f32_from_f16(c.bias[j]);
            y[(size_t) r * (size_t) ldy + j] = (float) value;
        }
    }
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
    {
        uint16_t word = 0;
        float value = 0;
        MatrixView view{&word, &word, &word, nullptr, 128, 128, 3, 2};
        auto rejects = [&](auto operation, const char* label) {
            try {
                operation();
                std::printf("size guard %s: FAIL\n", label);
                ++failures;
            } catch (const std::invalid_argument&) {
                std::printf("size guard %s: OK\n", label);
            }
        };
        MatrixView huge = view;
        huge.k = int64_t(1) << 62;
        huge.n = 256;
        rejects([&] { strata::kernels::exl3::exl3_gemv_scratch_elements(huge); }, "matrix overflow");
        rejects([&] { strata::kernels::exl3::exl3_gemm_scratch_elements(view, int64_t(1) << 62); }, "scratch overflow");
        rejects([&] {
            strata::kernels::exl3::exl3_gemm_fp32(queue, view, &value, int64_t(1) << 62,
                                                &value, 128, 2, nullptr, 0);
        }, "row stride overflow");
    }
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

        for (const int64_t rows : {int64_t(2), int64_t(4)}) {
            const int64_t ldx = (int64_t) c.k + 7;
            const int64_t ldy = (int64_t) c.n + 11;
            const float input_padding = -31337.25f;
            const float output_padding = 27182.75f;
            std::vector<float> batch_x((size_t) rows * (size_t) ldx, input_padding);
            std::vector<float> batch_y((size_t) rows * (size_t) ldy, output_padding);
            for (int64_t r = 0; r < rows; ++r)
                for (uint32_t i = 0; i < c.k; ++i)
                    batch_x[(size_t) r * (size_t) ldx + i] =
                        c.x[i] * (0.75f + 0.125f * (float) r) + 0.0025f * (float) ((int) (i % 9) - 4) * (float) (r + 1);
            std::vector<float> expected = batch_y;
            batch_reference(c, batch_x, ldx, expected, ldy, rows);

            float* d_batch_x = device.upload(batch_x);
            float* d_batch_y = device.upload(batch_y);
            const size_t batch_scratch_elements =
                strata::kernels::exl3::exl3_gemm_scratch_elements(view, rows);
            float* d_batch_scratch = device.alloc<float>(batch_scratch_elements);
            const sycl::event batch_ready = queue.ext_oneapi_submit_barrier();
            sycl::event batch_event = strata::kernels::exl3::exl3_gemm_fp32(
                queue, view, d_batch_x, ldx, d_batch_y, ldy, rows, d_batch_scratch, batch_scratch_elements, {}, {batch_ready});
            batch_event.wait_and_throw();
            std::vector<float> got_batch(batch_y.size());
            strata::parity::copy_out(got_batch.data(), d_batch_y, got_batch.size());
            size_t bad_batch = 0, bad_padding = 0;
            for (int64_t r = 0; r < rows; ++r) {
                for (uint32_t j = 0; j < c.n; ++j)
                    if (!close(got_batch[(size_t) r * (size_t) ldy + j], expected[(size_t) r * (size_t) ldy + j]))
                        ++bad_batch;
                for (int64_t j = c.n; j < ldy; ++j)
                    if (got_batch[(size_t) r * (size_t) ldy + j] != output_padding) ++bad_padding;
            }
            std::printf("case %2zu: batched rows %lld ldx %lld ldy %lld: %s (%zu value, %zu padding bad)\n", ci,
                        (long long) rows, (long long) ldx, (long long) ldy,
                        bad_batch == 0 && bad_padding == 0 ? "OK" : "FAIL", bad_batch, bad_padding);
            if (bad_batch != 0 || bad_padding != 0) ++failures;
        }
    }

    std::printf("exl3_parity: %zu cases, %d failures\n", cases.size(), failures);
    if (failures != 0) return 1;
    if (selftest) std::printf("exl3_parity OK\n");
    return 0;
}
