// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 n-gram fixture layout follows exllamav3's MIT-licensed specialized codec.
#include "strata/kernels/exl3_ngram.hpp"
#include "strata/kernels/cpu/exl3_ngram.hpp"

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

struct FixtureCase {
    uint32_t bits = 0, rows = 0, stride = 0, heads = 0;
    std::vector<int32_t> head_ids;
    std::vector<uint16_t> packed;
    std::vector<uint16_t> bias;
    std::vector<float> gold;
};

bool read_bytes(const std::vector<uint8_t>& data, size_t& at, void* dst, size_t bytes) {
    if (at > data.size() || bytes > data.size() - at) return false;
    std::memcpy(dst, data.data() + at, bytes);
    at += bytes;
    return true;
}

bool read_u32(const std::vector<uint8_t>& data, size_t& at, uint32_t& value) {
    uint8_t raw[4];
    if (!read_bytes(data, at, raw, sizeof(raw))) return false;
    value = (uint32_t) raw[0] | ((uint32_t) raw[1] << 8) | ((uint32_t) raw[2] << 16) | ((uint32_t) raw[3] << 24);
    return true;
}

template<class T> bool read_vector(const std::vector<uint8_t>& data, size_t& at, std::vector<T>& out, size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) return false;
    out.resize(count);
    return read_bytes(data, at, out.data(), count * sizeof(T));
}

bool load_fixture(const std::string& path, std::vector<FixtureCase>& cases, std::string& error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "cannot open fixture: " + path;
        return false;
    }
    const std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    size_t at = 0;
    uint8_t magic[8];
    uint32_t version = 0, count = 0;
    if (!read_bytes(data, at, magic, sizeof(magic)) || std::memcmp(magic, "EXL3NG01", 8) != 0 ||
        !read_u32(data, at, version) || !read_u32(data, at, count) || version != 1 || count == 0) {
        error = "invalid n-gram fixture header";
        return false;
    }
    cases.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
        FixtureCase c;
        if (!read_u32(data, at, c.bits) || !read_u32(data, at, c.rows) || !read_u32(data, at, c.stride) ||
            !read_u32(data, at, c.heads) || c.bits < 1 || c.bits > 8 || c.rows == 0 || c.heads == 0 ||
            c.stride < 1 + 10 * c.bits) {
            error = "invalid metadata at case " + std::to_string(index);
            return false;
        }
        if (!read_vector(data, at, c.head_ids, c.rows) || !read_vector(data, at, c.packed, (size_t) c.rows * c.stride) ||
            !read_vector(data, at, c.bias, (size_t) c.heads * 160) || !read_vector(data, at, c.gold, (size_t) c.rows * 160)) {
            error = "truncated data at case " + std::to_string(index);
            return false;
        }
        cases.push_back(std::move(c));
    }
    if (at != data.size()) {
        error = "fixture has trailing data";
        return false;
    }
    return true;
}

bool close(float got, float want) {
    if (!std::isfinite(got) || !std::isfinite(want)) return got == want;
    return std::fabs(got - want) <= 1e-6f * std::max(1.0f, std::fabs(want));
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
            std::fprintf(stderr, "usage: exl3_ngram_parity [--selftest] [--fixture PATH]\n");
            return 2;
        }
    }
    if (fixture_path.empty()) {
        std::fprintf(stderr, "usage: exl3_ngram_parity [--selftest] [--fixture PATH]\n");
        return 2;
    }

    std::vector<FixtureCase> cases;
    std::string error;
    if (!load_fixture(fixture_path, cases, error)) {
        std::fprintf(stderr, "exl3_ngram_parity: %s\n", error.c_str());
        return 2;
    }
    auto& queue = strata::core::Runtime::get().compute();
    int failures = 0;
    for (size_t index = 0; index < cases.size(); ++index) {
        const FixtureCase& c = cases[index];
        strata::parity::Device device;
        uint16_t* d_packed = device.upload(c.packed);
        int32_t* d_heads = device.upload(c.head_ids);
        uint16_t* d_bias = device.upload(c.bias);
        std::vector<float> initial((size_t) c.rows * 167, -1234.5f);
        float* d_out = device.upload(initial);
        strata::kernels::exl3_ngram::View view;
        view.packed_rows = d_packed;
        view.head_bias = d_bias;
        view.row_count = c.rows;
        view.words_per_row_stride = c.stride;
        view.num_heads = c.heads;
        view.bits = c.bits;
        sycl::event ready = queue.ext_oneapi_submit_barrier();
        sycl::event event =
            strata::kernels::exl3_ngram::exl3_ngram_dequant(queue, view, d_heads, d_out, 167, {ready});
        event.wait_and_throw();
        std::vector<float> got(initial.size());
        strata::parity::copy_out(got.data(), d_out, got.size());
        size_t bad_values = 0, bad_padding = 0;
        for (uint32_t row = 0; row < c.rows; ++row) {
            for (int i = 0; i < 160; ++i)
                if (!close(got[(size_t) row * 167 + i], c.gold[(size_t) row * 160 + i])) ++bad_values;
            for (int i = 160; i < 167; ++i)
                if (got[(size_t) row * 167 + i] != -1234.5f) ++bad_padding;
        }
        strata::kernels::exl3_ngram::View host_view;
        host_view.packed_rows = c.packed.data();
        host_view.row_count = c.rows;
        host_view.words_per_row_stride = c.stride;
        host_view.bits = (int) c.bits;
        host_view.head_bias = c.bias.data();
        host_view.num_heads = c.heads;
        std::vector<float> cpu_got(initial.size(), -1234.5f);
        const auto cpu_status = strata::kernels::cpu::exl3_ngram::decode_rows(
            host_view, nullptr, c.head_ids.data(), c.rows, cpu_got.data(), 167);
        size_t bad_cpu = cpu_status == strata::kernels::cpu::exl3_ngram::Status::ok ? 0 : 1;
        for (uint32_t row = 0; row < c.rows; ++row)
            for (int i = 0; i < 160; ++i)
                if (!close(cpu_got[(size_t) row * 167 + i], c.gold[(size_t) row * 160 + i])) ++bad_cpu;
        for (uint32_t row = 0; row < c.rows; ++row)
            for (int i = 160; i < 167; ++i)
                if (cpu_got[(size_t) row * 167 + i] != -1234.5f) ++bad_cpu;

        std::printf("case %zu: K=%u rows=%u stride=%u heads=%u: GPU values %s (%zu bad), padding %s (%zu bad), CPU %s (%zu bad)\n",
                    index, c.bits, c.rows, c.stride, c.heads, bad_values == 0 ? "OK" : "FAIL", bad_values,
                    bad_padding == 0 ? "OK" : "FAIL", bad_padding, bad_cpu == 0 ? "OK" : "FAIL", bad_cpu);
        if (bad_values != 0 || bad_padding != 0 || bad_cpu != 0) ++failures;

        if (index == 0) {
            for (const int32_t invalid_head : {int32_t(-1), (int32_t) c.heads}) {
                std::vector<int32_t> invalid_ids = c.head_ids;
                invalid_ids[0] = invalid_head;
                std::vector<float> invalid_initial(initial.size(), -1234.5f);
                int32_t* d_invalid_ids = device.upload(invalid_ids);
                float* d_invalid_out = device.upload(invalid_initial);
                sycl::event invalid_ready = queue.ext_oneapi_submit_barrier();
                sycl::event invalid_event = strata::kernels::exl3_ngram::exl3_ngram_dequant(
                    queue, view, d_invalid_ids, d_invalid_out, 167, {invalid_ready});
                invalid_event.wait_and_throw();
                std::vector<float> invalid_got(invalid_initial.size());
                strata::parity::copy_out(invalid_got.data(), d_invalid_out, invalid_got.size());
                size_t invalid_bad = 0;
                for (int i = 0; i < 160; ++i)
                    if (!std::isnan(invalid_got[i])) ++invalid_bad;
                for (uint32_t row = 1; row < c.rows; ++row)
                    for (int i = 0; i < 160; ++i)
                        if (!close(invalid_got[(size_t) row * 167 + i], c.gold[(size_t) row * 160 + i]))
                            ++invalid_bad;
                for (uint32_t row = 0; row < c.rows; ++row)
                    for (int i = 160; i < 167; ++i)
                        if (invalid_got[(size_t) row * 167 + i] != -1234.5f) ++invalid_bad;
                std::printf("  invalid head %d: %s (%zu bad, bias not read)\n", invalid_head,
                            invalid_bad == 0 ? "OK" : "FAIL", invalid_bad);
                if (invalid_bad != 0) ++failures;
            }
        }
    }
    std::printf("exl3_ngram_parity: %d failures\n", failures);
    if (failures != 0) return 1;
    if (selftest) std::printf("exl3_ngram_parity OK\n");
    return 0;
}
