// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 fixture layout follows exllamav3's MIT-licensed packed representation.
// Grouped routed-expert parity against an independent FP64 host oracle.
#include "strata/kernels/exl3_grouped.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/core/gpu.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

namespace {

using strata::kernels::exl3::ExpertBlobLayout;
using strata::kernels::exl3::ExpertComponentLayout;
using strata::kernels::exl3::ExpertProjectionLayout;
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

struct HostExpert {
    ExpertBlobLayout layout;
    std::vector<uint8_t> blob;
    HostMatrix gate, up, down;
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
    if (!file) { error = "cannot open fixture: " + path; return false; }
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
    for (uint32_t ci = 0; ci < count; ++ci) {
        FixtureCase c;
        if (!read_u32(bytes, at, c.bits) || !read_u32(bytes, at, c.codebook) || !read_u32(bytes, at, c.k) ||
            !read_u32(bytes, at, c.n) || c.bits < 1 || c.bits > 8 || c.codebook > 2 || c.k == 0 || c.n == 0 ||
            c.k % 16 != 0 || c.n % 16 != 0) {
            error = "invalid fixture metadata at case " + std::to_string(ci);
            return false;
        }
        const uint64_t tiles = (uint64_t) (c.k / 16) * (uint64_t) (c.n / 16);
        const uint64_t words = tiles * (uint64_t) (16 * c.bits);
        const uint64_t matrix = (uint64_t) c.k * (uint64_t) c.n;
        if (words > std::numeric_limits<size_t>::max() || matrix > std::numeric_limits<size_t>::max() ||
            !read_vector(bytes, at, c.trellis, (size_t) words) || !read_vector(bytes, at, c.suh, c.k) ||
            !read_vector(bytes, at, c.svh, c.n) || !read_vector(bytes, at, c.bias, c.n)) {
            error = "truncated fixture weights at case " + std::to_string(ci);
            return false;
        }
        std::vector<float> unused_x, unused_y;
        if (!read_vector(bytes, at, unused_x, c.k) || !read_vector(bytes, at, c.wgold, (size_t) matrix) ||
            !read_vector(bytes, at, unused_y, c.n)) {
            error = "truncated fixture oracle at case " + std::to_string(ci);
            return false;
        }
        cases.push_back(std::move(c));
    }
    if (at != bytes.size()) { error = "fixture has trailing bytes"; return false; }
    return true;
}

const FixtureCase* find_case(const std::vector<FixtureCase>& cases, uint32_t bits, uint32_t codebook, uint32_t k,
                             uint32_t n) {
    for (const FixtureCase& c : cases)
        if (c.bits == bits && c.codebook == codebook && c.k == k && c.n == n) return &c;
    return nullptr;
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

ExpertComponentLayout append_component(std::vector<uint8_t>& blob, const std::vector<uint16_t>& values) {
    ExpertComponentLayout result;
    result.offset = blob.size();
    result.bytes = values.size() * sizeof(uint16_t);
    const size_t old = blob.size();
    blob.resize(old + (size_t) result.bytes);
    std::memcpy(blob.data() + old, values.data(), (size_t) result.bytes);
    return result;
}

ExpertProjectionLayout append_projection(std::vector<uint8_t>& blob, const HostMatrix& matrix) {
    ExpertProjectionLayout result;
    result.k = matrix.k;
    result.n = matrix.n;
    result.bits = (int) matrix.bits;
    result.codebook = (int) matrix.codebook;
    result.trellis = append_component(blob, matrix.trellis);
    result.suh = append_component(blob, matrix.suh);
    result.svh = append_component(blob, matrix.svh);
    result.bias = append_component(blob, matrix.bias);
    return result;
}

HostExpert make_expert(HostMatrix gate, HostMatrix up, HostMatrix down) {
    HostExpert result;
    result.gate = std::move(gate);
    result.up = std::move(up);
    result.down = std::move(down);
    result.layout.alignment = 2;
    result.layout.gate = append_projection(result.blob, result.gate);
    result.layout.up = append_projection(result.blob, result.up);
    result.layout.down = append_projection(result.blob, result.down);
    result.layout.total_bytes = result.blob.size();
    return result;
}

int hadamard_sign(int row, int col) {
    unsigned bits = (unsigned) (row & col);
    int parity = 0;
    while (bits != 0) { parity ^= 1; bits &= bits - 1; }
    return parity == 0 ? 1 : -1;
}

void linear_reference(const HostMatrix& m, const float* x, int64_t ldx, float* y, int64_t ldy) {
    std::vector<float> input(m.k);
    for (uint32_t i = 0; i < m.k; ++i) {
        const uint32_t block = i & ~uint32_t(127);
        const int local = (int) (i & 127);
        float sum = 0.0f;
        for (int j = 0; j < 128; ++j)
            sum += (float) hadamard_sign(local, j) * x[(size_t) block + (size_t) j] *
                   f32_from_f16(m.suh[(size_t) block + (size_t) j]);
        input[i] = sum * kHadamardScale;
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
        for (int i = 0; i < 128; ++i) sum += (float) hadamard_sign(local, i) * dot[(size_t) block + (size_t) i];
        y[j] = sum * kHadamardScale * f32_from_f16(m.svh[j]) + f32_from_f16(m.bias[j]);
    }
    (void) ldx;
    (void) ldy;
}

void expert_reference(const HostExpert& expert, const float* x, int64_t ldx, float* y, int64_t ldy) {
    std::vector<float> gate(expert.gate.n), up(expert.up.n), activation(expert.gate.n);
    linear_reference(expert.gate, x, ldx, gate.data(), expert.gate.n);
    linear_reference(expert.up, x, ldx, up.data(), expert.up.n);
    for (size_t i = 0; i < activation.size(); ++i)
        activation[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
    linear_reference(expert.down, activation.data(), expert.down.k, y, ldy);
}

bool close(float got, float expected) {
    if (!std::isfinite(got) || !std::isfinite(expected)) return got == expected;
    return std::fabs(got - expected) <= 5e-4f * std::max(1.0f, std::fabs(expected));
}

bool check_rows(const std::vector<float>& got, const std::vector<float>& expected, int64_t ldy, int64_t hidden,
                float sentinel, const std::vector<int32_t>& active, const char* label) {
    size_t bad = 0, padding_bad = 0;
    size_t first_bad = 0;
    for (int32_t row : active) {
        for (int64_t j = 0; j < hidden; ++j)
            if (!close(got[(size_t) row * (size_t) ldy + (size_t) j], expected[(size_t) row * (size_t) ldy + (size_t) j])) {
                if (bad == 0) first_bad = (size_t) row * (size_t) ldy + (size_t) j;
                ++bad;
            }
        for (int64_t j = hidden; j < ldy; ++j)
            if (got[(size_t) row * (size_t) ldy + (size_t) j] != sentinel) ++padding_bad;
    }
    std::printf("%s: values %s (%zu bad), padding %s (%zu bad)\n", label, bad == 0 ? "OK" : "FAIL", bad,
                padding_bad == 0 ? "OK" : "FAIL", padding_bad);
    if (bad != 0)
        std::printf("  first mismatch row=%zu col=%zu got=%.9g expected=%.9g\n", first_bad / (size_t) ldy,
                    first_bad % (size_t) ldy, (double) got[first_bad], (double) expected[first_bad]);
    return bad == 0 && padding_bad == 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    std::string fixture_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--selftest") selftest = true;
        else if (arg == "--fixture" && i + 1 < argc) fixture_path = argv[++i];
        else if (fixture_path.empty() && arg != "--fixture") fixture_path = arg;
        else { std::fprintf(stderr, "usage: exl3_grouped_parity [--selftest] --fixture PATH\n"); return 2; }
    }
    if (fixture_path.empty()) { std::fprintf(stderr, "usage: exl3_grouped_parity [--selftest] --fixture PATH\n"); return 2; }

    std::vector<FixtureCase> cases;
    std::string error;
    if (!load_fixture(fixture_path, cases, error)) {
        std::fprintf(stderr, "exl3_grouped_parity: %s\n", error.c_str());
        return 2;
    }
    const FixtureCase* b2_256_128 = find_case(cases, 2, 0, 256, 128);
    const FixtureCase* b3_128_256 = find_case(cases, 3, 1, 128, 256);
    const FixtureCase* b5_128_256 = find_case(cases, 5, 2, 128, 256);
    if (b2_256_128 == nullptr || b3_128_256 == nullptr || b5_128_256 == nullptr) {
        std::fprintf(stderr, "exl3_grouped_parity: fixture lacks mixed-format source cases\n");
        return 2;
    }

    const HostExpert expert0 = make_expert(materialize(*b2_256_128, 128, 128),
                                           materialize(*b3_128_256, 128, 128),
                                           materialize(*b5_128_256, 128, 128));
    const HostExpert expert1 = make_expert(materialize(*b5_128_256, 128, 256),
                                           materialize(*b2_256_128, 128, 256),
                                           materialize(*b3_128_256, 256, 128));

    auto& queue = strata::core::Runtime::get().compute();
    strata::parity::Device device;
    uint8_t* d_blob0 = device.upload(expert0.blob);
    uint8_t* d_blob1 = device.upload(expert1.blob);
    const std::vector<uint64_t> blob_base{(uint64_t) (uintptr_t) d_blob0, (uint64_t) (uintptr_t) d_blob1,
                                          (uint64_t) (uintptr_t) d_blob0};
    const std::vector<int32_t> format_index{0, 1, 0};
    const std::vector<ExpertBlobLayout> formats{expert0.layout, expert1.layout};
    const std::vector<int32_t> starts{0, 2, 4, 5};
    const std::vector<int32_t> tok{2, 0, 2, 1, 2};
    const std::vector<int32_t> dst{4, 1, 3, 0, 2};
    const int32_t ngroups_value = 3, nentries_value = 5;
    uint64_t* d_blob_base = device.upload(blob_base);
    int32_t* d_format_index = device.upload(format_index);
    ExpertBlobLayout* d_formats = device.upload(formats);
    int32_t* d_starts = device.upload(starts);
    int32_t* d_tok = device.upload(tok);
    int32_t* d_dst = device.upload(dst);
    int32_t* d_ngroups = device.upload(std::vector<int32_t>{ngroups_value});
    int32_t* d_nentries = device.upload(std::vector<int32_t>{nentries_value});
    strata::kernels::exl3::GroupedPlan plan{d_blob_base, d_format_index, d_formats, d_starts, d_tok, d_dst,
                                            d_ngroups, d_nentries, 3, 5, 2, 3, 5, 128, 256};
    const int64_t hidden = 128, ldx = 131, ldy = 133;
    std::vector<float> x((size_t) plan.n_tokens * (size_t) ldx, -777.0f);
    for (int64_t r = 0; r < plan.n_tokens; ++r)
        for (int64_t i = 0; i < hidden; ++i) x[(size_t) r * (size_t) ldx + (size_t) i] =
            0.013f * (float) (((r + 3) * 17 + i * 11) % 101 - 50);
    const float sentinel = 24680.5f;
    std::vector<float> expected((size_t) plan.output_rows * (size_t) ldy, sentinel);
    for (size_t e = 0; e < tok.size(); ++e) {
        const HostExpert& expert = e < 2 || e == 4 ? expert0 : expert1;
        expert_reference(expert, x.data() + (size_t) tok[e] * (size_t) ldx, ldx, expected.data() + (size_t) dst[e] * (size_t) ldy,
                         ldy);
    }
    std::vector<float> host_y(expected.size(), sentinel);
    float* d_x = device.upload(x);
    float* d_y = device.upload(host_y);
    const size_t scratch_elements = strata::kernels::exl3::exl3_grouped_scratch_elements(plan);
    float* d_scratch = device.alloc<float>(scratch_elements);
    const sycl::event ready = queue.ext_oneapi_submit_barrier();
    sycl::event event = strata::kernels::exl3::exl3_grouped_fp32(
        queue, plan, d_x, ldx, d_y, ldy, d_scratch, scratch_elements, {}, {ready});
    event.wait_and_throw();
    strata::parity::copy_out(host_y.data(), d_y, host_y.size());
    int failures = 0;
    const std::vector<int32_t> active{0, 1, 2, 3, 4};
    if (!check_rows(host_y, expected, ldy, hidden, sentinel, active, "mixed formats rows=5")) ++failures;

    // The same graph-compatible capacities are reused after changing only device counts; group 2 / entry 4 remain
    // inactive and their destination row keeps its sentinel.  This exercises reordered/repeated token IDs and dst
    // permutation without rebuilding the kernel graph.
    const int32_t ngroups_small = 2, nentries_small = 4;
    strata::parity::copy_in(d_ngroups, &ngroups_small, 1);
    strata::parity::copy_in(d_nentries, &nentries_small, 1);
    std::fill(host_y.begin(), host_y.end(), sentinel);
    strata::parity::copy_in(d_y, host_y.data(), host_y.size());
    const float scratch_sentinel = -31415.25f;
    std::vector<float> scratch_before(scratch_elements, scratch_sentinel);
    strata::parity::copy_in(d_scratch, scratch_before.data(), scratch_before.size());
    sycl::event replay = strata::kernels::exl3::exl3_grouped_fp32(
        queue, plan, d_x, ldx, d_y, ldy, d_scratch, scratch_elements, {}, {});
    replay.wait_and_throw();
    strata::parity::copy_out(host_y.data(), d_y, host_y.size());
    const std::vector<int32_t> active_small{4, 1, 3, 0};
    if (!check_rows(host_y, expected, ldy, hidden, sentinel, active_small, "dynamic count rows=4")) ++failures;
    for (int64_t j = 0; j < ldy; ++j)
        if (host_y[(size_t) 2 * (size_t) ldy + (size_t) j] != sentinel) ++failures;
    std::vector<float> scratch_after(scratch_elements, 0.0f);
    strata::parity::copy_out(scratch_after.data(), d_scratch, scratch_after.size());
    size_t inactive_scratch_bad = 0;
    const size_t plane = (size_t) std::max<int64_t>(plan.max_ff, plan.hidden);
    for (int p = 0; p < 6; ++p)
        for (int64_t e = nentries_small; e < plan.max_entries; ++e)
            for (size_t j = 0; j < plane; ++j)
                if (scratch_after[(size_t) p * (size_t) plan.max_entries * plane + (size_t) e * plane + j] !=
                    scratch_sentinel)
                    ++inactive_scratch_bad;
    std::printf("inactive scratch sentinel: %s (%zu bad)\n", inactive_scratch_bad == 0 ? "OK" : "FAIL",
                inactive_scratch_bad);
    if (inactive_scratch_bad != 0) ++failures;

    // Capture the same bounded launch sequence on an engine stream.  The graph stores the static plan pointers,
    // while ngroups/nentries remain device scalars and can change between replays without recapturing.
    strata::gpu::Stream capture_stream = strata::gpu::stream_create();
    strata::gpu::Graph* grouped_graph = nullptr;
    bool graph_ok = capture_stream != nullptr;
    if (graph_ok) {
        try {
            auto& capture_queue = strata::core::Runtime::get().stream(capture_stream);
            const int32_t ngroups_full = 3, nentries_full = 5;
            strata::parity::copy_in(d_format_index, format_index.data(), format_index.size());
            strata::parity::copy_in(d_ngroups, &ngroups_full, 1);
            strata::parity::copy_in(d_nentries, &nentries_full, 1);
            std::fill(host_y.begin(), host_y.end(), sentinel);
            strata::parity::copy_in(d_y, host_y.data(), host_y.size());
            std::fill(scratch_before.begin(), scratch_before.end(), scratch_sentinel);
            strata::parity::copy_in(d_scratch, scratch_before.data(), scratch_before.size());
            graph_ok = strata::gpu::begin_capture(capture_stream);
            if (graph_ok) {
                sycl::event captured_event = strata::kernels::exl3::exl3_grouped_fp32(
                    capture_queue, plan, d_x, ldx, d_y, ldy, d_scratch, scratch_elements, {}, {});
                (void) captured_event;
                graph_ok = strata::gpu::end_capture(capture_stream, &grouped_graph);
            }
            graph_ok = graph_ok && grouped_graph != nullptr && strata::gpu::graph_nodes(grouped_graph) != 0;
            graph_ok = graph_ok && strata::gpu::graph_launch(grouped_graph, capture_stream) &&
                       strata::gpu::stream_sync(capture_stream);
            if (graph_ok) {
                strata::parity::copy_out(host_y.data(), d_y, host_y.size());
                graph_ok = check_rows(host_y, expected, ldy, hidden, sentinel, active, "captured graph rows=5");
            }

            // Replay the same executable graph with smaller device counts.  This is the graph-specific dynamic-count
            // check; no grouped API call or graph recapture occurs between these two launches.
            if (graph_ok) {
                const int32_t ngroups_replay = 2, nentries_replay = 4;
                strata::parity::copy_in(d_ngroups, &ngroups_replay, 1);
                strata::parity::copy_in(d_nentries, &nentries_replay, 1);
                std::fill(host_y.begin(), host_y.end(), sentinel);
                strata::parity::copy_in(d_y, host_y.data(), host_y.size());
                std::fill(scratch_before.begin(), scratch_before.end(), scratch_sentinel);
                strata::parity::copy_in(d_scratch, scratch_before.data(), scratch_before.size());
                graph_ok = strata::gpu::graph_launch(grouped_graph, capture_stream) &&
                           strata::gpu::stream_sync(capture_stream);
                if (graph_ok) {
                    strata::parity::copy_out(host_y.data(), d_y, host_y.size());
                    graph_ok = check_rows(host_y, expected, ldy, hidden, sentinel, active_small,
                                          "captured graph replay rows=4");
                    for (int64_t j = 0; j < ldy; ++j)
                        graph_ok = graph_ok && host_y[(size_t) 2 * (size_t) ldy + (size_t) j] == sentinel;
                    strata::parity::copy_out(scratch_after.data(), d_scratch, scratch_after.size());
                    for (int p = 0; p < 6; ++p)
                        for (int64_t e = nentries_replay; e < plan.max_entries; ++e)
                            for (size_t j = 0; j < plane; ++j)
                                graph_ok = graph_ok &&
                                           scratch_after[(size_t) p * (size_t) plan.max_entries * plane +
                                                        (size_t) e * plane + j] == scratch_sentinel;
                    std::printf("captured graph inactive output/scratch: %s\n", graph_ok ? "OK" : "FAIL");
                }
            }
        } catch (const std::exception& capture_error) {
            std::fprintf(stderr, "grouped graph capture exception: %s\n", capture_error.what());
            graph_ok = false;
            if (strata::gpu::is_capturing(capture_stream)) strata::gpu::abandon_capture(capture_stream);
        }
    }
    if (!graph_ok) {
        std::fprintf(stderr, "grouped graph capture/replay: FAIL (%s)\n", strata::gpu::last_error());
        ++failures;
    } else {
        std::printf("grouped graph capture/replay: OK\n");
    }
    if (grouped_graph != nullptr) strata::gpu::graph_destroy(grouped_graph);
    if (capture_stream != nullptr) strata::gpu::stream_destroy(capture_stream);

    // Invalid format indices are guarded on device: the valid route destination becomes NaN and no scratch range is
    // accessed.  Restore the original index for the process's final status only; no graph state is retained here.
    const int32_t bad_format = 9;
    strata::parity::copy_in(d_format_index, &bad_format, 1);
    const int32_t ngroups_bad = 1, nentries_bad = 2;
    strata::parity::copy_in(d_ngroups, &ngroups_bad, 1);
    strata::parity::copy_in(d_nentries, &nentries_bad, 1);
    std::fill(host_y.begin(), host_y.end(), sentinel);
    strata::parity::copy_in(d_y, host_y.data(), host_y.size());
    sycl::event malformed = strata::kernels::exl3::exl3_grouped_fp32(
        queue, plan, d_x, ldx, d_y, ldy, d_scratch, scratch_elements, {}, {});
    malformed.wait_and_throw();
    strata::parity::copy_out(host_y.data(), d_y, host_y.size());
    if (!std::isnan(host_y[(size_t) dst[0] * (size_t) ldy])) {
        std::printf("invalid device format guard: FAIL\n");
        ++failures;
    } else {
        std::printf("invalid device format guard: OK\n");
    }

    // An optional scale span must be absent or complete. A short nonempty
    // span must never be read as a full H128 scale vector.
    const int32_t good_format = 0;
    strata::parity::copy_in(d_format_index, &good_format, 1);
    auto short_scale = formats;
    short_scale[0].gate.suh.bytes = sizeof(uint16_t);
    strata::parity::copy_in(d_formats, short_scale.data(), short_scale.size());
    std::fill(host_y.begin(), host_y.end(), sentinel);
    strata::parity::copy_in(d_y, host_y.data(), host_y.size());
    strata::kernels::exl3::exl3_grouped_fp32(queue, plan, d_x, ldx, d_y, ldy, d_scratch,
                                            scratch_elements).wait_and_throw();
    strata::parity::copy_out(host_y.data(), d_y, host_y.size());
    const bool scale_guard = std::isnan(host_y[(size_t) dst[0] * (size_t) ldy]);
    std::printf("incomplete scale span guard: %s\n", scale_guard ? "OK" : "FAIL");
    if (!scale_guard) ++failures;

    std::printf("exl3_grouped_parity: %d failures\n", failures);
    if (failures != 0) return 1;
    if (selftest) std::printf("exl3_grouped_parity OK\n");
    return 0;
}
