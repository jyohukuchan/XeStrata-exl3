// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/exl3_cb2_bench.cpp - a micro benchmark of the codebook 2 decode: the two byte-sum forms.
//
// It decodes the same states with the same decode, once with the unsigned dot product and once with the
// signed one with the sign bits flipped, and reports the time of both.  It measures the decode itself, not
// the model: the states stay in the cache and every work item decodes many of them, so the time is the
// arithmetic of the decode and not the loading of the weights.
#include "xe/exl3_cb2.hpp"

#include "parity_device.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kMul1 = 0x83DCD12Du;

struct Options {
    size_t elements = 65536;
    int repeats = 256;
    int launches = 5;
    int rounds = 11;
    int warmup_ms = 400;
};

void usage() {
    std::fprintf(stderr,
                 "usage: exl3_cb2_bench [--elements N] [--repeats N] [--launches N] [--rounds N] [--warmup-ms N]\n");
}

bool parse_int(const char* text, long& out) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > 1000000000) return false;
    out = value;
    return true;
}

bool parse_options(int argc, char** argv, Options& out) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        long value = 0;
        if (key == "--elements" && i + 1 < argc && parse_int(argv[++i], value)) {
            out.elements = (size_t) value;
        } else if (key == "--repeats" && i + 1 < argc && parse_int(argv[++i], value)) {
            out.repeats = (int) value;
        } else if (key == "--launches" && i + 1 < argc && parse_int(argv[++i], value)) {
            out.launches = (int) value;
        } else if (key == "--rounds" && i + 1 < argc && parse_int(argv[++i], value)) {
            out.rounds = (int) value;
        } else if (key == "--warmup-ms" && i + 1 < argc && parse_int(argv[++i], value)) {
            out.warmup_ms = (int) value;
        } else {
            usage();
            return false;
        }
    }
    return out.elements > 0 && out.repeats > 0 && out.launches > 0 && out.rounds > 0;
}

// The decode of one state: the mul1 product, its byte sum and the codebook 2 value in the half form.  The
// two forms differ only in the byte sum.
template <bool Signed>
inline float decode_state(uint16_t state) {
    const uint32_t product = (uint32_t) state * kMul1;
    const int byte_sum = Signed ? strata::kernels::exl3::cb2_byte_sum_signed(product)
                                : strata::kernels::exl3::cb2_byte_sum(product);
    return strata::kernels::exl3::cb2_half_f32(byte_sum);
}

template <bool Signed>
void submit_decode(sycl::queue& queue, const uint16_t* states, float* out, size_t count, int repeats) {
    queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
            const uint16_t base = states[id[0]];
            // Four accumulators, so the sum is not a chain of dependent additions.
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (int r = 0; r < repeats; r += 4) {
                a0 += decode_state<Signed>((uint16_t) (base + r));
                a1 += decode_state<Signed>((uint16_t) (base + r + 1));
                a2 += decode_state<Signed>((uint16_t) (base + r + 2));
                a3 += decode_state<Signed>((uint16_t) (base + r + 3));
            }
            out[id[0]] = (a0 + a1) + (a2 + a3);
        });
    });
}

// The time of `launches` kernels, in milliseconds.
template <bool Signed>
double measure(sycl::queue& queue, const uint16_t* states, float* out, size_t count, int repeats, int launches) {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < launches; ++i) submit_decode<Signed>(queue, states, out, count, repeats);
    queue.wait_and_throw();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) return 2;

    auto& queue = strata::core::Runtime::get().compute();
    strata::parity::Device device;
    const sycl::device gpu = queue.get_device();
    std::printf("device: %s\n", gpu.get_info<sycl::info::device::name>().c_str());

    // The states: every 16-bit state, so every decode is one the kernels really do.
    std::vector<uint16_t> states(options.elements);
    for (size_t i = 0; i < states.size(); ++i) states[i] = (uint16_t) (i & 0xFFFFu);
    uint16_t* d_states = device.upload(states);
    float* d_out = device.alloc<float>(options.elements);
    int* d_native = device.alloc<int>(1);
    queue.submit([&](sycl::handler& h) {
        h.single_task([=] { d_native[0] = strata::kernels::exl3::cb2_unsigned_dot_native() ? 1 : 0; });
    }).wait_and_throw();
    std::vector<int> native(1, 0);
    strata::parity::copy_out(native.data(), d_native, 1);
    std::printf("byte sum: %s dot product\n", native[0] ? "unsigned" : "signed (sign bits flipped)");

    // The GPU raises its clock under load, so the warm-up runs until the clock has settled: otherwise the
    // first rounds are measured at the idle clock and the spread is the clock, not the decode.
    const auto warmup_begin = std::chrono::steady_clock::now();
    while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - warmup_begin).count() <
           (double) options.warmup_ms) {
        measure<false>(queue, d_states, d_out, options.elements, options.repeats, options.launches);
        measure<true>(queue, d_states, d_out, options.elements, options.repeats, options.launches);
    }

    // The two forms alternate, and the order changes every round, so a drift of the machine does not
    // favour one of them.
    std::vector<double> unsigned_times, signed_times;
    for (int round = 0; round < options.rounds; ++round) {
        if (round % 2 == 0) {
            unsigned_times.push_back(measure<false>(queue, d_states, d_out, options.elements, options.repeats,
                                                    options.launches));
            signed_times.push_back(measure<true>(queue, d_states, d_out, options.elements, options.repeats,
                                                  options.launches));
        } else {
            signed_times.push_back(measure<true>(queue, d_states, d_out, options.elements, options.repeats,
                                                  options.launches));
            unsigned_times.push_back(measure<false>(queue, d_states, d_out, options.elements, options.repeats,
                                                    options.launches));
        }
    }

    // The two forms must return the same values.
    std::vector<float> first(options.elements), second(options.elements);
    submit_decode<false>(queue, d_states, d_out, options.elements, options.repeats);
    queue.wait_and_throw();
    strata::parity::copy_out(first.data(), d_out, options.elements);
    submit_decode<true>(queue, d_states, d_out, options.elements, options.repeats);
    queue.wait_and_throw();
    strata::parity::copy_out(second.data(), d_out, options.elements);
    size_t different = 0;
    for (size_t i = 0; i < first.size(); ++i)
        if (std::memcmp(&first[i], &second[i], sizeof(float)) != 0) ++different;

    const double decodes = (double) options.elements * (double) options.repeats * (double) options.launches;
    const auto report = [&](const char* name, const std::vector<double>& times) {
        const double best = *std::min_element(times.begin(), times.end());
        const double worst = *std::max_element(times.begin(), times.end());
        const double mid = median(times);
        std::printf("%-9s: median %8.3f ms  min %8.3f ms  max %8.3f ms  spread %5.1f%%  %8.2f Gdecodes/s\n", name,
                    mid, best, worst, mid > 0.0 ? 100.0 * (worst - best) / mid : 0.0,
                    decodes / (mid * 1e6));
    };
    report("unsigned", unsigned_times);
    report("signed", signed_times);
    const double mid_unsigned = median(unsigned_times);
    const double mid_signed = median(signed_times);
    std::printf("signed / unsigned: %.4f  (rounds %d, launches %d, decodes %zu x %d per launch)\n",
                mid_signed / mid_unsigned, options.rounds, options.launches, options.elements, options.repeats);
    std::printf("values: %s (%zu of %zu differ)\n", different == 0 ? "same" : "DIFFER", different, first.size());
    return different == 0 ? 0 : 1;
}
