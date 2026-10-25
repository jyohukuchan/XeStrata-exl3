// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/shortconv_parity.cpp - the LFM2 short convolution against a host reference written
// from the lfm2moe semantics (in_proj [B|C|X] split, bx = B*X, depthwise causal conv, y = C*out,
// the state carried between windows).  Exact-tolerance FP32: the kernel's per-channel sum order
// is the tap order, the same as the reference's.
#include "strata/kernels/shortconv.hpp"

#include "parity_device.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

void reference(float* state, const float* bcx, const float* w, float* y, int ch, int lc, int rows) {
    std::vector<float> hist((size_t) ch * (size_t) lc);
    for (int c = 0; c < ch; ++c) {
        float h[8];
        for (int j = 0; j < lc - 1; ++j) h[j] = state[(size_t) j * ch + c];
        for (int t = 0; t < rows; ++t) {
            const float* row = bcx + (size_t) t * 3 * ch;
            h[lc - 1] = row[c] * row[2 * (size_t) ch + c];
            float acc = 0.0f;
            for (int j = 0; j < lc; ++j) acc += w[(size_t) c * lc + j] * h[j];
            y[(size_t) t * ch + c] = row[(size_t) ch + c] * acc;
            for (int j = 0; j + 1 < lc; ++j) h[j] = h[j + 1];
        }
        for (int j = 0; j < lc - 1; ++j) state[(size_t) j * ch + c] = h[j];
    }
}

int run_case(int ch, int lc, int rows, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    std::vector<float> bcx((size_t) rows * 3 * ch), w((size_t) ch * lc), st0((size_t) (lc - 1) * ch);
    for (auto& v : bcx) v = dist(rng);
    for (auto& v : w) v = dist(rng);
    for (auto& v : st0) v = dist(rng);
    std::vector<float> want_y((size_t) rows * ch), want_st = st0;
    reference(want_st.data(), bcx.data(), w.data(), want_y.data(), ch, lc, rows);
    // a second window on the carried state
    std::vector<float> bcx2((size_t) rows * 3 * ch);
    for (auto& v : bcx2) v = dist(rng);
    std::vector<float> want_y2((size_t) rows * ch), want_st2 = want_st;
    reference(want_st2.data(), bcx2.data(), w.data(), want_y2.data(), ch, lc, rows);

    strata::parity::Device dev;
    float* d_bcx = dev.upload(bcx);
    float* d_bcx2 = dev.upload(bcx2);
    float* d_w = dev.upload(w);
    float* d_st = dev.upload(st0);
    float* d_y = dev.alloc<float>((size_t) rows * ch);
    void* stream = strata::parity::stream();
    strata::kernels::shortconv_apply(d_st, d_bcx, d_w, d_y, ch, lc, rows, stream);
    std::vector<float> got_y((size_t) rows * ch), got_st((size_t) (lc - 1) * ch);
    strata::parity::copy_out(got_y.data(), d_y, got_y.size());
    strata::parity::copy_out(got_st.data(), d_st, got_st.size());
    strata::kernels::shortconv_apply(d_st, d_bcx2, d_w, d_y, ch, lc, rows, stream);
    std::vector<float> got_y2((size_t) rows * ch), got_st2((size_t) (lc - 1) * ch);
    strata::parity::copy_out(got_y2.data(), d_y, got_y2.size());
    strata::parity::copy_out(got_st2.data(), d_st, got_st2.size());
    auto close = [](const std::vector<float>& a, const std::vector<float>& b) {
        double mx = 0;
        for (size_t i = 0; i < a.size(); ++i) mx = std::max(mx, (double) std::fabs(a[i] - b[i]));
        return mx;
    };
    const double err = std::max({close(got_y, want_y), close(got_st, want_st), close(got_y2, want_y2),
                                 close(got_st2, want_st2)});
    std::printf("shortconv ch=%d L=%d rows=%d max_err=%.3g\n", ch, lc, rows, err);
    return err <= 1e-5 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) != "--selftest") { std::printf("usage: --selftest\n"); return 2; }
    std::mt19937 rng(12345);
    int fails = 0;
    fails += run_case(2048, 3, 1, rng);
    fails += run_case(2048, 3, 8, rng);
    fails += run_case(64, 3, 5, rng);
    fails += run_case(37, 4, 3, rng);
    std::printf(fails ? "shortconv parity: FAIL\n" : "shortconv parity: OK\n");
    return fails ? 1 : 0;
}
