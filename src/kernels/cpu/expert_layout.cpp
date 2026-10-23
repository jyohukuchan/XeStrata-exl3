// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/cpu/expert_layout.cpp - plan v0.3 P6: the per-layer expert table.  See the header.
#include "strata/kernels/cpu/expert_layout.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#include <fstream>
#include <sstream>

namespace strata::kernels::cpu {
namespace {
ExpertLayout g_layout;
}

const ExpertLayout& expert_layout() { return g_layout; }

void expert_layout_set(const ExpertLayout& layout) { g_layout = layout; }

bool cpu_avx512_ok() {
    static const bool ok = [] {
        if (const char* f = std::getenv("STRATA_FORCE_AVX2"); f != nullptr && f[0] == '1') return false;
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        if (!((r[2] >> 27) & 1u)) return false;             // OSXSAVE
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0xE6) != 0xE6) return false;          // the OS saves the AVX-512 state
        cpuid(7, 0);
        const unsigned ebx = r[1], ecx = r[2];
        return ((ebx >> 16) & 1u) && ((ebx >> 30) & 1u) && ((ebx >> 31) & 1u) && ((ecx >> 11) & 1u) && ((ecx >> 1) & 1u);
    }();
    return ok;
}

bool cpu_avx2_ok() {
    static const bool ok = [] {
        unsigned r[4] = {0, 0, 0, 0};
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
            __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
        };
        cpuid(0, 0);
        if (r[0] < 7) return false;
        cpuid(1, 0);
        const unsigned ecx1 = r[2];
        // FMA (12), OSXSAVE (27), AVX (28), F16C (29)
        if (!((ecx1 >> 12) & 1u) || !((ecx1 >> 27) & 1u) || !((ecx1 >> 28) & 1u) || !((ecx1 >> 29) & 1u)) return false;
#if defined(_MSC_VER)
        const unsigned long long xcr0 = _xgetbv(0);
#else
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        const unsigned long long xcr0 = ((unsigned long long) hi << 32) | lo;
#endif
        if ((xcr0 & 0x6) != 0x6) return false;            // the OS saves the SSE and AVX state
        cpuid(7, 0);
        return ((r[1] >> 5) & 1u) != 0;                    // AVX2
    }();
    return ok;
}

bool cpu_avxvnni_ok() {
#if defined(STRATA_HAVE_AVXVNNI)
    static const bool ok = [] {
        if (const char* e = std::getenv("STRATA_NO_AVXVNNI"); e != nullptr && e[0] == '1') return false;
        if (!cpu_avx2_ok()) return false;
        unsigned a = 0, b = 0, c = 0, d = 0;
        auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
            int x[4];
            __cpuidex(x, (int) leaf, (int) sub);
            a = (unsigned) x[0], b = (unsigned) x[1], c = (unsigned) x[2], d = (unsigned) x[3];
#else
            __cpuid_count(leaf, sub, a, b, c, d);
#endif
        };
        cpuid(0, 0);
        if (a < 7) return false;
        cpuid(7, 0);
        if (a < 1) return false;                           // no sub-leaf 1
        cpuid(7, 1);
        return ((a >> 4) & 1u) != 0;                       // AVX-VNNI
    }();
    return ok;
#else
    return false;
#endif
}

std::string cpu_name() {
    unsigned r[12] = {};
#if defined(_MSC_VER)
    int x[4];
    __cpuid(x, (int) 0x80000000u);
    if ((unsigned) x[0] < 0x80000004u) return "unknown";
    for (unsigned i = 0; i < 3; ++i) {
        __cpuid(x, (int) (0x80000002u + i));
        for (int j = 0; j < 4; ++j) r[i * 4 + j] = (unsigned) x[j];
    }
#else
    unsigned a = 0, b = 0, c = 0, d = 0;
    __cpuid(0x80000000u, a, b, c, d);
    if (a < 0x80000004u) return "unknown";
    for (unsigned i = 0; i < 3; ++i) __cpuid(0x80000002u + i, r[(size_t) i * 4], r[(size_t) i * 4 + 1], r[(size_t) i * 4 + 2], r[(size_t) i * 4 + 3]);
#endif
    char s[49] = {};
    std::memcpy(s, r, 48);
    std::string name(s);
    const size_t b0 = name.find_first_not_of(' '), b1 = name.find_last_not_of(' ');
    return b0 == std::string::npos ? std::string("unknown") : name.substr(b0, b1 - b0 + 1);
}

// ---- the AVX-512 probe and the oracle flag (#795) ----
//
// expert.cpp is compiled for AVX-512, and a TU compiled that way may use AVX-512 in ANY of its
// code.  The code that decides whether this CPU has AVX-512 - and the startup flag every CPU, AVX2-only ones
// included, runs before that decision - therefore lives here, in a file compiled for the x86-64 baseline.
namespace {
std::atomic<bool> g_oracle_q8_0{false};
}

void expert_set_oracle_q8_0(bool enabled) { g_oracle_q8_0.store(enabled, std::memory_order_relaxed); }

bool expert_oracle_q8_0_enabled() { return g_oracle_q8_0.load(std::memory_order_relaxed); }

const char* CpuFeatures::reason() const {
    if (usable()) return "ok";
    // Named individually: "AVX-512 not supported" sends a user looking for a new CPU when the machine may have
    // AVX-512F and be missing only VNNI, which is a much narrower and more explicable gap.
    static char buf[160];
    std::snprintf(buf, sizeof buf, "missing %s%s%s%s%s", avx512f ? "" : "AVX512F ",
                  avx512bw ? "" : "AVX512BW ", avx512vl ? "" : "AVX512VL ",
                  avx512_vnni ? "" : "AVX512-VNNI ", avx512_vbmi ? "" : "AVX512-VBMI");
    return buf;
}

CpuFeatures cpu_features() {
    CpuFeatures f;
    int reg[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
    __cpuid(reg, 0);
    if (reg[0] < 7) return f;
    __cpuidex(reg, 7, 0);
#else
    unsigned r[4] = {0, 0, 0, 0};
    __cpuid_count(0, 0, r[0], r[1], r[2], r[3]);
    if (r[0] < 7) return f;
    __cpuid_count(7, 0, r[0], r[1], r[2], r[3]);
    for (int i = 0; i < 4; ++i) reg[i] = (int) r[i];
#endif
    const unsigned ebx = (unsigned) reg[1], ecx = (unsigned) reg[2];
    f.avx512f = (ebx >> 16) & 1u;
    f.avx512bw = (ebx >> 30) & 1u;
    f.avx512vl = (ebx >> 31) & 1u;
    f.avx512_vnni = (ecx >> 11) & 1u;
    f.avx512_vbmi = (ecx >> 1) & 1u;
    return f;
}

void cpu_require_expert_support() {
    const CpuFeatures f = cpu_features();
    if (f.usable()) return;
    std::fprintf(stderr,
                 "strata: this CPU cannot run the expert kernel: %s.\n"
                 "        The engine needs AVX512-VNNI and AVX512-VBMI (Intel Ice Lake / AMD Zen 4 or newer).\n"
                 "        The scalar fallback exists for tests only and is far too slow to decode with.\n",
                 f.reason());
    std::exit(1);
}

void q2_rows_any(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                 int r0, int r1) {
    if (cpu_avx512_ok()) q2_0_gguf_rows_multi(w, row_bytes, nblocks, a, nt, out, r0, r1);
    else q2_0_gguf_rows_multi_avx2(w, row_bytes, nblocks, a, nt, out, r0, r1);
}

void act_quant_any(const float* x, int n, ActQ& a) {
    if (cpu_avx512_ok()) act_quant_q8_1(x, n, a);
    else act_quant_q8_1_avx2(x, n, a);
}


bool expert_layout_load(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    ExpertLayout L;
    L.n_layers = n_layers;
    L.n_expert = n_expert;
    std::ifstream in(pack_dir + "/native_experts.txt");
    if (in) {
        // A pack whose experts are their raw GGUF slices (the native pack form) needs the native (IQ) expert
        // kernels, which this engine no longer has; its blob sizes would be misread as the canonical ones.
        err = "this pack keeps its experts in their GGUF form (native_experts.txt), which this engine does not "
              "read: it computes no native (IQ) experts";
        return false;
    }
    L.total = (uint64_t) n_layers * (uint64_t) n_expert * (uint64_t) BLOB;
    g_layout = L;
    return true;
}

}  // namespace strata::kernels::cpu
