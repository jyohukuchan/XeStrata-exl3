// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Focused exact-format bucket and legacy ExpertCache regression test.
#include "strata/core/expert_cache.hpp"
#include "strata/core/runtime.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool check(bool value, const char* what) {
    if (!value) std::fprintf(stderr, "FAIL: %s\n", what);
    return value;
}

std::vector<uint8_t> bytes(size_t n, uint8_t seed) {
    std::vector<uint8_t> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = (uint8_t) (seed + 37 * i + (i >> 8));
    return out;
}

}  // namespace

int main() {
    try {
        strata::core::Runtime::get();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "skipped: %s\n", e.what());
        return 77;
    }

    int failures = 0;
    std::string error;
    strata::core::ExpertCache cache;
    const std::vector<strata::core::ExpertCacheBucket> buckets{{2, 2, 1001}, {5, 1, 2001}};
    failures += !check(cache.open_buckets(buckets, 1, 4, error), error.c_str());
    const int32_t s0 = cache.admit(0, 0, 2), s1 = cache.admit(0, 1, 5), s2 = cache.admit(0, 2, 2);
    failures += !check(s0 == 0 && s1 == 2 && s2 == 1, "exact bucket slot assignment");
    failures += !check(cache.slot_format(s0) == 2 && cache.slot_format(s1) == 5 && cache.slot_format(s2) == 2,
                        "slot format identities");
    failures += !check(cache.resident() == 3, "bucket resident count");
    failures += !check(cache.admit(0, 0, 5) == strata::core::kNotResident, "format mismatch cannot reuse resident slot");
    failures += !check(cache.admit(0, 3, 2) == strata::core::kNotResident, "full exact bucket refuses admission");

    const auto b2 = bytes(1001, 0x21), b5 = bytes(2001, 0x71);
    failures += !check(cache.fill_slot_blocking(s0, b2.data(), error, (int64_t) b2.size()), error.c_str());
    failures += !check(cache.fill_slot_blocking(s1, b5.data(), error, (int64_t) b5.size()), error.c_str());
    failures += !check(cache.verify_slot(s0, b2.data(), error, (int64_t) b2.size()), error.c_str());
    failures += !check(cache.verify_slot(s1, b5.data(), error, (int64_t) b5.size()), error.c_str());
    auto wrong = b2;
    wrong[17] ^= 0x5A;
    failures += !check(!cache.verify_slot(s0, wrong.data(), error, (int64_t) wrong.size()),
                        "wrong-format/wrong-bytes verification rejects");
    failures += !check(!cache.fill_slot(s0, b5.data(), nullptr, error, (int64_t) b5.size()),
                        "oversized bucket fill rejects");
    cache.close();

    strata::core::ExpertCache legacy;
    failures += !check(legacy.open(2, 1, 2, 1024, error), error.c_str());
    const int32_t legacy_slot = legacy.admit(0, 0);
    failures += !check(legacy_slot == 0 && legacy.slot_format(legacy_slot) == 0, "legacy open regression");
    const auto legacy_blob = bytes(1024, 0x33);
    failures += !check(legacy.fill_slot_blocking(legacy_slot, legacy_blob.data(), error), error.c_str());
    failures += !check(legacy.verify_slot(legacy_slot, legacy_blob.data(), error), error.c_str());
    legacy.close();

    std::printf("expert_cache_buckets_test: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
