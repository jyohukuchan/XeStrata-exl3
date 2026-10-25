// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
//
// The pieces fallback of ExpertCache::open_sized/open/open_buckets, ported from the upstream
// expert_cache_pieces_test to this tree's exact-format buckets: when the driver refuses the one large
// allocation (the B70's xe driver, 2026-10-22: 25.5 GiB refused with 26.4 GiB reported free, every smaller
// whole-arena retry refused after it), the cache is taken in pieces of an eighth (at least 256 MiB, at most
// the device's largest allocation), keeping the prefix the device gives.  STRATA_TEST_ALLOC_FAIL fakes the
// single allocation's refusal and counts down, STRATA_TEST_PIECE_FAIL the pieces' from the second on, so the
// cases run in this one process; a want beyond the card takes the pre-check's refusal.  A bucket past the kept
// prefix holds no slots, so its experts take the miss path instead of failing the open.
// Needs a GPU (exits 77 without one).
#include "strata/core/expert_cache.hpp"

#include "strata/core/runtime.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// the environment as the engine reads it (std::getenv); Windows has no setenv, and an empty value removes the variable
void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    if (value[0] == '\0') ::unsetenv(name);
    else ::setenv(name, value, 1);
#endif
}

int open_and_check(const char* name, const std::vector<int64_t>& slot_bytes, int64_t nl, int64_t ne,
                   int64_t want_slots) {
    std::string err;
    strata::core::ExpertCache cache;
    if (!cache.open_sized(slot_bytes, nl, ne, err)) {
        if (want_slots >= 0) {
            std::fprintf(stderr, "%s: refused, expected %lld slots: %s\n", name, (long long) want_slots,
                         err.c_str());
            return 1;
        }
        // the give-up: the refusal is said in full, naming the allocation or the card's free VRAM
        if (std::strstr(err.c_str(), "failed") == nullptr &&
            std::strstr(err.c_str(), "of VRAM is free") == nullptr) {
            std::fprintf(stderr, "%s: the refusal is not said in full: %s\n", name, err.c_str());
            return 1;
        }
        std::printf("%s: refused cleanly\n", name);
        return 0;
    }
    if (want_slots < 0) {
        std::fprintf(stderr, "%s: opened with %lld slots, expected a refusal\n", name, (long long) cache.slots());
        return 1;
    }
    if (cache.slots() != want_slots) {
        std::fprintf(stderr, "%s: %lld slots, expected %lld\n", name, (long long) cache.slots(),
                     (long long) want_slots);
        return 1;
    }
    if (cache.slot_offsets() == nullptr || cache.last_piece_first() < 0 ||
        cache.last_piece_first() >= cache.slots()) {
        std::fprintf(stderr, "%s: offsets %s, last piece at %lld of %lld\n", name,
                     cache.slot_offsets() ? "set" : "NULL", (long long) cache.last_piece_first(),
                     (long long) cache.slots());
        return 1;
    }
    // every slot the cache reports must have an address: the pieces' wrap-around arithmetic
    for (int64_t s = 0; s < cache.slots(); ++s)
        if (cache.device_slot((int32_t) s) == nullptr) {
            std::fprintf(stderr, "%s: slot %lld has no address\n", name, (long long) s);
            return 1;
        }
    if (!cache.zero_arena()) {
        std::fprintf(stderr, "%s: zero_arena failed\n", name);
        return 1;
    }
    std::printf("%s: %lld slots (last piece at %lld)\n", name, (long long) cache.slots(),
                (long long) cache.last_piece_first());
    return 0;
}

// The buckets form of the same cases: two exact formats in one arena, clipped to the kept prefix in bucket order.
int buckets_and_check(const char* name, int64_t want_slots, int64_t want_fmt2, int64_t want_fmt5) {
    std::string err;
    strata::core::ExpertCache cache;
    const std::vector<strata::core::ExpertCacheBucket> buckets{{2, 2, 256 << 20}, {5, 4, 256 << 20}};
    if (!cache.open_buckets(buckets, 3, 4, err)) {
        std::fprintf(stderr, "%s: refused, expected %lld slots: %s\n", name, (long long) want_slots, err.c_str());
        return 1;
    }
    if (cache.slots() != want_slots) {
        std::fprintf(stderr, "%s: %lld slots, expected %lld\n", name, (long long) cache.slots(),
                     (long long) want_slots);
        return 1;
    }
    int64_t got2 = 0, got5 = 0;
    for (int64_t e = 0; e < 4; ++e) {
        if (cache.admit(0, e, 2) != strata::core::kNotResident) ++got2;
        if (cache.admit(1, e, 5) != strata::core::kNotResident) ++got5;
    }
    if (got2 != want_fmt2 || got5 != want_fmt5) {
        std::fprintf(stderr, "%s: admitted %lld+%lld, expected %lld+%lld\n", name, (long long) got2,
                     (long long) got5, (long long) want_fmt2, (long long) want_fmt5);
        return 1;
    }
    for (int64_t s = 0; s < cache.slots(); ++s) {
        if (cache.device_slot((int32_t) s) == nullptr || cache.slot_format((int32_t) s) == 0) {
            std::fprintf(stderr, "%s: slot %lld has no address or format\n", name, (long long) s);
            return 1;
        }
    }
    if (!cache.zero_arena()) {
        std::fprintf(stderr, "%s: zero_arena failed\n", name);
        return 1;
    }
    std::printf("%s: %lld slots (%lld+%lld admitted)\n", name, (long long) cache.slots(), (long long) got2,
                (long long) got5);
    return 0;
}

}  // namespace

int main() {
    try { strata::core::Runtime::get(); } catch (const std::exception& e) {
        std::puts(e.what()); return 77;
    }
    // 6 slots of 256 MiB: one allocation of 1.5 GiB, the fallback's pieces 256 MiB each
    const std::vector<int64_t> slots(6, 256 << 20);
    int bad = 0;
    // plain: one allocation, no pieces
    bad += open_and_check("plain", slots, 3, 4, 6);
    // the one allocation refused: every piece taken, the whole cache in pieces
    set_env("STRATA_TEST_ALLOC_FAIL", "1");
    bad += open_and_check("fallback", slots, 3, 4, 6);
    // the same refusal, with the pieces from the second on refused: only the first piece's slots stay
    set_env("STRATA_TEST_ALLOC_FAIL", "1");
    set_env("STRATA_TEST_PIECE_FAIL", "5");
    bad += open_and_check("prefix", slots, 3, 4, 1);
    set_env("STRATA_TEST_PIECE_FAIL", "");
    set_env("STRATA_TEST_ALLOC_FAIL", "");
    // a want beyond any card here: the pre-check's refusal, said in full (Lower --expert-cache)
    bad += open_and_check("toobig", std::vector<int64_t>(6, 64ll << 30), 3, 4, -1);
    // the buckets in one arena: all of them, then the prefix only (the second format holds nothing)
    bad += buckets_and_check("buckets-plain", 6, 2, 4);
    set_env("STRATA_TEST_ALLOC_FAIL", "1");
    bad += buckets_and_check("buckets-fallback", 6, 2, 4);
    set_env("STRATA_TEST_ALLOC_FAIL", "1");
    set_env("STRATA_TEST_PIECE_FAIL", "5");
    bad += buckets_and_check("buckets-prefix", 1, 1, 0);
    set_env("STRATA_TEST_PIECE_FAIL", "");
    set_env("STRATA_TEST_ALLOC_FAIL", "");
    return bad == 0 ? 0 : 1;
}
