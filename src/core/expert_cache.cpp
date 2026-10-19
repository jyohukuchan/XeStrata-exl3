// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/core/expert_cache.cpp - R4's slot storage and residency table.  Read the header first.
#include "strata/core/expert_cache.hpp"

#include "strata/core/gpu.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <numeric>
#include <utility>
#include <cstring>

namespace strata::core {

bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "read_expert_profile: cannot open " + path;
        return false;
    }
    char magic[4] = {0, 0, 0, 0};
    uint32_t hdr[5] = {0, 0, 0, 0, 0};
    if (std::fread(magic, 1, 4, f) != 4 || std::fread(hdr, 4, 5, f) != 5) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " is too short to hold a header";
        return false;
    }
    if (std::memcmp(magic, "STRP", 4) != 0) {
        std::fclose(f);
        err = "read_expert_profile: " + path + " does not start with STRP";
        return false;
    }
    const uint32_t version = hdr[0], nl = hdr[1], ne = hdr[2], want = hdr[3], n_ranked = hdr[4];
    // the version comes first: the fields after it are only known to be these in a version 1 file, so a later
    // format is refused by its number rather than read as if it were this one
    if (version != 1) {
        std::fclose(f);
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "read_expert_profile: %s is a version %u profile but this engine reads version 1 - it was "
                      "written by a different release's tools/make_profile.py or --expert-profile-save",
                      path.c_str(), version);
        err = buf;
        return false;
    }
    if ((int64_t) nl != n_layers || (int64_t) ne != n_expert) {
        std::fclose(f);
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "read_expert_profile: %s is %ux%u but this model is %lldx%lld - it is a profile for a "
                      "different artifact", path.c_str(), nl, ne, (long long) n_layers, (long long) n_expert);
        err = buf;
        return false;
    }
    if (n_ranked > want) {
        std::fclose(f);
        err = "read_expert_profile: the header claims more ranked pairs than slots";
        return false;
    }
    ranked.assign(n_ranked, {0, 0});
    std::vector<uint16_t> raw((size_t) n_ranked * 2);
    if (n_ranked > 0 && std::fread(raw.data(), 2, (size_t) n_ranked * 2, f) != (size_t) n_ranked * 2) {
        std::fclose(f);
        err = "read_expert_profile: the ranked list is truncated";
        return false;
    }
    std::fclose(f);
    for (uint32_t i = 0; i < n_ranked; ++i) {
        const int32_t l = (int32_t) raw[(size_t) i * 2], e = (int32_t) raw[(size_t) i * 2 + 1];
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "read_expert_profile: pair %u is (layer %d, expert %d), out of range",
                          i, l, e);
            err = buf;
            return false;
        }
        ranked[(size_t) i] = {l, e};
    }
    slots = (int64_t) want;
    return true;
}

std::vector<std::pair<int32_t, int32_t>> rank_learned_profile(int64_t n_layers, int64_t n_expert,
                                                              const std::vector<uint8_t>& resident,
                                                              const std::vector<double>& heat,
                                                              const std::vector<std::pair<int32_t, int32_t>>& prior) {
    const size_t n = (size_t) (n_layers * n_expert);
    std::vector<int64_t> prior_rank(n, INT64_MAX);
    for (size_t r = 0; r < prior.size(); ++r) {
        const auto [l, e] = prior[r];
        if (l >= 0 && l < n_layers && e >= 0 && e < n_expert) {
            int64_t& pr = prior_rank[(size_t) (l * n_expert + e)];
            if (pr == INT64_MAX) pr = (int64_t) r;
        }
    }
    std::vector<int64_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = (int64_t) i;
    auto res = [&](int64_t i) { return (size_t) i < resident.size() && resident[(size_t) i] != 0; };
    auto ht = [&](int64_t i) { return (size_t) i < heat.size() ? heat[(size_t) i] : 0.0; };
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        if (res(a) != res(b)) return res(a);
        if (ht(a) != ht(b)) return ht(a) > ht(b);
        if (prior_rank[(size_t) a] != prior_rank[(size_t) b]) return prior_rank[(size_t) a] < prior_rank[(size_t) b];
        return a < b;
    });
    std::vector<std::pair<int32_t, int32_t>> ranked(n);
    for (size_t r = 0; r < n; ++r)
        ranked[r] = {(int32_t) (order[r] / n_expert), (int32_t) (order[r] % n_expert)};
    return ranked;
}

bool write_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                          const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
    if (n_layers <= 0 || n_expert <= 0 || n_layers > 65535 || n_expert > 65535) {
        err = "write_expert_profile: the model's layout does not fit the format";
        return false;
    }
    std::vector<int32_t> table((size_t) (n_layers * n_expert), -1);
    std::vector<uint16_t> pairs;
    pairs.reserve(ranked.size() * 2);
    for (size_t r = 0; r < ranked.size(); ++r) {
        const auto [l, e] = ranked[r];
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            err = "write_expert_profile: a ranked pair is out of range";
            return false;
        }
        table[(size_t) (l * n_expert + e)] = (int32_t) r;
        pairs.push_back((uint16_t) l);
        pairs.push_back((uint16_t) e);
    }
    const uint32_t hdr[5] = {1u, (uint32_t) n_layers, (uint32_t) n_expert, (uint32_t) ranked.size(),
                             (uint32_t) ranked.size()};
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        err = "write_expert_profile: cannot create " + tmp;
        return false;
    }
    // the format is little-endian (make_profile.py's "<"): so is every machine this engine runs on
    bool ok = std::fwrite("STRP", 1, 4, f) == 4 && std::fwrite(hdr, 4, 5, f) == 5 &&
              (pairs.empty() || std::fwrite(pairs.data(), 2, pairs.size(), f) == pairs.size()) &&
              std::fwrite(table.data(), 4, table.size(), f) == table.size();
    ok = (std::fclose(f) == 0) && ok;
    std::error_code ec;
    if (ok) std::filesystem::rename(tmp, path, ec);   // replaces an existing file (MoveFileEx / rename(2))
    if (!ok || ec) {
        std::filesystem::remove(tmp, ec);
        err = "write_expert_profile: cannot write " + path;
        return false;
    }
    return true;
}

ExpertCache::~ExpertCache() { close(); }

// ---- #533: the segmented arena (--vram-elastic): one address range for the whole arena, backed by physical
// segments, the tail's segments unmapped and mapped again later.  Nothing here runs unless a segment size was set.
bool ExpertCache::open_segmented(uint64_t want, std::string& err) {
    if (!strata::gpu::vmem_supported()) {
        err = "ExpertCache: --vram-elastic needs virtual memory (sycl_ext_oneapi_virtual_mem), which this GPU's "
              "runtime does not offer";
        return false;
    }
    const uint64_t gran = (uint64_t) strata::gpu::vmem_granularity();
    if (gran == 0) {
        err = std::string("ExpertCache: cannot read the allocation granularity: ") + strata::gpu::last_error();
        return false;
    }
    // a whole number of 2 MiB pages as well: Level Zero reports 64 KiB but refused a 0.49 GiB segment of 64 KiB
    // pages (UR_RESULT_ERROR_INVALID_VALUE, the B70), and CUDA's granularity is 2 MiB
    const uint64_t g = std::lcm(gran, (uint64_t) 2 << 20);
    const uint64_t total = (want + g - 1) / g * g;
    seg_ = (int64_t) (((uint64_t) seg_req_ + g - 1) / g * g);
    void* va = strata::gpu::vmem_reserve((size_t) total);
    if (va == nullptr) {
        err = std::string("ExpertCache: cannot reserve the segmented cache's address range: ") + strata::gpu::last_error();
        return false;
    }
    base_ = static_cast<uint8_t*>(va);
    reserved_ = total;
    for (uint64_t at = 0; at < total; at += (uint64_t) seg_) {
        segs_.push_back(nullptr);
        seg_size_.push_back((int64_t) std::min<uint64_t>((uint64_t) seg_, total - at));
    }
    for (size_t i = 0; i < segs_.size(); ++i) {
        segs_[i] = strata::gpu::vmem_map(base_ + (size_t) i * (size_t) seg_, (size_t) seg_size_[i]);
        if (segs_[i] == nullptr) {
            char buf[240];
            std::snprintf(buf, sizeof buf, "ExpertCache: device allocation failed: segment %zu of %zu (%.2f GiB) of "
                          "the segmented cache could not be backed: %s", i + 1, segs_.size(),
                          (double) seg_size_[i] / 1073741824.0, strata::gpu::last_error());
            err = buf;
            release_segmented();
            return false;
        }
        mapped_segs_ = (int64_t) i + 1;
    }
    return true;
}

void ExpertCache::release_segmented() {
    if (base_ != nullptr) strata::gpu::device_sync();
    for (size_t i = 0; i < segs_.size(); ++i)
        if (segs_[i] != nullptr) strata::gpu::vmem_unmap(base_ + (size_t) i * (size_t) seg_, (size_t) seg_size_[i], segs_[i]);
    if (base_ != nullptr && reserved_ > 0) strata::gpu::vmem_free(base_, (size_t) reserved_);
    segs_.clear();
    seg_size_.clear();
    mapped_segs_ = 0;
    reserved_ = 0;
    base_ = nullptr;
}

int64_t ExpertCache::mapped_bytes() const {
    if (segs_.empty()) return base_ != nullptr ? full_bytes() : 0;
    int64_t b = 0;
    for (int64_t i = 0; i < mapped_segs_; ++i) b += seg_size_[(size_t) i];
    return b;
}

int64_t ExpertCache::slots_within(int64_t bytes) const {
    if (bytes >= full_bytes()) return slots_;
    if (bytes <= 0) return 0;
    if (off_.empty()) return blob_ > 0 ? bytes / blob_ : 0;
    // off_[i + 1] is slot i's end: count the slots whose end is at most `bytes`
    return (int64_t) (std::upper_bound(off_.begin() + 1, off_.end(), (uint64_t) bytes) - (off_.begin() + 1));
}

bool ExpertCache::shrink(int64_t keep_bytes, std::string& err) {
    if (segs_.empty()) {
        err = "the expert cache is not segmented (the engine needs --vram-elastic)";
        return false;
    }
    int64_t keep = 0, at = 0;   // the segments [0, keep) hold the first keep_bytes
    while (keep < (int64_t) segs_.size() && at < keep_bytes) at += seg_size_[(size_t) keep++];
    if (keep >= mapped_segs_) return true;
    if (!strata::gpu::device_sync()) {
        err = std::string("the device failed before the cache shrank: ") + strata::gpu::last_error();
        return false;
    }
    for (int64_t i = mapped_segs_ - 1; i >= keep; --i) {
        strata::gpu::vmem_unmap(base_ + (size_t) i * (size_t) seg_, (size_t) seg_size_[(size_t) i], segs_[(size_t) i]);
        segs_[(size_t) i] = nullptr;
        mapped_segs_ = i;
    }
    live_slots_ = slots_within(mapped_bytes());
    return true;
}

bool ExpertCache::grow(int64_t want_bytes, std::string& err) {
    if (segs_.empty()) {
        err = "the expert cache is not segmented (the engine needs --vram-elastic)";
        return false;
    }
    int64_t at = mapped_bytes();
    while (mapped_segs_ < (int64_t) segs_.size() && at + seg_size_[(size_t) mapped_segs_] <= want_bytes) {
        const size_t i = (size_t) mapped_segs_;
        segs_[i] = strata::gpu::vmem_map(base_ + i * (size_t) seg_, (size_t) seg_size_[i]);
        if (segs_[i] == nullptr) {
            err = std::string("the device has no memory for another expert-cache segment: ") + strata::gpu::last_error();
            live_slots_ = slots_within(mapped_bytes());
            return false;
        }
        at += seg_size_[i];
        ++mapped_segs_;
    }
    live_slots_ = slots_within(mapped_bytes());
    return true;
}


namespace {
bool g_cache_vmm = false;   // set_vmm

bool valid_table_geometry(int64_t n_layers, int64_t n_expert) {
    if (n_layers <= 0 || n_expert <= 0) return false;
    const uint64_t layers = (uint64_t) n_layers, experts = (uint64_t) n_expert;
    if (layers > std::numeric_limits<uint64_t>::max() / experts) return false;
    const uint64_t entries = layers * experts;
    return entries <= (uint64_t) std::numeric_limits<size_t>::max() / sizeof(int32_t);
}
}  // namespace

void ExpertCache::set_vmm(bool enabled) { g_cache_vmm = enabled; }

bool ExpertCache::open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
                       std::string& err) {
    close();
    if (n_slots <= 0 || n_slots > std::numeric_limits<int32_t>::max()) {
        err = "ExpertCache: n_slots must be positive";
        return false;
    }
    if (!valid_table_geometry(n_layers, n_expert) || blob_bytes <= 0) {
        err = "ExpertCache: n_layers, n_expert and blob_bytes must all be positive";
        return false;
    }

    if ((uint64_t) n_slots > std::numeric_limits<uint64_t>::max() / (uint64_t) blob_bytes ||
        (uint64_t) n_slots * (uint64_t) blob_bytes > std::numeric_limits<size_t>::max()) {
        err = "ExpertCache: allocation size overflows";
        return false;
    }
    const uint64_t want = (uint64_t) n_slots * (uint64_t) blob_bytes;

    if (!allocate_arena(want, err)) return false;
    start_tables(n_slots, n_layers, n_expert, blob_bytes);
    return true;
}

bool ExpertCache::allocate_arena(uint64_t want, std::string& err) {
    if (want == 0 || want > (uint64_t) std::numeric_limits<size_t>::max()) {
        err = "ExpertCache: allocation size is not representable";
        return false;
    }
    // ---- **THE ALLOCATION IS CHECKED AGAINST THE CARD, NOT AGAINST THE REQUEST.**
    //
    // `cudaMalloc` failing is the easy case. The one that matters is a machine where the weights already own
    // most of VRAM: the cache then takes what is left and `slots()` would report the number ASKED FOR while
    // `device_slot()` walks off the end. So the free-VRAM figure is read and compared BEFORE the allocation,
    // and the two numbers are named in the refusal.
    size_t free_b = 0, total_b = 0;
    if (strata::gpu::mem_info(&free_b, &total_b)) {
        if ((uint64_t) free_b < want) {
            char buf[320];
            std::snprintf(buf, sizeof buf,
                          "ExpertCache: %.2f GiB requested, but only %.2f GiB of VRAM is free "
                          "(%.2f GiB of %.2f GiB total). Lower --expert-cache.",
                          (double) want / 1073741824.0, (double) free_b / 1073741824.0,
                          (double) (total_b - free_b) / 1073741824.0,
                          (double) total_b / 1073741824.0);
            err = buf;
            return false;
        }
    }

    if (seg_req_ > 0) {   // #533: --vram-elastic: physical segments behind one address range (zeroed below)
        if (!open_segmented(want, err)) return false;
    } else if (g_cache_vmm && vmm_available()) {
        // the elastic K/V: every chunk mapped now; the K/V may later take some of them (and give them back)
        auto r = std::make_unique<VmmRange>();
        if (!r->reserve(want) || !r->map_range(0, r->chunks(), [] { return (VmmChunk) 0; })) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "ExpertCache: mapping %.2f GiB of VRAM failed: out of memory",
                          (double) want / 1073741824.0);
            err = buf;
            return false;
        }
        base_ = r->base();
        vmm_ = std::move(r);
    } else if (!strata::gpu::alloc_device((void**) &base_, (size_t) want)) {
        base_ = nullptr;
        char buf[256];
        std::snprintf(buf, sizeof buf, "ExpertCache: device allocation of %.2f GiB failed: %s",
                      (double) want / 1073741824.0, strata::gpu::last_error());
        err = buf;
        return false;
    }
    // Zeroed so a slot read before it is filled is a DETERMINISTIC wrong answer rather than whatever the
    // allocator handed back.  A stale block of a previous process's memory would still sum to finite floats.
    if (!strata::gpu::memset(base_, 0, (size_t) want)) {
        err = std::string("ExpertCache: zeroing the slot arena failed: ") + strata::gpu::last_error();
        close();
        return false;
    }

    return true;
}

void ExpertCache::start_tables(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes) {
    residency_.assign((size_t) (n_layers * n_expert), kNotResident);
    slots_ = n_slots;
    live_slots_ = n_slots;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    blob_ = blob_bytes;
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;
    // R4.2g: each layer starts at the bottom of its own range.  Built here rather than lazily so `admit`
    // stays allocation-free on the token path.
    layer_next_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    for (int64_t l = 0; l < n_layers; ++l) {
        int64_t lo = 0, hi = 0;
        layer_slot_range(l, lo, hi);
        layer_next_[(size_t) l] = (int32_t) lo;
    }
}

// set_max_piece: the slots in runs of at most piece_max_ bytes, taken from the end so the last run (the prompt path's
// loan) is the largest; each run one allocation.  off_ gets each slot's address relative to slot 0's (modulo 2^64:
// the kernels add it to slot 0's address), size_off_ the offsets by size for bytes().
bool ExpertCache::open_pieces(const std::vector<uint64_t>& size_off, std::string& err) {
    const size_t n = size_off.size() - 1;
    const uint64_t want = size_off.back();
    size_t free_b = 0, total_b = 0;
    if (strata::gpu::mem_info(&free_b, &total_b) && (uint64_t) free_b < want) {
        char buf[256];
        std::snprintf(buf, sizeof buf, "ExpertCache: %.2f GiB of slots, but only %.2f GiB of VRAM is free",
                      (double) want / 1073741824.0, (double) free_b / 1073741824.0);
        err = buf;
        return false;
    }
    std::vector<size_t> first;   // each run's first slot, from the last run back
    for (size_t end = n; end > 0;) {
        size_t b = end;
        while (b > 0 && size_off[end] - size_off[b - 1] <= piece_max_) --b;
        if (b == end) { err = "ExpertCache: a slot is larger than the device's largest allocation"; return false; }
        first.push_back(b);
        end = b;
    }
    std::reverse(first.begin(), first.end());
    std::vector<uint64_t> addr(n + 1, 0);
    for (size_t r = 0; r < first.size(); ++r) {
        const size_t lo = first[r], hi = r + 1 < first.size() ? first[r + 1] : n;
        const uint64_t bytes = size_off[hi] - size_off[lo];
        void* p = nullptr;
        if (!strata::gpu::alloc_device(&p, (size_t) bytes) || !strata::gpu::memset(p, 0, (size_t) bytes)) {
            if (p != nullptr) strata::gpu::free(p);
            char buf[256];
            std::snprintf(buf, sizeof buf, "ExpertCache: device allocation of %.2f GiB (a piece of %.2f GiB) failed: %s",
                          (double) bytes / 1073741824.0, (double) want / 1073741824.0, strata::gpu::last_error());
            err = buf;
            close();
            return false;
        }
        if (r == 0) base_ = static_cast<uint8_t*>(p);
        else pieces_.push_back(p);
        const uint64_t at = (uint64_t) (uintptr_t) p - (uint64_t) (uintptr_t) base_;
        for (size_t i = lo; i <= hi; ++i) addr[i] = at + (size_off[i] - size_off[lo]);
    }
    last_piece_first_ = (int64_t) first.back();
    off_ = std::move(addr);
    size_off_ = size_off;
    return true;
}

bool ExpertCache::open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_layers, int64_t n_expert,
                             std::string& err) {
    if (slot_bytes.empty()) { err = "ExpertCache: no slots"; return false; }
    if (!valid_table_geometry(n_layers, n_expert) || slot_bytes.size() > (size_t) std::numeric_limits<int32_t>::max()) {
        err = "ExpertCache: invalid sized-cache geometry";
        return false;
    }
    int64_t mx = 0;
    std::vector<uint64_t> off(slot_bytes.size() + 1, 0);
    for (size_t i = 0; i < slot_bytes.size(); ++i) {
        if (slot_bytes[i] <= 0 || (uint64_t) slot_bytes[i] > std::numeric_limits<uint64_t>::max() - 255) {
            err = "ExpertCache: invalid slot byte size";
            return false;
        }
        // 256-byte aligned slots, so every blob starts where the kernels' vector loads expect it
        const uint64_t aligned = ((uint64_t) slot_bytes[i] + 255) / 256 * 256;
        if (off[i] > std::numeric_limits<uint64_t>::max() - aligned) {
            err = "ExpertCache: sized-cache allocation overflows";
            return false;
        }
        off[i + 1] = off[i] + aligned;
        mx = slot_bytes[i] > mx ? slot_bytes[i] : mx;
    }
    if (off.back() > (uint64_t) std::numeric_limits<int64_t>::max() ||
        off.back() > (uint64_t) std::numeric_limits<size_t>::max()) {
        err = "ExpertCache: sized-cache allocation exceeds addressable size";
        return false;
    }
    if (piece_max_ > 0 && off.back() > piece_max_ && seg_req_ == 0 && !(g_cache_vmm && vmm_available())) {
        close();
        if (!open_pieces(off, err)) return false;
        start_tables((int64_t) slot_bytes.size(), n_layers, n_expert, mx);
        slot_payload_bytes_.clear();
        for (int64_t bytes : slot_bytes) slot_payload_bytes_.push_back((uint64_t) bytes);
        // start_tables seeded the cursors on these slots: nothing below to redo
        return true;
    }
    // Allocate the aligned arena directly. `off.back()` is a byte count and can exceed INT32_MAX even though
    // the actual slot count is bounded above; routing it through open() would confuse those two quantities.
    close();
    if (!allocate_arena(off.back(), err)) return false;
    start_tables((int64_t) slot_bytes.size(), n_layers, n_expert, mx);
    slots_ = (int64_t) slot_bytes.size();
    live_slots_ = slots_;
    blob_ = mx;
    off_ = std::move(off);
    slot_payload_bytes_.clear();
    for (int64_t bytes : slot_bytes) slot_payload_bytes_.push_back((uint64_t) bytes);
    // start_tables() seeded cursors before the sized offsets were installed; rebuild them because the actual
    // slot count is the number of payloads, not the aligned arena byte count.
    layer_next_.assign((size_t) (n_layers > 0 ? n_layers : 0), 0);
    for (int64_t l = 0; l < n_layers; ++l) {
        int64_t lo = 0, hi = 0;
        layer_slot_range(l, lo, hi);
        layer_next_[(size_t) l] = (int32_t) lo;
    }
    return true;
}

bool ExpertCache::open_buckets(const std::vector<ExpertCacheBucket>& buckets, int64_t n_layers, int64_t n_expert,
                               std::string& err) {
    if (buckets.empty()) { err = "ExpertCache: no format buckets"; return false; }
    if (n_layers <= 0 || n_expert <= 0) { err = "ExpertCache: invalid bucket-cache geometry"; return false; }
    if ((uint64_t) n_layers > (uint64_t) std::numeric_limits<size_t>::max() ||
        buckets.size() > std::numeric_limits<size_t>::max() / (size_t) n_layers) {
        err = "ExpertCache: bucket residency table overflows";
        return false;
    }
    std::vector<int64_t> slot_bytes;
    std::vector<uint64_t> formats;
    std::vector<int32_t> first, counts;
    for (const ExpertCacheBucket& bucket : buckets) {
        if (bucket.slots <= 0 || bucket.blob_bytes <= 0 || bucket.slots > std::numeric_limits<int32_t>::max()) {
            err = "ExpertCache: invalid format bucket size";
            return false;
        }
        for (uint64_t format : formats)
            if (format == bucket.format_id) { err = "ExpertCache: duplicate format bucket"; return false; }
        if ((uint64_t) slot_bytes.size() + (uint64_t) bucket.slots > (uint64_t) std::numeric_limits<int32_t>::max() ||
            (uint64_t) slot_bytes.size() + (uint64_t) bucket.slots > std::numeric_limits<size_t>::max()) {
            err = "ExpertCache: format bucket slot count overflows";
            return false;
        }
        formats.push_back(bucket.format_id);
        first.push_back((int32_t) slot_bytes.size());
        counts.push_back((int32_t) bucket.slots);
        slot_bytes.insert(slot_bytes.end(), (size_t) bucket.slots, bucket.blob_bytes);
    }
    if (!open_sized(slot_bytes, n_layers, n_expert, err)) return false;
    bucket_formats_ = std::move(formats);
    bucket_first_ = std::move(first);
    bucket_slots_ = std::move(counts);
    bucket_next_.resize(bucket_formats_.size());
    bucket_layer_next_.assign(bucket_formats_.size() * (size_t) n_layers, 0);
    for (size_t b = 0; b < bucket_formats_.size(); ++b) {
        bucket_next_[b] = bucket_first_[b];
        const int64_t q = bucket_slots_[b] / n_layers;
        for (int64_t layer = 0; layer < n_layers; ++layer)
            bucket_layer_next_[b * (size_t) n_layers + (size_t) layer] =
                bucket_first_[b] + (int32_t) (q * layer);
    }
    slot_formats_.assign(slot_bytes.size(), 0);
    for (size_t b = 0; b < bucket_formats_.size(); ++b)
        for (int32_t s = 0; s < bucket_slots_[b]; ++s) slot_formats_[(size_t) bucket_first_[b] + (size_t) s] = bucket_formats_[b];
    return true;
}

void ExpertCache::close() {
    off_.clear();
    size_off_.clear();
    last_piece_first_ = 0;
    for (void* p : pieces_) strata::gpu::free(p);
    pieces_.clear();
    if (!segs_.empty()) {
        release_segmented();
    } else if (vmm_) {
        vmm_.reset();   // unmaps and frees every chunk it still holds
        base_ = nullptr;
    } else if (base_ != nullptr) {
        strata::gpu::free(base_);
        base_ = nullptr;
    }
    residency_.clear();
    slots_ = 0;
    live_slots_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    blob_ = 0;
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;
    layer_next_.clear();
    slot_formats_.clear();
    bucket_formats_.clear();
    bucket_first_.clear();
    bucket_slots_.clear();
    bucket_next_.clear();
    bucket_layer_next_.clear();
    slot_payload_bytes_.clear();
}

/// R4.2g.  Layer `l` owns `[l*q, (l+1)*q)` with `q = slots_ / n_layers_`; the LAST layer takes whatever is
/// left over, so the ranges always cover `0..slots_` exactly and no slot is orphaned by the division.
void ExpertCache::layer_slot_range(int64_t layer, int64_t& lo, int64_t& hi) const {
    lo = 0;
    hi = 0;
    if (n_layers_ <= 0 || slots_ <= 0 || layer < 0 || layer >= n_layers_) return;
    const int64_t q = slots_ / n_layers_;
    lo = layer * q;
    hi = (layer == n_layers_ - 1) ? slots_ : (layer + 1) * q;
}

int32_t ExpertCache::slot_of(int64_t layer, int64_t expert) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return kNotResident;
    return residency_[(size_t) (layer * n_expert_ + expert)];
}

int32_t ExpertCache::admit(int64_t layer, int64_t expert) {
    if (!bucket_formats_.empty()) return admit(layer, expert, 0);
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return kNotResident;
    const size_t at = (size_t) (layer * n_expert_ + expert);
    if (residency_[at] != kNotResident) return residency_[at];
    // R4.2g: THE PER-LAYER PATH.  Same "no eviction" rule, but the ceiling is this layer's own range rather
    // than one counter shared by all 48 - which is what confined the measured hit rate to 2.97%.
    if (per_layer_) {
        if (layer_next_.empty()) return kNotResident;
        int64_t lo = 0, hi = 0;
        layer_slot_range(layer, lo, hi);
        if ((int64_t) layer_next_[(size_t) layer] >= hi) return kNotResident;   // this layer's quota is full
        residency_[at] = layer_next_[(size_t) layer]++;
        ++admitted_;
        return residency_[at];
    }
    if (next_free_ >= live_slots_) return kNotResident;   // full: no eviction, deliberately - see the header
    residency_[at] = (int32_t) next_free_;
    return (int32_t) next_free_++;
}

int ExpertCache::bucket_of(uint64_t format_id) const {
    for (size_t i = 0; i < bucket_formats_.size(); ++i)
        if (bucket_formats_[i] == format_id) return (int) i;
    return -1;
}

int32_t ExpertCache::admit(int64_t layer, int64_t expert, uint64_t format_id) {
    if (bucket_formats_.empty()) return format_id == 0 ? admit(layer, expert) : kNotResident;
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return kNotResident;
    const size_t at = (size_t) (layer * n_expert_ + expert);
    if (residency_[at] != kNotResident)
        return slot_format(residency_[at]) == format_id ? residency_[at] : kNotResident;
    const int bucket = bucket_of(format_id);
    if (bucket < 0) return kNotResident;
    int32_t slot = kNotResident;
    if (per_layer_) {
        const int64_t q = bucket_slots_[(size_t) bucket] / n_layers_;
        const int64_t bucket_hi = layer == n_layers_ - 1
                               ? (int64_t) bucket_first_[(size_t) bucket] + bucket_slots_[(size_t) bucket]
                               : (int64_t) bucket_first_[(size_t) bucket] + q * (layer + 1);
        const int64_t hi = std::min<int64_t>(bucket_hi, live_slots_);
        int32_t& next = bucket_layer_next_[(size_t) bucket * (size_t) n_layers_ + (size_t) layer];
        if ((int64_t) next >= hi) return kNotResident;
        slot = next++;
    } else {
        int32_t& next = bucket_next_[(size_t) bucket];
        const int32_t end = (int32_t) std::min<int64_t>(
            (int64_t) bucket_first_[(size_t) bucket] + bucket_slots_[(size_t) bucket], live_slots_);
        if (next >= end) return kNotResident;
        slot = next++;
    }
    residency_[at] = slot;
    ++admitted_;
    return slot;
}

uint64_t ExpertCache::slot_format(int32_t slot) const {
    if (slot < 0 || (size_t) slot >= slot_formats_.size()) return 0;
    return slot_formats_[(size_t) slot];
}

uint64_t ExpertCache::slot_capacity(int32_t slot) const {
    if (slot < 0 || slot >= slots_) return 0;
    if ((size_t) slot < slot_payload_bytes_.size()) return slot_payload_bytes_[(size_t) slot];
    if (!size_off_.empty()) return size_off_[(size_t) slot + 1] - size_off_[(size_t) slot];
    if (!off_.empty()) return off_[(size_t) slot + 1] - off_[(size_t) slot];
    return (uint64_t) blob_;
}

uint8_t* ExpertCache::device_slot(int32_t slot) {
    if (slot < 0 || slot >= slots_) return nullptr;
    if (!off_.empty()) return base_ + off_[(size_t) slot];
    return base_ + (size_t) slot * (size_t) blob_;
}

const uint8_t* ExpertCache::device_slot(int32_t slot) const {
    if (slot < 0 || slot >= slots_) return nullptr;
    if (!off_.empty()) return base_ + off_[(size_t) slot];
    return base_ + (size_t) slot * (size_t) blob_;
}

bool ExpertCache::fill_slot(int32_t slot, const uint8_t* host_blob, void* stream, std::string& err, int64_t bytes) {
    const uint64_t capacity = slot_capacity(slot);
    if (capacity == 0 || (bytes > 0 && (uint64_t) bytes > capacity)) {
        err = "ExpertCache::fill_slot: bytes exceed slot capacity";
        return false;
    }
    const size_t n = (size_t) (bytes > 0 ? bytes : capacity);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr) {
        err = "ExpertCache::fill_slot: slot " + std::to_string(slot) + " is outside 0.." +
              std::to_string(slots_ - 1);
        return false;
    }
    if (host_blob == nullptr) {
        err = "ExpertCache::fill_slot: the host blob is null";
        return false;
    }
    const bool e = strata::gpu::copy_async(dst, host_blob, n, stream);
    if (!e) {
        err = std::string("ExpertCache::fill_slot: ") + strata::gpu::last_error();
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::fill_slot_blocking(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes) {
    const uint64_t capacity = slot_capacity(slot);
    if (capacity == 0 || (bytes > 0 && (uint64_t) bytes > capacity)) {
        err = "ExpertCache::fill_slot_blocking: bytes exceed slot capacity";
        return false;
    }
    const size_t n = (size_t) (bytes > 0 ? bytes : capacity);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr) {
        err = "ExpertCache::fill_slot_blocking: slot outside the arena";
        return false;
    }
    if (host_blob == nullptr) {
        err = "ExpertCache::fill_slot_blocking: the host blob is null";
        return false;
    }
    const bool e = strata::gpu::copy(dst, host_blob, n);
    if (!e) {
        err = std::string("ExpertCache::fill_slot_blocking: ") + strata::gpu::last_error();
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::fill_slot_queued(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes) {
    const uint64_t capacity = slot_capacity(slot);
    if (capacity == 0 || (bytes > 0 && (uint64_t) bytes > capacity)) {
        err = "ExpertCache::fill_slot_queued: bytes exceed slot capacity";
        return false;
    }
    const size_t n = (size_t) (bytes > 0 ? bytes : capacity);
    uint8_t* dst = device_slot(slot);
    if (dst == nullptr || host_blob == nullptr) {
        err = dst == nullptr ? "ExpertCache::fill_slot_queued: slot outside the arena"
                             : "ExpertCache::fill_slot_queued: the host blob is null";
        return false;
    }
    const bool e = strata::gpu::copy_async(dst, host_blob, n, 0);
    if (!e) {
        err = std::string("ExpertCache::fill_slot_queued: ") + strata::gpu::last_error();
        return false;
    }
    ++fills_;
    return true;
}

bool ExpertCache::sync_queued(std::string& err) {
    const bool e = strata::gpu::stream_sync(0);
    if (!e) {
        err = std::string("ExpertCache::sync_queued: ") + strata::gpu::last_error();
        return false;
    }
    return true;
}

bool ExpertCache::verify_slot(int32_t slot, const uint8_t* host_blob, std::string& err, int64_t bytes) {
    const uint64_t capacity = slot_capacity(slot);
    if (capacity == 0 || (bytes > 0 && (uint64_t) bytes > capacity)) {
        err = "ExpertCache::verify_slot: bytes exceed slot capacity";
        return false;
    }
    const int64_t nb = (int64_t) (bytes > 0 ? bytes : capacity);
    const uint8_t* src = device_slot(slot);
    if (src == nullptr) {
        err = "ExpertCache::verify_slot: slot outside the arena";
        return false;
    }
    // `cudaMemcpy` and not `cudaMemcpyAsync`: this is a startup check, and a check that can be read before it
    // has happened is not a check.  It also synchronises the fills queued before it, which is what makes the
    // comparison meaningful.
    std::vector<uint8_t> got((size_t) nb);
    const bool e = strata::gpu::copy(got.data(), src, (size_t) nb);
    if (!e) {
        err = std::string("ExpertCache::verify_slot: ") + strata::gpu::last_error();
        return false;
    }
    if (std::memcmp(got.data(), host_blob, (size_t) nb) != 0) {
        size_t first = 0;
        while (first < (size_t) nb && got[first] == host_blob[first]) ++first;
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "ExpertCache::verify_slot: slot %d differs from the arena at byte %llu (of %lld)",
                      (int) slot, (unsigned long long) first, (long long) blob_);
        err = buf;
        return false;
    }
    return true;
}

}  // namespace strata::core
