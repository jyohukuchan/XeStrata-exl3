// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Bounded EXL3 source/cache/CPU/GPU plan integration.
#include "strata/core/exl3_expert_runtime.hpp"
#include "strata/core/exl3_prepare_profile.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/host_memory_probe.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/device.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace strata::core {
namespace {

class HostTimelineScope {
public:
    HostTimelineScope(const char* phase, int64_t layer, int64_t hits = -1, int64_t misses = -1)
        : phase_(phase), layer_(layer), hits_(hits), misses_(misses) {
        static const bool enabled = [] {
            const char* flag = std::getenv("STRATA_EXL3_TIMELINE");
            return flag != nullptr && flag[0] != '\0' && std::strcmp(flag, "0") != 0;
        }();
        enabled_ = enabled;
        if (enabled_) emit("begin");
    }
    ~HostTimelineScope() { if (enabled_) emit("end"); }
    HostTimelineScope(const HostTimelineScope&) = delete;
    HostTimelineScope& operator=(const HostTimelineScope&) = delete;

private:
    void emit(const char* edge) const {
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        std::fprintf(stderr, "{\"kind\":\"exl3_host_timeline\",\"clock\":\"steady_ns\",\"ts_ns\":%lld,"
                             "\"phase\":\"%s\",\"edge\":\"%s\",\"layer\":%lld,\"hits\":%lld,\"misses\":%lld}\n",
                     (long long) ns, phase_, edge, (long long) layer_, (long long) hits_, (long long) misses_);
    }
    const char* phase_;
    int64_t layer_, hits_, misses_;
    bool enabled_ = false;
};

bool product_fits(int64_t a, int64_t b, int64_t limit, int64_t& result) {
    if (a <= 0 || b <= 0 || a > limit / b) return false;
    result = a * b;
    return true;
}

bool span_fits(int64_t rows, int64_t stride, int64_t width) {
    if (rows <= 0 || stride < width) return false;
    const uint64_t max_float = static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max()) / sizeof(float);
    return static_cast<uint64_t>(rows) <= max_float / static_cast<uint64_t>(stride);
}

}  // namespace

Exl3ExpertRuntime::~Exl3ExpertRuntime() { close(); }

void Exl3ExpertRuntime::close() {
    if (opened_) (void) strata::gpu::device_sync();
    cache_.close();
    if (direct_staging_ != nullptr) strata::gpu::free(direct_staging_);
    direct_staging_ = nullptr;
    direct_stride_ = 0;
    direct_capacity_ = 0;
    direct_ready_ = false;
    if (split_cache_ != nullptr) strata::gpu::free(split_cache_);
    split_cache_ = nullptr;
    split_cache_slots_ = 0;
    split_cache_index_.clear();
    split_cache_lru_.clear();
    split_cache_slot_pos_.clear();
    split_cache_slot_key_.clear();
    split_cache_free_.clear();
    split_cache_hits_ = 0;
    split_cache_lookups_ = 0;
    heat_.clear();
    total_hit_entries_ = 0;
    total_cpu_miss_entries_ = 0;
    total_direct_miss_entries_ = 0;
    direct_entries_ = 0;
    cpu_us_per_entry_ = 0.0;
    pcie_bytes_per_sec_ = 0.0;
    split_windows_ = 0;
    split_gpu_entries_ = 0;
    split_miss_entries_ = 0;
    split_last_gpu_ = 0;
    split_last_cpu_ = 0;
    if (d_scratch_ != nullptr) strata::gpu::free(d_scratch_);
    // The seven per-window plan arrays are views into the plan arenas, not separate allocations.
    if (d_plan_arena_ != nullptr) strata::gpu::free(d_plan_arena_);
    if (d_format_table_ != nullptr) strata::gpu::free(d_format_table_);
    if (d_entry_plans_ != nullptr) strata::gpu::free(d_entry_plans_);
    if (d_pair_blob_base_ != nullptr) strata::gpu::free(d_pair_blob_base_);
    if (d_pair_format_index_ != nullptr) strata::gpu::free(d_pair_format_index_);
    if (h_plan_arena_ != nullptr) strata::gpu::free(h_plan_arena_);
    if (h_format_table_ != nullptr) strata::gpu::free(h_format_table_);
    d_scratch_ = nullptr;
    h_plan_arena_ = nullptr; d_plan_arena_ = nullptr; plan_arena_bytes_ = 0;
    d_blob_base_ = nullptr; d_format_index_ = nullptr; d_format_table_ = nullptr;
    d_start_ = nullptr; d_tok_ = nullptr; d_dst_ = nullptr; d_ngroups_ = nullptr; d_nentries_ = nullptr;
    d_entry_plans_ = nullptr;
    d_pair_blob_base_ = nullptr; d_pair_format_index_ = nullptr;
    h_blob_base_ = nullptr; h_format_index_ = nullptr; h_format_table_ = nullptr;
    h_start_ = nullptr; h_tok_ = nullptr; h_dst_ = nullptr; h_ngroups_ = nullptr; h_nentries_ = nullptr;
    source_ = nullptr;
    pool_ = nullptr;
    n_layers_ = n_expert_ = max_tokens_ = routed_k_ = max_entries_ = hidden_ = max_ff_ = 0;
    pair_count_ = 0;
    cpu_rowparallel_jobs_ = 0;
    scratch_elements_ = 0;
    pair_formats_.clear(); pair_format_index_.clear(); format_table_host_.clear();
    cpu_jobs_.clear(); miss_ids_.clear(); registry_.clear();
    device_plan_ = {};
    device_runtime_ = nullptr;
    opened_ = prepared_ = cpu_ready_ = pending_gpu_ = false;
    all_resident_ = false;
    prepared_layer_ = -1;
    miss_entries_ = 0;
}

int64_t Exl3ExpertRuntime::plan_split_gpu(const SplitCandidate* candidates, int64_t n,
                                               double cpu_us_per_entry, double bytes_per_sec,
                                               int64_t capacity) {
    if (candidates == nullptr || n <= 0 || capacity <= 0) return 0;
    if (!(cpu_us_per_entry > 0.0) || !(bytes_per_sec > 0.0)) return 0;
    int64_t cap = n < capacity ? n : capacity;
    int64_t total_entries = 0;
    for (int64_t i = 0; i < n; ++i)
        if (candidates[i].entries > 0) total_entries += candidates[i].entries;
    if (total_entries <= 0) return 0;
    // k = 0 (every miss on the CPU) is the default: a larger k must strictly beat it.
    double best = static_cast<double>(total_entries) * cpu_us_per_entry;
    int64_t best_k = 0;
    uint64_t gpu_bytes = 0;
    int64_t gpu_entries = 0;
    for (int64_t k = 1; k <= cap; ++k) {
        if (candidates[k - 1].entries > 0) gpu_entries += candidates[k - 1].entries;
        gpu_bytes += candidates[k - 1].bytes;
        const double gpu_us = static_cast<double>(gpu_bytes) / bytes_per_sec * 1e6;
        const double cpu_us = static_cast<double>(total_entries - gpu_entries) * cpu_us_per_entry;
        const double cost = cpu_us > gpu_us ? cpu_us : gpu_us;
        if (cost < best) {
            best = cost;
            best_k = k;
        }
    }
    return best_k;
}

namespace {

/// STRATA_EXL3_SPLIT_CACHE_MIB: the pinned canonical-blob cache for Split's GPU share, in MiB.
/// Negative (the default) sizes it from the host RAM the machine reports at open; 0 keeps the
/// per-window staging (every window converts its GPU share anew).
int64_t split_cache_budget_mib() {
    const char* text = std::getenv("STRATA_EXL3_SPLIT_CACHE_MIB");
    if (text == nullptr || text[0] == '\0') return -1;
    char* end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    if (end == text || (end != nullptr && *end != '\0') || value < 0) return -1;
    return static_cast<int64_t>(value);
}

}  // namespace

void Exl3ExpertRuntime::measure_pcie_bandwidth() {
    pcie_bytes_per_sec_ = 0.0;
    if (direct_staging_ == nullptr || direct_stride_ == 0 || direct_capacity_ <= 0) return;
    const uint64_t arena = direct_stride_ * static_cast<uint64_t>(direct_capacity_);
    // The timed copy's own size comes from the staging arena, not a fixed constant: most of the
    // arena up to 8 MiB, so a small card's staging still measures itself.  A card whose staging
    // is smaller than one stride measures that stride.
    size_t n = static_cast<size_t>(arena > (8ull << 20) ? (8ull << 20) : arena);
    if (n == 0) return;
    uint8_t* dev = static_cast<uint8_t*>(strata::gpu::alloc_device(n));
    if (dev == nullptr) return;
    std::memset(direct_staging_, 0x5a, n);
    double best_s = 0.0;
    for (int iter = 0; iter < 3; ++iter) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!strata::gpu::copy_async(dev, direct_staging_, n, nullptr) ||
            !strata::gpu::stream_sync(nullptr)) {
            strata::gpu::free(dev);
            return;
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (s > 0.0 && (best_s == 0.0 || s < best_s)) best_s = s;
    }
    strata::gpu::free(dev);
    if (best_s > 0.0) pcie_bytes_per_sec_ = static_cast<double>(n) / best_s;
}

void Exl3ExpertRuntime::open_split_cache() {
    split_cache_ = nullptr;
    split_cache_slots_ = 0;
    split_cache_index_.clear();
    split_cache_lru_.clear();
    split_cache_slot_pos_.clear();
    split_cache_slot_key_.clear();
    split_cache_free_.clear();
    split_cache_hits_ = 0;
    split_cache_lookups_ = 0;
    if (direct_staging_ == nullptr || direct_stride_ == 0) return;
    // The budget is a share of the host RAM the process can get NOW - after the source's pool
    // owns its misses, which open() prewarmed just before - between bounds that keep a small
    // machine's pool intact and cap a large machine's take.  An explicit MiB budget overrides the
    // share; 0 keeps the per-window staging.  Whatever does not fit in pinned memory is simply
    // not cached: the per-window staging below stays in charge of it.
    uint64_t budget = 0;
    const int64_t asked_mib = split_cache_budget_mib();
    if (asked_mib >= 0) {
        budget = static_cast<uint64_t>(asked_mib) << 20;
    } else {
        detail::HostMemory memory;
        if (!detail::host_available_memory(memory)) {
            std::fprintf(stderr, "Exl3ExpertRuntime: split cache off (host memory unreadable); "
                                 "windows stage their GPU share anew\n");
            return;
        }
        budget = memory.available / 16;
        if (budget < (256ull << 20)) budget = 256ull << 20;
        if (budget > (4ull << 30)) budget = 4ull << 30;
    }
    int64_t slots = budget >= direct_stride_ ? static_cast<int64_t>(budget / direct_stride_) : 0;
    if (slots <= 0) {
        std::fprintf(stderr, "Exl3ExpertRuntime: split cache off (no room for one blob); "
                             "windows stage their GPU share anew\n");
        return;
    }
    // A refused arena is halved and tried again, the way the device cache plan is; only slots
    // the pinning actually granted become GPU candidates' cache.
    for (; slots > 0; slots /= 2) {
        const uint64_t bytes = static_cast<uint64_t>(slots) * direct_stride_;
        if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) continue;
        split_cache_ = static_cast<uint8_t*>(strata::gpu::alloc_host(static_cast<size_t>(bytes)));
        if (split_cache_ != nullptr) break;
    }
    if (split_cache_ == nullptr) {
        std::fprintf(stderr, "Exl3ExpertRuntime: split cache off (pinned allocation refused); "
                             "windows stage their GPU share anew\n");
        slots = 0;
        return;
    }
    split_cache_slots_ = slots;
    split_cache_slot_pos_.resize(static_cast<size_t>(slots));
    split_cache_slot_key_.assign(static_cast<size_t>(slots), -1);
    split_cache_free_.reserve(static_cast<size_t>(slots));
    for (int64_t slot = slots - 1; slot >= 0; --slot) split_cache_free_.push_back(slot);
    std::fprintf(stderr,
                 "Exl3ExpertRuntime: split cache %lld slots x %llu bytes (%.2f GiB pinned); "
                 "0 disables it with STRATA_EXL3_SPLIT_CACHE_MIB=0\n",
                 (long long) slots, (unsigned long long) direct_stride_,
                 static_cast<double>(static_cast<uint64_t>(slots) * direct_stride_) / 1073741824.0);
}

int64_t Exl3ExpertRuntime::split_cache_assign(int64_t layer, int32_t expert) {
    if (split_cache_ == nullptr || split_cache_slots_ <= 0 || source_ == nullptr) return -1;
    if (layer < 0 || layer >= (int64_t{1} << 31) || expert < 0) return -1;
    const int64_t key = (layer << 32) | static_cast<uint32_t>(expert);
    ++split_cache_lookups_;
    const auto found = split_cache_index_.find(key);
    if (found != split_cache_index_.end()) {
        const int64_t slot = found->second;
        split_cache_lru_.splice(split_cache_lru_.begin(), split_cache_lru_, split_cache_slot_pos_[(size_t) slot]);
        ++split_cache_hits_;
        return slot;
    }
    int64_t slot = -1;
    if (!split_cache_free_.empty()) {
        slot = split_cache_free_.back();
        split_cache_free_.pop_back();
    } else if (!split_cache_lru_.empty()) {
        // Evict the least recently used slot outside the current window: windows run one at a
        // time and a slot of this window is at the list's front, never its back.
        const int64_t victim = split_cache_lru_.back();
        const auto vit = split_cache_index_.find(victim);
        if (vit == split_cache_index_.end()) return -1;
        slot = vit->second;
        split_cache_index_.erase(vit);
        split_cache_lru_.pop_back();
    }
    if (slot < 0 || slot >= split_cache_slots_) return -1;
    uint8_t* staged = split_cache_ + static_cast<size_t>(slot) * direct_stride_;
    if (!source_->copy_blob(layer, expert, staged)) {
        split_cache_free_.push_back(slot);
        return -1;
    }
    split_cache_index_.emplace(key, slot);
    split_cache_slot_key_[(size_t) slot] = key;
    split_cache_lru_.push_front(key);
    split_cache_slot_pos_[(size_t) slot] = split_cache_lru_.begin();
    return slot;
}

bool Exl3ExpertRuntime::allocate_images(std::string& err) {
    const size_t groups = static_cast<size_t>(max_entries_);
    const size_t entries = static_cast<size_t>(max_entries_);
    const size_t formats = format_table_host_.size();
    auto host = [&](void** p, size_t bytes, const char* what) {
        if (*p == nullptr) *p = strata::gpu::alloc_host(bytes);
        if (*p == nullptr) { err = std::string("exl3 runtime: ") + what + " host allocation failed"; return false; }
        return true;
    };
    auto device = [&](void** p, size_t bytes, const char* what) {
        if (*p == nullptr) *p = strata::gpu::alloc_device(bytes);
        if (*p == nullptr) { err = std::string("exl3 runtime: ") + what + " device allocation failed"; return false; }
        return true;
    };
    if (groups > std::numeric_limits<size_t>::max() / sizeof(uint64_t) ||
        groups > std::numeric_limits<size_t>::max() / sizeof(int32_t) ||
        entries > std::numeric_limits<size_t>::max() / sizeof(int32_t) ||
        entries > std::numeric_limits<size_t>::max() / sizeof(strata::kernels::exl3::GroupedEntryPlan) ||
        formats > std::numeric_limits<size_t>::max() / sizeof(Exl3ExpertFormatDesc) ||
        groups + 1 > std::numeric_limits<size_t>::max() / sizeof(int32_t)) {
        err = "exl3 runtime: plan allocation overflows";
        return false;
    }
    // The per-window plan arena: the seven arrays copy_plan uploads, laid out once here and carved in the same
    // order on the host and the device.  blob_base (uint64) comes first so the allocation's own alignment covers
    // it; every later array starts at a multiple of its alignment.  Sizes come from max_entries_ (= max_tokens ×
    // routed_k), so every window width up to the bound the runtime opened with - one row or many - fits.
    size_t blob_off = 0, format_off = 0, start_off = 0, tok_off = 0, dst_off = 0, ngroups_off = 0, nentries_off = 0;
    size_t plan_bytes = 0;
    auto append = [&](size_t bytes, size_t align, size_t& at) {
        if (plan_bytes > std::numeric_limits<size_t>::max() - align) return false;
        const size_t start = (plan_bytes + align - 1) & ~(align - 1);
        if (bytes > std::numeric_limits<size_t>::max() - start) return false;
        at = start;
        plan_bytes = start + bytes;
        return true;
    };
    if (!append(groups * sizeof(uint64_t), alignof(uint64_t), blob_off) ||
        !append(groups * sizeof(int32_t), alignof(int32_t), format_off) ||
        !append((groups + 1) * sizeof(int32_t), alignof(int32_t), start_off) ||
        !append(entries * sizeof(int32_t), alignof(int32_t), tok_off) ||
        !append(entries * sizeof(int32_t), alignof(int32_t), dst_off) ||
        !append(sizeof(int32_t), alignof(int32_t), ngroups_off) ||
        !append(sizeof(int32_t), alignof(int32_t), nentries_off)) {
        err = "exl3 runtime: plan arena overflows";
        return false;
    }
    h_plan_arena_ = strata::gpu::alloc_host(plan_bytes);
    if (h_plan_arena_ == nullptr) { err = "exl3 runtime: host plan arena allocation failed"; return false; }
    d_plan_arena_ = strata::gpu::alloc_device(plan_bytes);
    if (d_plan_arena_ == nullptr) { err = "exl3 runtime: device plan arena allocation failed"; return false; }
    plan_arena_bytes_ = plan_bytes;
    auto* h_base = static_cast<uint8_t*>(h_plan_arena_);
    auto* d_base = static_cast<uint8_t*>(d_plan_arena_);
    h_blob_base_ = reinterpret_cast<uint64_t*>(h_base + blob_off);
    h_format_index_ = reinterpret_cast<int32_t*>(h_base + format_off);
    h_start_ = reinterpret_cast<int32_t*>(h_base + start_off);
    h_tok_ = reinterpret_cast<int32_t*>(h_base + tok_off);
    h_dst_ = reinterpret_cast<int32_t*>(h_base + dst_off);
    h_ngroups_ = reinterpret_cast<int32_t*>(h_base + ngroups_off);
    h_nentries_ = reinterpret_cast<int32_t*>(h_base + nentries_off);
    d_blob_base_ = reinterpret_cast<uint64_t*>(d_base + blob_off);
    d_format_index_ = reinterpret_cast<int32_t*>(d_base + format_off);
    d_start_ = reinterpret_cast<int32_t*>(d_base + start_off);
    d_tok_ = reinterpret_cast<int32_t*>(d_base + tok_off);
    d_dst_ = reinterpret_cast<int32_t*>(d_base + dst_off);
    d_ngroups_ = reinterpret_cast<int32_t*>(d_base + ngroups_off);
    d_nentries_ = reinterpret_cast<int32_t*>(d_base + nentries_off);
    if (!host(reinterpret_cast<void**>(&h_format_table_), formats * sizeof(Exl3ExpertFormatDesc), "format table") ||
        !device(reinterpret_cast<void**>(&d_format_table_), formats * sizeof(Exl3ExpertFormatDesc), "format table") ||
        !device(reinterpret_cast<void**>(&d_entry_plans_),
                entries * sizeof(strata::kernels::exl3::GroupedEntryPlan), "entry resolution"))
        return false;
    {
        if (pair_count_ <= 0 || static_cast<uint64_t>(pair_count_) > std::numeric_limits<size_t>::max() / sizeof(uint64_t) ||
            static_cast<uint64_t>(pair_count_) > std::numeric_limits<size_t>::max() / sizeof(int32_t) ||
            !device(reinterpret_cast<void**>(&d_pair_blob_base_), static_cast<size_t>(pair_count_) * sizeof(uint64_t),
                    "resident pair blob table") ||
            !device(reinterpret_cast<void**>(&d_pair_format_index_), static_cast<size_t>(pair_count_) * sizeof(int32_t),
                    "resident pair format table"))
            return false;
    }
    std::memset(h_plan_arena_, 0, plan_arena_bytes_);
    std::memset(h_format_table_, 0, formats * sizeof(Exl3ExpertFormatDesc));
    *h_ngroups_ = 0; *h_nentries_ = 0;
    std::memcpy(h_format_table_, format_table_host_.data(), formats * sizeof(Exl3ExpertFormatDesc));

    device_plan_.blob_base = d_blob_base_;
    device_plan_.format_index = d_format_index_;
    device_plan_.format_table = d_format_table_;
    device_plan_.start = d_start_;
    device_plan_.tok = d_tok_;
    device_plan_.dst = d_dst_;
    device_plan_.ngroups = d_ngroups_;
    device_plan_.nentries = d_nentries_;
    device_plan_.entry_plans = d_entry_plans_;
    device_plan_.max_groups = max_entries_;
    device_plan_.max_entries = max_entries_;
    device_plan_.max_formats = static_cast<int64_t>(formats);
    device_plan_.n_tokens = max_tokens_;
    device_plan_.output_rows = max_entries_;
    device_plan_.hidden = hidden_;
    device_plan_.max_ff = max_ff_;
    try {
        scratch_elements_ = strata::kernels::exl3::exl3_grouped_scratch_elements(device_plan_);
    } catch (const std::exception& e) {
        err = std::string("exl3 runtime: grouped scratch: ") + e.what();
        return false;
    }
    if (scratch_elements_ > std::numeric_limits<size_t>::max() / sizeof(float) ||
        !device(reinterpret_cast<void**>(&d_scratch_), scratch_elements_ * sizeof(float), "grouped scratch"))
        return false;
    device_runtime_ = Runtime::owner(d_scratch_);
    if (device_runtime_ == nullptr) {
        err = "exl3 runtime: grouped scratch has no owning runtime";
        return false;
    }
    const void* device_images[] = {d_blob_base_, d_format_index_, d_format_table_, d_start_, d_tok_, d_dst_,
                                  d_ngroups_, d_nentries_, d_entry_plans_};
    for (const void* image : device_images)
        if (Runtime::owner(image) != device_runtime_) {
            err = "exl3 runtime: plan image belongs to another runtime";
            return false;
        }
    if (Runtime::owner(d_pair_blob_base_) != device_runtime_ ||
        Runtime::owner(d_pair_format_index_) != device_runtime_) {
        err = "exl3 runtime: resident pair table belongs to another runtime";
        return false;
    }
    if (cache_.valid() && Runtime::owner(cache_.device_slot(0)) != device_runtime_) {
        err = "exl3 runtime: cache arena belongs to another runtime";
        return false;
    }
    {
        std::vector<uint64_t> pair_blobs(static_cast<size_t>(pair_count_), 0);
        std::vector<int32_t> pair_formats(static_cast<size_t>(pair_count_), -1);
        for (int64_t layer = 0; layer < n_layers_; ++layer) {
            for (int64_t expert = 0; expert < n_expert_; ++expert) {
                const size_t at = pair_index(layer, expert);
                const int32_t slot = cache_.valid() ? cache_.slot_of(layer, expert) : kNotResident;
                const uint8_t* blob = slot >= 0 ? cache_.device_slot(slot) : nullptr;
                // A pair with a cache slot must have its device address.  The coverage question -
                // whether every pair the runtime serves has a slot - is `all_resident_`'s, answered
                // from the cache's own admission before this runs; a pair of another stage's layers
                // has no slot here and no blob here, and that is not an error.
                if (blob == nullptr && slot >= 0) {
                    err = "exl3 runtime: resident pair table has a missing device slot";
                    return false;
                }
                pair_blobs[at] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(blob));
                if (blob != nullptr) pair_formats[at] = pair_format_index_[at];
            }
        }
        if (!strata::gpu::copy(d_pair_blob_base_, pair_blobs.data(), pair_blobs.size() * sizeof(uint64_t)) ||
            !strata::gpu::copy(d_pair_format_index_, pair_formats.data(), pair_formats.size() * sizeof(int32_t))) {
            err = std::string("exl3 runtime: copying resident pair tables: ") + strata::gpu::last_error();
            return false;
        }
    }
    if (!strata::gpu::copy(d_format_table_, h_format_table_, formats * sizeof(Exl3ExpertFormatDesc))) {
        err = std::string("exl3 runtime: copying format table: ") + strata::gpu::last_error();
        return false;
    }
    return true;
}

bool Exl3ExpertRuntime::open(Exl3ExpertSource& source, strata::kernels::cpu::ExpertPool& pool, int64_t max_tokens,
                             int64_t routed_k, const std::vector<ResidentPair>& resident, std::string& err,
                             int64_t prewarm_layer_begin, int64_t prewarm_layer_end) {
    close();
    err.clear();
    if (max_tokens <= 0 || routed_k <= 0 || max_tokens > std::numeric_limits<int32_t>::max() ||
        routed_k > std::numeric_limits<int32_t>::max() ||
        !product_fits(max_tokens, routed_k, std::numeric_limits<int32_t>::max(), max_entries_)) {
        err = "exl3 runtime: max_tokens and routed_k exceed the bounded plan";
        return false;
    }
    if (max_entries_ > strata::kernels::cpu::ExpertPool::kMaxTasks) {
        err = "exl3 runtime: max_tokens*routed_k exceeds the CPU pool task bound";
        return false;
    }
    source_ = &source;
    pool_ = &pool;
    n_layers_ = source.n_layers();
    n_expert_ = source.n_expert();
    max_tokens_ = max_tokens;
    routed_k_ = routed_k;
    if (n_layers_ <= 0 || n_expert_ <= 0) { err = "exl3 runtime: source geometry is invalid"; close(); return false; }
    if (prewarm_layer_end == -1) prewarm_layer_end = n_layers_;
    if (prewarm_layer_begin < 0 || prewarm_layer_begin > prewarm_layer_end || prewarm_layer_end > n_layers_) {
        err = "exl3 runtime: CPU prewarm ownership range is invalid"; close(); return false;
    }
    const uint64_t pairs = static_cast<uint64_t>(n_layers_) * static_cast<uint64_t>(n_expert_);
    if (pairs > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        pairs > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        err = "exl3 runtime: source pair table overflows"; close(); return false;
    }
    pair_count_ = static_cast<int64_t>(pairs);
    // Whether EVERY routed pair is resident is decided after the cache is filled, from the cache's own
    // admission.  The runtime serves the layers [prewarm_layer_begin, prewarm_layer_end) - the stage's own
    // range for a stage owner - and a routed expert of those layers is a CPU miss exactly when its pair has
    // no cache slot, so the served range is what the question is about: a stage whose own layers are fully
    // resident never misses even though another stage's pairs have no slot here.
    all_resident_ = false;
    pair_formats_.resize(static_cast<size_t>(pairs));
    pair_format_index_.resize(static_cast<size_t>(pairs));
    std::unordered_map<Exl3ExpertFormatID, int32_t> format_indices;
    for (int64_t layer = 0; layer < n_layers_; ++layer) {
        for (int64_t expert = 0; expert < n_expert_; ++expert) {
            const auto* layout = source.exl3_layout(layer, expert);
            if (layout == nullptr || layout->total_bytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                err = "exl3 runtime: source has no valid exact layout"; close(); return false;
            }
            std::string why;
            const Exl3ExpertFormatID id = registry_.register_format(*layout, why);
            if (id == 0) { err = "exl3 runtime: format registry: " + why; close(); return false; }
            auto [it, inserted] = format_indices.emplace(id, static_cast<int32_t>(format_table_host_.size()));
            if (inserted) format_table_host_.push_back(*layout);
            const size_t at = pair_index(layer, expert);
            pair_formats_[at] = id;
            pair_format_index_[at] = it->second;
            if (hidden_ == 0) hidden_ = layout->gate.k;
            max_ff_ = std::max(max_ff_, layout->gate.n);
        }
    }
    max_ff_ = 0;
    for (const auto& layout : format_table_host_) {
        if (layout.gate.k != hidden_ || layout.up.k != hidden_ || layout.down.n != hidden_ ||
            layout.gate.n != layout.up.n || layout.down.k != layout.gate.n) {
            err = "exl3 runtime: source formats do not share hidden-ff-hidden geometry"; close(); return false;
        }
        max_ff_ = std::max(max_ff_, layout.gate.n);
    }
    if (hidden_ <= 0 || hidden_ % 128 != 0 || max_ff_ <= 0 || max_ff_ > std::numeric_limits<int32_t>::max() ||
        format_table_host_.empty()) {
        err = "exl3 runtime: unsupported EXL3 runtime geometry"; close(); return false;
    }
    if (pool.prepare_exl3(1, hidden_, max_ff_) != strata::kernels::cpu::exl3::Status::ok) {
        err = "exl3 runtime: CPU pool EXL3 scratch preparation failed"; close(); return false;
    }
    const char* rowparallel = std::getenv("STRATA_EXL3_CPU_ROWPARALLEL");
    // The row-parallel phases are the default: they give the pool the tasks it
    // needs and split the dot along the K when a phase's tasks do not fill the
    // workers.  The values do not depend on the split (the ranges' sums are
    // exact integers), so the path is bit identical to the whole-expert one.
    // Setting the variable to 0 keeps the old behaviour.
    if (!(rowparallel != nullptr && rowparallel[0] == '0' && rowparallel[1] == '\0')) {
        // Every runtime miss entry is a one-row job. Bound the prepared job
        // count by the largest phase rather than disabling a large window:
        // a larger actual miss batch takes the existing whole-expert path.
        const uint64_t ff_blocks = (static_cast<uint64_t>(max_ff_) + 127u) / 128u;
        const uint64_t hidden_blocks = (static_cast<uint64_t>(hidden_) + 127u) / 128u;
        const uint64_t tasks_per_job = (std::max)({2u * ff_blocks, hidden_blocks, uint64_t{1}});
        const uint64_t bounded_jobs = static_cast<uint64_t>(strata::kernels::cpu::ExpertPool::kMaxTasks) / tasks_per_job;
        const int64_t row_jobs = static_cast<int64_t>((std::min)(static_cast<uint64_t>(max_entries_), bounded_jobs));
        if (row_jobs > 0) {
            const auto row_status = pool.prepare_exl3_rowparallel(1, row_jobs, hidden_, max_ff_);
            if (row_status == strata::kernels::cpu::exl3::Status::ok) {
                cpu_rowparallel_jobs_ = row_jobs;
            } else if (row_status != strata::kernels::cpu::exl3::Status::unsupported_layout) {
                err = std::string("exl3 runtime: CPU row scratch preparation: ") +
                      strata::kernels::cpu::exl3::status_string(row_status);
                close(); return false;
            }
        }
    }
    if (resident.size() > static_cast<size_t>(n_layers_) * static_cast<size_t>(n_expert_)) {
        err = "exl3 runtime: resident pair list is too large"; close(); return false;
    }
    std::vector<uint8_t> selected(static_cast<size_t>(pairs), 0);
    std::vector<ExpertCacheBucket> buckets;
    std::unordered_map<Exl3ExpertFormatID, size_t> bucket_indices;
    for (const ResidentPair& pair : resident) {
        if (pair.first < 0 || pair.first >= n_layers_ || pair.second < 0 || pair.second >= n_expert_) {
            err = "exl3 runtime: resident pair is out of range"; close(); return false;
        }
        const size_t at = pair_index(pair.first, pair.second);
        if (selected[at] != 0) { err = "exl3 runtime: resident pair is duplicated"; close(); return false; }
        selected[at] = 1;
        const Exl3ExpertFormatID id = pair_formats_[at];
        auto [it, inserted] = bucket_indices.emplace(id, buckets.size());
        if (inserted) {
            const auto& layout = format_table_host_[(size_t) pair_format_index_[at]];
            buckets.push_back(ExpertCacheBucket{id, 0, static_cast<int64_t>(layout.total_bytes)});
        }
        ++buckets[it->second].slots;
    }
    if (!buckets.empty()) {
        // The arena is one allocation unless the device's largest allocation is smaller than it: then it is
        // taken in pieces of that size, the way the plain model's tier does it.  When the one allocation is
        // refused the cache keeps the prefix the device gives in pieces (ExpertCache's best-effort fallback):
        // the resident pairs past the prefix are not admitted below and take the CPU miss path instead.
        cache_.set_max_piece(max_alloc_bytes());
        if (!cache_.open_buckets(buckets, n_layers_, n_expert_, err)) { close(); return false; }
        // The fills are QUEUED and waited for once, not one synchronous copy per expert: a cache that holds
        // thousands of experts would otherwise pay a device synchronisation per expert.  A queued copy reads the
        // host buffer until it is done, so the buffers are a small ring and the ring is waited for before a
        // buffer is reused.
        constexpr size_t kFillRing = 16;
        std::vector<std::vector<uint8_t>> ring(kFillRing);
        size_t next = 0;
        int64_t skipped = 0;   // resident pairs the best-effort prefix could not hold: CPU misses, not a failure
        for (const ResidentPair& pair : resident) {
            const size_t at = pair_index(pair.first, pair.second);
            const auto& layout = format_table_host_[(size_t) pair_format_index_[at]];
            const int32_t slot = cache_.admit(pair.first, pair.second, pair_formats_[at]);
            if (slot < 0) {
                ++skipped;
                selected[at] = 0;   // not resident after all: the CPU prewarm below must own this pair
                continue;
            }
            std::vector<uint8_t>& blob = ring[next];
            blob.resize(static_cast<size_t>(layout.total_bytes));
            if (!source.copy_blob(pair.first, pair.second, blob.data())) {
                err = "exl3 runtime: resident source copy failed"; close(); return false;
            }
            if (!cache_.fill_slot_queued(slot, blob.data(), err, static_cast<int64_t>(blob.size()))) {
                if (err.empty()) err = "exl3 runtime: resident cache admission failed";
                close(); return false;
            }
            next = (next + 1) % kFillRing;
            if (next == 0 && !cache_.sync_queued(err)) {
                if (err.empty()) err = "exl3 runtime: resident cache fill did not finish";
                close(); return false;
            }
        }
        if (next != 0 && !cache_.sync_queued(err)) {
            if (err.empty()) err = "exl3 runtime: resident cache fill did not finish";
            close(); return false;
        }
        if (skipped > 0) {
            all_resident_ = false;
            std::fprintf(stderr, "Exl3ExpertRuntime: %lld of %zu resident experts did not fit the device cache "
                                 "and take the CPU miss path\n", (long long) skipped, resident.size());
        }
    }
    cpu_jobs_.resize(static_cast<size_t>(max_entries_));
    miss_ids_.reserve(static_cast<size_t>(n_expert_));
    if (cache_.valid() && !resident.empty()) {
        all_resident_ = true;
        for (int64_t layer = prewarm_layer_begin; layer < prewarm_layer_end && all_resident_; ++layer)
            for (int64_t expert = 0; expert < n_expert_; ++expert)
                if (cache_.slot_of(layer, expert) < 0) { all_resident_ = false; break; }
    }
    // The Direct/Split miss path's pinned staging: one canonical blob per distinct miss of a window,
    // at the largest format's stride.  The GPU reads these host addresses over the bus; no VRAM is
    // spent on misses.  The arena covers one window only - a few MiB for a decode window - not the
    // whole model's misses: the source's CPU pool already holds those under its host-memory budget.
    // A staging that cannot be allocated, or host memory the kernels cannot read, leaves the CPU
    // miss path in charge and says so once here.  Split additionally times a pinned->device copy
    // through this arena at open, so its per-window share divides measured PCIe bytes/s, not a
    // configured number.
    if (miss_mode_ == MissMode::Direct || miss_mode_ == MissMode::Split) {
        uint64_t stride = 0;
        for (const auto& layout : format_table_host_) stride = std::max(stride, layout.total_bytes);
        direct_stride_ = (stride + 255u) & ~uint64_t(255u);
        direct_capacity_ = std::min<int64_t>(n_expert_, max_entries_);
        const uint64_t bytes = direct_stride_ * static_cast<uint64_t>(direct_capacity_);
        if (direct_stride_ > 0 && direct_capacity_ > 0 &&
            bytes <= static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            direct_staging_ = static_cast<uint8_t*>(strata::gpu::alloc_host(static_cast<size_t>(bytes)));
        }
        direct_ready_ = direct_staging_ != nullptr && strata::gpu::kernels_read_host(direct_staging_);
        if (!direct_ready_) {
            if (direct_staging_ != nullptr) { strata::gpu::free(direct_staging_); direct_staging_ = nullptr; }
            std::fprintf(stderr, "Exl3ExpertRuntime: direct miss read unavailable (pinned staging or host "
                                 "access failed); CPU misses stay in charge\n");
        } else if (miss_mode_ == MissMode::Split) {
            measure_pcie_bandwidth();
            std::fprintf(stderr,
                         "Exl3ExpertRuntime: split miss staging %lld experts x %llu bytes, PCIe %.2f GB/s\n",
                         (long long) direct_capacity_, (unsigned long long) direct_stride_,
                         pcie_bytes_per_sec_ / 1e9);
        }
    }
    if (!allocate_images(err)) { close(); return false; }
    // selected already excludes GPU residents. Do not preload another stage's
    // layer just because its pairs are absent from this stage's GPU cache.
    for (int64_t layer = 0; layer < n_layers_; ++layer)
        if (layer < prewarm_layer_begin || layer >= prewarm_layer_end)
            std::fill_n(selected.begin() + static_cast<ptrdiff_t>(pair_index(layer, 0)),
                        static_cast<size_t>(n_expert_), uint8_t{1});
    Exl3ExpertSource::PrewarmResult prewarm;
    std::string prewarm_why;
    const bool prewarmed = source.prewarm_cpu(selected, prewarm, prewarm_why);
    std::fprintf(stderr,
                 "Exl3ExpertRuntime: CPU source prewarm=%s layers=[%lld,%lld) experts=%llu bytes=%llu "
                 "admission=%llu loaded_experts=%llu loaded_bytes=%llu%s%s\n",
                 prewarmed ? "ready" : "lazy", (long long) prewarm_layer_begin, (long long) prewarm_layer_end,
                 (unsigned long long) prewarm.requested_experts, (unsigned long long) prewarm.requested_bytes,
                 (unsigned long long) prewarm.admission_bytes, (unsigned long long) prewarm.loaded_experts,
                 (unsigned long long) prewarm.loaded_bytes, prewarm_why.empty() ? "" : ": ", prewarm_why.c_str());
    // Split's persistent pinned cache comes last: the pool owns its misses by now, so the host
    // reading sizes the cache from what is left.  It never fails the open (0 slots keeps the
    // per-window staging).
    if (miss_mode_ == MissMode::Split && direct_ready_) open_split_cache();
    opened_ = true;
    return true;
}

bool Exl3ExpertRuntime::prepare_window(int64_t layer, const float* host_fp32_x, int64_t ldx,
                                       const int32_t* routed_ids, int64_t n_tokens, float* cpu_output, int64_t ldy,
                                       std::string& err) {
    const HostTimelineScope timeline("prepare", layer);
    detail::Exl3PrepareSample detail_sample;
    err.clear();
    if (!opened_ || source_ == nullptr || pool_ == nullptr) { err = "exl3 runtime: open is required"; return false; }
    if (pending_gpu_) { err = "exl3 runtime: wait_gpu must consume the previous plan first"; return false; }
    if (prepared_) { err = "exl3 runtime: previous plan was not run and enqueued"; return false; }
    if (!valid_layer(layer) || n_tokens <= 0 || n_tokens > max_tokens_ || host_fp32_x == nullptr || routed_ids == nullptr ||
        cpu_output == nullptr || !span_fits(n_tokens, ldx, hidden_) || !span_fits(max_entries_, ldy, hidden_)) {
        err = "exl3 runtime: invalid window spans or geometry";
        return false;
    }
    *h_ngroups_ = 0; *h_nentries_ = 0; h_start_[0] = 0;
    miss_entries_ = 0;
    hit_entries_ = 0;
    direct_entries_ = 0;
    miss_ids_.clear();
    if (record_heat_ && heat_.size() != static_cast<size_t>(pair_count_))
        heat_.assign(static_cast<size_t>(pair_count_), 0);
    std::vector<uint8_t> miss_seen(static_cast<size_t>(n_expert_), 0);
    std::vector<int32_t> group_slots;
    std::vector<std::vector<int32_t>> group_entries;
    std::unordered_map<int32_t, int32_t> groups_by_slot;
    std::vector<int32_t> miss_group_experts;
    std::vector<std::vector<int32_t>> miss_group_entries;
    std::unordered_map<int32_t, int32_t> miss_groups_by_expert;
    for (int64_t entry = 0; entry < n_tokens * routed_k_; ++entry) {
        const int32_t expert = routed_ids[entry];
        if (expert < 0 || expert >= n_expert_) { err = "exl3 runtime: routed expert is out of range"; return false; }
        const size_t at = pair_index(layer, expert);
        if (record_heat_) ++heat_[at];
        const int32_t slot = cache_.valid() ? cache_.slot_of(layer, expert) : kNotResident;
        if (slot >= 0) {
            ++hit_entries_;
            std::fill(cpu_output + entry * ldy, cpu_output + entry * ldy + ldy, 0.0f);
            auto [it, inserted] = groups_by_slot.emplace(slot, static_cast<int32_t>(group_slots.size()));
            if (inserted) { group_slots.push_back(slot); group_entries.emplace_back(); }
            group_entries[(size_t) it->second].push_back(static_cast<int32_t>(entry));
        } else {
            if (miss_seen[(size_t) expert] == 0) {
                miss_seen[(size_t) expert] = 1;
                miss_ids_.push_back(expert);
            }
            auto [it, inserted] = miss_groups_by_expert.emplace(expert, static_cast<int32_t>(miss_group_experts.size()));
            if (inserted) { miss_group_experts.push_back(expert); miss_group_entries.emplace_back(); }
            miss_group_entries[(size_t) it->second].push_back(static_cast<int32_t>(entry));
        }
    }
    detail_sample.mark(0);   // route partition, including temporary workspace
    // Direct: the GPU computes the misses from pinned host memory, so no CPU job and no window pin of
    // the source is needed - copy_blob hands out the canonical order the device kernels read.  The
    // staging holds one blob per distinct miss, which the capacity bounds.
    const bool direct_active = miss_mode_ == MissMode::Direct && direct_ready_ &&
                               static_cast<int64_t>(miss_group_experts.size()) <= direct_capacity_;
    // Split: the GPU takes some distinct misses (staged like Direct, in the same grouped launch as
    // the hits) and the CPU pool takes the rest, side by side.  Which experts go to the GPU is
    // decided below from measured costs: experts with the most routed entries first (one bus read
    // serves every entry that needs the expert, while the CPU pays per entry), ties broken by the
    // smaller blob (a cheaper bus read).  The count minimizes max(CPU, GPU) through plan_split_gpu;
    // it is capped to what the pinned staging holds, so a window larger than the staging still
    // moves the staged prefix instead of falling back whole.
    bool split_active = false;
    std::vector<uint8_t> split_is_gpu;
    std::vector<size_t> split_staged_order;
    std::vector<uint64_t> split_gpu_addr;
    int64_t split_gpu_entries = 0;
    if (!direct_active && miss_mode_ == MissMode::Split && direct_ready_ &&
        !miss_group_experts.empty()) {
        std::vector<SplitCandidate> candidates;
        candidates.reserve(miss_group_experts.size());
        for (size_t group = 0; group < miss_group_experts.size(); ++group) {
            const auto* layout = source_->exl3_layout(layer, miss_group_experts[group]);
            candidates.push_back({miss_group_experts[group],
                                  static_cast<int64_t>(miss_group_entries[group].size()),
                                  layout != nullptr ? layout->total_bytes : 0});
        }
        std::vector<size_t> order(candidates.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (candidates[a].entries != candidates[b].entries)
                return candidates[a].entries > candidates[b].entries;
            return candidates[a].bytes < candidates[b].bytes;
        });
        std::vector<SplitCandidate> sorted;
        sorted.reserve(candidates.size());
        for (size_t i : order) sorted.push_back(candidates[i]);
        // A cached window is capped by the cache's slots (evicting a slot the same window uses
        // is then impossible: at most `slots` assignments reuse them); an uncached one by the
        // per-window staging.
        const int64_t gpu_capacity = split_cache_slots_ > 0 ? split_cache_slots_ : direct_capacity_;
        const int64_t k =
            plan_split_gpu(sorted.data(), static_cast<int64_t>(sorted.size()), cpu_us_per_entry_,
                           pcie_bytes_per_sec_, gpu_capacity);
        if (k > 0) {
            split_is_gpu.assign(miss_group_experts.size(), 0);
            split_staged_order.reserve(static_cast<size_t>(k));
            split_gpu_addr.reserve(static_cast<size_t>(k));
            for (int64_t s = 0; s < k; ++s) {
                const size_t at = order[static_cast<size_t>(s)];
                const int32_t expert = miss_group_experts[at];
                uint64_t addr = 0;
                if (split_cache_slots_ > 0) {
                    // The persistent cache: a staged expert is reread on later tokens, so this
                    // window's host conversion is paid only on a cache miss.  A blob that cannot
                    // be staged stays on the CPU; its CPU read below reports the failure, if any.
                    const int64_t slot = split_cache_assign(layer, expert);
                    if (slot < 0) continue;
                    addr = static_cast<uint64_t>(
                        reinterpret_cast<uintptr_t>(split_cache_ + static_cast<size_t>(slot) * direct_stride_));
                } else {
                    uint8_t* staged = direct_staging_ + split_staged_order.size() * direct_stride_;
                    if (!source_->copy_blob(layer, expert, staged)) {
                        err = "exl3 runtime: split miss source copy failed";
                        return false;
                    }
                    addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(staged));
                }
                split_is_gpu[at] = 1;
                split_staged_order.push_back(at);
                split_gpu_addr.push_back(addr);
                split_gpu_entries += static_cast<int64_t>(miss_group_entries[at].size());
            }
            split_active = !split_staged_order.empty();
        }
    }
    if (direct_active) {
        for (size_t group = 0; group < miss_group_experts.size(); ++group) {
            uint8_t* staged = direct_staging_ + group * direct_stride_;
            if (!source_->copy_blob(layer, miss_group_experts[group], staged)) {
                err = "exl3 runtime: direct miss source copy failed";
                return false;
            }
        }
        for (const auto& entries : miss_group_entries)
            for (const int32_t entry : entries) {
                std::fill(cpu_output + entry * ldy, cpu_output + entry * ldy + ldy, 0.0f);
                ++direct_entries_;
            }
    } else if (split_active) {
        // The CPU reads only its own share below (begin_layer pins that share); the GPU share's
        // rows are zeroes the grouped kernel overwrites, so the merge's halves stay disjoint.
        std::vector<int32_t> cpu_ids;
        cpu_ids.reserve(miss_ids_.size());
        for (size_t group = 0; group < miss_group_experts.size(); ++group) {
            if (split_is_gpu[group] != 0) {
                for (const int32_t entry : miss_group_entries[group]) {
                    std::fill(cpu_output + entry * ldy, cpu_output + entry * ldy + ldy, 0.0f);
                    ++direct_entries_;
                }
            } else {
                cpu_ids.push_back(miss_group_experts[group]);
            }
        }
        if (!cpu_ids.empty()) {
            source_->begin_layer(layer, cpu_ids.data(), static_cast<int64_t>(cpu_ids.size()));
            if (!source_->last_error().empty()) { err = source_->last_error(); return false; }
        }
    } else if (!miss_ids_.empty()) {
        source_->begin_layer(layer, miss_ids_.data(), static_cast<int64_t>(miss_ids_.size()));
        if (!source_->last_error().empty()) { err = source_->last_error(); return false; }
    }
    detail_sample.mark(1);   // source begin (guard time is a nested component)
    if (!direct_active && !split_active)
    for (int64_t entry = 0; entry < n_tokens * routed_k_; ++entry) {
        const int32_t expert = routed_ids[entry];
        if (cache_.valid() && cache_.slot_of(layer, expert) >= 0) continue;
        const uint8_t* blob = source_->blob(layer, expert);
        const auto* layout = source_->exl3_layout(layer, expert);
        if (blob == nullptr || layout == nullptr) { err = "exl3 runtime: CPU miss source read failed"; return false; }
        auto& job = cpu_jobs_[(size_t) miss_entries_++];
        job.view = strata::kernels::exl3::expert_view(*layout, blob, source_->exl3_tile_order(layer, expert));
        job.rows = 1;
        job.x = host_fp32_x + (entry / routed_k_) * ldx;
        job.ldx = ldx;
        job.y = cpu_output + entry * ldy;
        job.ldy = ldy;
        job.slot = static_cast<int>(entry);
    }
    if (split_active)
    for (int64_t entry = 0; entry < n_tokens * routed_k_; ++entry) {
        const int32_t expert = routed_ids[entry];
        if (cache_.valid() && cache_.slot_of(layer, expert) >= 0) continue;
        auto it = miss_groups_by_expert.find(expert);
        if (it == miss_groups_by_expert.end() || split_is_gpu[(size_t) it->second] != 0) continue;
        const uint8_t* blob = source_->blob(layer, expert);
        const auto* layout = source_->exl3_layout(layer, expert);
        if (blob == nullptr || layout == nullptr) { err = "exl3 runtime: CPU miss source read failed"; return false; }
        auto& job = cpu_jobs_[(size_t) miss_entries_++];
        job.view = strata::kernels::exl3::expert_view(*layout, blob, source_->exl3_tile_order(layer, expert));
        job.rows = 1;
        job.x = host_fp32_x + (entry / routed_k_) * ldx;
        job.ldx = ldx;
        job.y = cpu_output + entry * ldy;
        job.ldy = ldy;
        job.slot = static_cast<int>(entry);
    }
    detail_sample.mark(2);   // borrowed source pointers and CPU job views
    int64_t cursor = 0;
    for (size_t group = 0; group < group_slots.size(); ++group) {
        const int32_t slot = group_slots[group];
        const uint8_t* device_blob = cache_.device_slot(slot);
        if (device_blob == nullptr) { err = "exl3 runtime: cache slot address is invalid"; return false; }
        const int32_t entry0 = group_entries[group].empty() ? 0 : group_entries[group][0];
        const int32_t expert = routed_ids[entry0];
        const size_t at = pair_index(layer, expert);
        h_blob_base_[group] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(device_blob));
        h_format_index_[group] = pair_format_index_[at];
        for (int32_t entry : group_entries[group]) {
            h_tok_[cursor] = entry / static_cast<int32_t>(routed_k_);
            h_dst_[cursor] = entry;
            ++cursor;
        }
        h_start_[group + 1] = static_cast<int32_t>(cursor);
    }
    if (direct_active || split_active) {
        // Miss groups follow the hit groups in the same plan: their blob address is the pinned host
        // staging, which host USM makes the device address too, so the grouped kernel reads the miss
        // weights over the bus while it reads the hits from VRAM.  Direct stages every miss in
        // routing order; Split stages its chosen share - from the persistent cache when it has one,
        // else converted anew above - in the staged order.
        auto append_gpu_group = [&](size_t miss, size_t staged_at, uint64_t blob) {
            const size_t group = group_slots.size() + staged_at;
            const size_t at = pair_index(layer, miss_group_experts[miss]);
            h_blob_base_[group] = blob;
            h_format_index_[group] = pair_format_index_[at];
            for (int32_t entry : miss_group_entries[miss]) {
                h_tok_[cursor] = entry / static_cast<int32_t>(routed_k_);
                h_dst_[cursor] = entry;
                ++cursor;
            }
            h_start_[group + 1] = static_cast<int32_t>(cursor);
        };
        if (direct_active) {
            for (size_t miss = 0; miss < miss_group_experts.size(); ++miss)
                append_gpu_group(miss, miss,
                                 static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
                                     direct_staging_ + miss * direct_stride_)));
        } else {
            for (size_t s = 0; s < split_staged_order.size(); ++s)
                append_gpu_group(split_staged_order[s], s, split_gpu_addr[s]);
        }
    }
    const size_t gpu_groups = direct_active ? miss_group_experts.size()
                                            : (split_active ? split_staged_order.size() : 0);
    *h_ngroups_ = static_cast<int32_t>(group_slots.size() + gpu_groups);
    *h_nentries_ = static_cast<int32_t>(cursor);
    total_hit_entries_ += hit_entries_;
    total_cpu_miss_entries_ += miss_entries_;
    total_direct_miss_entries_ += direct_entries_;
    if (miss_mode_ == MissMode::Split && !miss_group_experts.empty()) {
        // A Split window with no miss still records nothing; every miss window tunes the share,
        // including an all-CPU one (k = 0 while the controller has no costs yet).
        ++split_windows_;
        split_last_gpu_ = static_cast<int64_t>(split_staged_order.size());
        split_last_cpu_ = static_cast<int64_t>(miss_group_experts.size()) - split_last_gpu_;
        split_gpu_entries_ += split_gpu_entries;
        split_miss_entries_ += split_gpu_entries + miss_entries_;
    }
    prepared_layer_ = layer;
    prepared_ = true;
    cpu_ready_ = false;
    detail_sample.mark(3);   // host grouped hit plan
    detail_sample.success();
    return true;
}

bool Exl3ExpertRuntime::run_cpu(std::string& err) {
    const HostTimelineScope timeline("cpu", prepared_layer_, hit_entries_, miss_entries_);
    err.clear();
    if (!prepared_) { err = "exl3 runtime: prepare_window is required"; return false; }
    if (cpu_ready_) return true;
    using CpuStatus = strata::kernels::cpu::exl3::Status;
    const auto cpu_t0 = std::chrono::steady_clock::now();
    CpuStatus status;
    if (cpu_rowparallel_jobs_ > 0 && miss_entries_ <= cpu_rowparallel_jobs_) {
        status = pool_->run_exl3_rowparallel(cpu_jobs_.data(), static_cast<int>(miss_entries_));
        // The packed phase API validates raw/unsupported geometry and task
        // bounds before publishing or writing output. Other errors propagate.
        if (status == CpuStatus::unsupported_layout)
            status = pool_->run_exl3(cpu_jobs_.data(), static_cast<int>(miss_entries_));
    } else {
        status = pool_->run_exl3(cpu_jobs_.data(), static_cast<int>(miss_entries_));
    }
    if (status != strata::kernels::cpu::exl3::Status::ok) {
        err = std::string("exl3 runtime: CPU miss run: ") + strata::kernels::cpu::exl3::status_string(status);
        return false;
    }
    cpu_ready_ = true;
    if (miss_mode_ == MissMode::Split && miss_entries_ > 0) {
        // The split controller's CPU cost: this window's pool time per CPU miss entry, folded
        // into an EWMA so one slow layer does not swing the next window's share.
        const double us =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - cpu_t0).count();
        const double sample = us / static_cast<double>(miss_entries_);
        cpu_us_per_entry_ = cpu_us_per_entry_ > 0.0 ? 0.75 * cpu_us_per_entry_ + 0.25 * sample : sample;
    }
    return true;
}

bool Exl3ExpertRuntime::copy_plan(sycl::queue& queue, const std::vector<sycl::event>& deps,
                                  std::vector<sycl::event>& copies, std::string& err) {
    // One memcpy uploads the whole window plan: the host and device arenas share the layout allocate_images
    // carved, so the seven arrays travel in a single submit instead of seven.  Events from other streams still
    // order this copy explicitly; work already on this queue needs no event, in-order or captured.
    try {
        copies.push_back(queue.submit([&](sycl::handler& h) {
            if (!deps.empty()) h.depends_on(deps);
            h.memcpy(d_plan_arena_, h_plan_arena_, plan_arena_bytes_);
        }));
        return true;
    } catch (const std::exception& e) {
        err = std::string("exl3 runtime: plan copy: ") + e.what();
        return false;
    }
}

bool Exl3ExpertRuntime::enqueue(sycl::queue& queue, const float* device_fp32_x, int64_t ldx, float* gpu_output,
                                int64_t ldy, sycl::event& done, const std::vector<sycl::event>& deps,
                                std::string& err) {
    err.clear();
    return enqueue_impl(queue, device_fp32_x, ldx, gpu_output, ldy, done, deps, true, err);
}

bool Exl3ExpertRuntime::enqueue_hit_only(sycl::queue& queue, const float* device_fp32_x, int64_t ldx,
                                         float* gpu_output, int64_t ldy, sycl::event& done,
                                         const std::vector<sycl::event>& deps, std::string& err) {
    err.clear();
    return enqueue_impl(queue, device_fp32_x, ldx, gpu_output, ldy, done, deps, false, err);
}

bool Exl3ExpertRuntime::enqueue_impl(sycl::queue& queue, const float* device_fp32_x, int64_t ldx, float* gpu_output,
                                     int64_t ldy, sycl::event& done, const std::vector<sycl::event>& deps,
                                     bool require_cpu, std::string& err) {
    const HostTimelineScope timeline("gpu_submit", prepared_layer_, hit_entries_, miss_entries_);
    if (pending_gpu_) {
        err = "exl3 runtime: previous GPU plan must be consumed before enqueue";
        return false;
    }
    if (!prepared_ || (require_cpu && !cpu_ready_)) {
        err = require_cpu ? "exl3 runtime: run_cpu must consume the prepared window first"
                          : "exl3 runtime: prepare_window must precede hit enqueue";
        return false;
    }
    Runtime* queue_runtime = Runtime::owner_of_stream(&queue);
    if (queue_runtime == nullptr || queue_runtime != device_runtime_) {
        err = "exl3 runtime: enqueue queue belongs to another runtime/device";
        return false;
    }
    if (!queue.has_property<sycl::property::queue::in_order>()) {
        err = "exl3 runtime: enqueue requires an in-order runtime queue";
        return false;
    }
    if (device_fp32_x == nullptr || gpu_output == nullptr || !span_fits(max_tokens_, ldx, hidden_) ||
        !span_fits(max_entries_, ldy, hidden_)) {
        err = "exl3 runtime: invalid device activation/output spans";
        return false;
    }
    if (Runtime::owner(device_fp32_x) != queue_runtime || Runtime::owner(gpu_output) != queue_runtime) {
        err = "exl3 runtime: activation/output belongs to another runtime/device";
        return false;
    }
    std::vector<sycl::event> copies;
    if (!copy_plan(queue, deps, copies, err)) {
        try { queue_runtime->finish(queue); } catch (...) {}
        return false;
    }
    // No event dependency on the plan copy: this queue is verified in-order above, and copy_plan ordered the copy
    // after `deps` already, so the grouped kernels submitted next run after it transitively.  That keeps the single
    // copy submit - not the event edge - as the only added launch on this path.
    try {
        done = strata::kernels::exl3::exl3_grouped_fp32(queue, device_plan_, device_fp32_x, ldx, gpu_output, ldy,
                                                        d_scratch_, scratch_elements_, {}, {});
    } catch (const std::exception& e) {
        // copy_plan may already have submitted work. Drain this queue before a caller can tear down the cache or
        // return an error to a stage owner; the normal success path remains fully asynchronous.
        try { queue_runtime->finish(queue); } catch (...) {}
        err = std::string("exl3 runtime: grouped enqueue: ") + e.what();
        return false;
    }
    pending_gpu_ = true;
    return true;
}

bool Exl3ExpertRuntime::record_enqueue(sycl::queue& queue, const float* device_fp32_x, int64_t ldx,
                                       float* gpu_output, int64_t ldy, std::string& err) {
    err.clear();
    if (!opened_) { err = "exl3 runtime: open is required"; return false; }
    Runtime* queue_runtime = Runtime::owner_of_stream(&queue);
    if (queue_runtime == nullptr || queue_runtime != device_runtime_) {
        err = "exl3 runtime: graph queue belongs to another runtime/device";
        return false;
    }
    if (device_fp32_x == nullptr || gpu_output == nullptr || !span_fits(max_tokens_, ldx, hidden_) ||
        !span_fits(max_entries_, ldy, hidden_)) {
        err = "exl3 runtime: invalid graph activation/output spans";
        return false;
    }
    if (Runtime::owner(device_fp32_x) != queue_runtime || Runtime::owner(gpu_output) != queue_runtime) {
        err = "exl3 runtime: graph activation/output belongs to another runtime/device";
        return false;
    }
    std::vector<sycl::event> copies;
    if (!copy_plan(queue, {}, copies, err)) return false;
    try {
        (void) strata::kernels::exl3::exl3_grouped_fp32(queue, device_plan_, device_fp32_x, ldx, gpu_output, ldy,
                                                        d_scratch_, scratch_elements_, {}, copies);
    } catch (const std::exception& e) {
        err = std::string("exl3 runtime: graph grouped enqueue: ") + e.what();
        return false;
    }
    return true;
}

bool Exl3ExpertRuntime::record_resident_enqueue(sycl::queue& queue, int64_t layer, const float* device_fp32_x,
                                                const int32_t* device_ids, int64_t n_tokens, float* gpu_output,
                                                int64_t ldy, std::string& err) {
    err.clear();
    if (!opened_ || !all_resident_) {
        err = "exl3 runtime: the direct resident plan requires a fully resident runtime";
        return false;
    }
    return record_device_hit_enqueue(queue, layer, device_fp32_x, device_ids, n_tokens, gpu_output, ldy, err);
}

bool Exl3ExpertRuntime::submit_device_hit_plan(sycl::queue& queue, int64_t layer, const int32_t* device_ids,
                                              int64_t n_tokens, const std::vector<sycl::event>& deps,
                                              sycl::event& plan, std::string& err) {
    int64_t entries = 0;
    if (!product_fits(n_tokens, routed_k_, max_entries_, entries) || entries <= 0) {
        err = "exl3 runtime: device hit route count is out of range";
        return false;
    }
    try {
        // Each route is one group.  This intentionally avoids a device compaction/radix pass: duplicate experts
        // remain separate route groups, and the existing grouped kernel already guarantees unique destinations.
        uint64_t* const plan_blob = d_blob_base_;
        int32_t* const plan_format = d_format_index_;
        int32_t* const plan_start = d_start_;
        int32_t* const plan_tok = d_tok_;
        int32_t* const plan_dst = d_dst_;
        int32_t* const plan_groups = d_ngroups_;
        int32_t* const plan_entries = d_nentries_;
        const uint64_t* const pair_blob = d_pair_blob_base_;
        const int32_t* const pair_format = d_pair_format_index_;
        const int64_t experts = n_expert_;
        const int64_t route_k = routed_k_;
        const int64_t layer_id = layer;
        plan = queue.submit([&](sycl::handler& h) {
            if (!deps.empty()) h.depends_on(deps);
            h.parallel_for(sycl::range<1>(static_cast<size_t>(entries)), [=](sycl::id<1> id) {
                const int32_t entry = static_cast<int32_t>(id[0]);
                if (entry == 0) {
                    plan_start[0] = 0;
                    *plan_groups = static_cast<int32_t>(entries);
                    *plan_entries = static_cast<int32_t>(entries);
                }
                plan_start[entry + 1] = entry + 1;
                plan_tok[entry] = entry / static_cast<int32_t>(route_k);
                plan_dst[entry] = entry;
                const int32_t expert = device_ids[entry];
                const bool valid = expert >= 0 && expert < experts;
                if (!valid) {
                    plan_blob[entry] = 0;
                    plan_format[entry] = -1;
                    // Keep the destination valid so the grouped guard poisons this row with NaNs.
                    return;
                }
                const size_t pair = static_cast<size_t>(layer_id) * static_cast<size_t>(experts) +
                                    static_cast<size_t>(expert);
                plan_blob[entry] = pair_blob[pair];
                plan_format[entry] = pair_format[pair];
                // A valid CPU miss is absent from the GPU output. A valid destination with an invalid blob would
                // poison its row with NaNs; keep that sentinel only for an invalid routed expert above.
                if (pair_blob[pair] == 0) plan_dst[entry] = -1;
            });
        });
        return true;
    } catch (const std::exception& e) {
        err = std::string("exl3 runtime: device hit plan: ") + e.what();
        return false;
    }
}

bool Exl3ExpertRuntime::record_device_hit_enqueue(sycl::queue& queue, int64_t layer, const float* device_fp32_x,
                                                  const int32_t* device_ids, int64_t n_tokens, float* gpu_output,
                                                  int64_t ldy, std::string& err) {
    err.clear();
    if (!opened_) { err = "exl3 runtime: open is required for the device hit plan"; return false; }
    if (!valid_layer(layer) || n_tokens <= 0 || n_tokens > max_tokens_ || device_fp32_x == nullptr ||
        device_ids == nullptr || gpu_output == nullptr || !span_fits(n_tokens, hidden_, hidden_) ||
        !span_fits(max_entries_, ldy, hidden_)) {
        err = "exl3 runtime: invalid device hit plan geometry";
        return false;
    }
    if (!queue.has_property<sycl::property::queue::in_order>()) {
        err = "exl3 runtime: device hit replay requires an in-order queue";
        return false;
    }
    Runtime* queue_runtime = Runtime::owner_of_stream(&queue);
    if (queue_runtime == nullptr || queue_runtime != device_runtime_ || Runtime::owner(device_fp32_x) != queue_runtime ||
        Runtime::owner(device_ids) != queue_runtime || Runtime::owner(gpu_output) != queue_runtime) {
        err = "exl3 runtime: device hit pointers belong to another runtime/device";
        return false;
    }
    try {
        sycl::event plan;
        if (!submit_device_hit_plan(queue, layer, device_ids, n_tokens, {}, plan, err)) return false;
        (void) strata::kernels::exl3::exl3_grouped_fp32(queue, device_plan_, device_fp32_x, hidden_, gpu_output, ldy,
                                                        d_scratch_, scratch_elements_, {}, {plan});
    } catch (const std::exception& e) {
        err = std::string("exl3 runtime: device hit grouped enqueue: ") + e.what();
        return false;
    }
    return true;
}

bool Exl3ExpertRuntime::enqueue_device_hit(sycl::queue& queue, int64_t layer, const float* device_fp32_x,
                                           const int32_t* device_ids, int64_t n_tokens, float* gpu_output,
                                           int64_t ldy, sycl::event& done, const std::vector<sycl::event>& deps,
                                           std::string& err) {
    err.clear();
    if (!opened_) { err = "exl3 runtime: open is required for the device hit plan"; return false; }
    if (!valid_layer(layer) || n_tokens <= 0 || n_tokens > max_tokens_ || device_fp32_x == nullptr ||
        device_ids == nullptr || gpu_output == nullptr || !span_fits(n_tokens, hidden_, hidden_) ||
        !span_fits(max_entries_, ldy, hidden_)) {
        err = "exl3 runtime: invalid device hit plan geometry";
        return false;
    }
    if (!queue.has_property<sycl::property::queue::in_order>()) {
        err = "exl3 runtime: device hit enqueue requires an in-order queue";
        return false;
    }
    Runtime* queue_runtime = Runtime::owner_of_stream(&queue);
    if (queue_runtime == nullptr || queue_runtime != device_runtime_ || Runtime::owner(device_fp32_x) != queue_runtime ||
        Runtime::owner(device_ids) != queue_runtime || Runtime::owner(gpu_output) != queue_runtime) {
        err = "exl3 runtime: device hit pointers belong to another runtime/device";
        return false;
    }
    try {
        sycl::event plan;
        if (!submit_device_hit_plan(queue, layer, device_ids, n_tokens, deps, plan, err)) return false;
        done = strata::kernels::exl3::exl3_grouped_fp32(queue, device_plan_, device_fp32_x, hidden_, gpu_output, ldy,
                                                        d_scratch_, scratch_elements_, {}, {plan});
        return true;
    } catch (const std::exception& e) {
        err = std::string("exl3 runtime: device hit enqueue: ") + e.what();
        return false;
    }
}

bool Exl3ExpertRuntime::publish_window_plan(GpuPlanSink& sink, std::string& err) const {
    err.clear();
    if (!opened_ || !prepared_) { err = "exl3 runtime: prepare_window is required before plan publish"; return false; }
    if (sink.counts == nullptr || sink.start == nullptr || sink.dst == nullptr || sink.tok == nullptr ||
        sink.start2 == nullptr || sink.cap < max_entries_) {
        err = "exl3 runtime: legacy window sink is too small or incomplete";
        return false;
    }
    const int32_t entries = *h_nentries_;
    if (entries < 0 || entries > max_entries_) { err = "exl3 runtime: hit plan entry count is invalid"; return false; }
    sink.counts[0] = 0;
    sink.counts[1] = entries;
    sink.counts[2] = 0;
    sink.start[0] = 0;
    sink.start2[0] = 0;
    for (int32_t i = 0; i < entries; ++i) {
        sink.dst[i] = h_dst_[i];
        sink.tok[i] = h_tok_[i];
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (sink.before_cpu != nullptr && !sink.before_cpu(sink.ctx, err)) return false;
    if (sink.publish != nullptr) sink.publish(sink.ctx);
    // The grouped kernel computes the published rows (hits plus Direct/Split GPU-read misses);
    // the CPU share stays in the sink's zeroed rows for the caller's merge.  Publish completion
    // of the empty DMA share through the same window protocol as legacy plans.
    if (sink.fetch != nullptr) sink.fetch(sink.ctx, nullptr, 0, 0);
    return true;
}

bool Exl3ExpertRuntime::wait_gpu(sycl::event& done, std::string& err) {
    err.clear();
    if (!pending_gpu_) { err = "exl3 runtime: no pending GPU plan"; return false; }
    try {
        done.wait_and_throw();
    } catch (const std::exception& e) {
        err = std::string("exl3 runtime: GPU plan: ") + e.what();
        pending_gpu_ = false;
        prepared_ = false;
        cpu_ready_ = false;
        return false;
    }
    pending_gpu_ = false;
    prepared_ = false;
    cpu_ready_ = false;
    prepared_layer_ = -1;
    return true;
}

void Exl3ExpertRuntime::mark_gpu_consumed() {
    pending_gpu_ = false;
    prepared_ = false;
    cpu_ready_ = false;
    prepared_layer_ = -1;
}

}  // namespace strata::core
