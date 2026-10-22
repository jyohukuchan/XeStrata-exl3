// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Bounded EXL3 source/cache/CPU/GPU plan integration.
#include "strata/core/exl3_expert_runtime.hpp"
#include "strata/core/exl3_prepare_profile.hpp"

#include "strata/core/gpu.hpp"
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
    if (d_scratch_ != nullptr) strata::gpu::free(d_scratch_);
    if (d_blob_base_ != nullptr) strata::gpu::free(d_blob_base_);
    if (d_format_index_ != nullptr) strata::gpu::free(d_format_index_);
    if (d_format_table_ != nullptr) strata::gpu::free(d_format_table_);
    if (d_start_ != nullptr) strata::gpu::free(d_start_);
    if (d_tok_ != nullptr) strata::gpu::free(d_tok_);
    if (d_dst_ != nullptr) strata::gpu::free(d_dst_);
    if (d_ngroups_ != nullptr) strata::gpu::free(d_ngroups_);
    if (d_nentries_ != nullptr) strata::gpu::free(d_nentries_);
    if (d_entry_plans_ != nullptr) strata::gpu::free(d_entry_plans_);
    if (d_pair_blob_base_ != nullptr) strata::gpu::free(d_pair_blob_base_);
    if (d_pair_format_index_ != nullptr) strata::gpu::free(d_pair_format_index_);
    if (h_blob_base_ != nullptr) strata::gpu::free(h_blob_base_);
    if (h_format_index_ != nullptr) strata::gpu::free(h_format_index_);
    if (h_format_table_ != nullptr) strata::gpu::free(h_format_table_);
    if (h_start_ != nullptr) strata::gpu::free(h_start_);
    if (h_tok_ != nullptr) strata::gpu::free(h_tok_);
    if (h_dst_ != nullptr) strata::gpu::free(h_dst_);
    if (h_ngroups_ != nullptr) strata::gpu::free(h_ngroups_);
    if (h_nentries_ != nullptr) strata::gpu::free(h_nentries_);
    d_scratch_ = nullptr;
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
        formats > std::numeric_limits<size_t>::max() / sizeof(Exl3ExpertFormatDesc)) {
        err = "exl3 runtime: plan allocation overflows";
        return false;
    }
    if (!host(reinterpret_cast<void**>(&h_blob_base_), groups * sizeof(uint64_t), "blob plan") ||
        !host(reinterpret_cast<void**>(&h_format_index_), groups * sizeof(int32_t), "format plan") ||
        !host(reinterpret_cast<void**>(&h_format_table_), formats * sizeof(Exl3ExpertFormatDesc), "format table") ||
        !host(reinterpret_cast<void**>(&h_start_), (groups + 1) * sizeof(int32_t), "start plan") ||
        !host(reinterpret_cast<void**>(&h_tok_), entries * sizeof(int32_t), "token plan") ||
        !host(reinterpret_cast<void**>(&h_dst_), entries * sizeof(int32_t), "destination plan") ||
        !host(reinterpret_cast<void**>(&h_ngroups_), sizeof(int32_t), "group count") ||
        !host(reinterpret_cast<void**>(&h_nentries_), sizeof(int32_t), "entry count") ||
        !device(reinterpret_cast<void**>(&d_blob_base_), groups * sizeof(uint64_t), "blob plan") ||
        !device(reinterpret_cast<void**>(&d_format_index_), groups * sizeof(int32_t), "format plan") ||
        !device(reinterpret_cast<void**>(&d_format_table_), formats * sizeof(Exl3ExpertFormatDesc), "format table") ||
        !device(reinterpret_cast<void**>(&d_start_), (groups + 1) * sizeof(int32_t), "start plan") ||
        !device(reinterpret_cast<void**>(&d_tok_), entries * sizeof(int32_t), "token plan") ||
        !device(reinterpret_cast<void**>(&d_dst_), entries * sizeof(int32_t), "destination plan") ||
        !device(reinterpret_cast<void**>(&d_ngroups_), sizeof(int32_t), "group count") ||
        !device(reinterpret_cast<void**>(&d_nentries_), sizeof(int32_t), "entry count") ||
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
    std::memset(h_blob_base_, 0, groups * sizeof(uint64_t));
    std::memset(h_format_index_, 0, groups * sizeof(int32_t));
    std::memset(h_format_table_, 0, formats * sizeof(Exl3ExpertFormatDesc));
    std::memset(h_start_, 0, (groups + 1) * sizeof(int32_t));
    std::memset(h_tok_, 0, entries * sizeof(int32_t));
    std::memset(h_dst_, 0, entries * sizeof(int32_t));
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
                if (blob == nullptr && (slot >= 0 || all_resident_)) {
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
    all_resident_ = resident.size() == static_cast<size_t>(pairs);
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
        // taken in pieces of that size, the way the plain model's tier does it (a cache that does not fit in
        // one allocation is a cache that does not fit at all).
        cache_.set_max_piece(max_alloc_bytes());
        if (!cache_.open_buckets(buckets, n_layers_, n_expert_, err)) { close(); return false; }
        // The fills are QUEUED and waited for once, not one synchronous copy per expert: a cache that holds
        // thousands of experts would otherwise pay a device synchronisation per expert.  A queued copy reads the
        // host buffer until it is done, so the buffers are a small ring and the ring is waited for before a
        // buffer is reused.
        constexpr size_t kFillRing = 16;
        std::vector<std::vector<uint8_t>> ring(kFillRing);
        size_t next = 0;
        for (const ResidentPair& pair : resident) {
            const size_t at = pair_index(pair.first, pair.second);
            const auto& layout = format_table_host_[(size_t) pair_format_index_[at]];
            std::vector<uint8_t>& blob = ring[next];
            blob.resize(static_cast<size_t>(layout.total_bytes));
            if (!source.copy_blob(pair.first, pair.second, blob.data())) {
                err = "exl3 runtime: resident source copy failed"; close(); return false;
            }
            const int32_t slot = cache_.admit(pair.first, pair.second, pair_formats_[at]);
            if (slot < 0 || !cache_.fill_slot_queued(slot, blob.data(), err, static_cast<int64_t>(blob.size()))) {
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
    }
    cpu_jobs_.resize(static_cast<size_t>(max_entries_));
    miss_ids_.reserve(static_cast<size_t>(n_expert_));
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
    miss_ids_.clear();
    std::vector<uint8_t> miss_seen(static_cast<size_t>(n_expert_), 0);
    std::vector<int32_t> group_slots;
    std::vector<std::vector<int32_t>> group_entries;
    std::unordered_map<int32_t, int32_t> groups_by_slot;
    for (int64_t entry = 0; entry < n_tokens * routed_k_; ++entry) {
        const int32_t expert = routed_ids[entry];
        if (expert < 0 || expert >= n_expert_) { err = "exl3 runtime: routed expert is out of range"; return false; }
        const int64_t token = entry / routed_k_;
        const size_t at = pair_index(layer, expert);
        const int32_t slot = cache_.valid() ? cache_.slot_of(layer, expert) : kNotResident;
        if (slot >= 0) {
            ++hit_entries_;
            std::fill(cpu_output + entry * ldy, cpu_output + entry * ldy + ldy, 0.0f);
            auto [it, inserted] = groups_by_slot.emplace(slot, static_cast<int32_t>(group_slots.size()));
            if (inserted) { group_slots.push_back(slot); group_entries.emplace_back(); }
            group_entries[(size_t) it->second].push_back(static_cast<int32_t>(entry));
        } else if (miss_seen[(size_t) expert] == 0) {
            miss_seen[(size_t) expert] = 1;
            miss_ids_.push_back(expert);
        }
        (void) token;
        (void) at;
    }
    detail_sample.mark(0);   // route partition, including temporary workspace
    if (!miss_ids_.empty()) {
        source_->begin_layer(layer, miss_ids_.data(), static_cast<int64_t>(miss_ids_.size()));
        if (!source_->last_error().empty()) { err = source_->last_error(); return false; }
    }
    detail_sample.mark(1);   // source begin (guard time is a nested component)
    for (int64_t entry = 0; entry < n_tokens * routed_k_; ++entry) {
        const int32_t expert = routed_ids[entry];
        const size_t at = pair_index(layer, expert);
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
        (void) at;
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
    *h_ngroups_ = static_cast<int32_t>(group_slots.size());
    *h_nentries_ = static_cast<int32_t>(cursor);
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
    return true;
}

bool Exl3ExpertRuntime::copy_plan(sycl::queue& queue, const std::vector<sycl::event>& deps,
                                  std::vector<sycl::event>& copies, std::string& err) {
    try {
        auto copy = [&](void* dst, const void* src, size_t bytes) {
            copies.push_back(queue.submit([&](sycl::handler& h) {
                if (!deps.empty()) h.depends_on(deps);
                h.memcpy(dst, src, bytes);
            }));
        };
        copy(d_blob_base_, h_blob_base_, static_cast<size_t>(max_entries_) * sizeof(uint64_t));
        copy(d_format_index_, h_format_index_, static_cast<size_t>(max_entries_) * sizeof(int32_t));
        copy(d_start_, h_start_, (static_cast<size_t>(max_entries_) + 1) * sizeof(int32_t));
        copy(d_tok_, h_tok_, static_cast<size_t>(max_entries_) * sizeof(int32_t));
        copy(d_dst_, h_dst_, static_cast<size_t>(max_entries_) * sizeof(int32_t));
        copy(d_ngroups_, h_ngroups_, sizeof(int32_t));
        copy(d_nentries_, h_nentries_, sizeof(int32_t));
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
    try {
        done = strata::kernels::exl3::exl3_grouped_fp32(queue, device_plan_, device_fp32_x, ldx, gpu_output, ldy,
                                                        d_scratch_, scratch_elements_, {}, copies);
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
        const sycl::event plan = queue.submit([&](sycl::handler& h) {
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
        (void) strata::kernels::exl3::exl3_grouped_fp32(queue, device_plan_, device_fp32_x, hidden_, gpu_output, ldy,
                                                        d_scratch_, scratch_elements_, {}, {plan});
    } catch (const std::exception& e) {
        err = std::string("exl3 runtime: device hit grouped enqueue: ") + e.what();
        return false;
    }
    return true;
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
    // This plan contains cache hits and CPU misses only. Publish completion
    // of the empty PCIe share through the same window protocol as legacy plans.
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
