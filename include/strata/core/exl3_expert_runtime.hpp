// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Bounded EXL3 source/cache/CPU/GPU plan integration.
#pragma once

#include "strata/core/exl3_expert_formats.hpp"
#include "strata/core/exl3_expert_source.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/exl3_grouped.hpp"

#include <cstddef>
#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::core {

class Runtime;
struct GpuPlanSink;

/// Owns the persistent EXL3 plan images, cache, and grouped scratch.  The source and CPU pool are borrowed and must
/// outlive this object. Captured graphs retain pointers to this object's plan/cache/scratch, so the runtime must
/// outlive every graph that captured enqueue() until that graph has been destroyed. A prepared window must be
/// consumed with wait_gpu() or mark_gpu_consumed() after its readers finish, before prepare_window() is called again.
class Exl3ExpertRuntime {
public:
    using ResidentPair = std::pair<int32_t, int32_t>;

    /// How a non-resident (miss) expert is computed.  `Cpu` is the original path: the source's blob is
    /// computed by the CPU pool.  `Direct` stages the miss's canonical blob in pinned host memory and lets
    /// the GPU's grouped kernel read it from there over the bus (PCIe), so the GPU - not the CPU -
    /// computes the miss and no VRAM copy is made.  Direct falls back to `Cpu` per window when the pinned
    /// staging is unavailable or the device cannot read host memory.  `Split` stages some of a window's
    /// misses like `Direct` and computes the rest on the CPU pool, in the same grouped launch: the GPU
    /// and the CPU run beside each other and the per-window share is chosen at run time (see
    /// plan_split_gpu).  A window whose GPU share does not fit the pinned staging, or a device that
    /// cannot read host memory, keeps those misses on the CPU.
    enum class MissMode { Cpu, Direct, Split };

    Exl3ExpertRuntime() = default;
    ~Exl3ExpertRuntime();
    Exl3ExpertRuntime(const Exl3ExpertRuntime&) = delete;
    Exl3ExpertRuntime& operator=(const Exl3ExpertRuntime&) = delete;

    /// Discover every exact source layout, register format identities, build optional cache buckets, preload only the
    /// explicit resident pairs, and allocate permanent host/device plan images plus six-plane grouped scratch.
    /// After every runtime/workspace allocation, optionally prewarm CPU pairs
    /// in [prewarm_layer_begin, prewarm_layer_end); -1 means the source's end.
    /// Stage owners pass their own range; a single runtime defaults to all.
    bool open(Exl3ExpertSource& source, strata::kernels::cpu::ExpertPool& pool, int64_t max_tokens, int64_t routed_k,
              const std::vector<ResidentPair>& resident, std::string& err,
              int64_t prewarm_layer_begin = 0, int64_t prewarm_layer_end = -1);
    void close();

    /// Classify hits before touching the source.  Only unique CPU-miss IDs are passed to source.begin_layer(); cached
    /// experts are never read by the CPU path.  `cpu_output` is route-row storage with `ldy` stride.
    bool prepare_window(int64_t layer, const float* host_fp32_x, int64_t ldx, const int32_t* routed_ids,
                        int64_t n_tokens, float* cpu_output, int64_t ldy, std::string& err);
    /// Run the CPU miss jobs separately. It may overlap a previously enqueued hit-only GPU plan; it never waits for
    /// that plan. The existing enqueue() method still keeps its CPU-first contract for callers that use it directly.
    bool run_cpu(std::string& err);

    /// Copy the current persistent host plan image to the device image and enqueue grouped EXL3.  The caller must
    /// wait_gpu() on the returned event before preparing another window.  Device x/y pointers use the same static
    /// capacities as open(): x has max_tokens rows and gpu_output has max_tokens*routed_k rows.
    bool enqueue(sycl::queue& queue, const float* device_fp32_x, int64_t ldx, float* gpu_output, int64_t ldy,
                 sycl::event& done, const std::vector<sycl::event>& deps, std::string& err);
    /// Enqueue only the resident GPU rows after prepare_window/publish_window_plan and before run_cpu. This is the
    /// bounded overlap path; the caller must wait_gpu() or mark_gpu_consumed() after the merge segment completes.
    bool enqueue_hit_only(sycl::queue& queue, const float* device_fp32_x, int64_t ldx, float* gpu_output, int64_t ldy,
                          sycl::event& done, const std::vector<sycl::event>& deps, std::string& err);
    /// Record the persistent plan copies and grouped kernel during graph capture.  It deliberately does not require
    /// a prepared CPU window and does not change the runtime's in-flight state; replay reads the host plan image that
    /// the caller updates before launching the graph.
    bool record_enqueue(sycl::queue& queue, const float* device_fp32_x, int64_t ldx, float* gpu_output, int64_t ldy,
                        std::string& err);
    /// Record an all-resident device route-plan builder followed by the grouped kernel.  The route IDs are device
    /// data written by MoeWindow::publish; no host plan copy, allocation, callback, or CPU miss path is involved.
    /// The in-order queue serializes reuse of the persistent plan and scratch. The owner must outlive replay.
    bool record_resident_enqueue(sycl::queue& queue, int64_t layer, const float* device_fp32_x,
                                 const int32_t* device_ids, int64_t n_tokens, float* gpu_output, int64_t ldy,
                                 std::string& err);
    /// Record a device route-plan builder for the immutable cache's hits, followed by grouped EXL3. Valid misses
    /// have no GPU destination; the caller supplies their CPU rows and a separate compact hit list for merging.
    /// Host prepare_window() images are independent of this plan and must never be copied over it during replay.
    bool record_device_hit_enqueue(sycl::queue& queue, int64_t layer, const float* device_fp32_x,
                                   const int32_t* device_ids, int64_t n_tokens, float* gpu_output, int64_t ldy,
                                   std::string& err);
    /// The same device route-plan builder and grouped kernels, enqueued outside a capture and returning the
    /// final event: the routed ids are device data and the builder resolves each route against the immutable
    /// cache's pair table itself, so the host neither builds nor uploads a plan and nothing is read back.  A
    /// route the cache holds lands in its output row; one it misses gets no GPU destination and stays the CPU
    /// path's.  The ids reach the host through the doorbell's mapped memory, off the decode's critical path.
    bool enqueue_device_hit(sycl::queue& queue, int64_t layer, const float* device_fp32_x, const int32_t* device_ids,
                            int64_t n_tokens, float* gpu_output, int64_t ldy, sycl::event& done,
                            const std::vector<sycl::event>& deps, std::string& err);
    /// Publish the planned entry list through the legacy window sink so its CPU-row merge can reuse the existing
    /// mapped plan and flag protocol.  EXL3's own grouped plan remains in the persistent runtime images.
    /// The list holds every entry the grouped kernel computes (cache hits plus Direct/Split GPU-read
    /// misses); CPU-computed misses are the sink's zeroed rows, merged by the caller.
    bool publish_window_plan(GpuPlanSink& sink, std::string& err) const;
    bool wait_gpu(sycl::event& done, std::string& err);
    /// Mark a graph replay consumed after stream synchronization, or after observing the next ring from that same
    /// in-order graph (all previous plan/row/merge readers have finished). Graph events cannot be waited directly.
    void mark_gpu_consumed();

    /// Selects the miss path.  Call before open(); the default is `MissMode::Cpu`.
    void set_miss_mode(MissMode mode) { miss_mode_ = mode; }
    MissMode miss_mode() const { return miss_mode_; }
    /// True when the Direct staging was allocated and the device can read it (set by open()).
    bool direct_ready() const { return direct_ready_; }

    /// One distinct miss expert's cost inputs for the split decision: how many routed entries need
    /// it and how many bytes the GPU would read for it over the bus.
    struct SplitCandidate {
        int32_t expert = -1;
        int64_t entries = 0;
        uint64_t bytes = 0;
    };
    /// How many of `candidates` (most entries first) the GPU takes.  The k minimizing
    /// max(cpu, gpu) wins, with cpu = (all_entries - gpu_entries) * cpu_us_per_entry and
    /// gpu = gpu_bytes / bytes_per_sec; ties keep the smaller k (the CPU default).  `capacity`
    /// caps k to what the pinned staging holds.  No estimate yet (non-positive cpu_us_per_entry
    /// or bytes_per_sec) answers 0: every miss stays on the CPU until both sides have measured
    /// costs.  A pure function of its arguments, so tests call it directly.
    static int64_t plan_split_gpu(const SplitCandidate* candidates, int64_t n, double cpu_us_per_entry,
                                  double bytes_per_sec, int64_t capacity);
    /// The split controller's state: the CPU cost per miss entry (an EWMA of run_cpu, 0 before
    /// the first CPU sample), the PCIe bandwidth in bytes/s (a timed copy at open, 0 when not
    /// measured), and the windows decided so far.  All are derived at run time, never configured.
    double split_cpu_us_per_entry() const { return cpu_us_per_entry_; }
    double split_pcie_bytes_per_sec() const { return pcie_bytes_per_sec_; }
    int64_t split_windows() const { return split_windows_; }
    int64_t split_last_gpu_experts() const { return split_last_gpu_; }
    int64_t split_last_cpu_experts() const { return split_last_cpu_; }
    /// The GPU share so far: split GPU-read entries over all split-decided miss entries (0 with none).
    double split_gpu_share() const {
        return split_miss_entries_ > 0
                   ? static_cast<double>(split_gpu_entries_) / static_cast<double>(split_miss_entries_)
                   : 0.0;
    }
    /// Split-decided miss entries so far: the GPU-read ones and every one (GPU + CPU).
    int64_t split_gpu_miss_entries() const { return split_gpu_entries_; }
    int64_t split_all_miss_entries() const { return split_miss_entries_; }
    /// Split's persistent pinned cache: slots, lookups and hits since open (0 slots: every window
    /// stages its GPU share anew in the per-window staging instead).
    int64_t split_cache_slots() const { return split_cache_slots_; }
    int64_t split_cache_hits() const { return split_cache_hits_; }
    int64_t split_cache_lookups() const { return split_cache_lookups_; }
    double split_cache_hit_rate() const {
        return split_cache_lookups_ > 0
                   ? static_cast<double>(split_cache_hits_) / static_cast<double>(split_cache_lookups_)
                   : 0.0;
    }

    /// Records every routed (layer, expert) entry's count while set (before open()).  The counts are the
    /// firing-frequency profile a later run ranks its resident set with (write_expert_profile's format).
    void set_record_heat(bool on) { record_heat_ = on; }
    bool record_heat() const { return record_heat_; }
    /// Per-pair routed-entry counts, [n_layers * n_expert]; empty until a window is prepared with
    /// recording on.
    const std::vector<uint64_t>& heat() const { return heat_; }

    /// Cumulative routed-entry counts since open(): resident hits, CPU-computed misses and
    /// direct-read misses.  Their sum is every routed entry prepared.  A Split window's GPU share
    /// counts as direct-read misses and its CPU share as CPU-computed misses; the split_*() state
    /// above tells the two apart per window.
    int64_t total_hit_entries() const { return total_hit_entries_; }
    int64_t total_cpu_miss_entries() const { return total_cpu_miss_entries_; }
    int64_t total_direct_miss_entries() const { return total_direct_miss_entries_; }

    bool prepared() const { return prepared_; }
    bool cpu_ready() const { return cpu_ready_; }
    bool pending_gpu() const { return pending_gpu_; }

    int64_t hidden() const { return hidden_; }
    int64_t n_layers() const { return n_layers_; }
    int64_t n_expert() const { return n_expert_; }
    int64_t routed_k() const { return routed_k_; }
    int64_t max_ff() const { return max_ff_; }
    int64_t max_entries() const { return max_entries_; }
    int64_t miss_entries() const { return miss_entries_; }
    int64_t hit_entries() const { return hit_entries_; }
    int64_t miss_experts() const { return static_cast<int64_t>(miss_ids_.size()); }
    int64_t gpu_groups() const { return h_ngroups_ == nullptr ? 0 : *h_ngroups_; }
    int64_t gpu_entries() const { return h_nentries_ == nullptr ? 0 : *h_nentries_; }
    size_t grouped_scratch_elements() const { return scratch_elements_; }
    const ExpertCache& cache() const { return cache_; }
    const Exl3ExpertSource* source() const { return source_; }
    /// True only when every (layer, expert) pair was admitted to the persistent device cache at open().
    bool fully_resident() const { return all_resident_; }

    /// Device plan fields exposed for the future MoeWindow merge seam.
    const uint64_t* gpu_blob_base() const { return d_blob_base_; }
    const int32_t* gpu_format_index() const { return d_format_index_; }
    const int32_t* gpu_start() const { return d_start_; }
    const int32_t* gpu_tok() const { return d_tok_; }
    const int32_t* gpu_dst() const { return d_dst_; }
    const int32_t* gpu_ngroups() const { return d_ngroups_; }
    const int32_t* gpu_nentries() const { return d_nentries_; }
    int64_t gpu_group_capacity() const { return max_entries_; }
    int64_t gpu_entry_capacity() const { return max_entries_; }

private:
    bool enqueue_impl(sycl::queue& queue, const float* device_fp32_x, int64_t ldx, float* gpu_output, int64_t ldy,
                      sycl::event& done, const std::vector<sycl::event>& deps, bool require_cpu, std::string& err);
    bool submit_device_hit_plan(sycl::queue& queue, int64_t layer, const int32_t* device_ids, int64_t n_tokens,
                                const std::vector<sycl::event>& deps, sycl::event& plan, std::string& err);
    bool allocate_images(std::string& err);
    /// Times a pinned->device copy through the staging arena and records the PCIe bandwidth the
    /// split controller divides GPU bytes by.  Only for MissMode::Split with a usable staging.
    void measure_pcie_bandwidth();
    /// Opens Split's persistent pinned cache from the host RAM the machine reports now (or
    /// STRATA_EXL3_SPLIT_CACHE_MIB).  Never fails the open: 0 slots keeps the per-window staging.
    void open_split_cache();
    /// The cache slot holding (layer, expert)'s canonical blob, staging it on a miss.  -1 when the
    /// blob could not be staged: the caller keeps that expert on the CPU.
    int64_t split_cache_assign(int64_t layer, int32_t expert);
    bool copy_plan(sycl::queue& queue, const std::vector<sycl::event>& deps, std::vector<sycl::event>& copies,
                   std::string& err);
    bool valid_layer(int64_t layer) const { return layer >= 0 && layer < n_layers_; }
    size_t pair_index(int64_t layer, int64_t expert) const {
        return static_cast<size_t>(layer) * static_cast<size_t>(n_expert_) + static_cast<size_t>(expert);
    }

    Exl3ExpertSource* source_ = nullptr;
    strata::kernels::cpu::ExpertPool* pool_ = nullptr;
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
    int64_t max_tokens_ = 0;
    int64_t routed_k_ = 0;
    int64_t max_entries_ = 0;
    int64_t hidden_ = 0;
    int64_t max_ff_ = 0;
    // Optional packed output-block workspace; zero keeps whole-expert dispatch.
    int64_t cpu_rowparallel_jobs_ = 0;
    MissMode miss_mode_ = MissMode::Cpu;
    bool direct_ready_ = false;
    uint8_t* direct_staging_ = nullptr;   ///< pinned host arena the GPU reads miss blobs from (Direct/Split)
    uint64_t direct_stride_ = 0;          ///< one staged blob's stride in direct_staging_
    int64_t direct_capacity_ = 0;         ///< distinct miss experts the arena holds
    double cpu_us_per_entry_ = 0.0;       ///< EWMA of run_cpu per CPU miss entry; 0 before the first sample
    double pcie_bytes_per_sec_ = 0.0;     ///< timed pinned->device copy at open; 0 when not measured
    int64_t split_windows_ = 0;           ///< windows the split controller decided
    int64_t split_gpu_entries_ = 0;       ///< split windows' GPU-read miss entries
    int64_t split_miss_entries_ = 0;      ///< split windows' every miss entry (GPU + CPU)
    int64_t split_last_gpu_ = 0;          ///< distinct GPU experts of the last split window
    int64_t split_last_cpu_ = 0;          ///< distinct CPU experts of the last split window
    // Split's persistent pinned canonical-blob cache, keyed by (layer, expert): a GPU-share expert
    // staged once is reread from here on later tokens instead of converted again every window.
    // Empty (0 slots) keeps the per-window staging above as the only GPU path.
    uint8_t* split_cache_ = nullptr;     ///< pinned arena: split_cache_slots_ x direct_stride_ bytes
    int64_t split_cache_slots_ = 0;      ///< canonical-blob slots in the arena
    std::unordered_map<int64_t, int64_t> split_cache_index_;  ///< (layer, expert) key -> slot
    std::list<int64_t> split_cache_lru_;                      ///< keys, front = most recently used
    std::vector<std::list<int64_t>::iterator> split_cache_slot_pos_;  ///< per-slot LRU position
    std::vector<int64_t> split_cache_slot_key_;               ///< per-slot key, -1 when free
    std::vector<int64_t> split_cache_free_;                   ///< free slots, a stack
    int64_t split_cache_hits_ = 0;       ///< GPU-share assignments served from the cache
    int64_t split_cache_lookups_ = 0;    ///< every GPU-share assignment, hit or miss
    bool record_heat_ = false;
    std::vector<uint64_t> heat_;
    int64_t total_hit_entries_ = 0;
    int64_t total_cpu_miss_entries_ = 0;
    int64_t total_direct_miss_entries_ = 0;
    int64_t direct_entries_ = 0;          ///< this window's direct-read entries (not CPU jobs)

    bool opened_ = false;
    bool prepared_ = false;
    bool cpu_ready_ = false;
    bool pending_gpu_ = false;
    bool all_resident_ = false;
    int64_t prepared_layer_ = -1;
    int64_t miss_entries_ = 0;
    int64_t hit_entries_ = 0;

    Exl3ExpertFormatRegistry registry_;
    std::vector<Exl3ExpertFormatID> pair_formats_;
    std::vector<Exl3ExpertFormatDesc> format_table_host_;
    std::vector<int32_t> pair_format_index_;
    ExpertCache cache_;
    std::vector<strata::kernels::cpu::ExpertJobExl3> cpu_jobs_;
    std::vector<int32_t> miss_ids_;

    // The seven per-window plan arrays live in one contiguous host arena and one contiguous device arena,
    // carved out in the same order on both sides so one memcpy uploads the whole window plan.  The typed pointers
    // below are views into those arenas, not separate allocations.
    void* h_plan_arena_ = nullptr;
    void* d_plan_arena_ = nullptr;
    size_t plan_arena_bytes_ = 0;
    uint64_t* h_blob_base_ = nullptr;
    int32_t* h_format_index_ = nullptr;
    strata::kernels::exl3::ExpertBlobLayout* h_format_table_ = nullptr;
    int32_t* h_start_ = nullptr;
    int32_t* h_tok_ = nullptr;
    int32_t* h_dst_ = nullptr;
    int32_t* h_ngroups_ = nullptr;
    int32_t* h_nentries_ = nullptr;
    uint64_t* d_blob_base_ = nullptr;
    int32_t* d_format_index_ = nullptr;
    strata::kernels::exl3::ExpertBlobLayout* d_format_table_ = nullptr;
    int32_t* d_start_ = nullptr;
    int32_t* d_tok_ = nullptr;
    int32_t* d_dst_ = nullptr;
    int32_t* d_ngroups_ = nullptr;
    int32_t* d_nentries_ = nullptr;
    strata::kernels::exl3::GroupedEntryPlan* d_entry_plans_ = nullptr;  ///< [max_entries], device plan resolution
    uint64_t* d_pair_blob_base_ = nullptr;   ///< [n_layers*n_expert], immutable; zero for CPU-only pairs
    int32_t* d_pair_format_index_ = nullptr; ///< [n_layers*n_expert], immutable; -1 for CPU-only pairs
    int64_t pair_count_ = 0;
    float* d_scratch_ = nullptr;
    size_t scratch_elements_ = 0;
    strata::kernels::exl3::GroupedPlan device_plan_{};
    Runtime* device_runtime_ = nullptr;
};

}  // namespace strata::core
