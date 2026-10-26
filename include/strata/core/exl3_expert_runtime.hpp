// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Bounded EXL3 source/cache/CPU/GPU plan integration.
#pragma once

#include "strata/core/exl3_expert_formats.hpp"
#include "strata/core/exl3_expert_source.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/exl3_grouped.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
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
    /// staging is unavailable or the device cannot read host memory.
    enum class MissMode { Cpu, Direct };

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
    /// Publish only the hit entry list through the legacy window sink so its CPU-row merge can reuse the existing
    /// mapped plan and flag protocol.  EXL3's own grouped plan remains in the persistent runtime images.
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

    /// Records every routed (layer, expert) entry's count while set (before open()).  The counts are the
    /// firing-frequency profile a later run ranks its resident set with (write_expert_profile's format).
    void set_record_heat(bool on) { record_heat_ = on; }
    bool record_heat() const { return record_heat_; }
    /// Per-pair routed-entry counts, [n_layers * n_expert]; empty until a window is prepared with
    /// recording on.
    const std::vector<uint64_t>& heat() const { return heat_; }

    /// The adaptive tier: while the round period is positive (set before open()), prepare_window() counts
    /// every routed entry into the decayed usage table, and maybe_adapt() - called with the device idle,
    /// e.g. at a token boundary - moves hot non-resident experts into cold residents' cache slots.  `every`
    /// is the tokens between rounds (0: off); `swaps` caps one round's moves; `mib_per_token` caps the
    /// round's host-to-device copies per token read since the last round (0: no byte bound); `given` says
    /// the period was set explicitly (no automatic choice below).  The tier is XeStrata's own (see
    /// exl3_adaptive.hpp): no file from another project is read or copied.
    void configure_adapt(int64_t every, int64_t swaps, int64_t mib_per_token, bool given);
    bool adapt_enabled() const { return adapt_every_ > 0; }
    /// Runs an adaptive round when one is due (a no-op otherwise, and always when the tier is off or the
    /// cache holds every pair).  The caller waits for the device first: a round rewrites cache slots no
    /// GPU kernel may be reading.  False with `err` when a round's copies fail.
    bool maybe_adapt(std::string& err);
    /// This run's adaptive-tier totals, for the run report.
    struct AdaptStats {
        int64_t rounds = 0;          ///< rounds that ran (including ones that moved nothing)
        int64_t swaps = 0;           ///< experts moved into the device cache
        int64_t skipped_format = 0;  ///< chosen swaps refused by a slot format or size mismatch
        uint64_t bytes = 0;          ///< host-to-device bytes the swaps copied
        double copy_ms = 0.0;        ///< host time the swaps' copies took
        double select_ms = 0.0;      ///< host time the rounds' swap choice took
        int64_t period = 0;          ///< the tokens between rounds now in effect
        bool auto_lengthened = false;  ///< the automatic choice lengthened the period
    };
    AdaptStats adapt_stats() const;

    /// Cumulative routed-entry counts since open(): resident hits, CPU-computed misses and
    /// direct-read misses.  Their sum is every routed entry prepared.
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
    bool adapt_round(std::string& err);
    bool enqueue_impl(sycl::queue& queue, const float* device_fp32_x, int64_t ldx, float* gpu_output, int64_t ldy,
                      sycl::event& done, const std::vector<sycl::event>& deps, bool require_cpu, std::string& err);
    bool submit_device_hit_plan(sycl::queue& queue, int64_t layer, const int32_t* device_ids, int64_t n_tokens,
                                const std::vector<sycl::event>& deps, sycl::event& plan, std::string& err);
    bool allocate_images(std::string& err);
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
    uint8_t* direct_staging_ = nullptr;   ///< pinned host arena the GPU reads miss blobs from (Direct)
    uint64_t direct_stride_ = 0;          ///< one staged blob's stride in direct_staging_
    int64_t direct_capacity_ = 0;         ///< distinct miss experts the arena holds
    bool record_heat_ = false;
    std::vector<uint64_t> heat_;
    // The adaptive tier's state (configure_adapt).  `adapt_usage_` is the decayed per-pair routing count
    // the rounds choose swaps from; the served layer range is this runtime's prewarm range.
    int64_t adapt_every_ = 0;
    int64_t adapt_swaps_ = 96;
    int64_t adapt_mib_per_token_ = 48;
    bool adapt_every_given_ = false;
    int64_t adapt_period_ = 4;
    int64_t adapt_layer_begin_ = 0;
    int64_t adapt_layer_end_ = 0;
    uint64_t adapt_max_blob_ = 0;
    std::vector<double> adapt_usage_;
    int64_t adapt_tokens_since_round_ = 0;
    int64_t adapt_tokens_total_ = 0;
    bool adapt_auto_decided_ = false;
    bool adapt_auto_lengthened_ = false;
    double adapt_copy_s_per_byte_ = -1.0;   ///< measured host-to-device seconds per byte (< 0: unmeasured)
    uint8_t* adapt_ring_base_ = nullptr;   ///< persistent pinned staging ring (freed in close())
    size_t adapt_ring_stride_ = 0;         ///< one ring slot's bytes
    std::chrono::steady_clock::time_point adapt_last_round_{};
    int64_t adapt_rounds_ = 0;
    int64_t adapt_swaps_done_ = 0;
    int64_t adapt_skipped_format_ = 0;
    uint64_t adapt_bytes_ = 0;
    double adapt_copy_ms_ = 0.0;
    double adapt_select_ms_ = 0.0;
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
    uint64_t* d_pair_blob_base_ = nullptr;   ///< [n_layers*n_expert], zero for CPU-only pairs
    int32_t* d_pair_format_index_ = nullptr; ///< [n_layers*n_expert], -1 for CPU-only pairs
    /// The device pair tables' host shadows: allocate_images fills them, adaptive rounds rewrite the
    /// swapped pairs in them and upload the tables again.
    std::vector<uint64_t> pair_blob_host_;
    std::vector<int32_t> pair_format_host_;
    int64_t pair_count_ = 0;
    float* d_scratch_ = nullptr;
    size_t scratch_elements_ = 0;
    strata::kernels::exl3::GroupedPlan device_plan_{};
    Runtime* device_runtime_ = nullptr;
};

}  // namespace strata::core
