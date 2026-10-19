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
    /// Publish only the hit entry list through the legacy window sink so its CPU-row merge can reuse the existing
    /// mapped plan and flag protocol.  EXL3's own grouped plan remains in the persistent runtime images.
    bool publish_window_plan(GpuPlanSink& sink, std::string& err) const;
    bool wait_gpu(sycl::event& done, std::string& err);
    /// Mark a graph replay consumed after stream synchronization, or after observing the next ring from that same
    /// in-order graph (all previous plan/row/merge readers have finished). Graph events cannot be waited directly.
    void mark_gpu_consumed();

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
    uint64_t* d_pair_blob_base_ = nullptr;   ///< [n_layers*n_expert], immutable; zero for CPU-only pairs
    int32_t* d_pair_format_index_ = nullptr; ///< [n_layers*n_expert], immutable; -1 for CPU-only pairs
    int64_t pair_count_ = 0;
    float* d_scratch_ = nullptr;
    size_t scratch_elements_ = 0;
    strata::kernels::exl3::GroupedPlan device_plan_{};
    Runtime* device_runtime_ = nullptr;
};

}  // namespace strata::core
