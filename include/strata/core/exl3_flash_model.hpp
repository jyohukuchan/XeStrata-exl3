// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Explicit Qwen3.8 EXL3 Flash-model ownership and stage preparation.
#pragma once

#include "strata/artifact/exl3_model_map.hpp"
#include "strata/artifact/exl3_model_spec.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/weights.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace strata::artifact::exl3 {
class Archive;
}

namespace strata::kernels {
class PleTable;
}

namespace strata::core {

/// Softmax probability of `draft` under one row of `logits` (the draft
/// token's confidence).  Non-finite logits carry no probability mass; a
/// draft outside the row, or whose own logit is non-finite, gets 0.
inline float exl3_flash_draft_probability(const float* logits, size_t count, int32_t draft) {
    if (logits == nullptr || draft < 0 || static_cast<size_t>(draft) >= count) return 0.0f;
    const double draft_logit = static_cast<double>(logits[draft]);
    if (!std::isfinite(draft_logit)) return 0.0f;
    double best = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < count; ++i) {
        const double value = static_cast<double>(logits[i]);
        if (std::isfinite(value) && value > best) best = value;
    }
    if (!std::isfinite(best)) return 0.0f;
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double value = static_cast<double>(logits[i]);
        if (std::isfinite(value)) sum += std::exp(value - best);
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) return 0.0f;
    return static_cast<float>(std::exp(draft_logit - best) / sum);
}

/// Whether a draft chain extends past a draft of probability `draft_prob`
/// under the `--spec-min-p` threshold: at or above it the chain continues,
/// below it the chain stops and no further draft is proposed.  A threshold
/// of 0 (or less) never stops the chain, which is the plain `--spec` loop.
inline bool exl3_flash_spec_extends(float draft_prob, double min_p) {
    if (!(min_p > 0.0)) return true;
    return static_cast<double>(draft_prob) >= min_p;
}

class Exl3Dense;
class Exl3ExpertRuntime;
class Exl3ExpertSource;
class Exl3Ple;
class Exl3WeightSet;
class ExpertPool;
class NativeEmbed;

/// Startup policy for the first EXL3 Flash coordinator.
///
/// This deliberately describes one autoregressive token at a time.  MTP,
/// vision, and batched/prefill execution need their own graph contracts and
/// are rejected by open() until their callers register those contracts.
struct Exl3FlashModelOptions {
    std::vector<int> stage_devices;       ///< explicit runtime ordinals, one per layer stage
    std::vector<int64_t> stage_layers;    ///< explicit layer counts per stage, e.g. 18,18,12; empty: the even split
    int pool_workers = 0;                 ///< workers per stage; 0 preserves the automatic physical-core default
    int64_t max_rows = 1;                 ///< persistent projection/expert row capacity
    int64_t max_cells = 0;                ///< 0: config.max_position_embeddings
    int64_t routed_k = 0;                 ///< 0: text_config.num_experts_per_tok
    int64_t prompt_chunk = 0;             ///< 0: prompt runs token by token; >0: batched prefill chunk size
    bool include_mtp = false;             ///< bind the archive MTP map (draft/verify loop still AR-only)
    bool vision = false;                  ///< currently rejected
    bool allow_cpu_expert_miss = true;    ///< misses use Exl3ExpertSource + ExpertPool
    std::string expert_profile;            ///< empty: STRATA_EXL3_DEVICE_CACHE_PROFILE / spread order;
                                           ///< a path ranks the device cache's resident set by firing frequency
    std::string expert_profile_save;       ///< empty: no recording; a path records this run's routed-entry
                                           ///< counts and saves them as a profile for the next run
    bool expert_miss_direct = false;       ///< misses are read by the GPU from pinned host memory over
                                           ///< the bus instead of being computed on the CPU
    std::string embedding_name;            ///< empty: ModelMap's HF embedding prefix
    std::string ple_sidecar;              ///< empty: discover the specialized EXL3 sidecar
};

/// Non-owning view of one stage.  The coordinator owns every pointer for its
/// lifetime; a caller must destroy captured graphs before close().  `table`
/// contains only this stage's canonical rows, while `experts` can classify
/// all layers and has resident cache entries only for this stage's range.
struct Exl3FlashStageView {
    int device = -1;
    int64_t layer_begin = 0;
    int64_t layer_end = 0;
    WeightTable* table = nullptr;
    Exl3WeightSet* weights = nullptr;
    Exl3ExpertSource* expert_source = nullptr;
    Exl3ExpertRuntime* expert_runtime = nullptr;
};

/// The layer range one stage owns.  The layers are split as evenly as the count allows, so every
/// stage gets at least one layer as long as there are no more stages than layers; open() refuses
/// that case, and an empty range is what the helper answers for it.
struct Exl3FlashStageRange {
    int64_t layer_begin = 0;
    int64_t layer_end = 0;
};

/// A caller-owned full-sequence snapshot for the draft/verify loop.  It holds
/// every device buffer a forward mutates (the session arenas, the PLE owner's
/// device rows) plus the host QSA mirrors and the host sequence scalars, so a
/// rejected draft rewinds by restoring.  Save allocates (or grows) the
/// snapshot's own device buffers; release frees them.  KV streaming is not
/// covered: saving a streamed state fails closed.
struct Exl3FlashSnapshot {
    struct StageSnap {
        void* arena = nullptr;
        size_t arena_bytes = 0;
        std::vector<std::vector<int32_t>> qsa_step;  ///< per allocated state, kStepCount+1
        std::vector<std::vector<int32_t>> qsa_pos;   ///< per allocated state, n_head
        void* ple_emb_dev = nullptr;
        void* ple_scratch = nullptr;
    };
    std::vector<StageSnap> stages;
    void* mtp_arena = nullptr;
    size_t mtp_arena_bytes = 0;
    std::vector<std::vector<int32_t>> mtp_qsa_step;
    std::vector<std::vector<int32_t>> mtp_qsa_pos;
    int64_t next_position = 0;
    int32_t ple_prev[2] = {-1, -1};
    int32_t ple_token = -1;
    bool seeded = false;
    bool initialized = false;
};

/// The range of stage `stage_index` of `stage_count` over `layer_count` layers.  The split is the
/// same for every layer count and stage count, and it is the only place the split is written down.
Exl3FlashStageRange exl3_flash_stage_range(int64_t layer_count, int64_t stage_count, int64_t stage_index);

/// The chunk sizes of a batched prefill of `n` tokens at chunk size `chunk`: full chunks
/// first, then one short last chunk when `n` is not a multiple.  Empty when either is
/// nonpositive (the caller then runs the prompt token by token).
std::vector<int64_t> exl3_flash_prefill_chunk_sizes(int64_t n, int64_t chunk);

/// The stage ranges of one split: every stage's layers, contiguous and in order.
///
/// `counts` gives each stage its layer count (`--stage-layers 18,18,12`); empty takes the even
/// split of `exl3_flash_stage_range`.  A split where a stage gets no layer, or where the counts do
/// not cover the model's layers exactly, is refused with the reason in `err`, because the stage
/// that holds a layer decides where that layer's state lives and the last stage holds the head.
bool exl3_flash_stage_ranges(int64_t layer_count, int64_t stage_count, const std::vector<int64_t>& counts,
                             std::vector<Exl3FlashStageRange>& ranges, std::string& err);

/// Whether a Qwen3.8 MTP map carries every role the draft binding needs: one
/// layer, both mtp_fc projections, the full QSA set (q/k/v/out plus both
/// indexer halves), the eight hyper-connection entries, and the complete
/// gate/up/down set for a positive expert count.  Anything less is not a
/// usable draft block, so it fails closed with the reason in `err`.
bool exl3_flash_mtp_roles_complete(const strata::artifact::exl3::ModelMap& mtp, std::string& err);

/// The stage that runs the PLE layer: the PLE's HOME STAGE.
///
/// The PLE block belongs to one layer, and the layer split gives that layer to exactly one stage.
/// The home stage holds the PLE's device half - the key and value projections' owners, the
/// gathered row, the preprojection destinations, the scratch and the conv history - and it reads
/// the n-gram rows there, so the PLE never crosses a stage boundary.  The other stages do not
/// touch the PLE at all.  -1 when no stage holds the layer.
int64_t exl3_flash_ple_stage(const std::vector<Exl3FlashStageRange>& ranges);
/// The same for the even split of `exl3_flash_stage_range`.
int64_t exl3_flash_ple_stage(int64_t layer_count, int64_t stage_count);

/// Owns the explicit EXL3 archive-to-runtime bridge for Flash AR execution.  It validates typed geometry and canonical mapping, attaches
/// the exact EXL3/raw rows to one WeightTable per GPU stage, prepares the
/// embedding/PLE owners, and opens the mixed packed/raw expert runtime.
///
/// forward() records the existing block pre/post graphs, merges CPU expert
/// misses with resident GPU rows, and bridges the residual between stages.
/// Where a stage's cache is fully resident it instead replays one captured
/// graph for the stage's whole token.
/// The explicit expert-window API also supports callers with their own buffers.
/// All GPU allocations and teardown use the stage device's OnDevice scope.
class Exl3FlashModel {
public:
    Exl3FlashModel();
    ~Exl3FlashModel();
    Exl3FlashModel(const Exl3FlashModel&) = delete;
    Exl3FlashModel& operator=(const Exl3FlashModel&) = delete;

    bool open(const std::filesystem::path& directory, const Exl3FlashModelOptions& options, std::string& err);
    bool close(std::string& err) noexcept;
    void close() noexcept;

    /// Prepare every owner that has stream-local state before graph capture.
    /// The stream may be null for the current device's compute queue.
    bool prepare_stream(void* stream, std::string& err);

    /// Prepare one routed expert window, run CPU misses, and enqueue GPU hits.
    /// `cpu_output` is written for misses and `device_output` for resident
    /// hits; the caller performs the existing router-weighted merge.  The
    /// returned event belongs to the selected stage's runtime.
    bool prepare_expert_window(size_t stage, int64_t layer, const float* host_x, int64_t ldx,
                               const int32_t* routed_ids, int64_t n_tokens, float* cpu_output, int64_t ldy,
                               void* stream, const float* device_x, float* device_output, sycl::event& done,
                               std::string& err);
    bool wait_expert_window(size_t stage, sycl::event& done, std::string& err);

    /// Reset all stage-local recurrent/KV/PLE state.  The next forward call
    /// must use position 0 and supplies the first token's embedding.
    bool reset(std::string& err);
    /// Run one AR token through every stage, including the mapped router,
    /// exact expert merge, post graph, final EXL3 head, and host argmax.
    /// `host_logits` may be null; argmax is still computed internally.
    bool forward(int32_t token, int64_t pos, int32_t& next_token, float* host_logits, std::string& err);
    /// Run `n` prompt tokens at positions pos0 .. pos0+n-1 in chunks of the open-time
    /// prompt_chunk size: dense projections batched per chunk, stateful steps in token
    /// order, experts grouped per layer over the chunk.  Leaves the state the `n`
    /// single-token forwards would have left; `next_token`/`host_logits` are the last
    /// token's.  Requires prompt_chunk > 0 at open(); otherwise fails closed so the
    /// caller runs forward() per token.
    bool forward_prefill(const int32_t* tokens, int64_t n, int64_t pos0, int32_t& next_token,
                         float* host_logits, std::string& err);

    bool validate_token_request(int64_t n_tokens, int64_t batch, std::string& err) const;

    bool loaded() const;

    /// Routed-entry counts summed over the stages, for the run report: resident hits, misses computed
    /// on the CPU and misses the GPU read directly from pinned host memory, plus the resident pair count.
    struct ExpertCacheStats {
        int64_t hit_entries = 0;
        int64_t cpu_miss_entries = 0;
        int64_t direct_miss_entries = 0;
        int64_t resident_pairs = 0;
    };
    ExpertCacheStats expert_cache_stats() const;
    /// Writes the firing frequencies recorded this run (options.expert_profile_save) as a profile file
    /// ranking every (layer, expert) pair, hottest first.  False with `err` when nothing was recorded.
    bool save_expert_profile(const std::string& path, std::string& err) const;

    size_t stage_count() const;
    Exl3FlashStageView stage(size_t index) const;
    const strata::ModelSpec& spec() const;
    const ModelGeometry& geometry() const;
    const strata::artifact::exl3::ModelMap& model_map() const;
    /// Whether open() bound the Qwen3.8 MTP map.  Forward stays single-token
    /// AR until the draft/verify loop lands; the map is the binding it will use.
    bool has_mtp() const;
    const strata::artifact::exl3::ModelMap& mtp_map() const;
    /// The MTP draft block's dense weights, or null when include_mtp was not
    /// set.  The draft expert runtime holds a device cache of its own: when
    /// every draft expert is resident the draft body after the embedding
    /// gather replays as one captured graph on the GPU, otherwise misses run
    /// on the CPU pool.
    const Exl3WeightSet* mtp_weights() const;
    /// Run one MTP draft step from the current trunk state: the trunk must
    /// have just produced `next_token` at `trunk_pos`, so `next_position()`
    /// is `trunk_pos + 1`.  Only the MTP session advances; the trunk state is
    /// untouched.  The draft predicts the token after `next_token`; callers
    /// verify it with a trunk forward and rewind with a snapshot on reject.
    /// With `draft_prob`, it also receives the draft's softmax probability
    /// (see exl3_flash_draft_probability), the confidence `--spec-min-p`
    /// thresholds on; null leaves it uncomputed.
    bool mtp_draft(int32_t next_token, int64_t trunk_pos, int32_t& draft, float* host_logits,
                   std::string& err, float* draft_prob = nullptr);
    /// Save the full sequence state into the caller-owned snapshot, or restore
    /// it back.  Restoring a snapshot from another model, another stage split,
    /// or another arena size fails closed.  Forward stays valid after either.
    bool save_snapshot(Exl3FlashSnapshot& snap, std::string& err);
    bool restore_snapshot(const Exl3FlashSnapshot& snap, std::string& err);
    static void release_snapshot(Exl3FlashSnapshot& snap) noexcept;
    NativeEmbed* embedding() const;
    Exl3Ple* ple() const;
    strata::kernels::PleTable* ple_table() const;
    const std::shared_ptr<const strata::artifact::exl3::Archive>& archive() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::core
