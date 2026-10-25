// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Explicit Qwen3.8 EXL3 Flash-model ownership and stage preparation.
#include "strata/core/exl3_flash_model.hpp"

#include "strata/core/exl3_expert_runtime.hpp"
#include "strata/core/exl3_expert_source.hpp"
#include "strata/core/exl3_ple.hpp"
#include "strata/core/exl3_weight_set.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// `_mm_pause` for the doorbell spin.  Guarded because it is x86-only; a target without it still builds, the
// spin is just less polite to the pipeline.
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define STRATA_SPIN_PAUSE() _mm_pause()
#else
#define STRATA_SPIN_PAUSE() ((void) 0)
#endif

namespace strata::core {
namespace {

using Archive = strata::artifact::exl3::Archive;
using ExpertTensorNames = strata::artifact::exl3::ExpertTensorNames;
using ModelMap = strata::artifact::exl3::ModelMap;
using ProjectionKind = strata::artifact::exl3::ProjectionKind;

constexpr const char* kPlePending = "Qwen3.8 PLE ngram sidecar binding is pending in ModelMap";

bool product_size(int64_t a, int64_t b, size_t& result) {
    if (a <= 0 || b <= 0 || static_cast<uint64_t>(a) > std::numeric_limits<size_t>::max() /
                                      static_cast<uint64_t>(b))
        return false;
    result = static_cast<size_t>(a) * static_cast<size_t>(b);
    return true;
}

bool make_expert_names(const ModelMap& map, int64_t layers, int64_t experts,
                       std::vector<ExpertTensorNames>& result, std::string& err) {
    size_t count = 0;
    if (!product_size(layers, experts, count)) {
        err = "EXL3 Flash: expert layer/name table dimensions overflow";
        return false;
    }
    result.assign(count, {});
    for (const auto& projection : map.projections) {
        if (projection.layer < 0 || projection.expert < 0 || projection.layer >= layers || projection.expert >= experts)
            continue;
        std::string* destination = nullptr;
        switch (projection.kind) {
        case ProjectionKind::expert_gate: destination = &result[static_cast<size_t>(projection.layer * experts + projection.expert)].gate; break;
        case ProjectionKind::expert_up: destination = &result[static_cast<size_t>(projection.layer * experts + projection.expert)].up; break;
        case ProjectionKind::expert_down: destination = &result[static_cast<size_t>(projection.layer * experts + projection.expert)].down; break;
        default: break;
        }
        if (destination == nullptr) continue;
        if (!destination->empty() && *destination != projection.source_name) {
            err = "EXL3 Flash: duplicate expert projection mapping for layer " + std::to_string(projection.layer) +
                  " expert " + std::to_string(projection.expert);
            return false;
        }
        *destination = projection.source_name;
    }
    for (int64_t layer = 0; layer < layers; ++layer) {
        for (int64_t expert = 0; expert < experts; ++expert) {
            const auto& names = result[static_cast<size_t>(layer * experts + expert)];
            if (names.gate.empty() || names.up.empty() || names.down.empty()) {
                err = "EXL3 Flash: expert mapping is incomplete at layer " + std::to_string(layer) + " expert " +
                      std::to_string(expert);
                return false;
            }
        }
    }
    return true;
}

std::filesystem::path find_ple_sidecar(const Archive& archive, const std::string& requested) {
    if (!requested.empty()) {
        const std::filesystem::path path(requested);
        return path.is_absolute() ? path : archive.directory() / path;
    }
    for (const auto& sidecar : archive.sidecars())
        if (sidecar.role == strata::artifact::exl3::SidecarRole::specialized_ngram_codec)
            return archive.directory() / sidecar.file;
    return {};
}

bool capacity_error(const std::string& error) {
    return error.find("cache") != std::string::npos || error.find("allocation") != std::string::npos ||
           error.find("resident") != std::string::npos || error.find("VRAM") != std::string::npos;
}

// ---- THE DEVICE CACHE ---------------------------------------------------------------------------
//
// **WHAT IT IS.**  The experts the GPU keeps in VRAM.  A routed expert that is resident is computed by the
// GPU; one that is not is read from the source and computed on the CPU.  The cache therefore decides how much
// of the model the disk and the CPU have to carry.
//
// **WHY IT IS SIZED HERE.**  The old code asked for EVERY (layer, expert) pair of the model: for Qwen3.8 that
// is 48 x 512 = 24576 pairs, 45.77 GiB of compressed weights, which fits in no card the engine runs on.  The
// runtime refused the list for capacity, the caller read that as a capacity error and reopened with an empty
// list - the "no device cache" mode, in which the GPU computes no expert at all and every routed expert is
// read from the disk and computed on the CPU.  The room is now what the DEVICE reports, so a card with less
// VRAM gets a smaller cache and a card with more gets a larger one; no number in this file decides it.

/// The order the resident pairs are admitted in.  The routing is per layer and data dependent, so the order
/// decides which layers, and which experts of them, the cache covers.
enum class CacheOrder {
    Spread,   ///< every layer gets the same number of experts: the experts in index order, the layers in turn
    Visited,  ///< the order the layers are visited: a layer's experts before the next layer's
    Profile,  ///< the ranking a profile file gives: the hottest pairs first
};

const char* cache_order_name(CacheOrder order) {
    switch (order) {
    case CacheOrder::Spread: return "spread";
    case CacheOrder::Visited: return "visited";
    case CacheOrder::Profile: return "profile";
    }
    return "spread";
}

/// STRATA_EXL3_DEVICE_CACHE_ORDER: which order the resident pairs are admitted in.
CacheOrder device_cache_order() {
    const char* text = std::getenv("STRATA_EXL3_DEVICE_CACHE_ORDER");
    if (text == nullptr || text[0] == '\0') return CacheOrder::Spread;
    if (std::strcmp(text, "visited") == 0) return CacheOrder::Visited;
    if (std::strcmp(text, "profile") == 0) return CacheOrder::Profile;
    return CacheOrder::Spread;
}

/// STRATA_EXL3_DEVICE_CACHE: 0 turns the device cache off (every routed expert is computed on the CPU).
bool device_cache_enabled() {
    const char* text = std::getenv("STRATA_EXL3_DEVICE_CACHE");
    if (text == nullptr || text[0] == '\0') return true;
    return !(std::strcmp(text, "0") == 0 || std::strcmp(text, "off") == 0 || std::strcmp(text, "no") == 0 ||
             std::strcmp(text, "n") == 0 || std::strcmp(text, "false") == 0);
}

/// STRATA_EXL3_DEVICE_CACHE_MIB: the cache's budget in MiB; negative (the default) means the room the
/// device reports.  0 is the same as turning the cache off.
int64_t device_cache_budget_mib() {
    const char* text = std::getenv("STRATA_EXL3_DEVICE_CACHE_MIB");
    if (text == nullptr || text[0] == '\0') return -1;
    char* end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    if (end == text || (end != nullptr && *end != '\0') || value < 0) return -1;
    return static_cast<int64_t>(value);
}

/// STRATA_EXL3_DEVICE_CACHE_PROFILE: the profile file the `profile` order ranks the pairs with.
std::string device_cache_profile_path() {
    const char* text = std::getenv("STRATA_EXL3_DEVICE_CACHE_PROFILE");
    return text == nullptr ? std::string() : std::string(text);
}

/// The reserve the model keeps for everything but the expert cache: the driver's own use, the graph capture's
/// and the room the runtime's plan images take.  It is a share of the card between two bounds, so a small card
/// keeps a smaller absolute share for the experts than a large one.
uint64_t device_cache_reserve(uint64_t total_bytes) {
    return std::clamp<uint64_t>(total_bytes / 32, 192ull << 20, 768ull << 20);
}

/// The pairs the cache holds, and what they cost.
struct DeviceCachePlan {
    std::vector<Exl3ExpertRuntime::ResidentPair> resident;
    uint64_t bytes = 0;          ///< the bytes the resident pairs take (the slots are 256-byte aligned)
    uint64_t budget = 0;         ///< the bytes the plan was given
    int64_t layers = 0;          ///< the layers the stage owns
    int64_t experts = 0;         ///< the experts of a layer
    int64_t per_layer_min = 0;   ///< the fewest experts a layer got
    int64_t per_layer_max = 0;   ///< the most a layer got
};

/// Plans the resident pairs of one stage: the pairs that fit in `budget`, in the given order.
///
/// The order decides the shape of the cache, and the shape decides the hit rate:
///   * `Spread` gives every layer the same number of experts.  A position routes a few experts in EVERY layer,
///     so a cache that filled a few layers and left the rest empty would miss on those layers for the whole
///     run - the same confinement the native path's per-layer admission was written against.
///   * `Visited` fills a layer's experts before the next layer's: the order the layers are visited in.
///   * `Profile` takes the pairs a profile file ranks first (the hottest ones), the rest in `Spread` order.
///
/// Each pair is charged its own layout's size: the EXL3 formats differ per layer, so the largest blob is not
/// the size of a slot.
DeviceCachePlan plan_device_cache(const Exl3ExpertSource& source, int64_t layer_begin, int64_t layer_end,
                                  int64_t n_expert, uint64_t budget, CacheOrder order,
                                  const std::vector<std::pair<int32_t, int32_t>>& ranked) {
    DeviceCachePlan plan;
    plan.budget = budget;
    plan.layers = layer_end - layer_begin;
    plan.experts = n_expert;
    const int64_t layers = plan.layers;
    if (layers <= 0 || n_expert <= 0) return plan;
    const size_t pairs = static_cast<size_t>(layers) * static_cast<size_t>(n_expert);
    std::vector<uint64_t> bytes(pairs, 0);
    for (int64_t l = 0; l < layers; ++l)
        for (int64_t e = 0; e < n_expert; ++e) {
            const auto* layout = source.exl3_layout(layer_begin + l, e);
            if (layout == nullptr) continue;
            bytes[static_cast<size_t>(l) * static_cast<size_t>(n_expert) + static_cast<size_t>(e)] =
                (layout->total_bytes + 255u) & ~uint64_t(255u);
        }
    // The order the pairs are admitted in.
    std::vector<std::pair<int32_t, int32_t>> order_pairs;
    order_pairs.reserve(pairs);
    std::vector<uint8_t> taken(pairs, 0);
    if (order == CacheOrder::Profile) {
        for (const auto& pair : ranked) {
            if (pair.first < layer_begin || pair.first >= layer_end) continue;
            if (pair.second < 0 || pair.second >= n_expert) continue;
            const size_t at = static_cast<size_t>(pair.first - layer_begin) * static_cast<size_t>(n_expert) +
                              static_cast<size_t>(pair.second);
            if (taken[at] != 0) continue;
            taken[at] = 1;
            order_pairs.emplace_back(pair.first, pair.second);
        }
    }
    const bool by_layer = order == CacheOrder::Visited;
    for (int64_t outer = 0; outer < (by_layer ? layers : n_expert); ++outer)
        for (int64_t inner = 0; inner < (by_layer ? n_expert : layers); ++inner) {
            const int64_t l = by_layer ? outer : inner;
            const int64_t e = by_layer ? inner : outer;
            const size_t at = static_cast<size_t>(l) * static_cast<size_t>(n_expert) + static_cast<size_t>(e);
            if (taken[at] != 0) continue;
            taken[at] = 1;
            order_pairs.emplace_back(static_cast<int32_t>(layer_begin + l), static_cast<int32_t>(e));
        }
    // The pairs that fit.
    uint64_t left = budget;
    std::vector<int64_t> per_layer(static_cast<size_t>(layers), 0);
    for (const auto& pair : order_pairs) {
        const size_t at = static_cast<size_t>(pair.first - layer_begin) * static_cast<size_t>(n_expert) +
                          static_cast<size_t>(pair.second);
        const uint64_t cost = bytes[at];
        if (cost == 0 || cost > left) continue;
        left -= cost;
        plan.bytes += cost;
        plan.resident.push_back(pair);
        ++per_layer[static_cast<size_t>(pair.first - layer_begin)];
    }
    for (const int64_t count : per_layer) {
        plan.per_layer_min = std::min(plan.per_layer_min, count);
        plan.per_layer_max = std::max(plan.per_layer_max, count);
    }
    if (plan.resident.empty()) plan.per_layer_min = 0;
    return plan;
}

/// Opens one stage's expert runtime on a device cache sized from what the device reports.
///
/// The pairs that do not fit stay in the source and are computed on the CPU: that miss path is the one the
/// engine has always had, so a card with no room for the cache still runs.  It says so when it happens,
/// because the run's numbers depend on it (the prepare detail's miss count).
bool open_device_cache(Exl3ExpertSource& source, strata::kernels::cpu::ExpertPool& pool, Exl3ExpertRuntime& runtime,
                       size_t stage_index, int64_t layer_begin, int64_t layer_end, int64_t n_expert,
                       int64_t n_layers, int64_t max_rows, int64_t routed_k, bool allow_cpu_miss, std::string& err) {
    const auto no_cache = [&](const char* why) {
        if (!allow_cpu_miss) {
            err = "EXL3 Flash: stage " + std::to_string(stage_index) + " expert runtime: the device cache is " +
                  why + ", and a CPU miss is not allowed";
            return false;
        }
        std::string ignored;
        if (!runtime.open(source, pool, max_rows, routed_k, {}, ignored, layer_begin, layer_end)) {
            err = "EXL3 Flash: stage " + std::to_string(stage_index) + " expert runtime: " + ignored;
            return false;
        }
        std::fprintf(stderr, "strata exl3-flash: stage %zu device cache: %s; every routed expert is read from "
                             "the source and computed on the CPU\n", stage_index, why);
        return true;
    };
    if (!device_cache_enabled()) return no_cache("off (STRATA_EXL3_DEVICE_CACHE=0)");
    size_t free_b = 0, total_b = 0;
    if (!strata::gpu::mem_info(&free_b, &total_b))
        return no_cache("the device does not report its free memory");
    CacheOrder order = device_cache_order();
    std::vector<std::pair<int32_t, int32_t>> ranked;
    if (order == CacheOrder::Profile) {
        const std::string path = device_cache_profile_path();
        std::string profile_error;
        int64_t slots = 0;
        if (path.empty() || !read_expert_profile(path, n_layers, n_expert, ranked, slots, profile_error)) {
            std::fprintf(stderr, "strata exl3-flash: stage %zu device cache: the profile order needs a profile "
                                 "file for this model (%s); the spread order is used instead\n",
                         stage_index, path.empty() ? "STRATA_EXL3_DEVICE_CACHE_PROFILE is not set"
                                                   : profile_error.c_str());
            ranked.clear();
            order = CacheOrder::Spread;
        }
    }
    const uint64_t reserve = device_cache_reserve(static_cast<uint64_t>(total_b));
    uint64_t room = free_b > reserve ? static_cast<uint64_t>(free_b) - reserve : 0;
    const int64_t asked_mib = device_cache_budget_mib();
    if (asked_mib >= 0 && (static_cast<uint64_t>(asked_mib) << 20) < room)
        room = static_cast<uint64_t>(asked_mib) << 20;
    // The cache may still not fit: the free figure is what the driver reports, and the driver's own use can
    // grow.  Then the room is halved and the plan made again; a card with no room at all runs without a cache.
    for (int attempt = 0; attempt < 4; ++attempt) {
        const DeviceCachePlan plan =
            plan_device_cache(source, layer_begin, layer_end, n_expert, room, order, ranked);
        if (plan.resident.empty()) break;
        std::string open_error;
        if (runtime.open(source, pool, max_rows, routed_k, plan.resident, open_error, layer_begin, layer_end)) {
            std::fprintf(stderr,
                         "strata exl3-flash: stage %zu device cache: %lld experts resident (%.2f GiB) of %lld, "
                         "%.2f GiB free, %.2f GiB reserved, order %s, %lld-%lld experts per layer\n",
                         stage_index, (long long) plan.resident.size(), (double) plan.bytes / 1073741824.0,
                         (long long) (plan.layers * plan.experts), (double) free_b / 1073741824.0,
                         (double) reserve / 1073741824.0, cache_order_name(order),
                         (long long) plan.per_layer_min, (long long) plan.per_layer_max);
            return true;
        }
        if (!capacity_error(open_error)) {
            // Not a capacity question: the cache could not be filled (a read that failed, a pool that would
            // not prepare).  The cache is an optimisation; the miss path is the engine's own, so the run goes
            // on without a cache when a CPU miss is allowed, and says what it did.
            std::fprintf(stderr, "strata exl3-flash: stage %zu device cache: the cache could not be opened: %s\n",
                         stage_index, open_error.c_str());
            if (!allow_cpu_miss) {
                err = "EXL3 Flash: stage " + std::to_string(stage_index) + " expert runtime: " + open_error;
                return false;
            }
            return no_cache("it could not be opened");
        }
        std::fprintf(stderr, "strata exl3-flash: stage %zu device cache: the cache did not fit (%.2f GiB asked, "
                             "%.2f GiB free): %s\n", stage_index, (double) plan.bytes / 1073741824.0,
                     (double) free_b / 1073741824.0, open_error.c_str());
        room /= 2;
    }
    return no_cache("no room for the experts");
}

uint64_t gdn_state_floats(const ModelGeometry& geometry) {
    return static_cast<uint64_t>(geometry.ssm_state_size) * static_cast<uint64_t>(geometry.ssm_v_heads) *
               static_cast<uint64_t>(geometry.ssm_state_size) +
           static_cast<uint64_t>(geometry.ssm_conv_channels) * static_cast<uint64_t>(geometry.ssm_d_conv - 1);
}

void point_gdn(const ModelGeometry& geometry, int64_t layer, SessionState& session) {
    if (is_qsa_layer(geometry, layer)) return;
    int64_t ordinal = 0;
    for (int64_t l = 0; l < layer; ++l)
        if (!is_qsa_layer(geometry, l)) ++ordinal;
    session.gdn.state = session.gdn_state +
                        static_cast<size_t>(ordinal - session.gdn_ord0) * gdn_state_floats(geometry);
    session.gdn.conv_state = session.gdn.state +
                             static_cast<uint64_t>(geometry.ssm_state_size) * geometry.ssm_v_heads *
                                 geometry.ssm_state_size;
}

QsaState& qsa_for(const ModelGeometry& geometry, int64_t layer, SessionState& session) {
    int64_t ordinal = 0;
    for (int64_t l = 0; l < layer; ++l)
        if (is_qsa_layer(geometry, l)) ++ordinal;
    return session.qsa_states[ordinal];
}

void stage_token(const ModelGeometry& geometry, int64_t pos, int32_t pos_base, SessionState& session) {
    strata::kernels::QsaShapes shapes = strata::kernels::qsa_real_shapes();
    shapes.n_head = geometry.n_head;
    shapes.n_head_kv = geometry.n_head_kv;
    shapes.head_dim = geometry.head_dim;
    shapes.idx_n_head = geometry.idx_q_heads;
    shapes.idx_dim = geometry.idx_key_dim;
    for (int64_t j = 0; j < session.qsa_alloc; ++j) {
        QsaState& state = session.qsa_states[session.qsa_ord0 + j];
        strata::kernels::qsa_step_fill(state.host_step, pos, shapes);
        for (int64_t h = 0; h < geometry.n_head; ++h)
            state.host_pos[h] = static_cast<int32_t>(pos_base + pos);
    }
}

void destroy_graphs(std::vector<strata::gpu::Graph*>& graphs) noexcept {
    for (auto*& graph : graphs) {
        if (graph != nullptr) strata::gpu::graph_destroy(graph);
        graph = nullptr;
    }
    graphs.clear();
}

// Host-side timing of one forward pass, printed when STRATA_EXL3_FLASH_TIMING is set.
struct HostTiming {
    bool enabled = false;
    std::chrono::steady_clock::time_point mark;
    double embedding_ms = 0;
    double ple_issue_ms = 0;
    double ple_collect_ms = 0;
    double pre_graph_ms = 0;
    double expert_prepare_ms = 0;
    double expert_wait_ms = 0;
    double merge_ms = 0;
    double post_graph_ms = 0;
    double bridge_ms = 0;
    double head_ms = 0;

    double total_ms() const {
        return embedding_ms + ple_issue_ms + ple_collect_ms + pre_graph_ms + expert_prepare_ms + expert_wait_ms +
               merge_ms + post_graph_ms + bridge_ms + head_ms;
    }
};

bool host_timing_enabled() {
    static const bool on = std::getenv("STRATA_EXL3_FLASH_TIMING") != nullptr;
    return on;
}

/// STRATA_EXL3_DEVICE_PLAN: 0 keeps the per-layer host plan (the doorbell wait, the host hit plan and the
/// merge round trip) even where the device-resident plan would apply.
bool device_route_plan_enabled() {
    const char* text = std::getenv("STRATA_EXL3_DEVICE_PLAN");
    return !(text != nullptr && text[0] == '0' && text[1] == '\0');
}

/// STRATA_EXL3_TOKEN_GRAPH: 0 keeps the per-layer graph launches even where a stage's whole token
/// was captured as one graph - the same values, kept as the comparison path for the graph's effect.
bool token_graph_enabled() {
    const char* text = std::getenv("STRATA_EXL3_TOKEN_GRAPH");
    return !(text != nullptr && text[0] == '0' && text[1] == '\0');
}

/// STRATA_EXL3_RESIDENT_GRAPH: 0 submits the resident expert flow's launches one by one instead of
/// replaying the captured graph - the same values, kept as the comparison path for the graph's effect.
bool resident_flow_graph_enabled() {
    const char* text = std::getenv("STRATA_EXL3_RESIDENT_GRAPH");
    return !(text != nullptr && text[0] == '0' && text[1] == '\0');
}

double ms_since(std::chrono::steady_clock::time_point& mark) {
    const auto now = std::chrono::steady_clock::now();
    const double value = std::chrono::duration<double, std::milli>(now - mark).count();
    mark = now;
    return value;
}

/// Waits for a pre graph's doorbell ring the way the native session loop does: a poll that enters
/// the driver every iteration (on this driver a device write to mapped pinned memory is host
/// visible only after one), ending at the ring - which the pre graph may ring before its own end -
/// or at the graph's end.  A graph that ended without ringing is refused: the window the host
/// would read is then the previous layer's.
bool wait_pre_doorbell(const Doorbell& doorbell, const strata::gpu::Event* probe, std::string& err) {
    volatile const uint32_t* const seq = doorbell.h_seq;
    for (;;) {
        const strata::gpu::Status status = strata::gpu::event_status(probe);
        if (status == strata::gpu::Status::Failed) {
            err = strata::gpu::last_error();
            return false;
        }
        if (*seq != 0) return true;
        if (status == strata::gpu::Status::Done) {
            err = "the pre graph ended without ringing the doorbell";
            return false;
        }
        STRATA_SPIN_PAUSE();
    }
}

/// THE RESIDUAL'S PATH ACROSS A STAGE BOUNDARY, TAKEN FROM WHAT A REAL COPY DOES.
///
/// The residual is the only data that leaves one stage's device.  Where the runtime copies between
/// the two devices' memories directly - the driver moves the bytes and no host buffer is touched -
/// the boundary takes that copy; everywhere else the residual goes through the host bridge, which
/// is the engine's original path.  The choice is made with a copy that is issued and verified, not
/// from a capability flag: `peer_access` is REPORTED beside it, and the extension answers "no" on
/// hosts whose devices still copy directly.
bool residual_copy_direct(int from_device, int to_device, size_t bytes, bool& peer, std::string& detail) {
    peer = strata::gpu::peer_access(from_device, to_device);
    const size_t probe_bytes = std::min<size_t>(bytes, 64 * 1024);
    std::vector<unsigned char> pattern(probe_bytes);
    for (size_t i = 0; i < probe_bytes; ++i) pattern[i] = static_cast<unsigned char>(i * 31 + 7);
    std::vector<unsigned char> readback(probe_bytes, 0);
    void* src = nullptr;
    void* dst = nullptr;
    {
        const OnDevice on(from_device);
        src = strata::gpu::alloc_device(probe_bytes);
        if (src != nullptr && !strata::gpu::copy(src, pattern.data(), probe_bytes)) {
            strata::gpu::free(src);
            src = nullptr;
        }
    }
    {
        const OnDevice on(to_device);
        dst = strata::gpu::alloc_device(probe_bytes);
    }
    bool direct = src != nullptr && dst != nullptr;
    if (direct) {
        const OnDevice on(from_device);
        direct = strata::gpu::copy(dst, src, probe_bytes);
        if (!direct) detail = strata::gpu::last_error();
    } else {
        detail = "no room for the probe copy";
    }
    if (direct) {
        const OnDevice on(to_device);
        direct = strata::gpu::copy(readback.data(), dst, probe_bytes) &&
                 std::memcmp(readback.data(), pattern.data(), probe_bytes) == 0;
        if (!direct) detail = "the copy did not carry the bytes";
    }
    if (src != nullptr) strata::gpu::free(src);
    if (dst != nullptr) strata::gpu::free(dst);
    return direct;
}

/// Drains the PLE's row ticket if a forward fails between the issue and the
/// collect.  The table allows one token in flight, so a ticket left behind
/// would refuse the next token's read and the model would stop with an error
/// that names the symptom rather than the cause.
class PleTicketGuard {
public:
    void arm(strata::kernels::PleTable* table, float* scratch) {
        table_ = table;
        scratch_ = scratch;
        armed_ = true;
    }
    void disarm() { armed_ = false; }
    ~PleTicketGuard() {
        if (!armed_ || table_ == nullptr || scratch_ == nullptr) return;
        std::string ignored;
        try {
            (void) table_->collect(scratch_, ignored);
        } catch (...) {   // a destructor that throws while the stack unwinds would end the process
        }
    }

private:
    strata::kernels::PleTable* table_ = nullptr;
    float* scratch_ = nullptr;
    bool armed_ = false;
};

}  // namespace

Exl3FlashStageRange exl3_flash_stage_range(int64_t layer_count, int64_t stage_count, int64_t stage_index) {
    Exl3FlashStageRange range;
    if (layer_count <= 0 || stage_count <= 0 || stage_index < 0 || stage_index >= stage_count) return range;
    range.layer_begin = layer_count * stage_index / stage_count;
    range.layer_end = layer_count * (stage_index + 1) / stage_count;
    return range;
}

bool exl3_flash_stage_ranges(int64_t layer_count, int64_t stage_count, const std::vector<int64_t>& counts,
                             std::vector<Exl3FlashStageRange>& ranges, std::string& err) {
    ranges.clear();
    err.clear();
    if (layer_count <= 0) { err = "the model has no layers"; return false; }
    if (stage_count <= 0) { err = "at least one stage is required"; return false; }
    if (stage_count > layer_count) {
        err = "the stage count exceeds the model's layer count";
        return false;
    }
    if (!counts.empty() && static_cast<int64_t>(counts.size()) != stage_count) {
        err = "the layer counts name " + std::to_string(counts.size()) + " stages but " +
              std::to_string(stage_count) + " stages were selected";
        return false;
    }
    ranges.reserve(static_cast<size_t>(stage_count));
    int64_t begin = 0;
    for (int64_t index = 0; index < stage_count; ++index) {
        int64_t count = counts.empty() ? 0 : counts[static_cast<size_t>(index)];
        if (counts.empty()) {
            const Exl3FlashStageRange even = exl3_flash_stage_range(layer_count, stage_count, index);
            count = even.layer_end - even.layer_begin;
        }
        if (count <= 0) {
            err = "stage " + std::to_string(index) + " gets " + std::to_string(count) +
                  " layers; every stage must get at least one layer";
            ranges.clear();
            return false;
        }
        if (count > layer_count - begin) {
            err = "the layer counts cover more than the model's " + std::to_string(layer_count) + " layers";
            ranges.clear();
            return false;
        }
        ranges.push_back({begin, begin + count});
        begin += count;
    }
    if (begin != layer_count) {
        err = "the layer counts cover " + std::to_string(begin) + " layers but the model has " +
              std::to_string(layer_count);
        ranges.clear();
        return false;
    }
    return true;
}

int64_t exl3_flash_ple_stage(const std::vector<Exl3FlashStageRange>& ranges) {
    for (size_t index = 0; index < ranges.size(); ++index)
        if (ranges[index].layer_begin <= PLE_LAYER && PLE_LAYER < ranges[index].layer_end)
            return static_cast<int64_t>(index);
    return -1;
}

int64_t exl3_flash_ple_stage(int64_t layer_count, int64_t stage_count) {
    std::vector<Exl3FlashStageRange> ranges;
    std::string ignored;
    if (!exl3_flash_stage_ranges(layer_count, stage_count, {}, ranges, ignored)) return -1;
    return exl3_flash_ple_stage(ranges);
}

bool exl3_flash_mtp_roles_complete(const strata::artifact::exl3::ModelMap& mtp, std::string& err) {
    err.clear();
    if (mtp.geometry.layers != 1) {
        err = "the MTP map must hold exactly one layer";
        return false;
    }
    if (mtp.geometry.experts <= 0) {
        err = "the MTP map names no routed experts";
        return false;
    }
    size_t fc = 0, qsa = 0, hc = 0, experts = 0;
    for (const auto& item : mtp.projections) {
        fc += item.kind == ProjectionKind::mtp_fc;
        qsa += item.kind == ProjectionKind::qsa_q || item.kind == ProjectionKind::qsa_k ||
               item.kind == ProjectionKind::qsa_v || item.kind == ProjectionKind::qsa_out ||
               item.kind == ProjectionKind::qsa_indexer_q || item.kind == ProjectionKind::qsa_indexer_k;
        hc += item.kind == ProjectionKind::hc_norm || item.kind == ProjectionKind::hc_down ||
              item.kind == ProjectionKind::hc_up || item.kind == ProjectionKind::hc_inject;
        experts += item.kind == ProjectionKind::expert_gate || item.kind == ProjectionKind::expert_up ||
                   item.kind == ProjectionKind::expert_down;
    }
    if (fc != 2) {
        err = "the MTP map must carry both mtp_fc projections";
        return false;
    }
    if (qsa != 6) {
        err = "the MTP map must carry the full QSA set";
        return false;
    }
    if (hc != 8) {
        err = "the MTP map must carry the eight hyper-connection entries";
        return false;
    }
    if (experts != static_cast<size_t>(3 * mtp.geometry.experts)) {
        err = "the MTP map must carry the complete expert gate/up/down set";
        return false;
    }
    return true;
}

struct Exl3FlashModel::Impl {
    struct Stage {
        int device = -1;
        int64_t layer_begin = 0;
        int64_t layer_end = 0;
        WeightTable table;
        std::unique_ptr<Exl3WeightSet> weights;
        std::unique_ptr<Exl3ExpertSource> source;
        std::unique_ptr<strata::kernels::cpu::ExpertPool> pool;
        std::unique_ptr<Exl3ExpertRuntime> runtime;
        SessionState session;
        void* session_arena = nullptr;
        size_t session_arena_bytes = 0;
        Doorbell doorbell;
        void* parts_device = nullptr;
        std::vector<float> miss_output;
        std::vector<float> merge_output;
        std::vector<float> merge_upload;
        float* embedding_device = nullptr;
        std::vector<strata::gpu::Graph*> pre_graphs;
        std::vector<strata::gpu::Graph*> post_graphs;
        // The resident expert flow's own graphs (one per layer): the clear of the expert rows, the device
        // route plan and the grouped experts' kernels, captured where the cache is fully resident.
        std::vector<strata::gpu::Graph*> expert_graphs;
        // One graph for the stage's whole token (every layer's pre, resident expert flow and post in
        // sequence), captured where the cache is fully resident: one launch per token per stage
        // instead of three per layer.  Null where the stage is not fully resident.
        strata::gpu::Graph* token_graph = nullptr;
    };

    Exl3FlashModelOptions options;
    std::shared_ptr<const Archive> archive;
    strata::ModelSpec spec;
    ModelGeometry geometry;
    ModelMap map;
    ModelMap mtp_map;
    bool has_mtp = false;
    // THE MTP DRAFT BLOCK'S OWNERS.  Bound on the last stage's device when
    // include_mtp is set: dense weights over the one-layer MTP map, the MTP
    // expert names/source/pool, and one layer of session state (QSA every
    // layer).  The draft runtime gets a device cache of its own at open();
    // when every draft expert is resident the draft runs from its captured
    // expert graph, otherwise it falls back to the CPU pool.
    std::unique_ptr<WeightTable> mtp_table;
    std::unique_ptr<Exl3WeightSet> mtp_weights;
    std::vector<ExpertTensorNames> mtp_expert_names;
    std::unique_ptr<Exl3ExpertSource> mtp_source;
    std::unique_ptr<strata::kernels::cpu::ExpertPool> mtp_pool;
    ModelGeometry mtp_geometry;
    std::unique_ptr<Exl3ExpertRuntime> mtp_runtime;
    strata::gpu::Graph* mtp_expert_graph = nullptr;  ///< captured resident expert flow of the draft block
    strata::gpu::Graph* mtp_body_graph = nullptr;    ///< the whole draft body (norms, projections, block, head) in one graph
    bool mtp_device_draft = false;                   ///< draft experts run on the device cache, not the CPU pool
    strata::gpu::Stream mtp_stream = nullptr;
    void* mtp_parts_device = nullptr;
    void* mtp_fc_scratch = nullptr;
    std::vector<int32_t> mtp_host_ids;
    std::vector<float> mtp_host_weights;
    std::vector<float> mtp_host_x;
    std::vector<float> mtp_cpu_out;
    void* mtp_session_arena = nullptr;
    size_t mtp_session_arena_bytes = 0;
    SessionState mtp_session;
    int mtp_device = -1;
    std::vector<ExpertTensorNames> expert_names;
    std::vector<Stage> stages;
    std::unique_ptr<NativeEmbed> embedding;
    std::unique_ptr<Exl3Ple> ple;
    std::unique_ptr<strata::kernels::PleTable> ple_table;
    int embedding_device = -1;
    int ple_device = -1;
    int ple_stage_index = -1;
    // THE PLE'S TOKEN WINDOW: the token being decoded and the two before it, oldest first.  It is
    // the MODEL's, not a stage's: the n-gram context is read on the host, and the PLE's device half
    // lives on the stage that runs the PLE layer (the home stage), which is where the rows are
    // consumed.  A stage boundary therefore carries no part of the PLE.
    int32_t ple_prev[2] = {-1, -1};
    int32_t ple_token = -1;
    void* logits_device = nullptr;
    std::vector<float> logits_host;
    std::vector<float> bridge_host;
    // HOW THE RESIDUAL CROSSES A STAGE BOUNDARY.  `residual_direct` is per boundary: the devices
    // copy between their memories (the driver moves the bytes, no host buffer), else the residual
    // goes through `bridge_host`.  The crossing is issued on the sending device's queue right
    // after the stage's last post graph and marked with `cross_event`; the receiving stage waits
    // that one event where the data crosses.  `doorbell_probe` marks each pre graph so the host
    // can poll its doorbell ring instead of draining the queue.
    std::vector<bool> residual_direct;
    strata::gpu::Event* cross_event = nullptr;
    strata::gpu::Event* doorbell_probe = nullptr;
    bool initialized = false;
    bool seeded = false;
    int64_t next_position = 0;
    bool previous_fused_gr = false;
    bool previous_shared_early = false;
    bool policy_saved = false;
    strata::kernels::RopeScaling previous_rope;
    bool previous_gdn_silu = false;
    bool model_globals_saved = false;
    bool opened = false;
};

Exl3FlashModel::Exl3FlashModel() : impl_(std::make_unique<Impl>()) {}

Exl3FlashModel::~Exl3FlashModel() { close(); }

bool Exl3FlashModel::open(const std::filesystem::path& directory, const Exl3FlashModelOptions& options,
                          std::string& err) {
    close();
    err.clear();
    struct OpenCleanup {
        Exl3FlashModel* owner;
        ~OpenCleanup() { if (owner != nullptr) owner->close(); }
    } cleanup{this};
    try {
        if (directory.empty()) { err = "EXL3 Flash: archive directory is required"; return false; }
        if (options.vision) {
            err = "EXL3 Flash: vision input is not supported by the initial AR coordinator";
            return false;
        }
        if (options.pool_workers < 0) {
            err = "EXL3 Flash: pool_workers must be nonnegative";
            return false;
        }
        if (options.max_rows != 1) {
            err = "EXL3 Flash: the initial coordinator accepts exactly one autoregressive row";
            return false;
        }
        if (options.stage_devices.empty()) {
            const int count = Runtime::count();
            if (count <= 0) { err = "EXL3 Flash: no runtime GPU is available for stage construction"; return false; }
            impl_->options = options;
            impl_->options.stage_devices.resize(static_cast<size_t>(count));
            for (int i = 0; i < count; ++i) impl_->options.stage_devices[static_cast<size_t>(i)] = i;
        } else {
            impl_->options = options;
        }
        if (impl_->options.stage_devices.empty()) {
            err = "EXL3 Flash: at least one stage device is required";
            return false;
        }
        std::set<int> unique_devices;
        for (const int device : impl_->options.stage_devices) {
            if (device < 0 || device >= Runtime::count() || !unique_devices.insert(device).second) {
                err = "EXL3 Flash: stage device ordinals must be valid and unique";
                return false;
            }
        }

        impl_->archive = std::make_shared<Archive>(directory);
        if (!strata::artifact::exl3::read_exl3_model_spec(*impl_->archive, impl_->spec, err, false)) return false;
        if (impl_->spec.arch != strata::ModelArch::Qwen4Exp) {
            err = "EXL3 Flash: the initial coordinator requires the Qwen3.8 Qwen4Exp architecture";
            return false;
        }
        if (!geometry_from_spec(impl_->spec, impl_->geometry, err)) {
            err = "EXL3 Flash: " + err;
            return false;
        }
        const int64_t context_cells = options.max_cells > 0 ? options.max_cells : impl_->spec.context_length;
        if (context_cells <= 0 || context_cells > impl_->spec.context_length) {
            err = "EXL3 Flash: max_cells must be within max_position_embeddings";
            return false;
        }
        impl_->options.max_cells = context_cells;
        {
            using RST = strata::kernels::RopeScalingType;
            strata::kernels::RopeScaling scaling;
            scaling.freq_base = impl_->spec.rope_base;
            const std::string& type = impl_->spec.rope_scaling_type;
            if (type.empty() || type == "none") scaling.type = RST::None;
            else if (type == "linear") scaling.type = RST::Linear;
            else if (type == "yarn") scaling.type = RST::YaRN;
            else {
                err = "EXL3 Flash: unsupported rope scaling type " + type;
                return false;
            }
            if (scaling.type != RST::None) {
                scaling.factor = impl_->spec.rope_scaling_factor > 1.0 ? impl_->spec.rope_scaling_factor : 1.0;
                scaling.orig_ctx = impl_->spec.rope_scaling_orig_ctx >= 1.0
                                       ? impl_->spec.rope_scaling_orig_ctx
                                       : static_cast<double>(impl_->spec.context_length);
                scaling.ext_factor = scaling.type == RST::YaRN ? 1.0 : 0.0;
            }
            if (const char* why = strata::kernels::rope_scaling_invalid(scaling)) {
                err = std::string("EXL3 Flash: rope scaling: ") + why;
                return false;
            }
            impl_->previous_rope = strata::kernels::rope_scaling();
            impl_->previous_gdn_silu = strata::kernels::gdn_silu_gate();
            impl_->model_globals_saved = true;
            strata::kernels::rope_scaling_set(scaling);
            // Qwen3.8/Qwen4Exp uses sigmoid(z) in the GDN closing gate;
            // capture bakes this choice into every GDN graph.
            strata::kernels::gdn_set_silu_gate(false);
        }
        if (!strata::artifact::exl3::build_model_map(*impl_->archive, impl_->map, err, false)) {
            err = "EXL3 Flash: " + err;
            return false;
        }
        for (const auto& unsupported : impl_->map.unsupported) {
            if (unsupported != kPlePending) {
                err = "EXL3 Flash: model map is not runnable: " + unsupported;
                return false;
            }
        }
        if (impl_->options.include_mtp) {
            // The legacy include_mtp switch stays Qwen3.5-only: the Qwen3.8
            // draft block is a separate one-layer map built from the trunk.
            // Forward stays single-token AR until the draft/verify loop lands;
            // this only binds the map that loop will draft from.
            if (!strata::artifact::exl3::build_qwen38_mtp_map(*impl_->archive, impl_->map, impl_->mtp_map,
                                                             err)) {
                err = "EXL3 Flash MTP: " + err;
                return false;
            }
            for (const auto& unsupported : impl_->mtp_map.unsupported) {
                err = "EXL3 Flash MTP map is not runnable: " + unsupported;
                return false;
            }
            if (!exl3_flash_mtp_roles_complete(impl_->mtp_map, err)) {
                err = "EXL3 Flash MTP: " + err;
                return false;
            }
            impl_->has_mtp = true;
        }
        if (!make_expert_names(impl_->map, impl_->geometry.n_layers, impl_->geometry.n_expert,
                               impl_->expert_names, err)) return false;
        const int64_t routed_k = options.routed_k > 0 ? options.routed_k : impl_->map.geometry.experts_per_token;
        if (routed_k <= 0 || routed_k > impl_->geometry.n_expert) {
            err = "EXL3 Flash: routed expert count is outside model geometry";
            return false;
        }
        impl_->options.routed_k = routed_k;

        // The embedding is one shared raw table.  NativeEmbed creates a device
        // alias on this first stage and prepare_stream() creates other-stage
        // copies before a graph captures the gather.
        const auto layers_pos = impl_->map.source_prefix.rfind(".layers");
        if (layers_pos == std::string::npos) {
            err = "EXL3 Flash: ModelMap has no canonical model root for embedding";
            return false;
        }
        const std::string embedding_name = options.embedding_name.empty()
                                               ? impl_->map.source_prefix.substr(0, layers_pos) + ".embed_tokens"
                                               : options.embedding_name;
        impl_->embedding = std::make_unique<NativeEmbed>();
        {
            const OnDevice on(impl_->options.stage_devices.front());
            if (!impl_->embedding->load_exl3(*impl_->archive, embedding_name, impl_->geometry.n_embd,
                                             impl_->map.geometry.vocab_size, err)) {
                err = "EXL3 Flash: embedding: " + err;
                return false;
            }
        }
        impl_->embedding_device = impl_->options.stage_devices.front();

        const int64_t layer_count = impl_->geometry.n_layers;
        const int stage_count = static_cast<int>(impl_->options.stage_devices.size());
        if (stage_count > layer_count) {
            err = "EXL3 Flash: stage count cannot exceed the number of model layers";
            return false;
        }
        std::vector<Exl3FlashStageRange> stage_ranges;
        if (!exl3_flash_stage_ranges(layer_count, stage_count, impl_->options.stage_layers, stage_ranges, err)) {
            err = "EXL3 Flash: stage layer split: " + err;
            return false;
        }
        {
            std::string split;
            for (size_t index = 0; index < stage_ranges.size(); ++index) {
                if (index > 0) split += ", ";
                split += std::to_string(stage_ranges[index].layer_begin) + "-" +
                         std::to_string(stage_ranges[index].layer_end);
            }
            std::fprintf(stderr, "strata exl3-flash: %d stages over devices", stage_count);
            for (const int device : impl_->options.stage_devices) std::fprintf(stderr, " %d", device);
            std::fprintf(stderr, ": layers %s\n", split.c_str());
        }
        // Exl3WeightSet::table_ borrows the Stage's WeightTable.  Reserve the
        // complete vector before any Stage is constructed so later stages
        // cannot move an earlier table while it is already registered.
        impl_->stages.reserve(static_cast<size_t>(stage_count));
        for (int stage_index = 0; stage_index < stage_count; ++stage_index) {
            Impl::Stage stage;
            stage.device = impl_->options.stage_devices[static_cast<size_t>(stage_index)];
            const Exl3FlashStageRange& range = stage_ranges[static_cast<size_t>(stage_index)];
            stage.layer_begin = range.layer_begin;
            stage.layer_end = range.layer_end;
            stage.weights = std::make_unique<Exl3WeightSet>();
            stage.source = std::make_unique<Exl3ExpertSource>();
            stage.pool = std::make_unique<strata::kernels::cpu::ExpertPool>(options.pool_workers);
            stage.runtime = std::make_unique<Exl3ExpertRuntime>();
            impl_->stages.push_back(std::move(stage));
            Impl::Stage& current = impl_->stages.back();
            const OnDevice on(current.device);
            Exl3WeightSetOptions weight_options;
            weight_options.device = current.device;
            weight_options.layer_begin = current.layer_begin;
            weight_options.layer_end = current.layer_end;
            weight_options.include_head = current.layer_end == layer_count;
            weight_options.include_final_hc = current.layer_end == layer_count;
            weight_options.max_rows = options.max_rows;
            if (!current.weights->load(*impl_->archive, impl_->map, current.table, weight_options, err)) {
                err = "EXL3 Flash: stage " + std::to_string(stage_index) + " dense weights: " + err;
                return false;
            }
            if (!current.source->open(impl_->archive, layer_count, impl_->geometry.n_expert,
                                      impl_->expert_names, err)) {
                err = "EXL3 Flash: stage " + std::to_string(stage_index) + " expert source: " + err;
                return false;
            }
            // The expert runtime (and the device cache it holds) is opened at the END of open(), after every
            // other device allocation and after the graph capture: the cache takes what the device has left,
            // so it has to be the last thing that asks for VRAM.
        }

        // The Qwen3.8 PLE owner is stage-local at trunk layer 1, while the
        // sidecar row table remains one host reader shared by the session.
        //
        // THE PLE'S HOME STAGE.  The PLE block belongs to one layer, and the layer split gives that
        // layer to exactly one stage.  That stage holds the PLE's device half - the key and value
        // projections' owners, the gathered row, the preprojection destinations, the scratch and the
        // conv history - and it reads the n-gram rows there, so the PLE does not cross a stage
        // boundary: the other stages never touch it.  Only the residual crosses, through the host
        // bridge.
        const std::filesystem::path ple_path = find_ple_sidecar(*impl_->archive, options.ple_sidecar);
        if (ple_path.empty()) {
            err = "EXL3 Flash: the Qwen3.8 specialized n-gram PLE sidecar is required";
            return false;
        }
        const int64_t ple_home = exl3_flash_ple_stage(stage_ranges);
        if (ple_home < 0) {
            err = "EXL3 Flash: no stage owns the required Qwen3.8 PLE layer 1";
            return false;
        }
        const int ple_owner_stage_index = static_cast<int>(ple_home);
        impl_->ple_stage_index = ple_owner_stage_index;
        impl_->ple = std::make_unique<Exl3Ple>();
        impl_->ple_device = impl_->stages[static_cast<size_t>(ple_owner_stage_index)].device;
        {
            const OnDevice on(impl_->ple_device);
            if (!impl_->ple->load(*impl_->archive, impl_->map, impl_->ple_device, options.max_rows, err)) {
                err = "EXL3 Flash: PLE projection owner: " + err;
                return false;
            }
        }
        impl_->ple_table = std::make_unique<strata::kernels::PleTable>();
        if (!impl_->ple_table->open_exl3(ple_path.string(), err)) {
            err = "EXL3 Flash: PLE sidecar table: " + err;
            return false;
        }

        const size_t residual_elements = static_cast<size_t>(impl_->geometry.hc) *
                                         static_cast<size_t>(impl_->geometry.n_embd);
        const size_t parts_elements = static_cast<size_t>(routed_k) *
                                      static_cast<size_t>(impl_->geometry.n_embd);
        impl_->bridge_host.resize(residual_elements);
        impl_->logits_host.resize(static_cast<size_t>(impl_->map.geometry.vocab_size));
        for (size_t stage_index = 0; stage_index < impl_->stages.size(); ++stage_index) {
            auto& stage = impl_->stages[stage_index];
            const OnDevice on(stage.device);
            const uint64_t session_size = session_bytes(impl_->geometry, impl_->options.max_cells, routed_k,
                                                        stage.layer_begin, stage.layer_end);
            stage.session_arena_bytes = static_cast<size_t>(session_size);
            stage.session_arena = strata::gpu::alloc_device(static_cast<size_t>(session_size));
            if (stage.session_arena == nullptr || session_init(impl_->geometry, impl_->options.max_cells, routed_k,
                                                               stage.session_arena, stage.session, stage.layer_begin,
                                                               stage.layer_end) == 0) {
                err = "EXL3 Flash: stage " + std::to_string(stage_index) + " SessionState allocation failed";
                return false;
            }
            if (doorbell_init(impl_->geometry, routed_k, stage.doorbell) == 0) {
                err = "EXL3 Flash: stage " + std::to_string(stage_index) + " expert doorbell allocation failed";
                return false;
            }
            stage.session.db = &stage.doorbell;
            stage.parts_device = strata::gpu::alloc_device(parts_elements * sizeof(float));
            if (stage.parts_device == nullptr) {
                err = "EXL3 Flash: stage " + std::to_string(stage_index) + " expert parts allocation failed";
                return false;
            }
            stage.miss_output.assign(parts_elements, 0.0f);
            stage.merge_output.assign(parts_elements, 0.0f);
            stage.merge_upload.assign(parts_elements, 0.0f);
            if (stage_index == 0) {
                stage.embedding_device = static_cast<float*>(strata::gpu::alloc_device(
                    static_cast<size_t>(impl_->geometry.n_embd) * sizeof(float)));
                if (stage.embedding_device == nullptr) {
                    err = "EXL3 Flash: embedding activation allocation failed";
                    return false;
                }
            }
            session_zero(stage.session, impl_->geometry, nullptr, nullptr);
            if (stage_index == 0) {
                impl_->embedding->gather_one(0, stage.embedding_device, nullptr);
                session_zero(stage.session, impl_->geometry, stage.embedding_device, nullptr);
            }
        }
        // THE RESIDUAL'S PATH ACROSS EACH BOUNDARY, FROM A REAL COPY.  It is probed with the
        // residual's own size and verified, because the devices of one host differ in what their
        // runtime does with a copy between two devices' memories, and the host bridge is kept for
        // the ones that cannot copy directly.
        for (size_t boundary = 0; boundary + 1 < impl_->stages.size(); ++boundary) {
            const int from = impl_->stages[boundary].device;
            const int to = impl_->stages[boundary + 1].device;
            bool peer = false;
            std::string detail;
            const bool direct = residual_copy_direct(from, to, residual_elements * sizeof(float), peer, detail);
            impl_->residual_direct.push_back(direct);
            if (direct) {
                std::fprintf(stderr, "strata exl3-flash: stage %zu -> %zu residual: a direct device-to-device copy "
                                     "(peer access reports %s)\n", boundary, boundary + 1, peer ? "yes" : "no");
            } else {
                std::fprintf(stderr, "strata exl3-flash: stage %zu -> %zu residual: the host bridge (the devices do "
                                     "not copy directly: %s; peer access reports %s)\n", boundary, boundary + 1,
                             detail.c_str(), peer ? "yes" : "no");
            }
        }
        if (!strata::gpu::event_create(&impl_->cross_event, false) ||
            !strata::gpu::event_create(&impl_->doorbell_probe, false)) {
            err = "EXL3 Flash: crossing event allocation failed";
            return false;
        }
        {
            auto& ple_owner_stage = impl_->stages[static_cast<size_t>(ple_owner_stage_index)];
            const OnDevice on(ple_owner_stage.device);
            ple_owner_stage.session.ple.table = impl_->ple_table.get();
            ple_owner_stage.session.ple.w = impl_->ple->weights();
            ple_owner_stage.session.ple.key_exl3 = impl_->ple->key();
            ple_owner_stage.session.ple.value_exl3 = impl_->ple->value();
            ple_owner_stage.session.ple.pre_key_out = impl_->ple->pre_key();
            ple_owner_stage.session.ple.pre_value_out = impl_->ple->pre_value();
            // The token window is the model's; the history is the home stage's, carved from its
            // session arena like every other per-stage state.
            ple_owner_stage.session.ple.token = &impl_->ple_token;
            ple_owner_stage.session.ple.prev = impl_->ple_prev;
            ple_owner_stage.session.ple.hist = ple_owner_stage.session.ple_hist;
            ple_owner_stage.session.ple.consts = strata::kernels::ple_artifact_consts();
            ple_owner_stage.session.ple.emb_dev = static_cast<float*>(strata::gpu::alloc_device(
                static_cast<size_t>(strata::kernels::NG_N_EMBD) * sizeof(float)));
            ple_owner_stage.session.ple.emb_host = static_cast<float*>(strata::gpu::alloc_host(
                static_cast<size_t>(strata::kernels::NG_N_EMBD) * sizeof(float)));
            ple_owner_stage.session.ple.scratch = static_cast<float*>(strata::gpu::alloc_device(
                static_cast<size_t>(strata::core::ple_run_scratch_bytes())));
            if (ple_owner_stage.session.ple.emb_dev == nullptr || ple_owner_stage.session.ple.emb_host == nullptr ||
                ple_owner_stage.session.ple.scratch == nullptr) {
                err = "EXL3 Flash: PLE token workspace allocation failed";
                return false;
            }
        }

        {
            const auto& last = impl_->stages.back();
            const OnDevice on(last.device);
            impl_->logits_device = strata::gpu::alloc_device(impl_->logits_host.size() * sizeof(float));
            if (impl_->logits_device == nullptr) {
                err = "EXL3 Flash: output logits allocation failed";
                return false;
            }
        }

        // The initial coordinator captures the unfused, non-early form.  The
        // selected values are restored after capture; graph nodes retain the
        // correct branch and later callers do not inherit this policy.
        impl_->previous_fused_gr = layer_fused_gr();
        impl_->previous_shared_early = layer_shared_early();
        impl_->policy_saved = true;
        layer_set_fused_gr(false);
        layer_set_shared_early(false);
        for (size_t stage_index = 0; stage_index < impl_->stages.size(); ++stage_index) {
            auto& stage = impl_->stages[stage_index];
            const OnDevice on(stage.device);
            stage.pre_graphs.resize(static_cast<size_t>(stage.layer_end - stage.layer_begin), nullptr);
            stage.post_graphs.resize(static_cast<size_t>(stage.layer_end - stage.layer_begin), nullptr);
            if (!stage.weights->prepare_stream(nullptr, err)) {
                err = "EXL3 Flash: preparing stage graph stream: " + err;
                return false;
            }
            stage_token(impl_->geometry, 0, 0, stage.session);
            if (stage.session.ple.table != nullptr) {
                // The capture's PLE token: the window the model starts from, and the rows the
                // captured graph reads.  The window is the model's; the rows are the home stage's.
                impl_->ple_token = 0;
                impl_->ple_prev[0] = -1;
                impl_->ple_prev[1] = -1;
            }
            strata::gpu::Stream capture_stream = strata::gpu::stream_create();
            if (capture_stream == nullptr) {
                err = "EXL3 Flash: graph capture stream creation failed";
                return false;
            }
            if (!stage.weights->prepare_stream(capture_stream, err) ||
                (stage.session.ple.table != nullptr && !impl_->ple->prepare_stream(capture_stream, err))) {
                strata::gpu::stream_destroy(capture_stream);
                err = "EXL3 Flash: preparing graph owners: " + err;
                return false;
            }
            if (stage.session.ple.table != nullptr) {
                if (!ple_stage_token(stage.session.ple, capture_stream, err) ||
                    !strata::gpu::stream_sync(capture_stream)) {
                    strata::gpu::stream_destroy(capture_stream);
                    err = "EXL3 Flash: preparing PLE capture token: " + err;
                    return false;
                }
            }
            for (int64_t layer = stage.layer_begin; layer < stage.layer_end; ++layer) {
                point_gdn(impl_->geometry, layer, stage.session);
                QsaState& qsa = qsa_for(impl_->geometry, layer, stage.session);
                const size_t local = static_cast<size_t>(layer - stage.layer_begin);
                if (!strata::gpu::begin_capture(capture_stream)) {
                    err = "EXL3 Flash: begin pre graph capture failed at layer " + std::to_string(layer);
                    break;
                }
                if (!block_layer_pre(stage.table, impl_->geometry, layer, 0, 0, stage.session.gdn, qsa,
                                     stage.session.qsa_bufs, stage.session.moe, routed_k, stage.session.block,
                                     capture_stream, err, &stage.doorbell,
                                     stage.session.ple.table != nullptr ? &stage.session.ple : nullptr)) {
                    strata::gpu::abandon_capture(capture_stream);
                    err = "EXL3 Flash: pre graph layer " + std::to_string(layer) + ": " + err;
                    break;
                }
                if (!strata::gpu::end_capture(capture_stream, &stage.pre_graphs[local])) {
                    err = "EXL3 Flash: end pre graph layer " + std::to_string(layer) + ": " + strata::gpu::last_error();
                    break;
                }
                if (!strata::gpu::begin_capture(capture_stream)) {
                    err = "EXL3 Flash: begin post graph capture failed at layer " + std::to_string(layer);
                    break;
                }
                if (!block_layer_post(stage.table, impl_->geometry, layer, routed_k, stage.session.moe,
                                      stage.session.block, static_cast<const float*>(stage.parts_device),
                                      capture_stream, err)) {
                    strata::gpu::abandon_capture(capture_stream);
                    err = "EXL3 Flash: post graph layer " + std::to_string(layer) + ": " + err;
                    break;
                }
                if (!strata::gpu::end_capture(capture_stream, &stage.post_graphs[local])) {
                    err = "EXL3 Flash: end post graph layer " + std::to_string(layer) + ": " + strata::gpu::last_error();
                    break;
                }
            }
            strata::gpu::stream_destroy(capture_stream);
            if (!err.empty()) return false;
        }

        // ---- THE DEVICE CACHE, PLANNED FROM WHAT THE DEVICE REPORTS ----
        //
        // The MTP draft block's weights, expert source and session state are bound here, after the
        // trunk's graphs and before the device-cache opening: they are device allocations the cache
        // must see as taken.  The draft block gets no expert runtime (no device cache) until the
        // draft/verify loop lands; its experts serve through the source and the CPU pool.
        if (impl_->options.include_mtp) {
            if (!make_expert_names(impl_->mtp_map, 1, impl_->mtp_map.geometry.experts,
                                   impl_->mtp_expert_names, err)) {
                err = "EXL3 Flash MTP experts: " + err;
                return false;
            }
            impl_->mtp_device = impl_->stages.back().device;
            impl_->mtp_geometry = impl_->geometry;
            impl_->mtp_geometry.n_layers = 1;
            impl_->mtp_geometry.qsa_interval = 1;
            impl_->mtp_table = std::make_unique<WeightTable>();
            impl_->mtp_weights = std::make_unique<Exl3WeightSet>();
            impl_->mtp_source = std::make_unique<Exl3ExpertSource>();
            impl_->mtp_pool = std::make_unique<strata::kernels::cpu::ExpertPool>(options.pool_workers);
            const OnDevice on_mtp(impl_->mtp_device);
            Exl3WeightSetOptions mtp_weight_options;
            mtp_weight_options.device = impl_->mtp_device;
            mtp_weight_options.layer_begin = 0;
            mtp_weight_options.layer_end = 1;
            mtp_weight_options.include_mtp_block = true;
            mtp_weight_options.max_rows = options.max_rows;
            if (!impl_->mtp_weights->load(*impl_->archive, impl_->mtp_map, *impl_->mtp_table,
                                          mtp_weight_options, err)) {
                err = "EXL3 Flash MTP dense weights: " + err;
                return false;
            }
            if (!impl_->mtp_source->open(impl_->archive, 1, impl_->mtp_map.geometry.experts,
                                         impl_->mtp_expert_names, err)) {
                err = "EXL3 Flash MTP expert source: " + err;
                return false;
            }
            const uint64_t mtp_session_size =
                session_bytes(impl_->mtp_geometry, impl_->options.max_cells, routed_k, 0, 1);
            impl_->mtp_session_arena_bytes = static_cast<size_t>(mtp_session_size);
            impl_->mtp_session_arena = strata::gpu::alloc_device(static_cast<size_t>(mtp_session_size));
            if (impl_->mtp_session_arena == nullptr ||
                session_init(impl_->mtp_geometry, impl_->options.max_cells, routed_k,
                             impl_->mtp_session_arena, impl_->mtp_session, 0, 1) == 0) {
                err = "EXL3 Flash MTP SessionState allocation failed";
                return false;
            }
            session_zero(impl_->mtp_session, impl_->mtp_geometry, nullptr, nullptr);
            impl_->mtp_stream = strata::gpu::stream_create();
            if (impl_->mtp_stream == nullptr) {
                err = "EXL3 Flash MTP draft stream creation failed";
                return false;
            }
            if (!impl_->mtp_weights->prepare_stream(impl_->mtp_stream, err)) {
                err = "EXL3 Flash MTP draft stream weights: " + err;
                return false;
            }
            if (!impl_->stages.back().weights->prepare_stream(impl_->mtp_stream, err)) {
                err = "EXL3 Flash MTP draft head weights: " + err;
                return false;
            }
            impl_->mtp_runtime = std::make_unique<Exl3ExpertRuntime>();
            // The draft block's experts get a device cache of their own, planned from what the
            // device reports like a stage's (open_device_cache): one MTP layer is small, so on a
            // card with room every draft expert is resident and the draft runs on the GPU.
            // STRATA_EXL3_MTP_DEVICE_DRAFT=0 keeps the CPU-pool draft for comparison.
            {
                const char* toggle = std::getenv("STRATA_EXL3_MTP_DEVICE_DRAFT");
                const bool want_device = !(toggle != nullptr && toggle[0] == '0' && toggle[1] == '\0');
                if (want_device) {
                    if (!open_device_cache(*impl_->mtp_source, *impl_->mtp_pool, *impl_->mtp_runtime,
                                           impl_->stages.size(), 0, 1, impl_->geometry.n_expert,
                                           impl_->geometry.n_layers, options.max_rows, routed_k,
                                           options.allow_cpu_expert_miss, err)) {
                        err = "EXL3 Flash MTP expert runtime: " + err;
                        return false;
                    }
                    impl_->mtp_device_draft = impl_->mtp_runtime->fully_resident();
                } else if (!impl_->mtp_runtime->open(*impl_->mtp_source, *impl_->mtp_pool, 1, routed_k,
                                                     {}, err, 0, 1)) {
                    err = "EXL3 Flash MTP expert runtime: " + err;
                    return false;
                }
            }
            const size_t mtp_fc_floats = static_cast<size_t>(3 * impl_->geometry.n_embd +
                                                            2 * impl_->geometry.hc * impl_->geometry.n_embd);
            impl_->mtp_fc_scratch = strata::gpu::alloc_device(mtp_fc_floats * sizeof(float));
            impl_->mtp_parts_device = strata::gpu::alloc_device(static_cast<size_t>(routed_k) *
                                                                static_cast<size_t>(impl_->geometry.n_embd) *
                                                                sizeof(float));
            if (impl_->mtp_fc_scratch == nullptr || impl_->mtp_parts_device == nullptr) {
                err = "EXL3 Flash MTP draft scratch allocation failed";
                return false;
            }
            if (impl_->mtp_device_draft) {
                // Capture the resident expert flow (the clear of the expert rows, the device
                // route plan and the grouped experts) like a stage's resident flow: the pointers
                // are fixed, so one graph replays it for every draft on the draft stream.
                strata::gpu::Stream capture_stream = strata::gpu::stream_create();
                if (capture_stream == nullptr) {
                    err = "EXL3 Flash: MTP expert graph capture stream creation failed";
                    return false;
                }
                sycl::queue& queue = Runtime::at(impl_->mtp_device).stream(capture_stream);
                const size_t parts_floats = static_cast<size_t>(routed_k) *
                                            static_cast<size_t>(impl_->geometry.n_embd);
                if (!strata::gpu::begin_capture(capture_stream) ||
                    !strata::gpu::memset_async(impl_->mtp_parts_device, 0, parts_floats * sizeof(float),
                                               capture_stream) ||
                    !impl_->mtp_runtime->record_device_hit_enqueue(
                        queue, 0, static_cast<const float*>(impl_->mtp_session.block.mixed),
                        static_cast<const int32_t*>(impl_->mtp_session.moe.ids), 1,
                        static_cast<float*>(impl_->mtp_parts_device), impl_->geometry.n_embd, err)) {
                    strata::gpu::abandon_capture(capture_stream);
                    strata::gpu::stream_destroy(capture_stream);
                    err = "EXL3 Flash: MTP expert graph capture: " + err;
                    return false;
                }
                if (!strata::gpu::end_capture(capture_stream, &impl_->mtp_expert_graph)) {
                    strata::gpu::stream_destroy(capture_stream);
                    err = std::string("EXL3 Flash: end MTP expert graph capture: ") +
                          strata::gpu::last_error();
                    return false;
                }
                strata::gpu::stream_destroy(capture_stream);
                std::fprintf(stderr, "strata exl3-flash: MTP draft experts resident on the device "
                                     "(captured expert graph; no CPU pool in the draft loop)\n");
                // Capture the whole draft body after the embedding gather (the one step whose
                // token argument changes per draft) as a single graph: the two norms, the fc
                // projections, the block pre, the resident expert flow, the block post and the
                // head.  Every pointer is a buffer bound at open and never moved, and the graph
                // re-reads their contents at replay, so one capture serves every draft.  A
                // capture failure keeps the uncaptured device path below, not a failed open.
                {
                    const int64_t N = impl_->geometry.n_embd;
                    const int64_t HC = impl_->geometry.hc;
                    float* scratch = static_cast<float*>(impl_->mtp_fc_scratch);
                    float* emb = scratch;
                    float* en = emb + N;
                    float* hn = en + N;
                    float* e2 = hn + HC * N;
                    float* h2 = e2 + N;
                    const WeightRef* enorm = impl_->mtp_table->find("mtp.pre_fc_norm_embedding.weight");
                    const WeightRef* hnorm = impl_->mtp_table->find("mtp.pre_fc_norm_hidden.weight");
                    const WeightRef* fc_e = impl_->mtp_table->find("mtp.fc_embedding.weight");
                    const WeightRef* fc_h = impl_->mtp_table->find("mtp.fc_hidden.weight");
                    auto& last_stage = impl_->stages.back();
                    strata::gpu::Stream body_stream = strata::gpu::stream_create();
                    // Owners with stream-local state are prepared on the capture stream first,
                    // as the trunk's graph capture does: a first use inside the capture would
                    // allocate, which a capture forbids.
                    bool body_ok = body_stream != nullptr &&
                                   impl_->mtp_weights->prepare_stream(body_stream, err) &&
                                   last_stage.weights->prepare_stream(body_stream, err);
                    body_ok = body_ok && enorm != nullptr && hnorm != nullptr &&
                                   fc_e != nullptr && fc_h != nullptr && fc_e->exl3 != nullptr &&
                                   fc_h->exl3 != nullptr && strata::gpu::begin_capture(body_stream);
                    if (body_ok) {
                        const float eps = static_cast<float>(impl_->spec.rms_eps);
                        sycl::queue& queue = Runtime::at(impl_->mtp_device).stream(body_stream);
                        strata::kernels::native_qsa_rms_norm_weighted(
                            emb, static_cast<const float*>(enorm->data), en, static_cast<int>(N), 1, eps,
                            body_stream);
                        strata::kernels::native_qsa_rms_norm_weighted(
                            last_stage.session.block.R, static_cast<const float*>(hnorm->data), hn,
                            static_cast<int>(HC * N), 1, eps, body_stream);
                        body_ok = fc_e->exl3->run(en, N, e2, N, 1, body_stream, err);
                        for (int64_t s = 0; body_ok && s < HC; ++s)
                            body_ok = fc_h->exl3->run(hn + s * N, N, h2 + s * N, N, 1, body_stream, err);
                        if (body_ok) {
                            strata::kernels::add_streams_broadcast(h2, e2, impl_->mtp_session.block.R, N,
                                                                   static_cast<int>(HC), 1, body_stream);
                            QsaState& mq = impl_->mtp_session.qsa_states[impl_->mtp_session.qsa_ord0];
                            body_ok = block_layer_pre(*impl_->mtp_table, impl_->mtp_geometry, 0, 0, 0,
                                                      impl_->mtp_session.gdn, mq, impl_->mtp_session.qsa_bufs,
                                                      impl_->mtp_session.moe, routed_k,
                                                      impl_->mtp_session.block, body_stream, err);
                        }
                        if (body_ok)
                            body_ok = strata::gpu::memset_async(impl_->mtp_parts_device, 0,
                                                                parts_floats * sizeof(float), body_stream) &&
                                      impl_->mtp_runtime->record_device_hit_enqueue(
                                          queue, 0, static_cast<const float*>(impl_->mtp_session.block.mixed),
                                          static_cast<const int32_t*>(impl_->mtp_session.moe.ids), 1,
                                          static_cast<float*>(impl_->mtp_parts_device), N, err);
                        if (body_ok)
                            body_ok = block_layer_post(*impl_->mtp_table, impl_->mtp_geometry, 0, routed_k,
                                                       impl_->mtp_session.moe, impl_->mtp_session.block,
                                                       static_cast<const float*>(impl_->mtp_parts_device),
                                                       body_stream, err);
                        if (body_ok) {
                            const BlockBuffers head_bb = impl_->mtp_session.block;
                            body_ok = lm_head(last_stage.table, impl_->geometry, head_bb,
                                              static_cast<float*>(impl_->logits_device), body_stream, err);
                        }
                    }
                    if (body_ok && strata::gpu::end_capture(body_stream, &impl_->mtp_body_graph)) {
                        std::fprintf(stderr, "strata exl3-flash: MTP draft body captured as one graph\n");
                    } else {
                        if (body_stream != nullptr) strata::gpu::abandon_capture(body_stream);
                        impl_->mtp_body_graph = nullptr;
                        std::fprintf(stderr, "strata exl3-flash: MTP draft body capture unavailable (%s); "
                                             "the draft runs uncaptured on the device path\n",
                                     err.empty() ? strata::gpu::last_error() : err.c_str());
                        err.clear();
                    }
                    if (body_stream != nullptr) strata::gpu::stream_destroy(body_stream);
                }
            }
            impl_->mtp_host_ids.assign(static_cast<size_t>(routed_k), 0);
            impl_->mtp_host_weights.assign(static_cast<size_t>(routed_k), 0.0f);
            impl_->mtp_host_x.assign(static_cast<size_t>(impl_->geometry.n_embd), 0.0f);
            impl_->mtp_cpu_out.assign(static_cast<size_t>(routed_k) * static_cast<size_t>(impl_->geometry.n_embd),
                                      0.0f);
        }

        // It is opened LAST, after every other device allocation and after the graph capture, because it takes
        // what the device has left: the room it gets is the free VRAM at this point, less a reserve.
        for (size_t stage_index = 0; stage_index < impl_->stages.size(); ++stage_index) {
            auto& stage = impl_->stages[stage_index];
            const OnDevice on(stage.device);
            if (!open_device_cache(*stage.source, *stage.pool, *stage.runtime, stage_index,
                                   stage.layer_begin, stage.layer_end, impl_->geometry.n_expert,
                                   impl_->geometry.n_layers, options.max_rows, routed_k,
                                   options.allow_cpu_expert_miss, err)) {
                return false;
            }
        }

        // ---- THE RESIDENT EXPERT FLOW, CAPTURED LIKE THE LAYERS ARE ----
        //
        // A fully resident cache runs the same submissions for every layer of every token: the clear of the
        // expert rows, the device route plan and the grouped experts' kernels.  They sat between the pre and
        // post graphs at their own dispatch cost (a launch outside a graph replay measured 11 to 27 us of
        // boundary against 7.8 us inside one) while the flow never changes shape between tokens, so it is
        // captured per layer and replayed like the pre and post graphs.  Every pointer the body takes is a
        // buffer allocated before this point and never moved (graph.hpp: a graph re-reads its input buffers at
        // replay, but not its kernel arguments), and the plan kernel's layer is a property of the graph it is
        // captured into.  The cache is open by now: the plan's device tables live in it, and whether it holds
        // every expert is what decides whether this flow exists at all.
        for (size_t stage_index = 0; stage_index < impl_->stages.size(); ++stage_index) {
            auto& stage = impl_->stages[stage_index];
            const OnDevice on(stage.device);
            if (!stage.runtime->fully_resident()) continue;
            stage.expert_graphs.resize(static_cast<size_t>(stage.layer_end - stage.layer_begin), nullptr);
            for (int64_t layer = stage.layer_begin; layer < stage.layer_end; ++layer) {
                const size_t local = static_cast<size_t>(layer - stage.layer_begin);
                strata::gpu::Stream capture_stream = strata::gpu::stream_create();
                if (capture_stream == nullptr) {
                    err = "EXL3 Flash: expert graph capture stream creation failed";
                    return false;
                }
                sycl::queue& queue = Runtime::at(stage.device).stream(capture_stream);
                if (!strata::gpu::begin_capture(capture_stream)) {
                    strata::gpu::stream_destroy(capture_stream);
                    err = "EXL3 Flash: begin expert graph capture failed at layer " + std::to_string(layer);
                    return false;
                }
                if (!strata::gpu::memset_async(stage.parts_device, 0,
                                               stage.merge_output.size() * sizeof(float), capture_stream) ||
                    !stage.runtime->record_device_hit_enqueue(queue, layer,
                                                              static_cast<const float*>(stage.session.block.mixed),
                                                              stage.doorbell.d_ids, 1,
                                                              static_cast<float*>(stage.parts_device),
                                                              impl_->geometry.n_embd, err)) {
                    strata::gpu::abandon_capture(capture_stream);
                    strata::gpu::stream_destroy(capture_stream);
                    err = "EXL3 Flash: expert graph layer " + std::to_string(layer) + ": " + err;
                    return false;
                }
                if (!strata::gpu::end_capture(capture_stream, &stage.expert_graphs[local])) {
                    strata::gpu::stream_destroy(capture_stream);
                    err = "EXL3 Flash: end expert graph layer " + std::to_string(layer) + ": " +
                          strata::gpu::last_error();
                    return false;
                }
                strata::gpu::stream_destroy(capture_stream);
            }
        }

        // ---- ONE GRAPH FOR THE WHOLE TOKEN, WHERE THE CACHE IS FULLY RESIDENT ----
        //
        // The per-layer path above launches three graphs per layer (pre, expert flow, post); the
        // launches themselves cost a graph replay's boundary each.  Where every expert of a stage
        // is resident the token never needs the host between layers, so the same submissions are
        // captured again as one graph per stage and forward() replays that instead.  The capture
        // replays the per-layer bodies in order on one stream: every pointer is a session buffer
        // bound before this point, the position staging is re-read from its pinned buffers at
        // replay (stage_token runs per token, before the launch), and the PLE rows are uploaded
        // before the launch and read where the PLE layer sits inside the graph.
        if (token_graph_enabled()) {
            for (size_t stage_index = 0; stage_index < impl_->stages.size(); ++stage_index) {
                auto& stage = impl_->stages[stage_index];
                const OnDevice on(stage.device);
                if (!stage.runtime->fully_resident()) continue;
                strata::gpu::Stream capture_stream = strata::gpu::stream_create();
                if (capture_stream == nullptr) {
                    err = "EXL3 Flash: token graph capture stream creation failed";
                    return false;
                }
                if (!stage.weights->prepare_stream(capture_stream, err) ||
                    (stage.session.ple.table != nullptr && !impl_->ple->prepare_stream(capture_stream, err))) {
                    strata::gpu::stream_destroy(capture_stream);
                    err = "EXL3 Flash: preparing token graph owners: " + err;
                    return false;
                }
                stage_token(impl_->geometry, 0, 0, stage.session);
                sycl::queue& queue = Runtime::at(stage.device).stream(capture_stream);
                if (!strata::gpu::begin_capture(capture_stream)) {
                    strata::gpu::stream_destroy(capture_stream);
                    err = "EXL3 Flash: begin token graph capture failed";
                    return false;
                }
                bool captured = true;
                for (int64_t layer = stage.layer_begin; captured && layer < stage.layer_end; ++layer) {
                    point_gdn(impl_->geometry, layer, stage.session);
                    QsaState& qsa = qsa_for(impl_->geometry, layer, stage.session);
                    captured = block_layer_pre(stage.table, impl_->geometry, layer, 0, 0, stage.session.gdn, qsa,
                                               stage.session.qsa_bufs, stage.session.moe, routed_k,
                                               stage.session.block, capture_stream, err, &stage.doorbell,
                                               stage.session.ple.table != nullptr ? &stage.session.ple : nullptr) &&
                               strata::gpu::memset_async(stage.parts_device, 0,
                                                         stage.merge_output.size() * sizeof(float),
                                                         capture_stream) &&
                               stage.runtime->record_device_hit_enqueue(
                                   queue, layer, static_cast<const float*>(stage.session.block.mixed),
                                   stage.doorbell.d_ids, 1, static_cast<float*>(stage.parts_device),
                                   impl_->geometry.n_embd, err) &&
                               block_layer_post(stage.table, impl_->geometry, layer, routed_k, stage.session.moe,
                                                stage.session.block,
                                                static_cast<const float*>(stage.parts_device), capture_stream, err);
                    if (!captured) err = "EXL3 Flash: token graph layer " + std::to_string(layer) + ": " + err;
                }
                if (!captured || !strata::gpu::end_capture(capture_stream, &stage.token_graph)) {
                    if (captured) err = std::string("EXL3 Flash: end token graph capture: ") + strata::gpu::last_error();
                    else strata::gpu::abandon_capture(capture_stream);
                    strata::gpu::stream_destroy(capture_stream);
                    return false;
                }
                strata::gpu::stream_destroy(capture_stream);
            }
        }

        layer_set_fused_gr(impl_->previous_fused_gr);
        layer_set_shared_early(impl_->previous_shared_early);
        impl_->policy_saved = false;
        impl_->opened = true;
        impl_->initialized = false;
        cleanup.owner = nullptr;
        return true;
    } catch (const std::exception& e) {
        err = std::string("EXL3 Flash: ") + e.what();
        return false;
    }
}

bool Exl3FlashModel::close(std::string& err) noexcept {
    err.clear();
    if (!impl_) return true;
    bool ok = true;
    try {
        for (auto it = impl_->stages.rbegin(); it != impl_->stages.rend(); ++it) {
            auto& stage = *it;
            const OnDevice on(stage.device);
            if (stage.runtime) Runtime::at(stage.device).finish();
            destroy_graphs(stage.pre_graphs);
            destroy_graphs(stage.post_graphs);
            destroy_graphs(stage.expert_graphs);
            if (stage.token_graph != nullptr) {
                strata::gpu::graph_destroy(stage.token_graph);
                stage.token_graph = nullptr;
            }
            session_release(stage.session);
            if (stage.session.qsa_states != nullptr) {
                for (int64_t ordinal = 0; ordinal < impl_->geometry.n_qsa_layers(); ++ordinal) {
                    if (stage.session.qsa_states[ordinal].host_step != nullptr)
                        strata::gpu::free(stage.session.qsa_states[ordinal].host_step);
                    if (stage.session.qsa_states[ordinal].host_pos != nullptr)
                        strata::gpu::free(stage.session.qsa_states[ordinal].host_pos);
                }
                delete[] stage.session.qsa_states;
                stage.session.qsa_states = nullptr;
            }
            if (stage.session.ple.emb_dev != nullptr) strata::gpu::free(stage.session.ple.emb_dev);
            if (stage.session.ple.emb_host != nullptr) strata::gpu::free(stage.session.ple.emb_host);
            if (stage.session.ple.scratch != nullptr) strata::gpu::free(stage.session.ple.scratch);
            stage.session.ple.emb_dev = nullptr;
            stage.session.ple.emb_host = nullptr;
            stage.session.ple.scratch = nullptr;
            if (stage.embedding_device != nullptr) strata::gpu::free(stage.embedding_device);
            if (stage.parts_device != nullptr) strata::gpu::free(stage.parts_device);
            if (stage.session_arena != nullptr) strata::gpu::free(stage.session_arena);
            stage.embedding_device = nullptr;
            stage.parts_device = nullptr;
            stage.session_arena = nullptr;
            doorbell_free(stage.doorbell);
            if (stage.runtime) stage.runtime->close();
            stage.runtime.reset();
            stage.source.reset();
            stage.pool.reset();
            if (stage.weights) stage.weights->close();
            stage.weights.reset();
        }
        impl_->stages.clear();
        if (impl_->mtp_device >= 0) {
            const OnDevice on_mtp(impl_->mtp_device);
            session_release(impl_->mtp_session);
            if (impl_->mtp_session.qsa_states != nullptr) {
                for (int64_t ordinal = 0; ordinal < impl_->mtp_geometry.n_qsa_layers(); ++ordinal) {
                    if (impl_->mtp_session.qsa_states[ordinal].host_step != nullptr)
                        strata::gpu::free(impl_->mtp_session.qsa_states[ordinal].host_step);
                    if (impl_->mtp_session.qsa_states[ordinal].host_pos != nullptr)
                        strata::gpu::free(impl_->mtp_session.qsa_states[ordinal].host_pos);
                }
                delete[] impl_->mtp_session.qsa_states;
                impl_->mtp_session.qsa_states = nullptr;
            }
            if (impl_->mtp_session.ple.emb_dev != nullptr) strata::gpu::free(impl_->mtp_session.ple.emb_dev);
            if (impl_->mtp_session.ple.emb_host != nullptr) strata::gpu::free(impl_->mtp_session.ple.emb_host);
            if (impl_->mtp_session.ple.scratch != nullptr) strata::gpu::free(impl_->mtp_session.ple.scratch);
            impl_->mtp_session.ple.emb_dev = nullptr;
            impl_->mtp_session.ple.emb_host = nullptr;
            impl_->mtp_session.ple.scratch = nullptr;
            if (impl_->mtp_session_arena != nullptr) strata::gpu::free(impl_->mtp_session_arena);
            impl_->mtp_session_arena = nullptr;
            if (impl_->mtp_weights) impl_->mtp_weights->close();
            if (impl_->mtp_body_graph != nullptr) {
                strata::gpu::graph_destroy(impl_->mtp_body_graph);
                impl_->mtp_body_graph = nullptr;
            }
            if (impl_->mtp_expert_graph != nullptr) {
                strata::gpu::graph_destroy(impl_->mtp_expert_graph);
                impl_->mtp_expert_graph = nullptr;
            }
            impl_->mtp_device_draft = false;
            if (impl_->mtp_runtime) {
                Runtime::at(impl_->mtp_device).finish();
                impl_->mtp_runtime->close();
            }
            if (impl_->mtp_stream != nullptr) strata::gpu::stream_destroy(impl_->mtp_stream);
            impl_->mtp_stream = nullptr;
            impl_->mtp_runtime.reset();
            impl_->mtp_weights.reset();
            impl_->mtp_source.reset();
            impl_->mtp_pool.reset();
            if (impl_->mtp_fc_scratch != nullptr) strata::gpu::free(impl_->mtp_fc_scratch);
            if (impl_->mtp_parts_device != nullptr) strata::gpu::free(impl_->mtp_parts_device);
            impl_->mtp_fc_scratch = nullptr;
            impl_->mtp_parts_device = nullptr;
            impl_->mtp_host_ids.clear();
            impl_->mtp_host_weights.clear();
            impl_->mtp_host_x.clear();
            impl_->mtp_cpu_out.clear();
            impl_->mtp_table.reset();
            impl_->mtp_device = -1;
        }
        if (impl_->logits_device != nullptr) {
            const OnDevice on(impl_->options.stage_devices.empty() ? -1 : impl_->options.stage_devices.back());
            strata::gpu::free(impl_->logits_device);
            impl_->logits_device = nullptr;
        }
        if (impl_->ple) {
            const OnDevice on(impl_->ple_device);
            impl_->ple->close();
            impl_->ple.reset();
        }
        impl_->ple_table.reset();
        if (impl_->embedding) {
            const OnDevice on(impl_->embedding_device);
            impl_->embedding.reset();
        }
        if (impl_->cross_event != nullptr) {
            strata::gpu::event_destroy(impl_->cross_event);
            impl_->cross_event = nullptr;
        }
        if (impl_->doorbell_probe != nullptr) {
            strata::gpu::event_destroy(impl_->doorbell_probe);
            impl_->doorbell_probe = nullptr;
        }
        impl_->residual_direct.clear();
        impl_->archive.reset();
        impl_->expert_names.clear();
        impl_->map = {};
        impl_->mtp_map = {};
        impl_->has_mtp = false;
        impl_->mtp_expert_names.clear();
        impl_->mtp_geometry = {};
        impl_->spec = {};
        impl_->geometry = {};
        impl_->embedding_device = impl_->ple_device = -1;
        impl_->ple_stage_index = -1;
        if (impl_->policy_saved) {
            layer_set_fused_gr(impl_->previous_fused_gr);
            layer_set_shared_early(impl_->previous_shared_early);
            impl_->policy_saved = false;
        }
        if (impl_->model_globals_saved) {
            strata::kernels::rope_scaling_set(impl_->previous_rope);
            strata::kernels::gdn_set_silu_gate(impl_->previous_gdn_silu);
            impl_->model_globals_saved = false;
        }
        impl_->opened = false;
        impl_->initialized = false;
        impl_->seeded = false;
        impl_->next_position = 0;
    } catch (const std::exception& e) {
        ok = false;
        err = std::string("EXL3 Flash: close: ") + e.what();
        if (impl_->policy_saved) {
            try {
                layer_set_fused_gr(impl_->previous_fused_gr);
                layer_set_shared_early(impl_->previous_shared_early);
                impl_->policy_saved = false;
            } catch (...) {
            }
        }
        if (impl_->model_globals_saved) {
            try {
                strata::kernels::rope_scaling_set(impl_->previous_rope);
                strata::kernels::gdn_set_silu_gate(impl_->previous_gdn_silu);
                impl_->model_globals_saved = false;
            } catch (...) {
            }
        }
    }
    return ok;
}

void Exl3FlashModel::close() noexcept {
    std::string ignored;
    (void) close(ignored);
}

bool Exl3FlashModel::prepare_stream(void* stream, std::string& err) {
    err.clear();
    if (!loaded()) { err = "EXL3 Flash: open is required before prepare_stream"; return false; }
    try {
        Runtime* owner = stream == nullptr ? &Runtime::at(current_device()) : Runtime::owner_of_stream(stream);
        if (owner == nullptr) { err = "EXL3 Flash: stream has no Xe runtime owner"; return false; }
        const int device = owner->ordinal();
        bool found = false;
        for (auto& stage : impl_->stages) {
            if (stage.device != device) continue;
            found = true;
            const OnDevice on(device);
            if (!stage.weights->prepare_stream(stream, err)) return false;
            if (impl_->ple && impl_->ple_device == device && !impl_->ple->prepare_stream(stream, err)) return false;
        }
        if (impl_->embedding && !impl_->embedding->prepare_stream(stream, err)) return false;
        if (!found) {
            err = "EXL3 Flash: stream belongs to a device with no model stage";
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("EXL3 Flash: preparing stream: ") + e.what();
        return false;
    }
}

bool Exl3FlashModel::prepare_expert_window(size_t stage_index, int64_t layer, const float* host_x, int64_t ldx,
                                           const int32_t* routed_ids, int64_t n_tokens, float* cpu_output, int64_t ldy,
                                           void* stream, const float* device_x, float* device_output, sycl::event& done,
                                           std::string& err) {
    err.clear();
    if (!loaded() || stage_index >= impl_->stages.size()) {
        err = "EXL3 Flash: invalid stage or unopened model";
        return false;
    }
    if (!validate_token_request(n_tokens, 1, err)) return false;
    auto& stage = impl_->stages[stage_index];
    if (layer < stage.layer_begin || layer >= stage.layer_end) {
        err = "EXL3 Flash: expert layer does not belong to the selected stage";
        return false;
    }
    try {
        Runtime& runtime = Runtime::at(stage.device);
        if (stream != nullptr && Runtime::owner_of_stream(stream) != &runtime) {
            err = "EXL3 Flash: expert stream belongs to another device stage";
            return false;
        }
        const OnDevice on(stage.device);
        sycl::queue& queue = runtime.stream(stream);
        if (!stage.runtime->prepare_window(layer, host_x, ldx, routed_ids, n_tokens, cpu_output, ldy, err) ||
            !stage.runtime->enqueue_hit_only(queue, device_x, ldx, device_output, ldy, done, {}, err)) {
            err = "EXL3 Flash: expert window: " + err;
            return false;
        }
        if (!stage.runtime->run_cpu(err)) {
            const std::string cpu_error = err;
            if (stage.runtime->pending_gpu()) {
                std::string drain_error;
                if (!stage.runtime->wait_gpu(done, drain_error) && !drain_error.empty())
                    err = cpu_error + "; GPU drain: " + drain_error;
                else
                    err = cpu_error;
            } else {
                err = cpu_error;
            }
            err = "EXL3 Flash: expert window: " + err;
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("EXL3 Flash: expert window: ") + e.what();
        return false;
    }
}

bool Exl3FlashModel::wait_expert_window(size_t stage_index, sycl::event& done, std::string& err) {
    err.clear();
    if (!loaded() || stage_index >= impl_->stages.size()) {
        err = "EXL3 Flash: invalid stage or unopened model";
        return false;
    }
    try {
        const OnDevice on(impl_->stages[stage_index].device);
        return impl_->stages[stage_index].runtime->wait_gpu(done, err);
    } catch (const std::exception& e) {
        err = std::string("EXL3 Flash: waiting for expert window: ") + e.what();
        return false;
    }
}

bool Exl3FlashModel::reset(std::string& err) {
    err.clear();
    if (!loaded()) { err = "EXL3 Flash: model is not open"; return false; }
    try {
        for (auto& stage : impl_->stages) {
            const OnDevice on(stage.device);
            session_zero(stage.session, impl_->geometry, nullptr, nullptr);
            if (!strata::gpu::stream_sync(nullptr)) {
                err = std::string("EXL3 Flash: resetting stage state: ") + strata::gpu::last_error();
                return false;
            }
            std::fill(stage.miss_output.begin(), stage.miss_output.end(), 0.0f);
            std::fill(stage.merge_output.begin(), stage.merge_output.end(), 0.0f);
            std::fill(stage.merge_upload.begin(), stage.merge_upload.end(), 0.0f);
        }
        if (impl_->has_mtp) {
            const OnDevice on_mtp(impl_->mtp_device);
            session_zero(impl_->mtp_session, impl_->mtp_geometry, nullptr, nullptr);
            if (!strata::gpu::stream_sync(nullptr)) {
                err = std::string("EXL3 Flash: resetting MTP state: ") + strata::gpu::last_error();
                return false;
            }
        }
        // The PLE's token window is the model's: a new sequence starts with no predecessor, which
        // is what the n-gram lookup reads as the end of the context.
        impl_->ple_prev[0] = -1;
        impl_->ple_prev[1] = -1;
        impl_->ple_token = -1;
        impl_->initialized = true;
        impl_->seeded = false;
        impl_->next_position = 0;
        return true;
    } catch (const std::exception& e) {
        err = std::string("EXL3 Flash: reset: ") + e.what();
        return false;
    }
}

bool Exl3FlashModel::forward(int32_t token, int64_t pos, int32_t& next_token, float* host_logits,
                             std::string& err) {
    err.clear();
    next_token = -1;
    if (!loaded()) { err = "EXL3 Flash: model is not open"; return false; }
    if (!impl_->initialized && !reset(err)) return false;
    if (pos != impl_->next_position) {
        err = "EXL3 Flash: position " + std::to_string(pos) + " does not follow reset/previous forward at " +
              std::to_string(impl_->next_position);
        return false;
    }
    if (pos < 0 || pos >= impl_->options.max_cells) {
        err = "EXL3 Flash: position is outside the configured max_cells";
        return false;
    }
    if (token < 0 || static_cast<int64_t>(token) >= impl_->map.geometry.vocab_size) {
        err = "EXL3 Flash: token is outside the embedding vocabulary";
        return false;
    }
    PleTicketGuard ple_guard;
    try {
        auto& first = impl_->stages.front();
        HostTiming timing;
        timing.enabled = host_timing_enabled();
        auto mark = std::chrono::steady_clock::now();
        if (impl_->ple_stage_index < 0 || static_cast<size_t>(impl_->ple_stage_index) >= impl_->stages.size()) {
            err = "EXL3 Flash: PLE stage ownership is missing";
            return false;
        }
        auto& ple_stage = impl_->stages[static_cast<size_t>(impl_->ple_stage_index)];
        impl_->ple_token = token;
        bool ple_collected = ple_stage.session.ple.table == nullptr;
        if (ple_stage.session.ple.table != nullptr) {
            if (!ple_stage.session.ple.ready() ||
                !ple_stage.session.ple.table->is_open() ||
                !ple_stage.session.ple.hist ||
                !ple_stage.session.ple.emb_dev ||
                !ple_stage.session.ple.scratch ||
                !ple_stage.session.ple.prev) {
                err = "EXL3 Flash: PLE state is not ready before token staging";
                return false;
            }
            // THE ROW READ STARTS HERE AND IS COLLECTED JUST BEFORE THE LAYER THAT
            // CONSUMES IT.  The sixteen rows come from a 32.6 GB sidecar that is read
            // with unbuffered I/O, so the read costs the disk's latency; collecting it
            // here would put that latency in front of every layer.  The reader's own
            // worker thread does the read, and the layers in front of the PLE are
            // the work it hides behind - the same split the native path uses.
            mark = std::chrono::steady_clock::now();
            if (!ple_issue_token(ple_stage.session.ple, err)) {
                err = "EXL3 Flash: PLE token staging: " + err;
                return false;
            }
            timing.ple_issue_ms = ms_since(mark);
            ple_guard.arm(ple_stage.session.ple.table, ple_stage.session.ple.emb_host);
        }
        {
            const OnDevice on(first.device);
            impl_->embedding->gather_one(token, first.embedding_device, nullptr);
            for (int64_t stream = 0; stream < impl_->geometry.hc; ++stream) {
                if (!strata::gpu::copy_async(first.session.block.R +
                                                 static_cast<size_t>(stream) * impl_->geometry.n_embd,
                                             first.embedding_device,
                                             static_cast<size_t>(impl_->geometry.n_embd) * sizeof(float), nullptr)) {
                    err = std::string("EXL3 Flash: embedding residual broadcast failed: ") + strata::gpu::last_error();
                    return false;
                }
            }
            // Nothing here needs the host: the broadcast is queued on the first stage's queue and
            // the queue's order puts it in front of the first pre graph.  The host waits only
            // where data crosses to it - the doorbell, the merge, the logits, the stage boundary.
            impl_->seeded = true;
        }
        timing.embedding_ms = ms_since(mark);

        for (auto& stage : impl_->stages) {
            stage_token(impl_->geometry, pos, 0, stage.session);
        }
        // The whole token runs as one graph per stage where every stage captured one: the host
        // stages the position and the PLE rows, launches each stage's graph, and meets the data
        // only at the stage crossings and the head.  Any stage without a token graph (its cache
        // is not fully resident) keeps the per-layer path for the whole token.
        bool use_token_graph = token_graph_enabled() && device_route_plan_enabled();
        for (const auto& stage : impl_->stages) use_token_graph = use_token_graph && stage.token_graph != nullptr;
        for (size_t stage_index = 0; stage_index < impl_->stages.size(); ++stage_index) {
            auto& stage = impl_->stages[stage_index];
            const OnDevice on(stage.device);
            if (stage_index > 0) {
                // THE RESIDUAL IS THE ONLY THING THAT CROSSES A STAGE BOUNDARY.  The previous
                // stage issued the crossing on its own queue right after its last post graph (see
                // below), so it has been running while the host got here; this is the one
                // synchronisation at the crossing.  Devices that copy between their memories
                // directly have the residual in place by now; the others take it through
                // `bridge_host` here.  The PLE, the embedding and every per-layer state stay on
                // the stage that owns them.
                mark = std::chrono::steady_clock::now();
                if (!strata::gpu::event_sync(impl_->cross_event)) {
                    err = std::string("EXL3 Flash: stage boundary residual crossing: ") + strata::gpu::last_error();
                    return false;
                }
                if (!impl_->residual_direct[stage_index - 1] &&
                    !strata::gpu::copy_async(stage.session.block.R, impl_->bridge_host.data(),
                                             impl_->bridge_host.size() * sizeof(float), nullptr)) {
                    err = std::string("EXL3 Flash: stage boundary residual upload failed: ") + strata::gpu::last_error();
                    return false;
                }
                timing.bridge_ms += ms_since(mark);
            }
            const bool resident_route = stage.runtime->fully_resident();
            if (use_token_graph) {
                doorbell_reset(stage.doorbell);
                if (static_cast<int>(stage_index) == impl_->ple_stage_index &&
                    ple_stage.session.ple.table != nullptr) {
                    // The PLE rows sit inside the stage's graph where the PLE layer is, so the
                    // read is collected and uploaded before the launch instead of before that
                    // layer: the same rows, staged earlier on the same queue the graph runs on.
                    mark = std::chrono::steady_clock::now();
                    const bool finished = ple_finish_token(ple_stage.session.ple, nullptr, err);
                    timing.ple_collect_ms += ms_since(mark);
                    ple_guard.disarm();
                    if (!finished) {
                        err = "EXL3 Flash: PLE token staging: " + err;
                        return false;
                    }
                    ple_collected = true;
                }
                mark = std::chrono::steady_clock::now();
                if (!strata::gpu::graph_launch(stage.token_graph, nullptr)) {
                    err = "EXL3 Flash: token graph launch stage " + std::to_string(stage_index) + ": " +
                          strata::gpu::last_error();
                    return false;
                }
                timing.pre_graph_ms += ms_since(mark);
            } else for (int64_t layer = stage.layer_begin; layer < stage.layer_end; ++layer) {
                const size_t local = static_cast<size_t>(layer - stage.layer_begin);
                doorbell_reset(stage.doorbell);
                if (layer == PLE_LAYER && static_cast<int>(stage_index) == impl_->ple_stage_index &&
                    ple_stage.session.ple.table != nullptr) {
                    // The gathered rows are the input of this layer's pre graph, so the
                    // read is collected here and the upload is queued on the same queue
                    // the graph is launched on: the copy is ordered before the graph.
                    mark = std::chrono::steady_clock::now();
                    const bool finished = ple_finish_token(ple_stage.session.ple, nullptr, err);
                    timing.ple_collect_ms += ms_since(mark);
                    ple_guard.disarm();
                    if (!finished) {
                        err = "EXL3 Flash: PLE token staging: " + err;
                        return false;
                    }
                    ple_collected = true;
                }
                mark = std::chrono::steady_clock::now();
                if (!strata::gpu::graph_launch(stage.pre_graphs[local], nullptr)) {
                    err = "EXL3 Flash: pre graph launch layer " + std::to_string(layer) + ": " +
                          strata::gpu::last_error();
                    return false;
                }
                if (resident_route && device_route_plan_enabled()) {
                    // THE EXPERTS RUN FROM THE DEVICE-RESIDENT PLAN.  The pre graph published the
                    // routed ids and the layer's activation on the device, and the plan kernel
                    // resolves every route against the cache's pair table itself: the host does
                    // not wait at the doorbell, builds no plan and moves no rows.  A fully
                    // resident cache has no CPU miss to merge, so the hit rows land in their
                    // `parts` rows directly and the post graph follows them on the queue.  The
                    // doorbell keeps publishing the ids and the weights to its mapped memory, so
                    // the host still sees them and the split they imply - it sees them off the
                    // decode's critical path rather than in it.
                    timing.pre_graph_ms += ms_since(mark);
                    mark = std::chrono::steady_clock::now();
                    if (resident_flow_graph_enabled() && local < stage.expert_graphs.size() &&
                        stage.expert_graphs[local] != nullptr) {
                        // The flow is the same submissions every token, so its graph replays them
                        // as one launch: the clear of the expert rows, the plan kernel and the
                        // grouped experts' kernels run at a graph replay's boundaries.
                        if (!strata::gpu::graph_launch(stage.expert_graphs[local], nullptr)) {
                            err = "EXL3 Flash: expert graph launch layer " + std::to_string(layer) + ": " +
                                  strata::gpu::last_error();
                            return false;
                        }
                    } else {
                        if (!strata::gpu::memset_async(stage.parts_device, 0,
                                                       stage.merge_output.size() * sizeof(float), nullptr)) {
                            err = std::string("EXL3 Flash: clearing expert output: ") + strata::gpu::last_error();
                            return false;
                        }
                        sycl::event resident_done;
                        mark = std::chrono::steady_clock::now();
                        if (!stage.runtime->enqueue_device_hit(Runtime::at(stage.device).stream(nullptr), layer,
                                                               static_cast<const float*>(stage.session.block.mixed),
                                                               stage.doorbell.d_ids, 1,
                                                               static_cast<float*>(stage.parts_device),
                                                               impl_->geometry.n_embd, resident_done, {}, err)) {
                            err = "EXL3 Flash: expert window layer " + std::to_string(layer) + ": " + err;
                            return false;
                        }
                    }
                    timing.expert_prepare_ms += ms_since(mark);
                    mark = std::chrono::steady_clock::now();
                    if (!strata::gpu::graph_launch(stage.post_graphs[local], nullptr)) {
                        err = "EXL3 Flash: post graph launch layer " + std::to_string(layer) + ": " +
                              strata::gpu::last_error();
                        return false;
                    }
                    timing.post_graph_ms += ms_since(mark);
                    continue;
                }
                // The host waits at the doorbell - the window crossing to it - and not for the
                // queue: the poll returns at the ring, which the pre graph rings before its end,
                // so the expert work below starts beside the graph's tail.  The queue's order is
                // what carries the layer's state into the rest.  The probe marks the pre graph so
                // the poll is a query and not a drain; the resident path above waits on neither.
                if (!strata::gpu::event_record(impl_->doorbell_probe, nullptr)) {
                    err = "EXL3 Flash: pre graph doorbell probe layer " + std::to_string(layer) + ": " +
                          strata::gpu::last_error();
                    return false;
                }
                if (!wait_pre_doorbell(stage.doorbell, impl_->doorbell_probe, err)) {
                    err = "EXL3 Flash: pre graph layer " + std::to_string(layer) + ": " + err;
                    return false;
                }
                timing.pre_graph_ms += ms_since(mark);
                // The expert output is cleared on the queue, ordered before the hit kernels that
                // fill the entries they compute.
                if (!strata::gpu::memset_async(stage.parts_device, 0,
                                               stage.merge_output.size() * sizeof(float), nullptr)) {
                    err = std::string("EXL3 Flash: clearing expert output: ") + strata::gpu::last_error();
                    return false;
                }
                sycl::event expert_done;
                mark = std::chrono::steady_clock::now();
                if (!prepare_expert_window(stage_index, layer, stage.doorbell.h_x_f, impl_->geometry.n_embd,
                                           stage.doorbell.h_ids, 1, stage.miss_output.data(), impl_->geometry.n_embd,
                                           nullptr, static_cast<const float*>(stage.session.block.mixed),
                                           static_cast<float*>(stage.parts_device), expert_done, err)) {
                    err = "EXL3 Flash: expert window layer " + std::to_string(layer) + ": " + err;
                    return false;
                }
                timing.expert_prepare_ms += ms_since(mark);
                mark = std::chrono::steady_clock::now();
                if (!wait_expert_window(stage_index, expert_done, err)) {
                    err = "EXL3 Flash: expert window layer " + std::to_string(layer) + ": " + err;
                    return false;
                }
                timing.expert_wait_ms += ms_since(mark);
                mark = std::chrono::steady_clock::now();
                if (!strata::gpu::copy(stage.merge_output.data(), stage.parts_device,
                                       stage.merge_output.size() * sizeof(float))) {
                    err = std::string("EXL3 Flash: expert device-to-host merge copy failed: ") + strata::gpu::last_error();
                    return false;
                }
                for (size_t i = 0; i < stage.merge_output.size(); ++i)
                    stage.merge_output[i] += stage.miss_output[i];
                // The host swaps in a second buffer and the upload is queued, not waited: the post
                // graph runs behind the copy on this queue, and the next layer's download fills the
                // other buffer, so the host never rewrites what a pending upload still reads.
                std::swap(stage.merge_output, stage.merge_upload);
                if (!strata::gpu::copy_async(stage.parts_device, stage.merge_upload.data(),
                                             stage.merge_upload.size() * sizeof(float), nullptr)) {
                    err = std::string("EXL3 Flash: expert host-to-device merge copy failed: ") + strata::gpu::last_error();
                    return false;
                }
                timing.merge_ms += ms_since(mark);
                mark = std::chrono::steady_clock::now();
                // The post graph is issued and not waited: nothing behind it needs the host until
                // the next doorbell, and this queue runs it before whatever comes next - the next
                // layer's pre graph, or the residual handover below.
                if (!strata::gpu::graph_launch(stage.post_graphs[local], nullptr)) {
                    err = "EXL3 Flash: post graph launch layer " + std::to_string(layer) + ": " +
                          strata::gpu::last_error();
                    return false;
                }
                timing.post_graph_ms += ms_since(mark);
            }
            if (stage_index + 1 < impl_->stages.size()) {
                // THE HANDOVER IS ISSUED HERE AND WAITED WHERE THE DATA CROSSES.  The crossing is
                // queued behind this stage's last post graph and marked with `cross_event`; the
                // receiving stage starts with that mark.  A direct copy goes from this stage's
                // residual to the next stage's; the host bridge downloads into `bridge_host`, and
                // the receiving stage uploads from it after the same mark.
                mark = std::chrono::steady_clock::now();
                const size_t residual_bytes = impl_->bridge_host.size() * sizeof(float);
                const bool sent = impl_->residual_direct[stage_index]
                                      ? strata::gpu::copy_async(impl_->stages[stage_index + 1].session.block.R,
                                                                stage.session.block.R, residual_bytes, nullptr)
                                      : strata::gpu::copy_async(impl_->bridge_host.data(), stage.session.block.R,
                                                                residual_bytes, nullptr);
                if (!sent || !strata::gpu::event_record(impl_->cross_event, nullptr)) {
                    err = std::string("EXL3 Flash: stage boundary residual handover failed: ") +
                          strata::gpu::last_error();
                    return false;
                }
                timing.bridge_ms += ms_since(mark);
            }
        }

        if (!ple_collected) {
            err = "EXL3 Flash: the PLE layer was not reached, so its rows were never staged";
            return false;
        }

        auto& last = impl_->stages.back();
        mark = std::chrono::steady_clock::now();
        {
            const OnDevice on(last.device);
            if (!lm_head(last.table, impl_->geometry, last.session.block, static_cast<float*>(impl_->logits_device),
                         nullptr, err) || !strata::gpu::stream_sync(nullptr) ||
                !strata::gpu::copy(impl_->logits_host.data(), impl_->logits_device,
                                   impl_->logits_host.size() * sizeof(float))) {
                err = "EXL3 Flash: output head: " + (err.empty() ? strata::gpu::last_error() : err);
                return false;
            }
        }
        timing.head_ms = ms_since(mark);
        double best = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < impl_->logits_host.size(); ++i) {
            const double value = impl_->logits_host[i];
            if (std::isfinite(value) && value > best) {
                best = value;
                next_token = static_cast<int32_t>(i);
            }
        }
        if (next_token < 0) { err = "EXL3 Flash: output logits are all non-finite"; return false; }
        if (host_logits != nullptr)
            std::copy(impl_->logits_host.begin(), impl_->logits_host.end(), host_logits);
        if (ple_stage.session.ple.table != nullptr) {
            // THE TOKEN WINDOW ADVANCES ONCE PER TOKEN, ON THE MODEL.  The n-gram lookup reads
            // the two tokens before the one being decoded, oldest first.
            impl_->ple_prev[0] = impl_->ple_prev[1];
            impl_->ple_prev[1] = token;
        }
        ++impl_->next_position;
        if (timing.enabled) {
            const auto split = ple_stage.session.ple.table ? ple_stage.session.ple.table->last_collect_timing()
                                                            : strata::kernels::PleTable::CollectTiming{};
            std::fprintf(stderr,
                         "exl3-flash timing pos %lld: total %.1f ms | emb %.1f | ple issue %.1f collect %.1f "
                         "(wait %.1f, decode %.1f) | pre %.1f | expert_prep %.1f | expert_wait %.1f | merge %.1f | "
                         "post %.1f | bridge %.1f | head %.1f\n",
                         static_cast<long long>(pos), timing.total_ms(), timing.embedding_ms, timing.ple_issue_ms,
                         timing.ple_collect_ms, split.wait_us / 1000.0, split.decode_us / 1000.0, timing.pre_graph_ms,
                         timing.expert_prepare_ms, timing.expert_wait_ms, timing.merge_ms, timing.post_graph_ms,
                         timing.bridge_ms, timing.head_ms);
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("EXL3 Flash: forward: ") + e.what();
        return false;
    }
}

bool Exl3FlashModel::validate_token_request(int64_t n_tokens, int64_t batch, std::string& err) const {
    if (!loaded()) { err = "EXL3 Flash: model is not open"; return false; }
    if (n_tokens != 1 || batch != 1) {
        err = "EXL3 Flash: initial coordinator supports only one autoregressive token (batch=1)";
        return false;
    }
    return true;
}

bool Exl3FlashModel::loaded() const { return impl_ != nullptr && impl_->opened; }
size_t Exl3FlashModel::stage_count() const { return impl_ == nullptr ? 0 : impl_->stages.size(); }

Exl3FlashStageView Exl3FlashModel::stage(size_t index) const {
    if (impl_ == nullptr || index >= impl_->stages.size()) return {};
    const auto& stage = impl_->stages[index];
    return {stage.device, stage.layer_begin, stage.layer_end, const_cast<WeightTable*>(&stage.table),
            stage.weights.get(), stage.source.get(), stage.runtime.get()};
}

const strata::ModelSpec& Exl3FlashModel::spec() const {
    static const strata::ModelSpec empty;
    return impl_ == nullptr ? empty : impl_->spec;
}

const ModelGeometry& Exl3FlashModel::geometry() const {
    static const ModelGeometry empty;
    return impl_ == nullptr ? empty : impl_->geometry;
}

const ModelMap& Exl3FlashModel::model_map() const {
    static const ModelMap empty;
    return impl_ == nullptr ? empty : impl_->map;
}

bool Exl3FlashModel::has_mtp() const { return impl_ != nullptr && impl_->has_mtp; }

const ModelMap& Exl3FlashModel::mtp_map() const {
    static const ModelMap empty;
    return impl_ == nullptr ? empty : impl_->mtp_map;
}

const Exl3WeightSet* Exl3FlashModel::mtp_weights() const {
    return impl_ == nullptr ? nullptr : impl_->mtp_weights.get();
}

namespace {

bool snapshot_mirrors_from(const SessionState& session, int64_t n_head,
                           std::vector<std::vector<int32_t>>& steps,
                           std::vector<std::vector<int32_t>>& poss, std::string& err) {
    if (session.qsa_states == nullptr) {
        err = "snapshot needs allocated QSA states";
        return false;
    }
    steps.assign(static_cast<size_t>(session.qsa_alloc), {});
    poss.assign(static_cast<size_t>(session.qsa_alloc), {});
    for (int64_t j = 0; j < session.qsa_alloc; ++j) {
        const QsaState& q = session.qsa_states[session.qsa_ord0 + j];
        if (q.kv_mode != 0) {
            err = "snapshot does not cover streamed KV caches";
            return false;
        }
        if (q.host_step == nullptr || q.host_pos == nullptr) {
            err = "snapshot needs allocated QSA host mirrors";
            return false;
        }
        steps[static_cast<size_t>(j)].assign(q.host_step, q.host_step + (strata::kernels::kStepCount + 1));
        poss[static_cast<size_t>(j)].assign(q.host_pos, q.host_pos + n_head);
    }
    return true;
}

bool snapshot_mirrors_to(const SessionState& session, int64_t n_head,
                         const std::vector<std::vector<int32_t>>& steps,
                         const std::vector<std::vector<int32_t>>& poss, std::string& err) {
    if (session.qsa_states == nullptr) {
        err = "snapshot needs allocated QSA states";
        return false;
    }
    if (steps.size() != static_cast<size_t>(session.qsa_alloc) ||
        poss.size() != static_cast<size_t>(session.qsa_alloc)) {
        err = "snapshot QSA count does not match the model";
        return false;
    }
    for (int64_t j = 0; j < session.qsa_alloc; ++j) {
        const QsaState& q = session.qsa_states[session.qsa_ord0 + j];
        if (steps[static_cast<size_t>(j)].size() !=
                static_cast<size_t>(strata::kernels::kStepCount + 1) ||
            poss[static_cast<size_t>(j)].size() != static_cast<size_t>(n_head)) {
            err = "snapshot QSA mirror size does not match the model";
            return false;
        }
        if (q.host_step == nullptr || q.host_pos == nullptr) {
            err = "snapshot needs allocated QSA host mirrors";
            return false;
        }
        std::copy(steps[static_cast<size_t>(j)].begin(), steps[static_cast<size_t>(j)].end(), q.host_step);
        std::copy(poss[static_cast<size_t>(j)].begin(), poss[static_cast<size_t>(j)].end(), q.host_pos);
    }
    return true;
}

bool snapshot_ensure_device(void*& slot, size_t bytes, std::string& err, const char* what) {
    if (slot != nullptr) {
        strata::gpu::free(slot);
        slot = nullptr;
    }
    slot = strata::gpu::alloc_device(bytes);
    if (slot == nullptr) {
        err = std::string("snapshot cannot allocate ") + what;
        return false;
    }
    return true;
}

}  // namespace

bool Exl3FlashModel::save_snapshot(Exl3FlashSnapshot& snap, std::string& err) {
    err.clear();
    if (!loaded()) { err = "EXL3 Flash: model is not open"; return false; }
    try {
        snap.stages.resize(impl_->stages.size());
        for (size_t index = 0; index < impl_->stages.size(); ++index) {
            auto& stage = impl_->stages[index];
            auto& out = snap.stages[index];
            const OnDevice on(stage.device);
            if (stage.session_arena == nullptr || stage.session_arena_bytes == 0) {
                err = "EXL3 Flash: snapshot needs an allocated stage session";
                return false;
            }
            if (out.arena == nullptr || out.arena_bytes != stage.session_arena_bytes) {
                if (!snapshot_ensure_device(out.arena, stage.session_arena_bytes, err, "stage arena")) return false;
                out.arena_bytes = stage.session_arena_bytes;
            }
            if (!strata::gpu::copy(out.arena, stage.session_arena, stage.session_arena_bytes)) {
                err = std::string("EXL3 Flash: snapshot stage copy failed: ") + strata::gpu::last_error();
                return false;
            }
            if (stage.session.ple.emb_dev != nullptr) {
                if (out.ple_emb_dev == nullptr &&
                    !snapshot_ensure_device(out.ple_emb_dev,
                                            static_cast<size_t>(strata::kernels::NG_N_EMBD) * sizeof(float), err,
                                            "PLE embedding row"))
                    return false;
                if (!strata::gpu::copy(out.ple_emb_dev, stage.session.ple.emb_dev,
                                       static_cast<size_t>(strata::kernels::NG_N_EMBD) * sizeof(float))) {
                    err = std::string("EXL3 Flash: snapshot PLE copy failed: ") + strata::gpu::last_error();
                    return false;
                }
            } else if (out.ple_emb_dev != nullptr) {
                strata::gpu::free(out.ple_emb_dev);
                out.ple_emb_dev = nullptr;
            }
            if (stage.session.ple.scratch != nullptr) {
                if (out.ple_scratch == nullptr &&
                    !snapshot_ensure_device(out.ple_scratch, strata::core::ple_run_scratch_bytes(), err,
                                            "PLE scratch"))
                    return false;
                if (!strata::gpu::copy(out.ple_scratch, stage.session.ple.scratch,
                                       strata::core::ple_run_scratch_bytes())) {
                    err = std::string("EXL3 Flash: snapshot PLE scratch copy failed: ") + strata::gpu::last_error();
                    return false;
                }
            } else if (out.ple_scratch != nullptr) {
                strata::gpu::free(out.ple_scratch);
                out.ple_scratch = nullptr;
            }
            if (!snapshot_mirrors_from(stage.session, impl_->geometry.n_head, out.qsa_step, out.qsa_pos, err)) {
                err = "EXL3 Flash snapshot: " + err;
                return false;
            }
            if (!strata::gpu::stream_sync(nullptr)) {
                err = std::string("EXL3 Flash: snapshot stage sync failed: ") + strata::gpu::last_error();
                return false;
            }
        }
        if (impl_->has_mtp) {
            const OnDevice on_mtp(impl_->mtp_device);
            if (impl_->mtp_session_arena == nullptr || impl_->mtp_session_arena_bytes == 0) {
                err = "EXL3 Flash: snapshot needs an allocated MTP session";
                return false;
            }
            if (snap.mtp_arena == nullptr || snap.mtp_arena_bytes != impl_->mtp_session_arena_bytes) {
                if (!snapshot_ensure_device(snap.mtp_arena, impl_->mtp_session_arena_bytes, err, "MTP arena"))
                    return false;
                snap.mtp_arena_bytes = impl_->mtp_session_arena_bytes;
            }
            if (!strata::gpu::copy(snap.mtp_arena, impl_->mtp_session_arena, impl_->mtp_session_arena_bytes)) {
                err = std::string("EXL3 Flash: snapshot MTP copy failed: ") + strata::gpu::last_error();
                return false;
            }
            if (!snapshot_mirrors_from(impl_->mtp_session, impl_->mtp_geometry.n_head, snap.mtp_qsa_step,
                                       snap.mtp_qsa_pos, err)) {
                err = "EXL3 Flash snapshot: " + err;
                return false;
            }
            if (!strata::gpu::stream_sync(nullptr)) {
                err = std::string("EXL3 Flash: snapshot MTP sync failed: ") + strata::gpu::last_error();
                return false;
            }
        } else {
            if (snap.mtp_arena != nullptr) strata::gpu::free(snap.mtp_arena);
            snap.mtp_arena = nullptr;
            snap.mtp_arena_bytes = 0;
            snap.mtp_qsa_step.clear();
            snap.mtp_qsa_pos.clear();
        }
        snap.next_position = impl_->next_position;
        snap.ple_prev[0] = impl_->ple_prev[0];
        snap.ple_prev[1] = impl_->ple_prev[1];
        snap.ple_token = impl_->ple_token;
        snap.seeded = impl_->seeded;
        snap.initialized = impl_->initialized;
        return true;
    } catch (const std::exception& e) {
        err = std::string("EXL3 Flash snapshot save: ") + e.what();
        return false;
    }
}

bool Exl3FlashModel::restore_snapshot(const Exl3FlashSnapshot& snap, std::string& err) {
    err.clear();
    if (!loaded()) { err = "EXL3 Flash: model is not open"; return false; }
    try {
        if (snap.stages.size() != impl_->stages.size()) {
            err = "EXL3 Flash: snapshot stage count does not match the model";
            return false;
        }
        for (size_t index = 0; index < impl_->stages.size(); ++index) {
            auto& stage = impl_->stages[index];
            const auto& in = snap.stages[index];
            const OnDevice on(stage.device);
            if (in.arena == nullptr || in.arena_bytes != stage.session_arena_bytes ||
                stage.session_arena == nullptr) {
                err = "EXL3 Flash: snapshot arena does not match the model";
                return false;
            }
            if (!strata::gpu::copy(stage.session_arena, in.arena, stage.session_arena_bytes)) {
                err = std::string("EXL3 Flash: snapshot stage restore failed: ") + strata::gpu::last_error();
                return false;
            }
            if (stage.session.ple.emb_dev != nullptr) {
                if (in.ple_emb_dev == nullptr ||
                    !strata::gpu::copy(stage.session.ple.emb_dev, in.ple_emb_dev,
                                       static_cast<size_t>(strata::kernels::NG_N_EMBD) * sizeof(float))) {
                    err = "EXL3 Flash: snapshot PLE restore is incomplete";
                    return false;
                }
            }
            if (stage.session.ple.scratch != nullptr) {
                if (in.ple_scratch == nullptr ||
                    !strata::gpu::copy(stage.session.ple.scratch, in.ple_scratch,
                                       strata::core::ple_run_scratch_bytes())) {
                    err = "EXL3 Flash: snapshot PLE scratch restore is incomplete";
                    return false;
                }
            }
            if (!snapshot_mirrors_to(stage.session, impl_->geometry.n_head, in.qsa_step, in.qsa_pos, err)) {
                err = "EXL3 Flash snapshot: " + err;
                return false;
            }
            if (!strata::gpu::stream_sync(nullptr)) {
                err = std::string("EXL3 Flash: snapshot stage sync failed: ") + strata::gpu::last_error();
                return false;
            }
        }
        if (impl_->has_mtp) {
            const OnDevice on_mtp(impl_->mtp_device);
            if (snap.mtp_arena == nullptr || snap.mtp_arena_bytes != impl_->mtp_session_arena_bytes ||
                impl_->mtp_session_arena == nullptr) {
                err = "EXL3 Flash: snapshot MTP arena does not match the model";
                return false;
            }
            if (!strata::gpu::copy(impl_->mtp_session_arena, snap.mtp_arena, impl_->mtp_session_arena_bytes)) {
                err = std::string("EXL3 Flash: snapshot MTP restore failed: ") + strata::gpu::last_error();
                return false;
            }
            if (!snapshot_mirrors_to(impl_->mtp_session, impl_->mtp_geometry.n_head, snap.mtp_qsa_step,
                                     snap.mtp_qsa_pos, err)) {
                err = "EXL3 Flash snapshot: " + err;
                return false;
            }
            if (!strata::gpu::stream_sync(nullptr)) {
                err = std::string("EXL3 Flash: snapshot MTP sync failed: ") + strata::gpu::last_error();
                return false;
            }
        }
        impl_->next_position = snap.next_position;
        impl_->ple_prev[0] = snap.ple_prev[0];
        impl_->ple_prev[1] = snap.ple_prev[1];
        impl_->ple_token = snap.ple_token;
        impl_->seeded = snap.seeded;
        impl_->initialized = snap.initialized;
        return true;
    } catch (const std::exception& e) {
        err = std::string("EXL3 Flash snapshot restore: ") + e.what();
        return false;
    }
}

void Exl3FlashModel::release_snapshot(Exl3FlashSnapshot& snap) noexcept {
    try {
        for (auto& stage : snap.stages) {
            if (stage.arena != nullptr) strata::gpu::free(stage.arena);
            if (stage.ple_emb_dev != nullptr) strata::gpu::free(stage.ple_emb_dev);
            if (stage.ple_scratch != nullptr) strata::gpu::free(stage.ple_scratch);
            stage.arena = nullptr;
            stage.arena_bytes = 0;
            stage.ple_emb_dev = nullptr;
            stage.ple_scratch = nullptr;
            stage.qsa_step.clear();
            stage.qsa_pos.clear();
        }
        snap.stages.clear();
        if (snap.mtp_arena != nullptr) strata::gpu::free(snap.mtp_arena);
        snap.mtp_arena = nullptr;
        snap.mtp_arena_bytes = 0;
        snap.mtp_qsa_step.clear();
        snap.mtp_qsa_pos.clear();
    } catch (...) {
    }
}

bool Exl3FlashModel::mtp_draft(int32_t next_token, int64_t trunk_pos, int32_t& draft, float* host_logits,
                               std::string& err, float* draft_prob) {
    err.clear();
    draft = -1;
    if (draft_prob != nullptr) *draft_prob = 0.0f;
    if (!loaded()) { err = "EXL3 Flash: model is not open"; return false; }
    if (!impl_->has_mtp || !impl_->mtp_runtime || !impl_->mtp_weights || impl_->mtp_table == nullptr ||
        impl_->mtp_fc_scratch == nullptr || impl_->mtp_parts_device == nullptr) {
        err = "EXL3 Flash: MTP draft block is not bound";
        return false;
    }
    if (!validate_token_request(1, 1, err)) return false;
    if (next_token < 0 || static_cast<int64_t>(next_token) >= impl_->map.geometry.vocab_size) {
        err = "EXL3 Flash: draft token is outside the embedding vocabulary";
        return false;
    }
    if (trunk_pos < 0 || trunk_pos >= impl_->options.max_cells || impl_->next_position != trunk_pos + 1) {
        err = "EXL3 Flash: MTP draft needs the trunk just past trunk_pos";
        return false;
    }
    try {
        const int64_t N = impl_->geometry.n_embd;
        const int64_t HC = impl_->geometry.hc;
        const int64_t k = impl_->options.routed_k;
        auto& last = impl_->stages.back();
        const OnDevice on(impl_->mtp_device);
        float* scratch = static_cast<float*>(impl_->mtp_fc_scratch);
        float* emb = scratch;
        float* en = emb + N;
        float* hn = en + N;
        float* e2 = hn + HC * N;
        float* h2 = e2 + N;
        const WeightRef* enorm = impl_->mtp_table->find("mtp.pre_fc_norm_embedding.weight");
        const WeightRef* hnorm = impl_->mtp_table->find("mtp.pre_fc_norm_hidden.weight");
        const WeightRef* fc_e = impl_->mtp_table->find("mtp.fc_embedding.weight");
        const WeightRef* fc_h = impl_->mtp_table->find("mtp.fc_hidden.weight");
        if (enorm == nullptr || hnorm == nullptr || fc_e == nullptr || fc_h == nullptr ||
            fc_e->exl3 == nullptr || fc_h->exl3 == nullptr) {
            err = "EXL3 Flash: MTP draft projections are missing";
            return false;
        }
        void* ds = impl_->mtp_stream;
        impl_->embedding->gather_one(next_token, emb, ds);
        if (impl_->mtp_body_graph != nullptr) {
            // CAPTURED DRAFT: the gather above is the only step whose argument changes per
            // draft; everything after it replays as the one graph captured at open, and the
            // logits it leaves are read back by the shared tail below.
            if (!strata::gpu::graph_launch(impl_->mtp_body_graph, ds)) {
                err = std::string("EXL3 Flash MTP draft body graph launch failed: ") +
                      strata::gpu::last_error();
                return false;
            }
            if (!strata::gpu::stream_sync(ds) ||
                !strata::gpu::copy(impl_->logits_host.data(), impl_->logits_device,
                                   impl_->logits_host.size() * sizeof(float))) {
                err = std::string("EXL3 Flash MTP draft head readback failed: ") + strata::gpu::last_error();
                return false;
            }
            double best = -std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < impl_->logits_host.size(); ++i) {
                const double value = impl_->logits_host[i];
                if (std::isfinite(value) && value > best) {
                    best = value;
                    draft = static_cast<int32_t>(i);
                }
            }
            if (draft < 0) { err = "EXL3 Flash: MTP draft logits are all non-finite"; return false; }
            if (draft_prob != nullptr)
                *draft_prob = exl3_flash_draft_probability(impl_->logits_host.data(), impl_->logits_host.size(), draft);
            if (host_logits != nullptr)
                std::copy(impl_->logits_host.begin(), impl_->logits_host.end(), host_logits);
            return true;
        }
        const float eps = static_cast<float>(impl_->spec.rms_eps);
        strata::kernels::native_qsa_rms_norm_weighted(emb, static_cast<const float*>(enorm->data), en,
                                                      static_cast<int>(N), 1, eps, ds);
        strata::kernels::native_qsa_rms_norm_weighted(last.session.block.R,
                                                      static_cast<const float*>(hnorm->data), hn,
                                                      static_cast<int>(HC * N), 1, eps, ds);
        if (!fc_e->exl3->run(en, N, e2, N, 1, ds, err)) {
            err = "EXL3 Flash MTP draft embedding projection: " + err;
            return false;
        }
        for (int64_t s = 0; s < HC; ++s) {
            if (!fc_h->exl3->run(hn + s * N, N, h2 + s * N, N, 1, ds, err)) {
                err = "EXL3 Flash MTP draft hidden projection: " + err;
                return false;
            }
        }
        strata::kernels::add_streams_broadcast(h2, e2, impl_->mtp_session.block.R, N, static_cast<int>(HC), 1,
                                               ds);
        QsaState& mq = impl_->mtp_session.qsa_states[impl_->mtp_session.qsa_ord0];
        if (!block_layer_pre(*impl_->mtp_table, impl_->mtp_geometry, 0, trunk_pos, 0, impl_->mtp_session.gdn, mq,
                             impl_->mtp_session.qsa_bufs, impl_->mtp_session.moe, k, impl_->mtp_session.block,
                             ds, err)) {
            err = "EXL3 Flash MTP draft block: " + err;
            return false;
        }
        if (impl_->mtp_device_draft) {
            // DEVICE DRAFT: the routed ids and the activation stay on the device. The captured
            // expert graph resolves the routes against the draft cache's pair table on the
            // device and writes every expert row into mtp_parts_device; no stream sync, no
            // router readback and no CPU-pool work sits in the draft loop. The post block
            // follows on the same in-order stream.
            if (impl_->mtp_expert_graph != nullptr) {
                if (!strata::gpu::graph_launch(impl_->mtp_expert_graph, ds)) {
                    err = std::string("EXL3 Flash MTP draft expert graph launch failed: ") +
                          strata::gpu::last_error();
                    return false;
                }
            } else {
                sycl::event done;
                sycl::queue& queue = Runtime::at(impl_->mtp_device).stream(ds);
                if (!strata::gpu::memset_async(impl_->mtp_parts_device, 0,
                                               static_cast<size_t>(k) * static_cast<size_t>(N) * sizeof(float),
                                               ds) ||
                    !impl_->mtp_runtime->enqueue_device_hit(
                        queue, 0, static_cast<const float*>(impl_->mtp_session.block.mixed),
                        static_cast<const int32_t*>(impl_->mtp_session.moe.ids), 1,
                        static_cast<float*>(impl_->mtp_parts_device), N, done, {}, err)) {
                    err = "EXL3 Flash MTP draft experts: " + err;
                    return false;
                }
            }
        } else {
            if (!strata::gpu::stream_sync(ds)) {
                err = std::string("EXL3 Flash MTP draft pre sync failed: ") + strata::gpu::last_error();
                return false;
            }
            if (!strata::gpu::copy(impl_->mtp_host_ids.data(), impl_->mtp_session.moe.ids,
                                   static_cast<size_t>(k) * sizeof(int32_t)) ||
                !strata::gpu::copy(impl_->mtp_host_weights.data(), impl_->mtp_session.moe.weights,
                                   static_cast<size_t>(k) * sizeof(float)) ||
                !strata::gpu::copy(impl_->mtp_host_x.data(), impl_->mtp_session.block.mixed,
                                   static_cast<size_t>(N) * sizeof(float))) {
                err = std::string("EXL3 Flash MTP draft router readback failed: ") + strata::gpu::last_error();
                return false;
            }
            if (!impl_->mtp_runtime->prepare_window(0, impl_->mtp_host_x.data(), N, impl_->mtp_host_ids.data(), 1,
                                                    impl_->mtp_cpu_out.data(), N, err) ||
                !impl_->mtp_runtime->run_cpu(err)) {
                err = "EXL3 Flash MTP draft experts: " + err;
                return false;
            }
            impl_->mtp_runtime->mark_gpu_consumed();
            if (!strata::gpu::copy_async(static_cast<float*>(impl_->mtp_parts_device), impl_->mtp_cpu_out.data(),
                                         static_cast<size_t>(k) * static_cast<size_t>(N) * sizeof(float), ds)) {
                err = std::string("EXL3 Flash MTP draft expert upload failed: ") + strata::gpu::last_error();
                return false;
            }
        }
        if (!block_layer_post(*impl_->mtp_table, impl_->mtp_geometry, 0, k, impl_->mtp_session.moe,
                              impl_->mtp_session.block, static_cast<const float*>(impl_->mtp_parts_device),
                              ds, err)) {
            err = "EXL3 Flash MTP draft combine: " + err;
            return false;
        }
        const BlockBuffers head_bb = impl_->mtp_session.block;
        if (!lm_head(last.table, impl_->geometry, head_bb, static_cast<float*>(impl_->logits_device), ds,
                     err) ||
            !strata::gpu::stream_sync(ds) ||
            !strata::gpu::copy(impl_->logits_host.data(), impl_->logits_device,
                               impl_->logits_host.size() * sizeof(float))) {
            err = "EXL3 Flash MTP draft head: " + (err.empty() ? strata::gpu::last_error() : err);
            return false;
        }
        double best = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < impl_->logits_host.size(); ++i) {
            const double value = impl_->logits_host[i];
            if (std::isfinite(value) && value > best) {
                best = value;
                draft = static_cast<int32_t>(i);
            }
        }
        if (draft < 0) { err = "EXL3 Flash: MTP draft logits are all non-finite"; return false; }
        if (draft_prob != nullptr)
            *draft_prob = exl3_flash_draft_probability(impl_->logits_host.data(), impl_->logits_host.size(), draft);
        if (host_logits != nullptr)
            std::copy(impl_->logits_host.begin(), impl_->logits_host.end(), host_logits);
        return true;
    } catch (const std::exception& e) {
        err = std::string("EXL3 Flash MTP draft: ") + e.what();
        return false;
    }
}

NativeEmbed* Exl3FlashModel::embedding() const { return impl_ == nullptr ? nullptr : impl_->embedding.get(); }
Exl3Ple* Exl3FlashModel::ple() const { return impl_ == nullptr ? nullptr : impl_->ple.get(); }
strata::kernels::PleTable* Exl3FlashModel::ple_table() const {
    return impl_ == nullptr ? nullptr : impl_->ple_table.get();
}

const std::shared_ptr<const strata::artifact::exl3::Archive>& Exl3FlashModel::archive() const {
    static const std::shared_ptr<const strata::artifact::exl3::Archive> empty;
    return impl_ == nullptr ? empty : impl_->archive;
}

}  // namespace strata::core
