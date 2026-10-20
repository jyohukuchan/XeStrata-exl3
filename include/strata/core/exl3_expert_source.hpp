// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Bounded persistent compressed EXL3 ExpertSource, with a window-scoped fallback.
#pragma once

#include "strata/artifact/exl3_expert.hpp"
#include "strata/core/expert_source.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// The caller finishes the previous CPU jobs and consumes its GPU plan before
/// begin_layer(). Every blob returned in a window stays pinned until that call.
/// LRU residence does not extend this borrowed-pointer contract: transient()
/// remains true, and longer-lived consumers must use copy_blob().
class Exl3ExpertSource final : public ExpertSource {
public:
    using Names = std::vector<strata::artifact::exl3::ExpertTensorNames>;

    struct CachePolicy {
        /// Automatic uses at most the compressed archive size. The environment
        /// STRATA_EXL3_SOURCE_CACHE_MIB is an additional cap; zero disables it.
        uint64_t max_bytes = std::numeric_limits<uint64_t>::max();
        /// Match the existing resident expert policy. Reserve future allocations
        /// here; allocations already present are included in HostMemory.available.
        uint64_t headroom_bytes = 8ull << 30;
        uint64_t reserved_bytes = 0;
    };

    struct CacheStats {
        uint64_t requested_bytes = 0;
        uint64_t capacity_bytes = 0;
        uint64_t admission_bytes = 0;     // may shrink under later host pressure
        uint64_t resident_bytes = 0;       // ready + loading persistent blobs
        uint64_t peak_resident_bytes = 0;
        uint64_t transient_bytes = 0;      // only the active window fallback
        uint64_t hits = 0;                 // unique window requests already resident
        uint64_t misses = 0;               // unique window requests needing a load
        uint64_t evictions = 0;
        uint64_t fallbacks = 0;
        uint64_t materializations = 0;     // includes failed attempts and copy_blob
        uint64_t copy_hits = 0;
        uint64_t copy_materializations = 0;
        uint64_t payload_read_bytes = 0;   // successful source tensor spans, not padding
        uint64_t materialized_bytes = 0;   // successful compressed/raw layout bytes
        uint64_t materialization_ns = 0;
    };

    struct PrewarmResult {
        uint64_t requested_experts = 0;
        uint64_t requested_bytes = 0;
        uint64_t loaded_experts = 0;
        uint64_t loaded_bytes = 0;
        uint64_t admission_bytes = 0;
    };

    /// The files the capacity check reads: the host's RAM, its cgroup list and the cgroup tree.  The
    /// defaults are the real ones; a test points them at a fixture, the way host_available_memory's own
    /// parameters do.
    struct CapacityFiles {
        std::string meminfo = "/proc/meminfo";
        std::string self_cgroup = "/proc/self/cgroup";
        std::string cgroup_root = "/sys/fs/cgroup";
    };

    Exl3ExpertSource() = default;
    Exl3ExpertSource(std::shared_ptr<const strata::artifact::exl3::Archive> archive, int64_t n_layers,
                     int64_t n_expert, Names names);
    ~Exl3ExpertSource() override;
    Exl3ExpertSource(const Exl3ExpertSource&) = delete;
    Exl3ExpertSource& operator=(const Exl3ExpertSource&) = delete;

    bool open(std::shared_ptr<const strata::artifact::exl3::Archive> archive, int64_t n_layers, int64_t n_expert,
              Names names, std::string& err);
    bool open(std::shared_ptr<const strata::artifact::exl3::Archive> archive, int64_t n_layers, int64_t n_expert,
              Names names, std::string& err, const CachePolicy& policy);
    bool open(const std::filesystem::path& directory, int64_t n_layers, int64_t n_expert, Names names,
              std::string& err);

    const uint8_t* blob(int64_t layer, int64_t expert) override;
    bool transient(int64_t layer, int64_t expert) const override;
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    const strata::kernels::exl3::ExpertBlobLayout* exl3_layout(int64_t layer, int64_t expert) const override;
    void begin_layer(int64_t layer, const int32_t* ids, int64_t k) override;
    int64_t reads() const override { return reads_.load(std::memory_order_relaxed); }
    CacheStats cache_stats() const;

    /// Optional startup prewarm. Nonzero mask entries exclude GPU residents or
    /// pairs owned by another runtime; the mask has n_layers*n_expert entries.
    /// Only start if every remaining compressed/raw payload fits the current
    /// persistent budget. Never evict, pin a window, or use transient storage.
    /// False leaves a bounded lazy cache (possibly partially warmed after later
    /// memory pressure/I/O failure), and explains why; no active window changes.
    /// STRATA_EXL3_SOURCE_PREWARM=0 disables the optional preload.
    bool prewarm_cpu(const std::vector<uint8_t>& excluded, PrewarmResult& result, std::string& why);

    /// Point the capacity check at other files.  Set it before open(), which reads the host's room.
    void set_capacity_files(CapacityFiles files) { capacity_files_ = std::move(files); }

    const std::string& last_error() const { return failure_; }
    int64_t n_layers() const { return n_layers_; }
    int64_t n_expert() const { return n_expert_; }

private:
    static constexpr size_t kNone = std::numeric_limits<size_t>::max();
    enum class State : uint8_t { empty, loading, ready };
    struct Entry {
        std::unique_ptr<strata::artifact::exl3::ExpertBlob> blob;
        State state = State::empty;
        bool persistent = false;
        bool window_pin = false;
        bool counted_window = false;
        bool failed_window = false;
        size_t previous = kNone;
        size_t next = kNone;
    };

    bool valid_pair(int64_t layer, int64_t expert) const;
    size_t index(int64_t layer, int64_t expert) const;
    bool ensure_materialized_locked(size_t at, std::unique_lock<std::mutex>& lock);
    bool materialize(size_t at, std::unique_ptr<strata::artifact::exl3::ExpertBlob>& blob, std::string& err,
                     bool copy);
    void clear_locked();
    void release_window_locked();
    void touch_locked(size_t at);
    void unlink_locked(size_t at);
    bool evict_one_locked();
    void refresh_capacity_locked(bool evict = true);
    /// The bytes the source may still admit: the slack of the budget the last reading of the host's room
    /// gave.  Zero when the room could not be read: the source then admits nothing.
    uint64_t admission_headroom_locked() const {
        return stats_.admission_bytes > stats_.resident_bytes ? stats_.admission_bytes - stats_.resident_bytes : 0;
    }

    std::shared_ptr<const strata::artifact::exl3::Archive> archive_;
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
    Names names_;
    std::vector<strata::kernels::exl3::ExpertBlobLayout> layouts_;
    std::vector<uint64_t> payload_bytes_;
    std::vector<Entry> entries_;
    std::vector<size_t> active_indices_;
    /// open/begin_layer take exclusive ownership; concurrent blob/copy calls
    /// share it. Lock order is lifecycle_ then mutex_. Disk reads release only
    /// mutex_, so reset cannot invalidate metadata during a load.
    mutable std::shared_mutex lifecycle_;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    CachePolicy policy_;
    CacheStats stats_;
    uint64_t maximum_blob_bytes_ = 0;
    uint64_t startup_capacity_bytes_ = 0;
    /// True when the room was read again for the shortage at hand: a reading that did not give the room is
    /// not repeated for every blob that does not fit.
    bool room_reread_ = false;
    /// The layer of the last window, and whether there was one.  The layers are visited in order, so a
    /// layer that is not the successor of the last one starts a new pass over them.
    int64_t last_layer_ = -1;
    bool layer_known_ = false;
    /// The files the capacity check reads.
    CapacityFiles capacity_files_;
    size_t newest_ = kNone;
    size_t oldest_ = kNone;
    int64_t active_layer_ = -1;
    std::string failure_;
    std::atomic<int64_t> reads_{0};
};

}  // namespace strata::core
