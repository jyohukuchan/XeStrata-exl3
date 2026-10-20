// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/core/exl3_expert_source.hpp"
#include "strata/core/exl3_prepare_profile.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::core {
namespace {

uint64_t add_saturated(uint64_t a, uint64_t b) {
    return b > std::numeric_limits<uint64_t>::max() - a ? std::numeric_limits<uint64_t>::max() : a + b;
}

uint64_t subtract_floor(uint64_t a, uint64_t b) { return a > b ? a - b : 0; }

std::string projection_base(const std::string& name) {
    constexpr const char* suffix = ".weight";
    const size_t n = std::strlen(suffix);
    return name.size() > n && name.compare(name.size() - n, n, suffix) == 0 ? name.substr(0, name.size() - n) : name;
}

uint64_t projection_payload_bytes(const strata::artifact::exl3::Archive& archive, const std::string& name) {
    const std::string base = projection_base(name);
    if (const auto* packed = archive.find_linear(base)) {
        uint64_t bytes = packed->trellis.bytes;
        const auto& su = packed->suh ? packed->suh : packed->su;
        const auto& sv = packed->svh ? packed->svh : packed->sv;
        if (su) bytes = add_saturated(bytes, su->bytes);
        if (sv) bytes = add_saturated(bytes, sv->bytes);
        if (packed->bias) bytes = add_saturated(bytes, packed->bias->bytes);
        return bytes;
    }
    uint64_t bytes = 0;
    if (const auto* raw = archive.find_tensor(base + ".weight")) bytes = raw->bytes;
    if (const auto* bias = archive.find_tensor(base + ".bias")) bytes = add_saturated(bytes, bias->bytes);
    return bytes;
}

uint64_t environment_cap() {
    const char* text = std::getenv("STRATA_EXL3_SOURCE_CACHE_MIB");
    if (text == nullptr) return std::numeric_limits<uint64_t>::max();
    uint64_t mib = 0;
    const char* end = text + std::strlen(text);
    const auto result = std::from_chars(text, end, mib);
    if (text == end || result.ec != std::errc{} || result.ptr != end ||
        mib > std::numeric_limits<uint64_t>::max() / (1ull << 20)) {
        std::fprintf(stderr, "Exl3ExpertSource: invalid STRATA_EXL3_SOURCE_CACHE_MIB; persistent cache disabled\n");
        return 0;
    }
    return mib << 20;
}

/// The device's copy of a host blob: the same bytes in the canonical tile
/// order.  A resident blob may hold the CPU block order, and the device
/// kernels read the canonical one only.
void copy_blob_canonical(const strata::artifact::exl3::ExpertBlob& blob, uint8_t* dst) {
    std::memcpy(dst, blob.data(), blob.bytes());
    if (blob.tile_order != strata::kernels::exl3::TileOrder::cpu_block) return;
    const strata::kernels::exl3::ExpertProjectionLayout* projections[3] = {&blob.layout.gate, &blob.layout.up,
                                                                           &blob.layout.down};
    for (const auto* projection : projections) {
        if (projection->storage != strata::kernels::exl3::ProjectionStorage::packed_trellis) continue;
        if (projection->trellis.bytes == 0) continue;
        const auto* source = reinterpret_cast<const uint16_t*>(blob.data() + projection->trellis.offset);
        auto* destination = reinterpret_cast<uint16_t*>(dst + projection->trellis.offset);
        strata::artifact::exl3::copy_packed_tiles(source, destination, projection->k, projection->n, projection->bits,
                                                  strata::kernels::exl3::TileOrder::cpu_block,
                                                  strata::kernels::exl3::TileOrder::canonical);
    }
}

/// Whether the host cache keeps the CPU block tile order.  The switch is a
/// diagnostic knob: with it off the host cache holds the format's own order,
/// and every result is the same.  The order of a resident blob and the order
/// the CPU kernels read always agree.
bool cpu_block_order_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_EXL3_CPU_BLOCK_ORDER");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

}  // namespace

Exl3ExpertSource::Exl3ExpertSource(std::shared_ptr<const strata::artifact::exl3::Archive> archive,
                                   int64_t n_layers, int64_t n_expert, Names names) {
    std::string err;
    if (!open(std::move(archive), n_layers, n_expert, std::move(names), err)) failure_ = std::move(err);
}

Exl3ExpertSource::~Exl3ExpertSource() {
    const auto s = cache_stats();
    if (s.materializations == 0 && s.hits == 0) return;
    std::fprintf(stderr,
                 "Exl3ExpertSource: capacity=%llu resident=%llu peak=%llu hits=%llu misses=%llu "
                 "evictions=%llu fallbacks=%llu materializations=%llu payload_read_bytes=%llu "
                 "materialize_ms=%.3f copy_hits=%llu copy_materializations=%llu\n",
                 (unsigned long long) s.capacity_bytes, (unsigned long long) s.resident_bytes,
                 (unsigned long long) s.peak_resident_bytes, (unsigned long long) s.hits,
                 (unsigned long long) s.misses, (unsigned long long) s.evictions,
                 (unsigned long long) s.fallbacks, (unsigned long long) s.materializations,
                 (unsigned long long) s.payload_read_bytes, (double) s.materialization_ns / 1.0e6,
                 (unsigned long long) s.copy_hits, (unsigned long long) s.copy_materializations);
}

size_t Exl3ExpertSource::index(int64_t layer, int64_t expert) const {
    return static_cast<size_t>(layer) * static_cast<size_t>(n_expert_) + static_cast<size_t>(expert);
}

bool Exl3ExpertSource::valid_pair(int64_t layer, int64_t expert) const {
    return archive_ != nullptr && layer >= 0 && layer < n_layers_ && expert >= 0 && expert < n_expert_;
}

void Exl3ExpertSource::clear_locked() {
    archive_.reset(); names_.clear(); layouts_.clear(); payload_bytes_.clear(); entries_.clear(); active_indices_.clear();
    n_layers_ = n_expert_ = 0;
    newest_ = oldest_ = kNone;
    active_layer_ = -1;
    maximum_blob_bytes_ = startup_capacity_bytes_ = 0;
    room_reread_ = false;
    last_layer_ = -1;
    layer_known_ = false;
    stats_ = {};
    failure_.clear();
    reads_.store(0, std::memory_order_relaxed);
}

bool Exl3ExpertSource::open(std::shared_ptr<const strata::artifact::exl3::Archive> archive, int64_t n_layers,
                            int64_t n_expert, Names names, std::string& err) {
    return open(std::move(archive), n_layers, n_expert, std::move(names), err, CachePolicy{});
}

bool Exl3ExpertSource::open(std::shared_ptr<const strata::artifact::exl3::Archive> archive, int64_t n_layers,
                            int64_t n_expert, Names names, std::string& err, const CachePolicy& policy) {
    std::unique_lock<std::shared_mutex> lifecycle(lifecycle_);
    std::lock_guard<std::mutex> lock(mutex_);
    err.clear();
    clear_locked();
    const auto fail = [&](const std::string& why) {
        clear_locked();
        err = failure_ = why;
        return false;
    };
    if (archive == nullptr || n_layers <= 0 || n_expert <= 0)
        return fail("exl3 source: archive and positive dimensions are required");
    if (static_cast<uint64_t>(n_layers) > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(n_expert))
        return fail("exl3 source: layer/expert dimensions overflow");
    const uint64_t count = static_cast<uint64_t>(n_layers) * static_cast<uint64_t>(n_expert);
    if (count > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) || names.size() != static_cast<size_t>(count))
        return fail("exl3 source: flat name table size does not match layer/expert dimensions");
    try {
        archive_ = std::move(archive);
        n_layers_ = n_layers; n_expert_ = n_expert;
        names_ = std::move(names);
        policy_ = policy;
        layouts_.resize(names_.size());
        payload_bytes_.resize(names_.size());
        entries_.resize(names_.size());
        active_indices_.reserve(static_cast<size_t>(n_expert));
        uint64_t total = 0;
        for (size_t i = 0; i < names_.size(); ++i) {
            std::string why;
            if (!strata::artifact::exl3::describe_expert(*archive_, names_[i], layouts_[i], why))
                return fail("exl3 source: expert " + std::to_string(i) + ": " + why);
            total = add_saturated(total, layouts_[i].total_bytes);
            maximum_blob_bytes_ = std::max(maximum_blob_bytes_, layouts_[i].total_bytes);
            uint64_t bytes = projection_payload_bytes(*archive_, names_[i].gate);
            bytes = add_saturated(bytes, projection_payload_bytes(*archive_, names_[i].up));
            payload_bytes_[i] = add_saturated(bytes, projection_payload_bytes(*archive_, names_[i].down));
        }
        for (int64_t layer = 0; layer < n_layers; ++layer) {
            const int64_t hidden = layouts_[index(layer, 0)].gate.k;
            for (int64_t expert = 0; expert < n_expert; ++expert)
                if (layouts_[index(layer, expert)].down.n != hidden)
                    return fail("exl3 source: experts in one layer have different hidden dimensions");
        }
        stats_.requested_bytes = std::min({total, policy.max_bytes, environment_cap()});
        startup_capacity_bytes_ = stats_.requested_bytes;
        refresh_capacity_locked();
        startup_capacity_bytes_ = stats_.admission_bytes;
        stats_.capacity_bytes = startup_capacity_bytes_;
        std::fprintf(stderr,
                     "Exl3ExpertSource: requested=%llu capacity=%llu headroom=%llu reserved=%llu "
                     "materialization_scratch=%llu\n",
                     (unsigned long long) stats_.requested_bytes, (unsigned long long) stats_.capacity_bytes,
                     (unsigned long long) policy_.headroom_bytes, (unsigned long long) policy_.reserved_bytes,
                     (unsigned long long) maximum_blob_bytes_);
        return true;
    } catch (const std::exception& e) {
        return fail(std::string("exl3 source: ") + e.what());
    }
}

bool Exl3ExpertSource::open(const std::filesystem::path& directory, int64_t n_layers, int64_t n_expert,
                            Names names, std::string& err) {
    try {
        auto archive = std::make_shared<strata::artifact::exl3::Archive>(directory);
        return open(std::move(archive), n_layers, n_expert, std::move(names), err);
    } catch (const std::exception& e) {
        err = std::string("exl3 source: ") + e.what();
        std::unique_lock<std::shared_mutex> lifecycle(lifecycle_);
        std::lock_guard<std::mutex> lock(mutex_);
        clear_locked();
        failure_ = err;
        return false;
    }
}

void Exl3ExpertSource::unlink_locked(size_t at) {
    Entry& e = entries_[at];
    if (e.previous != kNone) entries_[e.previous].next = e.next;
    else newest_ = e.next;
    if (e.next != kNone) entries_[e.next].previous = e.previous;
    else oldest_ = e.previous;
    e.previous = e.next = kNone;
}

void Exl3ExpertSource::touch_locked(size_t at) {
    if (newest_ == at) return;
    Entry& e = entries_[at];
    if (e.previous != kNone || e.next != kNone || oldest_ == at) unlink_locked(at);
    e.next = newest_;
    if (newest_ != kNone) entries_[newest_].previous = at;
    else oldest_ = at;
    newest_ = at;
}

bool Exl3ExpertSource::evict_one_locked() {
    for (size_t at = oldest_; at != kNone; at = entries_[at].previous) {
        Entry& e = entries_[at];
        if (e.state != State::ready || e.window_pin) continue;
        unlink_locked(at);
        stats_.resident_bytes -= layouts_[at].total_bytes;
        e.blob.reset(); e.state = State::empty; e.persistent = false;
        ++stats_.evictions;
        return true;
    }
    return false;
}

void Exl3ExpertSource::refresh_capacity_locked(bool evict) {
    detail::HostMemory memory;
    uint64_t room = 0;
    auto* profile = detail::active_exl3_prepare_detail();
    const auto guard_begin = profile != nullptr ? detail::Exl3DetailClock::now()
                                                : detail::Exl3DetailClock::time_point{};
    const bool available = startup_capacity_bytes_ != 0 &&
                           detail::host_available_memory(memory, capacity_files_.meminfo,
                                                         capacity_files_.self_cgroup,
                                                         capacity_files_.cgroup_root);
    if (profile != nullptr && startup_capacity_bytes_ != 0) {
        profile->guard_ns += detail::exl3_detail_ns(guard_begin, detail::Exl3DetailClock::now());
        ++profile->guard_calls;
        if (!available) ++profile->guard_failures;
    }
    if (available) {
        room = add_saturated(std::min(memory.available, memory.cgroup_limit), stats_.resident_bytes);
        room = subtract_floor(room, policy_.headroom_bytes);
        room = subtract_floor(room, policy_.reserved_bytes);
        room = subtract_floor(room, maximum_blob_bytes_);
        // Same safety margin as FileExpertSource's available-RAM clamp.
        room = subtract_floor(room, 256ull << 20);
        room = subtract_floor(room, stats_.transient_bytes);
    }
    // The budget the source may hold: the room the host had, and never more than the cache the policy
    // allows.  A room that could not be read leaves no budget at all: the source then admits nothing.
    stats_.admission_bytes = std::min(startup_capacity_bytes_, room);
    if (evict)
        while (stats_.resident_bytes > stats_.admission_bytes && evict_one_locked()) {}
}

bool Exl3ExpertSource::prewarm_cpu(const std::vector<uint8_t>& excluded, PrewarmResult& result,
                                  std::string& why) {
    std::unique_lock<std::shared_mutex> lifecycle(lifecycle_);
    std::unique_lock<std::mutex> lock(mutex_);
    result = {};
    why.clear();
    if (archive_ == nullptr || excluded.size() != entries_.size()) {
        why = "source or prewarm exclusion mask is invalid";
        return false;
    }
    uint64_t missing = 0;
    for (size_t at = 0; at < entries_.size(); ++at) {
        if (excluded[at] != 0) continue;
        const uint64_t bytes = layouts_[at].total_bytes;
        if (bytes > std::numeric_limits<uint64_t>::max() - result.requested_bytes) {
            why = "CPU prewarm payload bytes overflow";
            return false;
        }
        ++result.requested_experts;
        result.requested_bytes += bytes;
        if (entries_[at].state != State::ready || !entries_[at].persistent) missing += bytes;
    }
    if (result.requested_experts == 0) return true;
    const char* enabled = std::getenv("STRATA_EXL3_SOURCE_PREWARM");
    if (enabled != nullptr && std::strcmp(enabled, "0") == 0) {
        why = "disabled by STRATA_EXL3_SOURCE_PREWARM=0";
        return false;
    }
    if (active_layer_ >= 0) {
        why = "an active source window must finish before prewarm";
        return false;
    }
    const auto fits = [&] {
        // A refused preload must not throw away an already warm cache. Unlike
        // begin_layer, this check only samples/clamps; it performs no eviction.
        // The room is read again only when the payload that is left does not fit the budget the last
        // reading gave: the loading itself is what the budget accounts for.
        if (missing > admission_headroom_locked() && !room_reread_) {
            refresh_capacity_locked(false);
            room_reread_ = missing > admission_headroom_locked();
        }
        result.admission_bytes = stats_.admission_bytes;
        return result.requested_bytes <= stats_.admission_bytes &&
               stats_.resident_bytes <= stats_.admission_bytes &&
               missing <= admission_headroom_locked();
    };
    if (!fits()) {
        why = "all owned CPU payloads do not fit the current source cache budget";
        return false;
    }
    for (size_t at = 0; at < entries_.size(); ++at) {
        Entry& e = entries_[at];
        if (excluded[at] != 0 || e.state == State::ready) continue;
        if (result.loaded_experts != 0 && !fits()) {
            why = "host memory room decreased during CPU prewarm";
            return false;
        }
        const uint64_t bytes = layouts_[at].total_bytes;
        e.state = State::loading;
        e.persistent = true;
        stats_.resident_bytes += bytes;
        stats_.peak_resident_bytes = std::max(stats_.peak_resident_bytes, stats_.resident_bytes);
        std::unique_ptr<strata::artifact::exl3::ExpertBlob> loaded;
        std::string err;
        lock.unlock();
        const bool ok = materialize(at, loaded, err, false);
        lock.lock();
        if (!ok) {
            stats_.resident_bytes -= bytes;
            e.state = State::empty;
            e.persistent = false;
            ready_.notify_all();
            why = "CPU prewarm stopped at expert " + std::to_string(at) + ": " + err;
            return false;
        }
        e.blob = std::move(loaded);
        e.state = State::ready;
        touch_locked(at);
        missing -= bytes;
        ++result.loaded_experts;
        result.loaded_bytes += bytes;
        ready_.notify_all();
    }
    return true;
}

void Exl3ExpertSource::release_window_locked() {
    for (const size_t at : active_indices_) {
        Entry& e = entries_[at];
        e.window_pin = e.counted_window = e.failed_window = false;
        if (!e.persistent) { e.blob.reset(); e.state = State::empty; }
    }
    active_indices_.clear();
    stats_.transient_bytes = 0;
    active_layer_ = -1;
}

bool Exl3ExpertSource::materialize(size_t at, std::unique_ptr<strata::artifact::exl3::ExpertBlob>& blob,
                                 std::string& err, bool copy) {
    const auto t0 = std::chrono::steady_clock::now();
    bool ok = false;
    try {
        blob = std::make_unique<strata::artifact::exl3::ExpertBlob>();
        // The host cache keeps the CPU block order; a copy for the device keeps
        // the format's own order, which is the one the device kernels read.
        const auto order = (copy || !cpu_block_order_enabled()) ? strata::kernels::exl3::TileOrder::canonical
                                                                 : strata::kernels::exl3::TileOrder::cpu_block;
        ok = strata::artifact::exl3::materialize_expert(*archive_, names_[at], *blob, err, 8, order);
        if (ok && blob->layout.total_bytes != layouts_[at].total_bytes) {
            err = "materialized blob layout changed after metadata discovery";
            ok = false;
        }
    } catch (const std::exception& e) { err = e.what(); }
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.materializations;
    if (copy) ++stats_.copy_materializations;
    stats_.materialization_ns = add_saturated(stats_.materialization_ns, static_cast<uint64_t>(ns));
    if (ok) {
        stats_.payload_read_bytes = add_saturated(stats_.payload_read_bytes, payload_bytes_[at]);
        stats_.materialized_bytes = add_saturated(stats_.materialized_bytes, layouts_[at].total_bytes);
    }
    return ok;
}

bool Exl3ExpertSource::ensure_materialized_locked(size_t at, std::unique_lock<std::mutex>& lock) {
    Entry& e = entries_[at];
    if (!e.window_pin) {
        // The active vector was reserved to this layer's expert count at open.
        active_indices_.push_back(at);
        e.window_pin = true;
    }
    ready_.wait(lock, [&] { return e.state != State::loading; });
    if (e.failed_window) return false;
    if (e.state == State::ready) {
        if (!e.counted_window) { ++stats_.hits; e.counted_window = true; }
        if (e.persistent) touch_locked(at);
        return true;
    }
    if (!e.counted_window) { ++stats_.misses; e.counted_window = true; }
    const uint64_t bytes = layouts_[at].total_bytes;
    // The guard: the host's RAM and cgroup capacity are read again when this blob would grow the
    // resident set past the budget the last reading gave.  A budget of zero admits nothing at all, so
    // there is nothing to read for; a window that serves its requests from the resident blobs admits
    // nothing, and reads nothing.
    if (bytes > admission_headroom_locked() && stats_.admission_bytes > 0 && !room_reread_) {
        refresh_capacity_locked();
        room_reread_ = bytes > admission_headroom_locked();
    }
    while (bytes <= stats_.admission_bytes && stats_.resident_bytes > stats_.admission_bytes - bytes &&
           evict_one_locked()) {}
    e.persistent = bytes <= admission_headroom_locked() && stats_.resident_bytes <= stats_.admission_bytes - bytes;
    if (e.persistent) {
        stats_.resident_bytes += bytes;
        stats_.peak_resident_bytes = std::max(stats_.peak_resident_bytes, stats_.resident_bytes);
    } else ++stats_.fallbacks;
    e.state = State::loading;
    std::unique_ptr<strata::artifact::exl3::ExpertBlob> loaded;
    std::string err;
    lock.unlock();
    bool ok = materialize(at, loaded, err, false);
    lock.lock();
    if (!ok) {
        // Allocation failures inside materialize_expert are reported as an
        // error string. Retry once as transient after dropping unpinned cache
        // entries; no format error is accepted and no retry loop is possible.
        bool retry = e.persistent;
        if (e.persistent) {
            stats_.resident_bytes -= bytes;
            e.persistent = false;
            ++stats_.fallbacks;
        }
        loaded.reset();
        while (evict_one_locked()) retry = true;
        if (retry) {
            lock.unlock();
            ok = materialize(at, loaded, err, false);
            lock.lock();
        }
    }
    if (!ok) {
        e.state = State::empty;
        e.persistent = false;
        e.failed_window = true;
        failure_ = "exl3 source: materializing expert " + std::to_string(at) + ": " + err;
        ready_.notify_all();
        return false;
    }
    e.blob = std::move(loaded);
    e.state = State::ready;
    if (e.persistent) touch_locked(at);
    else stats_.transient_bytes = add_saturated(stats_.transient_bytes, bytes);
    ready_.notify_all();
    return true;
}

void Exl3ExpertSource::begin_layer(int64_t layer, const int32_t* ids, int64_t k) {
    std::unique_lock<std::shared_mutex> lifecycle(lifecycle_);
    std::unique_lock<std::mutex> lock(mutex_);
    failure_.clear();
    release_window_locked();
    if (!valid_pair(layer, 0) || (ids == nullptr && k != 0) || k < 0) {
        failure_ = "exl3 source: invalid begin_layer request";
        return;
    }
    for (int64_t i = 0; i < k; ++i)
        if (ids[i] < 0 || ids[i] >= n_expert_) {
            failure_ = "exl3 source: routed expert id is out of range";
            return;
        }
    // Protect every requested hit before the first admission can evict a tail.
    for (int64_t i = 0; i < k; ++i) {
        const size_t at = index(layer, ids[i]);
        if (entries_[at].window_pin) continue;
        entries_[at].window_pin = true;
        active_indices_.push_back(at);
    }
    // The guard: the host's RAM and cgroup capacity.  The layers are visited in order, so a layer that is
    // not the successor of the last one starts a new pass over them: the room is read again there, and
    // the blobs the source admits in between are held to the budget that reading gave.
    if (!layer_known_ || layer != last_layer_ + 1) {
        refresh_capacity_locked();
        room_reread_ = false;
    }
    last_layer_ = layer;
    layer_known_ = true;
    active_layer_ = layer;
    for (const size_t at : active_indices_)
        if (!ensure_materialized_locked(at, lock)) {
            release_window_locked();
            return;
        }
}

const uint8_t* Exl3ExpertSource::blob(int64_t layer, int64_t expert) {
    std::shared_lock<std::shared_mutex> lifecycle(lifecycle_);
    std::unique_lock<std::mutex> lock(mutex_);
    if (active_layer_ != layer) {
        failure_ = "exl3 source: begin_layer must precede blob requests for a new layer";
        return nullptr;
    }
    if (!valid_pair(layer, expert)) { failure_ = "exl3 source: layer or expert is out of range"; return nullptr; }
    const size_t at = index(layer, expert);
    if (!ensure_materialized_locked(at, lock)) return nullptr;
    ++reads_;
    return entries_[at].blob->data();
}

bool Exl3ExpertSource::transient(int64_t layer, int64_t expert) const {
    std::shared_lock<std::shared_mutex> lifecycle(lifecycle_);
    return valid_pair(layer, expert);
}

strata::kernels::exl3::TileOrder Exl3ExpertSource::exl3_tile_order(int64_t layer, int64_t expert) const {
    std::shared_lock<std::shared_mutex> lifecycle(lifecycle_);
    if (!valid_pair(layer, expert)) return strata::kernels::exl3::TileOrder::canonical;
    const auto& layout = layouts_[index(layer, expert)];
    if (!cpu_block_order_enabled()) return strata::kernels::exl3::TileOrder::canonical;
    return strata::kernels::exl3::cpu_tile_order_applies(layout) ? strata::kernels::exl3::TileOrder::cpu_block
                                                                 : strata::kernels::exl3::TileOrder::canonical;
}

bool Exl3ExpertSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    std::shared_lock<std::shared_mutex> lifecycle(lifecycle_);
    if (dst == nullptr || !valid_pair(layer, expert)) return false;
    const size_t at = index(layer, expert);
    {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return entries_[at].state != State::loading; });
        Entry& e = entries_[at];
        if (e.state == State::ready) {
            copy_blob_canonical(*e.blob, dst);
            if (e.persistent) touch_locked(at);
            ++stats_.copy_hits;
            return true;
        }
    }
    // Startup GPU preloads must not populate a second host copy of GPU hits.
    std::unique_ptr<strata::artifact::exl3::ExpertBlob> local;
    std::string err;
    if (!materialize(at, local, err, true)) return false;
    std::memcpy(dst, local->data(), local->bytes());
    return true;
}

const strata::kernels::exl3::ExpertBlobLayout* Exl3ExpertSource::exl3_layout(int64_t layer, int64_t expert) const {
    std::shared_lock<std::shared_mutex> lifecycle(lifecycle_);
    if (!valid_pair(layer, expert)) return nullptr;
    return &layouts_[index(layer, expert)];
}

Exl3ExpertSource::CacheStats Exl3ExpertSource::cache_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

}  // namespace strata::core
