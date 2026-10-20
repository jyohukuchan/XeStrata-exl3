// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The RAM this process can get, read through file handles that stay open between the readings.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace strata::core::detail {

/// Required cgroup-v2 usage counters for the conservative cache-reclaim allowance.
struct CgroupMemoryStat {
    uint64_t current = 0;
    uint64_t inactive_file = 0;
    uint64_t file_dirty = 0;
    uint64_t file_writeback = 0;
    bool valid = false;
};

/// Calculate additional bytes under a finite cgroup limit after reclaiming only clean inactive file cache.
/// Returns false when the required memory.stat counters were unavailable.
bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes);

/// #633: the RAM this process can get.  `available`: MemAvailable, lowered to the room under the
/// tightest cgroup limit; `cgroup_limit`: that tightest limit itself (v2 memory.max of the group and
/// its ancestors, or v1 memory.limit_in_bytes), ~0 when there is none - what a container can never
/// exceed.
struct HostMemory {
    uint64_t available = 0;
    uint64_t cgroup_limit = ~uint64_t{0};
};

/// The host's RAM and cgroup capacity, read through file handles that stay open between the
/// readings.
///
/// The reading is a POINT-IN-TIME guard, not a reservation against concurrent allocations: the
/// numbers must be the ones the host has NOW, so every call re-reads the files.  What the probe
/// saves is the open and the close of each of them, which is the part a per-layer check paid 40
/// times a token: the values come from the handles, the handles are read again every time.
///
/// A handle is reopened when the path names another file than the one it was opened for (a cgroup
/// that was removed and recreated, a tree that was remounted): the numbers in the old handle are
/// the numbers of a group that is gone.  A group that moved is followed: the group path comes from
/// `/proc/self/cgroup`, which is read on every call, so the tree that is read is the tree the
/// process is in NOW.
///
/// A file that cannot be read makes the reading fail: the caller fails closed, as it does for a
/// missing file.
class HostMemoryProbe {
public:
    HostMemoryProbe();
    ~HostMemoryProbe();
    HostMemoryProbe(const HostMemoryProbe&) = delete;
    HostMemoryProbe& operator=(const HostMemoryProbe&) = delete;

    /// The RAM this process can get now.  `meminfo`, `self_cgroup` and `cgroup_root` are the files
    /// to read (the defaults are the real ones).  False when the RAM cannot be determined.
    bool read(HostMemory& m, const std::string& meminfo = "/proc/meminfo",
              const std::string& self_cgroup = "/proc/self/cgroup",
              const std::string& cgroup_root = "/sys/fs/cgroup");

    /// Close every handle the probe holds.  The next reading reopens what it needs.
    void close();

    /// The handles the probe holds open.  The probe never holds more than kMaxOpenFiles.
    size_t open_files() const;

    /// The bound on the open handles: a cgroup tree is a handful of files per level, and a tree
    /// that moved leaves its old files behind until they are closed.
    static constexpr size_t kMaxOpenFiles = 64;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// #633: the RAM this process can get.  Linux reads `meminfo`, `self_cgroup` and the cgroup tree
/// under `cgroup_root` (the parameters are for tests; the defaults are the real files).  cgroup v2
/// as before (an unreadable limit of a group that has one fails); cgroup v1's memory controller
/// (`<root>/memory/<path>`: memory.limit_in_bytes - memory.usage_in_bytes); no cgroup line at all is
/// MemAvailable alone.  False when the RAM cannot be determined.
///
/// The files stay open between the calls (a probe of its own, for the life of the process), and
/// their values are read again on every call: a cached number would be a stale one.
bool host_available_memory(HostMemory& m, const std::string& meminfo = "/proc/meminfo",
                           const std::string& self_cgroup = "/proc/self/cgroup",
                           const std::string& cgroup_root = "/sys/fs/cgroup");

}  // namespace strata::core::detail
