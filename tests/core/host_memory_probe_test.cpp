// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// tests/core/host_memory_probe_test.cpp - the host's RAM and cgroup capacity, through handles that stay open.
#include "strata/core/host_memory_probe.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct TempDirectory {
    fs::path path;

    TempDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("strata-host-memory-probe-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }

    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

// The fixture: a meminfo, a /proc/self/cgroup and the cgroup tree under it.
struct Fixture {
    TempDirectory dir;

    void put(const std::string& file, const std::string& text) {
        const fs::path path = dir.path / file;
        fs::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::trunc);
        out << text;
        require(bool(out), "could not write the host-memory fixture");
    }

    void put_bytes(const std::string& file, uint64_t bytes) { put(file, std::to_string(bytes) + "\n"); }

    // The file again as ANOTHER file (a cgroup removed and recreated): the probe must follow it.
    void replace(const std::string& file, const std::string& text) {
        std::error_code ignored;
        fs::remove(dir.path / file, ignored);
        put(file, text);
    }

    void remove(const std::string& file) {
        std::error_code ignored;
        fs::remove(dir.path / file, ignored);
    }

    std::string meminfo() const { return (dir.path / "meminfo").string(); }
    std::string self_cgroup() const { return (dir.path / "self_cgroup").string(); }
    std::string cgroup_root() const { return (dir.path / "cgroup").string(); }
};

#if defined(__linux__)

constexpr uint64_t MiB = 1ull << 20;

// The fixture of the cgroup tree: the root has no memory.max (as the host's root cgroup), the group
// /a/b has a limit, and /a/c has one too.
void build_tree(Fixture& fixture, const std::string& stat) {
    fixture.put("cgroup/cgroup.controllers", "memory\n");
    fixture.put("cgroup/a/memory.max", "max\n");
    fixture.put_bytes("cgroup/a/memory.current", 0);
    fixture.put_bytes("cgroup/a/b/memory.max", 64 * MiB);
    fixture.put_bytes("cgroup/a/b/memory.current", 48 * MiB);
    fixture.put("cgroup/a/b/memory.stat", stat);
    fixture.put_bytes("cgroup/a/c/memory.max", 16 * MiB);
    fixture.put_bytes("cgroup/a/c/memory.current", 4 * MiB);
    fixture.put("cgroup/a/c/memory.stat", "anon 0\ninactive_file 0\nfile_dirty 0\nfile_writeback 0\n");
}

// The values are read again on every call: a cached number would be a stale one.
void test_values_are_read_again() {
    using namespace strata::core::detail;
    Fixture fixture;
    const std::string stat = "anon 123\ninactive_file 4194304\nfile_dirty 0\nfile_writeback 0\n";
    fixture.put("meminfo", "MemTotal: 999999 kB\n  MemAvailable:\t65536 kB\n");
    fixture.put("self_cgroup", "0::/a/b\n");
    build_tree(fixture, stat);

    HostMemoryProbe probe;
    HostMemory memory;
    require(probe.read(memory, fixture.meminfo(), fixture.self_cgroup(), fixture.cgroup_root()),
            "the host-memory fixture was refused");
    require(memory.available == 20 * MiB, "the room under the cgroup limit was not the tightest one");
    require(memory.cgroup_limit == 64 * MiB, "the tightest cgroup limit was not reported");

    // the limit, the usage and MemAvailable all change between two calls
    fixture.put("meminfo", "MemTotal: 999999 kB\nMemAvailable: 8192 kB\n");
    fixture.put_bytes("cgroup/a/b/memory.max", 32 * MiB);
    fixture.put_bytes("cgroup/a/b/memory.current", 30 * MiB);
    require(probe.read(memory, fixture.meminfo(), fixture.self_cgroup(), fixture.cgroup_root()),
            "the changed fixture was refused");
    require(memory.available == 6 * MiB, "the changed limit, usage and MemAvailable were not read again");
    require(memory.cgroup_limit == 32 * MiB, "the changed limit was not reported");

    // the free function of the process reads the same files the same way
    require(host_available_memory(memory, fixture.meminfo(), fixture.self_cgroup(), fixture.cgroup_root()),
            "the host-memory function refused the fixture");
    require(memory.available == 6 * MiB && memory.cgroup_limit == 32 * MiB,
            "the host-memory function did not read the changed values");
}

// A group that moved is followed: the group path comes from /proc/self/cgroup, read on every call.
void test_a_group_that_moved() {
    using namespace strata::core::detail;
    Fixture fixture;
    const std::string stat = "anon 123\ninactive_file 4194304\nfile_dirty 0\nfile_writeback 0\n";
    fixture.put("meminfo", "MemTotal: 999999 kB\nMemAvailable: 65536 kB\n");
    fixture.put("self_cgroup", "0::/a/b\n");
    build_tree(fixture, stat);

    HostMemoryProbe probe;
    HostMemory memory;
    require(probe.read(memory, fixture.meminfo(), fixture.self_cgroup(), fixture.cgroup_root()),
            "the host-memory fixture was refused");
    require(memory.cgroup_limit == 64 * MiB, "the first group was not read");
    const size_t open_before = probe.open_files();

    fixture.put("self_cgroup", "0::/a/c\n");
    require(probe.read(memory, fixture.meminfo(), fixture.self_cgroup(), fixture.cgroup_root()),
            "the moved group was refused");
    require(memory.cgroup_limit == 16 * MiB, "the moved group was not followed");
    require(memory.available == 12 * MiB, "the moved group's room was not read");
    // the handles of the group that was left are closed: their number does not grow
    require(probe.open_files() <= open_before, "the handles of the group that was left stayed open");
}

// A handle is reopened when the path names another file: a cgroup removed and recreated.
void test_a_replaced_file_is_followed() {
    using namespace strata::core::detail;
    Fixture fixture;
    const std::string stat = "anon 123\ninactive_file 4194304\nfile_dirty 0\nfile_writeback 0\n";
    fixture.put("meminfo", "MemTotal: 999999 kB\nMemAvailable: 65536 kB\n");
    fixture.put("self_cgroup", "0::/a/b\n");
    build_tree(fixture, stat);

    HostMemoryProbe probe;
    HostMemory memory;
    require(probe.read(memory, fixture.meminfo(), fixture.self_cgroup(), fixture.cgroup_root()),
            "the host-memory fixture was refused");
    require(memory.cgroup_limit == 64 * MiB, "the first limit was not read");

    fixture.replace("cgroup/a/b/memory.max", std::to_string(8 * MiB) + "\n");
    fixture.put_bytes("cgroup/a/b/memory.current", 4 * MiB);
    require(probe.read(memory, fixture.meminfo(), fixture.self_cgroup(), fixture.cgroup_root()),
            "the replaced file was refused");
    require(memory.cgroup_limit == 8 * MiB, "the replaced file was not read again");
    require(memory.available == 8 * MiB, "the replaced limit was not applied");
}

// A read that fails makes the reading fail: the caller fails closed, as for a missing file.
void test_a_failing_read_fails_closed() {
    using namespace strata::core::detail;
    Fixture fixture;
    const std::string stat = "anon 123\ninactive_file 4194304\nfile_dirty 0\nfile_writeback 0\n";
    fixture.put("meminfo", "MemTotal: 999999 kB\nMemAvailable: 65536 kB\n");
    fixture.put("self_cgroup", "0::/a/b\n");
    build_tree(fixture, stat);

    HostMemoryProbe probe;
    HostMemory memory;
    require(probe.read(memory, fixture.meminfo(), fixture.self_cgroup(), fixture.cgroup_root()),
            "the host-memory fixture was refused");

    // the usage counters of a group with a limit are needed
    fixture.remove("cgroup/a/b/memory.stat");
    require(!probe.read(memory, fixture.meminfo(), fixture.self_cgroup(), fixture.cgroup_root()),
            "a missing memory.stat did not fail closed");
    fixture.put("cgroup/a/b/memory.stat", "inactive_file 4194304\nfile_dirty 0\n");
    require(!probe.read(memory, fixture.meminfo(), fixture.self_cgroup(), fixture.cgroup_root()),
            "a malformed memory.stat did not fail closed");

    // the group's limit is needed
    fixture.remove("cgroup/a/b/memory.max");
    require(!probe.read(memory, fixture.meminfo(), fixture.self_cgroup(), fixture.cgroup_root()),
            "a missing limit did not fail closed");
    fixture.put_bytes("cgroup/a/b/memory.max", 64 * MiB);
    fixture.put("cgroup/a/b/memory.stat", stat);

    // MemAvailable is needed
    fixture.remove("meminfo");
    require(!probe.read(memory, fixture.meminfo(), fixture.self_cgroup(), fixture.cgroup_root()),
            "a missing meminfo did not fail closed");
    fixture.put("meminfo", "MemTotal: 999999 kB\n");

    // the group the process is in is needed
    fixture.remove("self_cgroup");
    require(!probe.read(memory, fixture.meminfo(), fixture.self_cgroup(), fixture.cgroup_root()),
            "a missing cgroup list did not fail closed");
}

// The handles stay open between the readings, and they are few.
void test_the_handles_stay_open() {
    using namespace strata::core::detail;
    Fixture fixture;
    const std::string stat = "anon 123\ninactive_file 4194304\nfile_dirty 0\nfile_writeback 0\n";
    fixture.put("meminfo", "MemTotal: 999999 kB\nMemAvailable: 65536 kB\n");
    fixture.put("self_cgroup", "0::/a/b\n");
    build_tree(fixture, stat);

    HostMemoryProbe probe;
    HostMemory memory;
    require(probe.read(memory, fixture.meminfo(), fixture.self_cgroup(), fixture.cgroup_root()),
            "the host-memory fixture was refused");
    const size_t open_after = probe.open_files();
    require(open_after > 0, "no handle was kept open between the readings");
    require(open_after <= HostMemoryProbe::kMaxOpenFiles, "more handles than the bound");

    // the handles are the same ones: a second reading does not open more
    require(probe.read(memory, fixture.meminfo(), fixture.self_cgroup(), fixture.cgroup_root()),
            "the second reading was refused");
    require(probe.open_files() == open_after, "the second reading opened more handles");

    // close() gives every handle back
    probe.close();
    require(probe.open_files() == 0, "the handles were not closed");
    require(probe.read(memory, fixture.meminfo(), fixture.self_cgroup(), fixture.cgroup_root()),
            "the reading after close() was refused");
    require(probe.open_files() == open_after, "the reading after close() did not reopen the handles");
}

#endif  // __linux__

}  // namespace

int main() {
    try {
#if defined(__linux__)
        test_values_are_read_again();
        test_a_group_that_moved();
        test_a_replaced_file_is_followed();
        test_a_failing_read_fails_closed();
        test_the_handles_stay_open();
#endif
        std::cout << "host_memory_probe_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "host_memory_probe_test: " << error.what() << '\n';
        return 1;
    }
}
