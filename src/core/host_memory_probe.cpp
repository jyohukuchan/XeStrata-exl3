// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/host_memory_probe.cpp - the host's RAM and cgroup capacity, through handles that stay open.
#include "strata/core/host_memory_probe.hpp"

#include <algorithm>
#include <charconv>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace strata::core::detail {

bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes) {
    bytes = 0;
    if (!stat.valid) return false;

    // memory.stat's inactive_file can race memory.current, so bound it to charged usage first.
    uint64_t reclaimable = std::min(stat.inactive_file, stat.current);
    reclaimable = stat.file_dirty >= reclaimable ? 0 : reclaimable - stat.file_dirty;
    reclaimable = stat.file_writeback >= reclaimable ? 0 : reclaimable - stat.file_writeback;

    // Reclaiming clean file pages reduces usage; saturating subtraction also handles a transient over-limit read.
    const uint64_t usage_after_reclaim = stat.current - reclaimable;
    bytes = usage_after_reclaim < limit ? limit - usage_after_reclaim : 0;
    return true;
}

namespace {

#if !defined(_WIN32)

void skip_memory_space(std::string_view& text) {
    const size_t at = text.find_first_not_of(" \t\n\v\f\r");
    text.remove_prefix(at == std::string_view::npos ? text.size() : at);
}

std::string_view take_memory_field(std::string_view& text) {
    skip_memory_space(text);
    const size_t end = text.find_first_of(" \t\n\v\f\r");
    const std::string_view field = text.substr(0, end);
    text.remove_prefix(field.size());
    return field;
}

bool take_memory_u64(std::string_view& text, uint64_t& value) {
    skip_memory_space(text);
    if (text.empty()) return false;
    // Match unsigned stream extraction, including its optional leading sign.
    const bool negative = text.front() == '-';
    if (negative || text.front() == '+') text.remove_prefix(1);
    if (text.empty()) return false;
    uint64_t magnitude = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), magnitude);
    if (parsed.ec != std::errc{}) return false;
    text.remove_prefix(static_cast<size_t>(parsed.ptr - text.data()));
    value = negative ? uint64_t{0} - magnitude : magnitude;
    return true;
}

// The lines of a text, without the line ends (as std::getline gives them).
class TextLines {
public:
    explicit TextLines(const std::string& text) : text_(text) {}

    bool next(std::string_view& line) {
        if (pos_ >= text_.size()) return false;
        const size_t end = text_.find('\n', pos_);
        const size_t stop = end == std::string::npos ? text_.size() : end;
        line = std::string_view(text_.data() + pos_, stop - pos_);
        pos_ = end == std::string::npos ? text_.size() : end + 1;
        return true;
    }

private:
    const std::string& text_;
    size_t pos_ = 0;
};

// The size of the buffer the files are read into.  Every file read here is far smaller: the
// biggest is /proc/meminfo, at about 1.5 KiB.
constexpr size_t kReadBuffer = 64 * 1024;

// One file the probe keeps open: the path it was opened for, and the file that path named then.
struct OpenFile {
    std::string path;
    int fd = -1;
    dev_t dev = 0;
    ino_t ino = 0;
    uint64_t stamp = 0;   // when the handle was last taken (the probe's counter)
    bool used = false;    // taken by the reading in hand
};

// The files the probe holds open, by path.
class FileSet {
public:
    // The handle for `path`: opened when it is not open yet, and reopened when the path names
    // another file now (a cgroup that was removed and recreated, a tree that was remounted) - the
    // numbers in the old handle would be the numbers of a group that is gone.  Null when the file
    // cannot be opened; `error` gets the reason.
    OpenFile* take(const std::string& path, int* error = nullptr) {
        ++clock_;
        if (error != nullptr) *error = 0;
        auto it = files_.find(path);
        if (it != files_.end()) {
            OpenFile& file = it->second;
            file.stamp = clock_;
            file.used = true;
            struct stat st{};
            if (::stat(path.c_str(), &st) == 0 && st.st_dev == file.dev && st.st_ino == file.ino) return &file;
            ::close(file.fd);
            files_.erase(it);
        }
        // The bound on the open handles: the least used one goes when the set is full.
        while (files_.size() >= HostMemoryProbe::kMaxOpenFiles) {
            auto oldest = files_.begin();
            for (auto it2 = files_.begin(); it2 != files_.end(); ++it2)
                if (it2->second.stamp < oldest->second.stamp) oldest = it2;
            ::close(oldest->second.fd);
            files_.erase(oldest);
        }
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            if (error != nullptr) *error = errno;
            return nullptr;
        }
        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            if (error != nullptr) *error = errno;
            ::close(fd);
            return nullptr;
        }
        OpenFile file;
        file.path = path;
        file.fd = fd;
        file.dev = st.st_dev;
        file.ino = st.st_ino;
        file.stamp = clock_;
        file.used = true;
        return &files_.emplace(path, std::move(file)).first->second;
    }

    /// The reading in hand begins: the handles it does not take are closed when it ends.
    void begin() {
        for (auto& entry : files_) entry.second.used = false;
    }

    /// Close the handles the reading in hand did not take.
    void close_unused() {
        for (auto it = files_.begin(); it != files_.end();) {
            if (it->second.used) {
                ++it;
                continue;
            }
            ::close(it->second.fd);
            it = files_.erase(it);
        }
    }

    size_t open_count() const { return files_.size(); }

    void close_all() {
        for (auto& entry : files_) ::close(entry.second.fd);
        files_.clear();
    }

private:
    std::map<std::string, OpenFile> files_;
    uint64_t clock_ = 0;
};

// Read the whole file through the handle.  False when the read fails.
bool read_all(int fd, std::vector<char>& buffer, size_t& length) {
    length = 0;
    while (length < buffer.size()) {
        const size_t want = buffer.size() - length;
        const ssize_t n = ::pread(fd, buffer.data() + length, want, (off_t) length);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) break;             // the end of the file
        length += (size_t) n;
        // A short read: the file ends here (these files are not pipes).
        if ((size_t) n < want) break;
    }
    return true;
}

#endif  // !_WIN32

}  // namespace

struct HostMemoryProbe::Impl {
    mutable std::mutex mutex;

#if !defined(_WIN32)
    FileSet files;
    std::vector<char> buffer = std::vector<char>(kReadBuffer);   // the buffer the files are read into

    // Read the file at `path` through the handle the probe keeps.  False when the file cannot be
    // opened or read; `error` gets the reason of a file that cannot be opened.
    bool read_file(const std::string& path, std::string& text, int* error = nullptr) {
        OpenFile* file = files.take(path, error);
        if (file == nullptr) return false;
        size_t length = 0;
        if (!read_all(file->fd, buffer, length)) return false;
        text.assign(buffer.data(), length);
        return true;
    }

    // The file at `path` is there.
    bool exists(const std::string& path) const {
        struct stat st{};
        return ::stat(path.c_str(), &st) == 0;
    }

    // The directory at `path` is there.
    bool is_directory(const std::filesystem::path& path) const {
        struct stat st{};
        return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
    }

    // The cgroup v2 usage counters of the group at `path`, read through the handle the probe keeps.
    bool read_memory_stat(const std::filesystem::path& path, uint64_t current, CgroupMemoryStat& stat) {
        std::string text;
        if (!read_file((path / "memory.stat").string(), text)) return false;

        bool inactive_file = false, file_dirty = false, file_writeback = false;
        TextLines lines(text);
        std::string_view line;
        while (lines.next(line)) {
            std::string_view fields = line;
            const std::string_view key = take_memory_field(fields);
            uint64_t value = 0;
            // Unknown counters still need valid syntax, preserving the existing
            // fail-closed behavior without allocating a stream for every line.
            if (key.empty() || !take_memory_u64(fields, value)) return false;
            skip_memory_space(fields);
            if (!fields.empty()) return false;

            if (key == "inactive_file") {
                if (inactive_file) return false;
                inactive_file = true;
                stat.inactive_file = value;
            } else if (key == "file_dirty") {
                if (file_dirty) return false;
                file_dirty = true;
                stat.file_dirty = value;
            } else if (key == "file_writeback") {
                if (file_writeback) return false;
                file_writeback = true;
                stat.file_writeback = value;
            }
        }
        if (!inactive_file || !file_dirty || !file_writeback) return false;
        stat.current = current;
        stat.valid = true;
        return true;
    }
#endif  // !_WIN32

    bool sample(HostMemory& m, const std::string& meminfo, const std::string& self_cgroup,
                const std::string& cgroup_root);
};

#if !defined(_WIN32)

bool HostMemoryProbe::Impl::sample(HostMemory& m, const std::string& meminfo, const std::string& self_cgroup,
                                    const std::string& cgroup_root) {
    files.begin();
    std::string text;
    // MemAvailable includes reclaimable page cache, unlike _SC_AVPHYS_PAGES.
    if (!read_file(meminfo, text)) return false;
    uint64_t bytes = 0;
    {
        TextLines lines(text);
        std::string_view line;
        while (lines.next(line)) {
            std::string_view fields = line;
            if (take_memory_field(fields) != "MemAvailable:") continue;
            uint64_t value = 0;
            if (take_memory_u64(fields, value) && take_memory_field(fields) == "kB" &&
                value <= std::numeric_limits<uint64_t>::max() / 1024) bytes = value * 1024;
        }
    }
    if (bytes == 0) return false;
    // Account for the tightest cgroup ancestor limit when its normal mount is visible.
    // This is a point-in-time guard, not a reservation against concurrent allocations.
    if (!read_file(self_cgroup, text)) return false;
    const std::filesystem::path root(cgroup_root);
    bool v2 = false;
    std::string v1_path;   // the memory controller's group (cgroup v1), when there is no v2 line
    {
        TextLines lines(text);
        std::string_view line;
        while (lines.next(line)) {
            if (line.rfind("0::/", 0) != 0) {
                // v1: "N:controller[,controller]:/path"
                const size_t c1 = line.find(':'), c2 = c1 == std::string_view::npos ? c1 : line.find(':', c1 + 1);
                if (c2 == std::string_view::npos) continue;
                std::istringstream ctl(std::string(line.substr(c1 + 1, c2 - c1 - 1)));
                for (std::string c; std::getline(ctl, c, ',');)
                    if (c == "memory") v1_path = std::string(line.substr(c2 + 1));
                continue;
            }
            auto path = (root / std::string(line.substr(4))).lexically_normal();
            if (path.string().rfind(root.string(), 0) != 0 || !is_directory(path)) return false;
            v2 = true;
            while (path.string().rfind(root.string(), 0) == 0) {
                const std::string limit_path = (path / "memory.max").string();
                const std::string current_path = (path / "memory.current").string();
                std::string limit_text, current_text;
                int limit_error = 0;
                uint64_t current = 0;
                const bool limit_read = read_file(limit_path, limit_text, &limit_error);
                const bool current_read = read_file(current_path, current_text);
                std::string_view limit_view = limit_text;
                std::string_view current_view = current_text;
                const bool current_ok = current_read && take_memory_u64(current_view, current);
                const std::string_view limit = limit_read ? take_memory_field(limit_view) : std::string_view();
                const bool readable = !limit.empty() && current_ok;
                // The host's root cgroup has no memory.max; ordinary child groups must expose their limits.
                const bool root_without_limit =
                    path == root && limit_error == ENOENT && exists((path / "cgroup.controllers").string());
                if (!readable && !root_without_limit) return false;
                if (readable && limit != "max") {
                    const std::string cap_text(limit);
                    size_t consumed = 0;
                    uint64_t cap = 0;
                    try {
                        cap = std::stoull(cap_text, &consumed);
                    } catch (...) {
                        return false;
                    }
                    if (consumed != cap_text.size()) return false;
                    CgroupMemoryStat stat;
                    if (!read_memory_stat(path, current, stat)) return false;
                    uint64_t cgroup_available = 0;
                    if (!cgroup_available_bytes(cap, stat, cgroup_available)) return false;
                    bytes = std::min(bytes, cgroup_available);
                    m.cgroup_limit = std::min(m.cgroup_limit, cap);
                }
                if (path == root) break;
                path = path.parent_path();
            }
        }
    }
    if (!v2 && !v1_path.empty()) {
        // #633: cgroup v1 (older Docker hosts, RHEL 7/8): the memory controller's group and its ancestors.  An
        // unlimited group says a number near 2^63 (rounded to its page size); a group whose files are not
        // visible (no mount in this namespace) is skipped - MemAvailable alone, as with no cgroup at all.
        const std::filesystem::path mroot = root / "memory";
        auto path = (mroot / v1_path.substr(v1_path.rfind('/', 0) == 0 ? 1 : 0)).lexically_normal();
        while (path.string().rfind(mroot.string(), 0) == 0) {
            std::string limit_text, usage_text;
            uint64_t cap = 0, usage = 0;
            const bool limit_read = read_file((path / "memory.limit_in_bytes").string(), limit_text);
            const bool usage_read = read_file((path / "memory.usage_in_bytes").string(), usage_text);
            std::string_view limit_view = limit_text;
            std::string_view usage_view = usage_text;
            const bool cap_ok = limit_read && take_memory_u64(limit_view, cap);
            const bool usage_ok = usage_read && take_memory_u64(usage_view, usage);
            if (cap_ok && usage_ok && cap < (1ull << 62)) {
                bytes = std::min(bytes, usage < cap ? cap - usage : 0);
                m.cgroup_limit = std::min(m.cgroup_limit, cap);
            }
            if (path == mroot) break;
            path = path.parent_path();
        }
    }
    m.available = bytes;
    return true;
}

#else  // _WIN32

bool HostMemoryProbe::Impl::sample(HostMemory& m, const std::string&, const std::string&, const std::string&) {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return false;
    m.available = (uint64_t) status.ullAvailPhys;
    return m.available > 0;
}

#endif  // _WIN32

HostMemoryProbe::HostMemoryProbe() : impl_(std::make_unique<Impl>()) {}
HostMemoryProbe::~HostMemoryProbe() = default;

bool HostMemoryProbe::read(HostMemory& m, const std::string& meminfo, const std::string& self_cgroup,
                           const std::string& cgroup_root) {
    m = HostMemory{};
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const bool ok = impl_->sample(m, meminfo, self_cgroup, cgroup_root);
#if !defined(_WIN32)
    // The handles the reading did not take are of no use any more.
    impl_->files.close_unused();
#endif
    return ok;
}

void HostMemoryProbe::close() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#if !defined(_WIN32)
    impl_->files.close_all();
#endif
}

size_t HostMemoryProbe::open_files() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#if !defined(_WIN32)
    return impl_->files.open_count();
#else
    return 0;
#endif
}

bool host_available_memory(HostMemory& m, const std::string& meminfo, const std::string& self_cgroup,
                           const std::string& cgroup_root) {
    // The probe of the process: its handles stay open between the readings, and its values are read
    // again on every call.
    static HostMemoryProbe probe;
    return probe.read(m, meminfo, self_cgroup, cgroup_root);
}

}  // namespace strata::core::detail
