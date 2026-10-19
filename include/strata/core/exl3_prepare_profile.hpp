// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <functional>
#include <thread>

namespace strata::core::detail {

using Exl3DetailClock = std::chrono::steady_clock;
inline uint64_t exl3_detail_ns(Exl3DetailClock::time_point begin, Exl3DetailClock::time_point end) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

// Calling-host-thread counters only. No worker/weight-loop instrumentation or existing ABI layout changes.
struct Exl3PrepareDetail {
    struct Phase {
        uint64_t calls = 0, tasks_sum = 0, tasks_min = 0, tasks_max = 0, under_workers = 0;
        uint64_t wall_ns = 0, build_ns = 0, before_park_ns = 0, publish_done_ns = 0, after_park_ns = 0;
    } phase[4];
    unsigned active_prepare = 0;
    uint64_t windows = 0, failed_windows = 0, prepare_ns = 0, part_ns[4]{};
    uint64_t guard_calls = 0, guard_failures = 0, guard_ns = 0;
    uint64_t row_batches = 0, validation_ns = 0, miss_jobs_sum = 0, miss_jobs_min = 0, miss_jobs_max = 0;
    uint64_t row_wall_ns = 0, initial_park_ns = 0;
    uint64_t workers_min = 0, workers_max = 0, host_work_batches = 0;

    void row_batch(int jobs, int workers, bool host_works) {
        const auto j = static_cast<uint64_t>(jobs), w = static_cast<uint64_t>(workers);
        miss_jobs_min = row_batches == 0 ? j : std::min(miss_jobs_min, j);
        workers_min = row_batches == 0 ? w : std::min(workers_min, w);
        miss_jobs_max = std::max(miss_jobs_max, j); workers_max = std::max(workers_max, w);
        miss_jobs_sum += j; ++row_batches;
        if (host_works) ++host_work_batches;
    }

    ~Exl3PrepareDetail() {
        if (windows == 0 && row_batches == 0) return;
        // One bounded write keeps summaries from concurrent Flash host threads on separate lines.
        char line[8192];
        char* next = line;
        size_t left = sizeof(line);
        bool ok = append(next, left,
            "{\"kind\":\"exl3_prepare_summary\",\"version\":1,\"scope\":\"calling_host_thread\","
            "\"thread_id_hash\":%llu,\"time_unit\":\"ns\",\"windows\":%llu,\"failed_windows\":%llu,"
            "\"prepare_ns\":%llu,\"partition_ns\":%llu,\"source_begin_ns\":%llu,\"borrow_jobs_ns\":%llu,"
            "\"host_plan_ns\":%llu,\"guard_calls\":%llu,\"guard_failures\":%llu,\"guard_ns\":%llu,"
            "\"row_batches\":%llu,\"row_wall_ns\":%llu,\"validation_ns\":%llu,\"initial_park_ns\":%llu,"
            "\"miss_jobs_sum\":%llu,\"miss_jobs_min\":%llu,"
            "\"miss_jobs_max\":%llu,\"workers_min\":%llu,\"workers_max\":%llu,\"host_work_batches\":%llu,"
            "\"intdot_fallback_counts\":\"not_instrumented\",\"cpu_phases\":[",
            ull(std::hash<std::thread::id>{}(std::this_thread::get_id())), ull(windows), ull(failed_windows),
            ull(prepare_ns), ull(part_ns[0]), ull(part_ns[1]), ull(part_ns[2]), ull(part_ns[3]),
            ull(guard_calls), ull(guard_failures), ull(guard_ns), ull(row_batches), ull(row_wall_ns),
            ull(validation_ns), ull(initial_park_ns),
            ull(miss_jobs_sum), ull(miss_jobs_min), ull(miss_jobs_max), ull(workers_min), ull(workers_max),
            ull(host_work_batches));
        for (int i = 0; i < 4 && ok; ++i) {
            const Phase& p = phase[i];
            ok = append(next, left,
                "%s{\"mode\":%d,\"calls\":%llu,\"tasks_sum\":%llu,\"tasks_min\":%llu,\"tasks_max\":%llu,"
                "\"under_worker_threads\":%llu,\"wall_ns\":%llu,\"task_build_ns\":%llu,"
                "\"before_park_ns\":%llu,\"publish_through_done_ns\":%llu,\"after_park_ns\":%llu}",
                i == 0 ? "" : ",", 11 + i, ull(p.calls), ull(p.tasks_sum), ull(p.tasks_min), ull(p.tasks_max),
                ull(p.under_workers), ull(p.wall_ns), ull(p.build_ns), ull(p.before_park_ns),
                ull(p.publish_done_ns), ull(p.after_park_ns));
        }
        ok = ok && append(next, left, "%s", "]}\n");
        if (ok) (void) std::fwrite(line, 1, sizeof(line) - left, stderr);
        else std::fputs("{\"kind\":\"exl3_prepare_summary\",\"status\":\"format_error\"}\n", stderr);
    }

private:
    static unsigned long long ull(uint64_t v) { return static_cast<unsigned long long>(v); }
    template <class... Args>
    static bool append(char*& next, size_t& left, const char* format, Args... args) {
        const int n = std::snprintf(next, left, format, args...);
        if (n < 0 || static_cast<size_t>(n) >= left) return false;
        next += n; left -= static_cast<size_t>(n);
        return true;
    }
};

inline Exl3PrepareDetail* exl3_prepare_detail() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_EXL3_PREPARE_DETAIL");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    if (!enabled) return nullptr;
    thread_local Exl3PrepareDetail detail;
    return &detail;
}

inline Exl3PrepareDetail* active_exl3_prepare_detail() {
    auto* detail = exl3_prepare_detail();
    return detail != nullptr && detail->active_prepare != 0 ? detail : nullptr;
}

class Exl3PrepareSample {
public:
    Exl3PrepareSample() : detail_(exl3_prepare_detail()) {
        if (detail_ != nullptr) { ++detail_->active_prepare; begin_ = last_ = Exl3DetailClock::now(); }
    }
    ~Exl3PrepareSample() {
        if (detail_ == nullptr) return;
        detail_->prepare_ns += exl3_detail_ns(begin_, Exl3DetailClock::now());
        ++detail_->windows;
        if (!success_) ++detail_->failed_windows;
        --detail_->active_prepare;
    }
    void mark(unsigned part) {
        if (detail_ == nullptr) return;
        const auto now = Exl3DetailClock::now();
        detail_->part_ns[part] += exl3_detail_ns(last_, now);
        last_ = now;
    }
    void success() { success_ = true; }
    Exl3PrepareSample(const Exl3PrepareSample&) = delete;
    Exl3PrepareSample& operator=(const Exl3PrepareSample&) = delete;
private:
    Exl3PrepareDetail* detail_;
    Exl3DetailClock::time_point begin_{}, last_{};
    bool success_ = false;
};

}  // namespace strata::core::detail
