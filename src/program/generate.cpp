// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 command-line dispatch and process failure reporting.
#include "strata/core/gpu.hpp"
#include "strata/core/progress.hpp"
#include "strata/core/runtime.hpp"
#include "strata/program/exl3_flash_generate.hpp"
#include "strata/program/plain_generate.hpp"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <thread>

#if defined(_WIN32)
#include <io.h>
#else
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#endif

namespace {

void usage() {
    std::fprintf(stderr,
                 "strata --exl3 DIR --tokens \"1,2,3\" [options]\n"
                 "strata --exl3 DIR --tokens-file PATH [options]\n\n"
                 "  --exl3 DIR          EXL3 safetensors checkpoint\n"
                 "  --metadata-gguf P   metadata bootstrap (required for Qwen3.5)\n"
                 "  --max-new N         generated token count\n"
                 "  --max-context N     context capacity\n"
                 "  --pool-workers N    CPU expert workers\n"
                 "  --version, -V       print the engine version\n\n"
                 "Use --exl3 DIR --help for the selected model's options.\n");
}

// Report an uncaught exception without unwinding through a GPU that may no longer answer.
[[noreturn]] void report_and_end() {
    std::string report = "an unknown exception";
    if (const std::exception_ptr ep = std::current_exception()) {
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            report = strata::gpu::failure_report(e);
        } catch (...) {
            report = "an unknown exception";
        }
    } else {
        report = "std::terminate without an exception";
    }
    const strata::core::Progress& p = strata::core::progress();
    std::fprintf(stderr, "strata: FATAL while %s (detail %lld, prompt chunk at %lld): %s\n", p.where.load(),
                 (long long) p.detail.load(), (long long) p.chunk.load(), report.c_str());
    std::printf("ERR the engine stopped while %s (detail %lld, prompt chunk at %lld): %s\n", p.where.load(),
                 (long long) p.detail.load(), (long long) p.chunk.load(), report.c_str());
    std::fflush(stdout);
    std::fflush(stderr);
    strata::core::release_gpu_waits(stderr);
    std::_Exit(3);
}

const char* fatal_signal_name(int sig) {
    switch (sig) {
        case SIGSEGV: return "SIGSEGV (a bad memory access)";
#ifdef SIGBUS
        case SIGBUS: return "SIGBUS (a bad memory access)";
#endif
        case SIGILL: return "SIGILL (an instruction this CPU does not have)";
        case SIGFPE: return "SIGFPE (an arithmetic fault)";
        case SIGABRT: return "SIGABRT (an abort)";
        default: return "a fatal signal";
    }
}

// Only write(2) in a signal handler: a failed engine may already hold a library lock.
void fatal_signal(int sig) {
    const char* where = strata::core::progress().where.load(std::memory_order_relaxed);
    const char* name = fatal_signal_name(sig);
#if defined(_WIN32)
    auto put = [](int fd, const char* a) { if (a != nullptr) (void) !_write(fd, a, (unsigned) std::strlen(a)); };
#else
    auto put = [](int fd, const char* a) { if (a != nullptr) (void) !write(fd, a, std::strlen(a)); };
#endif
    put(2, "strata: FATAL signal ");
    put(2, name);
    put(2, " while ");
    put(2, where);
    put(2, "\n");
    put(1, "ERR the engine crashed on ");
    put(1, name);
    put(1, " while ");
    put(1, where);
    put(1, "\n");
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

void install_fatal_signals() {
#if defined(_WIN32)
    for (const int sig : {SIGSEGV, SIGILL, SIGFPE, SIGABRT}) (void) std::signal(sig, fatal_signal);
#else
    static char alt[64 * 1024];
    stack_t st{};
    st.ss_sp = alt;
    st.ss_size = sizeof alt;
    (void) sigaltstack(&st, nullptr);
    struct sigaction sa{};
    sa.sa_handler = fatal_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_ONSTACK | SA_RESETHAND;
    for (const int sig : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT}) (void) sigaction(sig, &sa, nullptr);
#endif
}

// Drivers can replace stop-signal handlers while compiling kernels. Drain the queues on a dedicated thread.
void route_stop_signals() {
#if !defined(_WIN32)
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGTERM);
    sigaddset(&set, SIGINT);
    if (pthread_sigmask(SIG_BLOCK, &set, nullptr) != 0) return;
    std::thread([set] {
        int sig = 0;
        while (sigwait(&set, &sig) != 0) {}
        std::thread([sig] {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            _exit(128 + sig);
        }).detach();
        strata::core::Runtime::finish_all();
        _exit(128 + sig);
    }).detach();
#endif
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::set_terminate(report_and_end);
    install_fatal_signals();
    strata::core::progress_at("starting: reading the options and opening the GPUs");
    if (argc > 1 && std::strcmp(argv[1], "generate") == 0) { --argc; ++argv; }
    if (argc == 2 && (std::strcmp(argv[1], "--version") == 0 || std::strcmp(argv[1], "-V") == 0)) {
        std::printf("strata %s\n", STRATA_VERSION);
        return 0;
    }
    if (argc == 1 || (argc == 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0))) {
        usage();
        return argc == 1 ? 2 : 0;
    }
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
#if defined(_WIN32)
        _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#else
        setenv("CUDA_MODULE_LOADING", "EAGER", 0);
#endif
    }
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], "--gpu") == 0) {
#if defined(_WIN32)
            _putenv_s("STRATA_GPU_PCI", argv[i + 1]);
#else
            setenv("STRATA_GPU_PCI", argv[i + 1], 1);
#endif
            break;
        }
    route_stop_signals();
    if (const int r = strata::program::exl3_flash_main(argc, argv); r >= 0) return r;
    if (const int r = strata::program::plain_main(argc, argv); r >= 0) return r;
    std::fprintf(stderr, "strata: an EXL3 checkpoint with a supported model type is required (--exl3 DIR)\n");
    usage();
    return 2;
}
