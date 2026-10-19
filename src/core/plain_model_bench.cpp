// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Fixed-token AR decode timing. No logits are copied to the host or written during measurement.
#include "strata/core/plain_model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Options {
    strata::core::PlainOptions model;
    std::string token_file;
    int64_t context = 512;
    int64_t steps = 128;
    int64_t warmup = 16;
    int repeats = 5;
};

void usage() {
    std::fprintf(stderr,
        "usage: plain_model_bench (--native GGUF [--expert-profile FILE] | --exl3 DIR --metadata-gguf GGUF) "
        "--tokens-file FILE [--context N] [--steps N] [--warmup N] [--repeats N] "
        "[--max-t N] [--pool-workers N]\n");
}

bool integer(const std::string& text, int64_t& out) {
    size_t end = 0;
    try { out = std::stoll(text, &end); } catch (...) { return false; }
    return end == text.size() && end != 0 && out >= 0;
}

bool parse(int argc, char** argv, Options& opt, std::string& err) {
    opt.model.prompt_chunk = 0;
    opt.model.spec = 0;
    opt.model.adapt_every = 0;
    opt.model.max_t = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string name = argv[i];
        if (name == "--help" || name == "-h") { usage(); return false; }
        if (i + 1 == argc) { err = "missing value for " + name; return false; }
        const std::string value = argv[++i];
        if (name == "--native") opt.model.gguf = value;
        else if (name == "--exl3") opt.model.exl3_dir = value;
        else if (name == "--metadata-gguf") opt.model.metadata_gguf = value;
        else if (name == "--expert-profile") opt.model.profile = value;
        else if (name == "--tokens-file") opt.token_file = value;
        else {
            int64_t n = 0;
            if (!integer(value, n)) { err = "invalid integer for " + name; return false; }
            if (name == "--context") opt.context = n;
            else if (name == "--steps") opt.steps = n;
            else if (name == "--warmup") opt.warmup = n;
            else if (name == "--repeats" || name == "--max-t" || name == "--pool-workers") {
                if (n > std::numeric_limits<int>::max()) { err = "integer too large for " + name; return false; }
                if (name == "--repeats") opt.repeats = static_cast<int>(n);
                else if (name == "--max-t") opt.model.max_t = static_cast<int>(n);
                else opt.model.pool_workers = static_cast<int>(n);
            } else { err = "unknown option: " + name; return false; }
        }
    }
    const bool exl3 = !opt.model.exl3_dir.empty();
    if (exl3 == !opt.model.gguf.empty() || opt.token_file.empty() ||
        (exl3 && (opt.model.metadata_gguf.empty() || !opt.model.profile.empty())) ||
        (!exl3 && !opt.model.metadata_gguf.empty())) {
        err = "choose one model source, its required metadata, and a token file; expert profiles require --native";
        return false;
    }
    if (opt.steps < 1 || opt.repeats < 1 || opt.model.max_t < 1 || opt.warmup > opt.context ||
        opt.context > std::numeric_limits<int64_t>::max() - opt.steps) {
        err = "invalid context, warmup, steps, repeats or max-t";
        return false;
    }
    opt.model.max_cells = (std::max)(int64_t(16), opt.context + opt.steps);
    return true;
}

bool read_tokens(const std::string& path, std::vector<int32_t>& tokens, std::string& err) {
    std::ifstream file(path);
    if (!file) { err = "cannot open token file"; return false; }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (file.bad()) { err = "reading token file failed"; return false; }
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream input(text);
    std::string item;
    while (input >> item) {
        int64_t token = 0;
        if (!integer(item, token) || token > std::numeric_limits<int32_t>::max()) {
            err = "token ID is outside the nonnegative int32 range";
            return false;
        }
        tokens.push_back(static_cast<int32_t>(token));
    }
    if (!input.eof() || tokens.empty()) { err = "token file is empty or malformed"; return false; }
    return true;
}

void statistics(const std::vector<double>& values) {
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const auto quantile = [&](double q) {
        const double at = q * static_cast<double>(sorted.size() - 1);
        const size_t lo = static_cast<size_t>(at);
        const size_t hi = (std::min)(lo + 1, sorted.size() - 1);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (at - static_cast<double>(lo));
    };
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    std::printf("\"positions\":%zu,\"mean_ms\":%.9f,\"median_ms\":%.9f,\"p90_ms\":%.9f,\"tok_s\":%.9f",
                values.size(), mean, quantile(.5), quantile(.9), 1000.0 / mean);
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    std::string err;
    if (!parse(argc, argv, opt, err)) {
        if (!err.empty()) { std::fprintf(stderr, "plain_model_bench: %s\n", err.c_str()); usage(); }
        return err.empty() ? 0 : 2;
    }
    std::vector<int32_t> tokens;
    if (!read_tokens(opt.token_file, tokens, err)) {
        std::fprintf(stderr, "plain_model_bench: %s\n", err.c_str());
        return 2;
    }
    strata::core::PlainModel model;
    if (!model.load(opt.model, err)) {
        std::fprintf(stderr, "plain_model_bench: load failed: %s\n", err.c_str());
        return 1;
    }
    for (const int32_t token : tokens) {
        if (token >= model.vocab()) { std::fprintf(stderr, "plain_model_bench: token outside model vocabulary\n"); return 2; }
    }
    std::vector<int32_t> window(static_cast<size_t>(opt.model.max_t));
    std::vector<int32_t> output(window.size());
    std::vector<std::vector<double>> runs;
    std::vector<int64_t> hit_counts, miss_counts;
    for (int repeat = 0; repeat < opt.repeats; ++repeat) {
        if (!model.reset(err)) { std::fprintf(stderr, "plain_model_bench: reset failed: %s\n", err.c_str()); return 1; }
        int64_t pos = 0;
        while (pos < opt.context) {
            const int64_t seed_end = opt.context - opt.warmup;
            const int count = pos < seed_end ? static_cast<int>((std::min)(int64_t(opt.model.max_t), seed_end - pos)) : 1;
            for (int j = 0; j < count; ++j) window[static_cast<size_t>(j)] = tokens[static_cast<size_t>(pos + j) % tokens.size()];
            if (!model.forward(window.data(), count, pos, output.data(), nullptr, err)) {
                std::fprintf(stderr, "plain_model_bench: context failed: %s\n", err.c_str()); return 1;
            }
            pos += count;
        }
        const int64_t hits = model.hits(), misses = model.misses();
        std::vector<double> times;
        for (int64_t step = 0; step < opt.steps; ++step, ++pos) {
            const int32_t token = tokens[static_cast<size_t>(pos) % tokens.size()];
            const auto start = std::chrono::steady_clock::now();
            if (!model.forward(&token, 1, pos, output.data(), nullptr, err)) {
                std::fprintf(stderr, "plain_model_bench: measured decode failed: %s\n", err.c_str()); return 1;
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (!(ms > 0.0) || !std::isfinite(ms)) { std::fprintf(stderr, "plain_model_bench: invalid time\n"); return 1; }
            if (output[0] < 0 || output[0] >= model.vocab()) {
                std::fprintf(stderr, "plain_model_bench: generated token outside model vocabulary\n"); return 1;
            }
            times.push_back(ms);
        }
        hit_counts.push_back(model.hits() - hits);
        miss_counts.push_back(model.misses() - misses);
        runs.push_back(std::move(times));
    }
    std::printf("{\"format\":\"xestrata-fixed-token-decode/1\",\"mode\":\"%s\",\"context\":%lld,"
                "\"steps\":%lld,\"warmup\":%lld,\"max_t\":%d,\"resident\":%lld,\"slots\":%lld,\"runs\":[",
                opt.model.exl3_dir.empty() ? "native" : "exl3", (long long) opt.context, (long long) opt.steps,
                (long long) opt.warmup, opt.model.max_t, (long long) model.resident(), (long long) model.slots());
    std::vector<double> all;
    for (size_t i = 0; i < runs.size(); ++i) {
        std::printf("%s{", i == 0 ? "" : ",");
        statistics(runs[i]);
        std::printf(",\"cache_hits\":%lld,\"cache_misses\":%lld}", (long long) hit_counts[i], (long long) miss_counts[i]);
        all.insert(all.end(), runs[i].begin(), runs[i].end());
    }
    std::printf("],\"summary\":{");
    statistics(all);
    std::printf("}}\n");
    return 0;
}
