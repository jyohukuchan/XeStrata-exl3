// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Full-layer EXL3 Flash smoke with original token IDs and optional FP32 logits.
#include "strata/core/exl3_flash_model.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

bool number(const std::string& text, int64_t& value) {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

bool ids(const std::string& text, std::vector<int32_t>& result) {
    size_t offset = 0;
    while (offset < text.size()) {
        offset = text.find_first_not_of(", \t\r\n", offset);
        if (offset == std::string::npos) break;
        const size_t end = text.find_first_of(", \t\r\n", offset);
        int64_t value = 0;
        if (!number(text.substr(offset, end == std::string::npos ? end : end - offset), value) ||
            value < 0 || value > std::numeric_limits<int32_t>::max()) return false;
        result.push_back(static_cast<int32_t>(value));
        if (end == std::string::npos) break;
        offset = end;
    }
    return !result.empty();
}

void usage() {
    std::cerr << "usage: exl3_flash_probe --exl3 DIR --tokens IDS [--stage-devices IDS] "
                 "[--stage-layers COUNTS] [--max-new N] [--max-context N] [--require-resident] [--dump-logits PATH]\n"
                 "Diagnostic full-layer AR execution; its logit copies are not a speed benchmark.\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string directory, dump_path;
    std::vector<int32_t> prompt;
    strata::core::Exl3FlashModelOptions options;
    int64_t max_new = 1, max_cells = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") { usage(); return 0; }
        if (arg == "--require-resident") { options.allow_cpu_expert_miss = false; continue; }
        if (i + 1 == argc) { usage(); return 2; }
        const std::string value = argv[++i];
        if (arg == "--exl3") directory = value;
        else if (arg == "--dump-logits") dump_path = value;
        else if (arg == "--tokens") {
            if (!prompt.empty() || !ids(value, prompt)) { usage(); return 2; }
        } else if (arg == "--stage-devices") {
            std::vector<int32_t> devices;
            if (!options.stage_devices.empty() || !ids(value, devices)) { usage(); return 2; }
            options.stage_devices.assign(devices.begin(), devices.end());
        } else if (arg == "--stage-layers") {
            std::vector<int32_t> counts;
            if (!options.stage_layers.empty() || !ids(value, counts)) { usage(); return 2; }
            options.stage_layers.assign(counts.begin(), counts.end());
        } else if (arg == "--max-new") {
            if (!number(value, max_new) || max_new < 1) { usage(); return 2; }
        } else if (arg == "--max-context") {
            if (!number(value, max_cells) || max_cells < 1) { usage(); return 2; }
        } else { std::cerr << "unknown option: " << arg << '\n'; return 2; }
    }
    if (directory.empty() || prompt.empty() ||
        prompt.size() >= static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - max_new)) {
        usage(); return 2;
    }
    const int64_t needed = static_cast<int64_t>(prompt.size()) + max_new + 1;
    if (max_cells == 0) max_cells = needed;
    if (max_cells < needed) { std::cerr << "context is shorter than the requested run\n"; return 2; }
    options.max_cells = max_cells;
    strata::core::Exl3FlashModel model;
    std::string error;
    if (!model.open(directory, options, error) || !model.reset(error)) {
        std::cerr << "EXL3 Flash open: " << error << '\n'; return 1;
    }
    const int64_t vocab = model.model_map().geometry.vocab_size;
    if (vocab <= 0 || static_cast<uint64_t>(vocab) > std::numeric_limits<size_t>::max() / sizeof(float)) {
        std::cerr << "invalid model vocabulary\n"; return 1;
    }
    std::vector<float> logits(static_cast<size_t>(vocab));
    std::ofstream dump;
    if (!dump_path.empty()) {
        dump.open(dump_path, std::ios::binary);
        if (!dump) { std::cerr << "cannot open logit dump\n"; return 1; }
    }
    std::vector<int32_t> output;
    int32_t token = prompt.front();
    for (int64_t pos = 0; output.size() < static_cast<uint64_t>(max_new); ++pos) {
        int32_t next = -1;
        if (!model.forward(token, pos, next, logits.data(), error)) {
            std::cerr << "EXL3 Flash position " << pos << ": " << error << '\n'; return 1;
        }
        for (float value : logits) {
            if (!std::isfinite(value)) {
                std::cerr << "nonfinite logits at position " << pos << '\n'; return 1;
            }
        }
        if (next < 0 || next >= vocab) { std::cerr << "invalid next token\n"; return 1; }
        const bool generated = pos + 1 >= static_cast<int64_t>(prompt.size());
        if (generated) {
            output.push_back(next);
            if (dump.is_open()) {
                dump.write(reinterpret_cast<const char*>(logits.data()),
                           static_cast<std::streamsize>(logits.size() * sizeof(float)));
                if (!dump) { std::cerr << "logit dump write failed\n"; return 1; }
            }
        }
        token = generated ? next : prompt[static_cast<size_t>(pos + 1)];
    }
    std::cout << "{\"format\":\"xestrata-exl3-flash-probe/1\",\"layers\":" << model.geometry().n_layers
              << ",\"stages\":" << model.stage_count() << ",\"vocab\":" << vocab
              << ",\"performance_gate\":false,\"tokens\":[";
    for (size_t i = 0; i < output.size(); ++i) std::cout << (i == 0 ? "" : ",") << output[i];
    std::cout << "]}\n";
    if (!model.close(error)) { std::cerr << error << '\n'; return 1; }
    return 0;
}
