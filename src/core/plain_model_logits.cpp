// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Forced-token logit dump for comparing PlainModel backends with one protocol.
#include "strata/core/plain_model.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;
using strata::core::PlainModel;
using strata::core::PlainOptions;

struct TraceRow {
    Json id;
    std::vector<int32_t> input_ids;
    std::vector<int32_t> response_ids;
};

struct Options {
    std::string exl3;
    std::string metadata_gguf;
    std::string native;
    std::string profile;
    std::string trace;
    std::string out_dir;
    std::vector<size_t> selected_rows;
    int max_t = 1;
    int pool_workers = 8;
    int prompt_chunk = 0;
    int spec = 0;
    int adapt_every = 0;
};

struct DumpedRow {
    size_t index = 0;
    Json id;
    size_t n_tokens = 0;
    size_t input_tokens = 0;
    size_t range_start = 0;
    size_t range_end = 0;
    std::string file;
};

bool parse_integer(const std::string& text, int64_t& value) {
    if (text.empty()) return false;
    size_t at = 0;
    try {
        value = std::stoll(text, &at, 10);
    } catch (...) {
        return false;
    }
    return at == text.size();
}

bool parse_size(const std::string& text, size_t& value) {
    int64_t parsed = 0;
    if (!parse_integer(text, parsed) || parsed < 0) return false;
    value = static_cast<size_t>(parsed);
    return static_cast<int64_t>(value) == parsed;
}

bool parse_rows(const std::string& text, std::vector<size_t>& rows, std::string& error) {
    rows.clear();
    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t comma = text.find(',', begin);
        const std::string item = text.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
        size_t index = 0;
        if (!parse_size(item, index)) {
            error = "--rows contains an invalid index: " + item;
            return false;
        }
        rows.push_back(index);
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    std::sort(rows.begin(), rows.end());
    if (std::adjacent_find(rows.begin(), rows.end()) != rows.end()) {
        error = "--rows contains a duplicate index";
        return false;
    }
    return true;
}

void usage() {
    std::cerr << "usage: plain_model_logits (--exl3 DIR --metadata-gguf GGUF | --native GGUF) "
                 "--trace TRACE.json --out-dir DIR [--rows I[,I...]] [--max-t N] [--pool-workers N] [--expert-profile FILE]\n";
}

bool parse_options(int argc, char** argv, Options& options, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char* name, std::string& destination) {
            if (i + 1 >= argc) {
                error = std::string(name) + " requires a value";
                return false;
            }
            destination = argv[++i];
            return true;
        };
        if (arg == "--help" || arg == "-h") {
            usage();
            std::exit(0);
        } else if (arg == "--exl3") {
            if (!need("--exl3", options.exl3)) return false;
        } else if (arg == "--metadata-gguf") {
            if (!need("--metadata-gguf", options.metadata_gguf)) return false;
        } else if (arg == "--native") {
            if (!need("--native", options.native)) return false;
        } else if (arg == "--expert-profile") {
            if (!need("--expert-profile", options.profile)) return false;
        } else if (arg == "--trace") {
            if (!need("--trace", options.trace)) return false;
        } else if (arg == "--out-dir") {
            if (!need("--out-dir", options.out_dir)) return false;
        } else if (arg == "--rows") {
            std::string value;
            if (!need("--rows", value) || !parse_rows(value, options.selected_rows, error)) return false;
        } else if (arg == "--max-t") {
            std::string value;
            if (!need("--max-t", value)) return false;
            int64_t parsed = 0;
            if (!parse_integer(value, parsed) || parsed < 1 || parsed > std::numeric_limits<int>::max()) {
                error = "--max-t must be a positive integer";
                return false;
            }
            options.max_t = static_cast<int>(parsed);
        } else if (arg == "--pool-workers") {
            std::string value;
            if (!need("--pool-workers", value)) return false;
            int64_t parsed = 0;
            if (!parse_integer(value, parsed) || parsed < 1 || parsed > std::numeric_limits<int>::max()) {
                error = "--pool-workers must be a positive integer";
                return false;
            }
            options.pool_workers = static_cast<int>(parsed);
        } else if (arg == "--prompt-chunk") {
            std::string value;
            if (!need("--prompt-chunk", value)) return false;
            int64_t parsed = 0;
            if (!parse_integer(value, parsed) || parsed != 0) {
                error = "--prompt-chunk must be 0 for forced-token decode";
                return false;
            }
            options.prompt_chunk = 0;
        } else if (arg == "--spec") {
            std::string value;
            if (!need("--spec", value)) return false;
            int64_t parsed = 0;
            if (!parse_integer(value, parsed) || parsed != 0) {
                error = "--spec must be 0 for forced-token decode";
                return false;
            }
            options.spec = 0;
        } else if (arg == "--adapt-every") {
            std::string value;
            if (!need("--adapt-every", value)) return false;
            int64_t parsed = 0;
            if (!parse_integer(value, parsed) || parsed != 0) {
                error = "--adapt-every must be 0 for a deterministic dump";
                return false;
            }
            options.adapt_every = 0;
        } else {
            error = "unknown argument: " + arg;
            return false;
        }
    }
    if (options.exl3.empty() == options.native.empty()) {
        error = "choose exactly one of --exl3 or --native";
        return false;
    }
    if (!options.exl3.empty() && options.metadata_gguf.empty()) {
        error = "--metadata-gguf is required with --exl3";
        return false;
    }
    if (!options.native.empty() && !options.metadata_gguf.empty()) {
        error = "--metadata-gguf is only valid with --exl3";
        return false;
    }
    if (!options.exl3.empty() && !options.profile.empty()) {
        error = "--expert-profile is currently supported only with --native";
        return false;
    }
    if (options.trace.empty() || options.out_dir.empty()) {
        error = "--trace and --out-dir are required";
        return false;
    }
    return true;
}

bool json_ids(const Json& value, const char* name, std::vector<int32_t>& ids, std::string& error) {
    if (!value.is_array()) {
        error = std::string("trace row ") + name + " must be an array";
        return false;
    }
    ids.clear();
    ids.reserve(value.size());
    for (const Json& item : value) {
        if (!item.is_number_integer()) {
            error = std::string("trace row ") + name + " contains a non-integer token";
            return false;
        }
        const int64_t id = item.get<int64_t>();
        if (id < std::numeric_limits<int32_t>::min() || id > std::numeric_limits<int32_t>::max()) {
            error = std::string("trace row ") + name + " contains an out-of-range int32 token";
            return false;
        }
        ids.push_back(static_cast<int32_t>(id));
    }
    return true;
}

bool load_trace(const std::string& path, std::vector<TraceRow>& rows, int64_t& declared_vocab, std::string& error) {
    std::ifstream input(path);
    if (!input) {
        error = "cannot open trace: " + path;
        return false;
    }
    Json root;
    try {
        input >> root;
    } catch (const std::exception& e) {
        error = std::string("trace JSON is malformed: ") + e.what();
        return false;
    }
    declared_vocab = -1;
    if (!root.is_object()) {
        error = "trace must be an object with vocab_size and rows";
        return false;
    }
    const auto vocab = root.find("vocab_size");
    if (vocab == root.end() || !vocab->is_number_integer()) {
        error = "trace vocab_size must be an integer";
        return false;
    }
    declared_vocab = vocab->get<int64_t>();
    if (declared_vocab <= 0) {
        error = "trace vocab_size must be positive";
        return false;
    }
    const Json* entries = &root;
    if (root.is_object()) {
        for (const char* key : {"rows", "records", "examples"}) {
            const auto it = root.find(key);
            if (it != root.end()) {
                entries = &*it;
                break;
            }
        }
    }
    if (!entries->is_array() || entries->empty()) {
        error = "trace must contain a nonempty rows array";
        return false;
    }
    rows.clear();
    rows.reserve(entries->size());
    for (size_t index = 0; index < entries->size(); ++index) {
        const Json& entry = (*entries)[index];
        if (!entry.is_object()) {
            error = "trace row " + std::to_string(index) + " must be an object";
            return false;
        }
        const auto input = entry.find("input_ids");
        const auto response = entry.find("response_ids");
        if (input == entry.end() || response == entry.end()) {
            error = "trace row " + std::to_string(index) + " needs input_ids and response_ids";
            return false;
        }
        TraceRow row;
        const auto id = entry.find("id");
        row.id = id == entry.end() ? Json(index) : *id;
        if (!json_ids(*input, "input_ids", row.input_ids, error) ||
            !json_ids(*response, "response_ids", row.response_ids, error))
            return false;
        if (row.input_ids.empty() && row.response_ids.empty()) {
            error = "trace row " + std::to_string(index) + " has no tokens";
            return false;
        }
        rows.push_back(std::move(row));
    }
    return true;
}

bool checked_logit_size(size_t rows, int64_t vocab, size_t& elements, std::string& error) {
    if (vocab <= 0 || static_cast<uint64_t>(vocab) > std::numeric_limits<size_t>::max()) {
        error = "model vocabulary is invalid";
        return false;
    }
    const size_t width = static_cast<size_t>(vocab);
    if (rows > std::numeric_limits<size_t>::max() / width) {
        error = "logit output size overflows size_t";
        return false;
    }
    elements = rows * width;
    return true;
}

bool write_logits(const std::filesystem::path& path, const std::vector<float>& logits, size_t rows, int64_t vocab,
                  std::string& error) {
    size_t elements = 0;
    if (!checked_logit_size(rows, vocab, elements, error) || logits.size() != elements) return false;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        error = "cannot open logit output: " + path.string();
        return false;
    }
    output.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(elements * sizeof(float)));
    if (!output) {
        error = "writing logit output failed: " + path.string();
        return false;
    }
    return true;
}

bool dump_row(PlainModel& model, const TraceRow& trace, const Options& options, int64_t vocab,
              const std::filesystem::path& output_dir, size_t index, DumpedRow& manifest_row, std::string& error) {
    std::vector<int32_t> tokens;
    tokens.reserve(trace.input_ids.size() + trace.response_ids.size());
    tokens.insert(tokens.end(), trace.input_ids.begin(), trace.input_ids.end());
    tokens.insert(tokens.end(), trace.response_ids.begin(), trace.response_ids.end());
    const size_t n_tokens = tokens.size();
    const size_t range_start = trace.input_ids.empty() ? 0 : trace.input_ids.size() - 1;
    const size_t range_end = n_tokens;
    if (range_start > range_end) {
        error = "forced logit range is invalid";
        return false;
    }

    if (!model.reset(error)) return false;
    const size_t range_rows = range_end - range_start;
    size_t row_elements = 0;
    if (!checked_logit_size(range_rows, vocab, row_elements, error)) return false;
    std::vector<float> selected(row_elements);
    size_t selected_row = 0;
    for (size_t pos = 0; pos < n_tokens; pos += static_cast<size_t>(options.max_t)) {
        const size_t remaining = n_tokens - pos;
        const int T = static_cast<int>((std::min)(static_cast<size_t>(options.max_t), remaining));
        size_t chunk_elements = 0;
        if (!checked_logit_size(static_cast<size_t>(T), vocab, chunk_elements, error)) return false;
        std::vector<float> logits(chunk_elements);
        std::vector<int32_t> output(static_cast<size_t>(T));
        if (!model.forward(tokens.data() + pos, T, static_cast<int64_t>(pos), output.data(), logits.data(), error))
            return false;
        for (int t = 0; t < T; ++t) {
            const size_t absolute = pos + static_cast<size_t>(t);
            if (absolute < range_start || absolute >= range_end) continue;
            const float* source = logits.data() + static_cast<size_t>(t) * static_cast<size_t>(vocab);
            float* destination = selected.data() + selected_row * static_cast<size_t>(vocab);
            for (int64_t col = 0; col < vocab; ++col) {
                if (!std::isfinite(source[col])) {
                    error = "non-finite logit at row " + std::to_string(absolute) + ", vocab " + std::to_string(col);
                    return false;
                }
                destination[col] = source[col];
            }
            ++selected_row;
        }
    }
    if (selected_row != range_rows) {
        error = "forced logit row count did not cover the requested range";
        return false;
    }
    std::ostringstream filename;
    filename << "row_" << std::setw(6) << std::setfill('0') << index << ".f32";
    const std::string relative = filename.str();
    if (!write_logits(output_dir / relative, selected, range_rows, vocab, error)) return false;
    manifest_row = {index, trace.id, n_tokens, trace.input_ids.size(), range_start, range_end, relative};
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    std::string error;
    if (!parse_options(argc, argv, options, error)) {
        usage();
        std::cerr << "plain_model_logits: " << error << '\n';
        return 2;
    }

    std::vector<TraceRow> rows;
    int64_t trace_vocab = -1;
    if (!load_trace(options.trace, rows, trace_vocab, error)) {
        std::cerr << "plain_model_logits: " << error << '\n';
        return 2;
    }
    if (options.selected_rows.empty()) {
        options.selected_rows.resize(rows.size());
        for (size_t i = 0; i < rows.size(); ++i) options.selected_rows[i] = i;
    }
    for (const size_t index : options.selected_rows) {
        if (index >= rows.size()) {
            std::cerr << "plain_model_logits: --rows index is outside the trace: " << index << '\n';
            return 2;
        }
    }

    size_t max_selected_tokens = 0;
    for (const size_t index : options.selected_rows)
        max_selected_tokens = (std::max)(max_selected_tokens, rows[index].input_ids.size() + rows[index].response_ids.size());
    if (max_selected_tokens > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        std::cerr << "plain_model_logits: selected trace length exceeds int64 context limits\n";
        return 2;
    }

    PlainOptions model_options;
    model_options.max_t = options.max_t;
    model_options.max_cells = (std::max)(int64_t(16), static_cast<int64_t>(max_selected_tokens));
    model_options.pool_workers = options.pool_workers;
    model_options.profile = options.profile;
    model_options.prompt_chunk = options.prompt_chunk;
    model_options.spec = options.spec;
    model_options.adapt_every = options.adapt_every;
    if (!options.exl3.empty()) {
        model_options.exl3_dir = options.exl3;
        model_options.metadata_gguf = options.metadata_gguf;
    } else {
        model_options.gguf = options.native;
    }

    PlainModel model;
    if (!model.load(model_options, error)) {
        std::cerr << "plain_model_logits: model load failed: " << error << '\n';
        return 1;
    }
    const int64_t vocab = model.vocab();
    if (vocab < trace_vocab) {
        std::cerr << "plain_model_logits: model vocabulary " << vocab << " is smaller than trace vocabulary "
                  << trace_vocab << '\n';
        return 1;
    }
    for (const TraceRow& row : rows) {
        for (const int32_t token : row.input_ids)
            if (token < 0 || token >= vocab) { error = "trace input token is outside model vocabulary"; break; }
        if (!error.empty()) break;
        for (const int32_t token : row.response_ids)
            if (token < 0 || token >= vocab) { error = "trace response token is outside model vocabulary"; break; }
        if (!error.empty()) break;
    }
    if (!error.empty()) {
        std::cerr << "plain_model_logits: " << error << '\n';
        return 2;
    }

    std::error_code filesystem_error;
    std::filesystem::create_directories(options.out_dir, filesystem_error);
    if (filesystem_error) {
        std::cerr << "plain_model_logits: cannot create output directory: " << filesystem_error.message() << '\n';
        return 2;
    }
    std::vector<DumpedRow> manifest_rows;
    manifest_rows.reserve(options.selected_rows.size());
    for (const size_t index : options.selected_rows) {
        DumpedRow dumped;
        if (!dump_row(model, rows[index], options, vocab, options.out_dir, index, dumped, error)) {
            std::cerr << "plain_model_logits: row " << index << " failed: " << error << '\n';
            return 1;
        }
        manifest_rows.push_back(std::move(dumped));
        std::cout << "completed row " << index << "\n";
    }

    Json manifest = Json{{"format", "xestrata-forced-logits/1"},
                         {"eval_vocab", trace_vocab},
                         {"model_vocab", vocab},
                         {"rows", Json::array()}};
    for (const DumpedRow& row : manifest_rows) {
        manifest["rows"].push_back(Json{{"index", row.index},
                                         {"row_index", row.index},
                                         {"id", row.id},
                                         {"n_tokens", row.n_tokens},
                                         {"input_tokens", row.input_tokens},
                                         {"range_start", row.range_start},
                                         {"range_end", row.range_end},
                                         {"file", row.file}});
    }
    std::ofstream output(std::filesystem::path(options.out_dir) / "manifest.json", std::ios::trunc);
    if (!output) {
        std::cerr << "plain_model_logits: cannot open manifest output\n";
        return 2;
    }
    output << manifest.dump(2) << '\n';
    if (!output) {
        std::cerr << "plain_model_logits: writing manifest failed\n";
        return 2;
    }
    return 0;
}
