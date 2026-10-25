// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Minimal Qwen3.8 EXL3 autoregressive command-line entry point.
#include "strata/program/exl3_flash_generate.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/exl3_flash_model.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace strata::program {
namespace {

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

bool parse_probability(std::string_view text, double& value) {
    if (text.empty()) return false;
    const std::string copy(text);
    char* end = nullptr;
    const double parsed = std::strtod(copy.c_str(), &end);
    if (end != copy.c_str() + copy.size() || !std::isfinite(parsed) || parsed < 0.0 || parsed > 1.0)
        return false;
    value = parsed;
    return true;
}

bool parse_i64(std::string_view text, int64_t& value) {
    if (text.empty()) return false;
    const auto begin = text.data();
    const auto end = begin + text.size();
    const auto parsed = std::from_chars(begin, end, value, 10);
    return parsed.ec == std::errc{} && parsed.ptr == end;
}

bool parse_ids(const std::string& text, std::vector<int32_t>& output) {
    output.clear();
    size_t cursor = 0;
    while (cursor < text.size()) {
        while (cursor < text.size() && (text[cursor] == ',' || text[cursor] == ' ' || text[cursor] == '\n' ||
                                        text[cursor] == '\r' || text[cursor] == '\t')) ++cursor;
        if (cursor == text.size()) break;
        const size_t start = cursor;
        while (cursor < text.size() && text[cursor] != ',' && text[cursor] != ' ' && text[cursor] != '\n' &&
               text[cursor] != '\r' && text[cursor] != '\t') ++cursor;
        int64_t value = 0;
        if (!parse_i64(std::string_view(text).substr(start, cursor - start), value) || value < 0 ||
            value > std::numeric_limits<int32_t>::max()) return false;
        output.push_back(static_cast<int32_t>(value));
    }
    return !output.empty();
}

bool parse_devices(const std::string& text, std::vector<int>& output) {
    std::vector<int32_t> values;
    if (!parse_ids(text, values)) return false;
    output.clear();
    for (const int32_t value : values) {
        if (value > std::numeric_limits<int>::max()) return false;
        output.push_back(static_cast<int>(value));
    }
    return !output.empty();
}

bool append_config_eos(const Json& object, std::vector<int32_t>& eos) {
    const auto it = object.find("eos_token_id");
    if (it == object.end()) return true;
    auto append = [&](const Json& value) {
        if (!value.is_number_integer() && !value.is_number_unsigned()) return false;
        const uint64_t raw = value.is_number_unsigned() ? value.get<uint64_t>() :
                             static_cast<uint64_t>(value.get<int64_t>());
        if (raw > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) return false;
        const int32_t id = static_cast<int32_t>(raw);
        if (std::find(eos.begin(), eos.end(), id) == eos.end()) eos.push_back(id);
        return true;
    };
    if (it->is_array()) {
        for (const auto& value : *it)
            if (!append(value)) return false;
        return true;
    }
    return append(*it);
}

bool read_config_info(const std::string& directory, std::string& type, std::vector<int32_t>& eos) {
    std::ifstream input(std::filesystem::path(directory) / "config.json");
    if (!input) return false;
    try {
        const Json root = Json::parse(std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>()));
        const Json* text = root.contains("text_config") && root["text_config"].is_object() ? &root["text_config"] : &root;
        if (!append_config_eos(*text, eos)) return false;
        if (text != &root && !text->contains("eos_token_id") && !append_config_eos(root, eos)) return false;
        if (text->contains("model_type") && (*text)["model_type"].is_string()) {
            type = (*text)["model_type"].get<std::string>();
            return true;
        }
        if (root.contains("model_type") && root["model_type"].is_string()) {
            type = root["model_type"].get<std::string>();
            return true;
        }
    } catch (const std::exception&) {
    }
    return false;
}

void usage() {
    std::fprintf(stderr,
                 "strata --exl3 DIR --tokens \"1,2,3\" [options]\n\n"
                 "  --exl3 DIR          Qwen3.8 EXL3 archive\n"
                 "  --tokens LIST       token IDs, comma or whitespace separated\n"
                 "  --tokens-file PATH  pretokenized token IDs\n"
                 "  --max-new N         generated token count (default 16)\n"
                 "  --context N         K/V context capacity (default model config)\n"
                 "  --stage-devices L   explicit runtime ordinals, e.g. 0,1\n"
                 "  --stage-layers L    explicit layer counts per stage, e.g. 18,18,12 (default: the even split)\n"
                 "  --pool-workers N    CPU expert workers per stage (default 0: automatic)\n"
                 "  --stop-eos          stop after an EOS token\n"
                 "  --eos ID            add an EOS token and enable stopping\n"
                 "  --eos-ids LIST      replace EOS IDs and enable stopping\n"
                 "  --metadata-gguf P   optional GGUF metadata/tokenizer EOS source\n"
                 "  --dump-logits PATH  write FP32 logits rows for generated choices\n"
                 "  --expert-profile P  rank the device cache's resident experts by this firing-frequency profile\n"
                 "                      (STRP format, as --expert-profile-save writes); without it the spread order is used\n"
                 "  --expert-profile-save P\n"
                 "                      record this run's expert firing frequencies and save them to P as a profile\n"
                 "  --expert-miss M     cpu (default): non-resident experts are computed on the CPU; direct: the GPU\n"
                 "                      reads non-resident experts from pinned host memory over the bus\n"
                 "  --spec 0..6         2..6 run the serial MTP draft loop with up to N-1 drafts per window; 0|1 stay plain AR\n"
                 "  --spec-min-p P      stop a window's draft chain after a draft whose softmax probability is below P\n"
                 "                      (0..1, default 0: never stop early; the first draft is always proposed)\n"
                 "  --max-t 1           single-token AR only\n"
                 "  --prompt-chunk N    batched prefill chunk size (0: token by token, default)\n");
}

bool read_metadata_eos(const std::string& path, std::vector<int32_t>& eos) {
    if (path.empty()) return true;
    try {
        const strata::GgufModel model = strata::GgufModel::open(path);
        for (const char* key : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                                "tokenizer.ggml.eom_token_id"}) {
            const auto* value = model.meta().get(key);
            if (value != nullptr && value->is_num()) {
                const int64_t id = static_cast<int64_t>(value->u);
                if (id >= 0 && id <= std::numeric_limits<int32_t>::max() &&
                    std::find(eos.begin(), eos.end(), static_cast<int32_t>(id)) == eos.end())
                    eos.push_back(static_cast<int32_t>(id));
            }
        }
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata exl3-flash: metadata GGUF: %s\n", e.what());
        return false;
    }
}

}  // namespace

int exl3_flash_main(int argc, char** argv) {
    std::string directory;
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--exl3") directory = argv[i + 1];
    if (directory.empty()) return -1;

    std::string model_type;
    std::vector<int32_t> config_eos;
    if (!read_config_info(directory, model_type, config_eos)) return -1;
    if (model_type == "qwen3_5_moe" || model_type == "qwen3_5_moe_text" || model_type == "qwen3_5" ||
        model_type == "qwen3_5_text")
        return -1;
    if (model_type != "qwen4_exp" && model_type != "qwen4_exp_text") return -1;

    std::vector<int32_t> prompt;
    std::vector<int> stage_devices;
    std::vector<int64_t> stage_layers;
    std::vector<int32_t> eos = config_eos;
    std::string metadata_gguf, dump_path;
    std::string expert_profile, expert_profile_save;
    bool expert_miss_direct = false;
    int64_t max_new = 16, context = 0;
    bool stop_eos = false;
    int spec = 0, pool_workers = 0;
    double spec_min_p = 0.0;
    int64_t prompt_chunk = 0;
    bool prompt_seen = false, eos_explicit = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata exl3-flash: %s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (argument == "--exl3") (void) next("--exl3");
        else if (argument == "--tokens") {
            if (prompt_seen) {
                std::fprintf(stderr, "strata exl3-flash: --tokens/--tokens-file may be specified only once\n");
                return 2;
            }
            prompt_seen = true;
            if (!parse_ids(next("--tokens"), prompt)) {
                std::fprintf(stderr, "strata exl3-flash: --tokens must be a nonempty ID list\n");
                return 2;
            }
        } else if (argument == "--tokens-file") {
            if (prompt_seen) {
                std::fprintf(stderr, "strata exl3-flash: --tokens/--tokens-file may be specified only once\n");
                return 2;
            }
            prompt_seen = true;
            std::ifstream input(next("--tokens-file"));
            const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            if ((!input.good() && !input.eof()) || !parse_ids(text, prompt)) {
                std::fprintf(stderr, "strata exl3-flash: cannot read --tokens-file ID list\n");
                return 2;
            }
        } else if (argument == "--max-new") {
            if (!parse_i64(next("--max-new"), max_new)) {
                std::fprintf(stderr, "strata exl3-flash: --max-new must be a strict int64\n");
                return 2;
            }
        } else if (argument == "--context" || argument == "--max-context") {
            if (!parse_i64(next(argument.c_str()), context)) {
                std::fprintf(stderr, "strata exl3-flash: %s must be a strict int64\n", argument.c_str());
                return 2;
            }
        }
        else if (argument == "--stage-devices") {
            if (!parse_devices(next("--stage-devices"), stage_devices)) {
                std::fprintf(stderr, "strata exl3-flash: --stage-devices must be a nonempty ordinal list\n");
                return 2;
            }
        } else if (argument == "--stage-layers") {
            std::vector<int32_t> counts;
            if (!parse_ids(next("--stage-layers"), counts)) {
                std::fprintf(stderr, "strata exl3-flash: --stage-layers must be a nonempty layer count list\n");
                return 2;
            }
            stage_layers.clear();
            for (const int32_t count : counts) stage_layers.push_back(count);
        } else if (argument == "--pool-workers") {
            int64_t value = 0;
            if (!parse_i64(next("--pool-workers"), value) || value < 0 ||
                value > std::numeric_limits<int>::max()) {
                std::fprintf(stderr, "strata exl3-flash: --pool-workers must be a nonnegative int32 (0: automatic)\n");
                return 2;
            }
            pool_workers = static_cast<int>(value);
        } else if (argument == "--stop-eos") stop_eos = true;
        else if (argument == "--eos") {
            int64_t value = 0;
            if (!parse_i64(next("--eos"), value) || value < 0 || value > std::numeric_limits<int32_t>::max()) {
                std::fprintf(stderr, "strata exl3-flash: --eos must be a strict int32 token ID\n");
                return 2;
            }
            eos = {static_cast<int32_t>(value)};
            eos_explicit = true;
            stop_eos = true;
        } else if (argument == "--eos-ids") {
            if (!parse_ids(next("--eos-ids"), eos)) {
                std::fprintf(stderr, "strata exl3-flash: --eos-ids must be a strict token ID list\n");
                return 2;
            }
            eos_explicit = true;
            stop_eos = true;
        } else if (argument == "--metadata-gguf") metadata_gguf = next("--metadata-gguf");
        else if (argument == "--expert-profile") expert_profile = next("--expert-profile");
        else if (argument == "--expert-profile-save") expert_profile_save = next("--expert-profile-save");
        else if (argument == "--expert-miss") {
            const std::string mode = next("--expert-miss");
            if (mode == "cpu") expert_miss_direct = false;
            else if (mode == "direct") expert_miss_direct = true;
            else {
                std::fprintf(stderr, "strata exl3-flash: --expert-miss accepts cpu or direct\n");
                return 2;
            }
        }
        else if (argument == "--dump-logits") dump_path = next("--dump-logits");
        else if (argument == "--spec0") spec = 0;
        else if (argument == "--spec1") spec = 1;
        else if (argument == "--spec") {
            int64_t value = 0;
            if (!parse_i64(next("--spec"), value) || value < std::numeric_limits<int>::min() ||
                value > std::numeric_limits<int>::max()) {
                std::fprintf(stderr, "strata exl3-flash: --spec must be a strict int32\n");
                return 2;
            }
            spec = static_cast<int>(value);
        } else if (argument == "--spec-min-p") {
            if (!parse_probability(next("--spec-min-p"), spec_min_p)) {
                std::fprintf(stderr, "strata exl3-flash: --spec-min-p must be a probability in [0, 1]\n");
                return 2;
            }
        } else if (argument.rfind("--spec-min-p=", 0) == 0) {
            if (!parse_probability(std::string_view(argument).substr(13), spec_min_p)) {
                std::fprintf(stderr, "strata exl3-flash: --spec-min-p must be a probability in [0, 1]\n");
                return 2;
            }
        } else if (argument == "--max-t") {
            int64_t value = 0;
            if (!parse_i64(next("--max-t"), value) || value != 1) {
                std::fprintf(stderr, "strata exl3-flash: only --max-t 1 is supported\n");
                return 2;
            }
        } else if (argument == "--max-t1" || argument == "--prompt-chunk0") {
        } else if (argument == "--prompt-chunk") {
            if (!parse_i64(next("--prompt-chunk"), prompt_chunk) || prompt_chunk < 0) {
                std::fprintf(stderr, "strata exl3-flash: --prompt-chunk must be a nonnegative int64\n");
                return 2;
            }
        } else if (argument == "--serve" || argument == "--batch" || argument == "--vision" || argument == "--mtp") {
            std::fprintf(stderr, "strata exl3-flash: %s is not supported by the Qwen3.8 EXL3 path\n", argument.c_str());
            return 2;
        } else if (argument == "--help" || argument == "-h") {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "strata exl3-flash: unsupported option %s\n", argument.c_str());
            return 2;
        }
    }
    if (spec < 0 || spec > 6) {
        std::fprintf(stderr, "strata exl3-flash: --spec accepts 0..6 (2..6 run the serial MTP draft loop)\n");
        return 2;
    }
    if (spec < 0 || max_new < 1 || prompt.empty()) {
        std::fprintf(stderr, "strata exl3-flash: --tokens and positive --max-new are required\n");
        return 2;
    }
    if (context < 0) {
        std::fprintf(stderr, "strata exl3-flash: --context must be nonnegative\n");
        return 2;
    }
    if (prompt.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        max_new - 1 > std::numeric_limits<int64_t>::max() - static_cast<int64_t>(prompt.size())) {
        std::fprintf(stderr, "strata exl3-flash: prompt plus max-new overflows int64\n");
        return 2;
    }
    const int64_t required_cells = static_cast<int64_t>(prompt.size()) + (max_new - 1);
    if (context > 0 && context < required_cells) {
        std::fprintf(stderr, "strata exl3-flash: context is smaller than prompt plus max-new\n");
        return 2;
    }
    if (!metadata_gguf.empty() && !eos_explicit) eos.clear();
    if (!eos_explicit && !read_metadata_eos(metadata_gguf, eos)) return 2;
    if (stop_eos && eos.empty()) {
        std::fprintf(stderr, "strata exl3-flash: --stop-eos needs EOS IDs from config, metadata GGUF, or --eos-ids\n");
        return 2;
    }

    strata::core::Exl3FlashModel model;
    strata::core::Exl3FlashModelOptions options;
    options.stage_devices = stage_devices;
    options.stage_layers = stage_layers;
    options.pool_workers = pool_workers;
    options.max_cells = context;
    options.prompt_chunk = prompt_chunk;
    options.include_mtp = (spec >= 2);
    options.expert_profile = expert_profile;
    options.expert_profile_save = expert_profile_save;
    options.expert_miss_direct = expert_miss_direct;
    options.vision = false;
    std::string err;
    const Clock::time_point load_begin = Clock::now();
    if (!model.open(directory, options, err)) {
        std::fprintf(stderr, "strata exl3-flash: %s\n", err.c_str());
        return 1;
    }
    if (required_cells > model.spec().context_length) {
        std::fprintf(stderr, "strata exl3-flash: prompt plus generation exceeds model context\n");
        return 2;
    }
    if (!model.reset(err)) {
        std::fprintf(stderr, "strata exl3-flash: reset: %s\n", err.c_str());
        return 1;
    }
    const int64_t vocab = model.model_map().geometry.vocab_size;
    std::vector<float> logits(static_cast<size_t>(vocab));
    std::ofstream dump;
    if (!dump_path.empty()) {
        dump.open(dump_path, std::ios::binary);
        if (!dump) {
            std::fprintf(stderr, "strata exl3-flash: cannot open --dump-logits path\n");
            return 2;
        }
    }
    int32_t next_token = -1;
    const Clock::time_point prompt_begin = Clock::now();
    if (prompt_chunk > 0) {
        if (!model.forward_prefill(prompt.data(), static_cast<int64_t>(prompt.size()), 0, next_token,
                                   dump_path.empty() ? nullptr : logits.data(), err)) {
            std::fprintf(stderr, "strata exl3-flash: prompt prefill: %s\n", err.c_str());
            return 1;
        }
    } else {
        for (size_t position = 0; position < prompt.size(); ++position) {
            if (!model.forward(prompt[position], static_cast<int64_t>(position), next_token,
                               dump_path.empty() ? nullptr : logits.data(), err)) {
                std::fprintf(stderr, "strata exl3-flash: prompt position %zu: %s\n", position, err.c_str());
                return 1;
            }
        }
    }
    const Clock::time_point prompt_end = Clock::now();
    std::vector<int32_t> generated;
    generated.reserve(static_cast<size_t>(max_new));
    const bool use_spec = (spec >= 2);
    strata::core::Exl3FlashSnapshot snap;
    size_t spec_drafted = 0;
    size_t spec_accepted = 0;
    size_t spec_cut = 0;
    const auto emit_token = [&](int32_t token) {
        generated.push_back(token);
        if (!dump_path.empty())
            dump.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * sizeof(float)));
    };
    const Clock::time_point decode_begin = Clock::now();
    int64_t position = static_cast<int64_t>(prompt.size());
    while (static_cast<int64_t>(generated.size()) < max_new) {
        const int32_t emitted = next_token;
        emit_token(emitted);
        if (stop_eos && std::find(eos.begin(), eos.end(), emitted) != eos.end()) break;
        if (static_cast<int64_t>(generated.size()) >= max_new) break;
        if (use_spec && spec >= 2 && static_cast<int64_t>(generated.size()) + 1 < max_new) {
            if (!model.save_snapshot(snap, err)) {
                std::fprintf(stderr, "strata exl3-flash: snapshot: %s\n", err.c_str());
                return 1;
            }
            int32_t cur = emitted;
            int64_t cur_pos = position;
            int attempted = 0;
            int accepted = 0;
            bool window_done = false;
            for (int round = 1; round < spec && !window_done; ++round) {
                if (static_cast<int64_t>(generated.size()) + 1 >= max_new) break;
                int32_t draft = -1;
                float draft_prob = 0.0f;
                if (!model.mtp_draft(cur, cur_pos - 1, draft, nullptr, err,
                                     spec_min_p > 0.0 ? &draft_prob : nullptr)) {
                    std::fprintf(stderr, "strata exl3-flash: draft position %lld: %s\n", (long long) cur_pos, err.c_str());
                    return 1;
                }
                ++attempted;
                ++spec_drafted;
                int32_t verified = -1;
                if (!model.forward(cur, cur_pos, verified, dump_path.empty() ? nullptr : logits.data(), err)) {
                    std::fprintf(stderr, "strata exl3-flash: verify position %lld: %s\n", (long long) cur_pos, err.c_str());
                    return 1;
                }
                if (draft != verified) break;
                ++accepted;
                ++spec_accepted;
                cur = verified;
                ++cur_pos;
                emit_token(cur);
                if (stop_eos && std::find(eos.begin(), eos.end(), cur) != eos.end()) window_done = true;
                if (static_cast<int64_t>(generated.size()) >= max_new) window_done = true;
                // --spec-min-p: a draft the MTP head itself is unsure of ends
                // the chain here; the drafts already accepted stand, and the
                // window closes as if every draft had been proposed.  The
                // token stream is unchanged: unproposed drafts would only
                // have been verified, never emitted unverified.
                if (!window_done && !strata::core::exl3_flash_spec_extends(draft_prob, spec_min_p)) {
                    ++spec_cut;
                    break;
                }
            }
            if (window_done) break;
            if (attempted > 0 && accepted == attempted) {
                if (!model.forward(cur, cur_pos, next_token, dump_path.empty() ? nullptr : logits.data(), err)) {
                    std::fprintf(stderr, "strata exl3-flash: decode position %lld: %s\n", (long long) cur_pos, err.c_str());
                    return 1;
                }
                position = cur_pos + 1;
                continue;
            }
            if (!model.restore_snapshot(snap, err)) {
                std::fprintf(stderr, "strata exl3-flash: restore: %s\n", err.c_str());
                return 1;
            }
            cur = emitted;
            cur_pos = position;
            for (int i = 0; i <= accepted; ++i) {
                if (!model.forward(cur, cur_pos, cur, dump_path.empty() ? nullptr : logits.data(), err)) {
                    std::fprintf(stderr, "strata exl3-flash: decode position %lld: %s\n", (long long) cur_pos, err.c_str());
                    return 1;
                }
                ++cur_pos;
            }
            next_token = cur;
            position = cur_pos;
            continue;
        }
        if (!model.forward(emitted, position, next_token, dump_path.empty() ? nullptr : logits.data(), err)) {
            std::fprintf(stderr, "strata exl3-flash: decode position %lld: %s\n", (long long) position, err.c_str());
            return 1;
        }
        ++position;
    }
    strata::core::Exl3FlashModel::release_snapshot(snap);
    const Clock::time_point decode_end = Clock::now();
    if (!dump_path.empty() && !dump.good()) {
        std::fprintf(stderr, "strata exl3-flash: writing --dump-logits failed\n");
        return 1;
    }
    std::printf("tokens: ");
    for (size_t i = 0; i < generated.size(); ++i) std::printf(i == 0 ? "%d" : ",%d", generated[i]);
    std::printf("\n");
    const double load_s = std::chrono::duration<double>(prompt_begin - load_begin).count();
    const double prompt_s = std::chrono::duration<double>(prompt_end - prompt_begin).count();
    const double decode_s = std::chrono::duration<double>(decode_end - decode_begin).count();
    const size_t decode_tokens = generated.empty() ? 0 : generated.size() - 1;
    std::fprintf(stderr, "strata (qwen4exp-exl3): load %.1f s, prompt %zu tokens %.2f s, %zu decode tokens %.2f s (%.2f tok/s)\n",
                 load_s, prompt.size(), prompt_s, decode_tokens, decode_s,
                 static_cast<double>(decode_tokens) / std::max(decode_s, 1e-9));
    if (use_spec)
        std::fprintf(stderr, "strata (qwen4exp-exl3): spec%d drafts %zu accepted %zu (%.1f %%)\n", spec,
                     spec_drafted, spec_accepted,
                     spec_drafted == 0 ? 0.0 : 100.0 * static_cast<double>(spec_accepted) / static_cast<double>(spec_drafted));
    if (use_spec && spec_min_p > 0.0)
        std::fprintf(stderr, "strata (qwen4exp-exl3): spec-min-p %.3f cut %zu windows\n", spec_min_p, spec_cut);
    {
        const auto cache = model.expert_cache_stats();
        const int64_t routed = cache.hit_entries + cache.cpu_miss_entries + cache.direct_miss_entries;
        std::fprintf(stderr,
                     "strata (qwen4exp-exl3): expert cache: %lld resident, %lld routed entries, %lld hits "
                     "(%.2f%%), %lld cpu misses, %lld direct misses, miss mode %s\n",
                     (long long) cache.resident_pairs, (long long) routed, (long long) cache.hit_entries,
                     routed > 0 ? 100.0 * static_cast<double>(cache.hit_entries) / static_cast<double>(routed) : 0.0,
                     (long long) cache.cpu_miss_entries, (long long) cache.direct_miss_entries,
                     expert_miss_direct ? "direct" : "cpu");
        if (!expert_profile_save.empty()) {
            std::string save_err;
            if (!model.save_expert_profile(expert_profile_save, save_err)) {
                std::fprintf(stderr, "strata exl3-flash: --expert-profile-save: %s\n", save_err.c_str());
                return 1;
            }
            std::fprintf(stderr, "strata (qwen4exp-exl3): expert profile saved to %s\n",
                         expert_profile_save.c_str());
        }
    }
    return 0;
}

}  // namespace strata::program
