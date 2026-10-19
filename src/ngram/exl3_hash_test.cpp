// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/artifact/exl3_ngram.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/ngram.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace format = strata::kernels::exl3_ngram;
namespace artifact = strata::artifact::exl3_ngram;
using Json = nlohmann::json;

std::vector<uint8_t> u16_bytes(const std::vector<uint16_t>& values) {
    std::vector<uint8_t> result(values.size() * 2);
    for (size_t i = 0; i < values.size(); ++i) {
        result[2 * i] = static_cast<uint8_t>(values[i]);
        result[2 * i + 1] = static_cast<uint8_t>(values[i] >> 8);
    }
    return result;
}

std::vector<uint8_t> i64_bytes(const std::vector<int64_t>& values) {
    std::vector<uint8_t> result(values.size() * sizeof(int64_t));
    for (size_t i = 0; i < values.size(); ++i) {
        uint64_t raw = 0;
        std::memcpy(&raw, &values[i], sizeof(raw));
        for (int b = 0; b < 8; ++b) result[i * 8 + static_cast<size_t>(b)] = static_cast<uint8_t>(raw >> (8 * b));
    }
    return result;
}

std::vector<uint8_t> packed_row(int bits, int seed) {
    const size_t words = format::words_per_row(bits);
    std::vector<uint16_t> row(words, 0);
    row[0] = strata::kernels::f16_from_f32(1.0f + static_cast<float>(seed) * 0.01f);
    return u16_bytes(row);
}

void write_synthetic_sidecar(const std::filesystem::path& path) {
    constexpr int bits = 2;
    constexpr int heads = strata::kernels::PLE_N_HEADS;
    constexpr int rows = 32;
    const std::string prefix = "model.layers.0.ple.ngram";
    std::vector<std::pair<std::string, std::pair<std::string, std::vector<uint64_t>>>> tensors;
    const std::vector<uint16_t> bias(static_cast<size_t>(heads) * format::kRowDim, 0);
    tensors.push_back({prefix + ".head_bias", {"F16", {heads, format::kRowDim}}});
    std::vector<int64_t> sizes(heads), offsets(heads);
    int64_t next = 0;
    for (int h = 0; h < heads; ++h) {
        sizes[static_cast<size_t>(h)] = h % 2 == 0 ? 1 : 2;
        offsets[static_cast<size_t>(h)] = next;
        next += sizes[static_cast<size_t>(h)];
    }
    tensors.push_back({prefix + ".head_offsets", {"I64", {heads}}});
    tensors.push_back({prefix + ".head_vocab_sizes", {"I64", {heads}}});
    tensors.push_back({prefix + ".layer_multipliers", {"I64", {3}}});
    std::vector<uint8_t> rows_bytes;
    for (int row = 0; row < rows; ++row) {
        const auto bytes = packed_row(bits, row);
        rows_bytes.insert(rows_bytes.end(), bytes.begin(), bytes.end());
    }
    tensors.push_back({prefix + ".shard_0.trellis", {"I16", {rows, format::words_per_row(bits)}}});
    Json header = Json::object();
    header["__metadata__"] = { {"format", "exl3_ngram_trellis"}, {"version", "1"}, {"K", "2"},
                                {"codebook", "mul1"}, {"row_dim", "160"}, {"rows", std::to_string(rows)},
                                {"shard_rows", std::to_string(rows)} };
    const std::vector<std::vector<uint8_t>> payloads = {
        u16_bytes(bias), i64_bytes(offsets), i64_bytes(sizes),
        i64_bytes({-9223372036854775807LL, 5, 11}), rows_bytes};
    uint64_t offset = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        header[tensors[i].first] = { {"dtype", tensors[i].second.first}, {"shape", tensors[i].second.second},
                                     {"data_offsets", {offset, offset + payloads[i].size()}} };
        offset += payloads[i].size();
    }
    const std::string text = header.dump();
    std::ofstream output(path, std::ios::binary);
    const uint64_t header_bytes = text.size();
    for (int b = 0; b < 8; ++b) output.put(static_cast<char>(header_bytes >> (8 * b)));
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    for (const auto& payload : payloads)
        output.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
}

std::vector<int64_t> signed_reference(const std::vector<int64_t>& tokens, const std::vector<int64_t>& multipliers,
                                      const std::vector<int64_t>& offsets, const std::vector<int64_t>& sizes,
                                      int64_t eos) {
    const size_t context = strata::kernels::NGRAM_SIZE - 1;
    std::vector<int64_t> result((tokens.size() - context) * offsets.size());
    for (size_t row = 0; row < tokens.size() - context; ++row) {
        const size_t position = row + context;
        int64_t previous_eos = -1;
        for (size_t candidate = 0; candidate < position; ++candidate)
            if (tokens[candidate] == eos) previous_eos = static_cast<int64_t>(candidate);
        for (int n = 2; n <= strata::kernels::NGRAM_SIZE; ++n) {
            uint64_t mixed = static_cast<uint64_t>(tokens[position]) * static_cast<uint64_t>(multipliers[0]);
            for (int s = 1; s < n; ++s) {
                const int64_t source = static_cast<int64_t>(position) - s;
                const int64_t value = source >= previous_eos + 1 ? tokens[static_cast<size_t>(source)] : eos;
                mixed ^= static_cast<uint64_t>(value) * static_cast<uint64_t>(multipliers[static_cast<size_t>(s)]);
            }
            int64_t signed_mixed = 0;
            std::memcpy(&signed_mixed, &mixed, sizeof(signed_mixed));
            const size_t base = static_cast<size_t>(n - 2) * strata::kernels::HEADS_PER_NGRAM;
            for (int g = 0; g < strata::kernels::HEADS_PER_NGRAM; ++g) {
                const size_t h = base + static_cast<size_t>(g);
                const int64_t rem = signed_mixed % sizes[h];
                result[row * offsets.size() + h] = offsets[h] + (rem < 0 ? rem + sizes[h] : rem);
            }
        }
    }
    return result;
}

bool compare_rows(const std::vector<int64_t>& expected, const std::vector<uint32_t>& got, size_t row_count,
                  const char* label) {
    if (expected.size() != got.size()) return false;
    for (size_t i = 0; i < got.size(); ++i)
        if (expected[i] < 0 || static_cast<uint64_t>(expected[i]) != got[i]) {
            std::cerr << label << " mismatch at " << i << " expected=" << expected[i] << " got=" << got[i] << '\n';
            return false;
        }
    return row_count > 0;
}

bool check_legacy_fallback() {
    strata::kernels::PleTable table;
    const int32_t tokens[] = {17, 19, 23, 29};
    const int32_t prev[] = {-1, -1, -1, 17, 19, 23, 23, 29};
    std::vector<uint32_t> expected(4 * strata::kernels::PLE_N_HEADS);
    std::vector<uint32_t> got(expected.size());
    const auto constants = strata::kernels::ple_artifact_consts();
    strata::kernels::ngram_rows(tokens, prev, 4, constants, expected.data());
    table.lookup_rows(tokens, prev, 4, constants, got.data());
    return expected == got;
}

bool check_sidecar(const std::filesystem::path& path) {
    strata::artifact::exl3_ngram::Reader reader(path);
    strata::kernels::PleTable table;
    strata::kernels::PleIoOptions io;
    io.max_inflight = 4;
    io.cache_rows = 0;
    io.io_thread = false;
    std::string err;
    const bool opened = table.open_exl3(path.string(), err, io);
    if (!opened) {
        std::cerr << "PleTable open failed: " << err << '\n';
        return false;
    }
    const int32_t tokens[] = {17, 19, strata::kernels::PLE_EOS_TOKEN_ID, 29, 31, 37, 41, 43};
    const int32_t highbit[] = {INT32_MAX, INT32_MAX - 1, 7, 19, 31, 37, 41, 43};
    for (const auto& sequence : {std::vector<int32_t>(std::begin(tokens), std::end(tokens)),
                                 std::vector<int32_t>(std::begin(highbit), std::end(highbit))}) {
        const size_t count = sequence.size();
        std::vector<int32_t> prev(count * 2, strata::kernels::TOKEN_NULL);
        for (size_t i = 0; i < count; ++i) {
            if (i >= 2) prev[i * 2] = sequence[i - 2];
            if (i >= 1) prev[i * 2 + 1] = sequence[i - 1];
        }
        std::vector<int64_t> expected;
        std::vector<uint32_t> got(count * strata::kernels::PLE_N_HEADS);
        std::vector<int64_t> as_i64(sequence.begin(), sequence.end());
        reader.lookup_ids(as_i64.data(), as_i64.size(), strata::kernels::NGRAM_SIZE,
                          strata::kernels::HEADS_PER_NGRAM, strata::kernels::PLE_EOS_TOKEN_ID, expected);
        const auto independent = signed_reference(as_i64, reader.layer_multipliers(), reader.head_offsets(),
                                                  reader.head_vocab_sizes(), strata::kernels::PLE_EOS_TOKEN_ID);
        if (expected != independent) {
            std::cerr << "Reader oracle disagrees with independent signed hash\n";
            return false;
        }
        table.lookup_rows(sequence.data(), prev.data(), static_cast<int>(count),
                          strata::kernels::ple_artifact_consts(), got.data());
        std::vector<uint32_t> selected(got.begin() + 2 * strata::kernels::PLE_N_HEADS, got.end());
        if (!compare_rows(expected, selected, count - 2, "EXL3 metadata hash")) return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (!check_legacy_fallback()) {
        std::cerr << "legacy fallback changed\n";
        return 1;
    }
    const auto root = std::filesystem::temp_directory_path() /
                      ("xestrata_ple_hash_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    const auto synthetic = root / "synthetic.safetensors";
    write_synthetic_sidecar(synthetic);
    const bool synthetic_ok = check_sidecar(synthetic);
    std::filesystem::remove_all(root);
    if (!synthetic_ok) return 1;
    if (argc > 1 && !check_sidecar(argv[1])) return 1;
    std::cout << "PLE hash lookup checks: OK\n";
    return 0;
}
