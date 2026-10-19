// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU-only EXL3 n-gram sidecar reader selftest.
#include "strata/artifact/exl3_ngram.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/ngram/ple_reader.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace artifact = strata::artifact::exl3_ngram;
namespace format = strata::kernels::exl3_ngram;
using Json = nlohmann::json;

namespace {

struct Tensor {
    std::string name;
    std::string dtype;
    std::vector<uint64_t> shape;
    std::vector<uint8_t> data;
};

int failures = 0;

void check(bool condition, const std::string& label) {
    std::cout << "  " << label << ": " << (condition ? "OK" : "FAIL") << '\n';
    if (!condition) ++failures;
}

std::vector<uint8_t> u16_words(const std::vector<uint16_t>& words) {
    std::vector<uint8_t> result(words.size() * 2);
    for (size_t i = 0; i < words.size(); ++i) {
        result[2 * i] = static_cast<uint8_t>(words[i]);
        result[2 * i + 1] = static_cast<uint8_t>(words[i] >> 8);
    }
    return result;
}

std::vector<uint8_t> i64_values(const std::vector<int64_t>& values) {
    std::vector<uint8_t> result(values.size() * 8);
    for (size_t i = 0; i < values.size(); ++i) {
        uint64_t raw;
        std::memcpy(&raw, &values[i], sizeof(raw));
        for (int byte = 0; byte < 8; ++byte) result[i * 8 + static_cast<size_t>(byte)] = static_cast<uint8_t>(raw >> (8 * byte));
    }
    return result;
}

std::vector<uint8_t> row(int seed, int bits, uint16_t scale) {
    const size_t words = format::words_per_row(bits);
    std::vector<uint16_t> packed(words, 0);
    packed[0] = scale;
    for (int stream_bit = 0; stream_bit < format::kRowDim * bits; ++stream_bit) {
        const int value = ((stream_bit * 17 + seed * 13) ^ (stream_bit >> 2)) & 1;
        packed[1 + stream_bit / 16] |= static_cast<uint16_t>(value << (stream_bit & 15));
    }
    return u16_words(packed);
}

void write_sidecar(const std::filesystem::path& path, int bits, bool gap_ranges = false,
                   const std::vector<int64_t>& multipliers = {1, 3, 5}) {
    const std::string prefix = "model.layers.0.ple.ngram";
    const size_t words = format::words_per_row(bits);
    const int rows = gap_ranges ? 5 : 5;
    std::vector<Tensor> tensors;
    std::vector<uint16_t> bias(2 * format::kRowDim, 0);
    for (int i = 0; i < format::kRowDim; ++i) bias[format::kRowDim + i] = 0x3c00;
    tensors.push_back({prefix + ".head_bias", "F16", {2, format::kRowDim}, u16_words(bias)});
    const std::vector<int64_t> offsets = gap_ranges ? std::vector<int64_t>{0, 1} : std::vector<int64_t>{0, 3};
    const std::vector<int64_t> sizes = gap_ranges ? std::vector<int64_t>{1, 1} : std::vector<int64_t>{3, 2};
    tensors.push_back({prefix + ".head_offsets", "I64", {2}, i64_values(offsets)});
    tensors.push_back({prefix + ".head_vocab_sizes", "I64", {2}, i64_values(sizes)});
    tensors.push_back({prefix + ".layer_multipliers", "I64", {multipliers.size()}, i64_values(multipliers)});
    std::vector<uint8_t> first, second;
    for (int i = 0; i < 3; ++i) { const auto bytes = row(i, bits, static_cast<uint16_t>(0x3c00 + i * 0x100)); first.insert(first.end(), bytes.begin(), bytes.end()); }
    for (int i = 3; i < rows; ++i) { const auto bytes = row(i, bits, static_cast<uint16_t>(0x3c00 + i * 0x100)); second.insert(second.end(), bytes.begin(), bytes.end()); }
    tensors.push_back({prefix + ".shard_0.trellis", "I16", {3, words}, std::move(first)});
    tensors.push_back({prefix + ".shard_1.trellis", "I16", {2, words}, std::move(second)});
    Json header = Json::object();
    header["__metadata__"] = {
        {"format", "exl3_ngram_trellis"}, {"version", "1"}, {"K", std::to_string(bits)},
        {"codebook", "mul1"}, {"row_dim", "160"}, {"rows", std::to_string(rows)}, {"shard_rows", "3"},
    };
    uint64_t offset = 0;
    for (const Tensor& tensor : tensors) {
        header[tensor.name] = { {"dtype", tensor.dtype}, {"shape", tensor.shape}, {"data_offsets", {offset, offset + tensor.data.size()}} };
        offset += tensor.data.size();
    }
    const std::string text = header.dump();
    std::ofstream output(path, std::ios::binary);
    const uint64_t header_size = text.size();
    for (int byte = 0; byte < 8; ++byte) output.put(static_cast<char>(header_size >> (8 * byte)));
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    for (const Tensor& tensor : tensors) output.write(reinterpret_cast<const char*>(tensor.data.data()), static_cast<std::streamsize>(tensor.data.size()));
}

void run_reader(const std::filesystem::path& root) {
    const auto path = root / "ngram.safetensors";
    write_sidecar(path, 2);
    artifact::Reader reader(path);
    check(reader.bits() == 2 && reader.words_per_row() == 21 && reader.row_count() == 5, "header K/words/rows");
    check(reader.row_spans().size() == 2 && reader.row_spans()[0].first_row == 0 && reader.row_spans()[0].n_rows == 3 &&
              reader.row_spans()[1].first_row == 3 && reader.row_spans()[1].file_offset > 0,
          "validated logical shard spans are exposed");
    const int64_t ids[] = {2, 3, 0};
    std::vector<uint16_t> packed;
    reader.read_rows(ids, 3, packed);
    check(packed.size() == 3 * reader.words_per_row() && packed[0] == 0x3e00, "reordered blocking row read");
    std::vector<strata::ngram::RowSpan> spans;
    for (const auto& span : reader.row_spans())
        spans.push_back({span.first_row, span.n_rows, span.file_offset});
    strata::ngram::PleReader async_reader;
    std::string io_error;
    const bool opened = async_reader.open_spans(path.string(), spans, (uint64_t) reader.row_count(), 4, 16,
                                                io_error, true, (uint32_t) (reader.words_per_row() * 2));
    check(opened, "validated sidecar spans open in shared async reader");
    if (opened) {
        const uint32_t requested[] = {2, 3, 0};
        std::vector<uint8_t> raw(packed.size() * 2, 0);
        const auto ticket = async_reader.issue(requested, 3, raw.data());
        check(async_reader.collect(ticket, io_error) && std::memcmp(raw.data(), packed.data(), raw.size()) == 0,
              "async sidecar row bytes match blocking reader");
    }
    const int32_t heads[] = {0, 1, 0};
    std::vector<float> output(3 * 167, -4.0f);
    const auto status = reader.decode_rows(ids, heads, 3, output.data(), 167);
    check(status == strata::kernels::cpu::exl3::Status::ok, "explicit-head decode");
    check(output[160] == -4.0f && output[167 + 160] == -4.0f, "decode padding remains untouched");
    std::vector<float> auto_output(3 * 160, 0.0f);
    check(reader.decode_rows(ids, nullptr, 3, auto_output.data(), 160) == strata::kernels::cpu::exl3::Status::ok,
          "automatic head decode in valid ranges");
    const int64_t tokens[] = {1, 2, 3};
    std::vector<int64_t> lookup;
    reader.lookup_ids(tokens, 3, 3, 1, 99, lookup);
    check(lookup.size() == 2 && lookup[0] == 2 && lookup[1] == 3, "signed hash lookup boundary");
    if (lookup.size() == 2) std::cout << "    lookup=" << lookup[0] << "," << lookup[1] << '\n';
    const auto negative = root / "ngram-negative.safetensors";
    write_sidecar(negative, 3, false,
                  {0x7fffffffffffffedLL, 0x4000000000000001LL, static_cast<int64_t>(0x8000000000000003ULL)});
    artifact::Reader negative_reader(negative);
    const int64_t negative_tokens[] = {248045, 846, 198, 20206, 248044, 198, 248045, 74455, 198, 42};
    std::vector<int64_t> negative_lookup;
    negative_reader.lookup_ids(negative_tokens, 10, 3, 1, 248044, negative_lookup);
    const int64_t negative_expected[] = {0, 4, 2, 3, 2, 3, 2, 3, 2, 4, 1, 3, 0, 3, 1, 4};
    check(negative_lookup.size() == 16 && std::equal(negative_lookup.begin(), negative_lookup.end(), negative_expected),
          "signed int64 hash and EOS boundary");

    const auto gap = root / "ngram-gap.safetensors";
    write_sidecar(gap, 5, true);
    artifact::Reader gap_reader(gap);
    const int64_t gap_row[] = {2};
    std::vector<float> gap_output(160, 0.0f);
    check(gap_reader.decode_rows(gap_row, nullptr, 1, gap_output.data(), 160) ==
              strata::kernels::cpu::exl3::Status::invalid_argument,
          "automatic head rejects gap/tail rows");
    const int32_t explicit_head[] = {0};
    check(gap_reader.decode_rows(gap_row, explicit_head, 1, gap_output.data(), 160) ==
              strata::kernels::cpu::exl3::Status::ok,
          "explicit head accepts gap row");
}

}  // namespace

int main() {
    std::cout << "exl3_ngram_reader_test\n";
    const auto root = std::filesystem::temp_directory_path() / "xestrata_exl3_ngram_reader_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    try {
        run_reader(root);
    } catch (const std::exception& e) {
        std::cout << "  unexpected exception: " << e.what() << '\n';
        ++failures;
    }
    std::filesystem::remove_all(root);
    std::cout << (failures == 0 ? "exl3_ngram_reader_test: all passed\n" : "exl3_ngram_reader_test: failures\n");
    return failures == 0 ? 0 : 1;
}
