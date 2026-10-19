// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// CPU-only EXL3 PLE table selftest.
#include "strata/artifact/exl3_ngram.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/ngram.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace format = strata::kernels::exl3_ngram;
namespace artifact = strata::artifact::exl3_ngram;
using Json = nlohmann::json;

namespace {

struct Tensor {
    std::string name;
    std::string dtype;
    std::vector<uint64_t> shape;
    std::vector<uint8_t> bytes;
    uint64_t offset = 0;
};

int failures = 0;

void check(bool condition, const std::string& label) {
    std::cout << "  " << label << ": " << (condition ? "OK" : "FAIL") << '\n';
    if (!condition) ++failures;
}

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
    for (int stream_bit = 0; stream_bit < format::kRowDim * bits; ++stream_bit) {
        const int value = ((stream_bit * 19 + seed * 7) ^ (stream_bit >> 3)) & 1;
        row[1 + static_cast<size_t>(stream_bit / 16)] |= static_cast<uint16_t>(value << (stream_bit & 15));
    }
    return u16_bytes(row);
}

void write_sidecar(const std::filesystem::path& path) {
    constexpr int bits = 2;
    constexpr int heads = strata::kernels::PLE_N_HEADS;
    constexpr int rows_per_shard = 16;
    constexpr int rows = rows_per_shard * 2;
    const std::string prefix = "model.layers.0.ple.ngram";
    std::vector<Tensor> tensors;

    std::vector<uint16_t> bias(static_cast<size_t>(heads) * format::kRowDim);
    for (int h = 0; h < heads; ++h)
        for (int i = 0; i < format::kRowDim; ++i)
            bias[static_cast<size_t>(h) * format::kRowDim + i] =
                strata::kernels::f16_from_f32(0.25f * static_cast<float>(h + 1) + 0.001f * static_cast<float>(i));
    tensors.push_back({prefix + ".head_bias", "F16", {heads, format::kRowDim}, u16_bytes(bias), 0});

    std::vector<int64_t> offsets(heads), sizes(heads, 1);
    for (int h = 0; h < heads; ++h) offsets[static_cast<size_t>(h)] = h;
    tensors.push_back({prefix + ".head_offsets", "I64", {heads}, i64_bytes(offsets), 5120});
    tensors.push_back({prefix + ".head_vocab_sizes", "I64", {heads}, i64_bytes(sizes), 5248});
    tensors.push_back({prefix + ".layer_multipliers", "I64", {3}, i64_bytes({23703573157769LL, 20109073645365LL, 8052911324071LL}), 5376});

    std::vector<uint8_t> shard0, shard1;
    for (int row = 0; row < rows_per_shard; ++row) {
        const auto bytes = packed_row(bits, row);
        shard0.insert(shard0.end(), bytes.begin(), bytes.end());
        const auto second = packed_row(bits, rows_per_shard + row);
        shard1.insert(shard1.end(), second.begin(), second.end());
    }
    const uint64_t shard1_offset = 6000;
    const uint64_t shard0_offset = shard1_offset + shard1.size() + 37;
    tensors.push_back({prefix + ".shard_0.trellis", "I16", {rows_per_shard, format::words_per_row(bits)}, shard0,
                       shard0_offset});
    tensors.push_back({prefix + ".shard_1.trellis", "I16", {rows_per_shard, format::words_per_row(bits)}, shard1,
                       shard1_offset});

    Json header = Json::object();
    header["__metadata__"] = {
        {"format", "exl3_ngram_trellis"}, {"version", "1"}, {"K", "2"}, {"codebook", "mul1"},
        {"row_dim", "160"}, {"rows", std::to_string(rows)}, {"shard_rows", std::to_string(rows_per_shard)},
    };
    uint64_t data_bytes = 0;
    for (const Tensor& tensor : tensors) {
        data_bytes = std::max(data_bytes, tensor.offset + static_cast<uint64_t>(tensor.bytes.size()));
        header[tensor.name] = {{"dtype", tensor.dtype}, {"shape", tensor.shape},
                               {"data_offsets", {tensor.offset, tensor.offset + tensor.bytes.size()}}};
    }
    const std::string text = header.dump();
    std::vector<uint8_t> payload(static_cast<size_t>(data_bytes), 0);
    for (const Tensor& tensor : tensors)
        std::copy(tensor.bytes.begin(), tensor.bytes.end(), payload.begin() + static_cast<ptrdiff_t>(tensor.offset));
    std::ofstream output(path, std::ios::binary);
    const uint64_t header_bytes = text.size();
    for (int b = 0; b < 8; ++b) output.put(static_cast<char>(header_bytes >> (8 * b)));
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    output.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
}

void f16_roundtrip(float* values, size_t count) {
    for (size_t i = 0; i < count; ++i) values[i] = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(values[i]));
}

void run(const std::filesystem::path& root) {
    const auto path = root / "ngram.safetensors";
    write_sidecar(path);

    artifact::Reader reader(path);
    check(reader.bits() == 2 && reader.num_heads() == 16 && reader.row_count() == 32, "EXL3 sidecar geometry");
    check(reader.row_spans().size() == 2 && reader.row_spans()[0].file_offset > reader.row_spans()[1].file_offset,
          "logical spans retain reordered physical shards");

    const int64_t ids[] = {0, 16, 1, 31};
    std::vector<uint16_t> packed;
    reader.read_rows(ids, 4, packed);
    const auto expected0 = packed_row(2, 0);
    check(packed.size() == 4 * reader.words_per_row() &&
              std::memcmp(packed.data(), expected0.data(), expected0.size()) == 0,
          "blocking row bytes match the fixture");

    const int32_t explicit_heads[] = {0, 0, 1, 15};
    std::vector<float> expected(4 * format::kRowDim);
    check(reader.decode_rows(ids, explicit_heads, 4, expected.data(), format::kRowDim) ==
              strata::kernels::cpu::exl3::Status::ok,
          "artifact reader explicit-head decode");
    f16_roundtrip(expected.data(), expected.size());

    std::vector<uint16_t> packed_copy = packed;
    std::vector<float> direct(4 * format::kRowDim, -3.0f);
    check(reader.decode_packed_rows(packed_copy.data(), 4, explicit_heads, direct.data(), format::kRowDim) ==
              strata::kernels::cpu::exl3::Status::ok,
          "decode already-fetched packed rows");
    f16_roundtrip(direct.data(), direct.size());
    check(std::memcmp(expected.data(), direct.data(), expected.size() * sizeof(float)) == 0,
          "packed decode agrees after FP16 roundtrip");
    const int32_t bad_head[] = {16, 0, 0, 0};
    check(reader.decode_packed_rows(packed_copy.data(), 4, bad_head, direct.data(), format::kRowDim) ==
              strata::kernels::cpu::exl3::Status::invalid_argument,
          "wrong explicit head is rejected");
    const int64_t bad_row[] = {32};
    bool out_of_range_rejected = false;
    try {
        (void) reader.decode_rows(bad_row, explicit_heads, 1, direct.data(), format::kRowDim);
    } catch (const std::exception&) {
        out_of_range_rejected = true;
    }
    check(out_of_range_rejected, "out-of-range row is rejected");

    strata::kernels::PleIoOptions io;
    io.max_inflight = 4;
    io.cache_rows = 32;
    io.io_thread = true;
    strata::kernels::PleTable table;
    std::string err;
    check(table.open_exl3(path.string(), err, io), "PleTable EXL3 direct open");
    check(table.is_open() && table.is_exl3() && table.exl3_bits() == 2 && table.rows() == 32,
          "PleTable exposes EXL3 metadata");
    check(table.exl3_layer_multipliers().size() == 3 && table.exl3_head_offsets().size() == 16,
          "PleTable exposes sidecar hash metadata");
    std::array<uint32_t, strata::kernels::PLE_N_HEADS> rows{};
    for (int h = 0; h < strata::kernels::PLE_N_HEADS; ++h) rows[static_cast<size_t>(h)] = static_cast<uint32_t>(h);
    check(table.issue(rows.data()), "async issue");
    std::vector<float> got(format::kRowDim * strata::kernels::PLE_N_HEADS, -7.0f);
    check(table.collect(got.data(), err), "async collect");
    std::vector<float> expected_heads(static_cast<size_t>(strata::kernels::PLE_N_HEADS) * format::kRowDim);
    const int64_t head_rows[strata::kernels::PLE_N_HEADS] = {0, 1, 2, 3, 4, 5, 6, 7,
                                                              8, 9, 10, 11, 12, 13, 14, 15};
    const int32_t head_ids[strata::kernels::PLE_N_HEADS] = {0, 1, 2, 3, 4, 5, 6, 7,
                                                             8, 9, 10, 11, 12, 13, 14, 15};
    check(reader.decode_rows(head_rows, head_ids, strata::kernels::PLE_N_HEADS, expected_heads.data(), format::kRowDim) ==
              strata::kernels::cpu::exl3::Status::ok,
          "head-row oracle decode");
    f16_roundtrip(expected_heads.data(), expected_heads.size());
    check(std::memcmp(got.data(), expected_heads.data(), got.size() * sizeof(float)) == 0,
          "PleTable collect matches FP16 oracle");

    auto invalid_rows = rows;
    invalid_rows.back() = 32;
    std::vector<float> invalid_output(got.size(), -9.0f);
    check(table.issue(invalid_rows.data()), "out-of-range collect issue");
    check(table.collect(invalid_output.data(), err), "out-of-range collect remains a successful zero-fill");
    check(std::memcmp(invalid_output.data(), expected_heads.data(), (got.size() - format::kRowDim) * sizeof(float)) == 0,
          "out-of-range collect preserves valid rows");
    bool invalid_zero = true;
    for (size_t i = got.size() - format::kRowDim; i < got.size(); ++i) invalid_zero &= invalid_output[i] == 0.0f;
    check(invalid_zero, "out-of-range collect zeros the invalid row");

    table.prefetch_rows(rows.data());
    std::vector<float> batch(static_cast<size_t>(2) * got.size(), -8.0f);
    const uint32_t batch_rows[] = {rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7],
                                   rows[8], rows[9], rows[10], rows[11], rows[12], rows[13], rows[14], rows[15]};
    check(table.gather_batch(batch_rows, 1, batch.data(), err), "prefetch gather batch");
    check(table.gather_batch(batch_rows, 1, batch.data() + got.size(), err), "repeated cached gather batch");
    std::array<uint32_t, 2 * strata::kernels::PLE_N_HEADS> two_token_rows{};
    for (int h = 0; h < strata::kernels::PLE_N_HEADS; ++h) {
        two_token_rows[static_cast<size_t>(h)] = static_cast<uint32_t>(h);
        two_token_rows[strata::kernels::PLE_N_HEADS + static_cast<size_t>(h)] =
            static_cast<uint32_t>(strata::kernels::PLE_N_HEADS + h);
    }
    std::vector<float> two_token_output(static_cast<size_t>(2) * got.size(), -8.0f);
    check(table.gather_batch(two_token_rows.data(), 2, two_token_output.data(), err),
          "two-token batch crosses physical shard and head padding");
    std::vector<float> two_token_expected(two_token_output.size());
    const int64_t two_token_ids[strata::kernels::PLE_N_HEADS * 2] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    const int32_t two_token_heads[strata::kernels::PLE_N_HEADS * 2] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    check(reader.decode_rows(two_token_ids, two_token_heads, 2 * strata::kernels::PLE_N_HEADS,
                             two_token_expected.data(), format::kRowDim) == strata::kernels::cpu::exl3::Status::ok,
          "two-token batch oracle decode");
    f16_roundtrip(two_token_expected.data(), two_token_expected.size());
    check(std::memcmp(two_token_output.data(), two_token_expected.data(), two_token_output.size() * sizeof(float)) == 0,
          "two-token batch matches explicit-head oracle");
    check(table.io_report().find("row-cache hits") != std::string::npos, "direct cache statistics available");

    strata::kernels::PleIoOptions mmap = io;
    mmap.mode = strata::kernels::PleIo::Mmap;
    check(!table.open_exl3(path.string(), err, mmap) && err.find("Direct") != std::string::npos,
          "EXL3 mmap mode is rejected explicitly");
    table.close();
}

}  // namespace

int main() {
    std::cout << "exl3_table_test\n";
    const auto root = std::filesystem::temp_directory_path() / "xestrata_exl3_table_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    try {
        run(root);
    } catch (const std::exception& e) {
        std::cout << "  unexpected exception: " << e.what() << '\n';
        ++failures;
    }
    std::filesystem::remove_all(root);
    std::cout << (failures == 0 ? "exl3_table_test: all passed\n" : "exl3_table_test: failures\n");
    return failures == 0 ? 0 : 1;
}
