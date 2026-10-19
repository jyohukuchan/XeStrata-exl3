// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU-only EXL3 expert blob layout selftest.
#include "strata/artifact/exl3_expert.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace artifact = strata::artifact::exl3;
namespace format = strata::kernels::exl3;
using Json = nlohmann::json;

namespace {

struct Tensor {
    std::string dtype;
    std::vector<uint64_t> shape;
    std::vector<uint8_t> data;
};

int failures = 0;

void check(bool condition, const std::string& label) {
    std::cout << "  " << label << ": " << (condition ? "OK" : "FAIL") << '\n';
    if (!condition) ++failures;
}

std::vector<uint8_t> u32(uint32_t value) {
    return {static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8), static_cast<uint8_t>(value >> 16),
            static_cast<uint8_t>(value >> 24)};
}

std::vector<uint8_t> words(size_t count, uint16_t seed) {
    std::vector<uint8_t> result(count * 2);
    for (size_t i = 0; i < count; ++i) {
        const uint16_t value = static_cast<uint16_t>(seed + i * 13);
        result[2 * i] = static_cast<uint8_t>(value);
        result[2 * i + 1] = static_cast<uint8_t>(value >> 8);
    }
    return result;
}

std::vector<uint8_t> f32_vector(size_t count, float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    std::vector<uint8_t> result(count * 4);
    for (size_t i = 0; i < count; ++i) {
        result[4 * i] = static_cast<uint8_t>(bits);
        result[4 * i + 1] = static_cast<uint8_t>(bits >> 8);
        result[4 * i + 2] = static_cast<uint8_t>(bits >> 16);
        result[4 * i + 3] = static_cast<uint8_t>(bits >> 24);
    }
    return result;
}

void write_safetensors(const std::filesystem::path& path, const std::vector<std::pair<std::string, Tensor>>& tensors) {
    Json header = Json::object();
    uint64_t offset = 0;
    for (const auto& [name, tensor] : tensors) {
        header[name] = Json{{"dtype", tensor.dtype}, {"shape", tensor.shape}, {"data_offsets", {offset, offset + tensor.data.size()}}};
        offset += tensor.data.size();
    }
    const std::string text = header.dump();
    std::ofstream output(path, std::ios::binary);
    const uint64_t size = text.size();
    for (int i = 0; i < 8; ++i) output.put(static_cast<char>(size >> (8 * i)));
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    for (const auto& [name, tensor] : tensors) {
        (void) name;
        output.write(reinterpret_cast<const char*>(tensor.data.data()), static_cast<std::streamsize>(tensor.data.size()));
    }
}

void add_linear(std::vector<std::pair<std::string, Tensor>>& tensors, const std::string& base, int k, int n, int bits,
                int codebook, bool legacy_scales, bool bias, bool f32_bias) {
    tensors.emplace_back(base + ".trellis", Tensor{"I16", {(uint64_t) k / 16, (uint64_t) n / 16, (uint64_t) 16 * bits},
                                                       words((size_t) (k / 16) * (size_t) (n / 16) * 16 * bits,
                                                             static_cast<uint16_t>(bits * 23))});
    if (legacy_scales) {
        tensors.emplace_back(base + ".su", Tensor{"I16", {(uint64_t) k / 16}, words((size_t) k / 16, 0x8001)});
        tensors.emplace_back(base + ".sv", Tensor{"I16", {(uint64_t) n / 16}, words((size_t) n / 16, 0x0001)});
    } else {
        tensors.emplace_back(base + ".suh", Tensor{"F16", {(uint64_t) k}, words((size_t) k, 0x3c00)});  // codespell:ignore suh
        tensors.emplace_back(base + ".svh", Tensor{"F16", {(uint64_t) n}, words((size_t) n, 0x3c00)});
    }
    if (bias) {
        if (f32_bias)
            tensors.emplace_back(base + ".bias", Tensor{"F32", {(uint64_t) n}, f32_vector((size_t) n, 0.125f)});
        else
            tensors.emplace_back(base + ".bias", Tensor{"F16", {(uint64_t) n}, words((size_t) n, 0x3400)});
    }
    if (codebook == 1) tensors.emplace_back(base + ".mcg", Tensor{"U32", {}, u32(0xCBAC1FEDu)});
    if (codebook == 2) tensors.emplace_back(base + ".mul1", Tensor{"I32", {1}, u32(0x83DCD12Du)});
}

void make_archive(const std::filesystem::path& root, int hidden, int ff, const int bits[3], const int codebooks[3],
                  const bool legacy[3], const bool biases[3], const bool f32_bias[3], bool malformed_down = false) {
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json") << R"({"model_type":"synthetic-exl3-expert"})";
    std::vector<std::pair<std::string, Tensor>> tensors;
    const std::string names[3] = {"expert.gate", "expert.up", "expert.down"};
    add_linear(tensors, names[0], hidden, ff, bits[0], codebooks[0], legacy[0], biases[0], f32_bias[0]);
    add_linear(tensors, names[1], hidden, ff, bits[1], codebooks[1], legacy[1], biases[1], f32_bias[1]);
    add_linear(tensors, names[2], malformed_down ? hidden : ff, hidden, bits[2], codebooks[2], legacy[2], biases[2], f32_bias[2]);
    write_safetensors(root / "model.safetensors", tensors);
}

const artifact::LinearDescriptor* linear(const artifact::Archive& archive, const std::string& name) {
    for (const auto& descriptor : archive.linears())
        if (descriptor.name == name) return &descriptor;
    return nullptr;
}

void compare_component(const artifact::ExpertBlob& blob, const format::ExpertComponentLayout& layout,
                       const std::vector<uint16_t>& expected, const std::string& label) {
    check(layout.bytes == expected.size() * sizeof(uint16_t), label + " byte length");
    check(expected.empty() || std::memcmp(blob.data() + layout.offset, expected.data(), layout.bytes) == 0,
          label + " byte exact");
}

void run_case(const std::filesystem::path& root, const artifact::ExpertTensorNames& names, size_t& total_bytes) {
    artifact::Archive archive(root);
    format::ExpertBlobLayout metadata;
    std::string metadata_error;
    check(artifact::describe_expert(archive, names, metadata, metadata_error), "metadata-only layout query: " + metadata_error);
    artifact::ExpertBlob blob;
    std::string err;
    check(artifact::materialize_expert(archive, names, blob, err), "materialize expert: " + err);
    if (blob.data() == nullptr) return;
    total_bytes = blob.bytes();
    check(blob.layout.alignment == 2 && blob.bytes() == blob.layout.total_bytes, "typed storage size and alignment");
    check(metadata.total_bytes == blob.layout.total_bytes && metadata.gate.trellis.bytes == blob.layout.gate.trellis.bytes &&
              metadata.up.trellis.bytes == blob.layout.up.trellis.bytes && metadata.down.trellis.bytes == blob.layout.down.trellis.bytes,
          "metadata/materialized layout agreement");
    const auto* gate = linear(archive, names.gate);
    const auto* up = linear(archive, names.up);
    const auto* down = linear(archive, names.down);
    artifact::MaterializedMatrix gate_data, up_data, down_data;
    archive.materialize(*gate, gate_data);
    archive.materialize(*up, up_data);
    archive.materialize(*down, down_data);
    compare_component(blob, blob.layout.gate.trellis, gate_data.trellis, "gate trellis");
    compare_component(blob, blob.layout.gate.suh, gate_data.suh, "gate suh");
    compare_component(blob, blob.layout.gate.svh, gate_data.svh, "gate svh");
    compare_component(blob, blob.layout.gate.bias, gate_data.bias, "gate bias");
    compare_component(blob, blob.layout.up.trellis, up_data.trellis, "up trellis");
    compare_component(blob, blob.layout.up.suh, up_data.suh, "up suh");
    compare_component(blob, blob.layout.up.svh, up_data.svh, "up svh");
    compare_component(blob, blob.layout.up.bias, up_data.bias, "up bias");
    compare_component(blob, blob.layout.down.trellis, down_data.trellis, "down trellis");
    compare_component(blob, blob.layout.down.suh, down_data.suh, "down suh");
    compare_component(blob, blob.layout.down.svh, down_data.svh, "down svh");
    compare_component(blob, blob.layout.down.bias, down_data.bias, "down bias");
    const auto view = blob.view();
    check(view.gate.trellis != nullptr && view.up.trellis != nullptr && view.down.trellis != nullptr,
          "rebased view trellis pointers");
    check(view.gate.suh != nullptr && view.up.suh != nullptr && view.down.suh != nullptr,
          "rebased view scale pointers");
    std::vector<uint16_t> copied = blob.words;
    const auto copied_view = format::expert_view(blob.layout, reinterpret_cast<const uint8_t*>(copied.data()));
    check(copied_view.gate.trellis == reinterpret_cast<const uint16_t*>(copied.data() + blob.layout.gate.trellis.offset),
          "copy/rebase gate pointer");
    check(copied_view.down.trellis != view.down.trellis, "copy/rebase changes base pointer");
}

}  // namespace

int main() {
    const auto base = std::filesystem::temp_directory_path() / "xestrata_exl3_expert_blob_test";
    std::filesystem::remove_all(base);
    const artifact::ExpertTensorNames names{"expert.gate", "expert.up", "expert.down"};
    const int bits_a[3] = {2, 3, 5}, cb_a[3] = {0, 1, 2};
    const bool legacy_a[3] = {false, false, true}, bias_a[3] = {true, false, true}, f32_a[3] = {false, false, true};
    make_archive(base / "a", 128, 256, bits_a, cb_a, legacy_a, bias_a, f32_a);
    size_t bytes_a = 0;
    run_case(base / "a", names, bytes_a);
    artifact::Archive archive_a(base / "a");
    artifact::ExpertBlob blob_a;
    std::string err;
    check(artifact::materialize_expert(archive_a, names, blob_a, err), "legacy scales and optional bias");
    check(blob_a.view().up.bias == nullptr, "optional bias remains absent");

    const int bits_b[3] = {5, 2, 2}, cb_b[3] = {2, 0, 1};
    const bool legacy_b[3] = {true, false, false}, bias_b[3] = {false, true, false}, f32_b[3] = {false, false, false};
    make_archive(base / "b", 128, 256, bits_b, cb_b, legacy_b, bias_b, f32_b);
    size_t bytes_b = 0;
    run_case(base / "b", names, bytes_b);
    check(bytes_a != bytes_b, "different per-projection bit allocation changes blob bytes");

    make_archive(base / "bad", 128, 256, bits_a, cb_a, legacy_a, bias_a, f32_a, true);
    artifact::Archive bad_archive(base / "bad");
    artifact::ExpertBlob bad_blob;
    check(!artifact::materialize_expert(bad_archive, names, bad_blob, err), "malformed hidden-ff-hidden shape rejected");

    std::filesystem::remove_all(base);
    std::cout << "exl3_expert_blob_test: " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
