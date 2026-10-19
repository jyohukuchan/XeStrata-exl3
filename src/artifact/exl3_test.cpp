// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU-only EXL3 archive loader selftest.  It writes tiny safetensors fixtures,
// exercises mixed sharded and legacy layouts, and checks malformed boundaries.
#include "strata/artifact/exl3.hpp"

#include "strata/kernels/f16_bits.hpp"

#include <nlohmann/json.hpp>

#include <array>
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
using Json = nlohmann::json;

namespace {

// gitleaks:allow -- EXL3 marker constants are public format values, never secrets.
constexpr const char* kPatchFile = "mtp_hyper_connection_mixer_patch.safetensors";

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

std::vector<uint8_t> repeated_words(size_t count, uint16_t value) {
    std::vector<uint8_t> result(count * 2);
    for (size_t i = 0; i < count; ++i) {
        result[2 * i] = static_cast<uint8_t>(value);
        result[2 * i + 1] = static_cast<uint8_t>(value >> 8);
    }
    return result;
}

std::vector<uint8_t> u32(uint32_t value) {
    return {static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8), static_cast<uint8_t>(value >> 16),
            static_cast<uint8_t>(value >> 24)};
}

std::vector<uint8_t> f32(float value) {
    uint32_t raw;
    std::memcpy(&raw, &value, sizeof(raw));
    return u32(raw);
}

void write_safetensors(const std::filesystem::path& path, const std::vector<std::pair<std::string, Tensor>>& tensors,
                       const std::map<std::string, std::string>& metadata = {}) {
    Json header = Json::object();
    if (!metadata.empty()) {
        Json m = Json::object();
        for (const auto& [key, value] : metadata) m[key] = value;
        header["__metadata__"] = std::move(m);
    }
    uint64_t offset = 0;
    for (const auto& [name, tensor] : tensors) {
        Json item = Json::object();
        item["dtype"] = tensor.dtype;
        item["shape"] = tensor.shape;
        item["data_offsets"] = {offset, offset + tensor.data.size()};
        header[name] = std::move(item);
        offset += tensor.data.size();
    }
    const std::string header_text = header.dump();
    std::ofstream output(path, std::ios::binary);
    const uint64_t header_size = header_text.size();
    for (int i = 0; i < 8; ++i) output.put(static_cast<char>(header_size >> (8 * i)));
    output.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    for (const auto& [name, tensor] : tensors) {
        (void) name;
        output.write(reinterpret_cast<const char*>(tensor.data.data()), static_cast<std::streamsize>(tensor.data.size()));
    }
}

Tensor trellis(int bits, uint16_t seed) {
    const size_t words_count = 8 * 8 * 16 * static_cast<size_t>(bits);
    Tensor result{"I16", {8, 8, static_cast<uint64_t>(16 * bits)}, std::vector<uint8_t>(words_count * 2)};
    for (size_t i = 0; i < words_count; ++i) {
        const uint16_t value = static_cast<uint16_t>(seed + i * 17);
        result.data[2 * i] = static_cast<uint8_t>(value);
        result.data[2 * i + 1] = static_cast<uint8_t>(value >> 8);
    }
    return result;
}

Tensor f16_vector(size_t count, uint16_t bits = 0x3c00) {
    return Tensor{"F16", {count}, repeated_words(count, bits)};
}

Tensor legacy_vector(size_t count_words) {
    return Tensor{"I16", {count_words}, repeated_words(count_words, 0x8001)};
}

void make_mixed_fixture(const std::filesystem::path& root) {
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json") << R"({"model_type":"synthetic-exl3","hidden_size":128})";

    std::vector<std::pair<std::string, Tensor>> shard0;
    std::vector<std::pair<std::string, Tensor>> shard1;
    const auto add_linear = [](std::vector<std::pair<std::string, Tensor>>& out, const std::string& base, int bits,
                               bool legacy, bool marker_mcg, bool marker_mul1) {
        out.emplace_back(base + ".trellis", trellis(bits, static_cast<uint16_t>(bits * 31)));
        if (legacy) {
            out.emplace_back(base + ".su", legacy_vector(8));
            out.emplace_back(base + ".sv", legacy_vector(8));
        } else {
            out.emplace_back(base + ".suh", f16_vector(128, 0x3c00));  // codespell:ignore suh
            out.emplace_back(base + ".svh", f16_vector(128, 0x3c00));
        }
        out.emplace_back(base + ".bias", Tensor{"F32", {128}, [&] {
            std::vector<uint8_t> bytes;
            for (int i = 0; i < 128; ++i) {
                const auto value = f32(0.25f + static_cast<float>(i) * 0.001f);
                bytes.insert(bytes.end(), value.begin(), value.end());
            }
            return bytes;
        }()});
        if (marker_mcg) out.emplace_back(base + ".mcg", Tensor{"U32", {}, u32(0xCBAC1FEDu)});
        if (marker_mul1) out.emplace_back(base + ".mul1", Tensor{"I32", {1}, u32(0x83DCD12Du)});
    };
    shard0.emplace_back("model.empty.raw", Tensor{"F32", {0}, {}});
    add_linear(shard0, "model.layers.0.mlp.experts.0.gate_proj", 2, false, false, false);
    add_linear(shard0, "model.layers.0.mlp.experts.0.up_proj", 3, false, true, false);
    add_linear(shard1, "model.layers.0.mlp.experts.0.down_proj", 4, true, false, false);
    add_linear(shard1, "model.layers.1.mlp.experts.0.gate_proj", 5, false, false, true);
    shard0.emplace_back("model.embed_tokens.weight", Tensor{"F32", {4, 4}, std::vector<uint8_t>(64, 0x11)});
    write_safetensors(root / "model-00001-of-00002.safetensors", shard0);
    write_safetensors(root / "model-00002-of-00002.safetensors", shard1);

    Json index;
    index["metadata"]["total_size"] = 1;  // deliberately stale, advisory only
    index["weight_map"] = Json::object();
    for (const auto& [name, tensor] : shard0) index["weight_map"][name] = "model-00001-of-00002.safetensors";
    for (const auto& [name, tensor] : shard1) index["weight_map"][name] = "model-00002-of-00002.safetensors";
    std::ofstream(root / "model.safetensors.index.json") << index.dump();

    // The known patch replaces one base bias and is applied after the index.
    write_safetensors(root / kPatchFile,
                      {{"model.layers.0.mlp.experts.0.gate_proj.bias", Tensor{"F16", {128}, repeated_words(128, 0x4000)}}});
    write_safetensors(root / "ngram_embedding.safetensors",
                      {{"model.ple.ngram.trellis", Tensor{"I16", {2, 16}, std::vector<uint8_t>(64, 0x22)}}},
                      {{"format", "exl3_ngram_trellis"}, {"codebook", "mul1"}});
}

template <class Fn> bool rejected(Fn&& function) {
    try {
        function();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

void run_mixed() {
    const auto root = std::filesystem::temp_directory_path() / "xestrata_exl3_archive_selftest_mixed";
    std::filesystem::remove_all(root);
    make_mixed_fixture(root);
    artifact::Archive archive(root);
    check(archive.linears().size() == 4, "four mixed linears across two shards");
    check(archive.tensors().size() > archive.linears().size(), "unquantized tensors remain visible");
    const auto* empty_tensor = archive.find_tensor("model.empty.raw");
    check(empty_tensor != nullptr && empty_tensor->shape == std::vector<uint64_t>{0} && empty_tensor->bytes == 0,
          "empty raw tensor at a shared offset is preserved");
    check(archive.has_declared_total_size() && archive.declared_total_size() == 1, "stale total_size is advisory");
    check(archive.sidecars().size() == 2, "known patch and ngram sidecars are reported");
    bool saw_ngram = false, saw_patch = false;
    for (const auto& sidecar : archive.sidecars()) {
        saw_ngram |= sidecar.role == artifact::SidecarRole::specialized_ngram_codec;
        saw_patch |= sidecar.role == artifact::SidecarRole::known_override_patch && sidecar.overrides_base;
    }
    check(saw_ngram && saw_patch, "sidecar roles are explicit");
    bool bits[6] = {};
    bool saw_legacy = false, saw_cb1 = false, saw_cb2 = false;
    for (const auto& linear : archive.linears()) {
        if (linear.bits >= 0 && linear.bits < 6) bits[linear.bits] = true;
        saw_legacy |= linear.su.has_value() && linear.sv.has_value();
        saw_cb1 |= linear.codebook == 1;
        saw_cb2 |= linear.codebook == 2;
        artifact::MaterializedMatrix materialized;
        archive.materialize(linear, materialized);
        check(materialized.trellis.size() == static_cast<size_t>(linear.k / 16 * (linear.n / 16) * 16 * linear.bits),
              "materialize trellis for " + linear.name);
        check(materialized.suh.size() == static_cast<size_t>(linear.k) && materialized.svh.size() == static_cast<size_t>(linear.n),
              "materialize scales for " + linear.name);
        const auto view = materialized.view();
        check(view.trellis != nullptr && view.suh != nullptr && view.svh != nullptr, "MatrixView pointers for " + linear.name);
        if (linear.name.find("gate_proj") != std::string::npos && linear.name.find("layers.0") != std::string::npos)
            check(materialized.bias.front() == 0x4000, "known patch takes precedence for bias");
    }
    check(bits[2] && bits[3] && bits[4] && bits[5], "2-5 bpw mixed precision is preserved");
    check(saw_legacy && saw_cb1 && saw_cb2, "legacy scales and both marker codebooks are preserved");
    std::vector<uint8_t> raw;
    const auto* embedding = archive.find_tensor("model.embed_tokens.weight");
    check(embedding != nullptr, "raw unquantized tensor descriptor is findable");
    if (embedding != nullptr) {
        archive.read_tensor(*embedding, raw);
        check(raw.size() == embedding->bytes, "one requested raw tensor reads exactly");
    }
    std::filesystem::remove_all(root);
}

void run_malformed() {
    const auto base = std::filesystem::temp_directory_path() / "xestrata_exl3_archive_selftest_bad";
    std::filesystem::remove_all(base);
    auto make_base = [&](const std::string& config) {
        std::filesystem::create_directories(base);
        std::ofstream(base / "config.json") << config;
    };
    make_base(R"({"model_type":"bad"})");
    write_safetensors(base / "model.safetensors",
                      {{"x.trellis", trellis(2, 1)}, {"x.suh", f16_vector(128)}, {"x.svh", f16_vector(128)}});
    check(!rejected([&] { artifact::Archive archive(base); }) && artifact::Archive(base).linears().front().codebook == 0,
          "missing marker selects the legacy cb0 codebook");
    std::filesystem::remove_all(base);

    make_base(R"({"model_type":"bad"})");
    write_safetensors(base / "model.safetensors",
                      {{"x.trellis", trellis(2, 1)}, {"x.suh", f16_vector(128)}, {"x.svh", f16_vector(128)},
                       {"x.mul1", Tensor{"U32", {}, u32(0x12345678u)}}});
    check(rejected([&] { artifact::Archive archive(base); }), "unknown marker value is rejected");
    std::filesystem::remove_all(base);

    make_base(R"({"model_type":"bad"})");
    write_safetensors(base / "model.safetensors",
                      {{"x.trellis", trellis(2, 1)}, {"x.suh", f16_vector(128)}, {"x.svh", f16_vector(128)},
                       {"x.mcg", Tensor{"U32", {}, u32(0xCBAC1FEDu)}}});
    {
        artifact::Archive archive(base);
        artifact::LinearDescriptor altered = archive.linears().front();
        altered.bits = 5;
        artifact::MaterializedMatrix materialized;
        check(rejected([&] { archive.materialize(altered, materialized); }), "altered linear descriptor is rejected");
    }
    std::filesystem::remove_all(base);

    make_base("// comments are not safetensors JSON\n{\"model_type\":\"bad\"}");
    write_safetensors(base / "model.safetensors",
                      {{"x.trellis", trellis(2, 1)}, {"x.suh", f16_vector(128)}, {"x.svh", f16_vector(128)},
                       {"x.mcg", Tensor{"U32", {}, u32(0xCBAC1FEDu)}}});
    check(rejected([&] { artifact::Archive archive(base); }), "JSON comments are rejected");
    std::filesystem::remove_all(base);

    make_base(R"({"model_type":"bad"})");
    write_safetensors(base / "model.safetensors",
                      {{"x.trellis", trellis(2, 1)}, {"x.suh", f16_vector(128)}, {"x.svh", f16_vector(128)},
                       {"x.mcg", Tensor{"U32", {}, u32(0xCBAC1FEDu)}}});
    const auto truncated_size = std::filesystem::file_size(base / "model.safetensors");
    std::filesystem::resize_file(base / "model.safetensors", truncated_size - 1);
    check(rejected([&] { artifact::Archive archive(base); }), "truncated tensor payload is rejected");
    std::filesystem::remove_all(base);

    make_base(R"({"model_type":"bad"})");
    write_safetensors(base / "model.safetensors",
                      {{"x.trellis", trellis(2, 1)}, {"x.suh", f16_vector(128)}, {"x.svh", f16_vector(128)},
                       {"x.mcg", Tensor{"U32", {}, u32(0xCBAC1FEDu)}}});
    write_safetensors(base / "unrecognized_override.safetensors",
                      {{"x.suh", f16_vector(128)}});
    check(rejected([&] { artifact::Archive archive(base); }), "unrecognized duplicate sidecar is rejected");
    std::filesystem::remove_all(base);

    make_base(R"({"model_type":"bad"})");
    write_safetensors(base / "model.safetensors",
                      {{"x.trellis", trellis(9, 1)}, {"x.suh", f16_vector(128)}, {"x.svh", f16_vector(128)}});
    check(rejected([&] { artifact::Archive archive(base); }), "bit width above 8 is rejected");
    std::filesystem::remove_all(base);

    make_base(R"({"model_type":"bad","duplicate":1,"duplicate":2})");
    write_safetensors(base / "model.safetensors", {{"x", Tensor{"F32", {1}, f32(1.0f)}}});
    check(rejected([&] { artifact::Archive archive(base); }), "duplicate config JSON key is rejected");
    std::filesystem::remove_all(base);
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << "exl3_archive_test\n";
    if (argc == 2) {
        try {
            const artifact::Archive archive(argv[1]);
            std::cout << "  indexed tensors: " << archive.indexed_tensors().size() << '\n';
            std::cout << "  effective tensors: " << archive.tensors().size() << '\n';
            std::cout << "  linears: " << archive.linears().size() << '\n';
            std::cout << "  sidecars: " << archive.sidecars().size() << '\n';
            std::cout << "  indexed payload bytes: " << archive.indexed_payload_bytes() << '\n';
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "exl3_archive_test: " << e.what() << '\n';
            return 2;
        }
    }
    run_mixed();
    run_malformed();
    std::cout << (failures == 0 ? "exl3_archive_test: all passed\n" : "exl3_archive_test: failures\n");
    return failures == 0 ? 0 : 1;
}
