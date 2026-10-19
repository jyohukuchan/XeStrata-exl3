// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/artifact/exl3_expert.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

namespace {

using Json = nlohmann::json;
using namespace strata::artifact::exl3;

std::vector<uint8_t> bf16_words(size_t count, uint16_t seed) {
    std::vector<uint8_t> result(count * 2);
    for (size_t i = 0; i < count; ++i) {
        const uint16_t value = static_cast<uint16_t>(seed + static_cast<uint16_t>(i * 13));
        result[2 * i] = static_cast<uint8_t>(value);
        result[2 * i + 1] = static_cast<uint8_t>(value >> 8);
    }
    return result;
}

std::filesystem::path synthetic_archive() {
    const auto root = std::filesystem::temp_directory_path() /
                      ("xestrata_raw_slice_" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (!std::filesystem::create_directory(root)) throw std::runtime_error("fixture directory already exists");
    std::ofstream(root / "config.json") << R"({"model_type":"raw-slice-fixture"})";
    const std::string gate_up = "mtp.layers.0.mlp.experts.gate_up_proj";
    const std::string down = "mtp.layers.0.mlp.experts.down_proj";
    const auto gate_up_bytes = bf16_words(2 * 4 * 4, 0x1000);
    const auto down_bytes = bf16_words(2 * 4 * 2, 0x3000);
    Json header = Json::object();
    uint64_t offset = 0;
    header[gate_up] = Json{{"dtype", "BF16"}, {"shape", {2, 4, 4}}, {"data_offsets", {offset, offset + gate_up_bytes.size()}}};
    offset += gate_up_bytes.size();
    header[down] = Json{{"dtype", "BF16"}, {"shape", {2, 4, 2}}, {"data_offsets", {offset, offset + down_bytes.size()}}};
    const std::string text = header.dump();
    std::ofstream output(root / "model.safetensors", std::ios::binary);
    const uint64_t header_bytes = text.size();
    for (int i = 0; i < 8; ++i) output.put(static_cast<char>(header_bytes >> (8 * i)));
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    output.write(reinterpret_cast<const char*>(gate_up_bytes.data()), static_cast<std::streamsize>(gate_up_bytes.size()));
    output.write(reinterpret_cast<const char*>(down_bytes.data()), static_cast<std::streamsize>(down_bytes.size()));
    return root;
}

bool synthetic_test() {
    const auto root = synthetic_archive();
    bool ok = true;
    try {
        Archive archive(root);
        RawExpertTensorNames names{"mtp.layers.0.mlp.experts.gate_up_proj",
                                   "mtp.layers.0.mlp.experts.down_proj", 2, 4, 2};
        RawExpertSlices slices;
        std::string err;
        ok = describe_raw_expert_slices(archive, names, 1, slices, err);
        ok = ok && slices.gate.relative_offset == 32 && slices.up.relative_offset == 48 &&
             slices.down.relative_offset == 16 && slices.gate.bytes == 16 && slices.down.bytes == 16;
        ExpertBlob blob;
        ok = ok && materialize_raw_expert_slices(archive, slices, blob, err);
        const auto view = blob.view();
        ok = ok && view.gate.is_raw() && view.up.is_raw() && view.down.is_raw();
        ok = ok && view.gate.raw.k == 4 && view.gate.raw.n == 2 && view.down.raw.k == 2 && view.down.raw.n == 4;
        ok = ok && view.gate.raw.dtype == strata::kernels::raw::DType::BF16;
        const auto* gate_bytes = static_cast<const uint8_t*>(view.gate.raw.data);
        const auto* up_bytes = static_cast<const uint8_t*>(view.up.raw.data);
        const auto* down_bytes = static_cast<const uint8_t*>(view.down.raw.data);
        ok = ok && gate_bytes != nullptr && up_bytes != nullptr && gate_bytes[0] == 0xd0 && gate_bytes[1] == 0x10 &&
             up_bytes[0] == 0x38 && up_bytes[1] == 0x11 && down_bytes != nullptr && down_bytes[0] == 0x68 &&
             down_bytes[1] == 0x30;
        std::vector<uint8_t> sentinel{0xaa, 0xbb, 0xcc};
        const auto before = sentinel;
        const TensorDescriptor* gate = archive.find_tensor(names.gate_up);
        try {
            archive.read_tensor_range(*gate, gate->bytes - 1, 2, sentinel);
            ok = false;
        } catch (const std::exception&) {
            ok = ok && sentinel == before;
        }
        ExpertBlob unchanged;
        unchanged.layout.total_bytes = 8;
        unchanged.storage.assign(1, 0x1122334455667788ull);
        const auto unchanged_storage = unchanged.storage;
        RawExpertSlices bad = slices;
        --bad.gate.bytes;
        ok = ok && !materialize_raw_expert_slices(archive, bad, unchanged, err) &&
             unchanged.storage == unchanged_storage;
        bad = slices;
        bad.down.relative_offset = 0;
        ok = ok && !materialize_raw_expert_slices(archive, bad, unchanged, err) &&
             unchanged.storage == unchanged_storage;
        if (!ok) std::cerr << "synthetic raw slice checks failed: " << err << '\n';
    } catch (const std::exception& e) {
        std::cerr << "synthetic raw slice exception: " << e.what() << '\n';
        ok = false;
    }
    std::filesystem::remove_all(root);
    return ok;
}

bool actual_test(const std::filesystem::path& path, const std::filesystem::path& dump) {
    try {
        const SafetensorsHeader header = inspect_safetensors(path);
        const TensorDescriptor* gate_up = header.find("mtp.layers.0.mlp.experts.gate_up_proj");
        std::vector<uint8_t> sentinel{0xde, 0xad, 0xbe, 0xef};
        const auto before = sentinel;
        try {
            read_tensor_range(header, *gate_up, gate_up->bytes - 1, 2, sentinel);
            std::cerr << "standalone header range accepted an out-of-bounds request\n";
            return false;
        } catch (const std::exception&) {
            if (sentinel != before) {
                std::cerr << "standalone header range changed output on failure\n";
                return false;
            }
        }
        RawExpertTensorNames names{"mtp.layers.0.mlp.experts.gate_up_proj",
                                   "mtp.layers.0.mlp.experts.down_proj", 512, 2560, 640};
        for (const int64_t expert : {int64_t(0), int64_t(511)}) {
            RawExpertSlices slices;
            std::string err;
            if (!describe_raw_expert_slices(header, names, expert, slices, err)) {
                std::cerr << "actual describe failed: " << err << '\n';
                return false;
            }
            ExpertBlob blob;
            if (!materialize_raw_expert_slices(header, slices, blob, err)) {
                std::cerr << "actual materialize failed: " << err << '\n';
                return false;
            }
            std::cout << "actual expert " << expert << ": gate=" << slices.gate.bytes << " up=" << slices.up.bytes
                      << " down=" << slices.down.bytes << '\n';
            if (!dump.empty()) {
                std::filesystem::create_directories(dump);
                const auto view = blob.view();
                const std::pair<const char*, const strata::kernels::exl3::ProjectionView*> parts[] = {
                    {"gate", &view.gate}, {"up", &view.up}, {"down", &view.down}};
                for (const auto& [role, projection] : parts) {
                    const auto* data = static_cast<const char*>(projection->raw.data);
                    std::ofstream out(dump / ("expert" + std::to_string(expert) + "_" + role + ".bin"),
                                      std::ios::binary);
                    out.write(data, static_cast<std::streamsize>(projection->raw.k * projection->raw.n * 2));
                }
            }
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "actual raw slice exception: " << e.what() << '\n';
        return false;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (!synthetic_test()) return 1;
    if (argc > 1) {
        const std::filesystem::path dump = argc > 2 ? argv[2] : std::filesystem::path{};
        if (!actual_test(argv[1], dump)) return 1;
    }
    std::cout << "raw fused expert slice checks: OK\n";
    return 0;
}
