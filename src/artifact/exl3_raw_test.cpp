// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/artifact/exl3_raw.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>

namespace exl3 = strata::artifact::exl3;

std::vector<uint8_t> u16(const std::vector<uint16_t>& values) {
    std::vector<uint8_t> bytes;
    for (uint16_t value : values) {
        bytes.push_back(static_cast<uint8_t>(value));
        bytes.push_back(static_cast<uint8_t>(value >> 8));
    }
    return bytes;
}

std::vector<uint8_t> f32(const std::vector<float>& values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

int main() {
    int failures = 0;
    std::string err;
    auto check = [&](bool ok, const char* name) {
        std::printf("%s: %s\n", name, ok ? "OK" : "FAIL");
        failures += !ok;
    };
    const exl3::RawTensor weight{"F16", {2, 3}, u16({0x3c00, 0x4000, 0x4200, 0xbc00, 0xc000, 0xc200})};
    exl3::RawTensor mapped;
    check(exl3::remap_raw(weight, {{1, 0}, {2, 0, 1}}, exl3::ValueTransform::identity, mapped, err) &&
          mapped.dtype == "F16" && mapped.shape == weight.shape &&
          mapped.data == u16({0xc200, 0xbc00, 0xc000, 0x4200, 0x3c00, 0x4000}), "raw axes preserve exact F16 bits");
    exl3::RawTensor alias = weight;
    check(exl3::remap_raw(alias, {{1}, {}}, exl3::ValueTransform::identity, alias, err) &&
          alias.shape == std::vector<uint64_t>{1, 3} && alias.data == u16({0xbc00, 0xc000, 0xc200}),
          "raw output subset permits source-destination alias");
    const exl3::RawTensor sentinel = mapped;
    check(!exl3::remap_raw(weight, {{0, 0}, {}}, exl3::ValueTransform::identity, mapped, err) &&
          mapped.data == sentinel.data && mapped.shape == sentinel.shape, "invalid permutation leaves output unchanged");
    exl3::RawTensor truncated = weight;
    truncated.data.pop_back();
    check(!exl3::remap_raw(truncated, {}, exl3::ValueTransform::identity, mapped, err) &&
          mapped.data == sentinel.data, "truncated raw payload rejected before indexing");

    const exl3::RawTensor a{"F32", {2}, f32({0.0f, 1.0f})};
    std::vector<float> values;
    check(exl3::remap_raw(a, {{1, 0}}, exl3::ValueTransform::negative_exp, mapped, err) &&
          exl3::raw_to_f32(mapped, values, err) && values.size() == 2 &&
          std::fabs(values[0] + 2.718281828f) < 1e-6f && values[1] == -1.0f, "A-log negative exponential follows axis mapping");

    const auto root = std::filesystem::temp_directory_path() /
        ("xestrata_exl3_raw_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    bool created = false;
    try {
        if (!std::filesystem::create_directory(root)) throw std::runtime_error("temporary directory exists");
        created = true;
        std::ofstream(root / "config.json") << R"({"model_type":"synthetic-raw"})";
        const std::string header = R"({"norm.weight":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]}})";
        const auto data = u16({0x3f00, 0xbf80, 0x4000, 0x3e80});
        {
            std::ofstream out(root / "model.safetensors", std::ios::binary);
            const uint64_t length = header.size();
            for (int i = 0; i < 8; ++i) out.put(static_cast<char>(length >> (i * 8)));
            out.write(header.data(), static_cast<std::streamsize>(header.size()));
            out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        }
        exl3::Archive archive(root);
        const auto* descriptor = archive.find_tensor("norm.weight");
        check(descriptor != nullptr && exl3::materialize_raw(archive, *descriptor, {{3, 1}},
              exl3::ValueTransform::add_one, mapped, err) && mapped.dtype == "F32" &&
              mapped.shape == std::vector<uint64_t>{2} && exl3::raw_to_f32(mapped, values, err) &&
              values == std::vector<float>{1.25f, 0.0f}, "archive BF16 norm becomes F32 only at add-one transform");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        ++failures;
    }
    if (created) std::filesystem::remove_all(root);
    std::printf("exl3_raw_test: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
