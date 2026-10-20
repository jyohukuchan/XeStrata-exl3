// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// tests/core/exl3_source_capacity_test.cpp - the EXL3 source's capacity guard: it fails closed and evicts.
#include "strata/core/exl3_expert_source.hpp"

#include "strata/core/exl3_prepare_profile.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Json = nlohmann::json;
using Tensor = std::tuple<std::string, std::string, std::vector<uint64_t>, std::vector<uint8_t>>;

constexpr uint64_t GiB = 1ull << 30;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

struct TempDirectory {
    fs::path path;

    TempDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("strata-exl3-source-capacity-test-" + std::to_string(stamp));
        fs::create_directories(path);
    }

    ~TempDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

// The fixture of the host's RAM: a meminfo, a /proc/self/cgroup and the cgroup tree under it.
struct Fixture {
    TempDirectory dir;

    void put(const std::string& file, const std::string& text) {
        const fs::path path = dir.path / file;
        fs::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::trunc);
        out << text;
        require(bool(out), "could not write the capacity fixture");
    }

    void put_bytes(const std::string& file, uint64_t bytes) { put(file, std::to_string(bytes) + "\n"); }

    void remove(const std::string& file) {
        std::error_code ignored;
        fs::remove(dir.path / file, ignored);
    }

    std::string meminfo() const { return (dir.path / "meminfo").string(); }
    std::string self_cgroup() const { return (dir.path / "self_cgroup").string(); }
    std::string cgroup_root() const { return (dir.path / "cgroup").string(); }
};

// The cgroup tree of the fixture: the root has no memory.max (as the host's root cgroup), /a is unlimited and
// /a/b carries the limit.  The room under that limit is what the source sees.
void build_tree(Fixture& fixture, uint64_t limit, uint64_t usage) {
    fixture.put("cgroup/cgroup.controllers", "memory\n");
    fixture.put("cgroup/a/memory.max", "max\n");
    fixture.put_bytes("cgroup/a/memory.current", 0);
    fixture.put_bytes("cgroup/a/b/memory.max", limit);
    fixture.put_bytes("cgroup/a/b/memory.current", usage);
    fixture.put("cgroup/a/b/memory.stat", "anon 0\ninactive_file 0\nfile_dirty 0\nfile_writeback 0\n");
}

std::vector<uint8_t> words(size_t n, uint16_t seed) {
    std::vector<uint8_t> out(n * 2);
    for (size_t i = 0; i < n; ++i) {
        const uint16_t v = static_cast<uint16_t>(seed + i * 7);
        out[2 * i] = static_cast<uint8_t>(v);
        out[2 * i + 1] = static_cast<uint8_t>(v >> 8);
    }
    return out;
}

std::vector<uint8_t> u32(uint32_t v) {
    return {static_cast<uint8_t>(v), static_cast<uint8_t>(v >> 8), static_cast<uint8_t>(v >> 16),
            static_cast<uint8_t>(v >> 24)};
}

void add_linear(std::vector<Tensor>& tensors, const std::string& name, int bits, int codebook) {
    tensors.emplace_back(name + ".trellis", "I16", std::vector<uint64_t>{8, 8, static_cast<uint64_t>(16 * bits)},
                         words(static_cast<size_t>(8 * 8 * 16 * bits), static_cast<uint16_t>(bits * 19)));
    tensors.emplace_back(name + ".suh", "F16", std::vector<uint64_t>{128}, words(128, 0x3c00));  // codespell:ignore suh
    tensors.emplace_back(name + ".svh", "F16", std::vector<uint64_t>{128}, words(128, 0x3c00));
    if (codebook == 1) tensors.emplace_back(name + ".mcg", "U32", std::vector<uint64_t>{}, u32(0xCBAC1FEDu));
    if (codebook == 2) tensors.emplace_back(name + ".mul1", "I32", std::vector<uint64_t>{1}, u32(0x83DCD12Du));
}

// A small EXL3 archive: four layers of two experts, so that a window can move away from what it holds.
void write_archive(const fs::path& root, std::vector<strata::artifact::exl3::ExpertTensorNames>& names) {
    fs::create_directories(root);
    std::ofstream(root / "config.json") << R"({"model_type":"synthetic-exl3-capacity"})";
    std::vector<Tensor> tensors;
    for (int layer = 0; layer < 4; ++layer)
        for (int expert = 0; expert < 2; ++expert) {
            const std::string base = "layers." + std::to_string(layer) + ".experts." + std::to_string(expert);
            names.push_back({base + ".gate", base + ".up", base + ".down"});
            const int bits = 2 + ((layer * 2 + expert) % 4);
            add_linear(tensors, base + ".gate", bits, layer % 3);
            add_linear(tensors, base + ".up", 2 + ((bits + 1) % 4), (layer + expert + 1) % 3);
            add_linear(tensors, base + ".down", 2 + ((bits + 2) % 4), (layer + expert + 2) % 3);
        }
    Json header = Json::object();
    uint64_t offset = 0;
    for (const auto& [name, dtype, shape, data] : tensors) {
        header[name] = Json{{"dtype", dtype}, {"shape", shape}, {"data_offsets", {offset, offset + data.size()}}};
        offset += data.size();
    }
    const std::string text = header.dump();
    std::ofstream output(root / "model.safetensors", std::ios::binary);
    for (int i = 0; i < 8; ++i) output.put(static_cast<char>(text.size() >> (8 * i)));
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    for (const auto& [name, dtype, shape, data] : tensors) {
        (void) name; (void) dtype; (void) shape;
        output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
}

// The window of a layer, with the guard's counters around it.
void window(strata::core::Exl3ExpertSource& source, int64_t layer, const int32_t* ids, int64_t k) {
    strata::core::detail::Exl3PrepareSample sample;
    source.begin_layer(layer, ids, k);
}

}  // namespace

int main() {
    // The guard's own counters: the number of readings of the host's room, and the readings that failed.
#if defined(_WIN32)
    _putenv_s("STRATA_EXL3_PREPARE_DETAIL", "1");
#else
    setenv("STRATA_EXL3_PREPARE_DETAIL", "1", 1);
#endif
    try {
        auto* detail = strata::core::detail::exl3_prepare_detail();
        require(detail != nullptr, "the prepare detail counters are not enabled");

        Fixture fixture;
        fixture.put("meminfo", "MemTotal: 999999 kB\nMemAvailable: 4194304 kB\n");
        fixture.put("self_cgroup", "0::/a/b\n");
        build_tree(fixture, 4 * GiB, 0);

        TempDirectory archive_root;
        std::vector<strata::artifact::exl3::ExpertTensorNames> names;
        write_archive(archive_root.path, names);
        auto archive = std::make_shared<strata::artifact::exl3::Archive>(archive_root.path);

        strata::core::Exl3ExpertSource::CachePolicy policy;
        policy.headroom_bytes = 0;
        strata::core::Exl3ExpertSource source;
        source.set_capacity_files({fixture.meminfo(), fixture.self_cgroup(), fixture.cgroup_root()});
        std::string err;
        require(source.open(archive, 4, 2, names, err, policy), "the source did not open: " + err);
        require(source.cache_stats().capacity_bytes > 0, "the source opened without a budget");

        const int32_t both[] = {0, 1};
        const int32_t one[] = {0};
        // The first window reads the host's room and admits what fits.
        uint64_t calls = detail->guard_calls;
        window(source, 0, both, 2);
        require(detail->guard_calls == calls + 1, "the first window did not read the host's room");
        require(detail->guard_failures == 0, "the first window failed to read the host's room");
        const auto admitted = source.cache_stats();
        require(admitted.resident_bytes > 0 && admitted.fallbacks == 0 && admitted.evictions == 0,
                "the first window did not admit what fits");

        // The next layer continues the pass over the layers: the room is not read again.
        calls = detail->guard_calls;
        window(source, 1, one, 1);
        require(detail->guard_calls == calls, "the window that continued the pass read the host's room");
        require(source.cache_stats().resident_bytes >= admitted.resident_bytes,
                "the window that continued the pass lost what was resident");

        // The room shrinks under the source: the next pass reads it again, and the source gives back what
        // it holds and admits nothing while the room is gone.
        build_tree(fixture, 4 * GiB, 4 * GiB);
        calls = detail->guard_calls;
        window(source, 3, one, 1);
        require(detail->guard_calls == calls + 1, "the new pass over the layers did not read the host's room");
        const auto shrunk = source.cache_stats();
        require(shrunk.resident_bytes == 0 && shrunk.evictions > 0 && shrunk.fallbacks > 0 &&
                    shrunk.transient_bytes > 0,
                "the source kept what it held while the host's room was gone");
        require(source.blob(3, 0) != nullptr, "the window that fits nothing does not publish its blob");

        // The host's RAM cannot be read: the source admits nothing, and it keeps admitting nothing.
        fixture.put("meminfo", "MemTotal: 999999 kB\nMemAvailable: 4194304 kB\n");
        build_tree(fixture, 4 * GiB, 0);
        strata::core::Exl3ExpertSource closed;
        closed.set_capacity_files({fixture.meminfo(), fixture.self_cgroup(), fixture.cgroup_root()});
        require(closed.open(archive, 4, 2, names, err, policy), "the second source did not open: " + err);
        require(closed.cache_stats().capacity_bytes > 0, "the second source opened without a budget");
        fixture.remove("meminfo");
        const uint64_t failures = detail->guard_failures;
        window(closed, 0, both, 2);
        require(detail->guard_failures > failures, "the unreadable host's room did not fail the guard");
        const auto refused = closed.cache_stats();
        require(refused.resident_bytes == 0 && refused.fallbacks > 0 && refused.transient_bytes > 0,
                "the source admitted a blob while the host's RAM was unknown");
        window(closed, 1, one, 1);
        const auto still_refused = closed.cache_stats();
        require(still_refused.resident_bytes == 0 && still_refused.fallbacks > refused.fallbacks,
                "the source admitted a blob after the guard failed");

        std::cout << "exl3_source_capacity_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "exl3_source_capacity_test: " << error.what() << '\n';
        return 1;
    }
}
