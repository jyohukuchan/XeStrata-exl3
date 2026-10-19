// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// CPU-only EXL3 ExpertSource adapter selftest.
#include "strata/core/exl3_expert_source.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace {
using Json = nlohmann::json;
using Tensor = std::tuple<std::string, std::string, std::vector<uint64_t>, std::vector<uint8_t>>;
int failures = 0;

void check(bool ok, const std::string& label) {
    std::cout << "  " << label << ": " << (ok ? "OK" : "FAIL") << '\n';
    if (!ok) ++failures;
}

bool cache_capacity(const strata::core::Exl3ExpertSource& source, uint64_t required, const char* label) {
    const uint64_t capacity = source.cache_stats().capacity_bytes;
    if (capacity >= required) return true;
    std::cout << "  " << label << ": SKIP (cache capacity " << capacity << " < " << required << ")\n";
    return false;
}

std::vector<uint8_t> words(size_t n, uint16_t seed) {
    std::vector<uint8_t> out(n * 2);
    for (size_t i = 0; i < n; ++i) {
        const uint16_t v = static_cast<uint16_t>(seed + i * 7);
        out[2 * i] = static_cast<uint8_t>(v); out[2 * i + 1] = static_cast<uint8_t>(v >> 8);
    }
    return out;
}

std::vector<uint8_t> u32(uint32_t v) {
    return {static_cast<uint8_t>(v), static_cast<uint8_t>(v >> 8), static_cast<uint8_t>(v >> 16), static_cast<uint8_t>(v >> 24)};
}

void add_linear(std::vector<Tensor>& tensors, const std::string& name, int bits, int codebook) {
    tensors.emplace_back(name + ".trellis", "I16", std::vector<uint64_t>{8, 8, static_cast<uint64_t>(16 * bits)},
                         words(static_cast<size_t>(8 * 8 * 16 * bits), static_cast<uint16_t>(bits * 19)));
    tensors.emplace_back(name + ".suh", "F16", std::vector<uint64_t>{128}, words(128, 0x3c00));  // codespell:ignore suh
    tensors.emplace_back(name + ".svh", "F16", std::vector<uint64_t>{128}, words(128, 0x3c00));
    if (codebook == 1) tensors.emplace_back(name + ".mcg", "U32", std::vector<uint64_t>{}, u32(0xCBAC1FEDu));
    if (codebook == 2) tensors.emplace_back(name + ".mul1", "I32", std::vector<uint64_t>{1}, u32(0x83DCD12Du));
}

void write_archive(const std::filesystem::path& root, std::vector<strata::artifact::exl3::ExpertTensorNames>& names) {
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json") << R"({"model_type":"synthetic-exl3-source"})";
    std::vector<Tensor> tensors;
    for (int layer = 0; layer < 2; ++layer)
        for (int expert = 0; expert < 2; ++expert) {
            const std::string base = "layers." + std::to_string(layer) + ".experts." + std::to_string(expert);
            names.push_back({base + ".gate", base + ".up", base + ".down"});
            const int bits = 2 + ((layer * 2 + expert) % 4);
            if (layer == 1 && expert == 1)
                tensors.emplace_back(base + ".gate.weight", "BF16", std::vector<uint64_t>{128, 128},
                                     words(128 * 128, 0x3f80));
            else add_linear(tensors, base + ".gate", bits, layer % 3);
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

}  // namespace

int main() {
    const auto root = std::filesystem::temp_directory_path() / "xestrata_exl3_source_test";
    std::filesystem::remove_all(root);
    std::vector<strata::artifact::exl3::ExpertTensorNames> names;
    write_archive(root, names);
    auto archive = std::make_shared<strata::artifact::exl3::Archive>(root);
    strata::core::Exl3ExpertSource source(archive, 2, 2, names);
    // Preserve the original window-scoped I/O-failure checks independently
    // of the new persistent-cache cases below.
    strata::core::Exl3ExpertSource::CachePolicy disabled;
    disabled.max_bytes = 0;
    std::string source_error;
    check(source.open(archive, 2, 2, names, source_error, disabled), "zero-budget open succeeds");
    check(source.last_error().empty(), "metadata open succeeds");
    int32_t ids[] = {0, 1, 1};
    source.begin_layer(0, ids, 3);
    check(source.last_error().empty(), "begin_layer materializes unique ids");
    const uint8_t* first = source.blob(0, 0);
    const uint8_t* repeated = source.blob(0, 0);
    const uint8_t* second = source.blob(0, 1);
    check(first != nullptr && second != nullptr && first == repeated, "repeated id has stable pointer");
    const auto* layout = source.exl3_layout(0, 0);
    check(layout != nullptr && layout->total_bytes > 0 && source.transient(0, 0), "layout and transient contract");
    std::vector<uint8_t> copied(layout == nullptr ? 0 : static_cast<size_t>(layout->total_bytes), 0);
    check(source.copy_blob(0, 0, copied.data()) && first != nullptr &&
              std::memcmp(first, copied.data(), copied.size()) == 0, "copy_blob is byte exact");
    std::vector<uint8_t> copied_other(layout == nullptr ? 0 : static_cast<size_t>(source.exl3_layout(1, 0)->total_bytes), 0);
    check(source.copy_blob(1, 0, copied_other.data()) && source.blob(0, 0) == first,
          "copy_blob other layer preserves active pointer");
    int32_t bad_ids[] = {0, 2};
    source.begin_layer(0, bad_ids, 2);
    check(!source.last_error().empty() && source.blob(0, 0) == nullptr, "failed begin publishes no partial blobs");
    std::filesystem::rename(root / "model.safetensors", root / "model.unavailable");
    source.begin_layer(0, ids, 2);
    const bool io_failed = !source.last_error().empty();
    std::filesystem::rename(root / "model.unavailable", root / "model.safetensors");
    check(io_failed && source.blob(0, 0) == nullptr,
          "failed payload read leaves the window unpublished after file recovery");
    source.begin_layer(1, ids, 2);
    check(source.blob(1, 0) != nullptr && source.blob(1, 1) != nullptr, "second layer materializes independently");
    std::string err;
    auto bad_names = names;
    bad_names.pop_back();
    strata::core::Exl3ExpertSource bad;
    check(!bad.open(archive, 2, 2, std::move(bad_names), err), "flat table mismatch is rejected");
    strata::core::Exl3ExpertSource oversized;
    check(!oversized.open(archive, std::numeric_limits<int64_t>::max(), 2, names, err),
          "oversized layer dimensions are rejected");

    // 1. Variable-width/codebook compressed experts survive another layer.
    // A missing archive must remain readable on hits without materialization.
    using Source = strata::core::Exl3ExpertSource;
    Source::CachePolicy cache_policy;
    cache_policy.headroom_bytes = 0;
    uint64_t all_bytes = 0;
    for (int layer = 0; layer < 2; ++layer)
        for (int expert = 0; expert < 2; ++expert)
            all_bytes += source.exl3_layout(layer, expert)->total_bytes;
    cache_policy.max_bytes = all_bytes;
    Source reused;
    check(reused.open(archive, 2, 2, names, err, cache_policy), "persistent source opens");
    const int32_t both[] = {0, 1};
    const int32_t zero[] = {0};
    // External cache caps and tight host/cgroup limits may intentionally disable
    // persistence. Keep the legacy/fallback checks below active in that case.
    const uint64_t reuse_required = all_bytes - source.exl3_layout(1, 1)->total_bytes;
    if (cache_capacity(reused, reuse_required, "persistent reuse")) {
        reused.begin_layer(0, both, 2);
        const uint8_t* kept = reused.blob(0, 0);
        reused.begin_layer(1, zero, 1);
        const auto before_reuse = reused.cache_stats();
        std::filesystem::rename(root / "model.safetensors", root / "model.unavailable");
        reused.begin_layer(0, both, 2);
        const bool hit_ok = kept != nullptr && reused.last_error().empty() && reused.blob(0, 0) == kept;
        const auto after_reuse = reused.cache_stats();
        std::filesystem::rename(root / "model.unavailable", root / "model.safetensors");
        check(hit_ok && after_reuse.hits == before_reuse.hits + 2 &&
                  after_reuse.materializations == before_reuse.materializations &&
                  after_reuse.payload_read_bytes == before_reuse.payload_read_bytes,
              "repeated layer uses compressed cache without source reads");
        check(after_reuse.resident_bytes <= after_reuse.capacity_bytes && reused.transient(0, 0),
              "persistent budget and borrowed-pointer contract are bounded");
    }

    // 2. The oldest entry is requested second. Pre-pinning the complete route
    // keeps it alive while the first requested miss evicts another layer.
    const uint64_t a = source.exl3_layout(0, 0)->total_bytes;
    const uint64_t b = source.exl3_layout(1, 0)->total_bytes;
    const uint64_t c = source.exl3_layout(0, 1)->total_bytes;
    cache_policy.max_bytes = std::max(a + b, a + c);
    Source evicting;
    check(evicting.open(archive, 2, 2, names, err, cache_policy), "LRU source opens");
    if (cache_capacity(evicting, cache_policy.max_bytes, "LRU eviction and pinning")) {
        evicting.begin_layer(0, zero, 1);
        const uint8_t* pinned = evicting.blob(0, 0);
        check(pinned != nullptr, "LRU first blob materializes");
        std::vector<uint8_t> pinned_bytes(static_cast<size_t>(a));
        if (pinned != nullptr) std::memcpy(pinned_bytes.data(), pinned, pinned_bytes.size());
        evicting.begin_layer(1, zero, 1);
        const int32_t miss_before_hit[] = {1, 0};
        evicting.begin_layer(0, miss_before_hit, 2);
        const auto evicted = evicting.cache_stats();
        check(pinned != nullptr && evicting.last_error().empty() && evicting.blob(0, 0) == pinned &&
                  std::memcmp(pinned, pinned_bytes.data(), pinned_bytes.size()) == 0 && evicted.evictions == 1 &&
                  evicted.resident_bytes <= evicted.capacity_bytes,
              "all requested ids are pinned before LRU admission");
        const auto before_invalid = evicting.cache_stats();
        evicting.begin_layer(0, bad_ids, 2);
        check(!evicting.last_error().empty() && evicting.blob(0, 0) == nullptr &&
                  evicting.cache_stats().transient_bytes == 0,
              "invalid window clears pins and transient state");
        evicting.begin_layer(1, zero, 1);
        check(evicting.last_error().empty() && evicting.blob(1, 0) != nullptr &&
                  evicting.cache_stats().evictions > before_invalid.evictions,
              "entries from a failed window are evictable again");
    }

    // All requested blobs remain valid even when their sum exceeds the cache.
    cache_policy.max_bytes = std::min(a, c);
    Source overflowing;
    check(overflowing.open(archive, 2, 2, names, err, cache_policy), "small-budget source opens");
    overflowing.begin_layer(0, both, 2);
    const uint8_t* small_first = overflowing.blob(0, 0);
    const uint8_t* small_second = overflowing.blob(0, 1);
    const auto overflowed = overflowing.cache_stats();
    check(small_first != nullptr && small_second != nullptr && overflowing.blob(0, 0) == small_first &&
              overflowed.fallbacks > 0 && overflowed.transient_bytes > 0 &&
              overflowed.resident_bytes <= overflowed.capacity_bytes,
          "over-budget window retains simultaneous transient fallbacks");

    // 3. A zero budget retains raw source bytes/dtypes and rereads next window;
    // cold copy_blob (used by GPU/MTP preload) never admits a CPU cache entry.
    Source raw_fallback;
    check(raw_fallback.open(archive, 2, 2, names, err, disabled), "raw fallback opens");
    const int32_t one[] = {1};
    raw_fallback.begin_layer(1, one, 1);
    const auto* raw_layout = raw_fallback.exl3_layout(1, 1);
    const uint8_t* raw_first = raw_fallback.blob(1, 1);
    std::vector<uint8_t> raw_copy(static_cast<size_t>(raw_layout->total_bytes));
    std::vector<uint8_t> raw_original;
    archive->read_tensor(*archive->find_tensor(names[3].gate + ".weight"), raw_original);
    check(raw_fallback.copy_blob(1, 1, raw_copy.data()) && raw_first != nullptr &&
              std::memcmp(raw_first, raw_copy.data(), raw_copy.size()) == 0 &&
              std::memcmp(raw_first + raw_layout->gate.raw_weights.offset, raw_original.data(),
                          raw_original.size()) == 0 &&
              raw_layout->gate.storage == strata::kernels::exl3::ProjectionStorage::raw &&
              raw_layout->gate.raw_dtype == strata::kernels::raw::DType::BF16,
          "zero-budget packed/raw mixture preserves source bytes and dtype");
    const auto before_raw = raw_fallback.cache_stats();
    raw_fallback.begin_layer(1, one, 1);
    const auto after_raw = raw_fallback.cache_stats();
    check(raw_fallback.last_error().empty() && after_raw.resident_bytes == 0 && after_raw.capacity_bytes == 0 &&
              after_raw.materializations == before_raw.materializations + 1,
          "zero-budget path materializes each window without persistent residence");
    raw_fallback.begin_layer(0, nullptr, 0);
    check(raw_fallback.cache_stats().transient_bytes == 0 &&
              raw_fallback.copy_blob(1, 1, raw_copy.data()) && raw_fallback.cache_stats().resident_bytes == 0,
          "cold copy preserves empty window and avoids host cache duplication");

    // Exercise the actual ExpertDispatch -> ExpertPool boundary.  The
    // expected rows are produced independently by the single-expert CPU
    // routine, so this covers source routing, FP32 input strides, and the
    // pool's persistent EXL3 scratch rather than only testing the adapter.
    std::vector<std::vector<uint8_t>> dispatch_blobs(2);
    std::vector<strata::kernels::exl3::ExpertView> dispatch_views;
    dispatch_views.reserve(2);
    bool copied_views = true;
    for (int expert = 0; expert < 2; ++expert) {
        const auto* expert_layout = source.exl3_layout(0, expert);
        if (expert_layout == nullptr) {
            copied_views = false;
            break;
        }
        dispatch_blobs[static_cast<size_t>(expert)].resize(static_cast<size_t>(expert_layout->total_bytes));
        if (!source.copy_blob(0, expert, dispatch_blobs[static_cast<size_t>(expert)].data())) {
            copied_views = false;
            break;
        }
        dispatch_views.push_back(strata::kernels::exl3::expert_view(
            *expert_layout, dispatch_blobs[static_cast<size_t>(expert)].data()));
    }
    check(copied_views && dispatch_views.size() == 2, "dispatch expected views copied independently");
    constexpr int64_t dispatch_tokens = 4;
    constexpr int64_t dispatch_k = 2;
    constexpr int64_t dispatch_hidden = 128;
    // ExpertDispatch's public multi-token contract is a contiguous
    // `(n_tok, hidden)` FP32 matrix; padding belongs to the lower-level
    // ExpertJobExl3 API and is intentionally covered by exl3_pool_test.
    constexpr int64_t dispatch_ldx = dispatch_hidden;
    const int32_t dispatch_ids[] = {0, 1, 1, 0, 1, 1, 0, 1};
    std::vector<float> dispatch_input(static_cast<size_t>(dispatch_tokens * dispatch_ldx));
    for (size_t i = 0; i < dispatch_input.size(); ++i)
        dispatch_input[i] = 0.03125f * static_cast<float>(static_cast<int>(i % 23) - 11);
    std::vector<float> dispatch_expected(static_cast<size_t>(dispatch_tokens * dispatch_k * dispatch_hidden), -1234.0f);
    std::vector<float> dispatch_actual = dispatch_expected;
    bool expected_ok = copied_views;
    for (int64_t entry = 0; entry < dispatch_tokens * dispatch_k && expected_ok; ++entry) {
        const int64_t token = entry / dispatch_k;
        const int expert = dispatch_ids[entry];
        expected_ok = strata::kernels::cpu::exl3::expert(
                          dispatch_views[static_cast<size_t>(expert)],
                          dispatch_input.data() + static_cast<size_t>(token * dispatch_ldx), dispatch_ldx,
                          dispatch_expected.data() + static_cast<size_t>(entry * dispatch_hidden), dispatch_hidden, 1) ==
                      strata::kernels::cpu::exl3::Status::ok;
    }
    check(expected_ok, "standalone routed expert reference succeeds");
    strata::kernels::cpu::ExpertPool dispatch_pool(1, false, false);
    check(dispatch_pool.prepare_exl3(1, dispatch_hidden, dispatch_hidden) == strata::kernels::cpu::exl3::Status::ok,
          "dispatch pool scratch prepared");
    strata::core::ExpertDispatch dispatch;
    dispatch.pool = &dispatch_pool;
    dispatch.src = &source;
    dispatch.n_expert = 2;
    strata::core::expert_pool_dispatch_multi(dispatch, dispatch_input.data(), dispatch_ids, dispatch_tokens, dispatch_k,
                                             dispatch_actual.data());
    check(!dispatch.failed && dispatch.layers == 1 && dispatch.experts == 8 && dispatch.multi_entries == 8 &&
              dispatch.multi_misses == 2,
          "actual EXL3 dispatch advances layer and counters");
    check(std::memcmp(dispatch_expected.data(), dispatch_actual.data(), dispatch_actual.size() * sizeof(float)) == 0,
          "actual EXL3 dispatch matches standalone rows");

    strata::core::ExpertDispatch invalid_dispatch;
    invalid_dispatch.pool = &dispatch_pool;
    invalid_dispatch.src = &source;
    invalid_dispatch.n_expert = 2;
    std::vector<float> invalid_output(dispatch_expected.size(), -4321.0f);
    const int32_t invalid_ids[] = {0, 2};
    strata::core::expert_pool_dispatch_multi(invalid_dispatch, dispatch_input.data(), invalid_ids, 1, 2,
                                             invalid_output.data());
    bool invalid_untouched = true;
    for (float value : invalid_output) invalid_untouched &= value == -4321.0f;
    check(invalid_dispatch.failed && invalid_dispatch.layers == 0 && invalid_untouched,
          "invalid routed id fails before writing outputs");
    std::filesystem::remove_all(root);
    std::cout << "exl3_expert_source_test: " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
