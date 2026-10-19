// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// CPU-side EXL3 owner exception and legacy WeightTable layout checks.
#include "strata/core/exl3_weight_set.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/runtime.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;
using namespace strata::core;
using namespace strata::artifact::exl3;

struct Row {
    std::string name;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
};

strata::core::ModelGeometry geometry() {
    strata::core::ModelGeometry g;
    g.n_embd = 4;
    g.n_layers = 1;
    g.qsa_interval = 4;
    g.ssm_state_size = 2;
    g.ssm_k_heads = 1;
    g.ssm_v_heads = 1;
    g.ssm_d_conv = 2;
    g.ssm_conv_channels = 6;
    g.ssm_value_dim = 2;
    g.n_head = 1;
    g.n_head_kv = 1;
    g.head_dim = 2;
    g.idx_q_heads = 1;
    g.idx_key_dim = 1;
    g.hc = 1;
    g.hc_lr = 1;
    g.n_expert = 2;
    g.n_ff = 3;
    return g;
}

std::vector<Row> rows(const strata::core::ModelGeometry& g) {
    const std::string p = "blk.0.";
    return {
        {p + "hc_attn_down.weight", g.hc_dim(), g.hc_lr},
        {p + "hc_attn_up.weight", g.hc_lr, g.hc_dim()},
        {p + "hc_attn_inject.weight", g.hc_dim(), g.hc},
        {p + "hc_ffn_down.weight", g.hc_dim(), g.hc_lr},
        {p + "hc_ffn_up.weight", g.hc_lr, g.hc_dim()},
        {p + "hc_ffn_inject.weight", g.hc_dim(), g.hc},
        {p + "ffn_gate_inp.weight", g.n_embd, g.n_expert},
        {p + "ffn_gate_shexp.weight", g.n_embd, g.n_ff},
        {p + "ffn_up_shexp.weight", g.n_embd, g.n_ff},
        {p + "ffn_down_shexp.weight", g.n_ff, g.n_embd},
        {p + "attn_qkv.weight", g.n_embd, g.ssm_conv_channels},
        {p + "attn_gate.weight", g.n_embd, g.ssm_value_dim},
        {p + "ssm_out.weight", g.ssm_value_dim, g.n_embd},
        {p + "ssm_conv1d.weight", g.ssm_d_conv, g.ssm_conv_channels},
        {p + "ssm_alpha.weight", g.n_embd, g.ssm_v_heads},
        {p + "ssm_beta.weight", g.n_embd, g.ssm_v_heads},
        {p + "hc_attn_norm.weight", g.hc_dim(), 1},
        {p + "hc_ffn_norm.weight", g.hc_dim(), 1},
        {p + "ffn_gate_inp_shexp.weight", g.n_embd, 1},
        {p + "ssm_a", g.ssm_v_heads, 1},
        {p + "ssm_dt.bias", g.ssm_v_heads, 1},
        {p + "ssm_norm.weight", g.ssm_state_size, 1},
    };
}

void write_pack(const std::filesystem::path& root, const strata::core::ModelGeometry& g, bool wrong_scalar = false,
                bool omit_scalar = false) {
    std::filesystem::create_directories(root);
    auto all_rows = rows(g);
    if (omit_scalar)
        all_rows.erase(std::remove_if(all_rows.begin(), all_rows.end(), [](const Row& row) {
            return row.name == "blk.0.ffn_gate_inp_shexp.weight";
        }), all_rows.end());
    uint64_t offset = 0;
    std::vector<uint8_t> dense;
    std::ofstream index(root / "index.txt");
    index << "# strata pack index v3 -- synthetic layout test\n";
    uint64_t pool = 0;
    for (const Row& row : all_rows) {
        const uint64_t bytes = static_cast<uint64_t>(row.ne0) * static_cast<uint64_t>(row.ne1) * sizeof(float);
        const bool scalar = row.name == "blk.0.ffn_gate_inp_shexp.weight";
        const uint64_t dst_bytes = scalar && !wrong_scalar ? bytes / 2 : bytes;
        const uint64_t dst = pool;
        index << row.name << " 0 " << (scalar && !wrong_scalar ? 1 : 2) << " " << offset << " " << bytes << " " << dst << " " << dst_bytes << " "
              << row.ne0 << " " << row.ne1 << " 0 0 0 0 0 0 0 0 0 0\n";
        dense.resize(static_cast<size_t>(offset + bytes), 0);
        offset += bytes;
        pool = (dst + dst_bytes + 31u) & ~uint64_t(31u);
    }
    index.seekp(0, std::ios::end);
    index.flush();
    std::ofstream(root / "dense.bin", std::ios::binary).write(reinterpret_cast<const char*>(dense.data()),
                                                               static_cast<std::streamsize>(dense.size()));
    // Rewrite the header after all rows are known; WeightTable only needs the
    // generated index text and a pool size consistent with its offsets.
    std::ifstream old(root / "index.txt");
    std::string body((std::istreambuf_iterator<char>(old)), std::istreambuf_iterator<char>());
    const size_t first_newline = body.find('\n');
    std::ofstream fixed(root / "index.txt", std::ios::trunc);
    fixed << "# strata pack index v3 -- synthetic layout test\n# align 32 pool " << pool << " tensors " << all_rows.size()
          << "\n# name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias group_elems codebook has_offset codes_bytes scales_bytes offset_bytes scales_fp16 act_kind\n";
    fixed << body.substr(first_newline + 1);
}

void write_archive(const std::filesystem::path& root) {
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json") << R"({"model_type":"synthetic-exl3-layout"})";
    Json header = Json::object();
    std::vector<uint8_t> scalar(8, 0x3c);
    std::vector<uint8_t> scalar_bad(8, 0x3c);
    header["shared.scalar.weight"] = Json{{"dtype", "F16"}, {"shape", {1, 4}}, {"data_offsets", {0, 8}}};
    header["shared.scalar_bad.weight"] = Json{{"dtype", "F16"}, {"shape", {2, 2}}, {"data_offsets", {8, 16}}};
    const std::string text = header.dump();
    std::ofstream output(root / "model.safetensors", std::ios::binary);
    for (int i = 0; i < 8; ++i) output.put(static_cast<char>(text.size() >> (8 * i)));
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    output.write(reinterpret_cast<const char*>(scalar.data()), static_cast<std::streamsize>(scalar.size()));
    output.write(reinterpret_cast<const char*>(scalar_bad.data()), static_cast<std::streamsize>(scalar_bad.size()));
}

ProjectionMapping scalar_mapping(const std::string& source, int64_t k) {
    ProjectionMapping mapping;
    mapping.kind = ProjectionKind::shared_scalar;
    mapping.transform = MappingTransform::raw_rows;
    mapping.layer = 0;
    mapping.source_name = source + ".weight";
    mapping.canonical_name = "blk.0.ffn_gate_inp_shexp.weight";
    mapping.target_k = k;
    mapping.target_n = source == "shared.scalar_bad" ? 2 : 1;
    mapping.raw_dtype = "F16";
    mapping.raw_shape = {1, k};
    mapping.raw_rows = 1;
    mapping.raw_cols = k;
    return mapping;
}

bool load_table(const std::filesystem::path& root, WeightTable& table, void*& arena, std::string& error) {
    uint64_t bytes = 0;
    if (!WeightTable::pool_bytes(root.string(), bytes, error)) return false;
    arena = strata::gpu::alloc_device(static_cast<size_t>(bytes));
    if (arena == nullptr) { error = "device arena allocation failed"; return false; }
    if (!table.load(root.string(), arena, bytes, error)) return false;
    return true;
}

}  // namespace

int main() {
    if (strata::core::Runtime::count() == 0) return 77;
    const auto root = std::filesystem::temp_directory_path() /
                      ("xestrata_exl3_layout_test_" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    if (!std::filesystem::create_directory(root)) return 1;
    const strata::core::ModelGeometry g = geometry();
    write_pack(root / "pack", g);
    write_archive(root / "archive");
    int failures = 0;
    std::string error;
    try {
        WeightTable table;
        void* arena = nullptr;
        if (!load_table(root / "pack", table, arena, error)) throw std::runtime_error(error);
        if (!check_layer(table, g, 0, error)) throw std::runtime_error("valid legacy layout: " + error);
        auto archive = Archive(root / "archive");
        ModelMap map;
        map.geometry.layers = 1;
        map.geometry.hidden_size = g.n_embd;
        map.projections.push_back(scalar_mapping("shared.scalar", g.n_embd));
        Exl3WeightSet set;
        Exl3WeightSetOptions options;
        options.layer_end = 1;
        if (!set.load(archive, map, table, options, error)) throw std::runtime_error(error);
        if (!check_layer(table, g, 0, error)) {
            std::fprintf(stderr, "valid EXL3 shared scalar rejected: %s\n", error.c_str());
            ++failures;
        }
        set.close();
        strata::gpu::free(arena);
        arena = nullptr;
        write_pack(root / "bad", g, false, true);
        WeightTable bad_table;
        if (!load_table(root / "bad", bad_table, arena, error)) throw std::runtime_error(error);
        map.projections = {scalar_mapping("shared.scalar_bad", 2)};
        if (!set.load(archive, map, bad_table, options, error)) throw std::runtime_error(error);
        if (check_layer(bad_table, g, 0, error) || error.find("EXL3 owner ne0") == std::string::npos) {
            std::fprintf(stderr, "malformed EXL3 scalar was not rejected by shape: %s\n", error.c_str());
            ++failures;
        }
        set.close();
        strata::gpu::free(arena);
        arena = nullptr;
        write_pack(root / "wrong", g, true);
        WeightTable wrong_table;
        if (!load_table(root / "wrong", wrong_table, arena, error)) throw std::runtime_error(error);
        if (check_layer(wrong_table, g, 0, error) || error.find("engine form") == std::string::npos) {
            std::fprintf(stderr, "legacy scalar kind was not rejected: %s\n", error.c_str());
            ++failures;
        }
        strata::gpu::free(arena);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "layout fixture failed: %s\n", e.what());
        ++failures;
    }
    std::filesystem::remove_all(root);
    std::printf("exl3_layout_test: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
