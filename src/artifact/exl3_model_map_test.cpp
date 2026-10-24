// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Header-only validation smoke test for the explicit Qwen EXL3 model map.

#include "strata/artifact/exl3_model_map.hpp"

#include <filesystem>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>

namespace {

bool has_mapping(const strata::artifact::exl3::ModelMap& map, strata::artifact::exl3::ProjectionKind kind,
                strata::artifact::exl3::ValueTransform value = strata::artifact::exl3::ValueTransform::identity) {
    for (const auto& item : map.projections)
        if (item.kind == kind && item.value_transform == value) return true;
    return false;
}

bool has_nontrivial_v_map(const strata::artifact::exl3::ModelMap& map,
                          strata::artifact::exl3::ProjectionKind kind) {
    for (const auto& item : map.projections) {
        if (item.kind != kind) continue;
        const auto& axis = kind == strata::artifact::exl3::ProjectionKind::gdn_out ? item.input_axis : item.output_axis;
        const int64_t key_heads = map.geometry.linear_k_heads;
        const int64_t value_heads = map.geometry.linear_v_heads;
        const int64_t head_dim = map.geometry.linear_v_dim;
        if (axis.transform != strata::artifact::exl3::AxisTransform::h128_block_permutation ||
            axis.block_size != head_dim || axis.block_permutation.size() != static_cast<size_t>(value_heads) ||
            axis.permutation.empty() || key_heads <= 0 || value_heads <= 0 || value_heads % key_heads != 0)
            return false;
        const int64_t values_per_key = value_heads / key_heads;
        for (int64_t target_head = 0; target_head < value_heads; ++target_head) {
            const int64_t expected_source_head = (target_head % key_heads) * values_per_key + target_head / key_heads;
            if (axis.block_permutation[static_cast<size_t>(target_head)] != expected_source_head)
                return false;
            const int64_t target = axis.block_offset + target_head * head_dim;
            const int64_t source = axis.block_offset + expected_source_head * head_dim;
            if (axis.permutation[static_cast<size_t>(target)] != source)
                return false;
        }
        return true;
    }
    return false;
}

bool check_v_head_fixture() {
    using namespace strata::artifact::exl3;
    const std::vector<int64_t> expected{0, 2, 1, 3};
    const auto check = [&](const AxisMapping& axis, int64_t total, int64_t offset, int64_t width, const char* label) {
        if (axis.transform != AxisTransform::h128_block_permutation || axis.block_size != width ||
            axis.block_offset != offset || axis.block_permutation != expected ||
            static_cast<int64_t>(axis.permutation.size()) != total)
            return false;
        for (int64_t h = 0; h < 4; ++h) {
            for (int64_t d = 0; d < width; ++d) {
                const int64_t target = offset + h * width + d;
                const int64_t source = offset + expected[static_cast<size_t>(h)] * width + d;
                if (axis.permutation[static_cast<size_t>(target)] != source) {
                    std::cerr << "V-head fixture mismatch in " << label << '\n';
                    return false;
                }
            }
        }
        return true;
    };
    constexpr int64_t qk = 2 * 2 * 128;
    constexpr int64_t v = 4 * 128;
    return check(make_v_head_axis(qk + v, qk, 2, 4, 128), qk + v, qk, 128, "qkv V segment") &&
           check(make_v_head_axis(v, 0, 2, 4, 128), v, 0, 128, "out input") &&
           check(make_v_head_axis(4, 0, 2, 4, 1), 4, 0, 1, "alpha rows") &&
           check(make_v_head_axis(qk + v, qk, 2, 4, 128), qk + v, qk, 128, "conv V channels");
}

struct FixtureTensor {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t bytes = 0;
};

bool check_raw_gdn_fixture() {
    using namespace strata::artifact::exl3;
    const auto root = std::filesystem::temp_directory_path() /
        ("xestrata_exl3_model_map_raw_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json") << R"({"model_type":"qwen3_5_moe_text","tie_word_embeddings":false,"text_config":{"hidden_size":4,"vocab_size":4,"num_hidden_layers":1,"num_experts":1,"num_experts_per_tok":1,"moe_intermediate_size":4,"shared_expert_intermediate_size":4,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":128,"linear_num_key_heads":1,"linear_num_value_heads":2,"linear_key_head_dim":128,"linear_value_head_dim":128,"full_attention_interval":4,"layer_types":["linear_attention"],"rms_norm_eps":1e-6}})";
    std::vector<FixtureTensor> tensors;
    const auto add = [&](const std::string& name, const std::string& dtype, std::vector<int64_t> shape) {
        uint64_t elements = 1;
        for (const int64_t dim : shape) elements *= static_cast<uint64_t>(dim);
        const uint64_t element_bytes = dtype == "F32" ? 4 : dtype == "I32" ? 4 : 2;
        tensors.push_back({name, dtype, std::move(shape), elements * element_bytes});
    };
    const std::string p = "model.language_model.layers.0.";
    add(p + "linear_attn.in_proj_qkv.weight", "F16", {512, 4});
    add(p + "linear_attn.in_proj_z.weight", "F16", {256, 4});
    add(p + "linear_attn.out_proj.weight", "F16", {4, 256});
    add(p + "linear_attn.in_proj_a.weight", "F16", {2, 4});
    add(p + "linear_attn.in_proj_b.weight", "F16", {2, 4});
    add(p + "linear_attn.norm.weight", "F32", {128});
    add(p + "linear_attn.A_log", "F32", {2});
    add(p + "linear_attn.dt_bias", "F32", {2});
    add(p + "linear_attn.conv1d.weight", "F16", {512, 1, 4});
    add(p + "input_layernorm.weight", "BF16", {4});
    add(p + "post_attention_layernorm.weight", "BF16", {4});
    add(p + "mlp.gate.weight", "F16", {1, 4});
    add(p + "mlp.shared_expert_gate.weight", "F16", {1, 4});
    add(p + "mlp.shared_expert.gate_proj.weight", "F16", {4, 4});
    add(p + "mlp.shared_expert.up_proj.weight", "F16", {4, 4});
    add(p + "mlp.shared_expert.down_proj.weight", "F16", {4, 4});
    add(p + "mlp.experts.0.gate_proj.weight", "F16", {4, 4});
    add(p + "mlp.experts.0.up_proj.weight", "F16", {4, 4});
    add(p + "mlp.experts.0.down_proj.weight", "F16", {4, 4});
    add("model.language_model.embed_tokens.weight", "F16", {4, 4});
    add("model.language_model.norm.weight", "BF16", {4});
    add("lm_head.weight", "F16", {4, 4});
    std::ostringstream header;
    header << '{';
    uint64_t offset = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (i != 0) header << ',';
        const auto& tensor = tensors[i];
        header << '"' << tensor.name << '"' << ':' << '{'
               << '"' << "dtype" << '"' << ':' << '"' << tensor.dtype << '"' << ','
               << '"' << "shape" << '"' << ':' << '[';
        for (size_t j = 0; j < tensor.shape.size(); ++j) {
            if (j != 0) header << ',';
            header << tensor.shape[j];
        }
        header << ']' << ',' << '"' << "data_offsets" << '"' << ':' << '['
               << offset << ',' << offset + tensor.bytes << ']' << '}';
        offset += tensor.bytes;
    }
    header << '}';
    const std::string raw_header = header.str();
    std::ofstream model(root / "model.safetensors", std::ios::binary);
    const uint64_t header_bytes = raw_header.size();
    for (int i = 0; i < 8; ++i) model.put(static_cast<char>(header_bytes >> (8 * i)));
    model.write(raw_header.data(), static_cast<std::streamsize>(raw_header.size()));
    std::vector<uint8_t> zeros(static_cast<size_t>(offset), 0);
    model.write(reinterpret_cast<const char*>(zeros.data()), static_cast<std::streamsize>(zeros.size()));
    model.close();

    bool ok = false;
    try {
        Archive archive(root);
        ModelMap map;
        std::string err;
        const bool built = build_model_map(archive, map, err);
        ok = built && map.unsupported.empty();
        if (!built) std::cerr << "raw GDN fixture build error: " << err << '\n';
        for (const auto& item : map.unsupported) std::cerr << "raw GDN fixture unsupported: " << item << '\n';
        const auto find = [&](ProjectionKind kind) -> const ProjectionMapping* {
            for (const auto& item : map.projections)
                if (item.kind == kind && item.layer == 0) return &item;
            return nullptr;
        };
        const auto find_canonical = [&](const char* canonical) -> const ProjectionMapping* {
            for (const auto& item : map.projections)
                if (item.layer == 0 && item.canonical_name == canonical) return &item;
            return nullptr;
        };
        const auto* qkv = find(ProjectionKind::gdn_qkv);
        const auto* z = find(ProjectionKind::gdn_z);
        const auto* out = find(ProjectionKind::gdn_out);
        const auto* alpha = find(ProjectionKind::gdn_alpha);
        const auto* conv = find_canonical("blk.0.ssm_conv1d.weight");
        ok = ok && qkv != nullptr && z != nullptr && out != nullptr && alpha != nullptr && conv != nullptr;
        ok = ok && qkv->transform == MappingTransform::raw_rows && qkv->output_axis.block_permutation == std::vector<int64_t>({0, 1});
        ok = ok && qkv->output_axis.block_offset == 256 && qkv->raw_to_canonical_axes == std::vector<int64_t>({1, 0});
        ok = ok && z->output_axis.block_offset == 0 && out->input_axis.block_offset == 0 &&
             out->input_axis.block_permutation == std::vector<int64_t>({0, 1});
        ok = ok && alpha->output_axis.block_size == 1 && alpha->raw_to_canonical_axes == std::vector<int64_t>({1, 0});
        ok = ok && conv->raw_shape == std::vector<int64_t>({512, 1, 4}) && conv->raw_axes.front().block_offset == 256;
        ModelMap mtp_map;
        std::string mtp_err;
        ok = ok && build_model_map(archive, mtp_map, mtp_err) && mtp_map.unsupported.empty();
        ModelMap requested_mtp;
        std::string requested_mtp_err;
        ok = ok && build_model_map(archive, requested_mtp, requested_mtp_err, true) &&
             !requested_mtp.unsupported.empty();
    } catch (const std::exception& e) {
        std::cerr << "raw GDN fixture exception: " << e.what() << '\n';
        ok = false;
    }
    std::filesystem::remove_all(root);
    return ok;
}

bool check_lfm2moe_fixture() {
    using namespace strata::artifact::exl3;
    const auto root = std::filesystem::temp_directory_path() /
        ("xestrata_exl3_model_map_lfm2_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json")
        << R"({"model_type":"lfm2_moe","text_config":{"hidden_size":4,"vocab_size":4,"num_hidden_layers":2,"num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":4,"intermediate_size":4,"conv_L_cache":3,"layer_types":["conv","full_attention"],"num_attention_heads":1,"num_key_value_heads":1,"num_dense_layers":1,"use_expert_bias":true}})";
    const std::string header =
        R"({"model.embed_tokens.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[0,32]},"lm_head.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[32,64]}})";
    std::ofstream model(root / "model.safetensors", std::ios::binary);
    const uint64_t header_bytes = header.size();
    for (int i = 0; i < 8; ++i) model.put(static_cast<char>(header_bytes >> (8 * i)));
    model.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (int i = 0; i < 64; ++i) model.put(0);
    model.close();

    bool ok = false;
    try {
        Archive archive(root);
        ModelMap map;
        std::string err;
        const bool built = build_model_map(archive, map, err);
        bool has_embed = false, has_head = false, names_conv = false;
        for (const auto& item : map.projections) {
            if (item.kind == ProjectionKind::embedding) has_embed = true;
            if (item.kind == ProjectionKind::lm_head) has_head = true;
        }
        for (const auto& item : map.unsupported)
            if (item.find("conv.in_proj") != std::string::npos) names_conv = true;
        std::string require_err;
        ok = built && map.family == ModelFamily::lfm2moe && map.geometry.hidden_size == 4 &&
             map.geometry.layers == 2 && map.geometry.experts == 2 && map.geometry.dense_layers == 1 &&
             map.geometry.dense_intermediate == 4 && map.geometry.shortconv_l_cache == 3 && has_embed &&
             has_head && names_conv && !map.unsupported.empty() && !map.require_supported(require_err) &&
             require_err.find("lfm2moe") != std::string::npos;
        if (!ok) std::cerr << "lfm2moe fixture: built=" << built << " err=" << err << '\n';
    } catch (const std::exception& e) {
        std::cerr << "lfm2moe fixture exception: " << e.what() << '\n';
        ok = false;
    }
    std::filesystem::remove_all(root);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    const bool include_mtp = argc > 1 && std::string(argv[1]) == "--include-mtp";
    const int first_path = include_mtp ? 2 : 1;
    if (argc == first_path) {
        if (!check_v_head_fixture()) {
            std::cout << "V-head permutation fixture: FAIL\n";
            return 1;
        }
        std::cout << "V-head permutation fixture: OK\n";
        if (!check_raw_gdn_fixture()) {
            std::cout << "raw GDN mapping fixture: FAIL\n";
            return 1;
        }
        std::cout << "raw GDN mapping fixture: OK\n";
        if (!check_lfm2moe_fixture()) {
            std::cout << "lfm2moe recognition fixture: FAIL\n";
            return 1;
        }
        std::cout << "lfm2moe recognition fixture: OK\n";
        const auto root = std::filesystem::temp_directory_path() /
            ("xestrata_exl3_model_map_unknown_" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        std::ofstream(root / "config.json") << R"({"model_type":"unsupported_map_fixture"})";
        {
            const std::string header = R"({"dummy.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
            std::ofstream out(root / "model.safetensors", std::ios::binary);
            const uint64_t size = header.size();
            for (int i = 0; i < 8; ++i) out.put(static_cast<char>(size >> (8 * i)));
            out.write(header.data(), static_cast<std::streamsize>(header.size()));
            for (int i = 0; i < 4; ++i) out.put(0);
        }
        bool rejected = false;
        try {
            strata::artifact::exl3::Archive archive(root);
            strata::artifact::exl3::ModelMap map;
            std::string err;
            rejected = !strata::artifact::exl3::build_model_map(archive, map, err) && !err.empty();
        } catch (const std::exception& e) {
            std::cerr << "unexpected archive failure: " << e.what() << '\n';
        }
        std::filesystem::remove_all(root);
        std::cout << "unknown model rejection: " << (rejected ? "OK" : "FAIL") << '\n';
        return rejected ? 0 : 1;
    }
    int failures = 0;
    for (int i = first_path; i < argc; ++i) {
        try {
            strata::artifact::exl3::Archive archive(argv[i]);
            strata::artifact::exl3::ModelMap map;
            std::string err;
            const bool built = strata::artifact::exl3::build_model_map(archive, map, err, include_mtp);
            bool useful = built && !map.projections.empty() && map.geometry.hidden_size > 0;
            if (useful && map.family == strata::artifact::exl3::ModelFamily::qwen35) {
                useful = has_nontrivial_v_map(map, strata::artifact::exl3::ProjectionKind::gdn_qkv) &&
                         has_nontrivial_v_map(map, strata::artifact::exl3::ProjectionKind::gdn_out) &&
                         has_mapping(map, strata::artifact::exl3::ProjectionKind::gdn_alpha) &&
                         has_mapping(map, strata::artifact::exl3::ProjectionKind::auxiliary,
                                     strata::artifact::exl3::ValueTransform::add_one) &&
                         has_mapping(map, strata::artifact::exl3::ProjectionKind::auxiliary,
                                     strata::artifact::exl3::ValueTransform::negative_exp);
            }
            if (useful && map.family == strata::artifact::exl3::ModelFamily::qwen38) {
                bool indexer_split = false;
                for (const auto& item : map.projections)
                    if (item.kind == strata::artifact::exl3::ProjectionKind::qsa_indexer_q &&
                        item.output_axis.transform == strata::artifact::exl3::AxisTransform::slices)
                        indexer_split = true;
                useful = indexer_split && has_nontrivial_v_map(map, strata::artifact::exl3::ProjectionKind::gdn_out);
                if (useful) {
                    // Qwen3.8 MTP is a separate one-layer map.  Build it from
                    // the trunk metadata without enabling the legacy public
                    // include_mtp switch (that switch remains Qwen3.5-only).
                    strata::artifact::exl3::ModelMap mtp;
                    std::string mtp_err;
                    const bool mtp_built = strata::artifact::exl3::build_qwen38_mtp_map(archive, map, mtp, mtp_err);
                    size_t experts = 0, fc_count = 0, qsa_count = 0, hc_count = 0, add_one = 0;
                    for (const auto& item : mtp.projections) {
                        experts += item.kind == strata::artifact::exl3::ProjectionKind::expert_gate ||
                                   item.kind == strata::artifact::exl3::ProjectionKind::expert_up ||
                                   item.kind == strata::artifact::exl3::ProjectionKind::expert_down;
                        fc_count += item.kind == strata::artifact::exl3::ProjectionKind::mtp_fc;
                        qsa_count += item.kind == strata::artifact::exl3::ProjectionKind::qsa_q ||
                               item.kind == strata::artifact::exl3::ProjectionKind::qsa_k ||
                               item.kind == strata::artifact::exl3::ProjectionKind::qsa_v ||
                               item.kind == strata::artifact::exl3::ProjectionKind::qsa_out ||
                               item.kind == strata::artifact::exl3::ProjectionKind::qsa_indexer_q ||
                               item.kind == strata::artifact::exl3::ProjectionKind::qsa_indexer_k;
                        hc_count += item.kind == strata::artifact::exl3::ProjectionKind::hc_norm ||
                              item.kind == strata::artifact::exl3::ProjectionKind::hc_down ||
                              item.kind == strata::artifact::exl3::ProjectionKind::hc_up ||
                              item.kind == strata::artifact::exl3::ProjectionKind::hc_inject;
                        add_one += item.value_transform == strata::artifact::exl3::ValueTransform::add_one;
                    }
                    const auto* fc = archive.find_linear("mtp.fc_embedding");
                    const auto* shared = archive.find_linear("mtp.layers.0.mlp.shared_expert.gate_proj");
                    const auto* expert = archive.find_linear("mtp.layers.0.mlp.experts.0.gate_proj");
                    const bool mixed_bits = fc != nullptr && shared != nullptr && expert != nullptr &&
                                            fc->bits == 4 && shared->bits == 5 && expert->bits == 3;
                    useful = mtp_built && mtp.unsupported.empty() && mtp.geometry.layers == 1 &&
                             experts == static_cast<size_t>(3 * map.geometry.experts) && fc_count == 2 && qsa_count == 6 &&
                             hc_count == 8 && add_one == 8 && mixed_bits;
                    if (!useful && !mtp_err.empty()) std::cerr << mtp_err << '\n';
                }
            }
            useful = useful && has_mapping(map, strata::artifact::exl3::ProjectionKind::embedding) &&
                     has_mapping(map, strata::artifact::exl3::ProjectionKind::lm_head);
            if (map.family == strata::artifact::exl3::ModelFamily::qwen35)
                useful = useful && has_mapping(map, strata::artifact::exl3::ProjectionKind::final_norm,
                                               strata::artifact::exl3::ValueTransform::add_one);
            if (useful && include_mtp && map.family == strata::artifact::exl3::ModelFamily::qwen35) {
                bool mtp_fc = false, enorm = false, hnorm = false, shared_head_norm = false;
                for (const auto& item : map.projections) {
                    mtp_fc = mtp_fc || (item.layer == map.geometry.layers &&
                                         item.kind == strata::artifact::exl3::ProjectionKind::mtp_fc &&
                                         item.canonical_name == "blk." + std::to_string(map.geometry.layers) + ".nextn.eh_proj.weight");
                    enorm = enorm || item.canonical_name == "blk." + std::to_string(map.geometry.layers) + ".nextn.enorm.weight";
                    hnorm = hnorm || item.canonical_name == "blk." + std::to_string(map.geometry.layers) + ".nextn.hnorm.weight";
                    shared_head_norm = shared_head_norm || item.canonical_name ==
                        "blk." + std::to_string(map.geometry.layers) + ".nextn.shared_head_norm.weight";
                }
                useful = map.geometry.mtp_layers == 1 && !map.geometry.dedicated_embedding &&
                         mtp_fc && enorm && hnorm && shared_head_norm;
            }
            std::cout << argv[i] << ": " << (useful ? "map OK" : "map FAIL") << ", family "
                      << strata::artifact::exl3::model_family_name(map.family) << ", projections "
                      << map.projections.size() << ", unsupported " << map.unsupported.size()
                      << ", include_mtp " << (include_mtp ? "yes" : "no") << '\n';
            if (!useful) {
                std::cerr << err << '\n';
                ++failures;
            }
        } catch (const std::exception& e) {
            std::cerr << argv[i] << ": " << e.what() << '\n';
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
