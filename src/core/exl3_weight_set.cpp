// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Per-device EXL3 dense and auxiliary-weight ownership.
#include "strata/core/exl3_weight_set.hpp"

#include "strata/artifact/exl3_raw.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/runtime.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::core {
namespace {

using namespace strata::artifact::exl3;

bool product(int64_t a, int64_t b, int64_t& result) {
    if (a <= 0 || b <= 0 || a > std::numeric_limits<int64_t>::max() / b) return false;
    result = a * b;
    return true;
}

bool product_u64(uint64_t a, uint64_t b, uint64_t& result) {
    if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b) return false;
    result = a * b;
    return true;
}

bool dense_kind(ProjectionKind kind) {
    switch (kind) {
        case ProjectionKind::conv_in:
        case ProjectionKind::conv_out:
        case ProjectionKind::dense_gate:
        case ProjectionKind::dense_up:
        case ProjectionKind::dense_down:
        case ProjectionKind::gdn_qkv:
        case ProjectionKind::gdn_z:
        case ProjectionKind::gdn_out:
        case ProjectionKind::gdn_alpha:
        case ProjectionKind::gdn_beta:
        case ProjectionKind::qsa_q:
        case ProjectionKind::qsa_k:
        case ProjectionKind::qsa_v:
        case ProjectionKind::qsa_out:
        case ProjectionKind::qsa_indexer_q:
        case ProjectionKind::qsa_indexer_k:
        case ProjectionKind::router:
        case ProjectionKind::shared_scalar:
        case ProjectionKind::shared_gate:
        case ProjectionKind::shared_up:
        case ProjectionKind::shared_down:
        case ProjectionKind::hc_down:
        case ProjectionKind::hc_up:
        case ProjectionKind::hc_inject:
        case ProjectionKind::lm_head:
            return true;
        default:
            return false;
    }
}

bool expert_kind(ProjectionKind kind) {
    return kind == ProjectionKind::expert_gate || kind == ProjectionKind::expert_up ||
           kind == ProjectionKind::expert_down;
}

bool axis_elements(const AxisMapping& axis, std::vector<int64_t>& result, std::string& err) {
    result.clear();
    if (!axis.permutation.empty()) {
        result = axis.permutation;
        return true;
    }
    if (axis.transform == AxisTransform::identity || axis.transform == AxisTransform::raw_rows) return true;
    if (axis.transform == AxisTransform::h128_block_permutation) {
        if (axis.block_size <= 0 || axis.block_offset < 0 || axis.target_size <= 0 ||
            axis.target_size < axis.block_offset || axis.source_size <= 0 ||
            axis.block_permutation.size() != static_cast<size_t>((axis.target_size - axis.block_offset) /
                                                                  axis.block_size) ||
            (axis.target_size - axis.block_offset) % axis.block_size != 0) {
            err = "EXL3 raw axis has an invalid block map";
            return false;
        }
        result.resize(static_cast<size_t>(axis.target_size));
        for (int64_t i = 0; i < axis.target_size; ++i) result[static_cast<size_t>(i)] = i;
        for (size_t block = 0; block < axis.block_permutation.size(); ++block)
            for (int64_t lane = 0; lane < axis.block_size; ++lane) {
                const int64_t target = axis.block_offset + static_cast<int64_t>(block) * axis.block_size + lane;
                const int64_t source = axis.block_offset + axis.block_permutation[block] * axis.block_size + lane;
                if (source < 0 || source >= axis.source_size) {
                    err = "EXL3 raw block map is outside the source";
                    return false;
                }
                result[static_cast<size_t>(target)] = source;
            }
        return true;
    }
    if (axis.transform == AxisTransform::slices) {
        if (axis.target_size <= 0) {
            err = "EXL3 raw slice map has no target";
            return false;
        }
        result.assign(static_cast<size_t>(axis.target_size), -1);
        for (const auto& slice : axis.slices)
            for (int64_t i = 0; i < slice.elements; ++i) {
                const int64_t target = slice.target_offset + i;
                const int64_t source = slice.source_offset + i;
                if (target < 0 || target >= axis.target_size || source < 0 || source >= axis.source_size ||
                    result[static_cast<size_t>(target)] != -1) {
                    err = "EXL3 raw slices overlap or are outside bounds";
                    return false;
                }
                result[static_cast<size_t>(target)] = source;
            }
        if (std::find(result.begin(), result.end(), -1) != result.end()) {
            err = "EXL3 raw slices do not cover the target";
            return false;
        }
        return true;
    }
    err = "EXL3 raw axis transform is unsupported";
    return false;
}

bool raw_axes_for(const ProjectionMapping& mapping, size_t rank,
                  std::vector<std::vector<int64_t>>& axes, std::string& err) {
    axes.clear();
    if (mapping.raw_axes.empty()) return true;
    if (mapping.raw_axes.size() != rank) {
        err = mapping.source_name + ": raw axis rank does not match tensor rank";
        return false;
    }
    axes.reserve(rank);
    for (const auto& axis : mapping.raw_axes) {
        std::vector<int64_t> values;
        if (!axis_elements(axis, values, err)) return false;
        axes.push_back(std::move(values));
    }
    return true;
}

bool packed_blocks(const AxisMapping& axis, std::vector<int64_t>& blocks, std::string& err) {
    std::vector<int64_t> elements;
    if (!axis_elements(axis, elements, err)) return false;
    if (elements.empty()) {
        blocks.clear();
        return true;
    }
    if (axis.source_size <= 0 || axis.source_size % 128 != 0 || elements.size() % 128 != 0) {
        err = "EXL3 packed axis map is not H128 aligned";
        return false;
    }
    blocks.resize(elements.size() / 128);
    for (size_t block = 0; block < blocks.size(); ++block) {
        const int64_t first = elements[block * 128];
        if (first < 0 || first % 128 != 0 || first / 128 >= axis.source_size / 128) {
            err = "EXL3 packed axis starts outside an H128 block";
            return false;
        }
        for (int lane = 0; lane < 128; ++lane)
            if (elements[block * 128 + static_cast<size_t>(lane)] != first + lane) {
                err = "EXL3 packed axis reorders values inside an H128 block";
                return false;
            }
        blocks[block] = first / 128;
    }
    return true;
}

bool packed_storage_bytes(const LinearDescriptor& descriptor, int64_t k, int64_t n, uint64_t& bytes,
                          std::string& err) {
    if (k <= 0 || n <= 0 || k % 16 != 0 || n % 16 != 0 || descriptor.bits < 1 || descriptor.bits > 8) {
        err = descriptor.name + ": invalid packed target geometry";
        return false;
    }
    uint64_t trellis = 0, tiles = 0, words = 0;
    if (!product_u64(static_cast<uint64_t>(k / 16), static_cast<uint64_t>(n / 16), tiles) ||
        !product_u64(tiles, static_cast<uint64_t>(16 * descriptor.bits), words) ||
        !product_u64(words, sizeof(uint16_t), trellis)) {
        err = descriptor.name + ": packed storage size overflows";
        return false;
    }
    uint64_t scales = 0, bias = 0;
    if (!product_u64(static_cast<uint64_t>(k), sizeof(uint16_t), scales) ||
        !product_u64(static_cast<uint64_t>(n), sizeof(uint16_t), bias)) {
        err = descriptor.name + ": packed companion size overflows";
        return false;
    }
    if (!descriptor.bias.has_value()) bias = 0;
    if (trellis > std::numeric_limits<uint64_t>::max() - scales ||
        trellis + scales > std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(n) * sizeof(uint16_t)) {
        err = descriptor.name + ": packed storage size overflows";
        return false;
    }
    bytes = trellis + scales + static_cast<uint64_t>(n) * sizeof(uint16_t);
    if (descriptor.bias.has_value()) bytes = trellis + scales + bias;
    return true;
}

bool raw_dtype_supported(const std::string& dtype) {
    return dtype == "F16" || dtype == "BF16" || dtype == "F32";
}

bool unrelated_unsupported(const std::string& reason) {
    // The first bundle intentionally leaves the specialized PLE and MTP
    // owners to their own bridges. A missing/invalid trunk projection must
    // still stop this bundle rather than silently leaving a null WeightRef.
    return reason == "Qwen3.8 PLE ngram sidecar binding is pending in ModelMap" ||
           reason.rfind("MTP ", 0) == 0;
}

std::string remove_weight_suffix(std::string name) {
    constexpr const char* suffix = ".weight";
    if (name.size() >= 7 && name.compare(name.size() - 7, 7, suffix) == 0) name.resize(name.size() - 7);
    return name;
}

}  // namespace

Exl3WeightSet::~Exl3WeightSet() { close(); }

bool Exl3WeightSet::register_ref(const std::string& name, const WeightRef& value, Exl3Dense* owner,
                                  const void* data, std::string& err) {
    if (table_ == nullptr || name.empty()) {
        err = "EXL3 weight set: table and canonical name are required";
        return false;
    }
    if ((owner == nullptr) == (data == nullptr)) {
        err = "EXL3 weight set: registration must have exactly one owner or data pointer: " + name;
        return false;
    }
    if (std::find_if(registrations_.begin(), registrations_.end(), [&](const Registration& item) {
            return item.name == name;
        }) != registrations_.end()) {
        err = "EXL3 weight set: canonical row is mapped more than once: " + name;
        return false;
    }
    auto it = table_->table_.find(name);
    Registration registration;
    registration.name = name;
    registration.owner = owner;
    registration.data = data;
    if (it == table_->table_.end()) {
        table_->table_.emplace(name, value);
        registration.inserted = true;
    } else {
        if (it->second.native_data != nullptr || it->second.exl3 != nullptr) {
            err = "EXL3 weight set: canonical row already has a native or EXL3 owner: " + name;
            return false;
        }
        if ((value.ne0 > 0 && it->second.ne0 > 0 && value.ne0 != it->second.ne0) ||
            (value.ne1 > 0 && it->second.ne1 > 0 && value.ne1 != it->second.ne1) ||
            (value.elements > 0 && it->second.elements > 0 && value.elements != it->second.elements)) {
            err = "EXL3 weight set: canonical row shape conflicts with the existing table: " + name;
            return false;
        }
        registration.before = it->second;
        it->second = value;
    }
    registrations_.push_back(std::move(registration));
    return true;
}

bool Exl3WeightSet::register_matrix(const std::string& canonical, Exl3Dense& owner, int64_t input,
                                    int64_t output, uint64_t bytes, std::string& err) {
    int64_t elements = 0;
    if (!product(input, output, elements)) {
        err = "EXL3 weight set: matrix shape overflows: " + canonical;
        return false;
    }
    WeightRef ref;
    ref.ne0 = input;
    ref.ne1 = output;
    ref.elements = elements;
    ref.bytes = bytes;
    ref.kind = WeightKind::Verbatim;
    ref.exl3 = &owner;
    ref.resident = true;
    return register_ref(canonical, ref, &owner, nullptr, err);
}

bool Exl3WeightSet::register_aux(const std::string& canonical, const float* data, int64_t elements,
                                 int64_t ne0, int64_t ne1, uint64_t bytes, std::string& err) {
    if (data == nullptr || elements <= 0 || ne0 <= 0 || ne1 <= 0) {
        err = "EXL3 weight set: invalid auxiliary registration: " + canonical;
        return false;
    }
    WeightRef ref;
    ref.data = data;
    ref.bytes = bytes;
    ref.ne0 = ne0;
    ref.ne1 = ne1;
    ref.elements = elements;
    ref.kind = WeightKind::F32;
    ref.resident = true;
    return register_ref(canonical, ref, nullptr, data, err);
}

bool Exl3WeightSet::attach_mapping(const Archive& archive, const ProjectionMapping& mapping, int64_t max_rows,
                                   std::string& err) {
    if (mapping.canonical_name.empty() || mapping.source_name.empty()) {
        err = "EXL3 weight set: mapping has an empty source or canonical name";
        return false;
    }
    if (mapping.transform == MappingTransform::raw_rows) {
        const TensorDescriptor* tensor = archive.find_tensor(mapping.source_name);
        if (tensor == nullptr) {
            err = "EXL3 weight set: raw tensor is missing: " + mapping.source_name;
            return false;
        }
        if (!raw_dtype_supported(tensor->dtype)) {
            err = "EXL3 weight set: unsupported raw dtype " + tensor->dtype + ": " + mapping.source_name;
            return false;
        }
        const std::string base = remove_weight_suffix(mapping.source_name);
        if (dense_kind(mapping.kind) && archive.find_linear(base) != nullptr) {
            err = "EXL3 weight set: raw and packed forms are ambiguous: " + base;
            return false;
        }
        std::vector<std::vector<int64_t>> axes;
        if (!raw_axes_for(mapping, tensor->shape.size(), axes, err)) return false;
        RawTensor raw;
        if (!materialize_raw(archive, *tensor, axes, mapping.value_transform, raw, err)) return false;
        if (dense_kind(mapping.kind)) {
            if (raw.shape.size() != 2 || mapping.target_k <= 0 || mapping.target_n <= 0 ||
                raw.shape[0] != static_cast<uint64_t>(mapping.target_n) ||
                raw.shape[1] != static_cast<uint64_t>(mapping.target_k)) {
                err = "EXL3 weight set: raw matrix shape does not match the canonical row: " + mapping.canonical_name;
                return false;
            }
            RawTensor bias;
            const RawTensor* bias_ptr = nullptr;
            if (const TensorDescriptor* bias_descriptor = archive.find_tensor(base + ".bias"); bias_descriptor != nullptr) {
                if (!raw_dtype_supported(bias_descriptor->dtype) || bias_descriptor->shape.size() != 1 ||
                    bias_descriptor->shape[0] != static_cast<uint64_t>(mapping.target_n)) {
                    err = "EXL3 weight set: raw bias does not match the canonical output width: " + base;
                    return false;
                }
                if (!materialize_raw(archive, *bias_descriptor, {}, ValueTransform::identity, bias, err)) return false;
                bias_ptr = &bias;
            }
            auto owner = std::make_unique<Exl3Dense>();
            if (!owner->load_raw_detached(raw, bias_ptr, err, max_rows)) return false;
            uint64_t bytes = raw.data.size();
            if (bias_ptr != nullptr) {
                uint64_t bias_bytes = 0;
                if (!product_u64(static_cast<uint64_t>(mapping.target_n), sizeof(float), bias_bytes) ||
                    bytes > std::numeric_limits<uint64_t>::max() - bias_bytes) {
                    err = "EXL3 weight set: raw matrix storage size overflows: " + mapping.canonical_name;
                    return false;
                }
                bytes += bias_bytes;
            }
            Exl3Dense* ptr = owner.get();
            if (!register_matrix(mapping.canonical_name, *ptr, mapping.target_k, mapping.target_n, bytes, err))
                return false;
            dense_.push_back(std::move(owner));
            return true;
        }
        std::vector<float> values;
        if (!raw_to_f32(raw, values, err)) return false;
        if (values.empty() || values.size() > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) / sizeof(float)) {
            err = "EXL3 weight set: raw auxiliary size is invalid: " + mapping.source_name;
            return false;
        }
        int64_t ne0 = 0, ne1 = 1, elements = 0;
        if (raw.shape.size() == 1) {
            if (raw.shape[0] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                err = "EXL3 weight set: auxiliary shape exceeds int64: " + mapping.source_name;
                return false;
            }
            ne0 = static_cast<int64_t>(raw.shape[0]);
            elements = ne0;
        } else if ((raw.shape.size() == 2 || (raw.shape.size() == 3 && raw.shape[1] == 1)) &&
                   mapping.canonical_name.find("ssm_conv1d.weight") != std::string::npos) {
            if (raw.shape.front() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
                raw.shape.back() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                err = "EXL3 weight set: convolution shape exceeds int64: " + mapping.source_name;
                return false;
            }
            // The canonical layer stores [d_conv, conv_channels], while the
            // HF raw tensor is [conv_channels, d_conv] or [conv_channels, 1, d_conv].
            ne0 = static_cast<int64_t>(raw.shape.back());
            ne1 = static_cast<int64_t>(raw.shape.front());
            if (!product(ne0, ne1, elements)) {
                err = "EXL3 weight set: convolution shape overflows: " + mapping.source_name;
                return false;
            }
        } else {
            err = "EXL3 weight set: unsupported auxiliary rank for " + mapping.canonical_name;
            return false;
        }
        if (static_cast<size_t>(elements) != values.size()) {
            err = "EXL3 weight set: auxiliary element count does not match " + mapping.canonical_name;
            return false;
        }
        const uint64_t bytes = static_cast<uint64_t>(values.size()) * sizeof(float);
        void* device = strata::gpu::alloc_device(static_cast<size_t>(bytes));
        if (device == nullptr || !strata::gpu::copy(device, values.data(), static_cast<size_t>(bytes))) {
            if (device != nullptr) strata::gpu::free(device);
            err = "EXL3 weight set: auxiliary upload failed for " + mapping.canonical_name + ": " +
                  strata::gpu::last_error();
            return false;
        }
        if (!register_aux(mapping.canonical_name, static_cast<const float*>(device), elements, ne0, ne1, bytes, err)) {
            strata::gpu::free(device);
            return false;
        }
        auxiliary_.push_back(device);
        return true;
    }

    const LinearDescriptor* descriptor = archive.find_linear(mapping.source_name);
    if (descriptor == nullptr) {
        err = "EXL3 weight set: packed linear is missing: " + mapping.source_name;
        return false;
    }
    if (archive.find_tensor(mapping.source_name + ".weight") != nullptr) {
        err = "EXL3 weight set: packed and raw forms are ambiguous: " + mapping.source_name;
        return false;
    }
    if (mapping.target_k <= 0 || mapping.target_n <= 0) {
        err = "EXL3 weight set: packed mapping has invalid target dimensions: " + mapping.canonical_name;
        return false;
    }
    std::vector<int64_t> input_blocks, output_blocks;
    if (!packed_blocks(mapping.input_axis, input_blocks, err) ||
        !packed_blocks(mapping.output_axis, output_blocks, err)) return false;
    auto owner = std::make_unique<Exl3Dense>();
    if (!owner->load_detached(archive, mapping.source_name, mapping.target_k, mapping.target_n, input_blocks,
                              output_blocks, err, max_rows)) return false;
    uint64_t bytes = 0;
    if (!packed_storage_bytes(*descriptor, mapping.target_k, mapping.target_n, bytes, err)) return false;
    Exl3Dense* ptr = owner.get();
    if (!register_matrix(mapping.canonical_name, *ptr, mapping.target_k, mapping.target_n, bytes, err)) return false;
    dense_.push_back(std::move(owner));
    return true;
}

bool Exl3WeightSet::attach_final_hc(const Archive& archive, const ModelMap& map, int64_t max_rows, std::string& err) {
    std::vector<const ProjectionMapping*> mapped;
    for (const auto& mapping : map.projections)
        if (mapping.canonical_name == "output_hc_norm.weight" || mapping.canonical_name == "output_hc_down.weight" ||
            mapping.canonical_name == "output_hc_up.weight") mapped.push_back(&mapping);
    if (!mapped.empty()) {
        if (mapped.size() != 3) {
            err = "EXL3 weight set: final hyper-connection mapping is incomplete";
            return false;
        }
        for (const ProjectionMapping* mapping : mapped)
            if (!attach_mapping(archive, *mapping, max_rows, err)) return false;
        return true;
    }
    const size_t layers_at = map.source_prefix.rfind(".layers");
    if (layers_at == std::string::npos) {
        err = "EXL3 weight set: cannot derive the final mixer source prefix";
        return false;
    }
    const std::string root = map.source_prefix.substr(0, layers_at);
    const std::string source = root + ".hyper_connection_mixer";
    const std::string names[] = {source + ".hc_norm.weight", source + ".input_mix_weight_down.weight",
                                 source + ".input_mix_weight_up.weight"};
    for (const std::string& name : names)
        if (archive.find_tensor(name) == nullptr) {
            err = "EXL3 weight set: final hyper-connection tensor is missing: " + name;
            return false;
        }
    ProjectionMapping norm;
    norm.kind = ProjectionKind::hc_norm;
    norm.transform = MappingTransform::raw_rows;
    norm.value_transform = ValueTransform::add_one;
    norm.source_name = names[0];
    norm.canonical_name = "output_hc_norm.weight";
    ProjectionMapping down;
    down.kind = ProjectionKind::hc_down;
    down.transform = MappingTransform::raw_rows;
    down.source_name = names[1];
    down.canonical_name = "output_hc_down.weight";
    down.target_k = map.geometry.hidden_size * map.geometry.hc_count;
    down.target_n = map.geometry.hc_rank;
    ProjectionMapping up;
    up.kind = ProjectionKind::hc_up;
    up.transform = MappingTransform::raw_rows;
    up.source_name = names[2];
    up.canonical_name = "output_hc_up.weight";
    up.target_k = map.geometry.hc_rank;
    up.target_n = map.geometry.hidden_size * map.geometry.hc_count;
    if (!attach_mapping(archive, norm, max_rows, err) || !attach_mapping(archive, down, max_rows, err) ||
        !attach_mapping(archive, up, max_rows, err)) return false;
    return true;
}

bool Exl3WeightSet::load(const Archive& archive, const ModelMap& map, WeightTable& table,
                         const Exl3WeightSetOptions& options, std::string& err) {
    err.clear();
    if (loaded()) {
        err = "EXL3 weight set: the owner is already loaded";
        return false;
    }
    if (options.max_rows <= 0) {
        err = "EXL3 weight set: max_rows must be positive";
        return false;
    }
    const int64_t end = options.layer_end < 0 ? map.geometry.layers : options.layer_end;
    if (map.geometry.layers <= 0 || options.layer_begin < 0 || end <= options.layer_begin ||
        end > map.geometry.layers) {
        err = "EXL3 weight set: layer range is outside the trunk geometry";
        return false;
    }
    device_ = options.device >= 0 ? options.device : current_device();
    try {
        const OnDevice on(device_);
        for (const std::string& reason : map.unsupported) {
            if (!unrelated_unsupported(reason)) {
                err = "EXL3 weight set: selected model mapping is unsupported: " + reason;
                close();
                return false;
            }
        }
        table_ = &table;
        bool selected = false;
        bool selected_head = false;
        for (const auto& mapping : map.projections) {
            if (mapping.kind == ProjectionKind::embedding ||
                (mapping.kind == ProjectionKind::mtp_fc && !options.include_mtp_block) ||
                expert_kind(mapping.kind)) continue;
            if (mapping.canonical_name.find(".nextn.") != std::string::npos) continue;
            if (mapping.kind == ProjectionKind::lm_head || mapping.kind == ProjectionKind::final_norm) {
                if (!options.include_head) continue;
                if (mapping.kind == ProjectionKind::lm_head) selected_head = true;
            } else if (mapping.layer < options.layer_begin || mapping.layer >= end) {
                // Layer<0 entries are archive globals (the MTP block's fc/norms); the trunk range
                // never takes them, and the MTP block takes them only when asked for explicitly.
                if (!options.include_mtp_block || mapping.layer >= 0) continue;
            }
            if (mapping.kind == ProjectionKind::hc_norm && mapping.layer < 0) continue;
            if (!attach_mapping(archive, mapping, options.max_rows, err)) {
                close();
                return false;
            }
            selected = true;
        }
        if (options.include_final_hc) {
            if (!attach_final_hc(archive, map, options.max_rows, err)) {
                close();
                return false;
            }
            selected = true;
        }
        if (options.include_head && !selected_head) {
            err = "EXL3 weight set: output.weight mapping is missing while the head was requested";
            close();
            return false;
        }
        if (!selected) {
            err = "EXL3 weight set: no selected dense or auxiliary mappings";
            close();
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("EXL3 weight set: ") + e.what();
        close();
        return false;
    }
}

bool Exl3WeightSet::prepare_stream(void* stream, std::string& err) {
    err.clear();
    if (!loaded()) {
        err = "EXL3 weight set: load is required before prepare_stream";
        return false;
    }
    try {
        const OnDevice on(device_);
        for (const auto& owner : dense_)
            if (!owner->prepare_stream(stream, err)) return false;
        if (std::find(prepared_streams_.begin(), prepared_streams_.end(), stream) == prepared_streams_.end())
            prepared_streams_.push_back(stream);
        return true;
    } catch (const std::exception& e) {
        err = std::string("EXL3 weight set: preparing stream: ") + e.what();
        return false;
    }
}

void Exl3WeightSet::reset_rows() noexcept {
    if (table_ == nullptr) return;
    for (auto it = registrations_.rbegin(); it != registrations_.rend(); ++it) {
        auto row = table_->table_.find(it->name);
        if (row == table_->table_.end()) continue;
        const bool ours = (it->owner != nullptr && row->second.exl3 == it->owner) ||
                          (it->data != nullptr && row->second.data == it->data);
        if (!ours) continue;
        if (it->inserted) table_->table_.erase(row);
        else row->second = it->before;
    }
    registrations_.clear();
}

void Exl3WeightSet::reset_owners() noexcept {
    for (void* data : auxiliary_)
        if (data != nullptr) strata::gpu::free(data);
    auxiliary_.clear();
    dense_.clear();
    prepared_streams_.clear();
}

void Exl3WeightSet::close() noexcept {
    if (table_ == nullptr && dense_.empty() && auxiliary_.empty()) return;
    try {
        const OnDevice on(device_ >= 0 ? device_ : -1);
        (void) strata::gpu::device_sync();
        reset_rows();
        reset_owners();
    } catch (...) {
        // Destruction cannot report an error. The normal path above is also
        // what makes gpu::free select the allocation's owning runtime.
        registrations_.clear();
        auxiliary_.clear();
        dense_.clear();
        prepared_streams_.clear();
    }
    table_ = nullptr;
    device_ = -1;
}

}  // namespace strata::core
