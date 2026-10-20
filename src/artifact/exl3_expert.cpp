// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU-side EXL3 expert blob materialization; packed rules derive from exllamav3's MIT implementation.
#include "strata/artifact/exl3_expert.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::artifact::exl3 {
namespace {

using Matrix = strata::kernels::exl3::ExpertProjectionLayout;
using ComponentLayout = strata::kernels::exl3::ExpertComponentLayout;
using Storage = strata::kernels::exl3::ProjectionStorage;
using DType = strata::kernels::raw::DType;
using Materialized = MaterializedMatrix;

struct ResolvedProjection {
    Storage storage = Storage::packed_trellis;
    const LinearDescriptor* packed = nullptr;
    const TensorDescriptor* weights = nullptr;
    const TensorDescriptor* bias = nullptr;
    DType dtype = DType::F32;
    DType bias_dtype = DType::F32;
};

bool checked_add(uint64_t a, uint64_t b, uint64_t& out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    out = a + b;
    return true;
}

uint64_t align_up(uint64_t value, size_t alignment) {
    const uint64_t a = static_cast<uint64_t>(alignment);
    const uint64_t remainder = value % a;
    return remainder == 0 ? value : value + (a - remainder);
}

bool dtype_of(const std::string& text, DType& dtype) {
    if (text == "F16") dtype = DType::F16;
    else if (text == "BF16") dtype = DType::BF16;
    else if (text == "F32") dtype = DType::F32;
    else return false;
    return true;
}

std::string packed_name(const std::string& name) {
    constexpr const char* suffix = ".weight";
    if (name.size() > std::strlen(suffix) && name.compare(name.size() - std::strlen(suffix), std::strlen(suffix), suffix) == 0)
        return name.substr(0, name.size() - std::strlen(suffix));
    return name;
}

std::string raw_name(const std::string& name) {
    constexpr const char* suffix = ".weight";
    if (name.size() > std::strlen(suffix) && name.compare(name.size() - std::strlen(suffix), std::strlen(suffix), suffix) == 0)
        return name;
    return name + suffix;
}

bool resolve_projection(const Archive& archive, const std::string& name, const char* role,
                        ResolvedProjection& result, std::string& err) {
    result = {};
    const std::string packed = packed_name(name);
    const std::string raw = raw_name(name);
    result.packed = archive.find_linear(packed);
    result.weights = archive.find_tensor(raw);
    if (result.packed != nullptr && result.weights != nullptr) {
        err = std::string("exl3 expert: ") + role + " has both packed and raw representations";
        return false;
    }
    if (result.packed != nullptr) {
        result.storage = Storage::packed_trellis;
        return true;
    }
    if (result.weights == nullptr) {
        err = std::string("exl3 expert: ") + role + " is missing packed trellis or raw .weight tensor";
        return false;
    }
    if (result.weights->shape.size() != 2 || result.weights->shape[0] == 0 || result.weights->shape[1] == 0) {
        err = std::string("exl3 expert: ") + role + " raw weight must have nonempty shape [n,k]";
        return false;
    }
    if (!dtype_of(result.weights->dtype, result.dtype)) {
        err = std::string("exl3 expert: ") + role + " raw weight dtype must be F16, BF16 or F32";
        return false;
    }
    const std::string bias = packed_name(name) + ".bias";
    result.bias = archive.find_tensor(bias);
    if (result.bias != nullptr) {
        if (result.bias->shape != std::vector<uint64_t>{result.weights->shape[0]} ||
            !dtype_of(result.bias->dtype, result.bias_dtype)) {
            err = std::string("exl3 expert: ") + role + " raw bias must be [n] with dtype F16, BF16 or F32";
            return false;
        }
    }
    result.storage = Storage::raw;
    return true;
}

void copy_bytes(std::vector<uint64_t>& destination, const ComponentLayout& component, const void* source,
                size_t bytes) {
    if (bytes != component.bytes) throw std::runtime_error("materialized component size differs from metadata layout");
    if (bytes == 0) return;
    if (source == nullptr) throw std::runtime_error("materialized component has a null source");
    std::memcpy(reinterpret_cast<uint8_t*>(destination.data()) + component.offset, source, bytes);
}

void metadata_component(Matrix& layout, ComponentLayout Matrix::*member, uint64_t bytes, uint64_t& cursor,
                        size_t alignment) {
    cursor = align_up(cursor, alignment);
    ComponentLayout& destination = layout.*member;
    destination.offset = cursor;
    destination.bytes = bytes;
    if (!checked_add(cursor, bytes, cursor)) throw std::runtime_error("expert blob size overflows uint64");
}

void metadata_packed(Matrix& layout, const LinearDescriptor& descriptor, uint64_t& cursor, size_t alignment) {
    layout = {};
    layout.storage = Storage::packed_trellis;
    layout.k = descriptor.k;
    layout.n = descriptor.n;
    layout.bits = descriptor.bits;
    layout.codebook = descriptor.codebook;
    metadata_component(layout, &Matrix::trellis, descriptor.trellis.bytes, cursor, alignment);
    metadata_component(layout, &Matrix::suh, static_cast<uint64_t>(descriptor.k) * sizeof(uint16_t), cursor, alignment);
    metadata_component(layout, &Matrix::svh, static_cast<uint64_t>(descriptor.n) * sizeof(uint16_t), cursor, alignment);
    metadata_component(layout, &Matrix::bias,
                       descriptor.bias.has_value() ? static_cast<uint64_t>(descriptor.n) * sizeof(uint16_t) : 0,
                       cursor, alignment);
}

void metadata_raw(Matrix& layout, const ResolvedProjection& descriptor, uint64_t& cursor, size_t alignment) {
    layout = {};
    layout.storage = Storage::raw;
    layout.k = static_cast<int64_t>(descriptor.weights->shape[1]);
    layout.n = static_cast<int64_t>(descriptor.weights->shape[0]);
    layout.raw_dtype = descriptor.dtype;
    layout.raw_bias_dtype = descriptor.bias_dtype;
    metadata_component(layout, &Matrix::raw_weights, descriptor.weights->bytes, cursor, alignment);
    metadata_component(layout, &Matrix::raw_bias, descriptor.bias == nullptr ? 0 : descriptor.bias->bytes, cursor, alignment);
}

bool valid_packed(const LinearDescriptor& descriptor, const char* role, std::string& err) {
    if (descriptor.k <= 0 || descriptor.n <= 0 || descriptor.k % 128 != 0 || descriptor.n % 128 != 0) {
        err = std::string("exl3 expert: ") + role + " packed projection is not 128-aligned";
        return false;
    }
    if (descriptor.bits < 1 || descriptor.bits > 8 || descriptor.codebook < 0 || descriptor.codebook > 2) {
        err = std::string("exl3 expert: ") + role + " packed projection has unsupported bits/codebook";
        return false;
    }
    return true;
}

bool valid_raw(const ResolvedProjection& descriptor, const char* role, std::string& err) {
    if (descriptor.weights == nullptr || descriptor.weights->shape.size() != 2 || descriptor.weights->shape[0] == 0 ||
        descriptor.weights->shape[1] == 0 || descriptor.weights->shape[0] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        descriptor.weights->shape[1] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        err = std::string("exl3 expert: ") + role + " raw projection geometry is invalid";
        return false;
    }
    return true;
}

bool projection_geometry(const ResolvedProjection& descriptor, int64_t& k, int64_t& n, const char* role,
                         std::string& err) {
    if (descriptor.storage == Storage::packed_trellis) {
        if (descriptor.packed == nullptr || !valid_packed(*descriptor.packed, role, err)) return false;
        k = descriptor.packed->k;
        n = descriptor.packed->n;
        return true;
    }
    if (!valid_raw(descriptor, role, err)) return false;
    k = static_cast<int64_t>(descriptor.weights->shape[1]);
    n = static_cast<int64_t>(descriptor.weights->shape[0]);
    return true;
}

bool checked_storage_words(uint64_t bytes, size_t& words) {
    if (bytes % sizeof(uint64_t) != 0 || bytes / sizeof(uint64_t) > std::numeric_limits<size_t>::max()) return false;
    words = static_cast<size_t>(bytes / sizeof(uint64_t));
    return true;
}

/// The packed trellis of one projection, in the given tile order.
bool packed_tiles_supported(int64_t k, int64_t n, int bits) {
    return k > 0 && n > 0 && k % 16 == 0 && n % 16 == 0 && bits >= 1 && bits <= 8;
}

}  // namespace

void copy_packed_tiles(const uint16_t* source, uint16_t* destination, int64_t k, int64_t n, int bits,
                        strata::kernels::exl3::TileOrder from, strata::kernels::exl3::TileOrder to) {
    if (source == nullptr || destination == nullptr || !packed_tiles_supported(k, n, bits)) return;
    const int64_t tiles_k = k / 16;
    const int64_t tiles_n = n / 16;
    const size_t words = static_cast<size_t>(16 * bits);
    for (int64_t tile_n = 0; tile_n < tiles_n; ++tile_n) {
        for (int64_t tile_k = 0; tile_k < tiles_k; ++tile_k) {
            const size_t source_tile = strata::kernels::exl3::trellis_tile_index(from, tile_k, tile_n, tiles_k, tiles_n);
            const size_t destination_tile =
                strata::kernels::exl3::trellis_tile_index(to, tile_k, tile_n, tiles_k, tiles_n);
            std::memcpy(destination + destination_tile * words, source + source_tile * words, words * sizeof(uint16_t));
        }
    }
}

bool describe_expert(const Archive& archive, const ExpertTensorNames& names,
                     strata::kernels::exl3::ExpertBlobLayout& layout, std::string& err, size_t alignment) {
    err.clear();
    if (alignment == 0 || alignment > 4096 || (alignment & (alignment - 1)) != 0 || alignment < alignof(uint32_t)) {
        err = "exl3 expert: alignment must be a power of two in 4..4096";
        return false;
    }
    if (names.gate.empty() || names.up.empty() || names.down.empty()) {
        err = "exl3 expert: gate/up/down names are required";
        return false;
    }
    try {
        ResolvedProjection gate, up, down;
        if (!resolve_projection(archive, names.gate, "gate", gate, err) ||
            !resolve_projection(archive, names.up, "up", up, err) ||
            !resolve_projection(archive, names.down, "down", down, err))
            return false;
        int64_t gate_k = 0, gate_n = 0, up_k = 0, up_n = 0, down_k = 0, down_n = 0;
        if (!projection_geometry(gate, gate_k, gate_n, "gate", err) ||
            !projection_geometry(up, up_k, up_n, "up", err) ||
            !projection_geometry(down, down_k, down_n, "down", err))
            return false;
        if (gate_k != up_k || gate_n != up_n || down_k != gate_n || down_n != gate_k) {
            err = "exl3 expert: gate/up/down shapes do not form hidden-ff-hidden";
            return false;
        }
        strata::kernels::exl3::ExpertBlobLayout result;
        result.alignment = static_cast<uint32_t>(alignment);
        uint64_t cursor = 0;
        if (gate.storage == Storage::packed_trellis) metadata_packed(result.gate, *gate.packed, cursor, alignment);
        else metadata_raw(result.gate, gate, cursor, alignment);
        if (up.storage == Storage::packed_trellis) metadata_packed(result.up, *up.packed, cursor, alignment);
        else metadata_raw(result.up, up, cursor, alignment);
        if (down.storage == Storage::packed_trellis) metadata_packed(result.down, *down.packed, cursor, alignment);
        else metadata_raw(result.down, down, cursor, alignment);
        result.total_bytes = align_up(cursor, alignment);
        layout = result;
        return true;
    } catch (const std::exception& e) {
        err = std::string("exl3 expert: ") + e.what();
        return false;
    }
}

bool materialize_expert(const Archive& archive, const ExpertTensorNames& names, ExpertBlob& output, std::string& err,
                        size_t alignment, strata::kernels::exl3::TileOrder order) {
    strata::kernels::exl3::ExpertBlobLayout metadata;
    if (!describe_expert(archive, names, metadata, err, alignment)) return false;
    // A CPU tile order is applied only where it is defined; every other
    // expert keeps the format's own order.
    const bool cpu_order = strata::kernels::exl3::tile_order_is_cpu(order) &&
                           strata::kernels::exl3::cpu_tile_order_applies(metadata);
    try {
        ResolvedProjection resolved[3];
        const std::string role_names[3] = {names.gate, names.up, names.down};
        const char* roles[3] = {"gate", "up", "down"};
        for (int i = 0; i < 3; ++i)
            if (!resolve_projection(archive, role_names[i], roles[i], resolved[i], err)) return false;

        ExpertBlob result;
        result.layout = metadata;
        result.tile_order = cpu_order ? order : strata::kernels::exl3::TileOrder::canonical;
        size_t words = 0;
        if (!checked_storage_words(metadata.total_bytes, words))
            throw std::runtime_error("expert blob size cannot be represented by aligned storage");
        result.storage.assign(words, 0);
        Matrix* layouts[3] = {&result.layout.gate, &result.layout.up, &result.layout.down};
        for (int i = 0; i < 3; ++i) {
            if (resolved[i].storage == Storage::packed_trellis) {
                Materialized matrix;
                archive.materialize(*resolved[i].packed, matrix);
                if (matrix.trellis.size() * sizeof(uint16_t) != layouts[i]->trellis.bytes ||
                    matrix.suh.size() * sizeof(uint16_t) != layouts[i]->suh.bytes ||
                    matrix.svh.size() * sizeof(uint16_t) != layouts[i]->svh.bytes ||
                    matrix.bias.size() * sizeof(uint16_t) != layouts[i]->bias.bytes)
                    throw std::runtime_error("materialized packed component size differs from metadata layout");
                if (cpu_order) {
                    // The same bytes, in the CPU tile order: only the order of
                    // the whole 16x16 tiles changes.
                    auto* base = reinterpret_cast<uint8_t*>(result.storage.data());
                    copy_packed_tiles(matrix.trellis.data(),
                                      reinterpret_cast<uint16_t*>(base + layouts[i]->trellis.offset), layouts[i]->k,
                                      layouts[i]->n, layouts[i]->bits, strata::kernels::exl3::TileOrder::canonical,
                                      order);
                } else {
                    copy_bytes(result.storage, layouts[i]->trellis, matrix.trellis.data(), layouts[i]->trellis.bytes);
                }
                copy_bytes(result.storage, layouts[i]->suh, matrix.suh.data(), layouts[i]->suh.bytes);
                copy_bytes(result.storage, layouts[i]->svh, matrix.svh.data(), layouts[i]->svh.bytes);
                copy_bytes(result.storage, layouts[i]->bias, matrix.bias.data(), layouts[i]->bias.bytes);
            } else {
                std::vector<uint8_t> raw;
                archive.read_tensor(*resolved[i].weights, raw);
                copy_bytes(result.storage, layouts[i]->raw_weights, raw.data(), layouts[i]->raw_weights.bytes);
                if (resolved[i].bias != nullptr) {
                    std::vector<uint8_t> bias;
                    archive.read_tensor(*resolved[i].bias, bias);
                    copy_bytes(result.storage, layouts[i]->raw_bias, bias.data(), layouts[i]->raw_bias.bytes);
                }
            }
        }
        output = std::move(result);
        return true;
    } catch (const std::exception& e) {
        err = std::string("exl3 expert: ") + e.what();
        return false;
    }
}

namespace {

bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t& result) {
    if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b) return false;
    result = a * b;
    return true;
}

uint64_t align_up_checked(uint64_t value, size_t alignment) {
    const uint64_t a = static_cast<uint64_t>(alignment);
    const uint64_t remainder = value % a;
    if (remainder == 0) return value;
    const uint64_t add = a - remainder;
    if (value > std::numeric_limits<uint64_t>::max() - add)
        throw std::runtime_error("raw expert alignment overflows uint64");
    return value + add;
}

template <typename Finder>
bool valid_slice(Finder&& find, const RawExpertSlice& slice, const char* role, std::string& err) {
    const TensorDescriptor* tensor = find(slice.tensor);
    if (tensor == nullptr) {
        err = std::string("exl3 raw expert: ") + role + " tensor is missing: " + slice.tensor;
        return false;
    }
    DType dtype;
    if (!dtype_of(tensor->dtype, dtype) || dtype != slice.dtype) {
        err = std::string("exl3 raw expert: ") + role + " dtype does not match its descriptor";
        return false;
    }
    if (slice.k <= 0 || slice.n <= 0 || static_cast<uint64_t>(slice.k) >
        std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(slice.n)) {
        err = std::string("exl3 raw expert: ") + role + " has invalid 2-D geometry";
        return false;
    }
    const uint64_t element_bytes = static_cast<uint64_t>(strata::kernels::raw::dtype_bytes(dtype));
    uint64_t elements = 0, expected_bytes = 0;
    if (!checked_mul_u64(static_cast<uint64_t>(slice.k), static_cast<uint64_t>(slice.n), elements) ||
        !checked_mul_u64(elements, element_bytes, expected_bytes) || expected_bytes != slice.bytes) {
        err = std::string("exl3 raw expert: ") + role + " byte geometry does not match its descriptor";
        return false;
    }
    if (slice.relative_offset > tensor->bytes || slice.bytes > tensor->bytes - slice.relative_offset) {
        err = std::string("exl3 raw expert: ") + role + " slice is outside its tensor payload";
        return false;
    }
    return true;
}

void set_raw_layout(Matrix& layout, const RawExpertSlice& slice, uint64_t& cursor, size_t alignment) {
    layout = {};
    layout.storage = Storage::raw;
    layout.k = slice.k;
    layout.n = slice.n;
    layout.raw_dtype = slice.dtype;
    cursor = align_up_checked(cursor, alignment);
    layout.raw_weights.offset = cursor;
    layout.raw_weights.bytes = slice.bytes;
    if (!checked_add(cursor, slice.bytes, cursor)) throw std::runtime_error("raw expert blob size overflows uint64");
}

}  // namespace

template <typename Finder>
bool describe_raw_expert_slices_impl(Finder&& find, const RawExpertTensorNames& names, int64_t expert,
                                     RawExpertSlices& output, std::string& err) {
    err.clear();
    if (names.gate_up.empty() || names.down.empty() || expert < 0 || names.experts < 0 || names.hidden < 0 ||
        names.feed_forward < 0) {
        err = "exl3 raw expert: fused tensor names and nonnegative expert are required";
        return false;
    }
    try {
        const TensorDescriptor* gate_up = find(names.gate_up);
        const TensorDescriptor* down = find(names.down);
        if (gate_up == nullptr || down == nullptr) {
            err = "exl3 raw expert: fused gate_up/down tensor is missing";
            return false;
        }
        DType gate_dtype, down_dtype;
        if (!dtype_of(gate_up->dtype, gate_dtype) || !dtype_of(down->dtype, down_dtype)) {
            err = "exl3 raw expert: fused tensors must be F16, BF16, or F32";
            return false;
        }
        if (gate_up->shape.size() != 3 || down->shape.size() != 3 || gate_up->shape[0] == 0 ||
            gate_up->shape[1] == 0 || gate_up->shape[2] == 0 || down->shape[0] != gate_up->shape[0] ||
            down->shape[1] != gate_up->shape[2] || down->shape[2] > std::numeric_limits<uint64_t>::max() / 2 ||
            down->shape[2] * 2 != gate_up->shape[1]) {
            err = "exl3 raw expert: fused tensors do not have [experts,2*ff,hidden]/[experts,hidden,ff] shape";
            return false;
        }
        const uint64_t experts = gate_up->shape[0];
        const uint64_t hidden = gate_up->shape[2];
        const uint64_t ff = down->shape[2];
        if (expert >= static_cast<int64_t>(experts) ||
            (names.experts > 0 && static_cast<uint64_t>(names.experts) != experts) ||
            (names.hidden > 0 && static_cast<uint64_t>(names.hidden) != hidden) ||
            (names.feed_forward > 0 && static_cast<uint64_t>(names.feed_forward) != ff)) {
            err = "exl3 raw expert: requested expert or configured geometry is out of range";
            return false;
        }
        uint64_t gate_elements = 0, gate_bytes_per_expert = 0;
        uint64_t down_elements = 0, down_bytes_per_expert = 0;
        if (!checked_mul_u64(static_cast<uint64_t>(gate_up->shape[1]), hidden, gate_elements) ||
            !checked_mul_u64(gate_elements, static_cast<uint64_t>(strata::kernels::raw::dtype_bytes(gate_dtype)),
                             gate_bytes_per_expert) ||
            !checked_mul_u64(hidden, ff, down_elements) ||
            !checked_mul_u64(down_elements, static_cast<uint64_t>(strata::kernels::raw::dtype_bytes(down_dtype)),
                             down_bytes_per_expert))
            throw std::runtime_error("fused expert byte size overflows uint64");
        if (gate_bytes_per_expert == 0 || down_bytes_per_expert == 0 ||
            static_cast<uint64_t>(expert) > std::numeric_limits<uint64_t>::max() / gate_bytes_per_expert ||
            static_cast<uint64_t>(expert) > std::numeric_limits<uint64_t>::max() / down_bytes_per_expert)
            throw std::runtime_error("fused expert offset overflows uint64");
        const uint64_t gate_base = static_cast<uint64_t>(expert) * gate_bytes_per_expert;
        const uint64_t down_base = static_cast<uint64_t>(expert) * down_bytes_per_expert;
        uint64_t gate_slice_elements = 0, gate_slice_bytes = 0;
        if (!checked_mul_u64(ff, hidden, gate_slice_elements) ||
            !checked_mul_u64(gate_slice_elements, static_cast<uint64_t>(strata::kernels::raw::dtype_bytes(gate_dtype)),
                             gate_slice_bytes))
            throw std::runtime_error("fused gate/up byte size overflows uint64");
        if (gate_slice_bytes == 0 || gate_base > std::numeric_limits<uint64_t>::max() - gate_slice_bytes)
            throw std::runtime_error("fused gate/up offset overflows uint64");
        RawExpertSlices result;
        result.gate = {names.gate_up, gate_base, gate_slice_bytes, static_cast<int64_t>(hidden),
                       static_cast<int64_t>(ff), gate_dtype};
        result.up = {names.gate_up, gate_base + gate_slice_bytes, gate_slice_bytes, static_cast<int64_t>(hidden),
                     static_cast<int64_t>(ff), gate_dtype};
        result.down = {names.down, down_base, down_bytes_per_expert, static_cast<int64_t>(ff),
                       static_cast<int64_t>(hidden), down_dtype};
        if (!valid_slice(find, result.gate, "gate", err) || !valid_slice(find, result.up, "up", err) ||
            !valid_slice(find, result.down, "down", err)) return false;
        output = std::move(result);
        return true;
    } catch (const std::exception& e) {
        err = std::string("exl3 raw expert: ") + e.what();
        return false;
    }
}

template <typename Finder, typename Reader>
bool materialize_raw_expert_slices_impl(Finder&& find, Reader&& read_range, const RawExpertSlices& slices,
                                        ExpertBlob& output, std::string& err, size_t alignment) {
    err.clear();
    if (alignment == 0 || alignment > 4096 || (alignment & (alignment - 1)) != 0 || alignment < alignof(uint32_t)) {
        err = "exl3 raw expert: alignment must be a power of two in 4..4096";
        return false;
    }
    try {
        if (!valid_slice(find, slices.gate, "gate", err) || !valid_slice(find, slices.up, "up", err) ||
            !valid_slice(find, slices.down, "down", err)) return false;
        if (slices.gate.tensor != slices.up.tensor ||
            slices.gate.relative_offset > std::numeric_limits<uint64_t>::max() - slices.gate.bytes ||
            slices.gate.relative_offset + slices.gate.bytes != slices.up.relative_offset) {
            err = "exl3 raw expert: gate/up slices are not contiguous within one fused tensor";
            return false;
        }
        if (slices.gate.k != slices.up.k || slices.gate.n != slices.up.n || slices.down.k != slices.gate.n ||
            slices.down.n != slices.gate.k) {
            err = "exl3 raw expert: slices do not form hidden-ff-hidden projections";
            return false;
        }
        const auto* gate_tensor = find(slices.gate.tensor);
        const auto* down_tensor = find(slices.down.tensor);
        if (gate_tensor == nullptr || down_tensor == nullptr || gate_tensor->shape.size() != 3 ||
            down_tensor->shape.size() != 3 || gate_tensor->shape[0] != down_tensor->shape[0] ||
            gate_tensor->shape[1] != 2 * static_cast<uint64_t>(slices.gate.n) ||
            gate_tensor->shape[2] != static_cast<uint64_t>(slices.gate.k) ||
            down_tensor->shape[1] != static_cast<uint64_t>(slices.down.n) ||
            down_tensor->shape[2] != static_cast<uint64_t>(slices.down.k) ||
            slices.gate.bytes > std::numeric_limits<uint64_t>::max() / 2 ||
            slices.gate.relative_offset % (2 * slices.gate.bytes) != 0 ||
            slices.down.relative_offset % slices.down.bytes != 0 ||
            slices.gate.relative_offset / (2 * slices.gate.bytes) !=
                slices.down.relative_offset / slices.down.bytes) {
            err = "exl3 raw expert: slices do not select the same expert from fused 3-D tensors";
            return false;
        }
        ExpertBlob result;
        result.layout.alignment = static_cast<uint32_t>(alignment);
        uint64_t cursor = 0;
        set_raw_layout(result.layout.gate, slices.gate, cursor, alignment);
        set_raw_layout(result.layout.up, slices.up, cursor, alignment);
        set_raw_layout(result.layout.down, slices.down, cursor, alignment);
        result.layout.total_bytes = align_up_checked(cursor, alignment);
        size_t words = 0;
        if (!checked_storage_words(result.layout.total_bytes, words))
            throw std::runtime_error("raw expert blob size cannot be represented by aligned storage");
        result.storage.assign(words, 0);
        const RawExpertSlice* source_slices[3] = {&slices.gate, &slices.up, &slices.down};
        ComponentLayout* destinations[3] = {&result.layout.gate.raw_weights, &result.layout.up.raw_weights,
                                            &result.layout.down.raw_weights};
        for (int i = 0; i < 3; ++i) {
            std::vector<uint8_t> raw;
            const TensorDescriptor* descriptor = find(source_slices[i]->tensor);
            if (descriptor == nullptr) throw std::runtime_error("fused slice tensor disappeared from archive");
            read_range(*descriptor, source_slices[i]->relative_offset, source_slices[i]->bytes, raw);
            if (raw.size() != destinations[i]->bytes) throw std::runtime_error("fused slice read length changed");
            copy_bytes(result.storage, *destinations[i], raw.data(), raw.size());
        }
        output = std::move(result);
        return true;
    } catch (const std::exception& e) {
        err = std::string("exl3 raw expert: ") + e.what();
        return false;
    }
}


bool describe_raw_expert_slices(const Archive& archive, const RawExpertTensorNames& names, int64_t expert,
                                RawExpertSlices& output, std::string& err) {
    return describe_raw_expert_slices_impl([&](const std::string& name) { return archive.find_tensor(name); }, names,
                                           expert, output, err);
}

bool describe_raw_expert_slices(const SafetensorsHeader& header, const RawExpertTensorNames& names, int64_t expert,
                                RawExpertSlices& output, std::string& err) {
    return describe_raw_expert_slices_impl([&](const std::string& name) { return header.find(name); }, names, expert,
                                           output, err);
}

bool materialize_raw_expert_slices(const Archive& archive, const RawExpertSlices& slices, ExpertBlob& output,
                                   std::string& err, size_t alignment) {
    return materialize_raw_expert_slices_impl(
        [&](const std::string& name) { return archive.find_tensor(name); },
        [&](const TensorDescriptor& descriptor, uint64_t offset, uint64_t bytes, std::vector<uint8_t>& result) {
            archive.read_tensor_range(descriptor, offset, bytes, result);
        },
        slices, output, err, alignment);
}

bool materialize_raw_expert_slices(const SafetensorsHeader& header, const RawExpertSlices& slices, ExpertBlob& output,
                                   std::string& err, size_t alignment) {
    return materialize_raw_expert_slices_impl(
        [&](const std::string& name) { return header.find(name); },
        [&](const TensorDescriptor& descriptor, uint64_t offset, uint64_t bytes, std::vector<uint8_t>& result) {
            read_tensor_range(header, descriptor, offset, bytes, result);
        },
        slices, output, err, alignment);
}

}  // namespace strata::artifact::exl3
