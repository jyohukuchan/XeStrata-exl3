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

}  // namespace

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
                        size_t alignment) {
    strata::kernels::exl3::ExpertBlobLayout metadata;
    if (!describe_expert(archive, names, metadata, err, alignment)) return false;
    try {
        ResolvedProjection resolved[3];
        const std::string role_names[3] = {names.gate, names.up, names.down};
        const char* roles[3] = {"gate", "up", "down"};
        for (int i = 0; i < 3; ++i)
            if (!resolve_projection(archive, role_names[i], roles[i], resolved[i], err)) return false;

        ExpertBlob result;
        result.layout = metadata;
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
                copy_bytes(result.storage, layouts[i]->trellis, matrix.trellis.data(), layouts[i]->trellis.bytes);
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

}  // namespace strata::artifact::exl3
