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
using Materialized = MaterializedMatrix;

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

Matrix projection_layout(const LinearDescriptor& descriptor) {
    Matrix result;
    result.k = descriptor.k;
    result.n = descriptor.n;
    result.bits = descriptor.bits;
    result.codebook = descriptor.codebook;
    return result;
}

void copy_component(std::vector<uint16_t>& destination, const Matrix& layout, ComponentLayout Matrix::*member,
                    const std::vector<uint16_t>& source) {
    const ComponentLayout& component_layout = layout.*member;
    if (source.empty()) return;
    std::memcpy(reinterpret_cast<uint8_t*>(destination.data()) + component_layout.offset, source.data(), component_layout.bytes);
}

bool valid_projection(const LinearDescriptor& descriptor, const char* role, std::string& err) {
    if (descriptor.k <= 0 || descriptor.n <= 0 || descriptor.k % 128 != 0 || descriptor.n % 128 != 0) {
        err = std::string("exl3 expert: ") + role + " is not 128-aligned";
        return false;
    }
    if (descriptor.bits < 1 || descriptor.bits > 8 || descriptor.codebook < 0 || descriptor.codebook > 2) {
        err = std::string("exl3 expert: ") + role + " has unsupported bits/codebook";
        return false;
    }
    return true;
}

void metadata_component(Matrix& layout, ComponentLayout Matrix::*member, uint64_t bytes, uint64_t& cursor,
                        size_t alignment) {
    cursor = align_up(cursor, alignment);
    ComponentLayout& destination = layout.*member;
    destination.offset = cursor;
    destination.bytes = bytes;
    if (!checked_add(cursor, bytes, cursor)) throw std::runtime_error("expert blob size overflows uint64");
}

void metadata_projection(Matrix& layout, const LinearDescriptor& descriptor, uint64_t& cursor, size_t alignment) {
    layout = projection_layout(descriptor);
    const uint64_t trellis_bytes = descriptor.trellis.bytes;
    const uint64_t scale_in_bytes = static_cast<uint64_t>(descriptor.k) * sizeof(uint16_t);
    const uint64_t scale_out_bytes = static_cast<uint64_t>(descriptor.n) * sizeof(uint16_t);
    const uint64_t bias_bytes = descriptor.bias.has_value() ? static_cast<uint64_t>(descriptor.n) * sizeof(uint16_t) : 0;
    metadata_component(layout, &Matrix::trellis, trellis_bytes, cursor, alignment);
    metadata_component(layout, &Matrix::suh, scale_in_bytes, cursor, alignment);  // codespell:ignore suh
    metadata_component(layout, &Matrix::svh, scale_out_bytes, cursor, alignment);
    metadata_component(layout, &Matrix::bias, bias_bytes, cursor, alignment);
}

}  // namespace

bool describe_expert(const Archive& archive, const ExpertTensorNames& names,
                     strata::kernels::exl3::ExpertBlobLayout& layout, std::string& err, size_t alignment) {
    err.clear();
    if (alignment != 2) {
        err = "exl3 expert: this typed blob storage supports alignment=2 only";
        return false;
    }
    if (names.gate.empty() || names.up.empty() || names.down.empty()) {
        err = "exl3 expert: gate/up/down names are required";
        return false;
    }
    try {
        const LinearDescriptor* gate = archive.find_linear(names.gate);
        const LinearDescriptor* up = archive.find_linear(names.up);
        const LinearDescriptor* down = archive.find_linear(names.down);
        if (gate == nullptr || up == nullptr || down == nullptr) {
            err = "exl3 expert: raw BF16/F16 expert codec is not supported; an explicit EXL3 trellis descriptor is missing";
            return false;
        }
        if (!valid_projection(*gate, "gate", err) || !valid_projection(*up, "up", err) ||
            !valid_projection(*down, "down", err))
            return false;
        if (gate->k != up->k || gate->n != up->n || down->k != gate->n || down->n != gate->k) {
            err = "exl3 expert: gate/up/down shapes do not form hidden-ff-hidden";
            return false;
        }
        strata::kernels::exl3::ExpertBlobLayout result;
        result.alignment = static_cast<uint32_t>(alignment);
        uint64_t cursor = 0;
        metadata_projection(result.gate, *gate, cursor, alignment);
        metadata_projection(result.up, *up, cursor, alignment);
        metadata_projection(result.down, *down, cursor, alignment);
        result.total_bytes = cursor;
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
        const LinearDescriptor* gate_descriptor = archive.find_linear(names.gate);
        const LinearDescriptor* up_descriptor = archive.find_linear(names.up);
        const LinearDescriptor* down_descriptor = archive.find_linear(names.down);
        Materialized gate, up, down;
        archive.materialize(*gate_descriptor, gate);
        archive.materialize(*up_descriptor, up);
        archive.materialize(*down_descriptor, down);
        ExpertBlob result;
        result.layout = metadata;
        if (metadata.total_bytes % sizeof(uint16_t) != 0 ||
            metadata.total_bytes / sizeof(uint16_t) > std::numeric_limits<size_t>::max())
            throw std::runtime_error("expert blob size cannot be represented by typed storage");
        result.words.assign(static_cast<size_t>(metadata.total_bytes / sizeof(uint16_t)), 0);
        if (gate.trellis.size() * sizeof(uint16_t) != metadata.gate.trellis.bytes ||
            gate.suh.size() * sizeof(uint16_t) != metadata.gate.suh.bytes ||
            gate.svh.size() * sizeof(uint16_t) != metadata.gate.svh.bytes ||
            gate.bias.size() * sizeof(uint16_t) != metadata.gate.bias.bytes ||
            up.trellis.size() * sizeof(uint16_t) != metadata.up.trellis.bytes ||
            up.suh.size() * sizeof(uint16_t) != metadata.up.suh.bytes ||
            up.svh.size() * sizeof(uint16_t) != metadata.up.svh.bytes ||
            up.bias.size() * sizeof(uint16_t) != metadata.up.bias.bytes ||
            down.trellis.size() * sizeof(uint16_t) != metadata.down.trellis.bytes ||
            down.suh.size() * sizeof(uint16_t) != metadata.down.suh.bytes ||
            down.svh.size() * sizeof(uint16_t) != metadata.down.svh.bytes ||
            down.bias.size() * sizeof(uint16_t) != metadata.down.bias.bytes)
            throw std::runtime_error("materialized component size differs from metadata layout");
        copy_component(result.words, result.layout.gate, &Matrix::trellis, gate.trellis);
        copy_component(result.words, result.layout.gate, &Matrix::suh, gate.suh);  // codespell:ignore suh
        copy_component(result.words, result.layout.gate, &Matrix::svh, gate.svh);
        copy_component(result.words, result.layout.gate, &Matrix::bias, gate.bias);
        copy_component(result.words, result.layout.up, &Matrix::trellis, up.trellis);
        copy_component(result.words, result.layout.up, &Matrix::suh, up.suh);  // codespell:ignore suh
        copy_component(result.words, result.layout.up, &Matrix::svh, up.svh);
        copy_component(result.words, result.layout.up, &Matrix::bias, up.bias);
        copy_component(result.words, result.layout.down, &Matrix::trellis, down.trellis);
        copy_component(result.words, result.layout.down, &Matrix::suh, down.suh);  // codespell:ignore suh
        copy_component(result.words, result.layout.down, &Matrix::svh, down.svh);
        copy_component(result.words, result.layout.down, &Matrix::bias, down.bias);
        output = std::move(result);
        return true;
    } catch (const std::exception& e) {
        err = std::string("exl3 expert: ") + e.what();
        return false;
    }
}

}  // namespace strata::artifact::exl3
