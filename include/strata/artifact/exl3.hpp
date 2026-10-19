// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU-side EXL3 safetensors archive descriptors and bounded materialization.
// The trellis storage rules originate in exllamav3's MIT implementation.
#pragma once

#include "strata/kernels/exl3_format.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <map>
#include <string>
#include <vector>

namespace strata::artifact::exl3 {

// gitleaks:allow -- EXL3 marker constants are public format values, never secrets.

/// A validated tensor entry from a safetensors header.  `data_offset` is
/// relative to the safetensors data section; `data_offset_absolute` is the
/// byte offset from the beginning of `file`.
struct TensorDescriptor {
    std::string name;
    std::string dtype;
    std::vector<uint64_t> shape;
    std::string file;
    uint64_t data_offset = 0;
    uint64_t data_offset_absolute = 0;
    uint64_t bytes = 0;
};

/// A single validated safetensors header.  This public inspector is useful for
/// specialized sidecars that must be opened without scanning an indexed model
/// archive.  JSON types stay private to the implementation.
struct SafetensorsHeader {
    std::filesystem::path path;
    std::string file;
    uint64_t data_start = 0;
    uint64_t file_bytes = 0;
    std::map<std::string, std::string> metadata;
    std::vector<TensorDescriptor> tensors;

    const TensorDescriptor* find(const std::string& name) const;
};

/// Validate one safetensors file's JSON, dtype/shape byte spans, duplicate
/// keys and non-overlap.  This performs header-only inspection.
SafetensorsHeader inspect_safetensors(const std::filesystem::path& path);

/// Blocking read of one descriptor from a previously inspected header.
void read_tensor(const SafetensorsHeader& header, const TensorDescriptor& tensor, std::vector<uint8_t>& bytes);

enum class SidecarRole {
    specialized_ngram_codec,
    known_override_patch,
    unindexed_sidecar,
};

struct SidecarDescriptor {
    std::string file;
    std::string format;
    SidecarRole role = SidecarRole::unindexed_sidecar;
    std::vector<TensorDescriptor> tensors;
    /// A known patch is applied after indexed tensors when `find_tensor` and
    /// linear descriptors are built.  It is kept visible in this record.
    bool overrides_base = false;
};

/// One EXL3 linear and its validated companion tensors.  `suh`/`svh` are the
/// unpacked F16 scale vectors; legacy `su`/`sv` are accepted and converted only
/// when this linear is materialized.  The marker values themselves are read at
/// open time because they are four-byte metadata, not model weights.
struct LinearDescriptor {
    std::string name;
    TensorDescriptor trellis;
    std::optional<TensorDescriptor> suh;  // codespell:ignore suh
    std::optional<TensorDescriptor> svh;
    std::optional<TensorDescriptor> su;
    std::optional<TensorDescriptor> sv;
    std::optional<TensorDescriptor> bias;
    int64_t k = 0;
    int64_t n = 0;
    int bits = 0;
    int codebook = 0;
    uint32_t marker = 0;
};

/// Owned host storage for one requested linear.  No archive-wide weight data
/// is retained by the loader.  `view()` is valid until this object is moved or
/// destroyed and returns the shared CPU/GPU-neutral MatrixView descriptor.
struct MaterializedMatrix {
    std::vector<uint16_t> trellis;
    std::vector<uint16_t> suh;  // codespell:ignore suh
    std::vector<uint16_t> svh;
    std::vector<uint16_t> bias;
    int64_t k = 0;
    int64_t n = 0;
    int bits = 0;
    int codebook = 0;

    strata::kernels::exl3::MatrixView view() const;
};

/// A CPU-only EXL3 archive inspector.  Construction reads config/index and
/// safetensors headers, plus four-byte codebook markers.  Trellis, scale and
/// bias payloads are read only by `read_tensor` or `materialize`.
class Archive {
public:
    // Opaque implementation state; JSON types stay private to the .cpp file.
    struct Impl;

    explicit Archive(const std::filesystem::path& directory);
    ~Archive();
    Archive(Archive&&) noexcept;
    Archive& operator=(Archive&&) noexcept;
    Archive(const Archive&) = delete;
    Archive& operator=(const Archive&) = delete;

    const std::filesystem::path& directory() const;
    const std::string& config_json() const;
    const std::string& quantization_config_json() const;

    /// Indexed/main tensors before applying a known patch sidecar.
    const std::vector<TensorDescriptor>& indexed_tensors() const;
    /// Effective tensor descriptors: a known patch's duplicate name replaces
    /// the indexed descriptor and is retained in its sidecar record.
    const std::vector<TensorDescriptor>& tensors() const;
    const std::vector<LinearDescriptor>& linears() const;
    const std::vector<SidecarDescriptor>& sidecars() const;
    const TensorDescriptor* find_tensor(const std::string& name) const;
    const LinearDescriptor* find_linear(const std::string& name) const;

    /// `metadata.total_size` is advisory.  A stale value is exposed to callers
    /// for reporting but never makes a valid header fail to open.
    bool has_declared_total_size() const;
    uint64_t declared_total_size() const;
    uint64_t indexed_payload_bytes() const;

    /// Read exactly one validated tensor payload into caller-owned storage.
    /// This is deliberately per-tensor and never loads a model or shard set.
    void read_tensor(const TensorDescriptor& tensor, std::vector<uint8_t>& bytes) const;

    /// Read and materialize one linear, converting legacy I16 sign bitfields
    /// and F32 bias values to binary16 bits.  Only the selected linear's
    /// components are read.
    void materialize(const LinearDescriptor& linear, MaterializedMatrix& output) const;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::artifact::exl3
