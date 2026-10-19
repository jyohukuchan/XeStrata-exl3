// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// CPU-side EXL3 expert blob materialization; packed rules derive from exllamav3's MIT implementation.
#pragma once

#include "strata/artifact/exl3.hpp"
#include "strata/kernels/exl3_expert_blob.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::artifact::exl3 {

struct ExpertTensorNames {
    std::string gate;
    std::string up;
    std::string down;
};

/// Names and optional geometry for a fused raw MTP expert source. The source
/// stores gate/up as [experts, 2*ff, hidden] and down as [experts, hidden, ff].
/// Zero geometry fields mean "derive from the validated headers".
struct RawExpertTensorNames {
    std::string gate_up;
    std::string down;
    int64_t experts = 0;
    int64_t hidden = 0;
    int64_t feed_forward = 0;
};

/// One 2-D projection slice inside a fused raw tensor. Offsets are relative
/// to that tensor's payload and are consumed by Archive::read_tensor_range.
struct RawExpertSlice {
    std::string tensor;
    uint64_t relative_offset = 0;
    uint64_t bytes = 0;
    int64_t k = 0;
    int64_t n = 0;
    strata::kernels::raw::DType dtype = strata::kernels::raw::DType::F32;
};

struct RawExpertSlices {
    RawExpertSlice gate;
    RawExpertSlice up;
    RawExpertSlice down;
};

/// Typed host storage for one packed or raw expert.  The uint64_t backing
/// storage provides a stable alignment for raw F32 projections while packed
/// component bytes remain byte exact.
struct ExpertBlob {
    strata::kernels::exl3::ExpertBlobLayout layout;
    std::vector<uint64_t> storage;

    const uint8_t* data() const { return storage.empty() ? nullptr : reinterpret_cast<const uint8_t*>(storage.data()); }
    size_t bytes() const { return static_cast<size_t>(layout.total_bytes); }
    strata::kernels::exl3::ExpertView view() const {
        return strata::kernels::exl3::expert_view(layout, data());
    }
};

/// Query the exact mixed packed/raw layout from descriptor metadata alone.  No tensor
/// payload is read; offsets and component sizes match `materialize_expert`.
bool describe_expert(const Archive& archive, const ExpertTensorNames& names,
                     strata::kernels::exl3::ExpertBlobLayout& layout, std::string& err, size_t alignment = 8);

/// Materialize only the three projections named by `names`. Packed components
/// retain their exact binary16 bits; raw F16/BF16/F32 components retain their
/// source bytes and dtype metadata. No full decoded matrix is created.
bool materialize_expert(const Archive& archive, const ExpertTensorNames& names, ExpertBlob& output, std::string& err,
                        size_t alignment = 8);

/// Describe one expert's three 2-D slices without reading tensor payloads.
bool describe_raw_expert_slices(const Archive& archive, const RawExpertTensorNames& names, int64_t expert,
                                RawExpertSlices& output, std::string& err);
bool describe_raw_expert_slices(const SafetensorsHeader& header, const RawExpertTensorNames& names, int64_t expert,
                                RawExpertSlices& output, std::string& err);

/// Read only the described slices into the existing raw ExpertBlob format.
/// The destination remains unchanged on failure and source dtypes are retained.
bool materialize_raw_expert_slices(const Archive& archive, const RawExpertSlices& slices, ExpertBlob& output,
                                   std::string& err, size_t alignment = 8);
bool materialize_raw_expert_slices(const SafetensorsHeader& header, const RawExpertSlices& slices, ExpertBlob& output,
                                   std::string& err, size_t alignment = 8);

}  // namespace strata::artifact::exl3
