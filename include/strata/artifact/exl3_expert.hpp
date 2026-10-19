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

/// Typed host storage for one packed expert.  `words` guarantees the minimum
/// two-byte alignment required by the component tensors; this bounded builder
/// intentionally accepts alignment=2 only until a separately owned aligned
/// allocator is needed by a device cache.
struct ExpertBlob {
    strata::kernels::exl3::ExpertBlobLayout layout;
    std::vector<uint16_t> words;

    const uint8_t* data() const { return words.empty() ? nullptr : reinterpret_cast<const uint8_t*>(words.data()); }
    size_t bytes() const { return words.size() * sizeof(uint16_t); }
    strata::kernels::exl3::ExpertView view() const {
        return strata::kernels::exl3::expert_view(layout, data());
    }
};

/// Query the exact packed layout from descriptor metadata alone.  No tensor
/// payload is read; offsets and component sizes match `materialize_expert`.
bool describe_expert(const Archive& archive, const ExpertTensorNames& names,
                     strata::kernels::exl3::ExpertBlobLayout& layout, std::string& err, size_t alignment = 2);

/// Materialize only the three packed EXL3 projections named by `names`.
/// Components retain their exact materialized binary16 bits and are appended
/// at their actual sizes; no full matrix or maximum-bit padding is created.
/// Raw BF16/F16 expert tensors are rejected as a distinct unsupported codec.
bool materialize_expert(const Archive& archive, const ExpertTensorNames& names, ExpertBlob& output, std::string& err,
                        size_t alignment = 2);

}  // namespace strata::artifact::exl3
