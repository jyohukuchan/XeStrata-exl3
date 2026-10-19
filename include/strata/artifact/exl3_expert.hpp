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

}  // namespace strata::artifact::exl3
