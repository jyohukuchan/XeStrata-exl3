// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Logical EXL3 expert format descriptors and their process-local registry.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "strata/kernels/exl3_expert_blob.hpp"

namespace strata::core {

/// The registry consumes the actual packed blob factory layout, including every role's trellis/suh/svh/bias span.
using Exl3ExpertFormatDesc = strata::kernels::exl3::ExpertBlobLayout;

using Exl3ExpertFormatID = uint64_t;

bool exl3_expert_format_equal(const Exl3ExpertFormatDesc& a, const Exl3ExpertFormatDesc& b);
/// Returns zero for an invalid descriptor; nonzero IDs are deterministic for the logical fields.
Exl3ExpertFormatID exl3_expert_format_id(const Exl3ExpertFormatDesc& desc);
bool validate_exl3_expert_format(const Exl3ExpertFormatDesc& desc, std::string& error);

/// Process-local collision-checking registry. IDs are not a persisted file format.
class Exl3ExpertFormatRegistry {
public:
    Exl3ExpertFormatID register_format(const Exl3ExpertFormatDesc& desc, std::string& error);
    const Exl3ExpertFormatDesc* find(Exl3ExpertFormatID id) const;
    size_t size() const { return entries_.size(); }
    void clear() { entries_.clear(); }

private:
    struct Entry {
        Exl3ExpertFormatID id = 0;
        Exl3ExpertFormatDesc desc{};
    };
    std::vector<Entry> entries_;
};

}  // namespace strata::core
