// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// Reorder packed EXL3 tiles without decoding or requantizing weights.
#pragma once

#include "strata/artifact/exl3.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::artifact::exl3 {

/// Destination H128 block -> source H128 block. An empty map is identity.
/// Input blocks must be a full permutation; output blocks may select a unique
/// subset, allowing a combined projection to be split at H128 boundaries.
/// H128 transforms commute with these whole-block moves. Scales and bias move
/// with their blocks, and trellis words remain bit-for-bit unchanged.
bool remap_matrix_blocks(const MaterializedMatrix& source, const std::vector<int64_t>& input_blocks,
                         const std::vector<int64_t>& output_blocks, MaterializedMatrix& destination,
                         std::string& error);

}  // namespace strata::artifact::exl3
