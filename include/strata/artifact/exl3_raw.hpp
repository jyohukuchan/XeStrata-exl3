// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "strata/artifact/exl3.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::artifact::exl3 {

enum class ValueTransform { identity, add_one, negative_exp };

struct RawTensor {
    std::string dtype;
    std::vector<uint64_t> shape;
    std::vector<uint8_t> data;
};

/// Maps each destination axis index to its source index. Empty axis maps are
/// identity; nonempty maps may select a unique subset. Raw storage is row-major.
/// Identity value transforms preserve every F16/BF16/F32 bit. Arithmetic value
/// transforms produce F32, retaining the source dtype until that operation.
/// Failure leaves destination unchanged; source and destination may alias.
bool remap_raw(const RawTensor& source, const std::vector<std::vector<int64_t>>& axes,
               ValueTransform transform, RawTensor& destination, std::string& err);

bool materialize_raw(const Archive& archive, const TensorDescriptor& descriptor,
                     const std::vector<std::vector<int64_t>>& axes, ValueTransform transform,
                     RawTensor& destination, std::string& err);

/// Explicit widening for kernels whose weight contract is F32. This does not
/// change raw projection or embedding storage implicitly.
bool raw_to_f32(const RawTensor& source, std::vector<float>& destination, std::string& err);

}  // namespace strata::artifact::exl3
