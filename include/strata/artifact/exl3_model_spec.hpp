// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Typed model geometry read directly from an EXL3 archive config.
#pragma once

#include "strata/artifact/exl3.hpp"
#include "strata/artifact/model_spec.hpp"

#include <string>

namespace strata::artifact::exl3 {

/// Read the Qwen3.5/Qwen3.8 (and LFM2 MoE, HF `lfm2_moe`) geometry needed by the Flash engine without a
/// GGUF metadata pack. When include_mtp is false, the returned ModelSpec is
/// explicitly trunk-only. Qwen3.8 MTP is rejected when requested because the
/// official raw MTP source and its override precedence are a separate loader
/// contract; it must never be inferred from a few `mtp.*` archive tensors.
bool read_exl3_model_spec(const Archive& archive, strata::ModelSpec& spec, std::string& err,
                          bool include_mtp = false);

}  // namespace strata::artifact::exl3
