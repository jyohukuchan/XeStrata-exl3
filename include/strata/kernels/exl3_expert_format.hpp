// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 jyohukuchan
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// EXL3 packed tiles follow exllamav3's MIT-licensed trellis storage rules.
#pragma once

#include "strata/kernels/exl3_format.hpp"

namespace strata::kernels::exl3 {

/// Neutral descriptor for one gate/up/down expert, usable by CPU and SYCL callers.
struct ExpertView {
    MatrixView gate;
    MatrixView up;
    MatrixView down;
};

}  // namespace strata::kernels::exl3
