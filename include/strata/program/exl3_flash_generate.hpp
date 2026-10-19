// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Minimal Qwen3.8 EXL3 autoregressive command-line entry point.
#pragma once

namespace strata::program {

/// Dispatch the direct Qwen3.8 EXL3 AR CLI.  Returns -1 when argv does not
/// name a Qwen3.8 EXL3 archive, so the existing generate/plain dispatch can
/// continue.  Qwen3.5 EXL3 deliberately returns -1 for its own path.
int exl3_flash_main(int argc, char** argv);

}  // namespace strata::program
