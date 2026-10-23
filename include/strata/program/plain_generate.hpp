// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/program/plain_generate.hpp - `strata generate` for the models of plain_model.hpp.
#pragma once

namespace strata::program {

/// Runs an EXL3 model supported by PlainModel, using its metadata bootstrap; returns -1 for other model families.
int plain_main(int argc, char** argv);

}  // namespace strata::program
