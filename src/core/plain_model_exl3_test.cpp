// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Focused EXL3 PlainModel mode contract test.  It intentionally stops before
// opening a model or GPU: prefill is rejected explicitly.
#include "strata/core/plain_model.hpp"

#include <cstdio>
#include <string>

int main() {
    strata::core::PlainOptions opt;
    opt.exl3_dir = "/definitely/missing/exl3";
    opt.metadata_gguf = "/definitely/missing/metadata.gguf";
    std::string err;
    strata::core::PlainModel model;

    opt.prompt_chunk = 1;
    opt.spec = 0;
    if (model.load(opt, err) || err.find("does not support prefill") == std::string::npos) {
        std::fprintf(stderr, "EXL3 prefill was not rejected explicitly: %s\n", err.c_str());
        return 1;
    }

    std::puts("plain_model_exl3_test: all passed");
    return 0;
}
