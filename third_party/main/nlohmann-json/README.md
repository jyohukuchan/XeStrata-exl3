<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# JSON parser for EXL3 metadata

`nlohmann/json.hpp` is the unmodified JSON for Modern C++ 3.12.0 single header,
taken from `vendor/nlohmann/json.hpp` at llama.cpp's pinned commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. Its MIT license and original copyright
notices are retained.

It reads EXL3 config, safetensors indexes, and tensor headers. It is bundled
independently so the artifact reader builds without a GPU runtime, Python,
or a llama.cpp download.

Compare the header's SHA256 with its source when updating it. `LICENSE` contains
the original license text.
