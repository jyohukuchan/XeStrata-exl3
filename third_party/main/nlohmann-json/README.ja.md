<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# EXL3 メタデータ用の JSON パーサー

`nlohmann/json.hpp` は JSON for Modern C++ 3.12.0 の変更していない単一ヘッダーで、
llama.cpp の固定 commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` の
`vendor/nlohmann/json.hpp` から取得した。MIT ライセンスと元の著作権表記を保持する。

EXL3 の config、safetensors index、tensor header を読み取る。GPU runtime、Python、
llama.cpp のダウンロードがなくても artifact reader をビルドできるよう、独立して同梱する。

ヘッダーの SHA256 は更新時に元ファイルと照合する。元のライセンス全文は `LICENSE` にある。
