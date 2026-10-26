<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# EXL3 の精度（KL）の計測記録

XeStrata の EXL3 と I-Quant の精度を、BF16 の元重みに対する `KL(P_BF16 || P_candidate)` で測った記録の要約です。
モデルはすべて Qwen3.5 35B の instruct（`qwen35-instruct`）です。Qwen3.8 Flash の KL はまだ取っていません。

## 最終の結果（`kl-final-7a749ef5/`）

固定の 6 holdout、2729 位置。教師は公式の BF16 重み、校正 token は 32×1024 で、両者で同じです。

| 4.5 bpw | EXL3 | Xe I-Quant |
| --- | ---: | ---: |
| 平均 KL | 0.00897974 | 0.02184257 |
| median | 0.00492543 | 0.01300026 |
| p90 | 0.02167341 | 0.05320558 |
| max | 0.19886886 | 0.33511669 |

保存量はほぼ同じで（I-Quant が 0.0989% 小さい）、平均 KL は EXL3 が 58.9% 低くなりました。
経緯と条件は `EXL3_GATE_7a749ef5.ja.md` の「精度ゲート」の節にあります。

## ファイル

- `kl-final-7a749ef5/pair.json`: 最終の対比較（両側の平均・median・p90・max、保存量、バイナリの SHA256）
- `kl-final-7a749ef5/{exl45-r9700,iq45-r9700}/kl.json`: 位置ごとの KL。`plan.json` と `executed-protocol.json` は計測の条件
- `qwen35-instruct/*-kl.json`: 最適化ごと、bpw ごと、VRAM 制限下などの KL
- `qwen35-instruct/*-paired.json`: 同じセッションで取った対比較
- `qwen35-instruct/*quality-record.json`: 各量子化の変換と評価の記録

## 入れていないもの

logits の capture（1 側 2.6 GB）と教師の cache は大きいため入れていません。
パス、GPU の識別子、コンテナ名は `<data>`、`<checkout>`、`<gpu-uuid>`、`<container>` などに置き換えてあります。
