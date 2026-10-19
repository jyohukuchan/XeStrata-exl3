<!-- SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# EXL3 compressed tileの先読み

[English](EXL3_WORD_TILE.md) | 日本語

EXL3のGPU decodeには、compressed tileのwordをsubgroupで共有する任意の経路がある。
有効にしても、入力のF32精度、decoded weightのF16 RNE境界、FP32加算順と保存形式を保つ。
decoded weight全体の展開や追加のglobal scratchは行わない。

## 有効化

`STRATA_EXL3_WORD_TILE_PRELOAD=1` をprocess開始前に設定する。既定は`0`で、未設定の場合も従来の経路を使う。
有効にする値は文字列`1`だけで、それ以外の値は無効として扱う。

```sh
export STRATA_EXL3_WORD_TILE_PRELOAD=1
```

Windows PowerShellでは次のように設定する。

```powershell
$env:STRATA_EXL3_WORD_TILE_PRELOAD = "1"
```

無効へ戻すには値を`0`にしてprocessを再起動する。
既にcaptureしたgraphは選択済みのkernelを保持するため、flagを変更するときはsessionを作り直す。

## 対応する経路

このflagはcodebook 2のpacked Dense projectionに適用する。
既存packed経路が要求するsubgroup・workgroup・matrix geometryに対応する場合だけ先読みkernelを選ぶ。
条件に合わないdevice/shape、他のcodebook、raw projectionは従来の経路を使う。

bit幅1〜8の循環trellisと2byte alignmentを保つ。
Hpre、Hpost、SVH、biasの処理とcaller-owned scratchの量は変わらない。
一般的なEXL3の設定と対応形式は[EXL3](EXL3.ja.md)を参照。
