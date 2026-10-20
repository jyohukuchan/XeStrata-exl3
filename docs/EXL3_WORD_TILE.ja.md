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

## Routed expert

SG32対応deviceのrouted expertでは、codebook 2のpacked GU・down行列積でも圧縮wordを共有する。
この経路は既定で有効で、`STRATA_EXL3_GROUPED_WORD_TILE=0` にすると従来の行列積へ戻る。
未設定または`0`以外の値は有効として扱う。Dense用flagとは独立している。

```sh
export STRATA_EXL3_GROUPED_WORD_TILE=0
```

subgroupは16出力列・2 K laneを担当し、16×16 tileの圧縮wordをレジスタへ先読みする。
bit幅1〜8をそれぞれ特殊化し、同じsubgroup内でwordを共有して各stateを復号する。
各laneに2本の累積値を持ち、従来の4 K laneのFP32加算順とreduction treeを再現する。
入力のF32精度とdecoded weightのF16 RNE境界も保つ。
GUのK分割境界が16要素に揃わない区間、他のcodebook、raw projectionは従来の計算を使う。
format table内でbit幅やprojection形式が混在する場合も、各projectionで経路を選ぶ。

追加のglobal scratchや展開済みweightを保持しない。既存のdevice上のplanとevent依存関係を使い、
Hpre・Hpost・biasとgroup/token/destinationの検証は変わらない。
flagはgraph capture時に評価するため、変更した場合はsessionを作り直す。
