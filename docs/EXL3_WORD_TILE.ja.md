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

## 出力幅の大きい射影

既存packed経路で処理しない出力幅の大きいcodebook 2の行列には、SG32内で圧縮wordと隣接する積を共有する経路を使う。
下位16 laneはK=0から順番に積を加算し、従来のscalar経路のFP32加算順を保つ。
bit幅1〜8、bias、Hpre/Hpostと複数行・leading dimensionに対応し、追加global scratchを要求しない。
SG32非対応deviceと他のcodebookは従来の経路を使う。

既定で有効。`STRATA_EXL3_WIDE_WORD_TILE=0` で従来のscalar経路へ戻る。Denseの先読みflagとは独立している。

## Denseのwave配置

`STRATA_EXL3_WORD_TILE_PRELOAD=1` のpacked codebook 2経路では、独立した出力tile数とdeviceのcompute unit数・workgroup容量から、
実際に実行するwave数を選ぶ。従来の論理K分割ごとの積和を保持し、同じ順序でpartialを加算する。
論理分割を複数担当するwaveも、分割ごとの累積値を独立して計算するため、FP32加算順は変わらない。
さらに、1つのwaveが読むK tileの数を16個（K要素256個）に制限する。waveがK全体を覆うとtileを順に読むためloadを重ねられず、
deviceを一度だけ埋めるwave数ではmemory latencyが露出する。wave数を変えても各論理分割の累積値とpartialの加算順は変わらない。

既定で有効。`STRATA_EXL3_DENSE_SCHEDULE=0` で従来のwave配置へ戻る。出力幅の大きいscalar経路には適用しない。

## codebook 2 の decode

codebook 2 (mul1) の decode は、byte sum B に対して (887 * B - 452608) * 2^-17 を返す。
値は 2^-17 の倍数で絶対値が 2^19 未満なので F32 で厳密に表せる。portable な形は F32 でその値を作り、
F16 へ一度だけ丸める。half の形は参照実装と同じ F16 の fused multiply add
(hfma16(F16(1024 + B), 0x1EEE, 0xC931)) で行い、同じ厳密値を一度だけ丸める。
device が FP16 の演算を報告するときは half の形を使い、報告しない device では portable な形を使う。
どちらの形も bit 単位で同じ値を返す。

既定で有効。`STRATA_EXL3_CB2_HALF_FMA=0` で portable な形を強制し、同じ binary で両方を比較できる。
Dense、routed expert、出力幅の大きい射影、n-gram の decode は同じ判定を使う。
F16 の RNE 境界、FP32 の加算順、Hadamard 変換、bias、重みの保存形式と bpw は変わらない。

## Raw射影

F16/BF16/F32のraw射影ではdtypeごとのkernelを使い、K方向のloadを展開する。
独立した出力が少なくK方向の並列性を増やせる場合は、device能力とlocal memory容量からworkgroupを選び、
一時的なF32の積をlocal memoryへ置く。従来の32 laneの加算順を保ち、global scratchは増やさない。
FP16に対応するdeviceでは、F16の正規数の変換にnative conversionを使う。
ゼロ・非正規数・非有限値は従来のsoftware変換を使い、FP16 denormal modeに依存しない。
入力はF32のままで、各laneの積和順とsubgroup reductionを保つ。重みの追加展開は行わない。

既定で有効。`STRATA_EXL3_RAW_SPECIALIZED=0` で従来のdtype共通kernelへ戻る。
各flagの変更後は、capture済みのgraphを使うsessionを作り直す。
