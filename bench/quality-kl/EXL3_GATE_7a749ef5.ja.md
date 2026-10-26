<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# EXL3 検証と速度ゲート（7a749ef5）

EXL3 の 2〜5 bpw と混合精度の読み込み・実行・KL 比較は 5508006b で完了済みで、
2255b625 の最適化 7 系統に 이어 **CPU exact decode の高速化（A）** と
**CPU expert の K 方向分割（B）** を統合した後の判定である。

**判定**

- **同実効 bpw で EXL3 の KL が IQuant より低い**という条件は**成立**。
  最終バイナリで取り直した 4.5 bpw の平均 KL は EXL3 0.00897974 対 IQuant 0.02184257 で
  **58.9% 低い**（保存量差は EXL3 の方が -0.099% 大きい）。
- **全 expert を GPU に常駐できる条件では decode 速度が IQuant の 98.9% に達した**
  （35B モデル、R9700、69.55 対 70.29 tok/s）。暫定合格の閾値 95% を満たす。
- **CPU/GPU 併用の条件（VRAM 上限 12 GiB）では IQuant の 73〜78%** に留まる。
  前回の判定（72〜76%）とほぼ同じ水準で、閾値には届かない。
  ただし基準バイナリに対する改善は 1.41〜1.42 倍に達している。

EXL3 形式による速度同等が不可能であると判定するものではない。

ソースは `7a749ef5`（upstream/main から 70 コミット、作業ブランチ `feature/integrated`、
ドキュメントは `3de7050b`）。測定は raw ログ・固定バイナリ・protocol・source identity を併記した。
開発機の設定と実行環境は repository の LOCAL.md を参照。

## 速度ゲート（35B モデル、同一セッション、ABBA 順）

比較は**同一セッションで ABBA 順**（基準→候補→候補→基準）に同じバイナリ対・同じモデル・
同じ token 列・同じ設定で走らせた。基準は 81334cf3、候補は 7a749ef5。

| 条件 | 基準 tok/s | 候補 tok/s | 改善 | IQuant 参照 | 候補/IQuant |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全 expert GPU 常駐（R9700、pool 8） | 59.25 → 59.46 | **68.88 → 69.55** | 1.170× | 70.29 | **98.9%** |
| CPU/GPU 併用（R9700、pool 48） | 35.12 → 35.82 | **49.63 → 50.57** | 1.412× | 64.59 | 78.3% |
| CPU/GPU 併用（V620 43:00.0、pool 32） | 32.51 → 33.35 | **45.98 → 47.33** | 1.419× | 64.79 | 73.1% |
| CPU/GPU 併用（V620 03:00.0、pool 32） | 33.74 → 34.03 | **47.52 → 48.23** | 1.418× | 63.91 | 75.5% |

比較条件は context 512、warmup 16、64 位置 × 2 repeat、max_t 1、route skip 0、HC_Q8 0、
CPU/GPU 併用では INTDOT と ROWPARALLEL を有効、VRAM 上限 12GiB、最大 allocation 4GiB、XMX 無効。
profiler・変換・採点・コンパイルは速度測定と重ねていない。

**参照の IQuant は各条件の pool worker で測っている**（全常駐は 8、併用は 48 / 32）。
IQuant の最良 worker（16）で測ると分母が上がり、比はもう少し下がる。

### VRAM 上限を 8 GiB から 12 GiB に上げた理由

ZFS の ARC を 16 GiB に絞った結果、free RAM が増えて
`src/core/plain_model.cpp` の expert を RAM に写す分岐（`resident_pin_fits`）に入る。
その状態で VRAM 上限 8 GiB だと expert tier が上がらず、native 参照が
"no VRAM is left for the expert tier" で**起動できなくなった**。
比較は両腕で同じ条件にする必要があるので、上限を 12 GiB に上げて両腕を測った。
8 GiB 条件の比較は ARC 変更前の `2255b625` の記録
（`records/paired-gate-final-2255b625-hybrid/`）が有効である。

### ABBA 内訳（records/paired-gate-final-7a749ef5-12g/）

| 条件 | baseline | candidate | candidate | baseline | IQuant |
| --- | ---: | ---: | ---: | ---: | ---: |
| fullgpu-r9700 | 59.247 | 68.875 | 69.547 | 59.457 | 70.293 |
| hybrid-r9700 | 35.117 | 49.633 | 50.572 | 35.824 | 64.592 |
| hybrid-v620 | 33.345 | 47.330 | 45.977 | 32.514 | 64.790 |
| hybrid-v620-03 | 34.025 | 47.516 | 48.233 | 33.743 | 63.909 |

**hit/miss 分割**: 全条件で基準と候補が一致（20480 組）。
**resident slot 数**: 全常駐 10240 で一致、併用 6033 対 6032（R9700）と 6055 対 6053（V620）。
cache の大きさはその時の VRAM の空きから決めるので 1〜2 slot 動く。これはノイズであり、
1.4 倍の差の説明にはならない。

## 精度ゲート

品質の指標は `KL(P_BF16 || P_candidate)` のみ。同一の公式 BF16 元重み・32×1024 校正 token・
固定 6 holdout の 2729 位置。**最終バイナリ（`plain_model_logits`、SHA256 `d400b894…`）で
strict pair を取り直した**（`records/quality-instruct/kl-final-7a749ef5/`）。

| EXL3 target | EXL3 mean KL | Xe IQuant mean KL | 低下率 | IQ 保存量差（対 EXL） |
| --- | ---: | ---: | ---: | ---: |
| 4.5 | 0.00897974 | 0.02184257 | **58.9%** | -0.0989% |

median は 0.00492543 対 0.01300026、p90 は 0.02167341 対 0.05320558、max は 0.19886886 対 0.33511669。

**IQuant 側の値はこの binary でも同じ**（0.02184257088865589、bit 一致）。
EXL3 側は 5508006b の測定（0.00900379）より 0.27% 下がった（median 2.2%、p90 1.8%、max は一致）。
つまり今回の A・B と 7 系統の最適化で**KL は改善**していて、悪化は無い。

**B（K 方向分割）は数値を一切変えない。** 部分和が正確な整数で、ブロックの変換は
足し合わせた後に一度だけ適用するため、足し算の順序が違っても値は同じ。
実重み checkpoint で expert 出力 10240 個のうち差のあるものは 0 個、
model 全体の logits（7946240 byte）はビット単位で一致した。

## 今回の 2 系統（2255b625 の 7 系統に追加）

- **A: CPU の exact decode の高速化**（`6014548c`、`8ec8a852`、`834bbd1c`）
  EXL3 の CPU 経路が GPU を持たない環境で exact（FP64）経路へ落ちる場合の
  per-expert コストが native の約 5 倍で、CPU 並列が足りていなかった。
  CPU 単体で **3.45〜3.88 倍**。
- **B: CPU expert の K 方向分割**（`a62e8db2`、統合 `7a749ef5`）
  128 出力ブロック 1 タスクでは、VRAM tier に載らず host へ落ちた expert が
  少数しかない層でタスク数が worker 数に届かず worker が空回りした。
  dot を K 範囲に分割し、各範囲を 1 タスクにして整数部分和を残す。
  タスク数が worker に満たない phase だけを `floor(worker数 / タスク数)` 分割する。
  CPU 単体（1 expert/layer）で 32 worker 4,656 → 3,250 µs（-30%）、
  48 worker 5,227 → 3,599 µs（-31%）。
  数値差は 0（上記）。

## build と test の検証

両モードを intel/llvm でビルドし、警告をエラーにする設定（`STRATA_WERROR=ON`で
新規の警告がないことを確認した。

| モード | build | 警告 | CTest |
| --- | --- | ---: | ---: |
| free | 成功 | 0 | 136/136 pass（2 件 skip） |
| contrib-llvm | 成功 | 0 | 136/136 pass（2 件 skip） |

skip の 2 件は `vmm_test` と `expert_multi_test`（AVX-512 の経路）。この host には
Intel SDE が無いため AVX-512 の実行は未検証。Windows については、触れた translation unit
（pool.cpp、exl3_mul1_intdot.cpp、exl3.cpp、pool と intdot の各 test、exl3_layout_parity.cpp、
exl3_flash_model.cpp、exl3_expert_runtime.cpp、generate.cpp）が MSVC ABI の COFF に
`-Wall -Wextra -Werror` でコンパイルできることを確認した
（`records/cpu-split-k-windows.json`）。full link と runtime は未検証
（`.tools/intel-llvm-win` がこの host に無い）。

この过程中で 1 件直した。`exl3_pool_test.cpp` が `setenv` を直接呼んでいたので、
Windows では他の test と同じ書き方（`_putenv_s`）に変えた
（commit `179d747b`）。

lint: license_class と reuse は clean、gitleaks も clean（`.lint/bin` を PATH に入れて実行）。
codespell、clang-tidy、cppcheck、mdl、lychee はこの host に未導入で skip。
lint script は gitleaks を PATH から呼び、道具が無いと「secret の可能性」と出すので、
`.lint/bin` を PATH に入れて実行した。

## 残ってる作業

- **Sage-A1.5 の EXL3 4.0 bpw 変換**（`<models>/safetensors/Sage-A1.5-exl3-4.0bpw`）。
  コンテナ `<container>` の watcher が `records/convert-allow` を置いて
  MemAvailable 45 GB 以上と GPU アイドルで走らせる。**`--resume` を付けること**：
  付けないと checkpoint を無視して先頭からやり直し、`qtensors/` を消す
  （exllamav3 の `convert_model.py` の resume 処理）。

- **CPU/GPU 併用条件の速度**（IQuant の 73〜78%）。残差の主因は trellis の復号代
  （16-bit window の抽出 → magic multiply → byte sum → F16 丸め）で、I-Quant の整数 dot
  （16 要素を 1 回の表参照で復元、scale は dot 後に 1 回）より per-element コストが大きい。
  codebook 2 の半精度 FMA は実装済みで、次の候補は 64-bit chunk による state 抽出の削減。
- **Qwen3.8 の教師 KL の取り直し**（device cache を埋めた状態で数値が下位ビットで変わるため）。
  35B の KL は取り直し済みで改善した。
- **Qwen3.8 の EXL3 MTP**（未接続。native は MTP 前提で動くため、比較を公平にするには必要）。
- 複数 GPU（層分割）：PLE の device buffer が stage 間で共有されず未対応。
- codebook 0/1 の実モデル全体の品質比較、serve、vision、Intel/NVIDIA 実行、
  Windows の full link/runtime。

## 記録

- 速度ゲート: `records/paired-gate-final-7a749ef5-12g/`（4 条件、ABBA、IQuant 参照つき）。
- 教師 KL: `records/quality-instruct/kl-final-7a749ef5/`（pair.json と両側の capture）。
- CPU split-K: `records/cpu-split-k/`（harness、raw 出力、表、host の load の記録）。
- 基準バイナリ: `records/baseline-81334cf3/`（plain_model_bench の SHA256
  `17d1b67211af7b260c33a7563245eecb8dec7d577708ec4fab6d6870367e1d1f`）。
  候補: `records/paired-gate-final-7a749ef5-12g/candidate-bench`
  （SHA256 `83f37dd491f70e3b115008d1c49d47dad8992833a9669b2a7fb282876b807566`）。

## 追記（2def8636 〜 cd5d1309）

判定の後、対象を広げる 3 系統を追加した。速度ゲートの数値は変えない
（いずれも測定した構成の外側か、この host では実行できない経路だからである）。

1. **高速経路を既定にする**（`2def8636`）
   整数内積、CPU の行分割、device-hit plan、word tile の先行読みを既定にした。
   速度ゲートと教師 KL はこの 4 つを入れて測ってあるので、既定が測定した構成になる。
   各々は環境変数を `=0` にすると従来の挙動に戻る。整数内積だけは値を変える
   （codebook の affine を和に掛けるのを、exact 経路と GPU は要素ごとに F16 に丸めてから和す）ため、
   CPU と GPU・exact を比べる test では整数内積を切ってあり、CMakeLists に明記した。
2. **AVX-512 版の CPU カーネル**（`0b72afe3`、統合 `becd3416`）
   整数内積と exact 経路の 512 ビット幅版。1 行 16 列を 1 本のレジスタで処理する。
   命令の対応は CPUID の特徴ビットで選び、`STRATA_FORCE_AVX2=1` で 256 ビット版に戻る。
   F16 変換は 256 ビットのまま（512 ビット版は AVX512-FP16 が要り、Zen 4 に無い）。
   AVX2 版と AVX-512 版の bit 一致はスカラーのモデルで bit 幅 1〜8 について確認した
   （`records/cpu-avx512-model/`）。**実機での AVX-512 の実行は未検証**（この host は Zen 3、
   Intel SDE が無い）。
3. **複数 GPU の EXL3**（`19334a35`、統合 `9f071f70`）
   起こっていた障害は PLE では無かった。allocation の記録がブロックの先頭アドレスしか
   名乗らず、中身のポインタ（stage のバッファは全てこれ）は context の走査に落ちて
   「runtime 0」と答え、stage 1 の dense owner が自分のバッファを拒否していた
   （"input or output belongs to another runtime"）。記録がブロックの大きさを持つようにし、
   `Runtime::owner` が領域内のポインタを解決するようにした。stage の境界を越えるのは
   従来通り残差だけで、PLE はその stage 内で完結する。
   **V620 2 枚（<gpu-pci> と <gpu-pci>）で 48 層を 24/24 に割って動作確認**
   （device cache なしでは 1 stage の参照と bit 一致、device cache ありでは常駐 expert が
   構成ごとに違う分だけ 1 stage の exact 対 cache の差と同じ。greedy token は 17/11 のまま）。
   3 stage 以上と Windows の実行、分割時の decode 速度は未検証。

## Sage-A1.5 の EXL3 4.0 bpw 変換

`<models>/safetensors/Sage-A1.5-exl3-4.0bpw` に置いた（19 GB、5 shard、
128346 tensor）。recipe は 35B の 4.0 bpw と同じ
（`-b 4.0 -hb 6 -mb 4 -cb mul1 -cr 32 -cc 1024 -ss 4096 -d 0 -cpi 300 --fast_head_capture --out_scales always`）。

読み込み確認（R9700）: 10240 / 10240 expert 常駐、routed expert のヒット 100%、
生成 1 トークン 70.68 tok/s（`records/quality-instruct/conversion-sage-a15-4.0.log`）。

変換は **`--resume` が要る**。付けないと exllamav3 の `conversion/convert_model.py` が
checkpoint を無視して先頭からやり直し、work フォルダの `qtensors/` を消す
（2026-10-22 にそれで layer 11 までの仕事を 1 回失した）。コンテナ内の watcher
（`records/conversion-watcher-container.sh`）は `--resume` を付けるようにしてある。
