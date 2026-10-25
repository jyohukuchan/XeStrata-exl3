<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# EXL3 の開発用読み込み経路

モデル全体の対応範囲と、同じ実効 bpw での精度・最適化後の decode 速度の判定は
[EXL3 対応計画](EXL3_PLAN.ja.md)に従う。以下は開発用の読み込み経路と数値照合の部品を説明する。Qwen3.5 MoE の single GPU AR/MTP 経路を
接続しており、全対象モデル・prefill の対応と性能の判定は段階的に進める。

## Artifact

`strata::artifact::exl3::Archive` はモデルディレクトリの config、safetensors index と header を読み、
shard、tensor の shape・dtype・範囲と codebook marker を検証する。`metadata.total_size` は参考値として
公開し、実際の tensor payload 量を別に数える。量子化設定の JSON 本文は保持するが、実行に使う bit 幅と
codebook は trellis と marker から決める。

`linears()` は各行列の独立した descriptor を返す。2～5bit に加え、head などの1・6・7・8bitの例外、
cb0、mcg、mul1、F16 scale、旧形式の符号 bitfield、F16/F32 biasを扱う。`materialize()` は指定した
行列の圧縮成分だけを読み、展開済みの重み全体は作らない。非量子化 tensor は raw descriptor として残る。
`find_linear()` は名前で descriptor を検索する。expert factory は gate/up/down を一つの blob にまとめ、
各成分の offset、bit 幅、codebook を保持する。blob を別のアドレスへコピーした後も同じ layout から
view を作れる。各 projection は packed trellis と raw を明示的に区別し、raw weight と bias の
F16/BF16/F32 を元の dtype と bit 列のまま保持する。raw を16bit trellis として代用しない。
同名の packed/raw 表現が同時にある場合は曖昧な入力として拒否する。

raw materializer は明示した軸ごとの並べ替え・部分選択を扱い、identity 変換では F16/BF16/F32 の
元の bit 列を保つ。norm の `+1` と GDN `A_log` の `-exp` は明示した場合だけ F32 で適用する。
F32 weight を要求する kernel 用の widening も別操作として提供し、shape と payload 長を検証する。

index 外の safetensors も列挙する。`exl3_ngram_trellis` は通常の linear と区別する。
既知の `mtp_hyper_connection_mixer_patch.safetensors` は本体より後に適用し、元のファイルと override を
sidecar descriptor に残す。その他の sidecar が同じ tensor 名を上書きする場合は拒否する。

Python の header inventory は重みをロードせず実行できる。

```sh
python3 tools/exl3_inventory.py /path/to/exl3-model > /path/to/inventory.json
```

## 演算部品と照合

CPU と SYCL は共通の `MatrixView` を使う。SYCL の FP32 GEMV/GEMM は圧縮 trellis を積和中に復号し、
input/output の H128、scale、bias を適用する。batched GEMM は padding を含む row stride と明示的な
event 依存を受け取り、scratch は呼び出し側が確保する。CPU の linear も行列全体を展開せずに計算する。
CPU は三つの codebook の全 state に対する復号表を一度だけ作り、linear では F16 の丸めを保った F32 値を
16×16 の一時 tile に読み出す。bit 幅ごとの展開で state 抽出の実行時除算を避け、従来の FP64 積和順序を保つ。
AVX2 対応 CPU では mul1 の packed tile を register permutation と小さい復号表で処理する。
`STRATA_NO_EXL3_AVX2=1` は汎用経路へ戻す。

GPU 側の codebook 2 (mul1) の decode は、byte sum B に対して (887 * B - 452608) * 2^-17 を返す。
この値は 2^-17 の倍数で絶対値が 2^19 未満なので、F32 で厳密に表せる。
GPU は同じ数値を返す2つの形を使う。portable な形は F32 で厳密な値を作って F16 へ一度だけ丸める。
half の形は参照実装と同じ F16 の fused multiply add (hfma16(F16(1024 + B), 0x1EEE, 0xC931)) で行う。
どちらも同じ厳密値を一度だけ F16 へ丸めるので、bit 単位で一致する。
device が FP16 の演算を報告するときは half の形を使い、報告しない device では portable な形を使う。
`STRATA_EXL3_CB2_HALF_FMA=0` は portable な形を強制する。
F16 の RNE 境界、FP32 の加算順、Hadamard 変換、bias、重みの保存形式と bpw はどちらの形でも変わらない。

`STRATA_EXL3_CPU_ROWPARALLEL=1` は CPU miss を Hpre、gate/up の出力 block、SiLU と down Hpre、
down の出力 block の4段階に分け、既存 pool で処理する。raw projection と準備容量を超える batch は
expert 単位の経路を使う。worker 数は既存の設定で選び、コア数を増やした効果は実モデルで確認する。
`STRATA_EXL3_CPU_INTDOT=1` は実験用の mul1 整数積和を有効にする。Hpre 後の入力を Q8 にし、
Gaussian codebook の affine 定数を補正するが、codebook の最終 F16 丸めは省略するため、既定の exact 経路と
精度契約が異なる。CB0/1、raw、未対応 CPU、非有限・過小入力と整数 overflow の可能性がある K は exact 経路へ戻す。
全行列の展開や保存 bpw の増加は行わない。品質と速度の採用判定には、この設定を明記した実モデルの KL と
CPU・GPU 併用時の decode 計測が必要になる。

`ExpertView` は gate/up/down の三つの packed/raw projection を独立に保持する。CPU と SYCL の expert 演算は
`down(silu(gate(x)) * up(x))` を計算し、router weight は適用しない。

`Exl3ExpertSource` は layer/expert ごとの明示的な名前表を受け取り、要求された expert の圧縮 blob だけを
読み込む。blob の pointer は layer/window 中は安定し、`copy_blob()` は別の layer の圧縮成分もコピーできる。
圧縮 blob は層をまたぐ LRU cache で再利用し、展開した重みは保持しない。容量は圧縮 payload 量と
現在の利用可能 host memory から決め、層を一巡するごとと、既知の容量を超えて blob を保持しようとした
ときに再確認する。`STRATA_EXL3_SOURCE_CACHE_MIB` は追加の
上限で、0 は window ごとの一時読み込みへ戻す。要求された全 expert を eviction 前に pin し、容量不足では
window 中だけ保持する。GPU preload の cold `copy_blob()` は CPU cache に追加しない。
終了時の source 統計は cache hit/miss、materialization 回数・時間と payload bytes を示す。payload bytes は
成功した tensor span の合計で、OS の実ディスク I/O 量ではない。
GPU cache と workspace の確保後、所有 layer の CPU 担当分を圧縮したまま事前読み込みする。
全量と既存 cache が現在の RAM 予算に収まる場合だけ開始し、不足時は既存データを退避せず lazy cache を使う。
GPU 常駐分と他 stage の layer は対象外で、途中で余裕が減った場合も予算を再確認して停止する。
`STRATA_EXL3_SOURCE_PREWARM=0` はこの事前読み込みを無効にする。既存の LRU cache は維持する。
`ExpertPool::prepare_exl3()` は worker の scratch を事前確保し、`run_exl3()` は FP32 入力で expert を計算する。
CPU dispatch は router の出力行を保ち、既存形式の GPU plan・residency との併用を拒否する。

format registry は実際の blob layout の全成分を比較し、process 内で使う ID を返す。`ExpertCache::open_buckets()`
は形式ごとの容量を保ち、別形式の slot への admission と容量を超えるコピーを拒否する。既存の
`open()` / `open_sized()` は format 0 のまま扱う。`exl3_grouped_fp32()` は device 上の形式表と動的な group/token/destination 計画を読み、各 expert の
独立した gate/up/down 形式を保持したまま FP32 出力を計算する。scratch は事前に確保する。
`Exl3ExpertRuntime` は明示した resident expert を形式別 cache に読み込み、hit を source I/O より前に判定し、
CPU miss と GPU hit の出力を router 順の行へ配置する。永続的な計画を graph に記録できる。
`STRATA_EXL3_TIMELINE=1` は host の prepare、GPU submit、CPU pool 区間の開始・終了を
steady-clock の ns で標準エラーへ JSON 行として出す。GPU submit の時間は host の投入時間であり、
GPU kernel の実行時間ではない。GPU profiler の timeline と照合して実行の重なりを確認する。
`STRATA_EXL3_PREPARE_DETAIL=1` は、呼び出す host thread ごとに prepare の partition、source begin、
borrow、host plan と CPU の4 phase の task 数・待機時間を集計し、thread 終了時に
`exl3_prepare_summary` JSON を標準エラーへ出す。RAM guard は source begin 内の部分時間で、
起動時の prewarm は含めない。CPU phase は row-parallel を使えた batch のみを集計し、fallback 回数は測らない。
Dense projection の圧縮 word を共有する任意の経路は、[tile の先読み](EXL3_WORD_TILE.ja.md)を参照。
`ExpertDispatch::exl3_runtime` と `MoeWindowTier::exl3_runtime` は、この runtime を既存の CPU-row/GPU-hit
merge に接続する。現在は1 group 構成を使い、2 group は拒否する。既定は segmented graph を使う。
segmented 経路では `publish_window_plan` の直前 callback が同じ in-order queue に hit-only grouped kernel を投入し、
その後 host の CPU miss 計算を開始する。CPU が戻ってから次 segment の既存 merge graph が CPU rows と GPU hit rows を
合成するため、mapped flag の可視性や device-side spin に依存しない。各 ring の merge segment と最後の segment を
同期してから plan/cache/scratch を再利用する。既存の CPU-first `enqueue()` 契約と all-resident direct path は維持する。
`STRATA_EXL3_DEVICE_HIT_PLAN=1` は、doorbell 可視性の実行時 probe に通った in-order queue で、
GPU が hit plan を作る monolithic graph を使う。CPU miss は GPU の出力対象から除外し、host は
GPU を enqueue せず CPU 結果と compact hit list を公開する。次 ring の doorbell を観測してから前 ring の
host state を再利用し、最後は stream を同期する。CPU 処理が失敗した場合も、残りの ring に空の plan と
ゼロの CPU rows を返して graph を drain してからエラーを返す。probe 不合格、または
`STRATA_VERIFY_SEGMENTED=1` の場合は segmented 経路を使う。GPU cache の residency は起動後に変更しない。
Qwen3.5 MoE の起動時 binding は以下の経路を使う。prefill は別に行う。

専用 n-gram decoder は、先頭の F16 row scale と160値の ringを通常の linear と区別して復号する。
CPU は要求した row ID を選べ、SYCL は取得済みの packed rowを処理する。head bias は明示した head IDで
選ぶ。FP32 出力と padding を扱い、呼び出し側はモデルの演算に必要な最後の F16 丸めを別に行う。
この row codec を非同期 I/O と cache へ接続する経路は次の reader と table が提供する。
専用の blocking `exl3_ngram::Reader` は sidecar の shard と head 範囲を検証し、要求された行だけを読む。
sidecar の multiplier、signed-int64 overflow、EOS 境界を使う row ID 計算も提供する。
`row_spans()` は検証済みの論理行とファイル内 offset の対応を公開する。既存の
`PleReader::open_spans()` へ渡すと、物理的な並び順や隙間がある分割 tensor を一つの非同期 I/O worker と
row cache で扱える。モデル起動時の sidecar 選択と binding は別に行う。
`PleTable::open_exl3()` はこの reader と decoder を既存の issue/collect、batch、prefetch に接続し、
head ごとの bias と最後の F16 丸めを適用する。現在は direct I/O を扱い、mmap と lock は明示的に拒否する。
モデルへの binding 用に sidecar の hash 定数も公開する。

NumPy のある環境で tests を有効にしてビルドすると、独立した fixture を使った CTest を実行できる。

```sh
cmake --build build/free --target exl3_artifact_test exl3_cpu_parity exl3_parity \
    exl3_cpu_expert_parity exl3_expert_parity exl3_cpu_ngram_parity exl3_ngram_parity \
    exl3_cb2_parity \
    exl3_pool_test exl3_expert_source_test exl3_expert_formats_test expert_cache_buckets_test
ctest --test-dir build/free -R '^exl3_' --output-on-failure
```

`exl3_cb2_parity` は codebook 2 の decode の2つの形を device 上で照合する。
全ての byte sum (0〜1020) と全ての16 bit state (0〜65535) について、half の形と portable な形の F16 の
bitが一致することを、独立に定義から計算した値とも照合する。
加えて、codebook 2 の射影を2つの形で実行した結果が bit 単位で一致することも確かめる。

実 checkpoint の一行列について、archive、CPU/GPU の復号と積和、engine の weight owner をまとめて
照合する probe もある。診断用の復号だけは選択した一行列を展開する。

```sh
cmake --build build/free --target exl3_archive_parity
build/free/exl3_archive_parity /path/to/exl3-model model.layers.0.mlp.experts.0.down_proj
```

## Qwen3.5 MoE の AR 接続

`--exl3 DIR --metadata-gguf GGUF` は `PlainModel` の embedding、GDN、attention、router、
shared/routed expert、norm と出力 head を EXL3 archive から読み込む。GGUF はモデルの幾何形状と
tokenizer のための bootstrap に使い、重みの fallback には使わない。tensor 数ゼロの metadata-only
GGUF も使える。config と bootstrap の形状を照合し、補助 tensor の要素数も kernel の要求に合わせて検証する。

bootstrap は EXL3 ディレクトリの config/tokenizer から作れる。元の BF16 weight は不要である。
converter の Python 依存を用意した環境で、まず tensor 数ゼロの GGUF を作り、補助ツールで config と照合する。
補助ツールは重み入り GGUF を拒否し、block/nextn 数以外の metadata を元の型・byte 列で保持する。

```sh
python3 third_party/main/llama.cpp/convert_hf_to_gguf.py /path/to/exl3-model \
    --vocab-only --no-nextn --outtype bf16 --outfile /path/to/vocab-only.gguf
python3 tools/exl3_metadata.py /path/to/vocab-only.gguf --exl3 /path/to/exl3-model \
    --output /path/to/metadata-only.gguf
```

```sh
strata --exl3 /path/to/exl3-model --metadata-gguf /path/to/metadata-only.gguf \
    --prompt-chunk 0 --spec 0 --max-t 1 --tokens 16,10,16,28 --max-new 2
```

packed projection は H128 block の並べ替えを行い、GDN の V-head の順序を canonical 配置へ移す。
raw projection は軸ごとの明示的な変換で元の F16/BF16/F32 を保持する。norm の `+1`、GDN の
`A_log` の `-exp` と FP32 を要求する補助 weight の widening だけを個別に適用する。
expert の residency は実際の blob byte 数と device の容量から決める。部分常駐時は既定で既存の segmented graph に接続する。
全 `(layer, expert)` が GPU に常駐する場合は、device 上の route ID から graph 内で経路表を作り、
CPU の ring 応答を省く。同じ owner の plan/scratch は in-order queue 上で順番に再利用する。
`STRATA_EXL3_RESIDENT_GRAPH=0` または `STRATA_VERIFY_SEGMENTED=1` で比較用に従来経路を選べる。
SG32 を使える device では grouped expert の H128 変換を register FWHT で実行し、他では既存の変換を使う。
gate/up の K 分割数は work-group、sub-group、local memory と register の上限から決める。
register の上限は device が報告する work-group あたりの register 数と、コンパイル済み kernel が
報告する work-item あたりの register 数および最大 work-group size から取る。device が報告しない
上限は拘束しない。
`STRATA_EXL3_GU_SPLIT_K` に正の整数を指定すると、その値を上限として device ごとに調整できる。
`1` は一つの sub-group で積和を実行する比較経路になる。

AR では `--prompt-chunk 0 --spec 0` を使う。`--spec 1` も MTP 無効として扱う。正の `--prompt-chunk` は
`max-t` 以下の decode window に分けて prefill を実行し、最後の行の出力を返す。圧縮 weight はそのまま
使い、専用の batched prefill とは区別する。MTP は次の接続を使う。
Qwen3.8 の projection map と n-gram codec/table はあり、同モデルの起動時 binding は別に行う。

開発用 `plain_model_logits` は `qbench-trace/1` の token ID を再 tokenize せず順に入力し、指定した行の
FP32 logits と manifest を保存する。`--exl3 DIR --metadata-gguf GGUF` と
`--trace TRACE.json --out-dir DIR` を指定する。`--rows 0,4` で対象行を限定できる。精度判定は元の BF16 教師に対する
`KL(P_BF16 || P_candidate)` だけを使う。速度の判定にはこの診断用 dump の実行時間を使わない。

## Qwen3.8 Flash の AR 接続と spec2

Qwen3.8 の `--exl3 DIR` は archive の HF config から直接起動し、metadata GGUF を必要としない。
`--metadata-gguf` を指定した場合は EOS metadata の取得に使う。`--tokens` または `--tokens-file` で
入力 ID を渡し、`--stage-devices` で層分割する device ordinal を選べる。
`--stage-layers` で各 stage の層数（例 `18,18,12`）を指定でき、既定は均等な分割である。
`--pool-workers N` は stage ごとの CPU worker 数を指定する。既定の0は物理 core に基づく自動設定で、
最適な値は実モデルで測定する。

```sh
strata --exl3 /path/to/qwen38-exl3 --tokens 16,10,16,28 --max-new 2 \
    --context 8 --stage-devices 0 --spec 0 --max-t 1 --prompt-chunk 0
```

この接続は既定で greedy の単 token AR を実行する。部分常駐時は CPU expert fallback を使う。
`--dump-logits FILE` は生成時の FP32 logits を保存し、`--stop-eos` または `--eos ID` で終了条件を指定する。
`--spec 2` から `--spec 6` は MTP の serial draft loop を実行する（窓あたり最大 N-1 draft）。verify に通った draft だけを採用し、通らない
window は snapshot から戻して AR で書き直すので、token 列と logits は AR と一致する。draft の expert は
自前の device cache を持ち、全 draft expert が常駐する場合は embedding gather の後の draft 本体
（norm、射影、block、常駐 expert flow、head）全体を 1 つの graph として replay する。常駐しない場合は
CPU pool 経路に落ちる。trunk は全 expert 常駐の stage では token 全体を 1 つの graph として replay する
（`STRATA_EXL3_TOKEN_GRAPH=0` で層ごとの起動に戻る）。vision、serve、batch と複数 token の
prompt chunk は引き続き要求時に拒否する。Qwen3.5 の CLI/MTP 接続は次の経路を使う。

`--spec-min-p P`（0〜1 の確率、既定 0）は draft loop を選択的にする。MTP head の下での
softmax 確率が P 未満の draft の後は、その窓ではそれ以上 draft を提案しない（窓の最初の
draft は常に出す）。既に accept された draft はそのまま残り、窓は全 draft を提案した
場合と全く同じに閉じるため、生成 token 列と logits は AR および通常の `--spec` と引き続き
一致し、変わるのは窓が verify する draft の数だけである。閾値 0 では連鎖は止まらず、
通常の `--spec` の動作になる。実行は閾値で打ち切った窓の数を報告する。

### device cache（Qwen3.8 Flash）

Qwen3.8 Flash の接続は、GPU が保持する expert の device cache を使う。cache の大きさは
**device が報告する空き VRAM** から決める: 予約（合計 VRAM の 1/32、192〜768 MiB の間）を
差し引いた room に収まる (layer, expert) 組だけを常駐させ、収まらない組は従来どおり source から
読んで CPU で計算する。VRAM の少ない card には小さい cache が入る。cache を開けないときは
CPU miss 経路だけで走り、その旨を表示する。

```sh
STRATA_EXL3_DEVICE_CACHE=0            # cache を使わない（全部 CPU）
STRATA_EXL3_DEVICE_CACHE_MIB=4096     # cache の予算を MiB で指定する
STRATA_EXL3_DEVICE_CACHE_ORDER=spread # spread | visited | profile
STRATA_EXL3_DEVICE_CACHE_PROFILE=PATH # profile order の profile ファイル
```

`spread`（既定）は各層に同じ数の expert を与える。`visited` は層を訪れる順に、層を丸ごと
埋めてから次の層へ進む。`profile` は `tools/make_profile.py` の profile（`STRP` 形式）の
順に、hot な組から入れる。

常駐セットはコマンドラインからも選べる（Strata の `--expert-profile` 相当）:

```sh
strata --exl3 DIR --tokens-file PROMPT --expert-profile-save profile.bin   # 計測実行: 発火頻度を記録
strata --exl3 DIR --tokens-file PROMPT --expert-profile profile.bin        # 頻度順に常駐
strata --exl3 DIR --tokens-file PROMPT --expert-miss direct                # 非常駐を GPU が直接読む
```

`--expert-profile-save PATH` はその実行の routed entry を (層, expert) ごとに数え、終了時に
全組を頻度順に並べた `STRP` profile を書く。`--expert-profile PATH` はその順で常駐セットを
決める。profile がない・読めない・モデルが違う場合は従来の `spread` にフォールバックし、
その旨を表示する。実行終了時に常駐数・routed entry 数・ヒット数と率・CPU/direct miss 数を
表示する。

`--expert-miss direct` は非常駐 expert の miss 経路を変える: miss の canonical blob を
pinned host メモリに置き、GPU の grouped kernel が PCIe 経由で直接読んで計算する
（VRAM へのコピーはしない）。pinned の確保や device の host 読取ができない場合は
CPU miss にフォールバックする。既定の `--expert-miss cpu` が従来経路である。

測定（V620 0000:43:00.0、Qwen3.8 EXL3 3.05 bpw、`STRATA_EXL3_CPU_INTDOT=1` と
`STRATA_EXL3_CPU_ROWPARALLEL=1`、context 512、63 decode token、engine 自身の
`STRATA_EXL3_PREPARE_DETAIL=1` の counter）:

| 条件 | CPU miss | decode |
| --- | ---: | ---: |
| device cache なし | 32160 / 3216 window（10.00 per window） | 3.67 tok/s |
| device cache 15404 expert（26.72 GiB）、`spread` | 11801（3.67 per window） | **12.88 tok/s** |
| 同、`visited` | 11978（3.73 per window） | 12.25 tok/s |
| 同、`profile`（`data/expert-profile.bin`） | 9429（2.93 per window） | 12.73 tok/s |

三つの順序の decode 速度は 5% の差で、順序は miss の数では決まらない: 層の CPU 計算は
層の中で並列に走るので、1 層でも miss があればその層の分だけ時間がかかるからである。
既定は `spread`（層に平等に配り、profile ファイルが要らない）。
cache の起動コストは expert を読む分（V620 で 34 s、cache に載る expert の read と
device 向けの並べ替え）で、decode は 1 token あたり 0.27 s 速くなる。

**数値**: cache に載った expert は GPU の FP32 積和で、CPU の exact 経路（FP64）とは
最後のビットが違う。device cache を使わない構成の logits は参照と bit 一致で、
cache を入れると logit の最大絶対差は 1.4e-4（相対 1e-5、greedy の token は不変）。

### 複数 GPU（Qwen3.8 Flash）

EXL3 Flash の層は `--stage-devices` で指定した device に分割される。指定がなければ、engine が
使える GPU の数だけ stage を作り、層は `layer_count * stage / stage_count` で切られるので、
どの stage も 1 層以上を持つ。`--stage-layers 18,18,12` で各 stage の層数を明示的に指定できる
（stage 数と device のリストは `--stage-devices` のまま）。指定は stage 数と層数の合計で検証され、
1 stage も層を持たないか、合計がモデルの層数と合わないときは、理由を出して起動を止める。
出力 head を持つ最後の stage の層を減らすと、その card に device cache 用の VRAM が多く残る。
stage ごとに device、session arena、dense weight、expert runtime、device cache が別々のものになる。

**stage 境界をまたぐのは residual だけである。** 前の stage の residual（`hc * n_embd` 個の
float）は次の stage の session へ copy される。copy は device 間の直接 copy が1回である:
2 台の device の memory の間を runtime が直接 copy できるときはその経路を使い、できないときは
これまで通り host の bridge buffer を経由する。どちらになるかは起動時に小さな copy を実際に流して
確かめて決め、採った経路を起動時に報告する。device ごとに address space が別なので、
ほかの状態を device 間で持ち回ることはできない。session arena（GDN state、QSA の KV cache、
PLE の conv history）、dense weight、expert runtime、device cache、embedding は、それぞれが属する
stage の device 上にある。

**host が待つのはデータがまたぐところだけである。** forward は各 stage の仕事を順に発行して
進み、host が待つのは (1) 各層の doorbell（pre graph が expert の入力を publish するところ。
鐘を鳴らすのを見て待つので、expert の CPU 側の仕事は graph の終わりを待たずに始まる）、
(2) expert の merge の device-to-host copy、(3) stage 境界の residual の受け渡し、(4) 最後の
logits の copy、の4か所だけである。residual の copy は前の stage の最後の post graph の直後に
queue へ積まれ、次の stage はその完了を1回だけ待って自分の最初の pre graph を積む。post graph の
後では待たないので、1 層の終わりから次の層の始めまでの host の手間は queue の順序にしか残らない。

**PLE は stage 境界をまたがない。** PLE の block は層 1 のもので、層分割はその層をちょうど
1 つの stage に与える。その stage（PLE の home stage）が PLE の device 側の半分を持つ: key と
value の projection の owner、n-gram から集めた row、preprojection の出力先、scratch、conv
history。n-gram の row は home stage で読み、同じ stage の層 1 の前 graph で消費されるので、
PLE の一部が境界をまたぐことはない。PLE の token window（現在の token とその 2 個前の token）は
host 側の読み出し用なので、model 自身に持つ。home stage は層分割から決めるので、stage 数や
device の順序が変わっても、PLE の層を持つ stage は必ず 1 つで、ほかの stage は PLE に
触らない。

**buffer の所有者は「確保した runtime」で名指しする。** engine の allocation は
`Runtime::note_allocation` が確保した runtime を記録し、`Runtime::owner` はその記録を先に
読む。記録は allocation の大きさも持つので、allocation の内側の pointer（session arena から
切り出した buffer、expert cache の slot）も同じ runtime のものとして名付けられる。context の
scan だけでは区別できない: device USM は `get_pointer_type` がどの device に対しても "device"
と答えるので、scan は最初の runtime を名指しし、2 枚目以降の card では自分の buffer を
「他の runtime のもの」と誤認する。EXL3 の複数 GPU 経路はそのため、stage 1 の dense owner が
自分の buffer を拒否して止まっていた（"input or output belongs to another runtime"）。

測定（V620 0000:03:00.0 と 0000:43:00.0、Qwen3.8 EXL3 3.05 bpw、48 層、2 stage では 24 層ずつ、
`STRATA_VRAM_LIMIT_MIB=8192`、プロンプト 16,10,16,28 の 2 token）:

| 条件 | stage | 出力 |
| --- | ---: | --- |
| device cache なし、CPU の exact 経路 | 1 | logits SHA256 `1fe719ac...`（単一 stage の参照値） |
| device cache なし、CPU の exact 経路 | 2 | logits SHA256 `1fe719ac...`（単一 stage と **bit 一致**） |
| device cache あり、CPU の整数 dot と row 並列 | 1 | stage 0 に 1871 expert（3.25 GiB）、logits SHA256 `e88713bc...` |
| 同 | 2 | stage 0 に 3196 expert（5.54 GiB）、stage 1 に 3011 expert（5.22 GiB）、logits SHA256 `51e69318...` |

device cache ありの 1 stage と 2 stage の値は一致しない: cache が常駐させる expert の組が
違うためで、GPU が計算した expert は FP32 の積和、CPU の exact 経路は FP64 だからである。
その差（logit の最大絶対差 1.06）は、単一 stage で exact と cache の差（0.83 と 0.98）と
同じ程度にとどまり、greedy の token は変わらない。device の順序を逆にしても
（`--stage-devices 1,0`）logits は bit 一致する。

**3 stage（V620 ×2 と R9700、48 層を 16/16/16）でも同じことが成り立つ。** device cache なしの
logits は単一 stage の参照と bit 一致し、device cache ありでも greedy token は変わらない。
常駐 expert は 3 枚で 3563 + 3602 + 3355 = 10520（1 枚では 1848）になる。

**速さ**（Qwen3.8 Flash EXL3 3.05 bpw、48 層、V620 0000:03:00.0 と 0000:43:00.0、R9700
0000:07:00.0、プロンプト 16,10,16,28 からの decode 7 トークンの中央値、`STRATA_EXL3_FLASH_TIMING=1`、
他に GPU を使うジョブがないとき。tok/s はその中央値からの逆数）:

| 条件 | 受け渡しを直す前 | 現在 |
| --- | ---: | ---: |
| 1 GPU（V620、VRAM 上限 8 GiB） | 87.9 ms（11.4 tok/s） | 85.1 ms（11.8 tok/s） |
| 1 GPU（上限なし） | 70.3 ms（14.2 tok/s） | 70.2 ms（14.2 tok/s） |
| 2 GPU（24/24、上限 8 GiB） | 111.0 ms（9.0 tok/s） | 100.6 ms（9.9 tok/s） |
| 2 GPU（上限なし） | 88.6 ms（11.3 tok/s） | 85.6 ms（11.7 tok/s） |
| 3 GPU（16/16/16、上限 8 GiB） | 105.4 ms（9.5 tok/s） | 97.1 ms（10.3 tok/s）† |
| 3 GPU（上限なし） | 83.0 ms（12.0 tok/s） | 80.3 ms（12.5 tok/s） |
| 3 GPU（上限 8 GiB、`--stage-layers 18,18,12`） | — | 101.3 ms（9.9 tok/s） |

† 測定中に別のプロセスが CPU を使う spike が2 トークン（167 と 183 ms）混ざった。それを除いた
中央値。除かない中央値は 110.2 ms。

**expert cache の配分も効く。** `--stage-layers` で層の分割を、`STRATA_EXL3_DEVICE_CACHE_ORDER`
で常駐させる expert の順を変えられる。3 GPU、VRAM 上限 8 GiB、同じ測定:

| 条件 | tok/s | 常駐 expert（stage 0/1/2） |
| --- | ---: | --- |
| 均等 16,16,16 + `spread`（既定） | 9.76 | 3563 / 3602 / 3355 |
| 均等 + `visited` | 7.48 | 3563 / 3602 / 457 |
| 18,18,12 + `spread` | 9.61 | 3470 / 3513 / 3539 |
| **18,18,12 + `visited`** | **10.75** | 3470 / 3513 / 3539 |

1 枚（R9700、同じ上限）は 11.14 tok/s なので、**3 枚の最良はその 96%** まで来る。
`visited` は層ごとに集中して積むので、head を持つ stage の層数を減らさないとその stage の
常駐が 457 に落ちて逆に遅くなる。層数を減らした stage に余地が残り、全 stage が約 3500 expert
を常駐させることが速さに効く。cache の予算（`STRATA_EXL3_DEVICE_CACHE_MIB`）を 6 GiB に
切っても 9.82 tok/s と変わらない（常駐 10,273 で十分）。

phase の内訳で目につくのは `post` の待ちが消えたことと、`bridge`（residual の受け渡し）が
0〜0.5 ms だけ残ったことである: `post` は 6〜8 ms が 0.3〜0.7 ms になる。pre graph は doorbell の
鐘で待つので、expert の CPU 側の仕事が graph の残りと並んで走り、層の終わりから次の層の始めまでの
同期も消える。それでも複数 GPU では各層に約 0.2 ms ずつ（48 層で約 10 ms）と、stage の最初の
pre graph に 2.5〜4.2 ms が残る。上限なしでは常駐 expert が3枚で多くなる分、expert の CPU 側の
仕事が 4.5 ms まで下がり（1 枚では 15.7 ms）、3 GPU が 1 GPU に迫る。`--stage-layers 18,18,12` は
最後の stage（head を持つ card）の層を減らすので、その card の device cache は層あたり0〜295 expert
を載せる（均等分割では 0〜210）。

Windows での実行は確かめていない。

### decode の速度と参照実装（Qwen3.8 Flash）

同じ checkpoint（qwen38-flash-next-exl3-3.05bpw）を同じ V620 2 枚で走らせる rocm_exl3
（exllamav3 の ROCm port）は batch-1 decode で 23.96 tok/s（41.7 ms/トークン）を出し、XeStrata の
11.4 tok/s（87 ms/トークン）より 2.09 倍速い。参照実装の構造を読み、移せるものを入れた。
参照の測定値はコンテナ rocm-exl3-rdna2 の `/work/runs/qwen38/first-bench-512.json`（commit 37a96d50、
512 トークン prompt、32 トークン decode、repeat 2 の中央値）。

**参照実装の decode の形。** 層ごとに 1 つの graph を replay するのは XeStrata と同じだが、層と層の
間にホストの同期がない。device ごとに worker プロセスが 1 トークンの全層を 1 回のディスパッチで
enqueue し、ホストが待つのは stage 境界の hidden state の受け渡し（bench の device copy は
1 トークン 1 回）と最後の argmax だけで、argmax は GPU 上で行う。EXL3 expert はルーティングから
計算まで GPU 上に閉じ、層の全 active expert を 1 つの cooperative kernel で計算する。1 層あたりの
カーネル数は約 22。trellis の復号は使う時にカーネル内で行い（XeStrata と同じ）、codebook 2 (mul1)
の byte sum も同じ式を dot product 命令で計算する。expert は層と同じ device に全常駐で、expert が
device をまたぐことはない。活性化は F16、hyper-connection の残差は FP32 のストリーム、KV cache は
paged の F16/量子化。

**XeStrata 側の内訳**（V620 2 枚、`--stage-devices 0,1`、同じ 512 トークン prompt の decode、device
cache 全常駐、GPU カーネルの trace の合計）: 1 トークン 3448 カーネルで GPU 実行の合計 64.6 ms、
壁時計 87 ms。多い順:

| 部品 | カーネル/トークン | GPU ms |
| --- | ---: | ---: |
| raw F16 射影（HC down/up/inject、alpha/beta、router） | 460 | 26.8 |
| EXL3 dense 射影（Hadamard 込みで 3 カーネル/射影） | 650 + 300 | 14.7 |
| expert の grouped 流れ（7 カーネル/層） | 336 | 9.7 |
| GDN の step・norm・conv など | ~400 | 6 |
| router の top-10、RoPE、copy、その他 | ~1200 | 7 |
| host の隙間（expert の hit 計画、merge の copy など） | - | 22 |

**適用したもの。** いずれも加算の順序を一つも変えないので、値は bit 一致:

* raw F16 射影の 2 カーネル（`linear_fp32_rows` / `linear_fp32_products`）が、32 ずつ離れた位置を
  1 つずつ読んでいたのを、8 段分の読み出しを先に出してから同じ順に加算する形に変えた。K を
  wave で分ける products 形と 1・2 行/wave の rows 形の選択も出力数に応じて切り替えた（三つの形は
  同じ lane の和と同じ subgroup の reduction で、同じ float を返す）。1 トークンの raw 射影は
  460 カーネルで 26.8 ms から 18.2 ms に、products 形は 17.6 µs から 8.9 µs/発に減った。
* expert の merge の upload を、2 枚のホスト buffer を交替に使う非同期 copy にした。ホストが次の層で
  書き始める buffer と upload が読んでいる buffer が重ならず、upload の待ちが消える。

trellis dot（word tile）で次の 4 tile の語を先に読む形も試したが、graph replay の中では
1 発 43 µs が 72 µs に遅くなり、入れなかった。単体での micro benchmark では速く見えたが、
変えない Hadamard カーネルも同時に 2.3 倍速く見える測定条件の差であり、graph replay 内の
trace が正しかった。

測定（V620 2 枚、512 トークン prompt、32 トークン decode、`STRATA_EXL3_FLASH_TIMING=1`）:

| 状態 | decode tok/s | 1 トークン | pre | expert_wait | expert_prep | merge | head/post/bridge/emb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 適用前 | 11.38 | 87.1 ms | 62.0 ms | 13.9 ms | 4.5 ms | 3.9 ms | 2.8 ms |
| 適用後 | 12.39 | 80.0 ms | 56.5 ms | 14.2 ms | 3.9 ms | 2.7 ms | 2.6 ms |

生成 token は適用前後で同じ。

値は変わらない: `exl3_flash_probe --tokens 16,10,16,28 --max-new 2 --max-context 8 --dump-logits` の
logits SHA256 は device cache なしで単一 stage・2 stage とも参照値のままで、device cache ありの
構成（全常駐・部分常駐）の dump もこの一手の前の run と bit 一致、`raw_linear_parity` も通る。

**decode のホストを層の外へ出す（device 常駐の route 計画）。** 次の一手として、routed expert の
route 計画を device 常駐の表に移した。pre の graph が routed id と層の活性化を device 上に出し、
計画を作るカーネルが各 route を device 常駐の (layer, expert) → blob の表で引く。cache が持つ
expert はその出力の行に直接書き、持たない route は GPU の行き先を持たない（その行は CPU の
FP64 経路のもの）。stage の層の route が全部 cache にある時（この計測の構成）、ホストは doorbell
を待たず、計画を組まず upload せず、merge の D2H + add + H2D の往復もしない。routed id は
doorbell の mapped memory に届き続けるので、CPU miss の処理も logits の dump も従来どおり
ホストから見える - decode の律速経路の外で見える、というだけの話。層ごとのホスト計画は
`STRATA_EXL3_DEVICE_PLAN=0` で残してある。層の全 pair が常駐かどうかは、その runtime が請け負う
層の範囲（stage ならその stage の層）で判定する。

計測（同じ 512 トークン prompt、32 トークン decode、V620 2 枚、`STRATA_EXL3_FLASH_TIMING=1`）:

| 状態 | decode tok/s | 1 トークン | pre | expert_prep | expert_wait | merge | post | bridge | head | emb/ple |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| この一手の前 | 12.50 | 79.45 ms | 56.30 | 3.42 | 14.42 | 2.65 | 0.36 | 0.10 | 1.86 | 0.30 |
| 後 | 13.69 | 72.53 ms | 0.66 | 2.44 | 0 | 0 | 0.30 | 32.71 | 35.51 | 0.92 |

後では GPU の仕事が bridge（stage 境界の残差の待ち）と head の待ちの中にそのまま現れる。減った
約 7 ms は、層ごとのホストの往復が GPU の時間軸に入れていた隙間である。生成 token は前後で同じ。

**Hadamard の GEMV への畳み込みは測って入れなかった。** 入力・出力の H128 を dot の 1 カーネルに
畳む形を 2 通り作り、実測でどちらも遅いので残さなかった。128 列の block を 1 つの work group が
持つ形（K 分割の部分和を LDS に置く）は trellis dot を 1 発 43 µs から 76 µs に、1 wave が 1 tile を
通しで持ち部分和の木をレジスタで作る形は 197 µs に遅くなった。grid 単位の barrier が無い以上、
128 列の block の K 分割は work group の中だけに閉じ、dot 単体の並列度（出力 tile ごとに K を
wave で割る）に戻せない。3 カーネルの連鎖（dot 43 µs + H128 2 発 2 ms）の方がこの GPU では速い。
GPU 上の argmax も入れていない: logits を dump する probe がホストの logits を必要とし、greedy の
同点の扱いを変えないためで、logits のホストへの転送は head の 1 回に残る。

**残りはカーネルの速さにある。** 今は壁時計が GPU の実行そのものに律速されている。同じ run の
カーネル trace で、1 トークンの実行の合計は 56.3 ms / 3160 カーネル（2 device の合計）、外の隙間が
その外に約 16 ms。多い順に raw F16 射影 244 カーネル 18.6 ms、trellis dot 300 カーネル 15.5 ms、
expert の grouped 流れ（計画のカーネルを含め 336 カーネル）約 9 ms、GDN・RoPE・router など約 8 ms、
H128 の 2 段 650 カーネル 2.5 ms。参照の 23.96 tok/s との差は、層ごとのホスト同期ではなくこれらの
カーネルの速さに残っている。expert の流れはまだ 7 カーネル/層で、ホストの計画は消えたがカーネルの
数は減っていない。

**raw 射影の K 分割を長い K に広げた。** raw F16 射影には三つの形（K を wave で分けて LDS に積む
products 形と、1・2 行/wave の rows 形）があり、それまでは出力数が device の wave の数より少ない
時にだけ products 形を取っていた。rows 形は1 wave が1出力の K を 32 lane の直列な列で歩き、その列が
128 本を超えると列の待ち時間がそのまま射影の時間になる。そこで K が大きい射影（4096 以上）では
wave 数に余裕があっても products 形を取るようにした。境界の値は 36 compute units のカードで測った
もの（320 出力・K 10240 の射影は半分、512 出力・K 2560 の射影は数パーセント悪化）で、他のカードで
測り直せる。加える順序は一つも変わらない（同じ lane の和と同じ subgroup の reduction）ので、値は
bit 一致:

計測（V620 2 枚、512 トークン prompt、32 トークン decode、`STRATA_EXL3_FLASH_TIMING=1`）:

| 状態 | decode tok/s | 1 トークン | pre | expert_prep | post | bridge | head | emb/ple |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 適用前 | 13.73 | 72.30 ms | 0.73 | 2.39 | 0.33 | 32.59 | 35.36 | 0.89 |
| 適用後 | 14.70 | 67.45 ms | 0.70 | 2.39 | 0.31 | 30.14 | 33.04 | 0.86 |

カーネル trace（1 decode トークン、2 device 合計、1 トークンを head の起動の間で切った区間）では
HC down（320 出力・K 10240）が 1 発 107 µs から 63 µs に、raw 射影の合計は 16.3 ms から 11.7 ms に、
カーネルの合計は 57.8 ms から 52.8 ms に減った。HC up（10240 出力・K 320）と router（512 出力）は
水準が変わらない（43 → 46 µs、36 → 36 µs）。変えないカーネルの対照（expert の gate_up dot 5.0 ms、
lm_head 1.23 ms）も同じままで、カーネルの外の隙間は 24.1 ms で変わっていない。生成 token は前後で同じ。

**測って入れなかったもの。** 広い射影に1 wave あたり 4・8 行を持たせる形は、8 行がレジスタを溢れ
させて（VGPR 128 に scratch 180）43 µs が 55 µs に、4 行は wave 数が半減して 168 µs が 963 µs に
遅くなった: 入力の読み回しより wave の数が効く。rows 形の加算を配列にして残りの段も 2 段に分けた
形は VGPR が 16 個増えて 43 → 48 µs になり、元の本体に戻した。trellis dot では1 lane が読む 8 値の
語を1 tile に一度だけ取ってレジスタから取り出す形を作ったが、実行時に決まる語の添え字が private
segment に落ちて1 発 114 µs が 343 µs になり、戻した。trellis dot は参照実装より語を多く取りに
行く（8 値の語は散っているため）ので、ここが次の一手の候補である。

**dot の間を流れる段の融合（カーネル数を 3160 から 2848 に）。** カーネルの外の隙間は 1 起動
あたりの dispatch の費用で、空のカーネルを同じ replay に入れると 1 境界 7.6 µs（カーネル自体は
2.6 µs、1 本の in-order stream、同じ環境で測定）かかる。ホストの queue 提出の費用ではないので、
減らせるのは提出の数すなわちカーネルの数である。そこで、dot と dot の間で値をそのまま受け渡す
段を、次の3か所で1起動にまとめた:

1. **expert の流れ**（7 → 5 カーネル/層）: gate/up dot の出力 basis 変換 + bias、活性化
   （`(g / (1 + exp(-g))) * u`）、down 射影の入力 basis 変換は、128 幅の block を 1 work group が
   持ち、レジスタの上で順に流せる。参照実装の fused kernel A の tail と同じ形。fill されない
   block の端の行は、分けた3カーネルが activation の plane に読み返す古い値と同じものを読み返す。
2. **shared expert**（10 → 7 カーネル/層）: gate/up dot の2つの出力 basis 変換 + SwiGLU +
   down 射影の入力 basis 変換を1起動に。
3. **GDN**（36 層、7 → 5 カーネル/層）: z dot の出力 basis 変換、`gdn_out_norm`、ssm_out 射影の
   入力 basis 変換を1起動に。norm の和は D（FP64 がある device では double）で同じ warp sum と
   同じ順に積む。1 work group が1つの GDN head（state 128）に対応するので、state が 128 でない
   モデルは従来の3カーネルを使う。

いずれも段ごとに元のカーネルと同じ式を同じ順に計算するだけ（bias を dot が足す形を tail が足す形
に移すのも、足す相手と順序は同じ）なので、値は bit 一致。`STRATA_EXL3_FUSED_TAILS=0` で分けた
カーネルに戻る。

計測（V620 2 枚、512 トークン prompt、32 トークン decode、`STRATA_EXL3_FLASH_TIMING=1`）:

| 状態 | decode tok/s | 1 トークン | pre | expert_prep | post | bridge | head | emb/ple |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 適用前 | 14.63 | 67.81 ms | 0.75 | 2.50 | 0.35 | 30.44 | 32.83 | 0.92 |
| 適用後 | 15.12 | 65.59 ms | 0.69 | 1.79 | 0.30 | 29.58 | 32.26 | 0.93 |

カーネル trace（1 decode トークン、2 device 合計、1 トークンを head の起動の間で切った区間）では
起動数が 3160 から 2848 に減り、カーネルの外の隙間が 23.8 ms から 21.7 ms に減った。融合した
3 起動のカーネル時間は置き換えた 1.9 ms より短い 1.4 ms だが、dot などの対照の合計は run 間の
揺れ（±5%）の中にあり、カーネルの合計は 52.4 ms から 53.5 ms に見える。起動の費用は graph replay
の中（1 境界 7.8 µs）より外（11〜27 µs、expert の流れの 8 カーネルはここで走る）の方が高く、
次の一手はこの expert の流れを graph に収めることである。

値は変わらない: device cache なしの probe SHA256 は単一 stage・2 stage とも参照値のままで、
生成 token も前後で同じ。device cache ありの部分常駐では常駐数そのものが前後で 2 expert ずれ
（fused graph の node が少ないぶん起動時の空き VRAM が変わり、常駐の締め切りが動く）、
GPU の FP32 と CPU の FP64 で計算される expert の分担が変わるので logits が変わる。常駐数を
`STRATA_EXL3_DEVICE_CACHE_MIB` で揃えた比較（14638 常駐）では、この一手の前・後、融合の off・on
の4通りがすべて bit 一致した。

**常駐 expert の流れを graph に収め、shared expert と GDN の小さな起動を畳んだ（起動数 2848 から
2656 に）。** カーネル1起動の外の境界は dispatch の費用なので、次の3か所で起動そのものを減らした:

1. **常駐 expert の流れを層ごとの graph に。** cache が全常駐の時、expert の流れ（expert 行の
   clear、device route 計画のカーネル、grouped expert の5カーネル）はどのトークンでも同じ形で、
   それまでは pre/post graph の間に8起動が graph の外の境界（11〜27 µs）で走っていた。同じ
   ポインタを毎回そのまま引数に渡す起動のままで層ごとにキャプチャし、pre/post graph と同じに
   replay する（計画の層はキャプチャした graph の属性なので、層ごとに1つ）。host は1回の
   graph launch を出すだけになり、起動の境界は graph replay の値になる。doorbell probe の
   event record は常駐経路では何も待たないのでそこでは落とした。`STRATA_EXL3_RESIDENT_GRAPH=0`
   で従来の起動に戻る（比較用）。
2. **shared expert の入力2変換を1起動に。** gate/up の入力 basis 変換は同じ x を読むので、
   参照実装の `had_r_128_dual` の形で1つのカーネルにした。1 work group が1つの 128 幅 block の
   2つの変換を持ち、それぞれの view の `suh` で単独のカーネルと同じ式を同じ順に計算するので、
   値は2つのカーネルのものそのままで、48 層で48起動少ない。
3. **GDN の層の小さな起動を畳んだ（36 層）。** conv とその activation への copy と SiLU は
   `gdn_conv_silu` の1起動（conv の加算と conv state の送りは `gdn_conv_step` の順のまま、
   活性化は `silu_inplace` の式）。`gdn_beta_gate` と `gdn_gate` は `gdn_gate_beta` の1起動
   （参照実装も2つの epilogue を1つの kernel で計算する）。q の 1/sqrt(S) の prescale は q の
   l2 norm の起動に乗り（scale の範囲は q だけで、k は従来どおり）、`scale_inplace` の起動が
   消える。`STRATA_EXL3_FUSED_TAILS=0` で2と dot の間の融合は従来のカーネルに戻る。

計測（V620 2 枚、512 トークン prompt、32 トークン decode、`STRATA_EXL3_FLASH_TIMING=1`）:

| 状態 | decode tok/s | 1 トークン | pre | expert_prep | post | bridge | head | emb/ple |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 適用前 | 14.96 | 66.29 ms | 0.79 | 1.88 | 0.35 | 30.27 | 32.09 | 0.93 |
| 適用後 | 15.63 | 63.72 ms | 0.30 | 0.28 | 0.26 | 29.75 | 32.17 | 0.95 |

カーネル trace（1 decode トークン、2 device 合計、1 トークンを head の起動の間で切った区間）では
起動数が 2848 から 2656 に減り、カーネルの外の隙間が 21.77 ms から 18.68 ms に減った。減った
192 起動は入力2変換のまとめ（96 → 48）、conv の畳み込み（3 起動 → 1、36 層）、
gate/beta（2 → 1、36 層）、prescale（norm 2 起動 + scale 1 起動 → 2、36 層）で、カーネルの合計は
run 間の揺れ（±5%）の中にあり 52.4 ms から 53.2 ms。expert の流れの8起動は graph の replay に
なり、層ごとのホストの提出も 1 回の launch になった（expert_prep のホスト時間 1.88 → 0.28 ms）。
生成 token は前後で同じ。

値は変わらない: device cache なしの probe SHA256 は単一 stage・2 stage とも参照値の
`1fe719aca5e9...`。device cache ありの dump は常駐数を `STRATA_EXL3_DEVICE_CACHE_MIB` で揃えた
比較（1 stage 14638 常駐、2 stage 7319 常駐×2）でも、常駐数がそのままの比較（1 stage 部分常駐・
2 stage 全常駐）でも前後で bit 一致。全常駐の2 stage では常駐 expert の流れがこの graph で走る。

## Qwen3.5 MoE の MTP 接続

`--spec 2` 以上は archive の `mtp.fc`、三つの norm、`mtp.layers.0` の attention・router・shared/routed
expert を読み込み、既存の draft/verify と GDN state の rollback に接続する。GGUF bootstrap は
`block_count=trunk+1`、`nextn_predict_layers=1` が必要で、EXL3 config の MTP 層数も一つであることを確認する。
`ModelMap` は MTP を明示した場合だけ追加し、MTP 無効時の trunk と head はそのまま使う。
bootstrap は `tools/exl3_metadata.py` に `--include-mtp` を付けて作る。

```sh
strata --exl3 /path/to/exl3-model --metadata-gguf /path/to/metadata-with-mtp.gguf \
    --prompt-chunk 0 --spec 2 --max-t 3 --tokens 16,10,16,28 --max-new 8
```

FC と各 projection は packed/raw と bit 幅を保持する。MTP の embedding と出力 head は trunk と共有し、
最後の norm は `mtp.norm.weight` の値を使う。現在の draft はフル head で計算し、`--draft-vocab` を指定した
場合はその旨を表示する。dedicated MTP embedding と複数の MTP 層は明示的に拒否する。

## 固定 token 列の decode 計測

`plain_model_bench` は token ID の comma/空白区切りファイルを受け取り、同じ列を必要な長さまで繰り返す。
指定した context を準備し、その末尾の `--warmup` token を一 token ずつ実行してから、`--steps` token の
AR decode を計測する。各 repeat で状態を reset し、context 準備・reset・モデル load は計測から除く。
計測中に logits の host copy やファイル書き出しは行わない。greedy token の取得と同期は含む。

```sh
plain_model_bench --exl3 /path/to/exl3-model --metadata-gguf /path/to/metadata-only.gguf \
    --tokens-file /path/to/token-ids.txt --context 512 --warmup 16 --steps 128 --repeats 5 --max-t 1
```

JSON 出力には各 repeat と全体の
mean/median/p90 latency、token/s、expert の hit/miss、resident/slot 数を含む。token 列、context、
max-t、常駐条件と実行環境をそろえ、最適化と精度確認の後に速度ゲートへ使う。

## 出力 head の早期接続

旧 pack/native 経路に EXL3 head だけを差し込む `--exl3-head` は、旧 CLI とともに削除した。
現在の `--exl3` は trunk、embedding、head を同じ archive から読み込む。

`Exl3Dense` が圧縮成分を device へ転送して所有する。非量子化の `lm_head.weight` がある場合は
F16/BF16/F32 の元 dtype を保って転送し、専用の raw linear で計算する。bias は shape を検証して FP32 にする。
使用する各 stream の scratch は graph capture 前に
準備し、owner はその graph より長く生存する。`lm_head()` は canonical の plane を解釈する前に EXL3 を
選び、head mixer の FP32 出力をそのまま渡す。
開発用の `Exl3Dense::load()` overload は任意の canonical 行名を明示でき、複数の owner を独立に保持する。
`load_remapped()` は H128 単位の入力並べ替えと出力の選択・並べ替えを行い、trellis と scale を
再量子化せずに移す。raw tensor の非空の block map は拒否する。
`load_detached()` は外部の canonical table を使わず、明示した shape と同じ検証で projection を所有する。

開発用の layer/verify 経路では、この owner を router、attention/GDN projection、shared expert と
hyper-connection read に明示的に接続できる。raw F16/BF16/F32 projection は元の dtype を保ち、
FP32 activation を受け取る。HC norm は別の F32 vector を使う。モデル全体の名前・軸順・norm の変換は
読み込み側が検証する必要があり、これらの部品だけでモデル全体を起動できるわけではない。
`Exl3WeightSet` は指定した device と trunk 層範囲の dense projection・FP32 補助 weight を所有し、
借用した `WeightTable` に登録する。head と最後の HC mixer は明示して追加できる。各 stream の準備を
capture 前に済ませ、解放時は device の処理完了を待って自分の登録だけを戻す。embedding、routed expert、
PLE、MTP は別の owner に接続する。
`read_exl3_model_spec()` は archive の HF config から Qwen3.5/Qwen3.8 の geometry を読み、
GGUF の weight や仮の pack を必要としない。trunk と MTP を区別し、Qwen3.8 の MTP は
専用 loader へ接続するまで要求時に拒否する。`build_qwen38_mtp_map()` は FC、QSA/indexer、
HC、shared/routed expert を別の一層 map に検証し、trunk の map へ暗黙に混ぜない。
`Exl3Ple` は PLE の key/value projection を packed/raw のまま所有し、三つの norm は `+1` を適用して
FP32、conv は元の F16 として保持する。layer/verify は事前投影した FP32 の key/value を既存 PLE
post-op に渡す。norm、conv、projection の shape と PLE config は読み込み時に検証する。
`NativeEmbed::load_exl3()` は raw F16/BF16/F32 embedding の元の dtype を保ち、host token と device token ID
の両方から行を取得する。`prepare_stream()` は必要な device ごとの host table copy を capture 前に用意する。
owner と embedding は、それを使う graph より長く生存させる。

`Exl3FlashModel` は Qwen3.8 の単 token AR を所有する。device ごとの層範囲、embedding、PLE、
expert の CPU/GPU merge、HC、head を接続し、stage 間では residual を転送する。開発用の
`exl3_flash_probe` は token ID を直接受け取る。Flash の expert window は hit enqueue を CPU miss 計算より先に
行い、CPU 完了後に event を待つ。現時点では MTP、vision、複数 token の window を要求すると拒否する。

この経路の数値・機能確認と、同じ元重み・実効保存量での品質比較は別に行う。最適化前の部品や初期接続の
速度は診断値として扱い、decode 速度を維持できるかの判定には使わない。
