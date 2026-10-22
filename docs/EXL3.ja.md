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
gate/up の K 分割数は work-group、sub-group と local memory の上限から決める。
`STRATA_EXL3_GU_SPLIT_K` に正の整数を指定すると、その値を上限として device ごとに調整できる。
`1` は一つの sub-group で積和を実行する比較経路になる。

AR では `--prompt-chunk 0 --spec 0` を使う。`--spec 1` も MTP 無効として扱う。正の `--prompt-chunk` は
`max-t` 以下の decode window に分けて prefill を実行し、最後の行の出力を返す。圧縮 weight はそのまま
使い、専用の batched prefill とは区別する。MTP は次の接続を使う。
Qwen3.8 の projection map と n-gram codec/table はあり、同モデルの起動時 binding は別に行う。

開発用 `plain_model_logits` は `qbench-trace/1` の token ID を再 tokenize せず順に入力し、指定した行の
FP32 logits と manifest を保存する。`--exl3 DIR --metadata-gguf GGUF` または `--native GGUF` を選び、
`--trace TRACE.json --out-dir DIR` を指定する。`--rows 0,4` で対象行を限定でき、native 経路の
`--expert-profile FILE` は同じ residency 条件で照合するために使える。精度判定は元の BF16 教師に対する
`KL(P_BF16 || P_candidate)` だけを使う。速度の判定にはこの診断用 dump の実行時間を使わない。

## Qwen3.8 Flash の AR 接続

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

この接続は greedy の単 token AR を実行する。部分常駐時は CPU expert fallback を使う。
`--dump-logits FILE` は生成時の FP32 logits を保存し、`--stop-eos` または `--eos ID` で終了条件を指定する。
MTP、vision、serve、batch と複数 token の prompt chunk はこの初期 CLI では要求時に拒否する。
Qwen3.5 の CLI/MTP 接続は次の経路を使う。

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

native IQuant は `--native GGUF --expert-profile FILE` に置き換える。JSON 出力には各 repeat と全体の
mean/median/p90 latency、token/s、expert の hit/miss、resident/slot 数を含む。token 列、context、
max-t、常駐条件と実行環境をそろえ、最適化と精度確認の後に速度ゲートへ使う。

## 出力 head の早期接続

Flash の CLI に `--exl3-head DIR` を加えると、その archive の `lm_head` を canonical の
`output.weight` に明示的に対応付ける。shape は一致が必要で、黙って行を切ったり別形式へ変換したりしない。
現在の接続は single GPU の CLI 用で、GGUF native head、MTP、serve、batch、pipeline windows との併用は
拒否する。trunk と embedding は既存の pack/native 経路を使う。

`Exl3Dense` が圧縮成分を device へ転送して所有する。非量子化の `lm_head.weight` がある場合は
F16/BF16/F32 の元 dtype を保って転送し、専用の raw linear で計算する。bias は shape を検証して FP32 にする。
使用する各 stream の scratch は graph capture 前に
準備し、owner はその graph より長く生存する。`lm_head()` は canonical の plane を解釈する前に EXL3 を
選び、head mixer の FP32 出力をそのまま渡す。
開発用の `Exl3Dense::load()` overload は任意の canonical 行名を明示でき、複数の owner を独立に保持する。
CLI の `--exl3-head` が選ぶ対象は引き続き `output.weight` である。
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
