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

`ExpertView` は gate/up/down の三つの packed/raw projection を独立に保持する。CPU と SYCL の expert 演算は
`down(silu(gate(x)) * up(x))` を計算し、router weight は適用しない。

`Exl3ExpertSource` は layer/expert ごとの明示的な名前表を受け取り、要求された expert の圧縮 blob だけを
読み込む。blob の pointer は layer/window 中は安定し、`copy_blob()` は別の layer の圧縮成分もコピーできる。
`ExpertPool::prepare_exl3()` は worker の scratch を事前確保し、`run_exl3()` は FP32 入力で expert を計算する。
CPU dispatch は router の出力行を保ち、既存形式の GPU plan・residency との併用を拒否する。

format registry は実際の blob layout の全成分を比較し、process 内で使う ID を返す。`ExpertCache::open_buckets()`
は形式ごとの容量を保ち、別形式の slot への admission と容量を超えるコピーを拒否する。既存の
`open()` / `open_sized()` は format 0 のまま扱う。`exl3_grouped_fp32()` は device 上の形式表と動的な group/token/destination 計画を読み、各 expert の
独立した gate/up/down 形式を保持したまま FP32 出力を計算する。scratch は事前に確保する。
`Exl3ExpertRuntime` は明示した resident expert を形式別 cache に読み込み、hit を source I/O より前に判定し、
CPU miss と GPU hit の出力を router 順の行へ配置する。永続的な計画を graph に記録できる。
`ExpertDispatch::exl3_runtime` と `MoeWindowTier::exl3_runtime` は、この runtime を既存の CPU-row/GPU-hit
merge に接続する。現在は segmented graph の1 group 構成を使い、2 group と monolithic graph は拒否する。
各 ring と最後の segment を同期してから永続計画を再利用する。Qwen3.5 MoE の起動時 binding は以下の経路を使う。prefill は別に行う。

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
    exl3_pool_test exl3_expert_source_test exl3_expert_formats_test expert_cache_buckets_test
ctest --test-dir build/free -R '^exl3_' --output-on-failure
```

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
expert の residency は実際の blob byte 数と device の容量から決める。部分常駐時は既存の segmented graph に接続する。
全 `(layer, expert)` が GPU に常駐する場合は、device 上の route ID から graph 内で経路表を作り、
CPU の ring 応答を省く。同じ owner の plan/scratch は in-order queue 上で順番に再利用する。
`STRATA_EXL3_RESIDENT_GRAPH=0` または `STRATA_VERIFY_SEGMENTED=1` で比較用に従来経路を選べる。
SG32 を使える device では grouped expert の H128 変換を register FWHT で実行し、他では既存の変換を使う。

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

```sh
strata --exl3 /path/to/qwen38-exl3 --tokens 16,10,16,28 --max-new 2 \
    --context 8 --stage-devices 0 --spec 0 --max-t 1 --prompt-chunk 0
```

この接続は greedy の単 token AR を実行する。部分常駐時は CPU expert fallback を使う。
`--dump-logits FILE` は生成時の FP32 logits を保存し、`--stop-eos` または `--eos ID` で終了条件を指定する。
MTP、vision、serve、batch と複数 token の prompt chunk はこの初期 CLI では要求時に拒否する。
Qwen3.5 の CLI/MTP 接続は次の経路を使う。

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
`exl3_flash_probe` は token ID を直接受け取る。現時点では MTP、vision、複数 token の window を要求すると拒否する。

この経路の数値・機能確認と、同じ元重み・実効保存量での品質比較は別に行う。最適化前の部品や初期接続の
速度は診断値として扱い、decode 速度を維持できるかの判定には使わない。
