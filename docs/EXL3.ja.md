<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# EXL3 の開発用読み込み経路

モデル全体の対応範囲と、同じ実効 bpw での精度・最適化後の decode 速度の判定は
[EXL3 対応計画](EXL3_PLAN.ja.md)に従う。以下は形式の読み込みと数値照合に使う開発用の部品であり、
モデル全体の EXL3 対応を示すものではない。

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
各 ring と最後の segment を同期してから永続計画を再利用する。モデル起動時の binding と prefill は別に行う。

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
`NativeEmbed::load_exl3()` は raw F16/BF16/F32 embedding の元の dtype を保ち、host token と device token ID
の両方から行を取得する。owner と embedding の stream は capture 前に準備する。

この経路の数値・機能確認と、同じ元重み・実効保存量での品質比較は別に行う。最適化前の部品や初期接続の
速度は診断値として扱い、decode 速度を維持できるかの判定には使わない。
