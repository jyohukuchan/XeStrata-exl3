<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# EXL3 対応計画

この計画は、XeStrata に EXL3 の 2～5 bpw と混合精度を組み込み、同じ実効 bpw で既存形式より高い精度を得ながら、最適化後の decode 速度を既存 XeStrata と同等以上にするためのものである。対象を途中で狭めず、未確認の範囲は `unverified` と記録する。

## 完了条件

- 2、3、4、5 bpw をそれぞれ単独で読み込み、prefill、decode、CPU expert、GPU expert、層分割で動かす。
- 主対象の 2、3、4、5 bpw で EXL3 の `mul1`、`mcg`、`legacy/default (cb0)` codebook を検証する。checkpoint に実在する output head の 1～8 または 16 bit は有効な例外として扱い、存在しない bit 幅を要件にしない。
- 層、行列、expert、MTP、head ごとのビット幅を持つ混合精度を読み書きし、可変サイズの配置、cache、転送、再構成を正しく扱う。
- dense、embedding、norm、output head、MTP head、vision など、量子化しないか別の精度を使うテンソルを明示的に扱う。checkpoint に記録された head/MTP の bit 数などの有効な例外を暗黙に丸めない。
- 同一の元モデル、revision、tokenizer、校正データ、サンプリング条件で比較する。
- bpw は重みだけでなく scale、codebook、trellis、metadata、padding、dense/head/MTP の保存量を含む実効値で揃える。
- 既存 XeStrata の同実効 bpw 形式に対して、BF16 基準の品質指標と固定評価文の結果で改善を示す。
- decode 速度は、最適化を完了した構成で同じ hardware、context、batch、KV、MTP 条件を測り、既存 XeStrata の 95% 以上を暫定合格とする。未最適化 prototype の速度だけで実現可能性を否定しない。
- 量子化精度の主比較では `STRATA_ROUTE_TAIL_SKIP=0` とし、全 routed expert の計算を維持する。
  条件により低順位の非resident expert を省略する既定の高速設定は別に記録し、同じ省略方針同士で速度を比べる。
  cache 配置や量子化形式で省略量が変わった結果を、そのまま量子化精度の改善と扱わない。

indexed shard の外にある safetensors sidecar も artifact の一部として扱う。`ngram_embedding.safetensors` の
`exl3_ngram_trellis` は通常の線形層の trellis ではなく、row 次元160の専用 n-gram codec なので、線形層の
descriptorへ混ぜず、専用decoderの対象として明示する。`mtp_hyper_connection_mixer_patch.safetensors` のような
override は本体の値へ黙って混ぜず、patch が base より優先される順序と実際に読んだファイルを記録する。同じ
実効 bpw の比較では、indexed linear の値と sidecar を含む artifact 全体の保存量を別々に示し、sidecarを除いた
値だけでモデル全体のbpwや精度を主張しない。

## 段階

### 0. 基準の固定と XeStrata の動作確認

XeStrata の対象 revision と、参照用に clone した Strata の tag/revision を記録する。環境構築中は、ソースと EXL3 metadata の静的調査、pack の形式一覧、Strata との構造比較を並行してよい。ただし、最初にモデルなしの device selftest、次に既存形式の小さな生成を実行し、GPU列挙、VRAM、runtime、CPU/GPU 経路、prefill、decode、MTP の基準が安定するまで EXL3 のモデル runtime 統合は開始しない。

記録する項目は、GPU ごとの device/backend、driver/runtime、build mode、CPU thread 数、モデル形式、実効 bpw、context、batch、KV 精度、MTP 設定、prefill tok/s、decode tok/s、VRAM/RAM、ログの主要な時間である。

### 1. 精度の成立性を先に確認

同じ元モデルから主対象の 2、3、4、5 bpw の EXL3 pack を作り、`mul1`、`mcg`、`legacy/default (cb0)` と代表的な混合精度 recipe を用意する。checkpoint に実在する head の bit 幅は付随する例外として保持する。BF16、既存 XeStrata 形式、EXL3 を同じ入力で比較し、perplexity または token log-likelihood、greedy token 一致率、固定した日本語・コード・長文・会話の評価を残す。

この段階で、少なくとも対象 bpw ごとに代表構成の改善方向、または改善できる bit 配分が確認できることを精度ゲートとする。改善がないときは kernel の実装不足と量子化 recipe の不足を分け、recipe を見直す。精度ゲート未達のまま大規模な高速化へ進めず、前提条件が不成立ならその証拠を記録する。

### 2. 形式と数値の最小統合

EXL3 metadata、tensor map、codebook、scale、trellis、alignment、shard を読み取るローダーを作り、CPU の参照 dequant と GPU の dequant/積和を照合する。まず固定 bit の `mul1` で一つの dense/expert 経路を XeStrata に接続し、モデル全体で一往復できる早期統合点を作る。

早期統合点では、各 codebook の小さな synthetic tensor、主対象の 2～5 bpw の境界、checkpoint に実在する head 例外、row 数と expert 数の違い、非量子化 tensor、MTP を検証する。CPU/GPU の中間値、logit、greedy 出力が基準実装と一致する範囲を記録し、差分がある場合は速度測定へ進まない。

### 3. decode 経路と混合精度の拡張

固定 bit の gate/up/down から、行列ごと、層ごと、expert ごとの bit を扱う。可変サイズの配置表を持たせ、padding で見かけだけ同じサイズにせず、cache eviction、PCIe/host transfer、prefill の複数 token、decode の一 token、MTP の draft/verify を同じ metadata から処理する。

主対象の 2～5 bpw、3 codebook、checkpoint に実在する mixed precision と dense/head/MTP の例外を組み合わせ、層分割と複数 GPU を含む互換性を確認する。対応できない入力を黙って別形式として読み込まず、具体的な理由を表示して失敗させる。

### 4. 最適化と速度判定

prototype の速度は診断値として保存する。復号、積和、Hadamard/scale、codebook 読み出し、kernel launch、graph、cache、CPU expert、PCIe/host transfer の時間を分解し、既存 XeStrata の同じ区間と比較する。速度低下がある間は、bit 幅を減らして隠すのではなく、ボトルネックを解消してから判定する。

各 bpw と各 GPU 構成について、最適化後の同一条件を複数回測る。中央値、ばらつき、warm-up 後の値、prefill と decode、MTP の acceptance と実効 token/s を残す。95% 未満なら未達として原因と次の最適化を記録し、未最適化の一回の測定だけで前提条件を不成立とはしない。

### 5. 最終確認

全 codebook、主対象の 2～5 bpw、checkpoint に実在する head 例外、代表的な混合精度、dense/MTP 例外、CPU/GPU、prefill/decode/MTP、複数 GPU を再確認する。品質と速度の両方を満たした bpw/hardware だけを達成とし、未確認の組み合わせは `unverified` とする。既存形式との比較表には、モデル全体と量子化対象の実効 bpw、精度、decode、prefill、VRAM/RAM、設定、最適化 revision を併記する。

## 前提条件を確認する時点

| 時点 | 確認 | 次の判断 |
| --- | --- | --- |
| 基準取得後 | XeStrata が安定し、同条件の比較値を取れるか | 不成立なら基盤を直す |
| 精度試験後 | 同 bpw の EXL3 に改善可能性があるか | recipe を再設計するか、前提不成立を記録 |
| 最小統合後 | 数値とモデル出力が基準実装に合うか | 合わなければ統合を止める |
| 最適化後 | 各 bpw/GPU で速度 95% 以上か | 未達原因を最適化し、再測定 |
| 最終確認後 | 品質・速度・互換性を全対象範囲で満たすか | 達成範囲と `unverified` を確定 |

## 停滞時の調査と証拠

同じ原因で実装または最適化が停滞したら、XeStrata の該当経路だけを見続けず、clone 済み Strata の対応する loader、layout、codebook kernel、MTP/head 処理、テスト、履歴を同じ入力で照合する。upstream の設計をそのまま採用できない場合も、差分の理由を記録する。二つの実装を確認しても原因が特定できない場合は、最小再現を作ってから次へ進む。

測定結果は AGENTS.md の規則に従い data folder の records に保存し、コードには持ち込まない。各記録に「既知」「検証済み」「unverified」「未達」を付け、未最適化 prototype、未対応入力、未測定 hardware を完了の証拠にしない。前提条件を緩めたり、未対応範囲を削ったりして完了扱いにしない。
