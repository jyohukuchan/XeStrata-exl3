<!-- SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# EXL3 CPU側の改善候補

CPU/GPU併用時のCPU経路を調査した設計メモ。以下の重み配置変更は診断専用の試作であり、通常のruntimeには統合していない。GPUカーネル、ディスク上のEXL3形式、量子化精度は変更しない。

## 圧縮重みの配置

CPUの整数内積は、16出力についてK方向のタイルを走査する。現在の圧縮タイルはK方向を外側、N方向を内側に格納するため、この走査はN幅に応じたstrideになる。CPU poolの1タスクは、出力側Hadamard変換と同じ128出力を担当する。

試作はRAMへ読み込んだCPU expertだけを128出力のブロック単位で並べ替える。16×16タイル内の圧縮ビット列、係数、biasには触れない。以下の添字はいずれもタイル単位で、末尾に1タイルの圧縮word数を掛ける。

```text
B = 128 / 16
canonical = k_tile * n_tiles + n_tile
CPU試作   = (n_tile / B) * k_tiles * B + k_tile * B + n_tile % B
```

1タスクが参照する圧縮データの範囲が狭まり、K方向の走査のstrideも小さくなる。全重みの復号や、重みの保持byte数の増加は必要ない。CPUへの読み込み時に圧縮タイルの並べ替えを行う費用は別途発生する。

## 単体測定

測定環境とモデルの所在はリポジトリの `LOCAL.md` を参照。同じ実重みを使い、H=2048、FF=512、rows=1、5 expertを1バッチとして、2巡のwarmup後に20巡を測った。40層分の200 expertは約355 MBで、1層だけを反復する測定と区別した。数値は1バッチあたりの中央値。

| 配置・処理 | worker数 | 1層反復 µs | 40層巡回 µs |
| --- | ---: | ---: | ---: |
| canonical、row-parallel | 48 | 167.064 | 376.899 |
| canonical、row-parallel | 24 | 193.345 | 338.544 |
| canonical、row-parallel | 8 | 388.720 | 538.432 |
| CPU試作、row-parallel | 48 | — | 229.263 |
| CPU試作、row-parallel | 24 | — | 247.375 |
| canonical、expert単位 | 48 | — | 875.963 |

配置変更はCPU48の40層巡回で約39%の時間短縮だった。200 expert分の出力はcanonicalとビット単位で一致し、すべて有限だった。同じ1層の反復だけでworker数を選ぶと、巡回時と傾向が逆転する。配置の変更後にもworker数を再評価する必要がある。

この差はキャッシュやアクセス順序に改善余地があることを示す。純粋なDRAM帯域と演算命令のどちらが限界かを、この測定だけから断定しない。

## 併用decodeでの確認

同じGPU経路、8GiBのVRAM上限、最大allocation 4GiB、matrix engine無効、context512、warmup16、maxT1、同じ64位置×2repeatで比較した。数値は全128位置の平均時間から求めたtok/s。

| 配置 | worker数 | tok/s |
| --- | ---: | ---: |
| canonical | 24 | 33.245 |
| canonical | 32 | 33.531 |
| canonical | 48 | 30.525 |
| CPU試作 | 48 | 37.171 |

今回測ったcanonicalの最速設定に対して約11%の改善だった。canonical48はmean32.761msに対してmedian31.607msとtailを含み、以前の同条件の32.416tok/sより低かった。この48だけを対照に大きな改善率を主張しない。試作の2repeatは36.066と38.345tok/sだった。

resident数3687、各repeatのhit7094/miss13386は全条件で一致した。日本語・Pythonの短い2holdout、計34位置×248320 logitsがすべて有限で、配置変更前後でビット単位で一致した。この範囲の確認を教師KLの全holdout評価の代わりにはしない。

## 通常経路への統合条件

試作はCPUのCB2整数内積だけを対象にし、別のCPU演算経路が必要になった場合は明示的に停止する。この制約を通常runtimeへ持ち込まない。

- CPU用の配置を識別するmetadataを持ち、canonicalとの混同を防ぐ。
- 整数内積を無効化した場合、非有限値や極小入力でexact経路へ戻る場合にも、正しい配置を読む。
- CB0/1、raw射影、混合形式、対応範囲外の形状は従来経路を維持する。
- CPU cacheからGPUへのコピーや再配置では、GPUが期待するcanonicalを渡す。
- cacheのeviction、再materialize、複数token、複数stageでも配置の所有権を保つ。
- CPU単体の一致だけでなく、同じexpert配分の併用decodeでlogitsと速度を比較する。

このため、試作の単体速度を通常runtimeの改善済み性能として扱わない。

## 容量確認とworker設定

現在のsourceは毎層RAMとcgroupの利用可能量を確認する。独立した1020回の診断では約110µs/回で、うちcgroup取得が約87µs、`MemAvailable`の取得・解析が約20µsだった。40層では約4.4msに相当するが、呼び出し費用の見積もりであり、全量をdecodeから削除できるという意味ではない。

候補はファイルを開き直す費用の削減。制限値・使用量そのものを古いまま使わず、再利用したhandleから毎回読み直す。通常経路へ入れる場合は、cgroup移動、mountやinodeの変化、読み取り失敗、handle数の上限と解放、Windows側の処理を含めて設計する。単純な確認省略は採用しない。

値を毎回再読し、開いたファイルだけを再利用する診断用試作では、約110→54µs/回になった。人工fixtureで容量上限、使用量、`MemAvailable`の変更への追随を確認した。40層の呼び出し費用では約2.3msの差に相当する。decode全体の効果は別に測り、配置変更の改善率と単純に加算しない。

タスクの作成やpark前後の時間は小さいが、publishから完了までの区間には演算と共有counterの費用が両方含まれる。同期の改善余地がないという意味ではない。また、128出力はHadamard変換の単位でもあるので、タスクだけを64出力へ分割すると変換前の新たな同期が必要になる。

worker数は固定の推奨値を追加せず、重み配置、expert数、cacheの状態とともに測る。単体harnessと通常のserveではhost threadのaffinityも揃えて判断する。
