<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# XeStrata

[English](README.md) | 日本語

> このブランチのエンジンは EXL3 専用です。GGUF native pack と `--pack` / `--native` は削除しました。
> EXL3 は下の[コマンドライン例](#exl3-の形式開発用)から起動してください。
> 以下の upstream 由来の setup・配布パッケージ・Web UI の案内は、EXL3 向けの導入手順ではありません。

1,250 億パラメーターの AI モデルを、Intel Arc、AMD Radeon、NVIDIA のどれかの GPU 1 枚と普通の PC で動かします。

XeStrata は、ふつうはサーバーで動かす大きな AI モデル
**[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** を、自分の PC で動かすためのソフトウェアです。
GPU は Intel Arc、AMD Radeon（RDNA2 以降）、NVIDIA のどれかを 1 枚、OS は Linux を使います。Ubuntu 26.04 と Fedora 44 には deb と rpm のパッケージがあります。
自由ソフトウェアで、Intel と AMD の GPU なら自由ソフトウェアだけでもビルドして動かせます。

> **目次:** [XeStrata とは](#xestrata-とは) · [必要なもの](#必要なもの) · [モデルの選び方](#モデルの選び方) ·
> [インストール](#インストール) · [使い方](#使い方) · [困ったとき](#困ったとき) · [しくみ](#しくみ) ·
> [開発する人へ](#開発する人へ) · [詳しい説明](docs/DETAILS.ja.md)

## XeStrata とは

CUDA で書いたソフトウェアは NVIDIA の GPU でしか動かず、使う人は 1 社の GPU に縛られます。
XeStrata は、オープンな標準の SYCL で書いた 1 つのコードで、Intel・AMD・NVIDIA のどの GPU でも動くことを目指します。

[Strata](https://github.com/Niko1221/Strata)（NVIDIA の GPU 向け）を、Intel の GPU に移したものです。
GPU の計算を CUDA から SYCL と Level Zero に書き直し、Intel の GPU で動くようにしました。
同じコードを、AMD の GPU 向け（free と contrib-llvm のビルド）と NVIDIA の GPU 向け（contrib-llvm のビルド）にもコンパイルします。
モデル、インストールの流れ、画面、API は Strata とほぼ同じです。

Strata との主な違い:

- **Intel Arc 向け**で、AMD と NVIDIA の GPU でも動きます。
- **GPU は 1 枚が基本**です。同じ PC の 2 枚以上の GPU に層を分けて載せることもできます（[MULTIGPU](docs/MULTIGPU.ja.md)）。
  Intel・AMD・NVIDIA のメーカーの違う GPU も組み合わせられます（例: Intel Arc の Battlemage と NVIDIA の Ada。NVIDIA を含むときは contrib-llvm のビルド）。
- **Linux と Windows** で動きます。Windows は Intel と NVIDIA の GPU で、AMD の GPU は Linux だけです（[Windows](#windows)）。WSL 2 は動くかもしれません（プルリクエスト歓迎）。
- **自由ソフトウェアだけでも動きます**（free のビルド、Debian main に入れられる程度に自由な構成。[ソースから作ります](docs/BUILD.ja.md#deb-と-rpm-のパッケージ)）。
  配っている contrib-llvm の版は、自由ソフトウェアでない Intel の oneMKL と NVIDIA の cuBLAS を、入っていれば使います（[インストール](#インストール)）。

## 必要なもの

| | |
| --- | --- |
| GPU | Intel Arc（A シリーズ、B シリーズ）、AMD Radeon（RDNA2 以降、[AMD の GPU](#amd-の-gpu)）か、NVIDIA の GPU（Pascal 以降。Tensor Core のない Pascal は遅い経路で動きます）。VRAM は 12 GB 以上を勧めます（それより少なくても動きますが、遅くなります）。 |
| CPU | x86-64 で AVX2 があるもの。AVX-512（F、BW、VL、VNNI、VBMI）があれば、CPU の計算に AVX-512 を使います。 |
| RAM | モデルのエキスパートが入るだけ（[モデルの選び方](#モデルの選び方)）。 |
| ディスク | モデルのチェックポイントの大きさだけ。SSD（NVMe）を強く勧めます。 |
| OS | Linux（WSL 2 は動くかもしれません。プルリクエスト歓迎）。パッケージは Ubuntu 26.04 と Fedora 44 向けです。ほかのディストリビューションでは[ソースから入れます](docs/BUILD.ja.md#ソースから入れる)。Windows 10 と 11（x64）は 7z で配ります（[Windows](#windows)）。 |
| BIOS | 単体の GPU では Above 4G Decoding と Re-Size BAR を有効にし、CSM を無効にします。 |

- **行列エンジン**: Intel の XMX、AMD の WMMA（RDNA3 以降）、NVIDIA の Tensor Core を使い、ない GPU では DP4a などの整数の内積の命令で計算します。
  どれを使うかは、GPU が報告する能力で決めます。行列エンジンがないと prefill が遅くなりますが、動きます。
- **確かめた GPU**: 開発と確認は Arc Pro B70（Xe2、32 GB）で行っています。NVIDIA は RTX 4070 と RTX 3070、AMD は RX 9060 XT（RDNA4）で確かめています。
  Arc A380 では計算の正しさだけを確かめ、モデルは動かしていません。Windows は RTX 3050 Laptop（4 GB）で確かめています。
  小さな GPU は、B70 で使えるメモリと XMX を制限して確かめています。
- **CPU 内蔵のグラフィックス**: 起動と prefill までを確かめています。メモリが RAM と共有で、
  XMX のないものが多いので、遅くなります。

### AMD の GPU

AMD の GPU では、prefill の密な行列積をディストリビューションの ROCm の rocBLAS と hipBLASLt で計算します。
そのため、XeStrata が動くのは、ディストリビューションの rocBLAS がコードを持っている GPU だけです。

| GPU | gfx | Ubuntu 26.04 | Fedora 44 |
| --- | --- | :-: | :-: |
| RX 6800 / 6900 | gfx1030 | ○ | ○ |
| RX 6700 | gfx1031 | × | ○ |
| Ryzen の内蔵 GPU（RDNA2） | gfx1035、gfx1036 | × | ○ |
| RX 7900 | gfx1100 | ○ | ○ |
| RX 7800 / 7700 | gfx1101 | ○ | ○ |
| RX 7600 | gfx1102 | × | ○ |
| Ryzen の内蔵 GPU（780M など） | gfx1103 | × | ○ |
| Ryzen AI の内蔵 GPU | gfx1150 | × | ○ |
| Ryzen AI Max の内蔵 GPU | gfx1151 | ○ | ○ |
| RX 9060 | gfx1200 | ○ | ○ |
| RX 9070 | gfx1201 | ○ | ○ |

- **RX 6700 と RX 7600 では Fedora 44 を使ってください。** Ubuntu 26.04 の rocBLAS には、これらの GPU のコードがありません。
- RX 6600（gfx1032）と RX 6500 / 6400（gfx1034）は、どちらのディストリビューションの rocBLAS にもコードがないので動きません。
- gfx1152 と gfx1153（Ryzen AI の一部の内蔵 GPU）は、Fedora 44 の rocBLAS にはコードがありますが、XeStrata が使う intel/llvm 7.1.1 にそのターゲットがないので動きません。
- 自分の GPU の gfx は、`xestrata --check` が GPU の名前の後ろに出します。
- 表は、どちらのディストリビューションも ROCm 7.1 の rocBLAS のものです。確かめたのは RX 9060 XT（gfx1200）だけです。

## モデルの選び方

### EXL3 の形式（開発用）

EXL3 は exllamav3 の量子化形式で、**同じ大きさなら I-Quant より精度が高くなります**。
XeStrata はこの形式をコマンドラインから読み込めます（サーバーの経路にはまだありません）。

- **読み込むもの**: EXL3 の checkpoint（safetensors の重み群）と、形の情報だけを持つ metadata GGUF
- **使えるモデル**: Qwen3.5 系の MoE（AR と MTP）、Qwen3.8 Flash（AR と MTP の serial speculative decoding）
- **精度**: 同じ元モデル、同じ 32×1024 の校正 token、同じ 6 つの holdout で
  `KL(P_BF16 || P_candidate)` を測ると、4.5 bpw の EXL3 は 0.0090、I-Quant の IQ4_XS 混合は 0.0218 で、
  EXL3 のほうが 59% 低い。2〜3 bpw では差がさらに大きく、5 bpw では同じ大きさで 60% 低い
- **速さ**: 32 GB の GPU にエキスパートを全部置ける条件では I-Quant の 98.9%。
  8 GiB か 12 GiB の上限を置いて CPU と GPU で分ける条件では 73〜78%。EXL3 は重みの復号が
  I-Quant より重いため、同じ速さにはなりません

```sh
strata --exl3 /path/to/exl3-model --metadata-gguf /path/to/metadata-only.gguf \
    --prompt-chunk 0 --spec 0 --max-t 1 --tokens 16,10,16,28 --max-new 2
```

Qwen3.8 Flash では `--spec 2` で MTP の serial speculative decoding を使う。
verify に通った draft だけを採用し、通らない window は AR で書き直すので、答えは AR と同じになる（今は AR より遅い）。

```sh
strata --exl3 /path/to/qwen38-exl3 --tokens 16,10,16,28 --max-new 8 \
    --context 16 --stage-devices 0 --spec 2 --max-t 1 --prompt-chunk 0
```

EXL3 の形式、精度の測り方、速さの差の分は [docs/EXL3.ja.md](docs/EXL3.ja.md) にあります。

## インストール

### Podman

ホストに入れるのは Podman と podman-compose だけです。XeStrata は Fedora 44 のコンテナの中に、Releases の rpm から入ります（rootless）。

```bash
git clone https://github.com/MistVVK/XeStrata && cd XeStrata/container
mkdir -p ~/.config/xestrata ~/.local/state/xestrata ~/.local/share/xestrata
podman compose build                         # 最新のリリースと GPU のライブラリを入れたイメージを作ります
podman compose run --rm xestrata --setup     # 質問に答えてモデルを取ってきます
podman compose up -d                         # http://127.0.0.1:8095（止めるときは podman compose down）
```

- **GPU**: Intel はこのままです。AMD は `-f compose.yaml -f compose.amd.yaml`、NVIDIA は `-f compose.yaml -f compose.nvidia.yaml` を
  `podman compose` の後に付けます（`export COMPOSE_FILE=compose.yaml:compose.nvidia.yaml` でも同じです）。
  NVIDIA はホストに NVIDIA Container Toolkit の CDI の設定が要ります（[docs/BUILD.ja.md](docs/BUILD.ja.md#podman)）。
  コンテナの中は cuda13.4 なので、Pascal と Volta（P100、P40、GTX 10 シリーズ、V100 など）は使えません。
- **ホストに要るもの**: GPU のドライバーと、自分が render グループに入っていること（`sudo usermod -aG render $USER`）です。
- **置き場所**: 設定・ログ・モデルのファイルはホストの同じ場所（下のパッケージと同じ）を使うので、パッケージ版とコンテナを入れ替えても、そのまま使えます。
- **ほかのコマンド**: `podman compose run --rm xestrata --bench`、`--calibrate`、`chat` のように、`xestrata` の後に付けるものを並べます。
- **更新**: `podman compose build --pull --no-cache` で最新のリリースを入れ直し、`podman compose up -d` で起動し直します。

### deb と rpm のパッケージ

1. GPU に合うパッケージを下の表から選んで、[GitHub の Releases](https://github.com/MistVVK/XeStrata/releases) から取ってきます（`SHA256SUMS` も一緒に）。
1. `sha256sum -c SHA256SUMS --ignore-missing` で確かめます。
1. 入れます: `sudo apt install ./xestrata-contrib-cuda13.1_*.deb`（Ubuntu）か `sudo dnf install ./xestrata-contrib-cuda13.4-*.rpm`（Fedora）。
1. 端末で `xestrata --setup` を実行します。足りない GPU のライブラリと render グループは、入れるコマンドを表示して止まるので、それを実行してからもう一度 `xestrata --setup` を実行します。

> [!NOTE]
> NVIDIA の GPU を使わない場合は、どちらの CUDA のバージョンを使っても変わりありません。

| GPU | Ubuntu 26.04 | Fedora 44 |
| --- | --- | --- |
| **おすすめ**: Intel、AMD（RDNA2 以降）、NVIDIA（Turing 以降） | `xestrata-contrib-cuda13.1` | `xestrata-contrib-cuda13.4` |
| Intel、AMD（RDNA2 以降）、NVIDIA の Pascal と Volta（P100、P40、GTX 10 シリーズ、V100 など） | `xestrata-contrib-cuda12.4` | なし（[ソースから](docs/BUILD.ja.md#ソースから入れる)） |

- パッケージは一度に 1 つだけ入ります。CUDA 13 の版には Pascal と Volta のコードがなく、これらを扱う NVIDIA のドライバーも 580 の系列（CUDA 12.4 に対応）までです。
- 自由ソフトウェアだけで動かす版（`xestrata-free`、Intel と AMD の GPU）は Releases にはありません。[docs/BUILD.ja.md](docs/BUILD.ja.md#deb-と-rpm-のパッケージ) の手順で作れます（`tools/package/build.sh ubuntu26.04 free` など）。

`xestrata --setup` の質問は、Enter を押せばおすすめの選択になります。

- **モデルと大きさ**: [モデルの選び方](#モデルの選び方)を見てください。
- **文脈の長さ**: 一度に覚えておける文章の量です。GPU に合った長さを勧めます。
- **画像**: 画像も読ませるかどうか。
- **実験的な速度向上用の射影**: 答え方が変わる実験的な機能で、既定ではオフです。使う前に
  [説明](docs/DETAILS.ja.md#実験的な速度向上用の射影実験的既定はオフ)を読んでください。

`xestrata` は、GPU と RAM と CPU を確かめ、モデルをダウンロードして起動します。
ブラウザで `http://127.0.0.1:8095` が開きます。

- **時間**: 初回はダウンロード（60〜110 GB）に時間がかかります。途中で止めても、次は続きから始まります。
  GPU のコードは初めて使うときにドライバーがコンパイルするので、最後にモデルを一度動かしてそれを済ませます。
- **起動中は PC が重くなります**: モデルを起動すると、RAM に 23〜50 GB を読み込みます。初回はとくに時間がかかり、
  1〜3 分ほど PC の反応が遅くなることがあります。ウィンドウを閉じずに待ってください。
- **置き場所**: モデルのファイルは `~/.local/share/xestrata`（`--data-dir` で変えられます）、設定は `~/.config/xestrata`、
  ログは `~/.local/state/xestrata` に置きます。

**次からは** `xestrata` を実行すると、すぐに起動します。止めるときはそのウィンドウを閉じます。

**ログインしている間ずっと動かす**: `systemctl --user enable --now xestrata` で、最後に設定したモデルを起動しておきます
（ブラウザは開きません）。先に `xestrata` で一度設定しておきます。

**更新**: 新しいリリースが出ると、起動したときと、動いている間は 6 時間ごとに調べて、端末とチャットのページの About に表示します（`xestrata --check-update` でも調べられます）。
新しいパッケージを [Releases](https://github.com/MistVVK/XeStrata/releases) から取ってきて、インストールと同じく `sudo apt install ./<ファイル>.deb` か `sudo dnf install ./<ファイル>.rpm` で入れます。
モデルの設定は次の起動で更新します。

**自分の PC に合わせる**: `xestrata --calibrate` は、エンジンの設定のいくつかをこの PC で測り、速いものを残します
（5〜10 分ほど）。

**別のパッケージに替える**: Ubuntu では入れたい方を `sudo apt install` すると入れ替わります。
Fedora では `sudo dnf swap xestrata-contrib-cuda13.4 ./xestrata-free-*.rpm` のように、入っているものと入れたいものを指定します。

**消す**: `xestrata --remove-data` でモデルのファイル・設定・ログを消してから（大きさを見せて確かめます）、パッケージを消します。

```bash
xestrata --remove-data
sudo apt purge xestrata-contrib-cuda13.1     # Ubuntu（入れたパッケージの名前）
sudo dnf remove xestrata-contrib-cuda13.4    # Fedora（同じ）
```

`xestrata --remove-data` をせずにパッケージを消すと、ホームのモデルのファイルと設定は残ります。

ソースから `./setup.sh` で入れる方法は [docs/BUILD.ja.md](docs/BUILD.ja.md#ソースから入れる) にあります。

### Windows

1. [GitHub の Releases](https://github.com/MistVVK/XeStrata/releases) から `XeStrata-<版>-windows-x64.7z` を取ってきて（`SHA256SUMS` も一緒に）、好きなフォルダーに展開します。
1. `xestrata.exe` を起動します。導入（GPU とモデルの選択、ダウンロード）と起動を画面で進めます。端末で使うときは `xestrata-cli.exe`（Linux の `xestrata` と同じ使い方）です。

- **要るもの**: [.NET 10 のランタイム](https://dotnet.microsoft.com/download/dotnet/10.0)と [Visual C++ の再頒布可能パッケージ](https://aka.ms/vs/17/release/vc_redist.x64.exe)、GPU のドライバー。
  Visual C++ の再頒布可能パッケージがなければ、`xestrata.exe` が知らせて止まります。
- **GPU**: Intel と NVIDIA です（NVIDIA は Pascal 以降、cuda12.4 のパッケージと同じコード）。AMD の GPU は Windows ではまだ使えません。
- **置き場所**: 7z のフォルダーには書き込みません。設定は `%APPDATA%\xestrata`、ログなどは `%LOCALAPPDATA%\xestrata`、モデルはデータのフォルダー（既定は `%LOCALAPPDATA%\xestrata\data`）に置きます。
- **更新**: 新しい 7z を別のフォルダーに展開して、そちらの `xestrata.exe` を起動します。設定とモデルはそのまま使います。

## 使い方

- **ブラウザ**: `http://127.0.0.1:8095`（モデルが起動すると自動で開きます）。**Chat**、モデルと GPU・CPU・RAM の様子を
  見る **Monitor**、設定とアドレスを見る **About** があります。
- **端末で会話**: `xestrata chat`
- **アプリやコーディングエージェント**: 「OpenAI 互換」の接続先として、ベース URL **`http://127.0.0.1:8095/v1`** を指定します。
  API キーとモデル名は何でも構いません。Anthropic の API を使うアプリは `http://127.0.0.1:8095/v1/messages` です。
- **思考の深さ**: モデルは答える前に考えます。深さは **Off、Low、Medium、High** から選べます。チャットの画面のメニュー、
  `chat.py` の `/think low`、アプリの「reasoning effort」の設定で変えられます。Off がいちばん速く、難しい質問には High が向きます。
- **画像**: チャットの画面では **Picture** を押します。`chat.py` では `/image <パス>` と入力します。アプリでは添付します。
- **スマートフォンや別の PC から**: `xestrata --host 0.0.0.0 --api-key <秘密の文字列>` を実行し、
  サーバーのウィンドウに出るアドレスを開きます（[詳細](docs/DETAILS.ja.md#ストリーミングと接続)）。

**知っておくと良いこと**: 要求は 1 つずつ処理します（複数を同時に処理するには `"parallel": 2`、[BATCHING](docs/BATCHING.ja.md)）。会話の最初のメッセージはすべて読み込みますが、
そのあとは会話を覚えておき、新しく足された部分だけを読むので、続きの返事はすぐに始まります。

## 困ったとき

**最初の起動で PC が固まった、とても遅くなった**
起動中、とくに初回はよくあります。モデルを RAM に読み込み、GPU に置くエキスパートの量を決めているところです。
ウィンドウを閉じずに待ってください。10 分たっても固まったままなら、PC を再起動し、ほかのプログラム
（とくにブラウザ）を閉じてからやり直してください。何度も起きるなら、小さいモデルを選んでください。

**ダウンロードやインストールの途中で止まった**
`xestrata` をもう一度実行してください。止まったところから続けます。

**GPU が見つからない、使えないと言われた**
次の 3 つを確かめてください。

- Intel の GPU のランタイム（Level Zero のドライバー）、AMD の GPU なら ROCm のライブラリ（HIP、rocBLAS、hipBLASLt）、
  NVIDIA の GPU ならそのドライバー（`nvidia-smi` で見えるか）が入っているか確かめてください。
  NVIDIA の GPU は contrib-llvm の版でだけ使えます。AMD の GPU は、[対応する GPU](#amd-の-gpu)かどうかも確かめてください。
- 自分のユーザーで GPU のデバイス（`/dev/dri/renderD*`、AMD の GPU では `/dev/kfd` も）を開けるか確かめてください（`render` グループ）。
- 単体の GPU が OS から見えないときは、BIOS の Above 4G Decoding と Re-Size BAR を確かめてください。

**XMX が使えないと言われた**
今のコンパイラでは、その GPU の行列エンジンが使えません。setup が出す選択肢から選んでください。
XMX なしでも動き、prefill が遅くなります（約 1.6 倍の時間）。

**起動時に VRAM の確保が拒否された（`alloc_device: ... refused`）**
B70 では、まれに起きることがあります。もう一度起動すれば、たいていは動きます。

**ポート 8095 はすでに使われていると言われた**
XeStrata がすでに動いています。そのウィンドウを探してください。ほかのプログラムが使っているなら、
`xestrata --port 8081` のように別のポートを選べます。

**とても遅く、ディスクのランプが点きっぱなし**
RAM が足りていません。ほかのプログラムを閉じるか、小さいモデルを選んでください。

**答えの途中で「the engine stopped unexpectedly」と出た**
たいていは RAM 不足です（Linux がエンジンを止めます）。同じメッセージをもう一度送れば、エンジンは自動で起動し直します。
何度も起きるなら、ほかのプログラムを閉じるか、小さい大きさを選んでください。

**プロンプトが文脈の長さを超えると言われた**
会話が、選んだ文脈の長さより長くなっています。新しい会話を始めるか、`xestrata --setup` で長い文脈を選び直してください。

**解決しないとき**
[詳しい対処の表](docs/DETAILS.ja.md#困ったとき)を見てください。
端末を操作できる LLM エージェントには、[復旧と原因調査の指示書](docs/DIAGNOSE.ja.md)と症状を渡せます。
ソース版と deb/rpm 版の両方に対応し、復旧できた場合も、詰まった場合も、原因と根拠を報告します。
開発者の対応が必要なら、秘密や個人情報を除いた報告を添えて [issue](https://github.com/MistVVK/XeStrata/issues) を立ててください。
エンジンのログの場所は、実際に使っているモデルの設定の `log` で確認できます。

## しくみ

このようなモデルは、ふつう数百 GB の GPU のメモリを持つサーバーで動かします。XeStrata は、**PC 全体で仕事を分け合う**ことで、
1 枚の GPU に収めています。

- **モデルは 24,576 個の小さな専門家（エキスパート）の集まり**で、1 語を書くのに使うのはそのうち 10 個だけです。
  そのため、すべてを一度に GPU に置く必要はありません。
- **GPU** は、どの語にも必要な計算と、よく使われる数千個のエキスパートを受け持ちます。どれがよく使われるかは、
  使っている間も学び続けます。
- **RAM** にはすべてのエキスパートを置きます。GPU にないエキスパートが要るときは、**CPU** がそれを計算します。
  GPU と同時に計算するので、互いに待ちません。
- **SSD** には大きな表を置き、1 語ごとにその数行だけを読みます。
- **推測してから確かめる**: モデルに組み込まれた小さな助手が次の数語を推測し、大きなモデルがそれをまとめて確かめます。
  合っているものは残し、次の 1 語は自分で書きます。決めるのはいつも大きなモデルなので、答えの品質は変わりません。
- **長い文章は大きな塊で読みます**（一度に最大 32,768 トークン）。

行列積は、Intel の GPU では XMX、AMD の GPU では WMMA（RDNA3 以降）、NVIDIA の GPU では Tensor Core で計算し、
行列エンジンのない GPU では DP4a などの整数の内積の命令を使います。
各部分の詳しい説明は [docs/DETAILS.ja.md](docs/DETAILS.ja.md#しくみ)、Intel の GPU での実装と確かめたことは
[docs/XE.ja.md](docs/XE.ja.md) にあります。

## 開発する人へ

報告やプルリクエストを送る前に、[CONTRIBUTING.ja.md](CONTRIBUTING.ja.md)（型番に依存しないこと、ライセンス、GPL 第 7 条の追加の許可、取り込む変更の条件、報告の仕方）を読んでください。

- **ビルド**: ソースから入れる方法、ビルドの 2 つの方式（free、contrib-llvm）、setup の振る舞いと要るパッケージ、
  deb と rpm のパッケージの作り方は [docs/BUILD.ja.md](docs/BUILD.ja.md) にあります。
  開発に使う道具（リント、Intel SDE、GPU のプロファイラー）は [docs/DEVTOOLS.ja.md](docs/DEVTOOLS.ja.md) にあります。
- **速さの測り方**: `xestrata --bench`、upstream 追従の計測、リリースの表の測り方は [docs/BENCH.ja.md](docs/BENCH.ja.md) にあります。
- **守ること**: 機器に依存しないこと（GPU と CPU が報告する能力で経路を選ぶ）、リント（`tools/lint/run.sh`）、
  テストの実行、ライセンスの表示は [AGENTS.md](AGENTS.md) にまとめてあります。
- **取り込む変更**: 誰が書いたものでも（自分のコミット、プルリクエスト、upstream からの移植）、intel/llvm の free と contrib-llvm の両方式でビルドと試験をしたもので、
  機器への依存ができるだけ少ないもの（CUDA などメーカーの API に頼らない、1 枚のカードで測った値を決め打ちしない）だけを入れます。
  そうでないものはそのままは入れず、参考にして XeStrata のコードとして書き直します（[AGENTS.md](AGENTS.md#what-goes-in)）。
  プルリクエストには、ビルドした方式（free、contrib-llvm）と、動かした GPU と CPU を書き、確かめていないことには `unverified` と付けてください。
- **upstream の取り込み**: Strata 0.1.42 までの変更を、Windows の実装を除いて Xe に移しています。
  速くするための経路は、B70・RTX 4070・RX 9060 XT で測って速くなったものを移します（[docs/XE.ja.md](docs/XE.ja.md#strata-0142-の取り込み)）。
- **README と docs**: 日本語版（README.ja.md、docs/*.ja.md）を先に書き、英語版（README.md、docs/*.md）はその訳です。

## クレジット

- **元のソフトウェア**: Niko1221 と Strata の貢献者による [Strata](https://github.com/Niko1221/Strata)。
- **モデル**:
   - Qwen チームの [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。
   - [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) による圧縮版と、
     [Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)。
   - UkisAI による [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)。
   - [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) による UD-IQ4_XS と UD-Q4_K_XL。

  モデルのファイルには、それぞれのライセンスが適用されます。
- **使っている部品**:
   - [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)（MIT）の一部。
   - [oneMath](https://github.com/uxlfoundation/oneMath)（Apache-2.0）: 密な行列積（contrib-llvm、free では AMD の GPU）。
     CMake が XeStrata の変更の入ったフォーク（[MistVVK/oneMath](https://github.com/MistVVK/oneMath)、同じライセンス）から取得します。
   - [intel/llvm](https://github.com/intel/llvm) の DPC++（Apache-2.0 WITH LLVM-exception）: free と contrib-llvm のコンパイラ。
     XeStrata の修正（`third_party/main/intel-llvm/patches/`、同じライセンス）を当ててビルドし、パッケージにはその SYCL の実行時を入れます。
   - AMD による [ROCm のデバイスライブラリ](https://github.com/ROCm/llvm-project/tree/rocm-7.2.4/amd/device-libs)（University of Illinois/NCSA）:
     intel/llvm と一緒にビルドし、AMD の GPU 向けのコードがリンクします。
     そのビルドに渡す機能の指定は、Debian の [rocm-llvm](https://salsa.debian.org/rocm-team/rocm-llvm) のパッケージ（Expat）の `debian/rules` から写しました。
   - 画面のフォント [Outfit](https://github.com/Outfitio/Outfit-Fonts)（SIL Open Font License 1.1）。
- **参考にした考え方**:
   - Inco AI と Splash の貢献者による [Splash](https://github.com/incoai/splash)。
   - Neroued による [ninfer](https://github.com/Neroued/ninfer)。
   - syv.ai と HyperQwen の貢献者による [HyperQwen](https://github.com/syv-ai/HyperQwen)。
   - mw00 と Project Maya の貢献者による [Project Maya](https://github.com/mw00/project-maya)（VRAM にないエキスパートを CPU と PCIe に分ける数の決め方）。
   - maxfridbe による [nextsycl](https://github.com/maxfridbe/nextsycl)（Qwen3.5 系のモデルの decode でカーネルをまとめるやり方）。
- 詳しくは [docs/DETAILS.ja.md](docs/DETAILS.ja.md#クレジットとライセンス) にあります。

## ライセンス

XeStrata（著作権は MistVVK と XeStrata の貢献者）は自由ソフトウェアで、全体を
[GNU Lesser General Public License の第 3 版以降](COPYING.LESSER)（`COPYING.LESSER` と `COPYING`）で配布します。

- **メーカーの計算ライブラリと OS のランタイムとのリンク**: GPU や CPU のメーカーが自社の製品向けに出している計算ライブラリとその実行時ライブラリ
  （oneMKL や Intel oneAPI のコンパイラの実行時ライブラリ、cuBLAS や CUDA のランタイムとドライバなど）、
  OS とその標準のコンパイラの C・C++ の実行時ライブラリ（Microsoft の Universal C Runtime や Visual C++ の実行時ライブラリなど）とリンクしたものも配布できます。
  GPL 第 3 版の第 7 条による追加の許可で、文面は [NOTICE](NOTICE) にあります。
- **Strata と ggml から来たコード**: MIT ライセンスの Strata と ggml / llama.cpp から来たコード（`src/` に書き写したカーネルを含む）は、
  MIT のままです。XeStrata が変えていない Strata のファイルは MIT だけで、XeStrata が手を入れたファイルや書き写したコードを含むファイルは、
  XeStrata の変更とファイル全体が LGPL です（ヘッダーは `LGPL-3.0-or-later AND MIT`）。著作権表示と許諾表示は [NOTICE](NOTICE) にあります。
  Debian の rocm-llvm のパッケージ（Expat）から `tools/intel_llvm_build.py` に写した部分も同じです。
- **例外**: `third_party/` の次のものは、元のライセンスのままです。プロジェクトごとのフォルダーに、ライセンスの文書と一緒に置いています。
   - intel/llvm への XeStrata の修正のパッチ（`third_party/main/intel-llvm/`）: Apache-2.0 WITH LLVM-exception。
   - oneMath への XeStrata の変更（フォーク、`third_party/main/oneMath/`）: Apache-2.0。
   - ggml の `ggml-common.h`（`third_party/main/ggml/`、変えていない写し）: MIT。
   - 画面のフォント Outfit（`third_party/main/outfit/`）: SIL Open Font License 1.1。
   - `third_party/nonfree/`: 自由ソフトウェアでないもの。元のモデルのチャットテンプレートと、実験的な速度向上用の射影のベクトルで、
     どちらも Qwen Community License 1.0 です。
- **`third_party/nonfree/` がなくても**、XeStrata はビルドでき、動きます。パッケージには入れていません。
- **パッケージに入れているほかのもの**: intel/llvm の SYCL の実行時（Apache-2.0 WITH LLVM-exception）、llama.cpp の gguf-py（MIT）、
  oneMath（Apache-2.0）を、それぞれのライセンスの文書と一緒に入れています。
  AMD の GPU 向けのコードは ROCm のデバイスライブラリ（NCSA）の一部を含むので、そのライセンスの文書も入れています。
  自由ソフトウェアでないもの（oneMKL、cuBLAS、NVIDIA のドライバー）は入れていません。
- **モデル**: このリポジトリには含みません。モデルのファイルには、それぞれのライセンスが適用されます。
