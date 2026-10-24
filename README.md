<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# XeStrata

English | [日本語](README.ja.md)

> This branch's engine is EXL3-only. GGUF native packs and `--pack` / `--native` have been removed.
> Start EXL3 with the [command-line example](#the-exl3-format-in-development) below.
> The inherited upstream setup, release-package and Web UI instructions below do not install this EXL3 engine.

Run a 125-billion-parameter AI model on one Intel Arc, AMD Radeon or NVIDIA GPU and an ordinary PC.

XeStrata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, a large AI model that
normally runs on a server, on your own PC.
It uses one Intel Arc, AMD Radeon (RDNA2 or later) or NVIDIA GPU and Linux; Ubuntu 26.04 and Fedora 44 have deb and rpm packages.
It is free software, and on an Intel or AMD GPU it also builds and runs with free software only.

> **Contents:** [About XeStrata](#about-xestrata) · [What you need](#what-you-need) ·
> [Choosing a model](#choosing-a-model) · [Install](#install) · [Using it](#using-it) · [When something goes wrong](#when-something-goes-wrong) ·
> [How it works](#how-it-works) · [For developers](#for-developers) · [All the details](docs/DETAILS.md)

## About XeStrata

Software written in CUDA runs only on NVIDIA's GPUs, which ties its users to one maker's hardware.
XeStrata aims to run on Intel, AMD and NVIDIA GPUs alike, from one code base written in SYCL, an open standard.

A port of [Strata](https://github.com/Niko1221/Strata) (for NVIDIA GPUs) to Intel GPUs.
Its GPU code is rewritten from CUDA to SYCL and Level Zero, so that it runs on Intel's GPUs.
The same code is compiled for AMD GPUs (the free and contrib-llvm builds) and NVIDIA GPUs (the contrib-llvm build) as well.
The models, the install, the app and the API are much the same as Strata's.

How it differs from Strata:

- **For Intel Arc,** and it runs on AMD and NVIDIA GPUs too.
- **One GPU, as a rule.** The layers can also be spread over two or more GPUs in the same PC ([MULTIGPU](docs/MULTIGPU.md)).
  The GPUs may be of different makers, Intel, AMD and NVIDIA (for example an Intel Arc Battlemage and an NVIDIA Ada; with NVIDIA, in the contrib-llvm build).
- **Linux and Windows.** On Windows, Intel and NVIDIA GPUs; AMD GPUs on Linux only ([Windows](#windows)). WSL 2 may work (pull requests welcome).
- **Runs with free software only, too** (the free build, free enough for Debian main; [made from the source](docs/BUILD.md#the-deb-and-rpm-packages)). The contrib-llvm packages use
  Intel's oneMKL and NVIDIA's cuBLAS, which are not free software, when they are installed ([Install](#install)).

## What you need

| | |
| --- | --- |
| GPU | Intel Arc (A series, B series), AMD Radeon (RDNA2 or later, [AMD GPUs](#amd-gpus)), or an NVIDIA GPU (Pascal or later; Pascal has no tensor cores and runs the slower paths). 12 GB of VRAM or more is recommended (less works, but slower). |
| CPU | x86-64 with AVX2. With AVX-512 (F, BW, VL, VNNI, VBMI) the CPU's part runs on AVX-512. |
| RAM | Enough for the model's experts ([Choosing a model](#choosing-a-model)). |
| Disk | About the size of the model's checkpoint. An SSD (NVMe) is strongly recommended. |
| OS | Linux (WSL 2 may work; pull requests welcome). The packages are for Ubuntu 26.04 and Fedora 44; on other distributions, [install from the source](docs/BUILD.md#installing-from-the-source). Windows 10 and 11 (x64) get a 7z ([Windows](#windows)). |
| BIOS | For a discrete GPU: Above 4G Decoding and Re-Size BAR on, CSM off. |

- **Matrix engines:** Intel's XMX, AMD's WMMA (RDNA3 or later) and NVIDIA's tensor cores are used; a GPU without
  them computes with integer dot-product instructions such as DP4a. Which one is chosen from what the GPU reports. Without matrix engines, prefill is slower,
  but it runs.
- **GPUs checked:** development and checks are done on an Arc Pro B70 (Xe2, 32 GB). On NVIDIA, an RTX 4070 and an
  RTX 3070 are checked; on AMD, an RX 9060 XT (RDNA4). On an Arc A380 only the arithmetic is checked, without running a model.
  Windows is checked on an RTX 3050 Laptop (4 GB).
  Smaller cards are checked by limiting the memory and the XMX the B70 may use.
- **The processor's own graphics:** checked up to starting and prefill. Its memory is shared with the RAM,
  and most have no XMX, so it is slow.

### AMD GPUs

On AMD GPUs, prefill's dense matrix products run in the distribution's ROCm rocBLAS and hipBLASLt.
So XeStrata runs only on the GPUs the distribution's rocBLAS has code for.

| GPU | gfx | Ubuntu 26.04 | Fedora 44 |
| --- | --- | :-: | :-: |
| RX 6800 / 6900 | gfx1030 | yes | yes |
| RX 6700 | gfx1031 | no | yes |
| Ryzen integrated graphics (RDNA2) | gfx1035, gfx1036 | no | yes |
| RX 7900 | gfx1100 | yes | yes |
| RX 7800 / 7700 | gfx1101 | yes | yes |
| RX 7600 | gfx1102 | no | yes |
| Ryzen integrated graphics (780M and others) | gfx1103 | no | yes |
| Ryzen AI integrated graphics | gfx1150 | no | yes |
| Ryzen AI Max integrated graphics | gfx1151 | yes | yes |
| RX 9060 | gfx1200 | yes | yes |
| RX 9070 | gfx1201 | yes | yes |

- **On an RX 6700 or an RX 7600, use Fedora 44.** Ubuntu 26.04's rocBLAS has no code for these GPUs.
- The RX 6600 (gfx1032) and the RX 6500 / 6400 (gfx1034) do not run: neither distribution's rocBLAS has code for them.
- gfx1152 and gfx1153 (some Ryzen AI integrated graphics) do not run: Fedora 44's rocBLAS has code for them, but the
  intel/llvm 7.1.1 XeStrata builds with has no target for them.
- `xestrata --check` shows your GPU's gfx after its name.
- The table is that of ROCm 7.1's rocBLAS on both distributions. Only the RX 9060 XT (gfx1200) has been checked.

## Choosing a model

### The EXL3 format (in development)

EXL3 is exllamav3's quantization format, and **at the same size it is more accurate than I-Quant**.
XeStrata reads it from the command line (the server path does not have it yet).

- **What it reads**: an EXL3 checkpoint (a set of safetensors weights) and a metadata GGUF that carries
  the shape only
- **The models it covers**: the Qwen3.5 family of MoE models (AR and MTP), and Qwen3.8 Flash
  (AR and serial-speculation MTP)
- **Accuracy**: with the same source model, the same 32x1024 calibration tokens and the same six holdouts,
  `KL(P_BF16 || P_candidate)` is 0.0090 for EXL3 at 4.5 bpw and 0.0218 for I-Quant's mixed IQ4_XS,
  so EXL3 is 59 percent lower. The gap is larger at 2 and 3 bpw, and at 5 bpw it is 60 percent lower at
  the same size
- **Speed**: with every expert resident on a 32 GB GPU it reaches 98.9 percent of I-Quant. With an 8 GiB
  or 12 GiB limit and the CPU and the GPU sharing the work it reaches 73 to 78 percent. EXL3 decodes its
  weights with more work than I-Quant does, so it does not reach the same speed

```sh
strata --exl3 /path/to/exl3-model --metadata-gguf /path/to/metadata-only.gguf \
    --prompt-chunk 0 --spec 0 --max-t 1 --tokens 16,10,16,28 --max-new 2
```

On Qwen3.8 Flash, `--spec 2` runs serial speculative decoding over MTP. Only drafts that pass
verification are kept; a rejected window is rewritten with AR, so the answer matches AR (currently
slower than AR).

```sh
strata --exl3 /path/to/qwen38-exl3 --tokens 16,10,16,28 --max-new 8 \
    --context 16 --stage-devices 0 --spec 2 --max-t 1 --prompt-chunk 0
```

The format, how the accuracy is measured and what the speed difference is are in
[docs/EXL3.md](docs/EXL3.md).

## Install

### Podman

Only Podman and podman-compose go on the host. XeStrata goes into a Fedora 44 container, from the release's rpm (rootless).

```bash
git clone https://github.com/MistVVK/XeStrata && cd XeStrata/container
mkdir -p ~/.config/xestrata ~/.local/state/xestrata ~/.local/share/xestrata
podman compose build                         # the image, with the latest release and the GPU libraries
podman compose run --rm xestrata --setup     # answer the questions; it downloads the model
podman compose up -d                         # http://127.0.0.1:8095 (podman compose down stops it)
```

- **GPU**: Intel's as it is. For AMD add `-f compose.yaml -f compose.amd.yaml` after `podman compose`, for NVIDIA
  `-f compose.yaml -f compose.nvidia.yaml` (or `export COMPOSE_FILE=compose.yaml:compose.nvidia.yaml`).
  NVIDIA needs the NVIDIA Container Toolkit's CDI specification on the host ([docs/BUILD.md](docs/BUILD.md#podman)).
  The container has cuda13.4, so Pascal and Volta (P100, P40, the GTX 10 series, V100) are not used.
- **What the host needs**: the GPU's driver, and your user in the render group (`sudo usermod -aG render $USER`).
- **Where things go**: the settings, logs and model files are the host's own (the same places as the packages below), so the packages and the container can take each other's place and keep them.
- **Other commands**: what follows `xestrata`, as in `podman compose run --rm xestrata --bench`, `--calibrate` or `chat`.
- **Updating**: `podman compose build --pull --no-cache` installs the latest release again, and `podman compose up -d` restarts it.

### The deb and rpm packages

1. Pick the package for your GPU from the table below and download it from [GitHub's Releases](https://github.com/MistVVK/XeStrata/releases) (with `SHA256SUMS`).
1. Check it: `sha256sum -c SHA256SUMS --ignore-missing`.
1. Install it: `sudo apt install ./xestrata-contrib-cuda13.1_*.deb` (Ubuntu) or `sudo dnf install ./xestrata-contrib-cuda13.4-*.rpm` (Fedora).
1. Run `xestrata --setup` in a terminal. If a GPU library or the render group is missing, it prints the command that installs it and stops; run that command, then `xestrata --setup` again.

> [!NOTE]
> Without an NVIDIA GPU, either CUDA version works the same.

| GPU | Ubuntu 26.04 | Fedora 44 |
| --- | --- | --- |
| **Recommended**: Intel, AMD (RDNA2 or later), NVIDIA (Turing or later) | `xestrata-contrib-cuda13.1` | `xestrata-contrib-cuda13.4` |
| Intel, AMD (RDNA2 or later), NVIDIA Pascal and Volta (P100, P40, GTX 10 series, V100 and others) | `xestrata-contrib-cuda12.4` | none ([from the source](docs/BUILD.md#installing-from-the-source)) |

- One package is installed at a time. The CUDA 13 packages have no code for Pascal and Volta, and NVIDIA's drivers for them end with the 580 series (CUDA 12.4 works with it).
- The package with free software only (`xestrata-free`, Intel and AMD GPUs) is not on Releases. Make it as [docs/BUILD.md](docs/BUILD.md#the-deb-and-rpm-packages) describes (`tools/package/build.sh ubuntu26.04 free`, for example).

In `xestrata --setup`'s questions, pressing Enter takes the recommended choice.

- **The model and its size:** see [Choosing a model](#choosing-a-model).
- **The context:** how much text it can keep in mind at once. It suggests one for your GPU.
- **Images:** whether it should also read pictures.
- **Experimental speed projection:** an experimental feature that changes how the model answers, off by default.
  Read [what it does](docs/DETAILS.md#experimental-speed-projection-experimental-off-by-default) before you turn it on.

`xestrata` checks the GPU, the RAM and the CPU, downloads the model and starts it. Your browser opens
`http://127.0.0.1:8095`.

- **Time:** the first time, the download (60-110 GB) takes a while. If you stop it, the next run continues where it
  stopped. The driver compiles the GPU code the first time it is used, so it ends by running the model once to get
  that done.
- **The PC is slow while the model starts:** it reads 23-50 GB into RAM. The first start takes longest, and the PC
  can respond slowly for 1-3 minutes. Don't close the window; wait.
- **Where things go:** the model files in `~/.local/share/xestrata` (`--data-dir` chooses another place), the settings
  in `~/.config/xestrata`, the logs in `~/.local/state/xestrata`.

**Next time**, run `xestrata`: it starts right away. Close its window to stop the model.

**Running it while you are logged in:** `systemctl --user enable --now xestrata` keeps the model set up last running
(no browser opens). Set it up with `xestrata` once first.

**Updating:** when a new release is out, XeStrata says so in its window and on the chat page's About tab (it checks
at the start and every six hours while it runs; `xestrata --check-update` checks now). Download the new package from
[Releases](https://github.com/MistVVK/XeStrata/releases) and install it as before, with `sudo apt install ./<file>.deb`
or `sudo dnf install ./<file>.rpm`. The model's settings are updated at the next start.

**Fitting it to your PC:** `xestrata --calibrate` measures some of the engine's settings on this PC and keeps the
fastest (about 5-10 minutes).

**Changing the package:** on Ubuntu, `sudo apt install` the other one and it replaces the first.
On Fedora, name the installed one and the new one, for example `sudo dnf swap xestrata-contrib-cuda13.4 ./xestrata-free-*.rpm`.

**Removing it:** delete the model files, settings and logs with `xestrata --remove-data` (it shows their sizes and asks),
then remove the packages.

```bash
xestrata --remove-data
sudo apt purge xestrata-contrib-cuda13.1     # Ubuntu (the package you installed)
sudo dnf remove xestrata-contrib-cuda13.4    # Fedora (the same)
```

Removed without `xestrata --remove-data`, the packages leave the model files and settings in your home folder.

Installing from the source with `./setup.sh` is in [docs/BUILD.md](docs/BUILD.md#installing-from-the-source).

### Windows

1. Download `XeStrata-<version>-windows-x64.7z` from [GitHub's Releases](https://github.com/MistVVK/XeStrata/releases) (with `SHA256SUMS`) and unpack it into a folder of your choice.
1. Start `xestrata.exe`. It takes you through the install (the GPU and the model, the download) and the start on screen. On the console, `xestrata-cli.exe` (used as `xestrata` on Linux).

- **What it needs:** the [.NET 10 runtime](https://dotnet.microsoft.com/download/dotnet/10.0), the [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) and the GPU's driver.
  Without the Visual C++ Redistributable, `xestrata.exe` says so and stops.
- **GPUs:** Intel and NVIDIA (NVIDIA from Pascal on, the code of the cuda12.4 package). AMD GPUs do not run on Windows yet.
- **Where things go:** nothing is written into the 7z's folder. The settings go to `%APPDATA%\xestrata`, the logs and the rest to `%LOCALAPPDATA%\xestrata`, the models to the data folder (by default `%LOCALAPPDATA%\xestrata\data`).
- **Updates:** unpack the new 7z into a folder of its own and start its `xestrata.exe`. The settings and the models carry over.

## Using it

- **In the browser:** `http://127.0.0.1:8095` (it opens by itself when the model starts): **Chat**, the **Monitor**
  of the model and the GPU, CPU and RAM, and **About** with the settings and addresses.
- **Chat in the terminal:** `xestrata chat`
- **Apps and coding agents:** add it as an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8095/v1`**. Any API key and model name work. Apps that use Anthropic's API:
  `http://127.0.0.1:8095/v1/messages`.
- **Thinking:** the model thinks before it answers. Choose **Off, Low, Medium or High** in the chat page's menu, with
  `/think low` in `chat.py`, or with your app's "reasoning effort" setting. Off is fastest; High suits hard questions.
- **Pictures:** in the chat page, click **Picture**; in `chat.py`, type `/image <path>`; in apps, attach them.
- **From a phone or another PC:** run `xestrata --host 0.0.0.0 --api-key <secret>` and open the address
  the server window prints ([details](docs/DETAILS.md#streaming-and-connecting)).

**Good to know:** it answers one request at a time (several at once is opt-in: `"parallel": 2`, [BATCHING](docs/BATCHING.md)). The first message of a conversation is read in full; after that
it keeps the conversation and reads only what is new, so follow-ups start right away.

## When something goes wrong

**My PC froze or got very slow the first time it started.**
That is common while it starts, most of all the first time: it is loading the model into RAM and working out how many
experts the GPU holds. Don't close the window; wait. Still frozen after 10 minutes? Restart the PC, close other
programs (browsers above all) and try again. If it keeps happening, pick a smaller model.

**It stopped while downloading or installing.**
Run `xestrata` again. It continues where it stopped.

**It says it cannot find or use the GPU.**
Check these three things:

- Check that Intel's GPU runtime (its Level Zero driver), for an AMD GPU the ROCm libraries (HIP, rocBLAS, hipBLASLt),
  or for an NVIDIA GPU its driver (whether `nvidia-smi` sees it), is installed. NVIDIA GPUs run with the contrib-llvm
  packages only. For an AMD GPU, check also that it is one [XeStrata runs on](#amd-gpus).
- Check that your user can open the GPU's device (`/dev/dri/renderD*`, and `/dev/kfd` for an AMD GPU; the `render`
  group).
- If the OS does not see a discrete GPU, check Above 4G Decoding and Re-Size BAR in the BIOS.

**It says XMX cannot be used.**
The current compiler cannot use that GPU's matrix engines. Choose one of the options setup offers. Without XMX it
still runs; prefill takes about 1.6 times as long.

**A VRAM allocation was refused at start (`alloc_device: ... refused`).**
This happens now and then on the B70. Starting again usually works.

**It says port 8095 is already in use.**
XeStrata is already running: look for its window. If another program uses the port, choose another one, for example
`xestrata --port 8081`.

**It is very slow and the disk light stays on.**
The RAM is not enough. Close other programs, or pick a smaller model.

**An answer stopped with "the engine stopped unexpectedly".**
Usually too little RAM (Linux stops the engine). Send your message again: the engine starts again by itself. If it
keeps happening, close other programs or pick a smaller size.

**It says the prompt exceeds the context.**
The conversation is longer than the context you chose. Start a new conversation, or run `xestrata --setup` and
choose a longer context.

**Still stuck?**
See the [full troubleshooting table](docs/DETAILS.md#troubleshooting).
Give an LLM agent that can use your terminal the [recovery and diagnosis instructions](docs/DIAGNOSE.md) and your symptoms.
They cover both source and deb/rpm installations and ask for the cause and evidence whether recovery succeeds or the agent gets stuck.
If developer help is needed, open an [issue](https://github.com/MistVVK/XeStrata/issues) with the report, after removing secrets and personal information.
Find the engine log's location in the `log` field of the model configuration actually in use.

## How it works

Models like this one normally run on servers with hundreds of gigabytes of GPU memory. XeStrata fits it on one GPU by
**sharing the work across the whole PC**.

- **The model is a team of 24,576 small specialists ("experts"),** and each word needs only 10 of them. So they do not
  all have to be on the GPU at once.
- **The GPU** does the work every word needs, and holds the few thousand experts used most often. It keeps learning
  which ones those are while you use it.
- **The RAM** holds every expert. When a word needs one the GPU does not have, **the CPU** computes it, at the same
  time as the GPU, so neither waits for the other.
- **The SSD** holds a large table, of which only a few rows are read per word.
- **Guess, then check:** a small helper built into the model guesses the next few words, and the big model checks
  them all at once. It keeps the ones that are right and writes the next word itself. The big model always decides,
  so the quality of the answer does not change.
- **Long texts are read in large pieces** (up to 32,768 tokens at a time).

Matrix products run on XMX on Intel GPUs, on WMMA on AMD GPUs (RDNA3 or later) and on tensor cores on NVIDIA GPUs;
a GPU without matrix engines uses integer dot-product instructions such as DP4a.
Every part is explained in [docs/DETAILS.md](docs/DETAILS.md#how-it-works), and how it is done on Intel GPUs and
what has been checked in [docs/XE.md](docs/XE.md).

## For developers

Before sending a report or a pull request, read [CONTRIBUTING.md](CONTRIBUTING.md) (no dependence on the model number, the licenses, the additional
permission under GPL section 7, what goes in, and how to report).

- **Building:** installing from the source, the two build modes (free, contrib-llvm), what setup does and the
  packages it needs, and how the deb and rpm packages are made are in [docs/BUILD.md](docs/BUILD.md).
  The development tools (lints, Intel SDE, GPU profilers): [docs/DEVTOOLS.md](docs/DEVTOOLS.md).
- **Measuring the speed:** `xestrata --bench`, the upstream merges' measurements and the release table are in [docs/BENCH.md](docs/BENCH.md).
- **Rules:** [AGENTS.md](AGENTS.md) covers them: independence from the hardware (code paths chosen from what the GPU
  and CPU report), the lints (`tools/lint/run.sh`), the tests, and the license notices.
- **What goes in:** whoever wrote it (the maintainer's commits, pull requests, ports from upstream), a change goes in only when
  built and tested with intel/llvm in both the free and contrib-llvm modes, and depending on the hardware as little as it can
  (no CUDA or other maker's API relied on, no value measured on one card hard-coded). Others do not go in as they are: they
  are a reference, written again as XeStrata's code ([AGENTS.md](AGENTS.md#what-goes-in)).
  A pull request says which modes it was built in (free, contrib-llvm) and on which GPUs and CPUs it ran,
  and marks what was not checked `unverified`.
- **Upstream integration:** the changes of Strata up to 0.1.42 are carried into Xe, except the Windows
  implementation. Of the paths meant to be faster, those measured faster on the B70, the RTX 4070 or the RX 9060 XT
  are carried ([docs/XE.md](docs/XE.md#integration-through-strata-0142)).
- **The README and docs:** the Japanese ones (README.ja.md, docs/*.ja.md) are written first, and the English ones (README.md, docs/*.md) are their translations.

## Credits

- **The original software:** [Strata](https://github.com/Niko1221/Strata), by Niko1221 and the Strata contributors.
- **The models:**
   - [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), by the Qwen team;
   - compressed versions and the [Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) by
     [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF);
   - [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), by UkisAI;
   - UD-IQ4_XS and UD-Q4_K_XL, by [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).

  Each model's own license applies to its files.
- **Parts it uses:**
   - parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT);
   - [oneMath](https://github.com/uxlfoundation/oneMath) (Apache-2.0): the dense matrix products (the contrib-llvm mode,
     and the free mode for AMD GPUs), which CMake fetches from XeStrata's fork with XeStrata's
     changes ([MistVVK/oneMath](https://github.com/MistVVK/oneMath), under the same license);
   - [intel/llvm](https://github.com/intel/llvm)'s DPC++ (Apache-2.0 WITH LLVM-exception): the free and contrib-llvm
     modes' compiler, built with XeStrata's fixes (`third_party/main/intel-llvm/patches/`, under the same license);
     the packages carry its SYCL runtime;
   - [ROCm's device libraries](https://github.com/ROCm/llvm-project/tree/rocm-7.2.4/amd/device-libs), by AMD
     (University of Illinois/NCSA): built with intel/llvm, and linked into the code for AMD GPUs; the target features
     their build is given are transcribed from the `debian/rules` of Debian's
     [rocm-llvm](https://salsa.debian.org/rocm-team/rocm-llvm) package (Expat);
   - the web app's font, [Outfit](https://github.com/Outfitio/Outfit-Fonts) (SIL Open Font License 1.1).
- **Ideas from:**
   - [Splash](https://github.com/incoai/splash), by Inco AI and the Splash contributors;
   - [ninfer](https://github.com/Neroued/ninfer), by Neroued;
   - [HyperQwen](https://github.com/syv-ai/HyperQwen), by syv.ai and the HyperQwen contributors;
   - [Project Maya](https://github.com/mw00/project-maya), by mw00 and the Project Maya contributors (how many of the
     experts missing from VRAM the CPU computes and how many are read over PCIe);
   - [nextsycl](https://github.com/maxfridbe/nextsycl), by maxfridbe (which kernels the decode of the Qwen3.5 family's
     models merges).
- More in [docs/DETAILS.md](docs/DETAILS.md#credits-and-licenses).

## License

XeStrata (copyright MistVVK and the XeStrata contributors) is free software; the whole of it is under the
[GNU Lesser General Public License, version 3 or later](COPYING.LESSER) (`COPYING.LESSER`, with `COPYING`).

- **Linking with the vendors' math libraries and the system's runtime:** what is linked with a math or compute library
  a GPU or CPU maker publishes for its processors, and with the runtime libraries it needs (oneMKL and the runtime
  libraries of Intel's oneAPI compilers, cuBLAS and the CUDA runtime and driver, for example), or with the C and C++
  runtime libraries of the operating system or of its standard compiler (Microsoft's Universal C Runtime and Visual C++
  runtime libraries, for example), may be distributed as well: an additional permission under section 7 of the GPL,
  version 3, worded in [NOTICE](NOTICE).
- **Code from Strata and ggml:** the code that comes from Strata and from ggml / llama.cpp, both MIT (the kernels
  transcribed into `src/` included), stays under MIT. Strata's files XeStrata has not changed are under MIT alone; in
  a file XeStrata has changed or that carries transcribed code, XeStrata's changes and the file as a whole are under
  the LGPL (its header says `LGPL-3.0-or-later AND MIT`). The copyright and permission notices are in [NOTICE](NOTICE).
  So is what `tools/intel_llvm_build.py` transcribes from Debian's rocm-llvm package (Expat).
- **Exceptions:** these, in `third_party/`, stay under their own licenses, one folder per project with its license text.
   - XeStrata's patches to intel/llvm (`third_party/main/intel-llvm/`): Apache-2.0 WITH LLVM-exception.
   - XeStrata's changes to oneMath (its fork, `third_party/main/oneMath/`): Apache-2.0.
   - ggml's `ggml-common.h` (`third_party/main/ggml/`, an unmodified copy): MIT.
   - The app's font Outfit (`third_party/main/outfit/`): SIL Open Font License 1.1.
   - `third_party/nonfree/`: what is not free software. The original model's chat template and the experimental
     speed projection's vector, both under the Qwen Community License 1.0.
- **Without `third_party/nonfree/`,** XeStrata still builds and runs. The packages leave it out.
- **What else the packages carry:** intel/llvm's SYCL runtime (Apache-2.0 WITH LLVM-exception), llama.cpp's gguf-py
  (MIT) and oneMath (Apache-2.0), each with its license text.
  The code for AMD GPUs carries parts of ROCm's device libraries (NCSA), so their license text is there too.
  Nothing that is not free software (oneMKL, cuBLAS, NVIDIA's driver) is in them.
- **The models** are not part of this repository; each model's own license applies to its files.
