<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# EXL3 quality (KL) measurements

A summary of the records that measured the quality of XeStrata's EXL3 and I-Quant models as `KL(P_BF16 || P_candidate)` against the BF16 source weights.
Every model here is the Qwen3.5 35B instruct (`qwen35-instruct`). The KL of Qwen3.8 Flash has not been measured yet.

## The final result (`kl-final-7a749ef5/`)

Six fixed holdouts, 2729 positions. The teacher is the official BF16 weights; the calibration tokens (32×1024) are the same for both sides.

| 4.5 bpw | EXL3 | Xe I-Quant |
| --- | ---: | ---: |
| mean KL | 0.00897974 | 0.02184257 |
| median | 0.00492543 | 0.01300026 |
| p90 | 0.02167341 | 0.05320558 |
| max | 0.19886886 | 0.33511669 |

The storage is nearly the same (I-Quant is 0.0989% smaller), and the mean KL of EXL3 is 58.9% lower.
The history and the conditions are in the "精度ゲート" (quality gate) section of `EXL3_GATE_7a749ef5.ja.md` (in Japanese).

## The files

- `kl-final-7a749ef5/pair.json`: the final pair (mean, median, p90 and max of both sides, the storage, the binary's SHA256)
- `kl-final-7a749ef5/{exl45-r9700,iq45-r9700}/kl.json`: the KL of every position; `plan.json` and `executed-protocol.json` are the run's conditions
- `qwen35-instruct/*-kl.json`: the KL per optimization, per bpw and under a VRAM limit
- `qwen35-instruct/*-paired.json`: pairs taken in the same session
- `qwen35-instruct/*quality-record.json`: the conversion and evaluation record of each quantization

## What is left out

The logits captures (2.6 GB per side) and the teacher's cache are too large and are not here.
Paths, GPU identifiers and container names are replaced by `<data>`, `<checkout>`, `<gpu-uuid>`, `<container>` and the like.
