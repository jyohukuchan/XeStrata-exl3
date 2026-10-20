<!-- SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# EXL3 CPU optimization candidates

Design notes from investigating the CPU path during hybrid CPU/GPU execution. The weight-layout change below is a diagnostic prototype; it is not integrated into the normal runtime. GPU kernels, the on-disk EXL3 format, and quantization precision stay the same.

## Packed weight layout

The CPU integer dot kernel walks K tiles for 16 outputs at a time. Packed tiles currently store K outside N, so this walk has a stride determined by the N dimension. One CPU pool task handles 128 outputs, matching the output Hadamard transform.

The prototype rearranges only CPU expert weights loaded into RAM into blocks of 128 outputs. It preserves the compressed bits inside each 16-by-16 tile, coefficients, and bias. These indices count tiles; multiply by the packed word count per tile afterward.

```text
B = 128 / 16
canonical = k_tile * n_tiles + n_tile
CPU prototype = (n_tile / B) * k_tiles * B + k_tile * B + n_tile % B
```

This narrows the packed-data span each task accesses and reduces the stride of its K walk. It requires neither decoding all weights nor increasing retained weight bytes. Rearranging the packed tiles adds CPU loading work.

## Standalone measurements

See the repository's `LOCAL.md` for the environment and model locations. Measurements used the same real weights, H=2048, FF=512, rows=1, and batches of five experts, with two warmup passes and 20 measured passes. The 200 experts from 40 layers occupy about 355 MB; this sweep is separate from repeatedly using one layer. Values are median time per batch.

| Layout and execution | Workers | One layer repeated µs | 40-layer sweep µs |
| --- | ---: | ---: | ---: |
| Canonical, row-parallel | 48 | 167.064 | 376.899 |
| Canonical, row-parallel | 24 | 193.345 | 338.544 |
| Canonical, row-parallel | 8 | 388.720 | 538.432 |
| CPU prototype, row-parallel | 48 | — | 229.263 |
| CPU prototype, row-parallel | 24 | — | 247.375 |
| Canonical, whole expert | 48 | — | 875.963 |

The layout change reduced CPU48 sweep time by about 39%. All 200 experts' outputs were finite and bit-identical to canonical. Choosing workers using only a repeated single layer can reverse the trend seen in a sweep. Worker counts also need reevaluation after changing layout.

This difference demonstrates room to improve cache behavior and access order. It does not establish whether pure DRAM bandwidth or arithmetic instructions are the ultimate limit.

## Hybrid decode check

Comparison used the same GPU path, an 8GiB VRAM limit, a 4GiB maximum allocation, matrix engines disabled, context 512, warmup 16, maxT 1, and the same 64 positions over two repeats. Throughput is derived from mean time across all 128 positions.

| Layout | Workers | tok/s |
| --- | ---: | ---: |
| canonical | 24 | 33.245 |
| canonical | 32 | 33.531 |
| canonical | 48 | 30.525 |
| CPU prototype | 48 | 37.171 |

The prototype improved throughput by about 11% over the fastest canonical worker setting measured here. Canonical 48 included a tail: mean 32.761ms versus median 31.607ms, below its earlier 32.416tok/s result under the same settings. Do not claim a large gain using only that 48 reference. The prototype's two repeats reached 36.066 and 38.345 tok/s.

All cases had 3687 residents and 7094 hits/13386 misses per repeat. Two short Japanese/Python holdouts produced 34 positions of 248320 logits, all finite and bit-identical before and after the layout change. This check does not replace teacher KL evaluation over all holdouts.

## Requirements for runtime integration

The prototype supports only the CPU CB2 integer dot path and explicitly stops if another CPU computation path is required. This restriction must not enter the normal runtime.

- Identify the CPU layout in metadata to prevent confusing it with canonical.
- Read the correct layout when integer dot is disabled or non-finite or tiny inputs require the exact path.
- Preserve existing paths for CB0/1, raw projections, mixed formats, and unsupported shapes.
- Supply canonical data when copying or moving CPU cache entries to the GPU.
- Preserve layout ownership across eviction, rematerialization, multiple tokens, and multiple stages.
- Compare logits and speed in hybrid decode with the same expert placement, beyond standalone CPU equality.

The prototype's standalone speed therefore does not describe an improvement already available in the normal runtime.

## Capacity checks and worker settings

The source currently checks available RAM and cgroup capacity every layer. An independent diagnostic of 1020 calls measured about 110µs per call: about 87µs for cgroup collection and 20µs for reading and parsing `MemAvailable`. Across 40 layers this is about 4.4ms of call cost, not a claim that all of it can be removed from decode.

One candidate is reducing repeated file-open costs. Read limits and usage afresh from reused handles rather than reusing stale values. Runtime integration must account for cgroup moves, mount or inode changes, read failures, handle bounds and disposal, and Windows behavior. Simply omitting checks is not accepted.

A diagnostic prototype that rereads values while reusing open files reduced call time from about 110 to 54µs. An artificial fixture confirmed that changed capacity limits, usage, and `MemAvailable` were observed. The difference in call cost is about 2.3ms across 40 layers. Measure its effect on full decode separately; do not simply add it to the layout improvement.

Task construction and the time before and after parking are small, but publish-to-completion includes both computation and shared-counter costs. This does not rule out synchronization improvements. Also, 128 outputs form one Hadamard transform, so merely splitting tasks into 64 outputs requires another synchronization before the transform.

Do not add a fixed recommended worker count. Measure it alongside layout, expert counts, and cache state. Match host-thread affinity between the standalone harness and normal serving when making that choice.
