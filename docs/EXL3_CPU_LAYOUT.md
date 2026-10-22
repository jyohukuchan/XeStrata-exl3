<!-- SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# EXL3 CPU optimization candidates

Design notes from investigating the CPU path during hybrid CPU/GPU execution. The weight-layout change below is integrated into the normal runtime's CPU-side cache. GPU kernels, the on-disk EXL3 format, and quantization precision stay the same.

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

This difference demonstrates room to improve cache behavior and access order. It does not establish whether pure DRAM bandwidth or arithmetic instructions are the ultimate limit. This table is kept as the prototype's record; the measurement after the integration is in "CPU-only measurement after the integration".

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

## Integration into the normal runtime

The layout is identified by a `TileOrder` enumeration. `canonical` is EXL3's own order, `cpu_block` is the K-major CPU order above, and `cpu_nmajor` is the N-major CPU order. The CPU kernels read all three; the GPU kernels accept `canonical` only and throw otherwise. The value is a property of a host blob: it appears neither in the on-disk format nor in the format descriptor handed to the device.

- It applies to an expert whose three projections are all MUL1 (codebook 2) packed trellis with an output width that is a multiple of 128. CB0/1, raw projections, mixed formats and widths outside that range keep `canonical` and the existing path.
- With the integer dot disabled, and with a non-finite or tiny input that sends the work back to the exact path, the same tiles are read, following that blob's `TileOrder`. Only the order of the tiles differs, so the results stay bit identical.
- The CPU cache holds `cpu_nmajor`. What the GPU receives is the byte sequence in the canonical order, so the device reads the order it always read. The admission of a GPU-resident expert into the cache and a re-layout out of the cache both go through that path.
- The order is a property of the blob and does not change across cache eviction, re-materialization, multiple tokens or multiple stages. A re-materialized expert gets the same order.
- The condition that 128 outputs are the unit of the Hadamard transform is unchanged. A task is never split below 128 outputs. The order of the tiles and the granularity of a task are separate things and are not mixed.
- `STRATA_EXL3_CPU_TILE_ORDER=0` selects `canonical`, `=1` selects `cpu_block`, and `=2` (the default) selects `cpu_nmajor`. Switching the order does not change any result. The older name `STRATA_EXL3_CPU_BLOCK_ORDER` still works when the new variable is not set: 0 is `canonical`, anything else is `cpu_block`.

## Comparing the compressed tile orders

The CPU reads a 128-output block as eight N tiles, and its K walk has the N tile outside and the K tile inside. In the `cpu_block` order the address of a fixed N tile advances by B tiles (1280 bytes) per K tile. The eight N tiles therefore advance at eight different rates: the CPU reads eight interleaved streams and walks the whole block eight times.

`cpu_nmajor` puts the K tiles of one N tile next to each other: the tile (tile_k, tile_n) sits at `tile_n * tiles_k + tile_k`. The K walk is then one sequential run, and the eight N tiles of a block follow one another, so a task reads one contiguous stretch.

The N-major order does not depend on B. `(tile_n / B) * tiles_k * B + (tile_n % B) * tiles_k + tile_k` is `tile_n * tiles_k + tile_k` for every B, because `(tile_n / B) * B` is `tile_n - (tile_n % B)`. The indices for B = 8, 16 and 32 were checked to agree on the measured model's shapes (k=2048, n=512 and k=512, n=2048). B = 16 and B = 32 are another name for the same order as B = 8.

A larger B in the K-major order lengthens the stride of the eight streams (2560 bytes at B = 16, 5120 bytes at B = 32). The measurement found it slower than B = 8 as well.

### Measurement

The same weights (H=2048, FF=512, rows=1, five experts per batch), a sweep of 40 layers of five experts, two warmup passes and 20 measured passes, five repeats. The host was shared with other work. The table gives the time of one 40-layer sweep in microseconds: the median of the five repeats, and their smallest and largest (the spread over the repeats). Only the order changed, on the same binary.

The five orders (with the K-major B=16 and B=32) were compared on a comparison build that produces the same bytes for the three orders the committed code has (0, 1, 2).

| Order | Workers | Median µs | Min µs | Max µs | Against canonical |
| --- | ---: | ---: | ---: | ---: | ---: |
| canonical | 8 | 26732 | 26552 | 27376 | — |
| cpu_block (B=8) | 8 | 23800 | 23691 | 24532 | -11.0% |
| cpu_nmajor | 8 | 18701 | 18597 | 21736 | -30.0% |
| K-major B=16 | 8 | 25143 | 24576 | 41024 | -5.9% |
| K-major B=32 | 8 | 27048 | 26322 | 33395 | +1.2% |
| canonical | 24 | 17694 | 17102 | 23578 | — |
| cpu_block (B=8) | 24 | 14693 | 14119 | 15913 | -17.0% |
| cpu_nmajor | 24 | 11700 | 11429 | 13433 | -33.9% |
| K-major B=16 | 24 | 15724 | 14901 | 16226 | -11.1% |
| K-major B=32 | 24 | 15690 | 15257 | 16664 | -11.3% |
| canonical | 48 | 17010 | 16235 | 18447 | — |
| cpu_block (B=8) | 48 | 14849 | 12973 | 16474 | -12.7% |
| cpu_nmajor | 48 | 12798 | 12610 | 13686 | -24.8% |
| K-major B=16 | 48 | 15256 | 14320 | 16732 | -10.3% |
| K-major B=32 | 48 | 15891 | 15619 | 18073 | -6.6% |

The N-major order is the fastest at every worker count: 21.4% shorter than `cpu_block` with 8 workers, 20.4% with 24 and 13.8% with 48. The spread over the repeats was 6.5% at most (canonical with 24 workers), and 1.1% for `cpu_nmajor` with 48 workers. One of the five repeats ran under a heavy host load, and canonical with 24 workers reached 23578µs. Comparing the best run alone does not count.

The committed build measured the same three orders again (nine repeats at 48 workers). The ranking is the same: 28207, 25212 and 18703µs with 8 workers, 19124, 14664 and 12334µs with 24, and 19451, 15299 and 16104µs with 48. The additional nine repeats at 48 workers give 19846, 15803 and 16071µs: the two CPU orders differ by 1.7%, inside the spread.

The exact path, with the integer dot disabled, ranks the same way. Its 40-layer sweep (µs) is 198622, 194196 and 190714 with 8 workers, 98640, 96598 and 92880 with 24, and 65449, 69653 and 67739 with 48. The exact path decodes each tile to F32 and adds in FP64, so arithmetic dominates and the layout matters little.

The measurement was CPU only. The maintainer runs the GPU decode measurements.

## CPU-only measurement after the integration

The same weights (H=2048, FF=512, rows=1, five experts per batch), a sweep of 40 layers of five experts, two warmup passes and 20 measured passes; the value is the median time per layer, over five repeats. The host was shared with other work.

| Condition | Workers | Canonical µs | CPU layout µs | Difference |
| --- | ---: | ---: | ---: | ---: |
| Integer dot on | 24 | 335.1 | 244.2 | -27.1% |
| Integer dot on | 48 | 320.6 | 256.4 | -20.0% |
| Integer dot off (exact path) | 24 | 2360.5 | 2301.8 | -2.5% |
| Integer dot off (exact path) | 48 | 1667.7 | 1467.3 | -12.0% |

The 48-worker measurements carry the host's noise. Nine repeats of the exact path with 48 workers give 1310.9 to 1754.0µs for canonical and 1263.8 to 1600.1µs for the CPU order; comparing the best cases only, the difference is 3.6%, close to the 2.5% of 24 workers. The integer dot with 48 workers falls from 298.4µs to 220.4µs at the best cases, likewise about 26%.

The layout change helps the integer dot path much and the exact path little. The exact path decodes each tile to F32 and adds in FP64, so arithmetic rather than memory layout dominates there. Materializing 200 experts (355MB) took 336.1ms in canonical order and 349.6ms in the CPU order: the reordering costs about 68µs per expert (about 4%).

The tile orders were compared afterwards, and the N-major order became the default; see "Comparing the compressed tile orders".

The measurement was CPU only. The maintainer runs the GPU decode measurements.

## More tasks by cutting the dot along K

The CPU expert pool's packed EXL3 row-parallel phase has one task per 128-output block per expert row. When a layer has only a few experts that miss the VRAM tier and fall back to the host, the gate/up phase has misses×8 tasks and the down phase misses×16, so many layers have fewer tasks than the pool's workers, and the workers idle.

The dot was therefore cut along K. The block's dot is split into ranges of K, and each range is a task of its own. Each task leaves its integer partial sums behind in a workspace, and the task that finishes the block's last range adds them up and applies the block's transform once.

The cut applies only when the phase's tasks do not fill the workers. The number of ranges is `floor(workers / tasks)`, which fills one round of them. A phase that already has the workers' worth of tasks is left alone: there the cut does not shorten a round, and the ranges' shorter memory runs cost more than the cut saves.

The partial sums are exact integers and the transform is applied once, after they are added up, so the order of the addition changes but the values do not. On a real checkpoint, 0 of 10240 expert output values differ and the whole model's logits (7946240 bytes) compare bit for bit equal. The teacher KL therefore does not have to be taken again.

CPU only, with the real weights (H 2048, FF 512, 40 layers, the integer dot on), the median of the 40-layer round over 3 repeats. The one-expert-per-layer shape (the starved one) changes as follows.

| workers | before µs | after µs |
| ---: | ---: | ---: |
| 8 | 4009 | 3978 |
| 24 | 3386 | 3016 |
| 32 | 4656 | 3250 |
| 48 | 5227 | 3599 |
| 64 | 4638 | 3342 |

The five-experts-per-layer shape already fills the workers in each phase, so the cut does not apply and the two arms are the same within the noise: +0.8%, +0.7%, +0.8%, -5.5% and +0.0% at 8, 24, 32, 48 and 64 workers.

The whole model's decode (the fixed-token bench, 128 positions, the 8 GiB VRAM limit, the matrix engines off, the integer dot and the row split on) gains 0.8% to 4.0%: 34.58 to 35.04 tok/s at 8 workers, 40.51 to 40.82 at 24, 38.75 to 39.23 at 32, 36.64 to 37.36 at 48 and 33.39 to 34.73 at 64. The curve's shape does not change and its peak is at 24 workers; the fall after that is memory bandwidth, not the task count.

The cut does not go below the 128-output block: 128 outputs form one Hadamard transform, so the block's dot would have to be assembled before the transform. The per-task bookkeeping is about 0.2µs, and forcing the cut where it does not pay made the gate/up phase 40% and the down phase 87% slower. It is therefore limited to the conditions where it pays.

The measurement environment and the model's location are in the repository's `LOCAL.md`.

## The CPU kernels in AVX-512

The EXL3 CPU kernels have two computation paths. The integer dot (codebook 2, MUL1) adds 16 outputs at a time in integers; the exact path (FP64) decodes a tile to F32 and adds in FP64. Both have an AVX-512 copy now, and a CPU that has AVX-512 walks them at 512 bits. The AVX2 kernels stay as they are and are the path of a CPU without AVX-512.

### The integer dot in AVX-512

The AVX2 kernel decodes row by row, eight columns at a time, into two 256-bit registers. The AVX-512 kernel decodes a row's sixteen columns into one 512-bit register. The tile's registers are read 256 bits at a time, as before, and a column's dword is picked with the same lane index, so it is the same dword. The funnel shift keeps the AVX2 form: the shift is shared by each eight-column half, and the two halves of the 512-bit vector keep their own shift.

The results are bit identical.

- The 16-bit codes come out of the same dword of the same tile with the same funnel shift.
- The byte sum adds the same four bytes of the same product in the same order, so it is the same integer, and the sum of integers does not depend on the order.
- The activation's rounding is VRNDSCALEPS with the same immediate as the AVX2 kernel's VRNDPS: nearest, ties to even, no exception. The scale q and the clamp to [-127,127] are the same.
- The affine conversion applies the same two operations to the same values. With -ffp-contract=off there is no fusion into an FMA.

### The exact path in AVX-512

The exact path computes the MUL1 codebook's values in arithmetic and adds them in FP64. The AVX-512 copy puts two groups of eight states into one 512-bit register, so the byte sum and the fused multiply-add cover sixteen states at a time. The conversion to F16 stays at 256 bits: the 512-bit form needs AVX512-FP16, which Zen 4 does not have. The FP64 addition is done at 512 bits, and the order of the addition of each column does not change.

### The choice at run time

The choice is made from the CPU's feature bits. `cpu_avx512_ok` reads AVX512F/BW/VL/VNNI/VBMI from CPUID and XCR0, and STRATA_FORCE_AVX2=1 drops the CPU to AVX2. It is never made from a vendor string or a model name. The integer dot's AVX-512 copy needs neither F16C nor FMA, so AVX512F/BW/VL are enough. The exact path's AVX-512 copy uses F16C's 256-bit conversion, so the F16C and FMA bits `cpu_avx2_ok` reads are asked for as well. A named kernel the CPU cannot run is refused, not attempted: the instruction would fault.

### What was checked on this host

The host's CPU has no AVX-512, and Intel SDE is not on this host either. The AVX-512 paths were not run. The following was checked instead.

- A model of the AVX-512 algorithm written in scalar code agrees with the AVX2 implementation bit for bit, for bit widths 1 to 8, over the decode, the byte sum, the addition and the affine conversion.
- The 512-bit byte sum and the 256-bit one give the same value per column. The pairing of VPMADDUBSW is inside each 64-bit granule and the pairing of VPMADDWD inside each 128-bit lane, so the four bytes of a column stay in one column at either width.
- The group pairs the exact path's AVX-512 copy reads (8p+u, 8p+4+u) sit where the AVX2 copy's m[v]=4v+u puts them.
- The rounding means what VRNDPS means. The definition of VRNDSCALEPS in the Intel SDM takes the rounding mode from imm8[1:0] when imm8[2] is 0, has no scaling at imm8[7:4]=0, and imm8[1:0]=00 is nearest even.

Running the AVX-512 paths on an AVX-512 CPU, their bit identity with the AVX2 kernels there, and running them under Intel SDE were not done on this host.

## Capacity checks and worker settings

The source currently checks available RAM and cgroup capacity every layer. An independent diagnostic of 1020 calls measured about 110µs per call: about 87µs for cgroup collection and 20µs for reading and parsing `MemAvailable`. Across 40 layers this is about 4.4ms of call cost, not a claim that all of it can be removed from decode.

One candidate is reducing repeated file-open costs. Read limits and usage afresh from reused handles rather than reusing stale values. Runtime integration must account for cgroup moves, mount or inode changes, read failures, handle bounds and disposal, and Windows behavior. Simply omitting checks is not accepted.

A diagnostic prototype that rereads values while reusing open files reduced call time from about 110 to 54µs. An artificial fixture confirmed that changed capacity limits, usage, and `MemAvailable` were observed. The difference in call cost is about 2.3ms across 40 layers. Measure its effect on full decode separately; do not simply add it to the layout improvement.

Task construction and the time before and after parking are small, but publish-to-completion includes both computation and shared-counter costs. This does not rule out synchronization improvements. Also, 128 outputs form one Hadamard transform, so merely splitting tasks into 64 outputs requires another synchronization before the transform.

Do not add a fixed recommended worker count. Measure it alongside layout, expert counts, and cache state. Match host-thread affinity between the standalone harness and normal serving when making that choice.
