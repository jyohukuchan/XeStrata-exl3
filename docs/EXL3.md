<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# EXL3 development loading path

Whole-model coverage, quality at matched effective bpw, and optimized decode speed follow the
[EXL3 support plan](EXL3_PLAN.md). The following describes the development loading path and numerical checks. Single-GPU AR/MTP paths are
connected for Qwen3.5 MoE; coverage and performance of all target models and prefill proceed in stages.

## Artifact

`strata::artifact::exl3::Archive` reads a model directory's config, safetensors index and headers, checking
shards, tensor shapes, dtypes, bounds and codebook markers. It exposes `metadata.total_size` as advisory
and counts the actual tensor payload separately. The quantization configuration JSON text is retained,
but execution bit widths and codebooks come from the trellis tensors and markers.

`linears()` returns an independent descriptor for each matrix. It handles 2–5 bits plus 1/6/7/8-bit head
exceptions, cb0, mcg, mul1, F16 scales, legacy sign bitfields and F16/F32 biases. `materialize()` reads only
the selected matrix's compressed components without producing an expanded weight matrix. Unquantized
tensors retain their raw descriptors.
`find_linear()` looks up descriptors by name. The expert factory packs gate/up/down into one blob,
preserving each component's offset, bit width and codebook. The same layout recreates views after
copying the blob to a different address. Each projection explicitly distinguishes packed trellis from raw
storage. Raw weights and biases retain their F16/BF16/F32 dtype and bits; raw storage is never represented
as a 16-bit trellis. Simultaneous packed and raw representations of the same name are rejected as ambiguous.

The raw materializer applies explicit per-axis permutations and subsets. Identity value transforms preserve
original F16/BF16/F32 bits. Norm `+1` and GDN `A_log` `-exp` transformations produce F32 only when explicitly
requested. Widening for kernels that require F32 weights is a separate operation; shapes and payload lengths
are validated.

Safetensors outside the index are enumerated as well. `exl3_ngram_trellis` is kept separate from ordinary
linears. The known `mtp_hyper_connection_mixer_patch.safetensors` is applied after the base, with its
source file and override recorded in a sidecar descriptor. Other sidecars that overwrite a tensor name
are rejected.

The Python header inventory runs without loading weights.

```sh
python3 tools/exl3_inventory.py /path/to/exl3-model > /path/to/inventory.json
```

## Operations and parity

CPU and SYCL share `MatrixView`. SYCL FP32 GEMV/GEMM decode packed trellis weights inside the dot product,
applying input/output H128, scales and bias. Batched GEMM accepts padded row strides and explicit event
dependencies, with caller-owned scratch. CPU linear operations also avoid expanding the full matrix.
The CPU initializes decoded-state tables for all three codebooks once. Linear operations read F32 values preserving
the F16 rounding boundary into a temporary 16×16 tile. Bit-width specialization avoids runtime division during state
extraction while keeping the existing FP64 accumulation order.
On AVX2 CPUs, mul1 packed tiles use register permutations and a small decode table.
`STRATA_NO_EXL3_AVX2=1` restores the generic path.

The GPU side codebook 2 (mul1) decode returns (887 * B - 452608) * 2^-17 for a byte sum B.
The value is a multiple of 2^-17 with an absolute value below 2^19, so F32 represents it exactly.
The GPU uses two forms that return the same number.  The portable form computes the exact value in F32 and
rounds it once to F16.  The half form is the reference decoder's fused multiply add in F16,
hfma16(F16(1024 + B), 0x1EEE, 0xC931).  Both round the same exact value once to F16, so they agree bit for
bit.  A device that reports FP16 arithmetic runs the half form; a device that does not runs the portable
form.  `STRATA_EXL3_CB2_HALF_FMA=0` forces the portable form.
The F16 RNE boundary, the FP32 accumulation order, the Hadamard transforms, the bias, the weight storage
format and the bpw are the same in both forms.

`STRATA_EXL3_CPU_ROWPARALLEL=1` processes CPU misses through four existing-pool phases: Hpre,
gate/up output blocks, SiLU plus down Hpre, and down output blocks. Raw projections and batches beyond
prepared capacity use whole-expert execution. Worker count remains configurable; check scaling on the actual model.
`STRATA_EXL3_CPU_INTDOT=1` enables experimental mul1 integer accumulation. It quantizes the transformed input to Q8
and corrects the Gaussian codebook affine constant, but omits the final codebook F16 rounding, so its precision
contract differs from the default exact path. CB0/1, raw projections, unsupported CPUs, nonfinite/tiny inputs and
K dimensions that could overflow the integer accumulator retain the exact path. Neither path expands the full
weight matrix or increases stored bpw. Adoption requires actual-model KL and cooperative CPU/GPU decode measurements
with these settings recorded.

`ExpertView` keeps gate/up/down as three independent packed or raw projections. CPU and SYCL expert operations compute
`down(silu(gate(x)) * up(x))` without applying a router weight.

`Exl3ExpertSource` accepts an explicit name table per layer/expert and loads compressed blobs only for
requested experts. Blob pointers stay stable throughout a layer/window, and `copy_blob()` can also copy
compressed components from another layer. A persistent LRU cache reuses compressed blobs across layers without
retaining expanded weights. Its capacity follows compressed payload size and available host memory, checked again
at each pass over the layers and whenever a blob would grow the resident set past the capacity last checked.
`STRATA_EXL3_SOURCE_CACHE_MIB` adds an upper bound; zero restores temporary window loading.
All requested experts are pinned before eviction, with window-only storage when capacity is insufficient.
Cold GPU-preload `copy_blob()` calls do not populate the CPU cache. Final source statistics report cache hits/misses,
materialization counts/times and payload bytes. Payload bytes count successfully materialized tensor spans, not
physical disk I/O. After GPU cache and workspace allocation, the runtime preloads compressed CPU-owned experts
within its layer range only if all of them plus existing cached data fit the current RAM budget. Otherwise it keeps
the lazy cache without eviction. GPU residents and other stages' layers are excluded; decreasing memory room also
stops an in-progress preload. `STRATA_EXL3_SOURCE_PREWARM=0` disables startup preloading while retaining the LRU cache.
`ExpertPool::prepare_exl3()` preallocates worker scratch, and
`run_exl3()` computes experts from FP32 inputs. CPU dispatch preserves routed output rows and rejects
combinations with legacy GPU plans or residency tables.

The format registry compares every component of the actual blob layout and returns process-local IDs.
`ExpertCache::open_buckets()` preserves each format's capacity, rejecting admission to a different format's
slot and copies exceeding slot capacity. Existing `open()` / `open_sized()` keep using format 0.
`exl3_grouped_fp32()` reads a device format table and dynamic group/token/destination plan, computing FP32
outputs while preserving each expert's independent gate/up/down formats. Scratch is allocated in advance.
`Exl3ExpertRuntime` preloads explicitly selected resident experts into format-specific cache buckets, classifies
hits before source I/O, and places CPU-miss and GPU-hit outputs in router order. Its persistent plan can be
recorded in a graph. `STRATA_EXL3_TIMELINE=1` emits JSON lines on standard error for the start and end of host
prepare, GPU submit and CPU pool intervals in steady-clock nanoseconds. GPU submit measures host submission time,
not GPU kernel execution. Correlate it with a GPU profiler timeline to verify execution overlap.
`STRATA_EXL3_PREPARE_DETAIL=1` aggregates prepare partition, source begin, borrowed views, host plan, and
task counts and waiting times for the four CPU phases on each calling host thread. At thread exit it emits an
`exl3_prepare_summary` JSON record on standard error. RAM guard time is nested within source begin and excludes
startup prewarm. CPU phases cover accepted row-parallel batches only; fallback counts are not measured.
See [compressed tile preload](EXL3_WORD_TILE.md) for the optional path sharing Dense projection words.
`ExpertDispatch::exl3_runtime` and `MoeWindowTier::exl3_runtime` connect it to the existing
CPU-row/GPU-hit merge. The current path uses one group and rejects two groups. Segmented graphs are the default.
In the segmented path, a `publish_window_plan` callback enqueues the hit-only grouped kernel on the
same in-order queue before the host computes CPU misses. After the CPU returns, the next segment's existing merge
graph combines CPU rows and GPU hit rows, so it does not depend on mapped-flag visibility or a device-side spin.
Each ring's merge segment and the final segment are synchronized before reusing the plan/cache/scratch. The existing
CPU-first `enqueue()` contract and all-resident direct path remain unchanged.
`STRATA_EXL3_DEVICE_HIT_PLAN=1` uses a monolithic graph with a device-built hit plan when an in-order queue passes
the runtime doorbell-visibility probe. CPU misses have no GPU output destination. The host publishes CPU results
and a compact hit list without enqueuing GPU work. Observing the next ring's doorbell permits reuse of the previous
ring's host state; the final stream synchronization remains. CPU failures still answer the remaining rings with
empty plans and zero CPU rows, drain the graph, and then return an error. A failed probe or
`STRATA_VERIFY_SEGMENTED=1` selects the segmented path. GPU cache residency remains immutable after startup.
Qwen3.5 MoE startup uses the path below. Prefill remains separate.

The dedicated n-gram decoder treats its leading F16 row scale and 160-value ring separately from ordinary
linears. CPU can select requested row IDs; SYCL processes packed rows already fetched by the caller.
Head bias uses explicit head IDs. Outputs are FP32 with padded strides; callers separately choose any final
F16 rounding required by model arithmetic. The following reader and table connect this row codec to asynchronous I/O and caching.
The dedicated blocking `exl3_ngram::Reader` validates sidecar shards and head ranges and reads only
requested rows. It also computes row IDs using the sidecar's multipliers, signed-int64 overflow and EOS boundaries.
`row_spans()` exposes validated mappings from logical rows to file offsets. Passing them to the existing
`PleReader::open_spans()` handles tensors with reordered physical spans or gaps through one asynchronous
I/O worker and row cache. Model-startup sidecar selection and binding remain separate.
`PleTable::open_exl3()` connects this reader and decoder to the existing issue/collect, batch and prefetch
APIs, applying per-head bias and final F16 rounding. It currently handles direct I/O and explicitly rejects
mmap and locking. Sidecar hash constants are exposed for model binding.

With NumPy available and tests enabled, build and run the CTests backed by independent fixtures.

```sh
cmake --build build/free --target exl3_artifact_test exl3_cpu_parity exl3_parity \
    exl3_cpu_expert_parity exl3_expert_parity exl3_cpu_ngram_parity exl3_ngram_parity \
    exl3_cb2_parity \
    exl3_pool_test exl3_expert_source_test exl3_expert_formats_test expert_cache_buckets_test
ctest --test-dir build/free -R '^exl3_' --output-on-failure
```

`exl3_cb2_parity` compares the two forms of the codebook 2 decode on the device.
For every byte sum (0 to 1020) and every 16-bit state (0 to 65535) it checks that the half form and the
portable form return the same F16 bits, and that both agree with the value computed from the definition.
It also checks that a codebook 2 projection returns the same bits with either form.

A checkpoint probe checks the archive, CPU/GPU reconstruction and linear operations, and the engine's
weight owner together for one requested matrix. Only diagnostic reconstruction expands that selected matrix.

```sh
cmake --build build/free --target exl3_archive_parity
build/free/exl3_archive_parity /path/to/exl3-model model.layers.0.mlp.experts.0.down_proj
```

## Qwen3.5 MoE AR integration

`--exl3 DIR --metadata-gguf GGUF` loads `PlainModel` embeddings, GDN, attention, routers, shared/routed
experts, norms and output head from the EXL3 archive. GGUF bootstraps model geometry and tokenizer metadata;
it never supplies fallback weights. A metadata-only GGUF containing zero tensors also works. Config and
bootstrap geometry are checked, including auxiliary tensor element counts required by each kernel.

The bootstrap can be built from the EXL3 directory's config/tokenizer without original BF16 weights.
In an environment with the converter's Python dependencies, first create a zero-tensor GGUF, then validate it
against the config with the helper. The helper rejects GGUFs containing weights and preserves the original types
and bytes of metadata other than block/nextn counts.

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

Packed projections reorder H128 blocks, moving GDN V-heads to canonical order. Raw projections use explicit
axis maps and retain their original F16/BF16/F32 dtype. Norm `+1`, GDN `A_log` `-exp`, and widening auxiliary
weights required by F32 kernels are separate operations. Expert residency derives from actual blob bytes
and device capacity. Partial residency uses the existing segmented graphs by default. When all `(layer, expert)` pairs
are GPU resident, the graph builds its route plan from device IDs without CPU ring service. The same owner's
plan/scratch is reused sequentially on an in-order queue. `STRATA_EXL3_RESIDENT_GRAPH=0` or
`STRATA_VERIFY_SEGMENTED=1` selects the earlier path for comparison. On devices supporting SG32, grouped
expert H128 transforms use a register FWHT; other devices retain the existing transform.
Gate/up K partitioning derives from work-group, sub-group and local-memory limits. A positive
`STRATA_EXL3_GU_SPLIT_K` sets an upper bound for per-device tuning; `1` selects a single-sub-group
comparison path.

AR uses `--prompt-chunk 0 --spec 0`; `--spec 1` also disables MTP. A positive `--prompt-chunk` runs prefill
in decode windows bounded by `max-t`, returning the last row. Compressed weights remain compressed; this is
a fallback distinct from dedicated batched prefill. MTP uses the integration below. Qwen3.8 projection maps and n-gram codecs/tables exist; its startup binding remains separate.

The development tool `plain_model_logits` feeds `qbench-trace/1` token IDs in order without retokenizing,
saving FP32 logits and a manifest for selected rows. Choose `--exl3 DIR --metadata-gguf GGUF` or `--native GGUF`,
and supply `--trace TRACE.json --out-dir DIR`. `--rows 0,4` limits rows; native `--expert-profile FILE` supports
checks with matched residency. Quality acceptance uses only `KL(P_BF16 || P_candidate)` against the original
BF16 teacher. The diagnostic dump's runtime is not a decode speed acceptance measurement.

## Qwen3.8 Flash AR integration

Qwen3.8 `--exl3 DIR` starts directly from the archive's HF config without requiring a metadata GGUF.
An optional `--metadata-gguf` supplies EOS metadata. Pass input IDs with `--tokens` or `--tokens-file`,
and choose layer-stage device ordinals with `--stage-devices`. `--stage-layers` names the layer
count per stage (for example `18,18,12`); the default is the even split. `--pool-workers N` sets CPU workers per stage.
The default zero derives the count from physical cores; measure the best value on the actual model.

```sh
strata --exl3 /path/to/qwen38-exl3 --tokens 16,10,16,28 --max-new 2 \
    --context 8 --stage-devices 0 --spec 0 --max-t 1 --prompt-chunk 0
```

This integration runs greedy single-token AR and uses CPU expert fallback with partial residency.
`--dump-logits FILE` saves generation FP32 logits; `--stop-eos` or `--eos ID` controls stopping.
This initial CLI rejects requests for MTP, vision, serve, batch and multi-token prompt chunks.
Qwen3.5 CLI/MTP uses the following path.

### The device cache (Qwen3.8 Flash)

The Qwen3.8 Flash integration keeps its experts in a device cache on the GPU. The cache's size comes from
**the free VRAM the device reports**: the (layer, expert) pairs that fit in the room left after a reserve
(a thirty-second of the card's total VRAM, between 192 and 768 MiB) are resident, and the pairs that do not
fit are read from the source and computed on the CPU as before. A card with less VRAM gets a smaller cache.
When the cache cannot be opened at all the run goes on with the CPU miss path alone, and says so.

```sh
STRATA_EXL3_DEVICE_CACHE=0            # no device cache (every expert on the CPU)
STRATA_EXL3_DEVICE_CACHE_MIB=4096     # the cache's budget in MiB
STRATA_EXL3_DEVICE_CACHE_ORDER=spread # spread | visited | profile
STRATA_EXL3_DEVICE_CACHE_PROFILE=PATH # the profile file the profile order ranks with
```

`spread` (the default) gives every layer the same number of experts. `visited` fills a layer's experts
before the next layer's, in the order the layers are visited. `profile` takes the pairs a profile
(`tools/make_profile.py`'s `STRP` format) ranks first, the hottest ones first.

Measured on the V620 at 0000:43:00.0 with the Qwen3.8 EXL3 3.05 bpw checkpoint, `STRATA_EXL3_CPU_INTDOT=1`
and `STRATA_EXL3_CPU_ROWPARALLEL=1`, context 512, 63 decode tokens, and the engine's own
`STRATA_EXL3_PREPARE_DETAIL=1` counters:

| Condition | CPU misses | decode |
| --- | ---: | ---: |
| no device cache | 32160 / 3216 windows (10.00 per window) | 3.67 tok/s |
| device cache of 15404 experts (26.72 GiB), `spread` | 11801 (3.67 per window) | **12.88 tok/s** |
| the same, `visited` | 11978 (3.73 per window) | 12.25 tok/s |
| the same, `profile` (`data/expert-profile.bin`) | 9429 (2.93 per window) | 12.73 tok/s |

The three orders are within 5% of each other in decode speed, and the order does not follow the miss count:
a layer's CPU misses run in parallel inside the layer, so a layer with one miss costs as much as a layer with
ten. The default is `spread`: it spreads the cache evenly over the layers and needs no profile file.
The cache's start-up costs the read of the experts it holds (34 s on the V620: the read and the device's
order), and the decode is 0.27 s per token faster.

**The values**: an expert in the cache is computed by the GPU with FP32 accumulation, and the CPU's exact
path accumulates in FP64, so the last bits differ. The logits of a run without a device cache are bit
identical with the reference; with the cache the largest absolute difference of a logit is 1.4e-4 (1e-5
relative), and the greedy tokens do not change.

### Several GPUs (Qwen3.8 Flash)

The EXL3 Flash layers are split over the devices named by `--stage-devices`. Without the option the
model makes as many stages as the engine has usable GPUs, and the layers are cut at
`layer_count * stage / stage_count`, so every stage owns at least one layer. `--stage-layers 18,18,12`
names the layer count per stage explicitly (the stage count and the device list still come from
`--stage-devices`). The split is validated against the stage count and the layer total, and a split
where a stage gets no layer or the counts do not cover the model stops the start with the reason.
A last stage with fewer layers leaves more of its card's VRAM to the device cache. Each stage has its
own device, session arena, dense weights, expert runtime and device cache.

**Only the residual crosses a stage boundary.** The previous stage's residual (`hc * n_embd` floats)
goes into the next stage's session in one device-to-device copy: where the runtime copies between
the two devices' memories directly, that path is taken, and where it cannot, the residual goes
through the host bridge buffer as before. Which one it is is decided at start-up with a small copy
that is actually issued and checked, and the path taken is reported. The devices do not share an
address space, so nothing else can be carried from one device to another. The session arena (the
GDN state, the QSA KV cache, the PLE's conv history), the dense weights, the expert runtime, the
device cache and the embedding stay on the device of the stage that owns them.

**The host waits only where data crosses it.** forward() issues the stages' work in order and waits
in four places: (1) each layer's doorbell, where the pre graph publishes the experts' input - the
wait is on the ring, so the CPU's expert work starts without waiting for the graph's end; (2) the
expert merge's device-to-host copy; (3) the residual handover at a stage boundary; (4) the final
logits copy. The residual copy is queued right after the sending stage's last post graph, and the
receiving stage waits that one completion before queueing its first pre graph. Nothing waits after a
post graph, so between one layer's end and the next layer's start only the queue's order remains.

**The PLE does not cross a stage boundary.** The PLE block belongs to layer 1, and the layer split
gives that layer to exactly one stage. That stage (the PLE's home stage) holds the PLE's device
half: the key and value projections' owners, the row the n-gram lookup gathered, the preprojection
destinations, the scratch and the conv history. The rows are read on the home stage and consumed
by the same stage's layer 1 pre graph, so no part of the PLE crosses a boundary. The PLE's token
window (the token being decoded and the two before it) is read on the host, so it belongs to the
model. The home stage comes from the layer split, so however the stage count and the device order
change, exactly one stage holds the PLE's layer and the other stages never touch the PLE.

**A buffer is named after the runtime that made it.** The engine's allocations are recorded with the
runtime that made them (`Runtime::note_allocation`), and `Runtime::owner` reads that record first.
The record holds the size of the block too, so a pointer *inside* an allocation (a buffer carved
out of a session arena, a slot of the expert cache) is named after the same runtime. A context scan
alone cannot tell them apart: device USM answers "device" to `get_pointer_type` for every device,
so the scan names the first runtime, and with a second card the stage's own buffers are taken for
another runtime's. That is why the multi-GPU EXL3 path stopped in the stage 1 dense owner, which
refused buffers that were its own ("input or output belongs to another runtime").

Measured on the V620s at 0000:03:00.0 and 0000:43:00.0 with the Qwen3.8 EXL3 3.05 bpw checkpoint,
48 layers, 24 layers per stage with two stages, `STRATA_VRAM_LIMIT_MIB=8192`, and the two generated
tokens of the prompt 16,10,16,28:

| Condition | Stages | Output |
| --- | ---: | --- |
| no device cache, the CPU's exact path | 1 | logits SHA256 `1fe719ac...` (the single-stage reference) |
| no device cache, the CPU's exact path | 2 | logits SHA256 `1fe719ac...` (**bit identical** to the single stage) |
| device cache, the CPU's integer dot and row-parallel phases | 1 | 1871 experts resident on stage 0 (3.25 GiB), logits SHA256 `e88713bc...` |
| the same | 2 | 3196 experts resident on stage 0 (5.54 GiB) and 3011 on stage 1 (5.22 GiB), logits SHA256 `51e69318...` |

The one-stage and two-stage values with the device cache do not agree: the cache holds a different
set of experts in the two configurations, and a GPU-computed expert accumulates in FP32 where the
CPU's exact path accumulates in FP64. The difference (the largest absolute difference of a logit,
1.06) is of the same order as the one-stage difference between the exact path and the cache
(0.83 and 0.98), and the greedy tokens do not change. The device order does not matter either: with
`--stage-devices 1,0` the logits are bit identical.

A split of three stages (the two V620s and the R9700, the 48 layers in 16/16/16) behaves the same
way: the logits without the device cache are bit identical to the one-stage reference, and with the
device cache the greedy tokens do not change either. The resident experts are 3563 + 3602 + 3355 =
10520 on three cards, against 1848 on one.

**Speed** (the Qwen3.8 Flash EXL3 3.05 bpw checkpoint, 48 layers, the V620s at 0000:03:00.0 and
0000:43:00.0 and the R9700 at 0000:07:00.0, the median of 7 decode tokens from the prompt
16,10,16,28, `STRATA_EXL3_FLASH_TIMING=1`, and no other job using a GPU; tok/s is the reciprocal
of that median):

| Condition | Before the handover rework | Now |
| --- | ---: | ---: |
| 1 GPU (V620, 8 GiB VRAM limit) | 87.9 ms (11.4 tok/s) | 85.1 ms (11.8 tok/s) |
| 1 GPU, no limit | 70.3 ms (14.2 tok/s) | 70.2 ms (14.2 tok/s) |
| 2 GPUs (24/24, 8 GiB limit) | 111.0 ms (9.0 tok/s) | 100.6 ms (9.9 tok/s) |
| 2 GPUs, no limit | 88.6 ms (11.3 tok/s) | 85.6 ms (11.7 tok/s) |
| 3 GPUs (16/16/16, 8 GiB limit) | 105.4 ms (9.5 tok/s) | 97.1 ms (10.3 tok/s) † |
| 3 GPUs, no limit | 83.0 ms (12.0 tok/s) | 80.3 ms (12.5 tok/s) |
| 3 GPUs, 8 GiB limit, `--stage-layers 18,18,12` | — | 101.3 ms (9.9 tok/s) |

† Another process took the CPU during the measurement and stretched two tokens (167 and 183 ms);
this is the median without them. With them the median is 110.2 ms.

**The expert cache's allocation matters too.** `--stage-layers` sets the layer split and
`STRATA_EXL3_DEVICE_CACHE_ORDER` the order the resident experts are admitted in. Three GPUs, the
8 GiB limit, the same measurement:

| Condition | tok/s | Resident experts (stage 0/1/2) |
| --- | ---: | --- |
| even 16,16,16 + `spread` (the default) | 9.76 | 3563 / 3602 / 3355 |
| even + `visited` | 7.48 | 3563 / 3602 / 457 |
| 18,18,12 + `spread` | 9.61 | 3470 / 3513 / 3539 |
| **18,18,12 + `visited`** | **10.75** | 3470 / 3513 / 3539 |

One card (the R9700, the same limit) reaches 11.14 tok/s, so **the best three-card run is at 96
percent of it**. `visited` piles the experts up layer by layer, so unless the stage that carries the
head runs fewer layers its residency falls to 457 and the run is slower. What pays is the room the
smaller layers leave on that card, so that every stage holds about 3500 experts. Cutting the cache
budget (`STRATA_EXL3_DEVICE_CACHE_MIB`) to 6 GiB makes no difference (9.82 tok/s, 10273 resident).

What the phase breakdown shows first is that the `post` wait is gone and only `bridge` (the residual
handover) remains, at 0 to 0.5 ms: `post` falls from 6-8 ms to 0.3-0.7 ms. The pre graph is awaited
at the doorbell ring, so the CPU's expert work runs beside the rest of the graph and the
synchronisation between one layer's end and the next layer's start disappears too. On several GPUs
about 0.2 ms per layer (about 10 ms over 48 layers) and 2.5-4.2 ms for a stage's first pre graph
remain. Without a limit the resident experts hold more of the model on three cards and the CPU's
expert work drops to 4.5 ms (15.7 ms on one), so three GPUs come close to one. With
`--stage-layers 18,18,12` the last stage (the card with the head) runs fewer layers, and its device
cache holds 0-295 experts per layer (0-210 with the even split).

Running on Windows has not been checked.

## Qwen3.5 MoE MTP integration

`--spec 2` or greater loads `mtp.fc`, three norms, and `mtp.layers.0` attention, routers and shared/routed experts
from the archive, connecting them to existing draft/verify and GDN state rollback. The GGUF bootstrap must have
`block_count=trunk+1` and `nextn_predict_layers=1`, and the EXL3 config must also specify one MTP layer.
`ModelMap` adds MTP only when requested; disabled MTP leaves the trunk and head unchanged.
Build its bootstrap by adding `--include-mtp` to `tools/exl3_metadata.py`.

```sh
strata --exl3 /path/to/exl3-model --metadata-gguf /path/to/metadata-with-mtp.gguf \
    --prompt-chunk 0 --spec 2 --max-t 3 --tokens 16,10,16,28 --max-new 8
```

FC and other projections retain packed/raw storage and bit widths. MTP shares the trunk embedding and output head,
using `mtp.norm.weight` for its final norm. Drafts currently compute the full head; requesting `--draft-vocab`
prints a message explaining this behavior. Dedicated MTP embeddings and multiple MTP layers are explicitly rejected.

## Fixed-token decode measurement

`plain_model_bench` reads comma/whitespace-separated token IDs, repeating the same sequence to reach the required
length. It prepares the requested context, runs its last `--warmup` tokens one at a time, then measures `--steps`
AR decode tokens. Each repeat resets the state; context preparation, reset and model loading are excluded from
timing. No logits are copied to the host or written during measurement. Greedy token retrieval and synchronization
are included.

```sh
plain_model_bench --exl3 /path/to/exl3-model --metadata-gguf /path/to/metadata-only.gguf \
    --tokens-file /path/to/token-ids.txt --context 512 --warmup 16 --steps 128 --repeats 5 --max-t 1
```

For native IQuant, replace the source with `--native GGUF --expert-profile FILE`. JSON output includes each repeat's
and aggregate mean/median/p90 latency and token/s, expert hit/miss counts, and resident/slot counts. Match token IDs,
context, max-t, residency and execution environment, then apply the speed gate after optimization and quality checks.

## Early output-head integration

The Flash CLI's `--exl3-head DIR` explicitly maps that archive's `lm_head` to canonical `output.weight`.
Shapes must match: rows are neither silently truncated nor converted to another format. This connection
currently serves single-GPU CLI runs and rejects combinations with a GGUF native head, MTP, serve, batch
or pipeline windows. The trunk and embedding keep using the existing pack/native paths.

`Exl3Dense` uploads and owns the compressed device components. When an unquantized `lm_head.weight`
is present, it retains the original F16/BF16/F32 dtype on upload and uses a dedicated raw linear operation.
Bias shapes are validated before conversion to FP32. Each participating stream's scratch is
prepared before graph capture, and the owner outlives those graphs. `lm_head()` selects EXL3 before
interpreting canonical planes and passes the head mixer's FP32 output directly.
The development `Exl3Dense::load()` overload accepts an explicit canonical row name, allowing multiple
independent owners. The CLI's `--exl3-head` continues to select `output.weight`.
`load_remapped()` permutes input H128 blocks and selects or permutes output blocks, moving trellis words and
scales without requantization. Raw tensors reject nonempty block maps.
`load_detached()` owns a projection without an external canonical table, retaining the same explicit shape validation.

Development layer/verify paths can explicitly attach these owners to routers, attention/GDN projections,
shared experts and hyper-connection reads. Raw F16/BF16/F32 projections preserve their original dtype and accept
FP32 activations. HC norms use separate F32 vectors. The loader must validate model-wide names, axis order and
norm transformations; these components alone do not start a complete EXL3 model.
`Exl3WeightSet` owns dense projections and FP32 auxiliary weights for a selected device and trunk layer range,
registering them in a borrowed `WeightTable`. The head and final HC mixer can be included explicitly. Prepare each
stream before capture; teardown waits for the device and restores only its own registrations. Embeddings, routed
experts, PLE and MTP connect through separate owners.
`read_exl3_model_spec()` reads Qwen3.5/Qwen3.8 geometry directly from the archive's HF config without GGUF
weights or a synthetic pack. It separates trunk and MTP; requesting Qwen3.8 MTP is rejected until
its dedicated loader is connected. `build_qwen38_mtp_map()` validates FC, QSA/indexer, HC and shared/routed
experts as a separate one-layer map, never implicitly adding them to the trunk map.
`Exl3Ple` owns PLE key/value projections in packed/raw storage, applies `+1` to its three norms for FP32 kernels,
and retains the original F16 convolution. Layer/verify paths pass preprojected FP32 keys and values to the existing
PLE post-ops. Loading validates norm, convolution and projection shapes and the PLE config.
`NativeEmbed::load_exl3()` preserves raw F16/BF16/F32 embedding dtypes and gathers rows from either host tokens
or device token IDs. `prepare_stream()` prepares any host-table copy needed by each device before capture.
Owners and embedding tables must outlive their captured graphs.

`Exl3FlashModel` owns single-token Qwen3.8 AR execution. It connects per-device layer ranges, embedding, PLE,
CPU/GPU expert merging, HC, and the head, transferring the residual between stages. Its expert window enqueues
resident hits before the blocking CPU miss calculation and waits for the event afterward. The developer
`exl3_flash_probe` accepts token IDs directly. Requests for MTP, vision, or multi-token windows are currently rejected.

Numerical and functional checks of this path are separate from quality comparisons at the same source
weights and effective storage. Speeds from components and early integration before optimization are
diagnostic and do not decide whether decode speed can be maintained.
