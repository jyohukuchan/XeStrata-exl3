<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# EXL3 development loading path

Whole-model coverage, quality at matched effective bpw, and optimized decode speed follow the
[EXL3 support plan](EXL3_PLAN.md). The following describes the development loading path and numerical checks. A single-GPU AR path is
connected for Qwen3.5 MoE; coverage and performance of all target models, prefill and MTP proceed in stages.

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

`ExpertView` keeps gate/up/down as three independent packed or raw projections. CPU and SYCL expert operations compute
`down(silu(gate(x)) * up(x))` without applying a router weight.

`Exl3ExpertSource` accepts an explicit name table per layer/expert and loads compressed blobs only for
requested experts. Blob pointers stay stable throughout a layer/window, and `copy_blob()` can also copy
compressed components from another layer. `ExpertPool::prepare_exl3()` preallocates worker scratch, and
`run_exl3()` computes experts from FP32 inputs. CPU dispatch preserves routed output rows and rejects
combinations with legacy GPU plans or residency tables.

The format registry compares every component of the actual blob layout and returns process-local IDs.
`ExpertCache::open_buckets()` preserves each format's capacity, rejecting admission to a different format's
slot and copies exceeding slot capacity. Existing `open()` / `open_sized()` keep using format 0.
`exl3_grouped_fp32()` reads a device format table and dynamic group/token/destination plan, computing FP32
outputs while preserving each expert's independent gate/up/down formats. Scratch is allocated in advance.
`Exl3ExpertRuntime` preloads explicitly selected resident experts into format-specific cache buckets, classifies
hits before source I/O, and places CPU-miss and GPU-hit outputs in router order. Its persistent plan can be
recorded in a graph. `ExpertDispatch::exl3_runtime` and `MoeWindowTier::exl3_runtime` connect it to the existing
CPU-row/GPU-hit merge. The current path uses one group and segmented graphs; two groups and monolithic graphs
are rejected. Each ring and the final segment are synchronized before reusing the persistent plan.
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
    exl3_pool_test exl3_expert_source_test exl3_expert_formats_test expert_cache_buckets_test
ctest --test-dir build/free -R '^exl3_' --output-on-failure
```

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

```sh
strata --exl3 /path/to/exl3-model --metadata-gguf /path/to/metadata-only.gguf \
    --prompt-chunk 0 --spec 0 --max-t 1 --tokens 16,10,16,28 --max-new 2
```

Packed projections reorder H128 blocks, moving GDN V-heads to canonical order. Raw projections use explicit
axis maps and retain their original F16/BF16/F32 dtype. Norm `+1`, GDN `A_log` `-exp`, and widening auxiliary
weights required by F32 kernels are separate operations. Expert residency derives from actual blob bytes
and device capacity and connects to the existing segmented graphs.

This CLI path currently uses `--prompt-chunk 0 --spec 0`. Positive prefill chunks and MTP are explicitly
rejected. Qwen3.8 projection maps and n-gram codecs/tables exist; its startup binding remains separate.

The development tool `plain_model_logits` feeds `qbench-trace/1` token IDs in order without retokenizing,
saving FP32 logits and a manifest for selected rows. Choose `--exl3 DIR --metadata-gguf GGUF` or `--native GGUF`,
and supply `--trace TRACE.json --out-dir DIR`. `--rows 0,4` limits rows; native `--expert-profile FILE` supports
checks with matched residency. Quality acceptance uses only `KL(P_BF16 || P_candidate)` against the original
BF16 teacher. The diagnostic dump's runtime is not a decode speed acceptance measurement.

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
`NativeEmbed::load_exl3()` preserves raw F16/BF16/F32 embedding dtypes and gathers rows from either host tokens
or device token IDs. Owner and embedding streams are prepared before capture.

Numerical and functional checks of this path are separate from quality comparisons at the same source
weights and effective storage. Speeds from components and early integration before optimization are
diagnostic and do not decide whether decode speed can be maintained.
