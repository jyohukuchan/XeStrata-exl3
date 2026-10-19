<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# EXL3 development loading path

Whole-model coverage, quality at matched effective bpw, and optimized decode speed follow the
[EXL3 support plan](EXL3_PLAN.md). The components below load the format and check numerical results;
they do not establish whole-model EXL3 support.

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
copying the blob to a different address. This factory does not include raw experts.

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

`ExpertView` keeps gate/up/down as three independent matrices. CPU and SYCL expert operations compute
`down(silu(gate(x)) * up(x))` without applying a router weight. Connecting the expert cache and pool is
separate from these operations.

The dedicated n-gram decoder treats its leading F16 row scale and 160-value ring separately from ordinary
linears. CPU can select requested row IDs; SYCL processes packed rows already fetched by the caller.
Head bias uses explicit head IDs. Outputs are FP32 with padded strides; callers separately choose any final
F16 rounding required by model arithmetic. This row codec is separate from model-runtime sidecar I/O and caches.
The dedicated blocking `exl3_ngram::Reader` validates sidecar shards and head ranges and reads only
requested rows. It also computes row IDs using the sidecar's multipliers, signed-int64 overflow and EOS boundaries.

With NumPy available and tests enabled, build and run the CTests backed by independent fixtures.

```sh
cmake --build build/free --target exl3_artifact_test exl3_cpu_parity exl3_parity \
    exl3_cpu_expert_parity exl3_expert_parity exl3_cpu_ngram_parity exl3_ngram_parity
ctest --test-dir build/free -R '^exl3_' --output-on-failure
```

A checkpoint probe checks the archive, CPU/GPU reconstruction and linear operations, and the engine's
weight owner together for one requested matrix. Only diagnostic reconstruction expands that selected matrix.

```sh
cmake --build build/free --target exl3_archive_parity
build/free/exl3_archive_parity /path/to/exl3-model model.layers.0.mlp.experts.0.down_proj
```

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

Numerical and functional checks of this path are separate from quality comparisons at the same source
weights and effective storage. Speeds from components and early integration before optimization are
diagnostic and do not decide whether decode speed can be maintained.
