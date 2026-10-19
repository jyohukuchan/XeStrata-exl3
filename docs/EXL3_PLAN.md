<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# EXL3 implementation plan

This plan adds EXL3 at 2–5 bpw and mixed precision to XeStrata, with higher quality than the existing format at the same effective bpw and decode speed at least comparable with existing XeStrata after optimization. The scope must not be narrowed during implementation; an unchecked area is recorded as `unverified`.

## Completion conditions

- Load and run 2, 3, 4 and 5 bpw independently through prefill, decode, CPU experts, GPU experts and layer split.
- Verify EXL3 `mul1`, `mcg` and `legacy/default (cb0)` codebooks at the primary 2, 3, 4 and 5 bpw targets. A valid output head at 1–8 or 16 bit that exists in a checkpoint is an exception; bit widths that do not exist in a checkpoint are not requirements.
- Read and execute mixed precision with a bit width per layer, matrix, expert, MTP component and head, including variable-size layout, cache, transfer and reconstruction.
- Explicitly handle tensors that stay unquantized or use another precision: dense layers, embeddings, norms, output head, MTP head and vision components. Do not silently round valid head or MTP bit settings recorded in a checkpoint.
- Compare the same source model, revision, tokenizer, calibration data and sampling conditions.
- Match bpw as an effective value including weights, scales, codebook, trellis, metadata, padding and dense/head/MTP storage.
- Show an improvement over the existing XeStrata format at the same effective bpw using BF16 reference metrics and fixed evaluation prompts.
- Measure decode with the optimized configuration under identical hardware, context, batch, KV and MTP settings. The provisional pass line is 95% or more of existing XeStrata. Do not reject feasibility from an unoptimized prototype alone.
- Set `STRATA_ROUTE_TAIL_SKIP=0` for the primary quantization-quality comparison so every routed expert contributes.
  Record the stock fast setting, which conditionally skips low-ranked nonresident experts, separately and compare
  speed under matching skip policies. Do not attribute changes in skipped work caused by cache placement or
  quantization format directly to better quantization quality.

Safetensors sidecars outside the indexed shards are also part of the artifact. `ngram_embedding.safetensors` with
`exl3_ngram_trellis` is not an ordinary linear trellis: it uses a dedicated n-gram codec with row dimension 160, so it
must stay outside the linear descriptors and be handled by a dedicated decoder. An override such as
`mtp_hyper_connection_mixer_patch.safetensors` must not be silently merged into the base; record that the patch takes
precedence and record the files actually read. For same-effective-bpw comparisons, show indexed linear storage and
sidecar-inclusive artifact storage separately. A value that excludes sidecars cannot support a model-wide bpw or quality
claim.

## Stages

### 0. Pin the baseline and run XeStrata

Record the XeStrata revision and the tag/revision of a cloned Strata reference. While the environment is being built, source and EXL3 metadata inspection, pack inventory and structural comparison with Strata may proceed in parallel. Before starting EXL3 model runtime integration, run the model-free device selftest and then a small generation with an existing format. Establish stable GPU enumeration, VRAM, runtime, CPU/GPU paths, prefill, decode and MTP baselines; if the existing format is not stable, fix that cause first.

Record each GPU's device/backend, driver/runtime, build mode, CPU thread count, model format, effective bpw, context, batch, KV precision, MTP settings, prefill tok/s, decode tok/s, VRAM/RAM and major logged timings.

### 1. Check precision feasibility first

Create primary 2, 3, 4 and 5 bpw EXL3 packs from the same source model, covering `mul1`, `mcg`, `legacy/default (cb0)` and representative mixed precision recipes. Preserve valid checkpoint head bit widths as associated exceptions. Compare BF16, the existing XeStrata format and EXL3 on identical inputs. Keep perplexity or token log-likelihood, greedy token agreement and fixed Japanese, code, long-document and conversation evaluations.

The precision gate is evidence of an improvement direction for a representative configuration at every target bpw, or a bit allocation that can produce it. If there is no improvement, distinguish a kernel limitation from a quantization recipe limitation and revisit the recipe. Do not proceed to large-scale performance work while the precision gate is missing; record evidence if the prerequisite is false.

### 2. Integrate the format and numerics minimally

Implement loading for EXL3 metadata, tensor maps, codebooks, scales, trellis data, alignment and shards, then compare a CPU reference dequantizer with GPU dequantization and multiply-accumulate. Start with a fixed-bit `mul1` dense/expert path and make an early integration point that completes one full model pass in XeStrata.

At the early integration point, test synthetic tensors for every codebook, primary 2–5 bpw boundaries, valid checkpoint head exceptions, different row and expert counts, unquantized tensors, the head and MTP. Record CPU/GPU intermediates, logits and greedy output against the reference. Do not measure speed while numerical differences remain unexplained.

### 3. Extend decode and mixed precision

Move from fixed-bit gate/up/down tensors to a bit width per matrix, layer and expert. Use variable-size layout metadata instead of hiding the problem with maximum-size padding. Process cache eviction, PCIe/host transfers, multi-token prefill, one-token decode and MTP draft/verify from the same metadata.

Check primary 2–5 bpw, all three codebooks, valid checkpoint mixed precision and dense/head/MTP exceptions, including layer split and multiple GPUs. Do not load an unsupported input as another format silently; fail with a specific reason.

### 4. Optimize before judging speed

Keep prototype speed as diagnostic evidence. Break down dequantization, multiply-accumulate, Hadamard/scale work, codebook reads, kernel launch, graph, cache, CPU experts and PCIe/host transfer, comparing each interval with the existing XeStrata path. Do not hide a slowdown by dropping bit widths; remove the measured bottleneck first.

For each bpw and GPU configuration, run the optimized configuration repeatedly under the same conditions. Record median, spread, post-warm-up values, prefill and decode, MTP acceptance and delivered token/s. A result below 95% is a miss with a recorded cause and next optimization; one unoptimized run is not evidence that the premise is false.

### 5. Final verification

Recheck every codebook, primary 2–5 bpw, valid checkpoint head exceptions, representative mixed precision, dense/MTP exceptions, CPU/GPU, prefill/decode/MTP and multiple GPUs. Mark only configurations meeting both quality and speed as achieved; mark unchecked combinations `unverified`. The comparison table must show model-wide and quantized-only effective bpw, quality, decode, prefill, VRAM/RAM, settings and the optimization revision.

## Checkpoints for the premises

| Checkpoint | Question | Decision |
| --- | --- | --- |
| After baseline | Is XeStrata stable and is a matched comparison possible? | Fix the foundation if false |
| After precision tests | Can EXL3 improve quality at the same bpw? | Redesign the recipe or record a failed premise |
| After minimal integration | Do numerics and model output match the reference? | Stop integration until explained |
| After optimization | Does each bpw/GPU reach 95% or more? | Optimize the cause and measure again |
| Final review | Are quality, speed and compatibility met across scope? | Freeze achieved and `unverified` ranges |

## Stalls and evidence

When implementation or optimization stalls on the same cause, inspect the corresponding loader, layout, codebook kernels, MTP/head handling, tests and history in both XeStrata and the cloned Strata instead of looking only at the XeStrata path. Record why an upstream design cannot be reused. If both implementations leave the cause unclear, create a minimal reproducer before proceeding.

Store measurements in the data folder's `records/` as required by AGENTS.md, never in the source tree. Label every record `known`, `verified`, `unverified` or `missed`; an unoptimized prototype, an unsupported input or an unmeasured hardware configuration cannot prove completion. Do not relax the premises or delete an unsupported range to claim completion.
