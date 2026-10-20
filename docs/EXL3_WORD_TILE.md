<!-- SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->
# EXL3 compressed tile preload

English | [日本語](EXL3_WORD_TILE.ja.md)

EXL3 GPU decode has an optional path that shares compressed tile words within a subgroup.
It preserves F32 input precision, the decoded weight's F16 RNE boundary, FP32 accumulation order and storage format.
It does not expand the decoded weight matrix or add global scratch.

## Enabling

Set `STRATA_EXL3_WORD_TILE_PRELOAD=1` before starting the process. The default is `0`; an unset flag also uses the existing path.
Only the exact string `1` enables the path. Other values disable it.

```sh
export STRATA_EXL3_WORD_TILE_PRELOAD=1
```

In Windows PowerShell, use:

```powershell
$env:STRATA_EXL3_WORD_TILE_PRELOAD = "1"
```

To disable it again, set the value to `0` and restart the process.
An already captured graph retains its selected kernels, so recreate the session when changing the flag.

## Supported paths

The flag applies to packed Dense projections using codebook 2.
The preload kernel is selected only when the device and matrix geometry satisfy the existing packed path's
subgroup and workgroup requirements. Other device/shape combinations, other codebooks and raw projections use the existing paths.

It preserves circular trellis layouts at bit widths 1–8 and 2-byte alignment.
Hpre, Hpost, SVH, bias and the amount of caller-owned scratch remain unchanged.
See [EXL3](EXL3.md) for general EXL3 settings and supported formats.

## Routed experts

On devices supporting SG32, routed expert packed GU and down matrix products also share compressed words for
codebook 2. This path is enabled by default; `STRATA_EXL3_GROUPED_WORD_TILE=0` restores the previous matrix products.
An unset variable or any value other than `0` enables it. This flag is independent of the Dense flag.

```sh
export STRATA_EXL3_GROUPED_WORD_TILE=0
```

A subgroup handles sixteen output columns and two K lanes, preloading the compressed words of a 16×16 tile
into registers. Bit widths 1–8 have separate specializations, and words are shared within the subgroup
to decode each state. Two accumulators per lane reconstruct the previous four-K-lane FP32 addition order and reduction tree.
The F32 input precision and decoded weight F16 RNE boundary are retained.
GU K partitions whose boundaries are not aligned to 16 elements, other codebooks and raw projections retain the
previous computation. Mixed bit widths and projection storage in the format table select the path per projection.

No additional global scratch or decoded weight cache is allocated. The existing device plan and event dependencies
are retained, as are Hpre, Hpost, bias and group/token/destination validation. The flag is evaluated during graph
capture; recreate the session after changing it.

## Wide output projections

For wide codebook 2 matrices that do not use the existing packed path, an SG32 path shares compressed words and
adjacent products. The lower sixteen lanes add products in ascending K order, retaining the previous scalar FP32
addition order. It supports bit widths 1–8, bias, Hpre/Hpost, multiple rows and leading dimensions without additional
global scratch. Devices without SG32 and other codebooks retain the previous path.

Enabled by default. Set `STRATA_EXL3_WIDE_WORD_TILE=0` to restore the previous scalar path. This is independent of
the Dense preload flag.

## Dense wave scheduling

With `STRATA_EXL3_WORD_TILE_PRELOAD=1`, the packed codebook 2 path selects the number of physical waves from the
independent output tile count and the device's compute unit count and work-group capacity. Each original logical K
partition retains its accumulation, and partials are added in the same order. A wave handling multiple logical
partitions computes their accumulators independently, preserving FP32 addition order. The number of K tiles one wave
reads is bounded to sixteen tiles (256 K elements): a wave that covers the whole K span reads its tiles one after
the other and cannot overlap the loads, which leaves the memory latency exposed when the wave count fills the device
once. The wave count does not change any logical partition's accumulation or the order the partials are added in.

Enabled by default. Set `STRATA_EXL3_DENSE_SCHEDULE=0` to restore the previous wave layout. This does not apply to
the wide scalar path.

## Raw projections

Raw F16/BF16/F32 projections use dtype-specific kernels and unrolled K loads. When independent outputs do not
provide enough parallelism, device capacity and local memory determine a work-group that stages temporary F32
products locally, preserving the original 32-lane addition order without extra global scratch.
On devices supporting FP16, normal
F16 values use native conversion. Zeros, subnormals and nonfinite values retain the software conversion independently
of the FP16 denormal mode. Inputs remain F32, and each lane's accumulation order and subgroup reduction are retained.
Weights are not expanded into an additional copy.

Enabled by default. Set `STRATA_EXL3_RAW_SPECIALIZED=0` to restore the previous dtype-generic kernel.
Recreate sessions using captured graphs after changing any of these flags.
