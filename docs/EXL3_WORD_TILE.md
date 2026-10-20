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
