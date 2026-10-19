#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare the NumPy EXL3 reference with an optional native extension.

The native imports are intentionally inside :func:`main`: importing this
module remains possible on a CPU-only machine.  The runner compares packed
2/3/4/5-bit tiles and cb0/cb1/cb2, plus FP32 input/output Hadamard calls and
the fused original-basis reconstruction.  It reports bit mismatches and
floating error; it makes no throughput claim.

Example in a prepared oracle container::

    python3 exl3_native_parity.py --device cuda:0 \
        --out exl3-native-parity-had.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import exl3_reference as ref


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="Torch device for the native calls")
    parser.add_argument("--out", default="exl3-native-parity.json", type=Path, help="JSON report path")
    parser.add_argument("--seed", default=20261019, type=int)
    parser.add_argument("--tolerance", default=0.0, type=float,
                        help="optional absolute error allowance; fused parity still requires zero bit mismatches")
    return parser


def _summary(got, expected, *, bitwise: bool = False) -> dict[str, Any]:
    got_np = got.detach().cpu().numpy()
    expected_np = np.asarray(expected)
    diff = got_np.astype(np.float32) - expected_np.astype(np.float32)
    result: dict[str, Any] = {
        "shape": list(got_np.shape),
        "dtype": str(got_np.dtype),
        "max_abs_error": float(np.max(np.abs(diff))) if diff.size else 0.0,
    }
    if bitwise:
        result["bit_mismatches"] = int(np.count_nonzero(got_np.view(np.uint16) != expected_np.view(np.uint16)))
    return result


def main() -> int:
    args = _parser().parse_args()

    # Optional runtime-only dependencies.  The runner is imported after the
    # reference module so that its CPU tests never need torch or the extension.
    import torch
    import exllamav3_ext as ext

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("native EXL3 parity requires a CUDA/HIP torch device")
    rng = np.random.default_rng(args.seed)
    cases: list[dict[str, Any]] = []

    for bits in (2, 3, 4, 5):
        packed = rng.integers(0, 65536, (8, 8, bits * 16), dtype=np.uint16)
        gpu_packed = torch.from_numpy(packed.view(np.int16)).to(device)
        for codebook in (0, 1, 2):
            native = torch.empty((128, 128), dtype=torch.float16, device=device)
            ext.reconstruct(native, gpu_packed, bits, codebook == 1, codebook == 2)
            torch.cuda.synchronize(device)
            expected = ref.reconstruct_weight(packed, (128, 128), bits, codebook)
            row = {"kind": "reconstruct", "bits": bits, "codebook": codebook}
            row.update(_summary(native, expected, bitwise=True))
            cases.append(row)

    values = rng.standard_normal((2, 128), dtype=np.float32)
    scale = rng.uniform(0.5, 1.5, 128).astype(np.float16)
    gpu_values = torch.from_numpy(values).to(device)
    gpu_scale = torch.from_numpy(scale).to(device)

    native_input = torch.empty_like(gpu_values)
    ext.had_r_128(gpu_values, native_input, gpu_scale, None, 1.0)
    torch.cuda.synchronize(device)
    row = {"kind": "hadamard_input"}
    row.update(_summary(native_input, ref.apply_input_transform(values, scale)))
    cases.append(row)

    native_output = torch.empty_like(gpu_values)
    ext.had_r_128(gpu_values, native_output, None, gpu_scale, 1.0)
    torch.cuda.synchronize(device)
    row = {"kind": "hadamard_output"}
    row.update(_summary(native_output, ref.apply_output_transform(values, scale)))
    cases.append(row)

    fused_cases: list[dict[str, Any]] = []
    fused_scale_in = rng.uniform(0.75, 1.25, 128).astype(np.float16)
    fused_scale_out = rng.uniform(0.75, 1.25, 128).astype(np.float16)
    gpu_suh = torch.from_numpy(fused_scale_in).to(device)
    gpu_svh = torch.from_numpy(fused_scale_out).to(device)
    for bits in (2, 3, 4, 5):
        packed = rng.integers(0, 65536, (8, 8, bits * 16), dtype=np.uint16)
        gpu_packed = torch.from_numpy(packed.view(np.int16)).to(device)
        for codebook in (0, 1, 2):
            native = torch.empty((128, 128), dtype=torch.float16, device=device)
            ext.reconstruct_had_slice(
                native, gpu_packed, gpu_suh, gpu_svh, bits, codebook == 1, codebook == 2, 0
            )
            torch.cuda.synchronize(device)
            expected = ref.reconstruct_weight(
                packed, (128, 128), bits, codebook, fused_scale_in, fused_scale_out
            )
            row = {"kind": "reconstruct_had", "bits": bits, "codebook": codebook}
            row.update(_summary(native, expected, bitwise=True))
            fused_cases.append(row)
    cases.extend(fused_cases)

    tolerance = args.tolerance
    passed = all(
        row.get("bit_mismatches", 0) == 0 and float(row["max_abs_error"]) <= tolerance
        for row in cases if row["kind"] == "reconstruct"
    ) and all(
        row.get("bit_mismatches", 0) == 0 and float(row["max_abs_error"]) <= tolerance
        for row in cases if row["kind"] == "reconstruct_had"
    ) and all(
        float(row["max_abs_error"]) <= 1.0e-3
        for row in cases if row["kind"].startswith("hadamard_")
    )
    report = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "native_extension": str(getattr(ext, "__file__", "unknown")),
        "seed": args.seed,
        "tolerance": tolerance,
        "cases": cases,
        "passed": passed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
