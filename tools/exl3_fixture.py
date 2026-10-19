# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Write deterministic EXL3 reconstruction and linear fixtures for C++ checks.

Little-endian EXL3FX01: u32 version, u32 case count, then each case's u32
bits/codebook/input width/output width. Arrays follow: uint16 trellis,
input/output scales, bias, float32 input, uint16 decoded weight matrix,
float32 output. Matrix storage and decoded weights are input-major.

The codebook and tile reference is checked separately against the native
extension. Linear targets use FP32 transforms and a double-precision dot,
rounded to FP32 before the output transform. This is a numerical fixture,
not a speed benchmark or a claim about model quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

import exl3_reference as ref


def write_fixture(path: Path, *, seed: int = 20261019, include_head_bits: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    widths = ref.SUPPORTED_BITS if include_head_bits else ref.PRIMARY_BITS
    cases = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"EXL3FX01" + struct.pack("<II", 1, len(widths) * 3))
        for bits in widths:
            k, n = (128, 256) if bits % 2 else (256, 128)
            for codebook in (0, 1, 2):
                packed = rng.integers(0, 65536, (k // 16, n // 16, 16 * bits), dtype=np.uint16)
                scale_in = (rng.uniform(0.5, 1.5, k) * rng.choice([-1, 1], k)).astype(np.float16)
                scale_out = (rng.uniform(0.5, 1.5, n) * rng.choice([-1, 1], n)).astype(np.float16)
                bias = rng.uniform(-0.05, 0.05, n).astype(np.float16)
                x = (rng.standard_normal(k) * 0.125).astype(np.float16).astype(np.float32)
                weight = ref.reconstruct_weight(packed, (k, n), bits, codebook)
                transformed = ref.apply_input_transform(x, scale_in)
                dot = (transformed.astype(np.float64) @ weight.astype(np.float64)).astype(np.float32)
                y = ref.apply_output_transform(dot, scale_out, bias)
                handle.write(struct.pack("<IIII", bits, codebook, k, n))
                for array in (packed, scale_in.view(np.uint16), scale_out.view(np.uint16),
                              bias.view(np.uint16), x, weight.view(np.uint16), y):
                    dtype = "<f4" if array.dtype == np.float32 else "<u2"
                    handle.write(np.asarray(array, dtype=dtype).tobytes(order="C"))
                cases.append({"bits": bits, "codebook": codebook, "k": k, "n": n})
    return {"format": "EXL3FX01", "seed": seed, "numpy": np.__version__, "cases": cases,
            "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "weight_comparison": "bit exact", "linear_atol": 5e-5, "linear_rtol": 5e-5}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--include-head-bits", action="store_true")
    args = parser.parse_args()
    report = write_fixture(args.out, seed=args.seed, include_head_bits=args.include_head_bits)
    args.out.with_suffix(args.out.suffix + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
