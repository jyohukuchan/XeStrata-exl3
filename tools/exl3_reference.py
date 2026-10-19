#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-FileCopyrightText: 2025 Turboderp
# SPDX-FileCopyrightText: 2026 jyohukuchan
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""Small, scalar EXL3 reference operations.

The trellis extraction and codebook portions are adapted from exllamav3,
Copyright (c) 2025 Turboderp and Copyright (c) 2026 jyohukuchan, under the
MIT License.  The source reference is
https://github.com/jyohukuchan/rocm_exl3; the standard MIT notice is also
present in ``LICENSES/MIT.txt``.  The adapted scalar code in this file
remains under XeStrata's LGPL terms.

The packed tile and codebook rules here mirror the standalone ROCm EXL3
implementation in ``rocm_exl3/exllamav3/exllamav3_ext``:

* ``quant/exl3_dq.cuh`` extracts 16-bit tail-biting trellis states;
* ``cpu/moe_mul1.cpp`` supplies the tensor-core permutation;
* ``quant/codebook.cuh`` supplies the three 3-instruction codebooks; and
* ``cpu/moe_mul1.cpp`` / ``quant/reconstruct.cu`` define the 128-point
  Hadamard and sign-scale transforms.

This module deliberately favours transparent scalar NumPy over throughput.
The codebook operations round through ``float16`` at the documented
boundaries.  NumPy does not expose the device's fused half FMA, so cb2 uses
one float32 intermediate followed by one float16 conversion.  The exact cb2
rounding should be checked against the native extension when that comparison
is available; this file does not claim a numerical error bound.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np


PRIMARY_BITS = (2, 3, 4, 5)
SUPPORTED_BITS = tuple(range(1, 9))
TILE_VALUES = 256
TILE_SIDE = 16
HADAMARD_SIDE = 128
HADAMARD_SCALE = np.float32(0.088388347648)  # 1 / sqrt(128), as in moe_mul1.cpp
# gitleaks:allow -- these are public EXL3 codebook multipliers, never secrets.


def _check_bits(bits: int) -> int:
    bits = int(bits)
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"EXL3 reference supports bits {SUPPORTED_BITS}, got {bits}")
    return bits


def _u16(values: np.ndarray | Sequence[int]) -> np.ndarray:
    out = np.asarray(values, dtype=np.uint16)
    if out.ndim != 1:
        raise ValueError(f"packed trellis data must be one-dimensional, got {out.shape}")
    return np.ascontiguousarray(out)


def tensor_core_permutation() -> np.ndarray:
    """Return the 256-entry EXL3 tensor-core to row-major permutation.

    State ``t`` in a trellis tile reconstructs matrix element
    ``out.flat[perm[t]]``.  This is ``make_tc_perm`` from ``moe_mul1.cpp``.
    """
    perm = np.empty(TILE_VALUES, dtype=np.intp)
    for t in range(32):
        r0 = (t % 4) * 2
        r1, r2, r3 = r0 + 1, r0 + 8, r0 + 9
        c0, c1 = t // 4, t // 4 + 8
        base = t * 8
        perm[base:base + 8] = (
            r0 * 16 + c0, r1 * 16 + c0, r2 * 16 + c0, r3 * 16 + c0,
            r0 * 16 + c1, r1 * 16 + c1, r2 * 16 + c1, r3 * 16 + c1,
        )
    return perm


_TC_PERM = tensor_core_permutation()


def unpack_trellis_states(packed: np.ndarray | Sequence[int], bits: int) -> np.ndarray:
    """Extract the 256 16-bit trellis states from one packed EXL3 tile.

    ``packed`` has ``16 * bits`` little-endian ``uint16`` words.  The state
    window advances by ``bits`` and wraps at the tile's ``256 * bits``-bit
    ring (256 code positions), including windows that cross the final word.
    The returned order is trellis state order; use :func:`unpack_trellis_tile`
    for matrix order.
    """
    bits = _check_bits(bits)
    words = _u16(packed)
    expected = 16 * bits
    if words.size != expected:
        raise ValueError(f"bits={bits} needs {expected} uint16 words, got {words.size}")

    # The CUDA code loads uint32 words from the uint16 buffer.  Normalising to
    # explicit little-endian bytes keeps this reference independent of host
    # endianness while retaining that load's bit order.
    words32 = np.frombuffer(words.astype("<u2", copy=False).tobytes(), dtype="<u4")
    words32_count = bits * 256 // 32
    states = np.empty(TILE_VALUES, dtype=np.uint16)
    for t in range(TILE_VALUES):
        b0 = t * bits + bits - 16 + 256 * bits
        b1 = b0 + 16
        i0 = (b0 // 32) % words32_count
        i1 = ((b1 - 1) // 32) % words32_count
        shift = ((b1 - 1) // 32 + 1) * 32 - b1
        merged = (int(words32[i0]) << 32) | int(words32[i1])
        states[t] = (merged >> shift) & 0xFFFF
    return states


def _lop3_6a(value: int) -> int:
    # codebook_rdna.hip.h: (~x & 0x3b603b60) | (x & (0x8fff8fff ^ 0x3b603b60))
    return ((~value & 0x3B603B60) | (value & (0x8FFF8FFF ^ 0x3B603B60))) & 0xFFFFFFFF


def _half_bits(value: int) -> np.float16:
    return cast(np.float16, np.array([value & 0xFFFF], dtype=np.uint16).view(np.float16)[0])


def _half_add(a: np.float16, b: np.float16) -> np.float16:
    # Array operands ensure NumPy performs the add in half precision.
    return cast(np.float16, np.asarray(a, dtype=np.float16) + np.asarray(b, dtype=np.float16))


def _half_add_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(np.asarray(a, dtype=np.float16) + np.asarray(b, dtype=np.float16), dtype=np.float16)


def _half_sub_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(np.asarray(a, dtype=np.float16) - np.asarray(b, dtype=np.float16), dtype=np.float16)


def _half_mul_array(a: np.ndarray, b: np.ndarray | np.float16) -> np.ndarray:
    return np.asarray(np.asarray(a, dtype=np.float16) * np.asarray(b, dtype=np.float16), dtype=np.float16)


def _half_sign_flip(values: np.ndarray) -> np.ndarray:
    """Apply the sign-bit xor used before each native half shuffle add."""
    return (np.asarray(values, dtype=np.float16).view(np.uint16) ^ np.uint16(0x8000)).view(np.float16)


def decode_codebook(states: np.ndarray | Sequence[int], codebook: int) -> np.ndarray:
    """Decode state words with cb0, cb1 or cb2, returning ``float16`` values.

    cb0/cb1 are the two half lanes produced by the ``lop3`` codebook and
    summed in half precision.  cb2 uses the byte sum from the mul1 codebook,
    interprets the integer accumulator as half bits, then applies the half
    inverse scale and bias.  Its fused half FMA is represented by the
    portable float32-intermediate rule described in the module docstring.
    """
    codebook = int(codebook)
    if codebook not in (0, 1, 2):
        raise ValueError(f"EXL3 codebook must be 0, 1 or 2, got {codebook}")
    src = np.asarray(states, dtype=np.uint16)
    out = np.empty(src.shape, dtype=np.float16)
    flat = src.reshape(-1)
    dst = out.reshape(-1)
    for i, state in enumerate(flat):
        x = int(state)
        if codebook == 0:
            x = (x * 89226354 + 64248484) & 0xFFFFFFFF
            x = _lop3_6a(x)
            dst[i] = _half_add(_half_bits(x), _half_bits(x >> 16))
        elif codebook == 1:
            x = (x * 0xCBAC1FED) & 0xFFFFFFFF
            x = _lop3_6a(x)
            dst[i] = _half_add(_half_bits(x), _half_bits(x >> 16))
        else:
            x = (x * 0x83DCD12D) & 0xFFFFFFFF
            byte_sum = sum((x >> (8 * lane)) & 0xFF for lane in range(4))
            # __dp4a(..., acc=0x6400) leaves the accumulator's low 16 bits
            # as the half bit pattern 0x6400..0x67ff.
            h = _half_bits(0x6400 + byte_sum)
            inv = _half_bits(0x1EEE)
            bias = _half_bits(0xC931)
            dst[i] = np.float16(np.float32(h) * np.float32(inv) + np.float32(bias))
    return out


def unpack_trellis_tile(
    packed: np.ndarray | Sequence[int], bits: int, codebook: int | None = None
) -> np.ndarray:
    """Unpack one tile to a ``(16, 16)`` matrix.

    With ``codebook=None`` the matrix contains the raw uint16 states.  With a
    codebook it contains decoded float16 values.  The tensor-core permutation
    is always applied, matching ``make_tc_perm`` and the reconstruct kernels.
    """
    states = unpack_trellis_states(packed, bits)
    values: np.ndarray = states if codebook is None else decode_codebook(states, codebook)
    out = np.empty(TILE_VALUES, dtype=values.dtype)
    out[_TC_PERM] = values
    return out.reshape(TILE_SIDE, TILE_SIDE)


def _hadamard_blocks(values: np.ndarray, axis: int, *, dtype: Any) -> np.ndarray:
    """Apply unnormalised Sylvester H128 along one axis in the requested dtype."""
    out = np.asarray(values, dtype=dtype).copy()
    out = np.moveaxis(out, axis, -1)
    if out.shape[-1] % HADAMARD_SIDE:
        raise ValueError(f"Hadamard axis must be divisible by 128, got {out.shape[-1]}")
    shape = out.shape
    out = out.reshape(*shape[:-1], -1, HADAMARD_SIDE)
    for width in (1, 2, 4, 8, 16, 32, 64):
        for base in range(0, HADAMARD_SIDE, 2 * width):
            left = out[..., base:base + width].copy()
            right = out[..., base + width:base + 2 * width].copy()
            out[..., base:base + width] = left + right
            out[..., base + width:base + 2 * width] = left - right
    out = out.reshape(shape)
    out = np.moveaxis(out, -1, axis)
    return out


def hadamard_128(values: np.ndarray | Sequence[float], axis: int = -1, *, normalize: bool = True,
                 dtype: Any = np.float32) -> np.ndarray:
    """Apply the scalar 128-point Sylvester transform.

    ``dtype=np.float16`` follows the half arithmetic used by the fused GPU
    reconstruct path.  The default float32 path follows the CPU transform
    used by the EXL3 MoE fallback.  ``normalize`` multiplies by
    ``1/sqrt(128)`` after the butterflies.
    """
    out = _hadamard_blocks(np.asarray(values), axis, dtype=dtype)
    if normalize:
        out = out * np.asarray(HADAMARD_SCALE, dtype=dtype)
    return out


# codespell:ignore-next-line suh
def apply_input_transform(values: np.ndarray | Sequence[float], suh: np.ndarray | Sequence[float]) -> np.ndarray:
    # codespell:ignore-next-line suh
    """Apply EXL3's input transform: ``H128 * diag(suh) * x`` per block."""
    x = np.asarray(values, dtype=np.float32)
    scale = np.asarray(suh, dtype=np.float16).astype(np.float32)  # codespell:ignore suh
    if x.shape[-1] != scale.size:
        # codespell:ignore-next-line suh
        raise ValueError(f"input width {x.shape[-1]} does not match suh size {scale.size}")
    x = x * scale
    return hadamard_128(x, axis=-1, dtype=np.float32)


def apply_output_transform(
    values: np.ndarray | Sequence[float], svh: np.ndarray | Sequence[float],
    bias: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    """Apply ``H128 * 1/sqrt(128) * diag(svh)`` and optional fp16 bias."""
    y = hadamard_128(np.asarray(values, dtype=np.float32), axis=-1, dtype=np.float32)
    scale = np.asarray(svh, dtype=np.float16).astype(np.float32)
    if y.shape[-1] != scale.size:
        raise ValueError(f"output width {y.shape[-1]} does not match svh size {scale.size}")
    y = y * scale
    if bias is not None:
        b = np.asarray(bias, dtype=np.float16).astype(np.float32)
        if b.size != scale.size:
            raise ValueError(f"bias size {b.size} does not match svh size {scale.size}")
        y = y + b
    return y


# codespell:ignore-next-line suh
def _restore_basis(out: np.ndarray, suh: np.ndarray | Sequence[float], svh: np.ndarray | Sequence[float]) -> np.ndarray:
    """Apply the fused ROCm kernel's FP16/FP32 butterfly schedule.

    This intentionally does not call :func:`hadamard_128`.  The native fused
    kernel scales the first four-element butterflies before the five lane
    butterflies, uses FP16 for the column side, uses FP32 for the row side's
    first four-element butterfly, then casts to FP16 before the row lane
    butterflies and applies the input/output sign scales as sequential FP16
    multiplies.
    """
    rows, cols = out.shape
    if rows % 128 or cols % 128:
        raise ValueError("suh/svh basis reconstruction requires dimensions divisible by 128")  # codespell:ignore suh
    su = np.asarray(suh, dtype=np.float16)  # codespell:ignore suh
    sv = np.asarray(svh, dtype=np.float16)
    if su.size != rows or sv.size != cols:
        raise ValueError("suh/svh sizes must match reconstructed matrix dimensions")  # codespell:ignore suh

    rs2 = np.float16(0.08838834764831845)

    # Column transform: four-row FP16 butterflies, early scale, then the five
    # lane XOR butterflies from shuffle_had_h2x32.
    col = np.asarray(out, dtype=np.float16).copy()
    s0 = _half_add_array(col[0::4], col[1::4])
    d0 = _half_sub_array(col[0::4], col[1::4])
    s1 = _half_add_array(col[2::4], col[3::4])
    d1 = _half_sub_array(col[2::4], col[3::4])
    col4 = np.empty_like(col)
    col4[0::4] = _half_mul_array(_half_add_array(s0, s1), rs2)
    col4[1::4] = _half_mul_array(_half_add_array(d0, d1), rs2)
    col4[2::4] = _half_mul_array(_half_sub_array(s0, s1), rs2)
    col4[3::4] = _half_mul_array(_half_sub_array(d0, d1), rs2)
    for lane_bit in (1, 2, 4, 8, 16):
        previous = col4.copy()
        for lane in range(32):
            current = previous[lane * 4:(lane + 1) * 4]
            if lane & lane_bit:
                current = _half_sign_flip(current)
            partner_start = (lane ^ lane_bit) * 4
            col4[lane * 4:(lane + 1) * 4] = _half_add_array(
                current, previous[partner_start:partner_start + 4]
            )

    # Row transform: the source converts the first four-column butterfly from
    # FP32 to half, then performs the same five half lane butterflies.
    transformed = np.empty_like(col4)
    for r0 in range(0, rows, 128):
        for c0 in range(0, cols, 128):
            block = col4[r0:r0 + 128, c0:c0 + 128]
            row_out = np.empty_like(block)
            for row in range(128):
                values = block[row]
                s0f = values[0::4].astype(np.float32) + values[1::4].astype(np.float32)
                d0f = values[0::4].astype(np.float32) - values[1::4].astype(np.float32)
                s1f = values[2::4].astype(np.float32) + values[3::4].astype(np.float32)
                d1f = values[2::4].astype(np.float32) - values[3::4].astype(np.float32)
                row4 = np.empty(128, dtype=np.float16)
                row4[0::4] = np.asarray(s0f + s1f, dtype=np.float16) * rs2
                row4[1::4] = np.asarray(d0f + d1f, dtype=np.float16) * rs2
                row4[2::4] = np.asarray(s0f - s1f, dtype=np.float16) * rs2
                row4[3::4] = np.asarray(d0f - d1f, dtype=np.float16) * rs2
                for lane_bit in (1, 2, 4, 8, 16):
                    previous = row4.copy()
                    for lane in range(32):
                        current = previous[lane * 4:(lane + 1) * 4]
                        if lane & lane_bit:
                            current = _half_sign_flip(current)
                        partner_start = (lane ^ lane_bit) * 4
                        row4[lane * 4:(lane + 1) * 4] = _half_add_array(
                            current, previous[partner_start:partner_start + 4]
                        )
                row4 = _half_mul_array(row4, su[r0 + row])
                row4 = _half_mul_array(row4, sv[c0:c0 + 128])
                row_out[row] = row4
            transformed[r0:r0 + 128, c0:c0 + 128] = row_out
    return transformed


def reconstruct_mixed_weight(
    packed_tiles: Sequence[Sequence[np.ndarray | Sequence[int]]], shape: tuple[int, int],
    bits: np.ndarray | Sequence[Sequence[int]], codebooks: int | np.ndarray | Sequence[Sequence[int]],
    suh: np.ndarray | Sequence[float] | None = None,  # codespell:ignore suh
    svh: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    """Reconstruct a matrix whose 16x16 tiles independently use 2, 3, 4 or 5 bits.

    ``packed_tiles[r][c]`` may therefore have a different length for every
    tile.  ``bits`` and ``codebooks`` are tile grids; a scalar codebook is
    broadcast to all tiles.  This is a correctness helper for mixed arrays,
    with the same optional 128-basis transform as :func:`reconstruct_weight`.
    """
    rows, cols = int(shape[0]), int(shape[1])
    if rows <= 0 or cols <= 0 or rows % 16 or cols % 16:
        raise ValueError(f"weight shape must be positive multiples of 16, got {shape}")
    tile_rows, tile_cols = rows // 16, cols // 16
    bit_grid = np.asarray(bits, dtype=np.int64)
    if bit_grid.shape != (tile_rows, tile_cols):
        raise ValueError(f"bits grid must be {(tile_rows, tile_cols)}, got {bit_grid.shape}")
    if np.isscalar(codebooks):
        cb_grid = np.full((tile_rows, tile_cols), int(cast(Any, codebooks)), dtype=np.int64)
    else:
        cb_grid = np.asarray(codebooks, dtype=np.int64)
    if cb_grid.shape != (tile_rows, tile_cols):
        raise ValueError(f"codebooks grid must be {(tile_rows, tile_cols)}, got {cb_grid.shape}")
    if len(packed_tiles) != tile_rows or any(len(row) != tile_cols for row in packed_tiles):
        raise ValueError("packed_tiles nesting does not match the matrix tile shape")
    out = np.empty((rows, cols), dtype=np.float16)
    for kr in range(tile_rows):
        for nc in range(tile_cols):
            bits_ = _check_bits(int(bit_grid[kr, nc]))
            out[kr * 16:(kr + 1) * 16, nc * 16:(nc + 1) * 16] = unpack_trellis_tile(
                packed_tiles[kr][nc], bits_, int(cb_grid[kr, nc])
            )
    if suh is None and svh is None:  # codespell:ignore suh
        return out
    if suh is None or svh is None:  # codespell:ignore suh
        raise ValueError("suh and svh must be supplied together")  # codespell:ignore suh
    return _restore_basis(out, suh, svh)  # codespell:ignore suh


def reconstruct_weight(
    packed_tiles: np.ndarray | Sequence[int], shape: tuple[int, int], bits: int, codebook: int,
    suh: np.ndarray | Sequence[float] | None = None,  # codespell:ignore suh
    svh: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    """Reconstruct an EXL3 matrix, optionally returning the original basis.

    ``packed_tiles`` is either one tile (a flat ``uint16`` array) or a
    ``(shape[0]//16, shape[1]//16, 16*bits)`` array.  Without ``suh``/``svh``  # codespell:ignore suh
    this returns decoded tiles in the transformed basis.  When both are
    supplied, each 128x128 block receives the fused operation from
    ``reconstruct.cu``: ``diag(suh) H W_hat H diag(svh)`` with half-valued  # codespell:ignore suh
    weights and half butterfly arithmetic.  Dimensions must be multiples of
    128 for the transformed form.
    """
    bits = _check_bits(bits)
    rows, cols = (int(shape[0]), int(shape[1]))
    if rows <= 0 or cols <= 0 or rows % 16 or cols % 16:
        raise ValueError(f"weight shape must be positive multiples of 16, got {shape}")
    tiles = np.asarray(packed_tiles, dtype=np.uint16)
    words = 16 * bits
    if tiles.ndim == 1:
        if rows != 16 or cols != 16 or tiles.size != words:
            raise ValueError("flat packed input is only valid for one 16x16 tile")
        tiles = tiles.reshape(1, 1, words)
    expected = (rows // 16, cols // 16, words)
    if tiles.shape != expected:
        raise ValueError(f"packed tile shape must be {expected}, got {tiles.shape}")
    out = np.empty((rows, cols), dtype=np.float16)
    for kr in range(rows // 16):
        for nc in range(cols // 16):
            out[kr * 16:(kr + 1) * 16, nc * 16:(nc + 1) * 16] = unpack_trellis_tile(
                tiles[kr, nc], bits, codebook
            )
    if suh is None and svh is None:  # codespell:ignore suh
        return out
    if suh is None or svh is None or rows % 128 or cols % 128:  # codespell:ignore suh
        raise ValueError("suh and svh are both required for 128x128 basis reconstruction")  # codespell:ignore suh
    return _restore_basis(out, suh, svh)  # codespell:ignore suh
