#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Turboderp
# SPDX-FileCopyrightText: 2026 jyohukuchan
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""Generate small CPU/GPU EXL3 n-gram packed-row fixtures.

The binary format is intentionally independent of safetensors and torch:

    magic[8] = EXL3NG01, uint32 version=1, uint32 case_count
    per case: uint32 bits, rows, row_stride_words, num_heads
               int32[rows] heads
               uint16[rows * row_stride_words] packed rows
               uint16[num_heads * 160] F16 head bias bits
               float32[rows * 160] independent FP32 gold

Each packed row starts with one F16 scale word and then 10*K U16 ring words.
Words after the ring are padding and are never consumed by the decoder.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import struct

import numpy as np

ROW_DIM = 160
MUL1 = 0x83DCD12D
INV_BITS = np.uint16(0x1EEE)
BIAS_BITS = np.uint16(0xC931)


def words_per_row(bits: int) -> int:
    if not 1 <= bits <= 8:
        raise ValueError(bits)
    return 1 + ROW_DIM * bits // 16


def f16_bits(value: float) -> np.uint16:
    return np.asarray([value], dtype=np.float16).view(np.uint16)[0]


def codebook_mul1(state: int) -> np.float32:
    product = (int(state) * MUL1) & 0xFFFFFFFF
    byte_sum = sum((product >> shift) & 0xFF for shift in (0, 8, 16, 24))
    h = np.asarray([np.uint16(0x6400 + byte_sum)], dtype=np.uint16).view(np.float16).astype(np.float32)[0]
    inv = np.asarray([INV_BITS], dtype=np.uint16).view(np.float16).astype(np.float32)[0]
    bias = np.asarray([BIAS_BITS], dtype=np.uint16).view(np.float16).astype(np.float32)[0]
    return np.asarray(np.float32(h * inv + bias), dtype=np.float16).astype(np.float32)[()]


def unpack_state(row_ring: np.ndarray, bits: int, element: int) -> int:
    state = 0
    for m in range(16):
        position = (element - m // bits) % ROW_DIM
        stream_bit = position * bits + m % bits
        bit = (int(row_ring[stream_bit >> 4]) >> (stream_bit & 15)) & 1
        state |= bit << m
    return state


def make_case(bits: int) -> tuple[int, int, int, int, bytes, bytes, bytes, bytes]:
    rows = 2 + bits % 3
    heads = 3 + bits % 2
    words = words_per_row(bits)
    stride = words + 2 + bits % 4
    head_ids = np.asarray([(row * 3 + bits) % heads for row in range(rows)], dtype="<i4")
    packed = np.full((rows, stride), np.uint16(0xA55A ^ bits), dtype="<u2")
    scales = np.empty(rows, dtype=np.float16)
    for row in range(rows):
        scales[row] = np.float16(0.0 if (row + bits) % 4 == 0 else 0.25 + 0.03125 * (row + 1))
        packed[row, 0] = scales[row].view(np.uint16)
        for stream_bit in range(ROW_DIM * bits):
            bit = ((stream_bit * 13 + row * 29 + bits * 7) ^ (stream_bit >> 3)) & 1
            packed[row, 1 + (stream_bit >> 4)] |= np.uint16(bit << (stream_bit & 15))

    head_bias = np.empty((heads, ROW_DIM), dtype=np.float16)
    for head in range(heads):
        head_bias[head] = np.asarray(
            (head + 1) * 0.0375 + ((np.arange(ROW_DIM) % 13) - 6) * 0.0015, dtype=np.float16
        )

    gold = np.empty((rows, ROW_DIM), dtype=np.float32)
    for row in range(rows):
        scale = np.asarray(scales[row], dtype=np.float16).astype(np.float32)
        head = head_bias[head_ids[row]].astype(np.float32)
        for element in range(ROW_DIM):
            value = np.float32(codebook_mul1(unpack_state(packed[row, 1:words], bits, element)) * scale)
            gold[row, element] = np.float32(value + np.float32(head[element]))
    return (
        bits,
        rows,
        stride,
        heads,
        head_ids.tobytes(),
        packed.tobytes(),
        head_bias.view(np.uint16).astype("<u2", copy=False).tobytes(),
        gold.astype("<f4", copy=False).tobytes(),
    )


def write_fixture(path: Path) -> None:
    cases = [make_case(bits) for bits in range(1, 9)]
    with path.open("wb") as out:
        out.write(struct.pack("<8sII", b"EXL3NG01", 1, len(cases)))
        for bits, rows, stride, heads, head_ids, packed, bias, gold in cases:
            out.write(struct.pack("<IIII", bits, rows, stride, heads))
            out.write(head_ids)
            out.write(packed)
            out.write(bias)
            out.write(gold)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_fixture(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
