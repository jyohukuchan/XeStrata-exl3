# SPDX-FileCopyrightText: 2025 Turboderp
# SPDX-FileCopyrightText: 2026 jyohukuchan
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""Synthetic boundary tests for tools/exl3_ngram_reference.py."""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exl3_ngram_reference import MUL1, NgramReference, ROW_DIM, words_per_row  # noqa: E402


def pack_row(seed: int, scale: float, k: int) -> bytes:
    """Create a nontrivial packed ring row with deterministic stream bits."""
    words = np.zeros(words_per_row(k), dtype=np.uint16)
    words[0] = np.asarray([scale], dtype=np.float16).view(np.uint16)[0]
    for stream_bit in range(ROW_DIM * k):
        bit = ((stream_bit * 17 + seed * 13) ^ (stream_bit >> 2)) & 1
        words[1 + stream_bit // 16] |= np.uint16(bit << (stream_bit % 16))
    return words.view("<i2").tobytes()


def decoded_state(words: np.ndarray, k: int, i: int) -> int:
    state = 0
    for m in range(16):
        position = (i - m // k) % ROW_DIM
        stream_bit = position * k + m % k
        bit = (int(words[stream_bit // 16]) >> (stream_bit % 16)) & 1
        state |= bit << m
    return state


def write_sidecar(
    path: Path,
    k: int,
    rows: list[bytes],
    metadata_extra: dict[str, str] | None = None,
    multipliers: list[int] | None = None,
) -> None:
    prefix = "model.layers.0.ple.ngram"
    words = words_per_row(k)
    tensors: list[tuple[str, str, list[int], bytes]] = [
        (f"{prefix}.head_bias", "F16", [2, ROW_DIM], np.zeros((2, ROW_DIM), dtype="<f2").tobytes()),
        (f"{prefix}.head_offsets", "I64", [2], np.array([0, 3], dtype="<i8").tobytes()),
        (f"{prefix}.head_vocab_sizes", "I64", [2], np.array([3, 2], dtype="<i8").tobytes()),
        (
            f"{prefix}.layer_multipliers",
            "I64",
            [3],
            np.asarray(multipliers or [1, 3, 5], dtype=np.uint64).view("<i8").tobytes(),
        ),
    ]
    tensors.extend([
        (f"{prefix}.shard_0.trellis", "I16", [3, words], b"".join(rows[:3])),
        (f"{prefix}.shard_1.trellis", "I16", [len(rows) - 3, words], b"".join(rows[3:])),
    ])
    metadata = {
        "format": "exl3_ngram_trellis",
        "version": "1",
        "K": str(k),
        "codebook": "mul1",
        "row_dim": str(ROW_DIM),
        "rows": str(len(rows)),
        "shard_rows": "3",
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    header: dict[str, Any] = {"__metadata__": metadata}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        begin = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [begin, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def expected_codebook(state: int) -> float:
    product = (state * MUL1) & 0xFFFFFFFF
    byte_sum = sum((product >> shift) & 0xFF for shift in (0, 8, 16, 24))
    k_inv = np.array([0x1EEE], dtype=np.uint16).view(np.float16).astype(np.float32)[0]
    k_bias = np.array([0xC931], dtype=np.uint16).view(np.float16).astype(np.float32)[0]
    return np.float16((1024.0 + byte_sum) * k_inv + k_bias).astype(np.float32).item()


class NgramReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def test_k2_decode_crosses_shard_boundary(self):
        path = Path(self.temp.name) / "ngram.safetensors"
        write_sidecar(path, 2, [pack_row(row, 0.5 + row * 0.25, 2) for row in range(5)])
        reader = NgramReference(path)
        got = reader.decode_rows([2, 3, 0], heads=[0, 1, 0])
        packed = reader.read_rows([2, 3, 0])[:, 1:].view(np.uint16)
        expected = np.array(
            [expected_codebook(decoded_state(packed[index], 2, 0)) * np.float16(scale)
             for index, scale in enumerate((1.0, 1.25, 0.5))], dtype=np.float32
        )
        np.testing.assert_allclose(got[:, 0], expected, rtol=0.0, atol=0.0)
        self.assertEqual(reader.read_rows([4]).shape, (1, words_per_row(2)))

    def test_k5_last_row_and_ring_bits(self):
        path = Path(self.temp.name) / "ngram-k5.safetensors"
        write_sidecar(path, 5, [pack_row(row, 1.0, 5) for row in range(5)])
        reader = NgramReference(path)
        decoded = reader.decode_rows([4], heads=[1])
        self.assertEqual(decoded.shape, (1, ROW_DIM))
        packed = reader.read_rows([4])[0, 1:].view(np.uint16)
        self.assertEqual(decoded[0, 0], expected_codebook(decoded_state(packed, 5, 0)))
        self.assertEqual(reader.metadata["codebook"], "mul1")

    def test_lookup_uses_head_ranges_and_multipliers(self):
        path = Path(self.temp.name) / "ngram-lookup.safetensors"
        write_sidecar(path, 3, [pack_row(row, 1.0, 3) for row in range(5)])
        reader = NgramReference(path)
        got = reader.lookup_ids([1, 2, 3], ngram_size=3, heads_per_ngram=1, eos_token_id=99)
        mixed = (3 * 1) ^ (2 * 3)
        expected0 = mixed % 3
        mixed ^= 1 * 5
        expected1 = mixed % 2 + 3
        np.testing.assert_array_equal(got, [[expected0, expected1]])

    def test_lookup_signed_int64_and_eos_boundary(self):
        path = Path(self.temp.name) / "ngram-signed.safetensors"
        write_sidecar(
            path,
            3,
            [pack_row(row, 1.0, 3) for row in range(5)],
            multipliers=[0x7FFFFFFFFFFFFFED, 0x4000000000000001, 0x8000000000000003],
        )
        reader = NgramReference(path)
        tokens = [248045, 846, 198, 20206, 248044, 198, 248045, 74455, 198, 42]
        expected = np.array(
            [[0, 4], [2, 3], [2, 3], [2, 3], [2, 4], [1, 3], [0, 3], [1, 4]],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(reader.lookup_ids(tokens, 3, 1, 248044), expected)

    @unittest.skipUnless(os.environ.get("EXL3_NGRAM_NATIVE_HASH_MODEL"), "native oracle model not configured")
    def test_native_hash_cpu_matches_real_metadata(self):
        try:
            import torch
            import exllamav3_ext
        except ImportError as exc:
            self.skipTest(f"native extension unavailable: {exc}")
        reader = NgramReference(os.environ["EXL3_NGRAM_NATIVE_HASH_MODEL"])
        tokens = [248045, 846, 198, 20206, 248044, 198, 248045, 74455, 198, 42]
        seq_len = len(tokens) - 2
        capacity = seq_len * len(reader.head_offsets)
        uids = torch.empty(capacity, dtype=torch.int64)
        inverse = torch.empty(capacity, dtype=torch.int64)
        heads = torch.empty(capacity, dtype=torch.int32)
        used = exllamav3_ext.ngram_hash_cpu(
            torch.tensor([tokens], dtype=torch.int64),
            seq_len,
            torch.from_numpy(reader.layer_multipliers.copy()),
            torch.from_numpy(reader.head_offsets.copy()),
            torch.from_numpy(reader.head_vocab_sizes.copy()),
            len(reader.head_offsets) // (len(reader.layer_multipliers) - 1),
            248044,
            uids,
            inverse,
            heads,
        )
        ours = reader.lookup_ids(
            tokens,
            len(reader.layer_multipliers),
            len(reader.head_offsets) // (len(reader.layer_multipliers) - 1),
            248044,
        ).reshape(-1)
        unique, ours_inverse = np.unique(ours, return_inverse=True)
        ours_heads = np.searchsorted(reader.head_offsets, unique, side="right") - 1
        np.testing.assert_array_equal(uids[:used].numpy(), unique)
        np.testing.assert_array_equal(inverse[:capacity].numpy(), ours_inverse)
        np.testing.assert_array_equal(heads[:used].numpy(), ours_heads)


if __name__ == "__main__":
    unittest.main()
