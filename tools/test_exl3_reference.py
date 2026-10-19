# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-FileCopyrightText: 2025 Turboderp
# SPDX-FileCopyrightText: 2026 jyohukuchan
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""Independent scalar checks for :mod:`exl3_reference`.

Run from the repository root with::

    python3 -m unittest tools.test_exl3_reference -v

The trellis oracle below reads individual little-endian bits.  It does not
reuse the reference's uint32 funnel-shift implementation, so ring-wrap and
word-crossing errors are visible.  Codebook checks use fixed half-bit answers
captured from the ROCm reference constants, rather than calling the module to
construct expected values.
"""
from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exl3_reference as ref  # noqa: E402


def bit_oracle(packed: np.ndarray, bits: int) -> np.ndarray:
    """Independent bit-at-a-time equivalent of exl3_dq.cuh's ring window."""
    raw = np.asarray(packed, dtype=np.uint16).astype("<u2", copy=False).tobytes()
    ring = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
    words = bits * 256 // 32
    out = np.empty(256, dtype=np.uint16)
    for t in range(256):
        b0 = t * bits + bits - 16 + 256 * bits
        b1 = b0 + 16
        # These index expressions are intentionally present in the oracle:
        # they check the source's uint32 word selection independently of the
        # reference's merged uint64 operation.
        i0 = (b0 // 32) % words
        i1 = ((b1 - 1) // 32) % words
        if not (0 <= i0 < words and 0 <= i1 < words):
            raise AssertionError("oracle word index escaped the ring")
        # The source loads ``a = ptr[i0]`` and ``b = ptr[i1]`` and applies a
        # right funnel shift with (b, a).  Spell that as bit selection rather
        # than reproducing the reference's merged uint64 expression.
        value = 0
        for bit in range(16):
            shift = ((b1 - 1) // 32 + 1) * 32 - b1
            source = ring[i1 * 32 + shift + bit] if shift + bit < 32 else ring[i0 * 32 + shift + bit - 32]
            value |= int(source) << bit
        out[t] = value
    return out


def oracle_hadamard(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Reference butterflies written independently from the module helper."""
    a = np.asarray(values, dtype=np.float32).copy()
    a = np.moveaxis(a, axis, -1)
    side = a.shape[-1]
    if side % 128:
        raise ValueError(side)
    a = a.reshape(*a.shape[:-1], -1, 128)
    width = 1
    while width < 128:
        for base in range(0, 128, width * 2):
            left = a[..., base:base + width].copy()
            right = a[..., base + width:base + width * 2].copy()
            a[..., base:base + width] = left + right
            a[..., base + width:base + width * 2] = left - right
        width *= 2
    return np.moveaxis(a.reshape(*np.asarray(values).shape), -1, axis)


class TrellisReferenceTests(unittest.TestCase):
    def test_tail_biting_ring_and_cross_word_windows(self):
        rng = np.random.default_rng(0xE3)
        for bits in (2, 3, 4, 5):
            packed = rng.integers(0, 1 << 16, size=16 * bits, dtype=np.uint16)
            np.testing.assert_array_equal(ref.unpack_trellis_states(packed, bits), bit_oracle(packed, bits))

    def test_tensor_core_permutation_has_known_edges(self):
        perm = ref.tensor_core_permutation()
        self.assertEqual(perm.tolist()[:8], [0, 16, 128, 144, 8, 24, 136, 152])
        self.assertEqual(perm.tolist()[-8:], [103, 119, 231, 247, 111, 127, 239, 255])
        self.assertEqual(sorted(perm.tolist()), list(range(256)))

    def test_unpack_shape_and_permutation_for_each_2_to_5_bit_format(self):
        for bits in (2, 3, 4, 5):
            packed = np.arange(16 * bits, dtype=np.uint16)
            got = ref.unpack_trellis_tile(packed, bits)
            self.assertEqual(got.shape, (16, 16))
            self.assertEqual(got.dtype, np.uint16)
            np.testing.assert_array_equal(got.reshape(-1)[ref.tensor_core_permutation()],
                                          ref.unpack_trellis_states(packed, bits))

    def test_codebooks_use_reference_half_bits(self):
        states = np.array([0, 1, 0x1234, 0xAAAA, 0xFFFF, 0x8000], dtype=np.uint16)
        expected = {
            0: [0x3A25, 0xBB5B, 0xC066, 0xBB35, 0xB110, 0xB7F8],
            1: [0x3F60, 0x304E, 0x3E0F, 0xBFBA, 0x3ACD, 0xB915],
            2: [0xC2E8, 0x3921, 0xB89E, 0x39B9, 0xB936, 0xBAF2],
        }
        for codebook, bits in expected.items():
            got = ref.decode_codebook(states, codebook).view(np.uint16)
            np.testing.assert_array_equal(got, np.array(bits, dtype=np.uint16))

    def test_each_codebook_and_bitwidth_returns_finite_tile(self):
        packed = np.arange(80, dtype=np.uint16)
        for bits in (2, 3, 4, 5):
            tile = packed[:16 * bits]
            for codebook in (0, 1, 2):
                got = ref.unpack_trellis_tile(tile, bits, codebook)
                self.assertEqual(got.shape, (16, 16))
                self.assertEqual(got.dtype, np.float16)
                self.assertTrue(np.isfinite(got).all())

    def test_mixed_tile_grid_accepts_2_to_5_bit_tiles_independently(self):
        bits = np.array([[2, 3], [4, 5]], dtype=np.int64)
        codebooks = np.array([[0, 1], [2, 0]], dtype=np.int64)
        tiles = [[
            np.arange(16 * int(bits[r, c]), dtype=np.uint16) + 17 * (r * 2 + c)
            for c in range(2)
        ] for r in range(2)]
        got = ref.reconstruct_mixed_weight(tiles, (32, 32), bits, codebooks)
        self.assertEqual(got.shape, (32, 32))
        for r in range(2):
            for c in range(2):
                expected = ref.unpack_trellis_tile(tiles[r][c], int(bits[r, c]), int(codebooks[r, c]))
                np.testing.assert_array_equal(got[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16], expected)


class TransformReferenceTests(unittest.TestCase):
    def test_hadamard_matches_independent_butterflies_and_is_involutory(self):
        x = np.linspace(-1.0, 1.0, 256, dtype=np.float32).reshape(2, 128)
        expected = oracle_hadamard(x) * np.float32(0.088388347648)
        got = ref.hadamard_128(x)
        np.testing.assert_allclose(got, expected, rtol=0, atol=0)
        restored = ref.hadamard_128(got)
        np.testing.assert_allclose(restored, x, rtol=0, atol=3e-6)

    def test_input_output_transforms_and_bias(self):
        x = np.linspace(-0.7, 0.9, 128, dtype=np.float32)
        scale = np.linspace(0.5, 1.5, 128, dtype=np.float32).astype(np.float16)
        bias = np.linspace(-0.2, 0.2, 128, dtype=np.float32).astype(np.float16)
        hscale = np.float32(0.088388347648)
        input_expected = oracle_hadamard(x * scale.astype(np.float32)) * hscale
        output_expected = oracle_hadamard(x) * hscale * scale.astype(np.float32) + bias.astype(np.float32)
        np.testing.assert_allclose(ref.apply_input_transform(x, scale), input_expected, rtol=0, atol=0)
        np.testing.assert_allclose(ref.apply_output_transform(x, scale, bias), output_expected, rtol=0, atol=0)

    def test_reconstruct_transformed_128_square(self):
        bits = 2
        packed = np.arange(64 * 16 * bits, dtype=np.uint16).reshape(8, 8, 16 * bits)
        raw = ref.reconstruct_weight(packed, (128, 128), bits, 1)
        suh = np.linspace(0.75, 1.25, 128, dtype=np.float32).astype(np.float16)  # codespell:ignore suh
        svh = np.linspace(1.25, 0.75, 128, dtype=np.float32).astype(np.float16)
        got = ref.reconstruct_weight(packed, (128, 128), bits, 1, suh, svh)  # codespell:ignore suh
        # The output is half arithmetic.  Compute the expected matrix through
        # the public scalar transform with float32 only for a tolerance check;
        # this also checks dimensions and scale placement independently of the
        # tile loader and the codebook oracle above.
        expected = oracle_hadamard(oracle_hadamard(raw.astype(np.float32), axis=0), axis=1)
        expected = expected * np.float32(0.088388347648) ** 2
        expected = expected * suh.astype(np.float32)[:, None] * svh.astype(np.float32)[None, :]  # codespell:ignore suh
        np.testing.assert_allclose(got.astype(np.float32), expected, rtol=0, atol=3e-2)


if __name__ == "__main__":
    unittest.main()
