# SPDX-FileCopyrightText: 2025 Turboderp
# SPDX-FileCopyrightText: 2026 jyohukuchan
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""Small CPU reference reader for ``exl3_ngram_trellis`` sidecars.

The n-gram table uses a specialized 160-value tail-biting ring.  Its rows are
not ordinary EXL3 linear trellises: word zero is an FP16 row scale and the
remaining words contain the K-bit ring stream.  This module reads only the
requested rows from safetensors shards and reconstructs them with NumPy.
It is a correctness helper, not an inference or optimized runtime path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
from typing import Any, Iterable

import numpy as np

try:
    from .exl3_reference import decode_codebook
except ImportError:  # direct ``python tools/exl3_ngram_reference.py`` execution
    from exl3_reference import decode_codebook


ROW_DIM = 160
MUL1 = 0x83DCD12D
UINT64_MASK = (1 << 64) - 1
INT64_SIGN = 1 << 63
MAX_HEADER_BYTES = 100 * 1024 * 1024
DTYPE_BYTES = {"I16": 2, "I64": 8, "F16": 2, "F32": 4}


class NgramFormatError(ValueError):
    """The sidecar is malformed or uses an unsupported layout."""


def words_per_row(k: int) -> int:
    if not 1 <= k <= 8:
        raise NgramFormatError(f"K must be in 1..8, got {k}")
    return 1 + ROW_DIM * k // 16


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _product(shape: list[int]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result


def _load_header(path: Path) -> tuple[list[dict[str, Any]], dict[str, str], int, int]:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise NgramFormatError(f"{path}: truncated safetensors prefix")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 0 < header_size <= MAX_HEADER_BYTES or 8 + header_size > file_size:
                raise NgramFormatError(f"{path}: invalid safetensors header size {header_size}")
            raw = f.read(header_size)
    except OSError as exc:
        raise NgramFormatError(f"{path}: cannot read: {exc}") from exc
    if len(raw) != header_size:
        raise NgramFormatError(f"{path}: truncated safetensors header")
    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs_no_duplicates)
    except (UnicodeDecodeError, ValueError) as exc:
        raise NgramFormatError(f"{path}: invalid header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise NgramFormatError(f"{path}: header is not an object")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()
    ):
        raise NgramFormatError(f"{path}: __metadata__ must be string-to-string")
    data_start = 8 + header_size
    tensors: list[dict[str, Any]] = []
    spans: list[tuple[int, int, str]] = []
    for name, item in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(item, dict):
            raise NgramFormatError(f"{path}:{name}: tensor entry is not an object")
        dtype = item.get("dtype")
        shape = item.get("shape")
        offsets = item.get("data_offsets")
        if (
            dtype not in DTYPE_BYTES
            or not isinstance(shape, list)
            or not all(isinstance(x, int) and x >= 0 for x in shape)
        ):
            raise NgramFormatError(f"{path}:{name}: unsupported tensor declaration")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(x, int) for x in offsets):
            raise NgramFormatError(f"{path}:{name}: invalid data_offsets")
        begin, end = offsets
        expected = _product(shape) * DTYPE_BYTES[dtype]
        if begin < 0 or end < begin or end - begin != expected or data_start + end > file_size:
            raise NgramFormatError(f"{path}:{name}: invalid or out-of-bounds tensor span")
        tensors.append({
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "begin": begin,
            "absolute": data_start + begin,
            "bytes": end - begin,
        })
        spans.append((begin, end, name))
    spans.sort()
    for previous, current in zip(spans, spans[1:]):
        if current[0] < previous[1]:
            raise NgramFormatError(f"{path}: overlapping tensors {previous[2]} and {current[2]}")
    return tensors, metadata, data_start, file_size


def _scalar_text(metadata: dict[str, str], key: str) -> int:
    try:
        return int(metadata[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise NgramFormatError(f"metadata {key!r} is not an integer") from exc


class NgramReference:
    """Read and decode selected rows from an EXL3 n-gram sidecar."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        tensors, metadata, self.data_start, self.file_bytes = _load_header(self.path)
        if metadata.get("format") != "exl3_ngram_trellis":
            raise NgramFormatError(f"{self.path}: metadata format is not exl3_ngram_trellis")
        if metadata.get("codebook") != "mul1":
            raise NgramFormatError(f"{self.path}: n-gram trellis requires the mul1 codebook")
        self.metadata = metadata
        self.K = _scalar_text(metadata, "K")
        if not 1 <= self.K <= 8:
            raise NgramFormatError(f"metadata K={self.K} is outside 1..8")
        self.row_dim = _scalar_text(metadata, "row_dim")
        if self.row_dim != ROW_DIM:
            raise NgramFormatError(f"metadata row_dim={self.row_dim}, expected {ROW_DIM}")
        self._tensors = {item["name"]: item for item in tensors}
        trellis = []
        for name, item in self._tensors.items():
            marker = ".shard_"
            if not name.endswith(".trellis") or marker not in name:
                continue
            prefix, suffix = name.rsplit(marker, 1)
            if not suffix.endswith(".trellis"):
                continue
            try:
                shard_index = int(suffix[: -len(".trellis")])
            except ValueError:
                continue
            trellis.append((shard_index, prefix, item))
        if not trellis:
            raise NgramFormatError(f"{self.path}: no shard_N.trellis tensors")
        trellis.sort()
        if [x[0] for x in trellis] != list(range(len(trellis))):
            raise NgramFormatError(f"{self.path}: shard_N.trellis numbering is not contiguous")
        prefixes = {x[1] for x in trellis}
        if len(prefixes) != 1:
            raise NgramFormatError(f"{self.path}: shard trellises have different prefixes")
        self.prefix = trellis[0][1]
        self.shards: list[dict[str, Any]] = [x[2] for x in trellis]
        expected_words = words_per_row(self.K)
        for shard_info in self.shards:
            shape = shard_info["shape"]
            if shard_info["dtype"] != "I16" or len(shape) != 2 or shape[1] != expected_words or shape[0] <= 0:
                raise NgramFormatError(f"{self.path}:{shard_info['name']}: invalid trellis shard shape")
        self.rows_per_shard = self.shards[0]["shape"][0]
        if any(shard_info["shape"][0] != self.rows_per_shard for shard_info in self.shards[:-1]):
            raise NgramFormatError(f"{self.path}: all non-final shards must have equal row counts")
        self.shard_rows = [shard_info["shape"][0] for shard_info in self.shards]
        self.num_rows = sum(self.shard_rows)
        declared_rows = _scalar_text(metadata, "rows") if "rows" in metadata else self.num_rows
        if declared_rows != self.num_rows:
            raise NgramFormatError(f"{self.path}: metadata rows={declared_rows}, header rows={self.num_rows}")
        declared_shard_rows = _scalar_text(metadata, "shard_rows") if "shard_rows" in metadata else self.rows_per_shard
        if declared_shard_rows != self.rows_per_shard:
            raise NgramFormatError(f"{self.path}: metadata shard_rows disagrees with the header")
        self._cumulative = np.cumsum(self.shard_rows, dtype=np.int64)
        self.head_bias = self._small(f"{self.prefix}.head_bias", "F16", (None, ROW_DIM))
        self.head_offsets = self._small(f"{self.prefix}.head_offsets", "I64", (None,))
        self.head_vocab_sizes = self._small(f"{self.prefix}.head_vocab_sizes", "I64", (None,))
        self.layer_multipliers = self._small(f"{self.prefix}.layer_multipliers", "I64", (None,))
        if self.head_offsets.shape != self.head_vocab_sizes.shape:
            raise NgramFormatError(f"{self.path}: head offsets and sizes differ")
        if len(self.layer_multipliers) < 2:
            raise NgramFormatError(f"{self.path}: layer_multipliers is too short")
        if len(self.head_offsets) == 0 or self.head_offsets[0] != 0:
            raise NgramFormatError(f"{self.path}: head_offsets must start at zero")
        if np.any(self.head_offsets[1:] < self.head_offsets[:-1]):
            raise NgramFormatError(f"{self.path}: head_offsets are not sorted")
        if np.any(self.head_vocab_sizes <= 0):
            raise NgramFormatError(f"{self.path}: head_vocab_sizes must be positive")
        self.head_ends = self.head_offsets + self.head_vocab_sizes
        if np.any(self.head_offsets[1:] < self.head_ends[:-1]):
            raise NgramFormatError(f"{self.path}: head ranges overlap")
        if np.any(self.head_ends > self.num_rows):
            raise NgramFormatError(f"{self.path}: a head range exceeds the table")
        self.head_range_end = int(np.max(self.head_ends))
        self.unassigned_trailing_rows = self.num_rows - self.head_range_end
        self._codebook = _mul1_codebook()

    def _small(self, name: str, dtype: str, shape: tuple[int | None, ...]) -> np.ndarray:
        item = self._tensors.get(name)
        if item is None:
            raise NgramFormatError(f"{self.path}: missing {name}")
        actual = item["shape"]
        if item["dtype"] != dtype or len(actual) != len(shape) or any(
            expected is not None and actual[i] != expected for i, expected in enumerate(shape)
        ):
            raise NgramFormatError(f"{self.path}:{name}: unexpected dtype or shape")
        with self.path.open("rb") as f:
            f.seek(item["absolute"])
            raw = f.read(item["bytes"])
        if len(raw) != item["bytes"]:
            raise NgramFormatError(f"{self.path}:{name}: short read")
        np_dtype = {"F16": "<f2", "I64": "<i8"}[dtype]
        return np.frombuffer(raw, dtype=np_dtype).reshape(actual).copy()

    def read_rows(self, row_ids: Iterable[int]) -> np.ndarray:
        """Read selected packed rows in the caller's order, without loading the table."""
        ids = np.asarray(list(row_ids), dtype=np.int64)
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= self.num_rows):
            raise IndexError("row id outside the n-gram table")
        words = words_per_row(self.K)
        out = np.empty((len(ids), words), dtype="<i2")
        if not len(ids):
            return out
        shard_ids = np.searchsorted(self._cumulative, ids, side="right")
        starts = np.concatenate((np.array([0], dtype=np.int64), self._cumulative[:-1]))
        for shard_id in np.unique(shard_ids):
            output_positions = np.flatnonzero(shard_ids == shard_id)
            local = ids[output_positions] - starts[shard_id]
            order = np.argsort(local, kind="stable")
            sorted_positions = output_positions[order]
            sorted_local = local[order]
            run_start = 0
            while run_start < len(sorted_local):
                run_end = run_start + 1
                while run_end < len(sorted_local) and sorted_local[run_end] == sorted_local[run_end - 1] + 1:
                    run_end += 1
                first = int(sorted_local[run_start])
                count = run_end - run_start
                shard = self.shards[int(shard_id)]
                absolute = self.data_start + shard["begin"] + first * words * 2
                with self.path.open("rb") as f:
                    f.seek(absolute)
                    raw = f.read(count * words * 2)
                if len(raw) != count * words * 2:
                    raise NgramFormatError(f"{self.path}:{shard['name']}: short row read")
                rows = np.frombuffer(raw, dtype="<i2").reshape(count, words)
                out[sorted_positions[run_start:run_end]] = rows
                run_start = run_end
        return out

    def decode_rows(self, row_ids: Iterable[int], heads: Iterable[int] | None = None,
                    output_dtype: np.dtype = np.dtype(np.float32)) -> np.ndarray:
        """Decode selected rows with the exact mul1/fp16-codebook operation."""
        ids = np.asarray(list(row_ids), dtype=np.int64)
        packed = self.read_rows(ids)
        if heads is None:
            if np.any(ids >= self.head_range_end):
                raise NgramFormatError("a requested row has no head-bias range")
            head_ids = np.searchsorted(self.head_offsets, ids, side="right") - 1
            if np.any(ids < self.head_offsets[head_ids]) or np.any(ids >= self.head_ends[head_ids]):
                raise NgramFormatError("a requested row falls in a gap without head-bias metadata")
        else:
            head_ids = np.asarray(list(heads), dtype=np.int64)
            if len(head_ids) != len(ids):
                raise ValueError("heads and row_ids have different lengths")
        if np.any(head_ids < 0) or np.any(head_ids >= len(self.head_offsets)):
            raise IndexError("head id outside the n-gram metadata")
        states = _unpack_states(packed[:, 1:], self.K)
        scales = packed[:, 0].view(np.uint16).view(np.float16).astype(np.float32)
        out = self._codebook[states].astype(np.float32) * scales[:, None]
        out += self.head_bias[head_ids].astype(np.float32)
        return out.astype(output_dtype, copy=False)

    def lookup_ids(self, token_history: Iterable[int], ngram_size: int, heads_per_ngram: int,
                   eos_token_id: int) -> np.ndarray:
        """Return global row IDs for each output position and hash head."""
        tokens = np.asarray(list(token_history), dtype=np.int64)
        context = ngram_size - 1
        if ngram_size != len(self.layer_multipliers) or heads_per_ngram * context != len(self.head_offsets):
            raise ValueError("ngram metadata dimensions disagree")
        out_len = len(tokens) - context
        if out_len < 0:
            raise ValueError("token history is shorter than the n-gram context")
        result = np.empty((out_len, len(self.head_offsets)), dtype=np.int64)
        for out_index, position in enumerate(range(context, len(tokens))):
            previous_eos = -1
            for candidate in range(position):
                if tokens[candidate] == eos_token_id:
                    previous_eos = candidate
            mixed = (int(tokens[position]) * int(self.layer_multipliers[0])) & UINT64_MASK
            for s in range(1, ngram_size):
                source = position - s
                value = tokens[source] if source >= previous_eos + 1 and source >= 0 else eos_token_id
                mixed ^= (int(value) * int(self.layer_multipliers[s])) & UINT64_MASK
                lo = (s - 1) * heads_per_ngram
                hi = lo + heads_per_ngram
                signed_mixed = mixed if mixed < INT64_SIGN else mixed - (1 << 64)
                result[out_index, lo:hi] = (
                    signed_mixed % self.head_vocab_sizes[lo:hi] + self.head_offsets[lo:hi]
                )
        return result

    def embed_history(self, token_history: Iterable[int], ngram_size: int, heads_per_ngram: int,
                      eos_token_id: int, output_dtype: np.dtype = np.dtype(np.float32)) -> np.ndarray:
        """Hash, fetch, decode, and concatenate one token history's n-gram rows."""
        row_ids = self.lookup_ids(token_history, ngram_size, heads_per_ngram, eos_token_id)
        flat = row_ids.reshape(-1)
        unique, inverse = np.unique(flat, return_inverse=True)
        rows = self.decode_rows(unique, output_dtype=np.dtype(np.float32))
        return rows[inverse].reshape(row_ids.shape[0], row_ids.shape[1] * ROW_DIM).astype(output_dtype)


def _unpack_states(words: np.ndarray, k: int) -> np.ndarray:
    """Invert the ring packing in ngram_codec.unpack_rows for selected rows."""
    states = np.zeros((words.shape[0], ROW_DIM), dtype=np.uint16)
    words_u = words.view(np.uint16)
    for i in range(ROW_DIM):
        for m in range(16):
            position = (i - m // k) % ROW_DIM
            stream_bit = position * k + m % k
            bit = (words_u[:, stream_bit >> 4] >> (stream_bit & 15)) & 1
            states[:, i] |= (bit.astype(np.uint16) << m)
    return states


_CODEBOOK: np.ndarray | None = None


def _mul1_codebook() -> np.ndarray:
    global _CODEBOOK
    if _CODEBOOK is None:
        _CODEBOOK = decode_codebook(np.arange(65536, dtype=np.uint16), 2)
    return _CODEBOOK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decode selected rows from an EXL3 n-gram sidecar")
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("rows", nargs="+", type=int)
    parser.add_argument("--heads", nargs="*", type=int)
    args = parser.parse_args(argv)
    reader = NgramReference(args.sidecar)
    values = reader.decode_rows(args.rows, args.heads)
    print(json.dumps({"K": reader.K, "rows": args.rows, "shape": list(values.shape), "values": values.tolist()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
