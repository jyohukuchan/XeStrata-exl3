# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Prepare a zero-tensor GGUF metadata bootstrap for a Qwen3.5 EXL3 archive.

The input is intentionally required to contain no tensors.  Metadata entries
are copied as raw GGUF bytes; only qwen35moe.block_count and
qwen35moe.nextn_predict_layers are changed or appended.  This keeps tokenizer
arrays and unrelated metadata values/types intact without opening weights.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import struct
from typing import Any

GGUF_MAGIC = 0x46554747
META_TYPES = {
    0: ("u8", 1), 1: ("i8", 1), 2: ("u16", 2), 3: ("i16", 2), 4: ("u32", 4), 5: ("i32", 4),
    6: ("f32", 4), 7: ("bool", 1), 8: ("string", None), 9: ("array", None), 10: ("u64", 8),
    11: ("i64", 8), 12: ("f64", 8),
}
SCALAR_FORMATS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
    10: "<Q", 11: "<q", 12: "<d",
}


class MetadataError(ValueError):
    """A malformed or incompatible metadata bootstrap input."""


@dataclasses.dataclass(frozen=True)
class Entry:
    key: str
    type_id: int
    value_start: int
    value_end: int
    raw_start: int
    raw_end: int
    value: Any


@dataclasses.dataclass
class Header:
    data: bytes
    version: int
    tensor_count: int
    entries: list[Entry]
    alignment: int
    data_start: int

    @property
    def metadata(self) -> dict[str, Any]:
        return {entry.key: entry.value for entry in self.entries}


class _Cursor:
    def __init__(self, data: bytes, offset: int = 0):
        self.data = data
        self.offset = offset

    def take(self, size: int) -> bytes:
        if size < 0 or self.offset > len(self.data) - size:
            raise MetadataError("GGUF header is truncated")
        result = self.data[self.offset:self.offset + size]
        self.offset += size
        return result

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        size = self.u64()
        try:
            return self.take(size).decode("utf-8")
        except UnicodeDecodeError as error:
            raise MetadataError("GGUF metadata contains invalid UTF-8") from error


def _decode_value(cursor: _Cursor, type_id: int) -> Any:
    if type_id in SCALAR_FORMATS:
        fmt = SCALAR_FORMATS[type_id]
        return struct.unpack(fmt, cursor.take(struct.calcsize(fmt)))[0]
    if type_id == 8:
        return cursor.string()
    if type_id != 9:
        raise MetadataError(f"unsupported GGUF metadata type {type_id}")
    element_type = cursor.u32()
    count = cursor.u64()
    if count > 10_000_000:
        raise MetadataError("GGUF metadata array is unreasonably large")
    return [_decode_value(cursor, element_type) for _ in range(count)]


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def read_header(path: pathlib.Path) -> Header:
    data = path.read_bytes()
    if len(data) < 24:
        raise MetadataError("GGUF file is truncated before its fixed header")
    magic, version, tensor_count, kv_count = struct.unpack_from("<IIQQ", data, 0)
    if magic != GGUF_MAGIC:
        raise MetadataError("input is not GGUF")
    if version != 3:
        raise MetadataError(f"GGUF version {version} is unsupported; expected v3")
    if kv_count > 1_000_000:
        raise MetadataError("GGUF metadata entry count is unreasonable")
    cursor = _Cursor(data, 24)
    entries: list[Entry] = []
    seen: set[str] = set()
    for _ in range(kv_count):
        raw_start = cursor.offset
        key = cursor.string()
        if key in seen:
            raise MetadataError(f"duplicate GGUF metadata key: {key}")
        seen.add(key)
        type_id = cursor.u32()
        value_start = cursor.offset
        value = _decode_value(cursor, type_id)
        entries.append(Entry(key, type_id, value_start, cursor.offset, raw_start, cursor.offset, value))
    tensor_directory_start = cursor.offset
    # Parse enough of every tensor directory entry to reject truncated input.
    # Nonzero tensor counts are rejected after this check; their payloads are
    # never inspected or copied.
    if tensor_count > 1_000_000:
        raise MetadataError("GGUF tensor count is unreasonable")
    for _ in range(tensor_count):
        cursor.string()
        dimensions = cursor.u32()
        if dimensions > 128:
            raise MetadataError("GGUF tensor has an unreasonable rank")
        cursor.take(8 * dimensions)
        cursor.u32()
        cursor.u64()
    metadata = {entry.key: entry.value for entry in entries}
    alignment = metadata.get("general.alignment", 32)
    if not isinstance(alignment, int) or alignment <= 0 or alignment > 1 << 20 or alignment & (alignment - 1):
        raise MetadataError("general.alignment must be a positive power of two")
    data_start = _align(cursor.offset, alignment)
    if tensor_count == 0 and len(data) > data_start:
        raise MetadataError("zero-tensor GGUF has trailing payload bytes")
    if tensor_count != 0:
        raise MetadataError("input metadata bootstrap contains tensors; refusing stale offsets")
    if data_start < tensor_directory_start:
        raise MetadataError("GGUF data start is before its tensor directory")
    return Header(data, version, tensor_count, entries, alignment, data_start)


def _scalar_bytes(type_id: int, value: int) -> bytes:
    if type_id not in (0, 1, 2, 3, 4, 5, 10, 11):
        raise MetadataError(f"metadata replacement requires an integer scalar, got type {type_id}")
    try:
        return struct.pack(SCALAR_FORMATS[type_id], value)
    except struct.error as error:
        raise MetadataError(f"metadata replacement does not fit type {type_id}") from error


def _config_geometry(config_path: pathlib.Path, include_mtp: bool) -> dict[str, Any]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MetadataError(f"cannot read config.json: {error}") from error
    if not isinstance(config, dict):
        raise MetadataError("config.json must contain an object")
    if config.get("model_type") not in ("qwen3_5_moe", "qwen35moe"):
        raise MetadataError("config model_type is not Qwen3.5 MoE")
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise MetadataError("config.json lacks text_config")
    if text.get("model_type") not in ("qwen3_5_moe_text", "qwen35moe"):
        raise MetadataError("config text_config model_type is not Qwen3.5 MoE")

    def integer(key: str, *, positive: bool = True) -> int:
        value = text.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise MetadataError(f"config text_config.{key} must be an integer")
        if positive and value <= 0:
            raise MetadataError(f"config text_config.{key} must be positive")
        return value

    trunk = integer("num_hidden_layers")
    mtp_layers = integer("mtp_num_hidden_layers", positive=False)
    if mtp_layers not in (0, 1):
        raise MetadataError("only zero or one MTP layer is supported")
    if bool(text.get("mtp_use_dedicated_embeddings", False)):
        raise MetadataError("dedicated MTP embeddings are unsupported")
    if include_mtp and mtp_layers != 1:
        raise MetadataError("--include-mtp requested but config has no one-layer MTP")
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != trunk or any(
        item not in ("linear_attention", "full_attention") for item in layer_types
    ):
        raise MetadataError("config layer_types does not describe the Qwen3.5 trunk")

    def optional_integer(key: str) -> int | None:
        value = text.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise MetadataError(f"config text_config.{key} must be an integer")
        return value

    hidden = integer("hidden_size")
    experts = integer("num_experts")
    experts_used = integer("num_experts_per_tok")
    moe_ff = integer("moe_intermediate_size")
    shared_ff = integer("shared_expert_intermediate_size")
    heads = integer("num_attention_heads")
    kv_heads = integer("num_key_value_heads")
    head_dim = integer("head_dim")
    context = integer("max_position_embeddings")
    full_interval = integer("full_attention_interval")
    vocab = integer("vocab_size")
    linear_key_heads = integer("linear_num_key_heads")
    linear_value_heads = integer("linear_num_value_heads")
    linear_key_dim = integer("linear_key_head_dim")
    linear_value_dim = integer("linear_value_head_dim")
    conv = integer("linear_conv_kernel_dim")
    eps = text.get("rms_norm_eps")
    rope = text.get("rope_parameters")
    if not isinstance(eps, (int, float)) or not math.isfinite(float(eps)):
        raise MetadataError("config text_config.rms_norm_eps must be finite")
    if not isinstance(rope, dict) or not isinstance(rope.get("rope_theta"), (int, float)):
        raise MetadataError("config rope_parameters.rope_theta is missing")
    return {
        "block_count": trunk + (1 if include_mtp else 0),
        "nextn_predict_layers": 1 if include_mtp else 0,
        "embedding_length": hidden,
        "context_length": context,
        "attention.head_count": heads,
        "attention.head_count_kv": kv_heads,
        "attention.key_length": head_dim,
        "attention.value_length": head_dim,
        "expert_count": experts,
        "expert_used_count": experts_used,
        "expert_feed_forward_length": moe_ff,
        "expert_shared_feed_forward_length": shared_ff,
        "full_attention_interval": full_interval,
        "ssm.conv_kernel": conv,
        "ssm.inner_size": linear_value_heads * linear_value_dim,
        "ssm.state_size": linear_key_dim,
        "ssm.time_step_rank": linear_value_heads,
        "ssm.group_count": linear_key_heads,
        "rope.dimension_count": int(round(head_dim * float(rope.get("partial_rotary_factor", 1.0)))),
        "rope.freq_base": int(round(float(rope["rope_theta"]))),
        "attention.layer_norm_rms_epsilon": float(eps),
        "vocab_size": vocab,
        "_trunk_block_count": trunk,
        "_mtp_layers": mtp_layers,
        "_optional_leading_dense": optional_integer("num_hidden_layers"),
    }


def _metadata_value(metadata: dict[str, Any], architecture: str, key: str) -> Any:
    full = f"{architecture}.{key}"
    if full not in metadata:
        raise MetadataError(f"input metadata is missing {full}")
    return metadata[full]


def _validate_geometry(header: Header, config_path: pathlib.Path, include_mtp: bool) -> tuple[int, int]:
    geometry = _config_geometry(config_path, include_mtp)
    metadata = header.metadata
    if metadata.get("general.architecture") != "qwen35moe":
        raise MetadataError("input GGUF architecture must be general.architecture=qwen35moe")
    for key, expected in geometry.items():
        if key.startswith("_") or key in ("vocab_size", "block_count", "nextn_predict_layers"):
            continue
        actual = _metadata_value(metadata, "qwen35moe", key)
        if isinstance(expected, float):
            if (not isinstance(actual, (int, float)) or
                    not math.isclose(float(actual), expected, rel_tol=2e-5, abs_tol=1e-12)):
                raise MetadataError(f"metadata qwen35moe.{key} disagrees with config ({actual!r} != {expected!r})")
        elif actual != expected:
            raise MetadataError(f"metadata qwen35moe.{key} disagrees with config ({actual!r} != {expected!r})")
    source_block_count = _metadata_value(metadata, "qwen35moe", "block_count")
    allowed_blocks = {geometry["_trunk_block_count"]}
    if geometry["_mtp_layers"] == 1:
        allowed_blocks.add(geometry["_trunk_block_count"] + 1)
    if source_block_count not in allowed_blocks:
        raise MetadataError(f"metadata qwen35moe.block_count disagrees with config ({source_block_count!r})")
    source_nextn = metadata.get("qwen35moe.nextn_predict_layers", 0)
    if source_nextn not in (0, 1) or source_nextn != source_block_count - geometry["_trunk_block_count"]:
        raise MetadataError("metadata qwen35moe.nextn_predict_layers is incompatible with config")
    return geometry["block_count"], geometry["nextn_predict_layers"]


def prepare_metadata(input_path: pathlib.Path, exl3_dir: pathlib.Path, output_path: pathlib.Path,
                     include_mtp: bool = False) -> Header:
    header = read_header(input_path)
    block_count, nextn = _validate_geometry(header, exl3_dir / "config.json", include_mtp)
    replacements = {
        "qwen35moe.block_count": block_count,
        "qwen35moe.nextn_predict_layers": nextn,
    }
    entries_by_key = {entry.key: entry for entry in header.entries}
    if "qwen35moe.block_count" not in entries_by_key:
        raise MetadataError("input metadata is missing qwen35moe.block_count")
    if entries_by_key["qwen35moe.block_count"].type_id not in (4, 10):
        raise MetadataError("qwen35moe.block_count is not an unsigned integer scalar")
    if ("qwen35moe.nextn_predict_layers" in entries_by_key and
            entries_by_key["qwen35moe.nextn_predict_layers"].type_id not in (4, 10)):
        raise MetadataError("qwen35moe.nextn_predict_layers is not an unsigned integer scalar")

    chunks: list[bytes] = []
    for entry in header.entries:
        raw = header.data[entry.raw_start:entry.raw_end]
        if entry.key in replacements:
            replacement = _scalar_bytes(entry.type_id, replacements[entry.key])
            raw = raw[:entry.value_start - entry.raw_start] + replacement + raw[entry.value_end - entry.raw_start:]
        chunks.append(raw)
    if "qwen35moe.nextn_predict_layers" not in entries_by_key:
        key = b"qwen35moe.nextn_predict_layers"
        chunks.append(struct.pack("<Q", len(key)) + key + struct.pack("<I", 4) + struct.pack("<I", nextn))

    fixed = struct.pack("<IIQQ", GGUF_MAGIC, 3, 0, len(chunks)) + b"".join(chunks)
    fixed += b"\0" * ((_align(len(fixed), header.alignment)) - len(fixed))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(fixed)
    # Re-read the result, so callers never receive a path whose header was not
    # accepted by the same strict parser.
    return read_header(output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=pathlib.Path)
    parser.add_argument("--input", dest="input_option", type=pathlib.Path)
    parser.add_argument("--exl3", "--exl3-dir", dest="exl3_dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", "--out", dest="output", required=True, type=pathlib.Path)
    parser.add_argument("--include-mtp", action="store_true")
    args = parser.parse_args(argv)
    input_path = args.input_option or args.input
    if input_path is None:
        parser.error("an input metadata GGUF is required")
    try:
        output = prepare_metadata(input_path, args.exl3_dir, args.output, args.include_mtp)
        print(f"wrote {args.output} tensors={output.tensor_count} "
              f"block_count={output.metadata['qwen35moe.block_count']} "
              f"nextn_predict_layers={output.metadata['qwen35moe.nextn_predict_layers']}")
        return 0
    except (OSError, MetadataError) as error:
        print(f"exl3_metadata: {error}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
