# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Inspect an EXL3 safetensors directory without loading model weights.

This is an artifact inspector, not an EXL3 runtime.  It reads JSON and
safetensors headers, validates offsets and tensor geometry, and reports the
trellis, scale, bias, and codebook marker tensors belonging to each EXL3
linear.  Marker scalars are the only tensor bytes read after header parsing;
they are four-byte metadata values, never weight data.

The effective-bpw figure is deliberately scoped.  Its denominator is the sum
of ``in_features * out_features`` for recognized EXL3 linears.  It therefore
does not claim to be the model's source parameter count: embeddings, norms,
unquantized tensors, tied weights, and tensors in an unrecognized storage
layout are reported separately.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter
from bisect import bisect_left, bisect_right
from itertools import pairwise
from pathlib import Path
from typing import Any


MAX_HEADER_BYTES = 100 * 1024 * 1024
PRIMARY_K_MIN = 2
PRIMARY_K_MAX = 5
EXL3_K_MIN = 1
EXL3_K_MAX = 8

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "U32": 4,
    "I32": 4,
    "U64": 8,
    "I64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}
FLOAT_DTYPES = {"F8_E4M3", "F8_E5M2", "F16", "BF16", "F32", "F64"}
MARKER_DTYPES = {"I32", "U32"}
MARKERS = {
    "mcg": 0xCBAC1FED,
    "mul1": 0x83DCD12D,
}
COMPONENT_SUFFIXES = (".trellis", ".suh", ".svh", ".su", ".sv", ".mcg", ".mul1", ".bias")  # codespell:ignore suh


class InventoryError(ValueError):
    """A malformed or unsupported artifact."""


def _fail(message: str) -> InventoryError:
    return InventoryError(message)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f, object_pairs_hook=_reject_duplicate_pairs)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise _fail(f"{label}: cannot read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise _fail(f"{label}: top level must be an object")
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _product(shape: list[int]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result


def _safe_filename(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value in (".", ".."):
        raise _fail(f"{label}: safetensors filename is invalid")
    if value.startswith("/") or "\\" in value or "/" in value or "\x00" in value:
        raise _fail(f"{label}: safetensors filename must be a plain basename: {value!r}")
    return value


def _header(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise _fail(f"{path.name}: file is shorter than the safetensors header prefix")
            header_bytes = struct.unpack("<Q", prefix)[0]
            if not 0 < header_bytes <= MAX_HEADER_BYTES:
                raise _fail(f"{path.name}: invalid header size {header_bytes}")
            if 8 + header_bytes > file_size:
                raise _fail(f"{path.name}: header extends past the file")
            raw_header = f.read(header_bytes)
    except OSError as exc:
        raise _fail(f"{path.name}: cannot read file: {exc}") from exc

    if len(raw_header) != header_bytes:
        raise _fail(f"{path.name}: truncated header")
    try:
        text = raw_header.decode("utf-8")
        decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_pairs)
        value, end = decoder.raw_decode(text)
        if text[end:].strip():
            raise _fail(f"{path.name}: trailing bytes after the header JSON")
    except (UnicodeDecodeError, ValueError) as exc:
        raise _fail(f"{path.name}: malformed header JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise _fail(f"{path.name}: header must be an object")

    metadata = value.get("__metadata__", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()
    ):
        raise _fail(f"{path.name}: __metadata__ must be a string-to-string object")

    tensors: list[dict[str, Any]] = []
    spans: list[tuple[int, int, str]] = []
    data_start = 8 + header_bytes
    for name, entry in value.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not name:
            raise _fail(f"{path.name}: tensor name is invalid")
        if not isinstance(entry, dict):
            raise _fail(f"{path.name}:{name}: tensor entry must be an object")
        dtype = entry.get("dtype")
        if dtype not in DTYPE_BYTES:
            raise _fail(f"{path.name}:{name}: unsupported dtype {dtype!r}")
        shape = entry.get("shape")
        if not isinstance(shape, list) or not all(_is_int(x) and x >= 0 for x in shape):
            raise _fail(f"{path.name}:{name}: shape must be a list of nonnegative integers")
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(_is_int(x) for x in offsets):
            raise _fail(f"{path.name}:{name}: data_offsets must contain two integers")
        begin, end = offsets
        if begin < 0 or end < begin:
            raise _fail(f"{path.name}:{name}: invalid data_offsets {offsets!r}")
        payload_bytes = end - begin
        expected_bytes = _product(shape) * DTYPE_BYTES[dtype]
        if payload_bytes != expected_bytes:
            raise _fail(
                f"{path.name}:{name}: data_offsets span {payload_bytes} bytes, "
                f"but {dtype}{shape} requires {expected_bytes}"
            )
        if data_start + end > file_size:
            raise _fail(f"{path.name}:{name}: tensor data extends past the file")
        spans.append((begin, end, name))
        tensors.append(
            {
                "name": name,
                "file": path.name,
                "dtype": dtype,
                "shape": shape,
                "data_offset": begin,
                "data_offset_absolute": data_start + begin,
                "payload_bytes": payload_bytes,
                "storage_bytes": payload_bytes,
                "logical_elements": _product(shape),
            }
        )

    spans.sort()
    for previous, current in pairwise(spans):
        if current[0] < previous[1]:
            raise _fail(f"{path.name}: tensor data overlaps: {previous[2]} and {current[2]}")
    return tensors, {
        "header_bytes": header_bytes,
        "data_start": data_start,
        "file_bytes": file_size,
        "metadata": metadata,
    }


def _read_marker(model_dir: Path, tensor: dict[str, Any]) -> int:
    if tensor["dtype"] not in MARKER_DTYPES or tensor["shape"] not in ([], [1]) or tensor["payload_bytes"] != 4:
        raise _fail(f"{tensor['name']}: codebook marker must be one I32/U32 scalar")
    path = model_dir / tensor["file"]
    try:
        with path.open("rb") as f:
            f.seek(tensor["data_offset_absolute"])
            raw = f.read(4)
    except OSError as exc:
        raise _fail(f"{tensor['name']}: cannot read marker: {exc}") from exc
    if len(raw) != 4:
        raise _fail(f"{tensor['name']}: marker is truncated")
    return struct.unpack("<I", raw)[0]


def _declared_storage(qconfig: dict[str, Any], base: str) -> dict[str, Any] | None:
    storage = qconfig.get("tensor_storage")
    if not isinstance(storage, dict):
        return None
    value = storage.get(base)
    return value if isinstance(value, dict) else None


def _validate_declared(base: str, declared: dict[str, Any] | None, bits: int, codebook: str) -> dict[str, Any] | None:
    if declared is None:
        return None
    result: dict[str, Any] = {}
    quant_format = declared.get("quant_format")
    if quant_format is not None:
        if quant_format != "exl3":
            raise _fail(f"{base}: tensor_storage quant_format is {quant_format!r}, not 'exl3'")
        result["quant_format"] = quant_format
    declared_bits = declared.get("bits_per_weight")
    if declared_bits is not None:
        if (
            not isinstance(declared_bits, (int, float))
            or isinstance(declared_bits, bool)
            or int(declared_bits) != declared_bits
        ):
            raise _fail(f"{base}: bits_per_weight is not an integer")
        if int(declared_bits) != bits:
            raise _fail(f"{base}: config says {declared_bits} bits, trellis says {bits}")
        result["bits_per_weight"] = int(declared_bits)
    expected = {"mcg": MARKERS["mcg"], "mul1": MARKERS["mul1"]}
    for key, marker in expected.items():
        field = f"{key}_multiplier"
        value = declared.get(field)
        if value is None:
            continue
        if not isinstance(value, int) or value != marker:
            raise _fail(f"{base}: {field} is not the expected EXL3 marker")
        if codebook != key:
            raise _fail(f"{base}: {field} disagrees with tensor marker {codebook}")
        result[field] = value
    return result


def _expert_like(base: str) -> bool:
    lowered = base.lower()
    return ".experts." in lowered or ".expert." in lowered or "experts." in lowered


def _find_config(model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise _fail(f"{model_dir}: config.json is required")
    config = _json_file(config_path, "config.json")
    quant_path = model_dir / "quantization_config.json"
    quant = _json_file(quant_path, "quantization_config.json") if quant_path.is_file() else {}
    nested = config.get("quantization_config")
    if isinstance(nested, dict):
        merged = dict(nested)
        merged.update(quant)
        quant = merged
    return config, quant


def _files_from_index(model_dir: Path) -> tuple[list[Path], dict[str, str] | None, dict[str, Any]]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = _json_file(index_path, "model.safetensors.index.json")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map or not all(isinstance(k, str) for k in weight_map):
            raise _fail("model.safetensors.index.json: weight_map must be a nonempty object")
        files = sorted({_safe_filename(name, "model.safetensors.index.json") for name in weight_map.values()})
        paths = []
        for name in files:
            path = model_dir / name
            if not path.is_file():
                raise _fail(f"model.safetensors.index.json: missing shard {name}")
            paths.append(path)
        return paths, weight_map, index.get("metadata", {}) if isinstance(index.get("metadata", {}), dict) else {}

    single = model_dir / "model.safetensors"
    paths = [single] if single.is_file() else sorted(model_dir.glob("*.safetensors"))
    if not paths:
        raise _fail(f"{model_dir}: no safetensors file or model.safetensors.index.json")
    return paths, None, {}


def _sidecar_inventory(model_dir: Path, indexed_files: set[str]) -> list[dict[str, Any]]:
    """Inventory unindexed safetensors without treating their tensors as EXL3 linears."""
    sidecars: list[dict[str, Any]] = []
    for path in sorted(model_dir.glob("*.safetensors")):
        if path.name in indexed_files:
            continue
        tensors, header = _header(path)
        metadata = header["metadata"]
        artifact_format = metadata.get("format", "safetensors")
        if artifact_format == "exl3_ngram_trellis":
            role = "specialized_ngram_codec"
        elif "patch" in path.stem.lower():
            role = "override_sidecar"
        else:
            role = "unindexed_sidecar"
        sidecars.append(
            {
                "name": path.name,
                "file_bytes": path.stat().st_size,
                "format": artifact_format,
                "role": role,
                "override_precedence": "patch overrides base tensor with the same name"
                if role == "override_sidecar"
                else None,
                "metadata": metadata,
                "raw_tensor_count": len(tensors),
                "raw_tensor_payload_bytes": sum(t["payload_bytes"] for t in tensors),
                "tensor_names": sorted(t["name"] for t in tensors),
            }
        )
    return sidecars


def inventory_model(model_dir: str | Path, *, strict_metadata_size: bool = False) -> dict[str, Any]:
    """Return a JSON-serializable inventory for a model directory."""
    root = Path(model_dir).resolve()
    if not root.is_dir():
        raise _fail(f"model directory does not exist: {root}")
    config, qconfig = _find_config(root)
    paths, weight_map, index_metadata = _files_from_index(root)
    sidecars = _sidecar_inventory(root, {path.name for path in paths})
    all_tensors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        tensors, _ = _header(path)
        for tensor in tensors:
            if tensor["name"] in seen:
                raise _fail(f"duplicate tensor across shards: {tensor['name']}")
            seen.add(tensor["name"])
            all_tensors.append(tensor)
    raw_by_name = {tensor["name"]: tensor for tensor in all_tensors}
    raw_names = sorted(raw_by_name)
    if weight_map is not None:
        if set(weight_map) != set(raw_by_name):
            missing = sorted(set(weight_map) - set(raw_by_name))
            extra = sorted(set(raw_by_name) - set(weight_map))
            raise _fail(f"weight_map/header tensor mismatch: missing={missing[:3]} extra={extra[:3]}")
        for name, filename in weight_map.items():
            if raw_by_name[name]["file"] != filename:
                raise _fail(f"weight_map points {name} to {filename}, header is in {raw_by_name[name]['file']}")
    declared_total = index_metadata.get("total_size")
    payload_total = sum(t["payload_bytes"] for t in all_tensors)
    metadata_size_finding: str | None = None
    metadata_size_matches: bool | None = None
    declared_total_int: int | None = None
    if declared_total is not None:
        valid_total = (
            isinstance(declared_total, (int, float))
            and not isinstance(declared_total, bool)
            and int(declared_total) == declared_total
        )
        if not valid_total:
            metadata_size_finding = "model.safetensors.index.json: metadata.total_size is not an integer"
            if strict_metadata_size:
                raise _fail(metadata_size_finding)
        else:
            declared_total_int = int(declared_total)
            metadata_size_matches = declared_total_int == payload_total
            if not metadata_size_matches:
                metadata_size_finding = (
                    "model.safetensors.index.json: metadata.total_size "
                    f"{declared_total_int} != measured header payload {payload_total}"
                )
                if strict_metadata_size:
                    raise _fail(metadata_size_finding)

    descriptors: list[dict[str, Any]] = []
    component_names: set[str] = set()
    for name in sorted(raw_by_name):
        if not name.endswith(".trellis"):
            continue
        base = name[: -len(".trellis")]
        trellis = raw_by_name[name]
        if trellis["dtype"] != "I16" or len(trellis["shape"]) != 3:
            raise _fail(f"{name}: trellis must be a 3-D I16 tensor")
        tk, tn, packed = trellis["shape"]
        if not tk or not tn or packed % 16:
            raise _fail(f"{name}: trellis dimensions must be positive and the last dimension divisible by 16")
        bits = packed // 16
        if not EXL3_K_MIN <= bits <= EXL3_K_MAX:
            raise _fail(f"{name}: trellis K={bits} is outside the supported 1..8 range")
        in_features, out_features = tk * 16, tn * 16
        components: dict[str, dict[str, Any]] = {"trellis": trellis}
        for key in ("suh", "svh", "su", "sv", "mcg", "mul1", "bias"):  # codespell:ignore suh
            candidate = f"{base}.{key}"
            if candidate in raw_by_name:
                components[key] = raw_by_name[candidate]
                component_names.add(candidate)
        component_names.add(name)
        for side, expected in (("suh", in_features), ("svh", out_features)):  # codespell:ignore suh
            packed_side = "su" if side == "suh" else "sv"  # codespell:ignore suh
            if side in components and packed_side in components:
                raise _fail(f"{base}: both {side} and {packed_side} are present")
            actual = components.get(side) or components.get(packed_side)
            if actual is None:
                raise _fail(f"{base}: missing {side} or {packed_side}")
            if side in components:
                if actual["dtype"] not in FLOAT_DTYPES or actual["shape"] != [expected]:
                    raise _fail(f"{actual['name']}: {side} must be a floating tensor of shape [{expected}]")
            else:
                if actual["dtype"] not in {"I16", "U16"} or actual["shape"] != [expected // 16]:
                    raise _fail(f"{actual['name']}: {packed_side} must be a 16-bit tensor of shape [{expected // 16}]")
        if "bias" in components:
            bias = components["bias"]
            if bias["dtype"] not in FLOAT_DTYPES or bias["shape"] != [out_features]:
                raise _fail(f"{bias['name']}: bias must be floating and shape [{out_features}]")
        marker_kinds = [key for key in MARKERS if key in components]
        if len(marker_kinds) > 1:
            raise _fail(f"{base}: mcg and mul1 markers cannot both be present")
        codebook = "cb0"
        if marker_kinds:
            marker_kind = marker_kinds[0]
            marker = components[marker_kind]
            value = _read_marker(root, marker)
            if value != MARKERS[marker_kind]:
                raise _fail(f"{marker['name']}: marker 0x{value:08x} is not the {marker_kind} constant")
            codebook = f"cb{1 if marker_kind == 'mcg' else 2}"
        declared = _declared_storage(qconfig, base)
        declared_codebook = "mcg" if codebook == "cb1" else "mul1" if codebook == "cb2" else "cb0"
        declared_info = _validate_declared(base, declared, bits, declared_codebook)
        known = {f"{base}{suffix}" for suffix in COMPONENT_SUFFIXES}
        prefix = base + "."
        first = bisect_left(raw_names, prefix)
        last = bisect_right(raw_names, prefix + "\U0010ffff")
        extras = [n for n in raw_names[first:last] if n not in known]
        trellis_bytes = trellis["payload_bytes"]
        auxiliary_bytes = sum(value["payload_bytes"] for key, value in components.items() if key != "trellis")
        descriptor = {
            "name": base,
            "expert_like": _expert_like(base),
            "in_features": in_features,
            "out_features": out_features,
            "logical_weight_elements": in_features * out_features,
            "K": bits,
            "primary_K": PRIMARY_K_MIN <= bits <= PRIMARY_K_MAX,
            "codebook": codebook,
            "components": {key: value for key, value in sorted(components.items())},
            "trellis_payload_bytes": trellis_bytes,
            "auxiliary_payload_bytes": auxiliary_bytes,
            "tensor_payload_bytes": trellis_bytes + auxiliary_bytes,
            "storage_bytes": trellis_bytes + auxiliary_bytes,
            "extra_component_tensors": extras,
        }
        if declared_info is not None:
            descriptor["declared_quantization"] = declared_info
        descriptors.append(descriptor)

    for tensor in all_tensors:
        tensor["recognized_exl3_component"] = tensor["name"] in component_names
    descriptor_storage = sum(d["storage_bytes"] for d in descriptors)
    descriptor_elements = sum(d["logical_weight_elements"] for d in descriptors)
    raw_elements = sum(t["logical_elements"] for t in all_tensors)
    file_bytes = sum(path.stat().st_size for path in paths)
    sidecar_file_bytes = sum(sidecar["file_bytes"] for sidecar in sidecars)
    sidecar_payload_bytes = sum(sidecar["raw_tensor_payload_bytes"] for sidecar in sidecars)
    k_hist = Counter(str(d["K"]) for d in descriptors)
    cb_hist = Counter(d["codebook"] for d in descriptors)
    pair_hist = Counter(f"K{d['K']}/{d['codebook']}" for d in descriptors)
    text_config = config.get("text_config")
    geometry_source: dict[str, Any] = text_config if isinstance(text_config, dict) else config
    geometry_keys = (
        "model_type", "hidden_size", "moe_intermediate_size", "num_experts", "num_experts_per_tok",
        "num_hidden_layers", "vocab_size",
    )
    geometry = {key: geometry_source[key] for key in geometry_keys if key in geometry_source}
    findings = [metadata_size_finding] if metadata_size_finding else []
    findings.extend(
        "sidecar content not included in indexed linear accounting: "
        f"{sidecar['name']} (format={sidecar['format']})"
        for sidecar in sidecars
    )
    findings.extend(
        "override sidecar requires explicit precedence and is not merged by inventory: "
        f"{sidecar['name']}"
        for sidecar in sidecars
        if sidecar["role"] == "override_sidecar"
    )
    return {
        "model_dir": str(root),
        "safetensors_files": [{"name": path.name, "file_bytes": path.stat().st_size} for path in paths],
        "sidecar_files": sidecars,
        "config": {"geometry": geometry, "has_quantization_config": bool(qconfig)},
        "raw_tensors": sorted(all_tensors, key=lambda t: t["name"]),
        "linear_descriptors": descriptors,
        "histogram": {
            "K": dict(sorted(k_hist.items())),
            "codebook": dict(sorted(cb_hist.items())),
            "K_codebook": dict(sorted(pair_hist.items())),
        },
        "accounting": {
            "safetensors_file_bytes": file_bytes,
            "indexed_safetensors_file_bytes": file_bytes,
            "unindexed_sidecar_file_bytes": sidecar_file_bytes,
            "all_safetensors_file_bytes": file_bytes + sidecar_file_bytes,
            "unindexed_sidecar_tensor_payload_bytes": sidecar_payload_bytes,
            "raw_tensor_payload_bytes": payload_total,
            "raw_tensor_logical_elements": raw_elements,
            "recognized_exl3_storage_bytes": descriptor_storage,
            "recognized_exl3_logical_weight_elements": descriptor_elements,
            "recognized_exl3_effective_bpw": (
                descriptor_storage * 8.0 / descriptor_elements if descriptor_elements else None
            ),
            "raw_tensor_payload_bpw": payload_total * 8.0 / raw_elements if raw_elements else None,
            "denominator_definition": (
                "EXL3 bpw uses sum(in_features*out_features) over recognized .trellis linears in indexed weight_map "
                "tensors; unindexed sidecars are excluded. Raw tensor bpw uses sum(product(shape)) and is not a "
                "source parameter count"
            ),
            "source_parameter_count": config.get("num_parameters") if _is_int(config.get("num_parameters")) else None,
        },
        "scope": {
            "primary_K_range": [PRIMARY_K_MIN, PRIMARY_K_MAX],
            "accepted_K_range": [EXL3_K_MIN, EXL3_K_MAX],
            "weights_loaded": False,
            "runtime_support": False,
            "linear_accounting_scope": "indexed weight_map tensors only; unindexed sidecars are reported separately",
            "sidecars_reported_separately": bool(sidecars),
        },
        "index_metadata_size": {
            "declared_total_size": declared_total_int,
            "measured_tensor_payload_bytes": payload_total,
            "difference_bytes": (payload_total - declared_total_int) if declared_total_int is not None else None,
            "matches": metadata_size_matches,
            "advisory": True,
        },
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--strict-metadata-size",
        action="store_true",
        help="reject an index metadata.total_size mismatch instead of recording it as advisory",
    )
    args = parser.parse_args(argv)
    try:
        result = inventory_model(args.model_dir, strict_metadata_size=args.strict_metadata_size)
    except InventoryError as exc:
        print(f"exl3_inventory: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
