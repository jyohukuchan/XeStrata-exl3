# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for tools/exl3_inventory.py using tiny synthetic safetensors shards."""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
EI = import_module("exl3_inventory")


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, list[int], bytes]],
    metadata: dict[str, str] | None = None,
) -> int:
    header: dict[str, object] = {"__metadata__": metadata} if metadata is not None else {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        begin = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [begin, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return len(payload)


def tensor(dtype: str, shape: list[int], fill: int = 0) -> tuple[str, list[int], bytes]:
    size = EI.DTYPE_BYTES[dtype]
    count = 1
    for value in shape:
        count *= value
    return dtype, shape, bytes([fill & 0xFF]) * (count * size)


class Exl3InventoryTests(unittest.TestCase):
    def make_model(self) -> Path:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        config = {
            "model_type": "qwen4_exp",
            "hidden_size": 2560,
            "moe_intermediate_size": 640,
            "num_experts": 512,
            "num_experts_per_tok": 10,
        }
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        shard1 = {
            "model.layers.0.mlp.experts.0.gate_proj.trellis": tensor("I16", [8, 4, 32], 1),
            "model.layers.0.mlp.experts.0.gate_proj.suh": tensor("F16", [128], 2),  # codespell:ignore suh
            "model.layers.0.mlp.experts.0.gate_proj.svh": tensor("F16", [64], 3),
            "model.layers.0.mlp.experts.0.gate_proj.mcg": ("I32", [], struct.pack("<I", 0xCBAC1FED)),
        }
        shard2 = {
            "model.layers.0.mlp.experts.1.up_proj.trellis": tensor("I16", [8, 4, 80], 4),
            "model.layers.0.mlp.experts.1.up_proj.su": tensor("I16", [8], 5),
            "model.layers.0.mlp.experts.1.up_proj.sv": tensor("I16", [4], 6),
            "model.layers.0.mlp.experts.1.up_proj.mul1": ("I32", [1], struct.pack("<I", 0x83DCD12D)),
            "model.layers.0.mlp.experts.1.up_proj.bias": tensor("F16", [64], 7),
            "model.embed_tokens.weight": tensor("BF16", [4, 128], 8),
        }
        size1 = write_safetensors(root / "model-00001-of-00002.safetensors", shard1)
        size2 = write_safetensors(root / "model-00002-of-00002.safetensors", shard2)
        weight_map = {name: "model-00001-of-00002.safetensors" for name in shard1}
        weight_map.update({name: "model-00002-of-00002.safetensors" for name in shard2})
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": size1 + size2}, "weight_map": weight_map}), encoding="utf-8"
        )
        write_safetensors(
            root / "ngram_embedding.safetensors",
            {"model.layers.0.ple.ngram.shard_0.trellis": tensor("I16", [2, 51], 9)},
            {"format": "exl3_ngram_trellis", "K": "5", "row_dim": "160"},
        )
        (root / "quantization_config.json").write_text(
            json.dumps(
                {
                    "tensor_storage": {
                        "model.layers.0.mlp.experts.0.gate_proj": {
                            "quant_format": "exl3",
                            "bits_per_weight": 2,
                            "mcg_multiplier": 0xCBAC1FED,
                        },
                        "model.layers.0.mlp.experts.1.up_proj": {
                            "quant_format": "exl3",
                            "bits_per_weight": 5,
                            "mul1_multiplier": 0x83DCD12D,
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        return root

    def tearDown(self):
        if hasattr(self, "temp"):
            self.temp.cleanup()

    def test_mixed_bits_and_codebooks_across_shards(self):
        result = EI.inventory_model(self.make_model())
        self.assertEqual(len(result["linear_descriptors"]), 2)
        self.assertEqual(result["histogram"]["K"], {"2": 1, "5": 1})
        self.assertEqual(result["histogram"]["codebook"], {"cb1": 1, "cb2": 1})
        by_name = {x["name"]: x for x in result["linear_descriptors"]}
        first = by_name["model.layers.0.mlp.experts.0.gate_proj"]
        second = by_name["model.layers.0.mlp.experts.1.up_proj"]
        self.assertEqual((first["in_features"], first["out_features"], first["K"]), (128, 64, 2))
        self.assertEqual((second["in_features"], second["out_features"], second["K"]), (128, 64, 5))
        self.assertIn("su", second["components"])
        self.assertIn("bias", second["components"])
        self.assertEqual(first["declared_quantization"]["bits_per_weight"], 2)
        self.assertGreater(result["accounting"]["recognized_exl3_effective_bpw"], 0)
        self.assertEqual(len(result["raw_tensors"]), 10)
        self.assertFalse(result["scope"]["weights_loaded"])

    def test_truncated_shard_is_rejected(self):
        root = self.make_model()
        path = root / "model-00002-of-00002.safetensors"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(EI.InventoryError):
            EI.inventory_model(root)

    def test_wrong_codebook_marker_is_rejected(self):
        root = self.make_model()
        path = root / "model-00001-of-00002.safetensors"
        raw = bytearray(path.read_bytes())
        header_size = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + header_size])
        offset = header["model.layers.0.mlp.experts.0.gate_proj.mcg"]["data_offsets"][0]
        raw[8 + header_size + offset:8 + header_size + offset + 4] = struct.pack("<I", 0x12345678)
        path.write_bytes(raw)
        with self.assertRaises(EI.InventoryError):
            EI.inventory_model(root)

    def test_duplicate_json_key_is_rejected(self):
        root = self.make_model()
        (root / "model.safetensors.index.json").write_text(
            '{"weight_map": {}, "weight_map": {}}', encoding="utf-8"
        )
        with self.assertRaises(EI.InventoryError):
            EI.inventory_model(root)

    def test_stale_total_size_is_advisory_by_default(self):
        root = self.make_model()
        index_path = root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["metadata"]["total_size"] += 123
        index_path.write_text(json.dumps(index), encoding="utf-8")
        result = EI.inventory_model(root)
        self.assertFalse(result["index_metadata_size"]["matches"])
        self.assertEqual(result["index_metadata_size"]["difference_bytes"], -123)
        self.assertTrue(any("metadata.total_size" in finding for finding in result["findings"]))
        with self.assertRaises(EI.InventoryError):
            EI.inventory_model(root, strict_metadata_size=True)

    def test_unindexed_ngram_sidecar_is_reported_separately(self):
        result = EI.inventory_model(self.make_model())
        self.assertEqual(len(result["linear_descriptors"]), 2)
        self.assertEqual(len(result["sidecar_files"]), 1)
        sidecar = result["sidecar_files"][0]
        self.assertEqual(sidecar["format"], "exl3_ngram_trellis")
        self.assertIn("model.layers.0.ple.ngram.shard_0.trellis", sidecar["tensor_names"])
        self.assertGreater(result["accounting"]["unindexed_sidecar_file_bytes"], 0)
        self.assertTrue(any("sidecar content not included" in finding for finding in result["findings"]))


if __name__ == "__main__":
    unittest.main()
