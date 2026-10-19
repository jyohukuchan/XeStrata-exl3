# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for tools.exl3_metadata (CPU-only; no model weights or GPU)."""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from exl3_metadata import MetadataError, prepare_metadata, read_header  # noqa: E402


def scalar(type_id: int, value) -> bytes:
    formats = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
               10: "<Q", 11: "<q", 12: "<d"}
    return struct.pack(formats[type_id], value)


def string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def array_string(values: list[str]) -> bytes:
    return struct.pack("<IQ", 8, len(values)) + b"".join(string(v) for v in values)


def entry(key: str, type_id: int, value: bytes) -> bytes:
    return string(key) + struct.pack("<I", type_id) + value


def config(dedicated: bool = False, mtp_layers: int = 1) -> dict:
    text = {
        "model_type": "qwen3_5_moe_text", "hidden_size": 2048, "num_hidden_layers": 40,
        "mtp_num_hidden_layers": mtp_layers, "mtp_use_dedicated_embeddings": dedicated,
        "num_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512, "num_attention_heads": 16, "num_key_value_heads": 2,
        "head_dim": 256, "max_position_embeddings": 262144, "full_attention_interval": 4,
        "vocab_size": 248320, "linear_num_key_heads": 16, "linear_num_value_heads": 32,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
        "rms_norm_eps": 1e-6,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 10,
        "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25},
    }
    return {"model_type": "qwen3_5_moe", "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": text}


def make_metadata(path: Path, block_count: int = 40, nextn: int | None = None, tensors: bool = False) -> None:
    values = [
        entry("general.architecture", 8, string("qwen35moe")),
        entry("general.alignment", 4, scalar(4, 32)),
        entry("general.name", 8, string("metadata exactness fixture")),
        entry("tokenizer.ggml.tokens", 9, array_string(["a", "b", "c"])),
        entry("qwen35moe.block_count", 4, scalar(4, block_count)),
        entry("qwen35moe.embedding_length", 4, scalar(4, 2048)),
        entry("qwen35moe.context_length", 4, scalar(4, 262144)),
        entry("qwen35moe.attention.head_count", 4, scalar(4, 16)),
        entry("qwen35moe.attention.head_count_kv", 4, scalar(4, 2)),
        entry("qwen35moe.attention.key_length", 4, scalar(4, 256)),
        entry("qwen35moe.attention.value_length", 4, scalar(4, 256)),
        entry("qwen35moe.attention.layer_norm_rms_epsilon", 6, scalar(6, 1e-6)),
        entry("qwen35moe.expert_count", 4, scalar(4, 256)),
        entry("qwen35moe.expert_used_count", 4, scalar(4, 8)),
        entry("qwen35moe.expert_feed_forward_length", 4, scalar(4, 512)),
        entry("qwen35moe.expert_shared_feed_forward_length", 4, scalar(4, 512)),
        entry("qwen35moe.full_attention_interval", 4, scalar(4, 4)),
        entry("qwen35moe.ssm.conv_kernel", 4, scalar(4, 4)),
        entry("qwen35moe.ssm.inner_size", 4, scalar(4, 4096)),
        entry("qwen35moe.ssm.state_size", 4, scalar(4, 128)),
        entry("qwen35moe.ssm.time_step_rank", 4, scalar(4, 32)),
        entry("qwen35moe.ssm.group_count", 4, scalar(4, 16)),
        entry("qwen35moe.rope.dimension_count", 4, scalar(4, 64)),
        entry("qwen35moe.rope.freq_base", 6, scalar(6, 10000000.0)),
    ]
    if nextn is not None:
        values.append(entry("qwen35moe.nextn_predict_layers", 4, scalar(4, nextn)))
    body = bytearray(struct.pack("<IIQQ", 0x46554747, 3, 1 if tensors else 0, len(values)))
    body += b"".join(values)
    if tensors:
        body += string("dummy") + struct.pack("<I", 0) + struct.pack("<IQ", 0, 0)
    body += b"\0" * ((-len(body)) % 32)
    path.write_bytes(body)


class MetadataToolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps(config()), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_preserves_unrelated_raw_entries_and_sets_mtp(self):
        source = self.root / "base.gguf"
        output = self.root / "mtp.gguf"
        make_metadata(source)
        prepare_metadata(source, self.model, output, include_mtp=True)
        before, after = read_header(source), read_header(output)
        self.assertEqual(after.tensor_count, 0)
        self.assertEqual(after.metadata["qwen35moe.block_count"], 41)
        self.assertEqual(after.metadata["qwen35moe.nextn_predict_layers"], 1)
        before_raw = {entry.key: before.data[entry.raw_start:entry.raw_end] for entry in before.entries}
        after_raw = {entry.key: after.data[entry.raw_start:entry.raw_end] for entry in after.entries}
        for key in ("general.architecture", "general.name", "tokenizer.ggml.tokens"):
            self.assertEqual(after_raw[key], before_raw[key], key)

    def test_rejects_nonzero_and_truncated_inputs(self):
        nonzero = self.root / "nonzero.gguf"
        make_metadata(nonzero, tensors=True)
        with self.assertRaises(MetadataError):
            prepare_metadata(nonzero, self.model, self.root / "out.gguf")
        truncated = self.root / "truncated.gguf"
        truncated.write_bytes(b"GGUF")
        with self.assertRaises(MetadataError):
            prepare_metadata(truncated, self.model, self.root / "out2.gguf")

    def test_rejects_unsupported_mtp_configuration(self):
        dedicated = self.root / "dedicated"
        dedicated.mkdir()
        (dedicated / "config.json").write_text(json.dumps(config(dedicated=True)), encoding="utf-8")
        source = self.root / "base.gguf"
        make_metadata(source)
        with self.assertRaises(MetadataError):
            prepare_metadata(source, dedicated, self.root / "dedicated.gguf", include_mtp=True)
        no_mtp = self.root / "no_mtp"
        no_mtp.mkdir()
        (no_mtp / "config.json").write_text(json.dumps(config(mtp_layers=0)), encoding="utf-8")
        with self.assertRaises(MetadataError):
            prepare_metadata(source, no_mtp, self.root / "no_mtp.gguf", include_mtp=True)

    def test_rejects_changed_norm_epsilon(self):
        source = self.root / "wrong-epsilon.gguf"
        make_metadata(source)
        header = read_header(source)
        epsilon = next(entry for entry in header.entries
                       if entry.key == "qwen35moe.attention.layer_norm_rms_epsilon")
        data = bytearray(header.data)
        struct.pack_into("<f", data, epsilon.value_start, 0.0)
        source.write_bytes(data)
        with self.assertRaises(MetadataError):
            prepare_metadata(source, self.model, self.root / "out.gguf")

    def test_real_zero_tensor_bootstrap_roundtrip(self):
        real_base = os.environ.get("STRATA_EXL3_METADATA_TEST_INPUT")
        real_config = os.environ.get("STRATA_EXL3_METADATA_TEST_MODEL")
        if not real_base or not real_config:
            self.skipTest("set STRATA_EXL3_METADATA_TEST_INPUT and STRATA_EXL3_METADATA_TEST_MODEL")
        real_base_path = Path(real_base)
        real_config_path = Path(real_config)
        if not real_base_path.is_file() or not (real_config_path / "config.json").is_file():
            self.skipTest("real metadata bootstrap paths are unavailable")
        output = self.root / "real-mtp.gguf"
        prepare_metadata(real_base_path, real_config_path, output, include_mtp=True)
        result = read_header(output)
        self.assertEqual(result.tensor_count, 0)
        self.assertEqual(result.metadata["qwen35moe.block_count"], 41)
        self.assertEqual(result.metadata["qwen35moe.nextn_predict_layers"], 1)


if __name__ == "__main__":
    unittest.main()
