#!/usr/bin/env python3
"""Offline MoE extraction checks using only tiny, synthetic local files.

No model execution, checkpoint access, backend initialization or network is
needed. Run this file directly with Python from any working directory.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay = load_module("pmi_moe_replay_extractor", "extract_moe_replay.py")
fixture = load_module("pmi_moe_replay_fixture", "make_qwen_fixture.py")
headers = load_module("pmi_moe_replay_test_headers", "inspect_remote_headers.py")

LAYER = 1
EVALUATION = 7
EXPERT_IDS = [3, 1]
PRIVATE_PROMPT = "PRIVATE-PROMPT-must-not-appear-in-replay-provenance"
PRIVATE_TOKEN = 987654321
LABELS = ("ffn_input", "ffn_moe_logits", "ffn_moe_topk",
          "ffn_moe_weights_norm", "ffn_moe_out", "ffn_out")
WEIGHTS = {
    "router.weight": "ffn_gate_inp.weight",
    "gate.weight": "ffn_gate_exps.weight",
    "up.weight": "ffn_up_exps.weight",
    "down.weight": "ffn_down_exps.weight",
    "shared_router.weight": "ffn_gate_inp_shexp.weight",
    "shared_gate.weight": "ffn_gate_shexp.weight",
    "shared_up.weight": "ffn_up_shexp.weight",
    "shared_down.weight": "ffn_down_shexp.weight",
}
ROUTED_WEIGHTS = {"gate.weight", "up.weight", "down.weight"}
SNAPSHOTS = {
    "replay.input": ("reference", "ffn_input"),
    "replay.ids": ("reference", "ffn_moe_topk"),
    "replay.logits": ("reference", "ffn_moe_logits"),
    "replay.weights": ("reference", "ffn_moe_weights_norm"),
    "replay.reference_routed": ("reference", "ffn_moe_out"),
    "replay.trial_routed": ("trial", "ffn_moe_out"),
    "replay.reference_output": ("reference", "ffn_out"),
    "replay.trial_output": ("trial", "ffn_out"),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class BytesRangeReader(headers.RangeReader):
    """Independent in-memory source for the existing GGUF header parser."""

    def __init__(self, raw):
        self.raw = raw
        super().__init__("", len(raw), headers.Budget(headers.MAX_HEADER_BYTES), opener=None)

    def fetch(self, end):
        start = len(self.data)
        if not start < end <= self.file_size:
            raise AssertionError("out-of-file read in test header parser")
        self.budget.reserve(end - start)
        self.data.extend(self.raw[start:end])
        self.budget.received += end - start


def inspect_file(path):
    raw = path.read_bytes()
    header = headers.parse_header(BytesRangeReader(raw))
    return raw, header, {tensor["name"]: tensor for tensor in header["tensors"]}


def trace(trial=False):
    values = {
        "ffn_input": [(index - 16) / 32 for index in range(32)],
        "ffn_moe_logits": [-0.25, 0.5, -0.5, 1.0],
        "ffn_moe_topk": EXPERT_IDS[:],
        "ffn_moe_weights_norm": [0.625, 0.375],
        "ffn_moe_out": [index / 64 + (1 / 128 if trial else 0) for index in range(32)],
        "ffn_out": [index / 64 + (0.25 if trial else 0.125) for index in range(32)],
    }
    return {"prompt": PRIVATE_PROMPT, "tokens": [PRIVATE_TOKEN],
            "metrics": {"prompt_tokens": 12345, "private_path": "/private/raw-prompt.txt"},
            "records": [{"evaluation": EVALUATION, "phase": "prefill", "name": f"{label}-{LAYER}",
                         "type": "i32" if label == "ffn_moe_topk" else "f32",
                         "shape": [len(data), 1, 1, 1], "values": data}
                        for label, data in values.items()]}


def record(document, label):
    rows = [row for row in document["records"]
            if row["name"] == f"{label}-{LAYER}" and row["evaluation"] == EVALUATION]
    if len(rows) != 1:
        raise AssertionError("test expected exactly one selected trace record")
    return rows[0]


class MoEReplayExtractionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pmi-moe-replay-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "selected.gguf"
        self.sidecar = self.output.with_suffix(".provenance.json")
        self.reference_path = self.root / "private-reference-trace.json"
        self.trial_path = self.root / "private-trial-trace.json"
        self.reference, self.trial = trace(), trace(trial=True)
        self.enterContext(mock.patch.object(socket, "create_connection", side_effect=AssertionError("network forbidden")))
        self.enterContext(mock.patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")))
        self.make_shards()
        self.write_traces()

    def make_shards(self, expert_type="q8_0", scale=None):
        metadata, tensors = fixture.fixture(expert_type=expert_type)
        if scale is not None:
            metadata.append(("qwen4exp.expert_weights_scale", 6, scale))
        self.source_tensors = {name: (shape, kind, data) for name, shape, kind, data in tensors}
        self.shards = []
        for index in range(2):
            path = self.root / f"synthetic-source-{index + 1:05d}-of-00002.gguf"
            split = [("split.no", 2, index), ("split.count", 2, 2), ("split.tensors.count", 5, len(tensors))]
            fixture.write_gguf(path, metadata + split, tensors[index::2])
            self.shards.append(path)

    def write_traces(self):
        for path, value in ((self.reference_path, self.reference), (self.trial_path, self.trial)):
            # allow_nan deliberately permits negative-test fixtures containing
            # nonfinite JSON constants; extraction must reject them.
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    def extract(self, shards=None):
        return replay.extract(self.shards[::-1] if shards is None else shards,
                              self.reference_path, self.trial_path, LAYER, EVALUATION, self.output)

    def reject(self, shards=None):
        self.write_traces()
        with self.assertRaises(replay.ReplayError):
            self.extract(shards)
        self.assertFalse(self.output.exists(), "failed extraction published a GGUF")
        self.assertFalse(self.sidecar.exists(), "failed extraction published provenance")

    def assert_bundle(self, provenance, scale=1.0):
        raw, header, output_tensors = inspect_file(self.output)
        self.assertEqual(header["metadata"], {
            "general.architecture": "pmi_moe_replay", "general.alignment": 32,
            "replay.version": 1, "replay.layer": LAYER, "replay.evaluation": EVALUATION,
            "replay.expert_weights_scale": scale,
        })
        self.assertEqual(set(output_tensors), set(WEIGHTS) | set(SNAPSHOTS))
        self.assertEqual(header["tensor_count"], 16)
        self.assertEqual(json.loads(self.sidecar.read_text()), provenance)
        self.assertEqual(provenance["schema_version"], 1)
        self.assertEqual(provenance["scope"], "pmi_moe_replay_extraction")
        self.assertEqual(provenance["layer"], LAYER)
        self.assertEqual(provenance["evaluation"], EVALUATION)
        self.assertEqual(provenance["expert_ids"], EXPERT_IDS)
        self.assertEqual(provenance["fixture"], {"name": self.output.name, "bytes": len(raw), "sha256": sha256(raw)})
        self.assertEqual(provenance["traces"], {
            label: {"bytes": path.stat().st_size, "sha256": sha256(path.read_bytes())}
            for label, path in (("reference", self.reference_path), ("trial", self.trial_path))
        })

        sources, expected_shards = {}, []
        for index, path in enumerate(self.shards):
            source_raw, source_header, source_tensors = inspect_file(path)
            sources.update({name: (index, tensor, source_raw) for name, tensor in source_tensors.items()})
            expected_shards.append({"shard_index": index, "file_bytes": len(source_raw),
                                    "header_bytes": source_header["header_bytes"],
                                    "header_sha256": sha256(source_raw[:source_header["header_bytes"]])})
        self.assertEqual(provenance["shards"], expected_shards)
        expected_ranges = []
        expected_source_bytes = 0
        for output_name, suffix in WEIGHTS.items():
            source_name = f"blk.{LAYER}.{suffix}"
            shape, kind, payload = self.source_tensors[source_name]
            shard_index, source_tensor, source_raw = sources[source_name]
            routed = output_name in ROUTED_WEIGHTS
            selected = EXPERT_IDS if routed else [None]
            stride = len(payload) // 4 if routed else len(payload)
            pieces = []
            for expert_id in selected:
                relative = expert_id * stride if routed else 0
                piece = payload[relative:relative + stride]
                absolute = source_tensor["absolute_offset"] + relative
                self.assertEqual(source_raw[absolute:absolute + stride], piece)
                pieces.append(piece)
                item = {"output_tensor": output_name, "source_tensor": source_name,
                        "shard_index": shard_index, "source_offset": absolute,
                        "bytes": stride, "sha256": sha256(piece)}
                if routed:
                    item["expert_id"] = expert_id
                expected_ranges.append(item)
            expected_payload = b"".join(pieces)
            expected_shape = list(shape[:-1]) + [2] if routed else list(shape)
            if output_name == "shared_router.weight":
                expected_shape = [32, 1]
            tensor = output_tensors[output_name]
            self.assertEqual(tensor["shape"], expected_shape, output_name)
            self.assertEqual(tensor["ggml_type"], kind, output_name)
            self.assertEqual(tensor["bytes"], len(expected_payload), output_name)
            start = tensor["absolute_offset"]
            self.assertEqual(raw[start:start + tensor["bytes"]], expected_payload, output_name)
            expected_source_bytes += len(expected_payload)
        self.assertEqual(provenance["source_payload_bytes"], expected_source_bytes)
        self.assertEqual(len(provenance["ranges"]), len(expected_ranges))
        key = lambda item: (item["output_tensor"], item.get("expert_id", -1))
        self.assertEqual(sorted(provenance["ranges"], key=key), sorted(expected_ranges, key=key))
        for name in ROUTED_WEIGHTS:
            self.assertEqual([item["expert_id"] for item in provenance["ranges"] if item["output_tensor"] == name],
                             EXPERT_IDS, "packed expert order must retain global ID order")
        self.assertEqual({item["shard_index"] for item in expected_ranges}, {0, 1})

        for name, (side, label) in SNAPSHOTS.items():
            values = record(getattr(self, side), label)["values"]
            integer = label == "ffn_moe_topk"
            expected_payload = struct.pack("<" + ("i" if integer else "f") * len(values), *values)
            tensor = output_tensors[name]
            self.assertEqual(tensor["shape"], [len(values)], name)
            self.assertEqual(tensor["ggml_type"], 26 if integer else 0, name)
            start = tensor["absolute_offset"]
            self.assertEqual(raw[start:start + tensor["bytes"]], expected_payload, name)
        self.assertNotEqual(record(self.reference, "ffn_moe_out")["values"], record(self.trial, "ffn_moe_out")["values"])
        self.assertNotEqual(record(self.reference, "ffn_out")["values"], record(self.trial, "ffn_out")["values"])

        serialized = self.sidecar.read_text()
        for secret in (PRIVATE_PROMPT, str(PRIVATE_TOKEN), "/private/raw-prompt.txt", str(self.root),
                       self.reference_path.name, self.trial_path.name, *(path.name for path in self.shards)):
            self.assertNotIn(secret, serialized)
        def inspect_keys(value):
            if isinstance(value, dict):
                self.assertFalse({"prompt", "tokens", "token_ids", "prompt_tokens", "path", "private_path"} & value.keys())
                for child in value.values():
                    inspect_keys(child)
            elif isinstance(value, list):
                for child in value:
                    inspect_keys(child)
        inspect_keys(provenance)

    def test_two_shards_shuffled_preserve_quantized_slices_and_provenance(self):
        self.assert_bundle(self.extract())

    def test_f32_and_f16_expert_payloads_are_copied_without_conversion(self):
        for kind in ("f32", "f16"):
            with self.subTest(kind=kind):
                self.make_shards(expert_type=kind)
                self.assert_bundle(self.extract())
                self.output.unlink()
                self.sidecar.unlink()

    def test_zero_scale_and_normalized_weight_reshape_are_preserved(self):
        self.make_shards(scale=0.0)
        record(self.trial, "ffn_moe_weights_norm")["shape"] = [1, 2, 1, 1]
        self.write_traces()
        self.assert_bundle(self.extract(), scale=0.0)

    def test_mismatched_common_traces_are_rejected(self):
        for label in LABELS[:4]:
            with self.subTest(label=label):
                self.trial = trace(trial=True)
                row = record(self.trial, label)
                row["values"][0] = 2 if label == "ffn_moe_topk" else row["values"][0] + 0.125
                self.reject()

    def test_missing_and_duplicate_selected_records_are_rejected(self):
        for side in ("reference", "trial"):
            for label in LABELS:
                for mutation in ("missing", "duplicate"):
                    with self.subTest(side=side, label=label, mutation=mutation):
                        self.reference, self.trial = trace(), trace(trial=True)
                        document = getattr(self, side)
                        row = record(document, label)
                        if mutation == "missing":
                            document["records"].remove(row)
                        else:
                            document["records"].append(copy.deepcopy(row))
                        self.reject()

    def test_invalid_global_ids_are_rejected(self):
        for ids in ([3, 3], [4, 1], [-1, 1], [3.0, 1], [True, 1]):
            with self.subTest(ids=ids):
                self.reference, self.trial = trace(), trace(trial=True)
                for value in (self.reference, self.trial):
                    record(value, "ffn_moe_topk")["values"] = ids[:]
                self.reject()

    def test_missing_duplicate_and_copied_duplicate_shards_are_rejected(self):
        duplicate = self.root / "copied-first-shard.gguf"
        duplicate.write_bytes(self.shards[0].read_bytes())
        for shards in ([], self.shards[:1], [self.shards[0], self.shards[0]], [self.shards[0], duplicate]):
            with self.subTest(shards=[path.name for path in shards]):
                self.reject(shards)

    def test_nonfinite_and_unrepresentable_float_values_are_rejected(self):
        for side in ("reference", "trial"):
            for label in (name for name in LABELS if name != "ffn_moe_topk"):
                for value in (float("nan"), float("inf"), -float("inf"), 1e39):
                    with self.subTest(side=side, label=label, value=value):
                        self.reference, self.trial = trace(), trace(trial=True)
                        record(getattr(self, side), label)["values"][0] = value
                        self.reject()

    def test_multitoken_shapes_are_rejected(self):
        for side in ("reference", "trial"):
            for label in LABELS:
                with self.subTest(side=side, label=label):
                    self.reference, self.trial = trace(), trace(trial=True)
                    row = record(getattr(self, side), label)
                    row["shape"][1] = 2
                    row["values"] *= 2
                    self.reject()

    def test_wrong_types_extents_and_phases_are_rejected(self):
        for mutation in ("type", "extent", "shape_type", "within_phase", "between_phase"):
            with self.subTest(mutation=mutation):
                self.reference, self.trial = trace(), trace(trial=True)
                row = record(self.trial, "ffn_input")
                if mutation == "type":
                    row["type"] = "f16"
                elif mutation == "extent":
                    row["values"].pop()
                elif mutation == "shape_type":
                    row["shape"][1] = True
                elif mutation == "within_phase":
                    row["phase"] = "decode"
                else:
                    for item in self.trial["records"]:
                        item["phase"] = "decode"
                self.reject()

    def test_existing_gguf_or_provenance_is_preserved(self):
        sentinel = b"existing user artifact must survive\n"
        for existing, absent in ((self.output, self.sidecar), (self.sidecar, self.output)):
            with self.subTest(existing=existing.name):
                existing.write_bytes(sentinel)
                with self.assertRaises(replay.ReplayError):
                    self.extract()
                self.assertEqual(existing.read_bytes(), sentinel)
                self.assertFalse(absent.exists())
                existing.unlink()

    def test_dangling_output_symlink_is_preserved(self):
        destination = self.root / "missing-user-file"
        self.output.symlink_to(destination)
        with self.assertRaises(replay.ReplayError):
            self.extract()
        self.assertTrue(self.output.is_symlink())
        self.assertEqual(self.output.readlink(), destination)
        self.assertFalse(destination.exists())
        self.assertFalse(self.sidecar.exists())

    def test_tiny_limits_reject_before_any_output_file_is_opened(self):
        original_open = Path.open
        for limit in ("MAX_TRACE_BYTES", "MAX_HEADER_BYTES", "MAX_SOURCE_PAYLOAD_BYTES",
                      "MAX_MATRIX_BYTES", "MAX_FIXTURE_BYTES"):
            with self.subTest(limit=limit):
                writes = []
                def guarded_open(path, mode="r", *args, **kwargs):
                    if any(flag in mode for flag in "wax+"):
                        writes.append((path, mode))
                        raise AssertionError("bounds must be checked before opening output files")
                    return original_open(path, mode, *args, **kwargs)
                with mock.patch.object(replay, limit, 1), mock.patch.object(Path, "open", guarded_open):
                    with self.assertRaises(replay.ReplayError):
                        self.extract()
                self.assertEqual(writes, [])
                self.assertFalse(self.output.exists())
                self.assertFalse(self.sidecar.exists())


if __name__ == "__main__":
    unittest.main()
