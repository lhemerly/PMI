#!/usr/bin/env python3
"""Extract a bounded, local MoE replay fixture without loading a model.

Only exact GGUF header fields and the selected tensor ranges are read. The
provenance records range/header hashes, not full checkpoint hashes or prompts.
"""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys

_SPEC = importlib.util.spec_from_file_location(
    "pmi_replay_header_parser", Path(__file__).with_name("inspect_remote_headers.py"))
headers = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(headers)

MAX_TRACE_BYTES = 32 * 1024 * 1024
MAX_HEADER_BYTES = 32 * 1024 * 1024
MAX_SOURCE_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_MATRIX_BYTES = 32 * 1024 * 1024
MAX_FIXTURE_BYTES = 64 * 1024 * 1024
MAX_WIDTH = 16384
MAX_TOPK = 64
MAX_SHARDS = 64
# Storage types with standalone ggml CPU matrix-vector support. Sidecar-scaled
# NVFP4 and integer/workspace-only formats are outside this fixture contract.
MATRIX_TYPES = {0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18,
                19, 20, 21, 22, 23, 29, 30, 34, 35, 39, 41, 42}
TRACE_LABELS = ("ffn_input", "ffn_moe_logits", "ffn_moe_topk",
                "ffn_moe_weights_norm", "ffn_moe_out", "ffn_out")


class ReplayError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ReplayError(message)


def uint(value, name, maximum=2**32 - 1):
    require(type(value) is int and 0 <= value <= maximum, "invalid " + name)
    return value


def finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def fingerprint(stream):
    value = os.fstat(stream.fileno())
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class LocalRangeReader(headers.RangeReader):
    """The parser's exact-field interface backed only by a local raw file."""
    def __init__(self, stream, budget):
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode), "source shard must be a regular file")
        self.stream = stream
        super().__init__("", info.st_size, budget, opener=None)

    def fetch(self, end):
        start = len(self.data)
        require(start < end <= self.file_size, "invalid local header range")
        count = end - start
        self.budget.reserve(count)
        self.stream.seek(start)
        raw = self.stream.read(count)
        self.budget.received += len(raw)
        require(len(raw) == count, "short local header read")
        self.data.extend(raw)
        self.requests.append({"start": start, "bytes": count})


def read_headers(paths, stack):
    require(0 < len(paths) <= MAX_SHARDS, "source shard count exceeds bound")
    budget = headers.Budget(MAX_HEADER_BYTES)
    sources, identities = [], set()
    for path in paths:
        stream = stack.enter_context(Path(path).open("rb", buffering=0))
        before = fingerprint(stream)
        require(before[:2] not in identities, "duplicate source shard")
        identities.add(before[:2])
        reader = LocalRangeReader(stream, budget)
        try:
            header = headers.parse_header(reader)
        except (headers.InspectionError, UnicodeError, struct.error) as error:
            raise ReplayError("invalid GGUF header: " + str(error)) from error
        require(fingerprint(stream) == before, "source changed while reading header")
        metadata = header["metadata"]
        count = uint(metadata.get("split.count", 1), "split.count", MAX_SHARDS)
        count = count or 1  # An upstream-merged GGUF can retain split.count=0.
        index = uint(metadata.get("split.no", 0), "split.no", MAX_SHARDS - 1)
        require(count == len(paths) and index < count, "incomplete or inconsistent shard set")
        if count > 1:
            require("split.no" in metadata and "split.tensors.count" in metadata,
                    "missing split metadata")
        sources.append({"stream": stream, "fingerprint": before, "header": header, "index": index})
    sources.sort(key=lambda source: source["index"])
    require([source["index"] for source in sources] == list(range(len(sources))), "duplicate or missing split.no")
    primary = sources[0]["header"]["metadata"]
    require(primary.get("general.architecture") == "qwen4exp", "architecture must be qwen4exp")
    count = sum(source["header"]["tensor_count"] for source in sources)
    physical = {}
    for source in sources:
        metadata = source["header"]["metadata"]
        if "split.tensors.count" in metadata:
            require(uint(metadata["split.tensors.count"], "split.tensors.count") == count,
                    "aggregate split tensor count mismatch")
        for key, value in metadata.items():
            require(not key.startswith("general.tensor_extra."), "unsupported per-tensor precision metadata")
            if key == "general.architecture" or key.startswith("qwen4exp."):
                require(key in primary and primary[key] == value, "shards disagree on " + key)
        for tensor in source["header"]["tensors"]:
            require(tensor["name"] not in physical, "duplicate tensor across shards: " + tensor["name"])
            physical[tensor["name"]] = (source, tensor)
    return sources, primary, physical


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate trace JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ReplayError("nonfinite trace JSON constant: " + value)


def read_trace(path, layer, evaluation, phase):
    with Path(path).open("rb", buffering=0) as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_TRACE_BYTES, "trace exceeds regular-file byte bound")
        before = fingerprint(stream)
        raw = stream.read(MAX_TRACE_BYTES + 1)
        require(len(raw) <= MAX_TRACE_BYTES and fingerprint(stream) == before, "trace grew or changed while reading")
    try:
        document = json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ReplayError("invalid trace JSON: " + str(error)) from error
    require(isinstance(document, dict) and isinstance(document.get("records"), list), "trace records missing")
    names = {label + "-" + str(layer): label for label in TRACE_LABELS}
    selected = {}
    for record in document["records"]:
        require(isinstance(record, dict), "invalid trace record")
        require(isinstance(record.get("name"), str), "invalid trace record name")
        if record.get("evaluation") != evaluation or record.get("name") not in names:
            continue
        uint(record["evaluation"], "trace evaluation")
        label = names[record["name"]]
        require(label not in selected, "duplicate selected trace record: " + label)
        require(record.get("phase") == phase, "selected trace phase differs from requested phase")
        selected[label] = record
    require(set(selected) == set(TRACE_LABELS), "missing canonical selected trace records")
    require(len({record["phase"] for record in selected.values()}) == 1, "selected trace phases disagree")
    return selected, {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def f32(values):
    require(isinstance(values, list), "trace values must be an array")
    out = bytearray()
    for value in values:
        require(finite_number(value), "nonfinite or nonnumeric trace value")
        try:
            encoded = struct.pack("<f", value)
        except (OverflowError, struct.error) as error:
            raise ReplayError("trace value is outside F32 range") from error
        require(math.isfinite(struct.unpack("<f", encoded)[0]), "nonfinite F32 trace value")
        out.extend(encoded)
    return bytes(out)


def trace_tensor(record, width, kind="f32", *, normalized=False):
    expected = [width, 1, 1, 1]
    shape = record.get("shape")
    require(isinstance(shape, list) and all(type(x) is int for x in shape), "invalid trace tensor shape")
    require(shape == expected or (normalized and shape == [1, width, 1, 1]), "trace must contain one token with expected shape")
    values = record.get("values")
    require(record.get("type") == kind and isinstance(values, list) and len(values) == width,
            "trace tensor type or extent mismatch")
    if kind == "i32":
        require(all(type(value) is int and 0 <= value <= 2**31 - 1 for value in values), "invalid global expert ID")
        return struct.pack("<" + "i" * width, *values)
    return f32(values)


def validate(physical, metadata, reference, trial, layer):
    prefix = "blk." + str(layer) + "."
    weight_names = {"router": "ffn_gate_inp", "gate": "ffn_gate_exps", "up": "ffn_up_exps",
                    "down": "ffn_down_exps", "shared_router": "ffn_gate_inp_shexp",
                    "shared_gate": "ffn_gate_shexp", "shared_up": "ffn_up_shexp", "shared_down": "ffn_down_shexp"}
    expected_names = {prefix + value + ".weight" for value in weight_names.values()}
    for name in physical:
        if name.startswith((prefix + "ffn_", prefix + "exp_probs")):
            require(name in expected_names, "unsupported FFN tensor or sidecar scale: " + name)
    require(expected_names <= physical.keys(), "missing selected-layer weight")
    weights = {name: physical[prefix + value + ".weight"] for name, value in weight_names.items()}
    router = weights["router"][1]["shape"]
    gate = weights["gate"][1]["shape"]
    shared = weights["shared_gate"][1]["shape"]
    require(len(router) == 2 and len(gate) == 3 and len(shared) == 2, "unsupported physical matrix rank")
    dim, experts = router
    hidden, shared_hidden = gate[1], shared[1]
    require(all(0 < value <= MAX_WIDTH for value in (dim, hidden, shared_hidden)) and 0 < experts <= 2**31 - 1,
            "matrix dimensions exceed replay bound")
    topk = reference["ffn_moe_topk"].get("values")
    require(isinstance(topk, list), "top-k trace values must be an array")
    k = len(topk)
    require(0 < k <= min(MAX_TOPK, experts), "selected expert count exceeds replay bound")
    expected_shapes = {"router": [dim, experts], "gate": [dim, hidden, experts], "up": [dim, hidden, experts],
                       "down": [hidden, dim, experts], "shared_router": [dim],
                       "shared_gate": [dim, shared_hidden], "shared_up": [dim, shared_hidden], "shared_down": [shared_hidden, dim]}
    for name, (_, tensor) in weights.items():
        require(tensor["shape"] == expected_shapes[name], "physical matrix shape mismatch: " + name)
        require(tensor["ggml_type"] in MATRIX_TYPES, "unsupported matrix storage type: " + name)
    for key, expected in (("embedding_length", dim), ("expert_count", experts), ("expert_used_count", k),
                          ("expert_feed_forward_length", hidden), ("expert_shared_feed_forward_length", shared_hidden)):
        value = metadata.get("qwen4exp." + key)
        require(type(value) is int and value == expected, "model metadata disagrees with trace/weights: " + key)
    groups = metadata.get("qwen4exp.expert_group_count", 1)
    require(type(groups) is int and groups in (0, 1), "unsupported expert grouping")
    for key in ("qwen4exp.swiglu_clamp_exp", "qwen4exp.swiglu_clamp_shexp"):
        require(key not in metadata or metadata[key] == 0, "unsupported activation clamp metadata")
    scale = metadata.get("qwen4exp.expert_weights_scale", 1.0)
    require(finite_number(scale) and scale >= 0, "unsupported expert weight scale")
    scale = struct.unpack("<f", f32([scale]))[0]
    shapes = {"ffn_input": dim, "ffn_moe_logits": experts, "ffn_moe_topk": k,
              "ffn_moe_weights_norm": k, "ffn_moe_out": dim, "ffn_out": dim}
    encoded = []
    for selected in (reference, trial):
        encoded.append({label: trace_tensor(selected[label], width,
                         "i32" if label == "ffn_moe_topk" else "f32", normalized=label == "ffn_moe_weights_norm")
                        for label, width in shapes.items()})
    for label in TRACE_LABELS[:4]:
        require(reference[label]["phase"] == trial[label]["phase"] and
                reference[label]["values"] == trial[label]["values"] and encoded[0][label] == encoded[1][label],
                "reference and trial differ in " + label)
    ids = reference["ffn_moe_topk"]["values"]
    require(len(set(ids)) == k and all(value < experts for value in ids), "duplicate or out-of-range expert IDs")
    require(all(value >= 0 for value in reference["ffn_moe_weights_norm"]["values"]), "negative normalized expert weight")
    return weights, ids, scale, encoded, (dim, experts, hidden, shared_hidden)


def gguf_string(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def make_header(tensors, layer, evaluation, scale):
    metadata = [("general.architecture", 8, "pmi_moe_replay"), ("general.alignment", 4, 32),
                ("replay.version", 4, 1), ("replay.layer", 4, layer), ("replay.evaluation", 4, evaluation),
                ("replay.expert_weights_scale", 6, scale)]
    out = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(metadata)))
    for name, kind, value in metadata:
        out.extend(gguf_string(name) + struct.pack("<I", kind))
        out.extend(gguf_string(value) if kind == 8 else struct.pack("<f" if kind == 6 else "<I", value))
    offset = 0
    for tensor in tensors:
        shape = tensor["shape"]
        out.extend(gguf_string(tensor["name"]) + struct.pack("<I", len(shape)))
        out.extend(struct.pack("<" + "Q" * len(shape), *shape))
        out.extend(struct.pack("<IQ", tensor["kind"], offset))
        offset += (tensor["bytes"] + 31) // 32 * 32
    out.extend(bytes(-len(out) % 32))
    require(len(out) + offset <= MAX_FIXTURE_BYTES, "fixture total exceeds byte bound")
    return bytes(out), len(out) + offset


def extract(shards, reference_trace, trial_trace, layer, evaluation, output, phase="prefill"):
    """Write one exclusive replay GGUF and its exclusive provenance sidecar."""
    layer, evaluation = uint(layer, "layer"), uint(evaluation, "evaluation")
    require(phase in ("prefill", "decode"), "invalid requested trace phase")
    output = Path(output)
    require(output.suffix.lower() == ".gguf", "output must be a new .gguf path")
    provenance_path = output.with_suffix(".provenance.json")
    require(not os.path.lexists(output) and not os.path.lexists(provenance_path), "refusing output overwrite")
    with ExitStack() as stack:
        sources, metadata, physical = read_headers(shards, stack)
        reference, reference_hash = read_trace(reference_trace, layer, evaluation, phase)
        trial, trial_hash = read_trace(trial_trace, layer, evaluation, phase)
        weights, ids, scale, encoded, dimensions = validate(physical, metadata, reference, trial, layer)
        dim, experts, hidden, shared_hidden = dimensions
        tensors, ranges = [], []
        source_payload_bytes = 0
        for name, (source, tensor) in weights.items():
            routed = name in ("gate", "up", "down")
            shape = tensor["shape"][:-1] + [len(ids)] if routed else tensor["shape"][:]
            if name == "shared_router":
                shape = [dim, 1]
            stride = tensor["bytes"] // experts if routed else tensor["bytes"]
            selected = ids if routed else [None]
            nbytes = stride * len(selected)
            require(nbytes <= MAX_MATRIX_BYTES, "selected matrix exceeds byte bound")
            source_payload_bytes += nbytes
            require(source_payload_bytes <= MAX_SOURCE_PAYLOAD_BYTES, "source payload exceeds byte bound")
            pieces = []
            for expert_id in selected:
                offset = tensor["absolute_offset"] + (expert_id * stride if routed else 0)
                require(offset + stride <= tensor["absolute_offset"] + tensor["bytes"], "selected slice exceeds packed matrix")
                item = {"output_tensor": name + ".weight", "source_tensor": tensor["name"],
                        "shard_index": source["index"], "source_offset": offset, "bytes": stride}
                if routed:
                    item["expert_id"] = expert_id
                ranges.append(item)
                pieces.append((source, item))
            tensors.append({"name": name + ".weight", "shape": shape, "kind": tensor["ggml_type"],
                            "bytes": nbytes, "pieces": pieces})
        for name, label, width, kind, index in (
                ("input", "ffn_input", dim, 0, 0), ("ids", "ffn_moe_topk", len(ids), 26, 0),
                ("logits", "ffn_moe_logits", experts, 0, 0), ("weights", "ffn_moe_weights_norm", len(ids), 0, 0),
                ("reference_routed", "ffn_moe_out", dim, 0, 0), ("trial_routed", "ffn_moe_out", dim, 0, 1),
                ("reference_output", "ffn_out", dim, 0, 0), ("trial_output", "ffn_out", dim, 0, 1)):
            data = encoded[index][label]
            tensors.append({"name": "replay." + name, "shape": [width], "kind": kind, "bytes": len(data), "data": data})
        require(len(tensors) == 16, "unexpected replay tensor inventory")
        header, file_bytes = make_header(tensors, layer, evaluation, scale)
        provenance = {"schema_version": 1, "scope": "pmi_moe_replay_extraction", "layer": layer,
                      "evaluation": evaluation, "expert_ids": ids, "source_payload_bytes": source_payload_bytes,
                      "fixture": {"name": output.name, "bytes": file_bytes},
                      "traces": {"reference": reference_hash, "trial": trial_hash},
                      "shards": [{"shard_index": source["index"], "file_bytes": source["fingerprint"][2],
                                  "header_bytes": source["header"]["header_bytes"],
                                  "header_sha256": source["header"]["header_sha256"]} for source in sources],
                      "ranges": ranges}
        created = []
        try:
            with output.open("xb") as target:
                created.append(output)
                with provenance_path.open("x", encoding="utf-8") as sidecar:
                    created.append(provenance_path)
                    digest = hashlib.sha256()
                    def write(raw):
                        target.write(raw)
                        digest.update(raw)
                    write(header)
                    for tensor in tensors:
                        if "data" in tensor:
                            write(tensor["data"])
                        else:
                            for source, item in tensor["pieces"]:
                                stream = source["stream"]
                                require(fingerprint(stream) == source["fingerprint"], "source changed before payload read")
                                stream.seek(item["source_offset"])
                                raw = stream.read(item["bytes"])
                                require(len(raw) == item["bytes"], "short selected tensor read")
                                item["sha256"] = hashlib.sha256(raw).hexdigest()
                                write(raw)
                        write(bytes(-tensor["bytes"] % 32))
                    require(target.tell() == file_bytes, "fixture byte count mismatch")
                    for source in sources:
                        require(fingerprint(source["stream"]) == source["fingerprint"], "source changed during extraction")
                    provenance["fixture"]["sha256"] = digest.hexdigest()
                    json.dump(provenance, sidecar, indent=2, allow_nan=False)
                    sidecar.write("\n")
            return provenance
        except BaseException:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--reference-trace", required=True, type=Path)
    parser.add_argument("--trial-trace", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--evaluation", required=True, type=int)
    parser.add_argument("--phase", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = extract(args.shards, args.reference_trace, args.trial_trace, args.layer, args.evaluation, args.output, args.phase)
    except (ReplayError, OSError, headers.InspectionError) as error:
        print("MoE replay extraction failed: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps({"fixture": result["fixture"], "source_payload_bytes": result["source_payload_bytes"],
                      "expert_ids": result["expert_ids"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
