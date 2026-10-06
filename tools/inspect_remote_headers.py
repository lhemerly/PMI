#!/usr/bin/env python3
"""Inspect pinned GGUF headers with validated HTTP ranges and no tensor reads.

The parser only reads a field's exact size plus a conservative lower bound on
the remaining header. This permits buffered metadata parsing without guessing
where the tensor data starts. Response status, range, total size, encoding and
length are checked before reading any response body. No retry is automatic.
"""

import argparse
import collections
import datetime
import hashlib
import io
import json
import math
import re
import struct
import sys
import urllib.parse
import urllib.request
from pathlib import Path


MAX_HEADER_BYTES = 32 * 1024 * 1024
REPO = "unsloth/Qwen3.8-Flash-Next-GGUF"
REVISION = "38bb39ee97821de2c9009abb7e93950eec396e66"
SHARDS = [
    ("UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf", 10946624,
     "5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4"),
    ("UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf", 49835229856,
     "577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7"),
    ("UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf", 43836407744,
     "d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833"),
]
# (name, values per block, bytes per block), matching ggml tensor row storage.
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18), 42: ("Q2_0", 64, 18),
}
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
           6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class InspectionError(Exception):
    pass


class Budget:
    def __init__(self, limit):
        if not 0 < limit <= MAX_HEADER_BYTES:
            raise InspectionError("The authorized maximum is 32 MiB")
        self.limit = limit
        self.requested = 0
        self.received = 0
        self.requests = []

    def reserve(self, count):
        if count > self.limit - self.requested:
            raise InspectionError("Header budget exhausted; no further request issued")
        self.requested += count


class RangeReader:
    def __init__(self, url, file_size, budget, opener=urllib.request.urlopen):
        self.url, self.file_size, self.budget, self.opener = url, file_size, budget, opener
        self.data = bytearray()
        self.pos = 0
        self.requests = []

    def fetch(self, end):
        start = len(self.data)
        if not start < end <= self.file_size:
            raise InspectionError("Invalid or out-of-file metadata range")
        count = end - start
        self.budget.reserve(count)
        record = {"start": start, "end_inclusive": end - 1, "requested_bytes": count}
        self.requests.append(record)
        self.budget.requests.append(record)
        request = urllib.request.Request(self.url, headers={
            "Range": f"bytes={start}-{end - 1}", "Accept-Encoding": "identity",
            "User-Agent": "PMI-GGUF-header-inspector/1.0",
        })
        # HTTP errors, ignored Range responses, and mismatched lengths abort.
        # Do not print response URLs: redirected CDN URLs may contain signatures.
        with self.opener(request, timeout=30) as response:
            record["status"] = response.status
            record["content_range"] = response.headers.get("Content-Range")
            if response.status != 206:
                raise InspectionError(f"Server ignored Range (HTTP {response.status}); body not read")
            expected = f"bytes {start}-{end - 1}/{self.file_size}"
            if response.headers.get("Content-Range") != expected:
                raise InspectionError("Content-Range does not match requested bounds/expected file size")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise InspectionError("Encoded range response refused")
            if response.headers.get("Content-Length") != str(count):
                raise InspectionError("Content-Length does not match requested range")
            body = response.read(count)
            self.budget.received += len(body)
            record["received_bytes"] = len(body)
            if len(body) != count:
                raise InspectionError("Truncated range response")
            self.data.extend(body)

    def read(self, count, remaining_minimum=0):
        if count < 0 or remaining_minimum < 0:
            raise InspectionError("Negative field length")
        need = self.pos + count
        safe_end = need + remaining_minimum
        if safe_end > self.file_size:
            raise InspectionError("Header field/counts exceed declared file size")
        while len(self.data) < need:
            # All lookahead belongs to mandatory, as-yet-unparsed header fields.
            end = min(safe_end, len(self.data) + 256 * 1024)
            self.fetch(end)
        value = self.data[self.pos:need]
        self.pos = need
        return value

    def scalar(self, fmt, tail=0):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt), tail))[0]

    def string(self, tail=0):
        length = self.scalar("Q", tail)
        if length > MAX_HEADER_BYTES:
            raise InspectionError("String alone exceeds authorized header budget")
        return bytes(self.read(length, tail)).decode("utf-8", errors="strict")


def metadata_value(reader, kind, tail):
    if kind in SCALARS:
        return reader.scalar(SCALARS[kind], tail)
    if kind == 8:
        return reader.string(tail)
    if kind != 9:
        raise InspectionError(f"Unsupported GGUF metadata value type {kind}")
    element_kind = reader.scalar("I", tail + 8)
    count = reader.scalar("Q", tail)
    if element_kind == 8:
        if count > MAX_HEADER_BYTES // 8:
            raise InspectionError("Implausible string array count")
        first = []
        digest = hashlib.sha256()
        for index in range(count):
            text = reader.string(tail + (count - index - 1) * 8)
            digest.update(text.encode("utf-8"))
            digest.update(b"\0")
            if index < 8:
                first.append(text)
        return {"array_type": element_kind, "count": count, "first": first,
                "nul_delimited_utf8_sha256": digest.hexdigest()}
    if element_kind not in SCALARS:
        raise InspectionError("Nested/unknown metadata array type refused")
    fmt = SCALARS[element_kind]
    width = struct.calcsize("<" + fmt)
    raw = reader.read(count * width, tail)
    first = [x[0] for x in struct.iter_unpack("<" + fmt, raw[: min(count, 64) * width])]
    return {"array_type": element_kind, "count": count, "first": first,
            "raw_sha256": hashlib.sha256(raw).hexdigest()}


def tensor_bytes(kind, shape):
    if kind not in TYPES:
        raise InspectionError(f"Unsupported tensor storage type {kind}")
    name, block, size = TYPES[kind]
    if not shape or any(x <= 0 for x in shape) or shape[0] % block:
        raise InspectionError("Invalid shape or quantization row alignment")
    return name, shape[0] // block * size * math.prod(shape[1:])


def parse_header(reader):
    magic = bytes(reader.read(4, 20))
    if magic != b"GGUF":
        raise InspectionError("Not little-endian GGUF")
    version = reader.scalar("I", 16)
    if version not in (2, 3):
        raise InspectionError("Unsupported GGUF version")
    n_tensors = reader.scalar("Q", 8)
    n_metadata = reader.scalar("Q")
    if n_tensors > MAX_HEADER_BYTES // 24 or n_metadata > MAX_HEADER_BYTES // 13:
        raise InspectionError("Implausible GGUF counts")
    metadata = {}
    for index in range(n_metadata):
        tail = (n_metadata - index - 1) * 13 + n_tensors * 24
        name = reader.string(tail + 5)
        kind = reader.scalar("I", tail + 1)
        if name in metadata:
            raise InspectionError("Duplicate metadata name")
        metadata[name] = metadata_value(reader, kind, tail)
    tensors = []
    seen = set()
    for index in range(n_tensors):
        tail = (n_tensors - index - 1) * 24
        name = reader.string(tail + 16)
        dimensions = reader.scalar("I", tail + 12)
        if not name or name in seen or not 1 <= dimensions <= 4:
            raise InspectionError("Invalid/duplicate tensor name or dimension count")
        seen.add(name)
        shape = [reader.scalar("Q", tail + 12 + (dimensions - j - 1) * 8)
                 for j in range(dimensions)]
        kind = reader.scalar("I", tail + 8)
        offset = reader.scalar("Q", tail)
        type_name, nbytes = tensor_bytes(kind, shape)
        tensors.append({"name": name, "shape": shape, "ggml_type": kind,
                        "type": type_name, "relative_offset": offset, "bytes": nbytes})
    if len(reader.data) != reader.pos:
        raise InspectionError("Parser lookahead exceeded exact header; refusing report")
    alignment = metadata.get("general.alignment", 32)
    if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
        raise InspectionError("Invalid tensor alignment")
    data_offset = (reader.pos + alignment - 1) // alignment * alignment
    for tensor in tensors:
        tensor["absolute_offset"] = data_offset + tensor["relative_offset"]
        if tensor["relative_offset"] % alignment:
            raise InspectionError("Unaligned tensor range")
        if tensor["absolute_offset"] + tensor["bytes"] > reader.file_size:
            raise InspectionError("Tensor descriptor exceeds known file size")
    ordered = sorted(tensors, key=lambda x: x["absolute_offset"])
    for left, right in zip(ordered, ordered[1:]):
        if left["absolute_offset"] + left["bytes"] > right["absolute_offset"]:
            raise InspectionError("Overlapping tensor descriptors")
    return {"version": version, "tensor_count": n_tensors, "metadata_count": n_metadata,
            "header_bytes": reader.pos, "tensor_data_offset": data_offset,
            "header_sha256": hashlib.sha256(reader.data).hexdigest(),
            "metadata": metadata, "tensors": tensors, "requests": reader.requests}


def classify(name):
    if re.search(r"(?:mtp|vision|nextn)", name):
        return "optional_mtp_or_vision"
    if "_exps" in name:
        return "routed_expert_weights"
    if name == "per_layer_token_embd.weight":
        return "sparse_lookup_table"
    if "shexp" in name:
        return "shared_expert_weights"
    if "ffn_gate_inp" in name or "router" in name:
        return "router_weights"
    if name == "token_embd.weight":
        return "token_embeddings"
    if name == "output.weight":
        return "output_projection"
    if "hc_" in name:
        return "hyper_connection"
    if any(x in name for x in ("attn_", "ssm_", "indexer_", "indexer.")):
        return "attention_ssm_and_indexer"
    if ".ple_" in name:
        return "ple_dense_projection_and_norm"
    return "dense_other"


def summarize(shards):
    tensors = [dict(t, shard_id=i) for i, shard in enumerate(shards) for t in shard["tensors"]]
    if len({t["name"] for t in tensors}) != len(tensors):
        raise InspectionError("Duplicate tensor across shards")
    primary = shards[0]["metadata"]
    for i, shard in enumerate(shards):
        metadata = shard["metadata"]
        if metadata.get("split.no") != i or metadata.get("split.count") != len(shards):
            raise InspectionError("Shard identity/count mismatch")
        if metadata.get("split.tensors.count") != len(tensors):
            raise InspectionError("Aggregate tensor count mismatch")
    if primary.get("general.architecture") != "qwen4exp":
        raise InspectionError("Unexpected target architecture")
    experts = primary.get("qwen4exp.expert_count")
    if experts != 512:
        raise InspectionError("Unexpected target expert count")
    classes, types = {}, {}
    layouts, attention_layouts = {}, {}
    bundles = collections.defaultdict(int)
    for tensor in tensors:
        category = classify(tensor["name"])
        tensor["class"] = category
        for key, mapping in ((category, classes), (tensor["type"], types)):
            entry = mapping.setdefault(key, {"tensor_count": 0, "bytes": 0})
            entry["tensor_count"] += 1
            entry["bytes"] += tensor["bytes"]
        if category == "routed_expert_weights":
            shape = tensor["shape"]
            if len(shape) != 3 or shape[-1] != experts:
                raise InspectionError("Unrecognized routed expert layout; manual review required")
            _, page_bytes = tensor_bytes(tensor["ggml_type"], shape[:-1])
            if page_bytes * experts != tensor["bytes"]:
                raise InspectionError("Expert stride mismatch")
            tensor["expert_slice_bytes"] = page_bytes
            match = re.fullmatch(r"blk\.(\d+)\.(.+)", tensor["name"])
            if not match:
                raise InspectionError("Unrecognized routed layer name")
            bundles[int(match[1])] += page_bytes
            key = (match[2], tuple(shape), tensor["type"])
            layout = layouts.setdefault(key, {"name_pattern": "blk.*." + match[2],
                "shape": shape, "type": tensor["type"], "layer_count": 0,
                "expert_slice_bytes": page_bytes, "packed_bytes_each": tensor["bytes"]})
            layout["layer_count"] += 1
        if "attn_qkv" in tensor["name"]:
            pattern = re.sub(r"^blk\.\d+\.", "blk.*.", tensor["name"])
            key = (pattern, tuple(tensor["shape"]), tensor["type"])
            layout = attention_layouts.setdefault(key, {"name_pattern": pattern,
                "shape": tensor["shape"], "type": tensor["type"], "layer_count": 0,
                "bytes_each": tensor["bytes"]})
            layout["layer_count"] += 1
    excluded = {"routed_expert_weights", "sparse_lookup_table", "optional_mtp_or_vision"}
    sparse_tables = []
    for tensor in tensors:
        if tensor["class"] == "sparse_lookup_table":
            _, row_bytes = tensor_bytes(tensor["ggml_type"], tensor["shape"][:1])
            sparse_tables.append({"name": tensor["name"], "shape": tensor["shape"],
                                  "type": tensor["type"], "row_bytes": row_bytes,
                                  "table_bytes": tensor["bytes"]})
    active_experts = primary.get("qwen4exp.expert_used_count")
    if type(active_experts) is not int or not 0 < active_experts <= experts:
        raise InspectionError("Invalid active expert count")
    return {"tensor_count": len(tensors), "tensor_payload_bytes": sum(t["bytes"] for t in tensors),
            "class_totals": classes, "type_totals": types,
            "non_expert_dense_weight_bytes": sum(x["bytes"] for k, x in classes.items() if k not in excluded),
            "largest_expert_component_bytes": max(t.get("expert_slice_bytes", 0) for t in tensors),
            "largest_per_layer_expert_bundle_bytes": max(bundles.values(), default=0),
            "per_layer_expert_bundle_bytes": dict(sorted(bundles.items())),
            "routed_layouts": list(layouts.values()),
            "packed_attention_layouts": list(attention_layouts.values()),
            "sparse_tables": sparse_tables,
            "derived_cold_routed_logical_bytes_per_token": sum(bundles.values()) * active_experts,
            "derived_cold_routed_note": "One token, ten distinct selected experts per layer, all three matrices, zero cache hits. Logical payload only; not a disk measurement or throughput prediction.",
            "ple_layer_index_note": "HF ple_layer_ids is one-based; conversion/qwen4exp.py subtracts one. This GGUF has qwen4exp.ple.layers=[1] and blk.1.ple_* tensors.",
            "fused_gate_up_expert_tensors": sum("gate_up_exps" in t["name"] for t in tensors),
            "largest_tensors": sorted(tensors, key=lambda t: t["bytes"], reverse=True)[:12],
            "tensors": tensors}


def self_test():
    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw
    meta = string("tokens") + struct.pack("<IIQ", 9, 8, 3)
    meta += b"".join(string(x) for x in ("x", "y" * 300000, "z"))
    descriptor = string("sample.weight") + struct.pack("<IQIQ", 1, 4, 0, 0)
    header = struct.pack("<4sIQQ", b"GGUF", 3, 1, 1) + meta + descriptor
    data_offset = (len(header) + 31) // 32 * 32
    fixture = header + b"\0" * (data_offset - len(header)) + b"\xff" * 16
    class Response(io.BytesIO):
        def __init__(self, data, status, headers):
            super().__init__(data)
            self.status, self.headers = status, headers
    def opener(request, timeout):
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["Range"]).groups())
        assert end < len(header), "Inspector attempted padding/tensor payload read"
        return Response(fixture[start:end + 1], 206,
                        {"Content-Range": f"bytes {start}-{end}/{len(fixture)}", "Content-Length": str(end-start+1)})
    budget = Budget(MAX_HEADER_BYTES)
    result = parse_header(RangeReader("https://example.invalid/model", len(fixture), budget, opener))
    assert result["header_bytes"] == len(header)
    assert budget.received == len(header)
    assert result["tensors"][0]["bytes"] == 16
    for status, content_range, length in ((200, "bytes 0-23/24", "24"),
                                         (206, "bytes 1-24/25", "24"),
                                         (206, "bytes 0-23/24", "25")):
        class Unreadable(Response):
            def read(self, n=-1):
                raise AssertionError("Invalid response body must not be read")
        def bad_opener(request, timeout):
            return Unreadable(b"", status, {"Content-Range": content_range, "Content-Length": length})
        try:
            RangeReader("https://example.invalid/model", 24, Budget(24), bad_opener).read(24)
        except InspectionError:
            pass
        else:
            raise AssertionError("Invalid response accepted")
    try:
        RangeReader("https://example.invalid/model", len(fixture), Budget(8), opener).read(24)
    except InspectionError:
        pass
    else:
        raise AssertionError("Budget exceeded")
    print("PASS: header-only lookahead, ignored/mismatched ranges, lengths, and byte budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summarize-report", type=Path,
                        help="Recompute summaries from a saved successful report, without network")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output is None:
        parser.error("--output is required for remote inspection")
    if args.summarize_report is not None:
        report = json.loads(args.summarize_report.read_text())
        if report.get("status") != "complete":
            raise InspectionError("Only a complete saved report can be summarized")
        report["summary"] = summarize(report["shards"])
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return
    budget = Budget(MAX_HEADER_BYTES)
    report = {"repository": REPO, "revision": REVISION,
              "inspected_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "limit_bytes": MAX_HEADER_BYTES, "tensor_payload_read": False,
              "full_file_sha256_verified": False,
              "classification_note": "Name/layout classification, not a resident-memory measurement. Dense total excludes routed expert arrays, the PLE table, and optional tensors. Graph/KV/recurrent/staging allocations are additional.",
              "shards": []}
    try:
        for path, size, sha256 in SHARDS:
            print(f"Inspecting header: {path}", file=sys.stderr, flush=True)
            url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{urllib.parse.quote(path)}"
            reader = RangeReader(url, size, budget)
            shard = parse_header(reader)
            shard.update(path=path, file_size=size, source_sha256=sha256, source_url=url)
            report["shards"].append(shard)
            print(f"  {shard['tensor_count']} tensors; {shard['header_bytes']} header bytes; {len(reader.requests)} ranges", file=sys.stderr, flush=True)
        report["summary"] = summarize(report["shards"])
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "aborted"
        # Avoid exposing signed redirect URLs from arbitrary network exception text.
        report["error"] = str(error) if isinstance(error, InspectionError) else type(error).__name__
        raise
    finally:
        report["requested_header_bytes"] = budget.requested
        report["received_header_bytes"] = budget.received
        report["request_count"] = len(budget.requests)
        report["request_audit"] = budget.requests
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
