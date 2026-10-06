#!/usr/bin/env python3
"""CPU-only dense direct-I/O contract on generated, untrained two-shard models.

No trained weights, GPU discovery, or downloads. Unsupported strict direct I/O
is a test failure, not a buffered fallback or a skipped correctness assertion.
Returned read bytes are distinct from logical dense bytes and physical disk I/O.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("full_model_test", SOURCE / "tests/full_model_test.py")
full = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(full)

SCOPE = "llama_dense_tensor_payload_reads_excludes_gguf_metadata_and_pmi_provider"
ACTUAL_SCOPE = "read_or_fread_returned_bytes_including_direct_alignment_overlap"
STAGING_SCOPE = "max_single_aligned_read_allocation"
COUNTERS = ("payload_requested_bytes", "payload_completed_bytes", "direct_read_bytes",
            "buffered_read_bytes", "fallback_count")


def hashes(paths):
    # Only the tiny generated fixtures are read; never accept checkpoint inputs.
    assert all(path.stat().st_size < 4 * 1024 * 1024 for path in paths)
    return [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]


def verify_loader(metrics, expected_bytes, *, direct):
    load = metrics["dense_load"]
    assert load["requested"] == ("direct-required" if direct else "default")
    assert load["effective"] == ("direct-required" if direct else "buffered")
    assert load["scope"] == SCOPE
    assert load["actual_read_scope"] == ACTUAL_SCOPE
    assert load["staging_peak_scope"] == STAGING_SCOPE
    shards = load["shards"]
    assert len(shards) == 2
    assert [row["shard_index"] for row in shards] == [0, 1]
    for row, expected in zip(shards, expected_bytes):
        for field in ("direct_io_requested", "direct_io_required", "direct_io_active"):
            assert row[field] is direct, (field, row)
        assert row["payload_requested_bytes"] == row["payload_completed_bytes"] == expected, row
        assert row["fallback_count"] == 0
        if direct:
            assert row["buffered_read_bytes"] == 0
            assert row["direct_read_bytes"] >= expected
            assert 0 < row["staging_peak_bytes"] <= 64 * 1024 * 1024
        else:
            assert row["buffered_read_bytes"] == expected
            assert row["direct_read_bytes"] == row["staging_peak_bytes"] == 0
    for field in COUNTERS:
        assert load[field] == sum(row[field] for row in shards), field
    assert load["staging_peak_bytes"] == max(row["staging_peak_bytes"] for row in shards)
    assert load["payload_completed_bytes"] == sum(expected_bytes)
    return load


def routing_rows(path, expected_load):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["type"] == "run"
    assert records[0]["config"]["dense_load"] == expected_load
    assert records[-1]["type"] == "complete"
    assert records[-1]["complete"] is True and records[-1]["truncated"] is False
    totals = [r for r in records if r["type"] == "metrics" and r.get("eval_id") is None]
    assert len(totals) == 1 and totals[0]["metrics"]["dense_load"] == expected_load
    return [{k: v for k, v in row.items() if k not in ("run_id", "event_id")}
            for row in records if row["type"] == "routing"]


def run_case(pmi, root, expert_type, ple_type, chunk, *, explicit_default):
    case = root / f"{expert_type}-{ple_type}-chunk{chunk}"
    case.mkdir(parents=True, exist_ok=True)
    paths = full.fixture.generate(case / "tiny.gguf", shards=2,
                                  expert_type=expert_type, ple_type=ple_type)
    _, tensors = full.fixture.fixture(expert_type=expert_type, ple_type=ple_type)
    dense_bytes = [sum(len(data) for name, _, _, data in tensors[shard::2]
                       if name != "per_layer_token_embd.weight" and not name.endswith("_exps.weight"))
                   for shard in range(2)]
    before = hashes(paths)
    traces, routes, loads = [], [], []
    # Shuffled CLI inputs test canonical shard indexing in the retained snapshot.
    base = ["run", *reversed(paths), "--prompt", "hello world", "--max-tokens", 8,
            "--context", 64, "--prefill-chunk", chunk, "--cache-bytes", 16384,
            "--cpu-repack", "off", "--projection-backend", "cpu", "--trace-mode", "full",
            "--profile-paging", "on"]
    for mode in ("default", "direct-required"):
        trace_path, route_path = case / (mode + ".trace.json"), case / (mode + ".routing.jsonl")
        # The live writer deliberately requires a new destination.
        route_path.unlink(missing_ok=True)
        flags = ["--dense-load", mode] if explicit_default or mode != "default" else []
        tokens = full.invoke(pmi, case, mode, [*base, *flags, "--trace-json", trace_path,
                                             "--routing-jsonl", route_path])
        trace = json.loads(trace_path.read_text())
        # Full capture fails the run at its byte cap; it never silently truncates.
        assert trace["tokens"] == tokens and trace["trace_mode"] == "full"
        load = verify_loader(trace["metrics"], dense_bytes, direct=mode == "direct-required")
        assert trace["metrics"]["projection_backend"] == "cpu"
        assert trace["metrics"]["projection_device"] is None
        assert trace["metrics"]["evictions"] > 0
        # The routing validator expects canonical callback names, not full
        # graph aliases such as "ffn_moe_weights_norm-0 (reshaped)". Restrict
        # only its view; preserve every original record for the differential
        # signature/float comparison below.
        routing_trace = {**trace, "records": [row for row in trace["records"]
                                               if full.compact_label(row["name"])]}
        full.check_routing_jsonl(route_path, routing_trace)
        traces.append(trace)
        loads.append(load)
        routes.append(routing_rows(route_path, load))
    assert hashes(paths) == before, "model source bytes changed"
    left, right = traces
    signature = lambda row: (row["evaluation"], row["phase"], row["name"], row["type"], tuple(row["shape"]))
    assert Counter(map(signature, left["records"])) == Counter(map(signature, right["records"])), \
        "default/direct traces did not capture identical graph boundaries"
    comparison = full.compare(left, right)  # Frozen original atol=2e-5, rtol=1e-4.
    assert routes[0] and routes[0] == routes[1], "routing IDs/weights changed with dense I/O policy"
    for key in ("page_requests", "expert_requests", "lookup_requests", "deferred_weight_bytes",
                "io_bytes", "cache_hits", "cache_misses", "evictions"):
        assert left["metrics"][key] == right["metrics"][key], ("paging counter changed", key)
    return {"expert_type": expert_type, "ple_type": ple_type, "prefill_chunk": chunk,
            "fixture_sha256": before, "fixture_source_bytes_unchanged": True,
            "expected_logical_dense_bytes_per_shard": dense_bytes,
            "logical_dense_payload_bytes_exact": True, "routing_rows_exact": len(routes[0]),
            "comparison": comparison, "default": loads[0], "direct_required": loads[1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pmi", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pmi, root = args.pmi.resolve(), args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Invalid policy and reference usage must fail before trying the absent file.
    missing = root / "must-not-open-missing.gguf"
    for label, command, policy, expected in (
            ("invalid-policy", "run", "direct", "dense-load must be default or direct-required"),
            ("reference-direct", "reference", "direct-required", "dense-load is only valid for the run command"),
            ("reference-default", "reference", "default", "dense-load is only valid for the run command")):
        full.invoke(pmi, root, label, [command, missing, "--prompt", "hello", "--dense-load", policy], success=False)
        error = (root / (label + ".stderr")).read_text()
        assert expected in error, (label, error)
    cases = [run_case(pmi, root, "f32", "f32", 1, explicit_default=False),
             run_case(pmi, root, "q8_0", "iq4_nl", 4, explicit_default=True)]
    report = {"passed": True, "scope": "synthetic_untrained_two_shard_cpu_only",
              "validation_before_file_access_cases": 3, "cases": cases}
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "cases": len(cases),
                      "max_abs_error": max(c["comparison"]["max_abs_error"] for c in cases),
                      "results": str(root / "results.json")}))


if __name__ == "__main__":
    main()
