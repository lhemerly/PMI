#!/usr/bin/env python3
"""Differential complete-graph tests using generated, untrained Qwen fixtures.

Reference execution uses llama.cpp's resident operations. PMI execution must
agree on logits, intermediate activations, exact routing IDs and greedy tokens
while loading expert/PLE pages through the bounded runtime. This is correctness
coverage for architecture wiring, not language quality or real-model evidence.
"""

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_qwen_fixture", SOURCE / "tools/make_qwen_fixture.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def invoke(pmi, root, name, args, *, success=True):
    proc = subprocess.run([str(pmi), *map(str, args)], capture_output=True, timeout=60)
    (root / (name + ".stdout")).write_bytes(proc.stdout)
    (root / (name + ".stderr")).write_bytes(proc.stderr)
    if success:
        assert proc.returncode == 0, (name, proc.returncode, proc.stderr.decode(errors="replace")[-5000:])
    else:
        assert proc.returncode != 0, f"{name}: invalid request unexpectedly succeeded"
        assert b"\nTOKENS" not in proc.stdout, f"{name}: failed run printed successful token result"
    tokens = None
    for line in proc.stdout.decode(errors="replace").splitlines():
        if line == "TOKENS" or line.startswith("TOKENS "):
            tokens = [int(value) for value in line.split()[1:]]
    if success:
        assert tokens is not None and len(tokens) == 8, f"{name}: generation did not exercise eight decode steps"
    return tokens


COMPACT_LABELS = {
    "result_output", "ffn_input", "ffn_moe_logits", "ffn_moe_topk",
    "ffn_moe_weights_norm", "ffn_moe_weights_scaled", "ffn_moe_out", "ffn_out",
    "ple_embd", "ple_gated_value",
}


def compact_label(name):
    label, separator, layer = name.rpartition("-")
    return name in COMPACT_LABELS or (separator and layer.isdecimal() and label in COMPACT_LABELS)


def compare(reference, actual, *, atol=2e-5, rtol=1e-4, compact=False):
    assert reference["tokens"] == actual["tokens"], "greedy token mismatch"

    def index(records):
        result, seen = {}, Counter()
        for record in records:
            signature = (record["evaluation"], record["phase"], record["name"], record["type"], tuple(record["shape"]))
            occurrence = seen[signature]
            seen[signature] += 1
            result[signature + (occurrence,)] = record["values"]
        return result

    left, right = index(reference["records"]), index(actual["records"])
    common = left.keys() & right.keys()
    assert common, "no comparable graph snapshots captured"
    # Custom paged nodes replace internal expert matmul nodes. Require all final
    # logits and the shared graph boundaries; extra operation-local nodes may
    # legitimately exist only in the resident graph.
    if compact:
        required = {key for key in left if compact_label(key[2])}
        assert all(compact_label(key[2]) for key in right), "compact trace retained full-only activations"
        assert right.keys() <= left.keys(), "compact trace has records missing from its reference"
    else:
        required = {key for key in left if key[2] == "result_output" or
                    key[2].startswith(("ffn_out", "ffn_moe_out", "ple_embd", "ple_gate", "ple_conv_out", "indexer_top_k"))}
    assert required <= right.keys(), f"missing shared graph snapshots: {sorted(required - right.keys())[:8]}"
    assert any(key[2] == "result_output" for key in common), "no logits snapshots"
    assert any(key[2].startswith("ple_") for key in common), "PLE path not traced"
    assert any(key[2].startswith("ffn_") for key in common), "MoE path not traced"
    assert any(key[3] == "i32" for key in common), "routing IDs not traced"
    maximum, f32_count, i32_count = 0.0, 0, 0
    names = set()
    for key in sorted(common):
        a, b = left[key], right[key]
        assert len(a) == len(b) == math.prod(key[4]), f"trace extent mismatch: {key}"
        names.add(key[2])
        if key[3] == "i32":
            assert a == b, f"routing mismatch: {key}"
            i32_count += len(a)
        else:
            for j, (expected, got) in enumerate(zip(a, b)):
                assert math.isfinite(expected) and math.isfinite(got), f"nonfinite activation: {key} element {j}"
                error = abs(expected - got)
                maximum = max(maximum, error)
                assert error <= atol + rtol * abs(expected), (
                    f"activation mismatch: {key} element {j}: expected {expected}, got {got}, abs error {error}"
                )
            f32_count += len(a)
    return {"compared_records": len(common), "compared_floats": f32_count, "compared_ids": i32_count,
            "max_abs_error": maximum, "absolute_tolerance": atol, "relative_tolerance": rtol,
            "trace_names": sorted(names)}


def check_evaluation_metrics(trace, *, paged, repack, reference_mode, trace_mode, chunk):
    metrics = trace["metrics"]
    assert metrics["path"] == ("pmi_paged_qwen_cpu" if paged else "llama_reference")
    assert metrics["cpu_repack"] is repack
    assert metrics["reference_mode"] == reference_mode
    assert trace["trace_mode"] == metrics["trace_mode"] == trace_mode
    assert metrics["generated_tokens"] == len(trace["tokens"]) == 8
    for name in ("elapsed_ms", "setup_ms", "prefill_ms", "decode_ms", "ttft_ms", "first_token_after_setup_ms"):
        assert isinstance(metrics[name], (int, float)) and math.isfinite(metrics[name]) and metrics[name] >= 0, name
    assert metrics["setup_ms"] + metrics["prefill_ms"] <= metrics["ttft_ms"] + 1e-6
    assert metrics["ttft_ms"] <= metrics["elapsed_ms"] + 1e-6
    assert math.isclose(metrics["ttft_ms"] - metrics["setup_ms"], metrics["first_token_after_setup_ms"],
                        rel_tol=1e-8, abs_tol=1e-6), "inconsistent first-token timing origins"

    evaluations = metrics["evaluations"]
    prefill_count, decode_count = metrics["prefill_evaluations"], metrics["decode_evaluations"]
    assert prefill_count > 0 and decode_count == 7
    assert len(evaluations) == prefill_count + decode_count
    assert [item["evaluation"] for item in evaluations] == list(range(len(evaluations)))
    assert [item["phase"] for item in evaluations] == ["prefill"] * prefill_count + ["decode"] * decode_count
    assert sum(item["input_tokens"] for item in evaluations[:prefill_count]) == metrics["prompt_tokens"]
    assert all(0 < item["input_tokens"] <= chunk for item in evaluations[:prefill_count])
    assert all(item["input_tokens"] == 1 for item in evaluations[prefill_count:])
    for phase in ("prefill", "decode"):
        duration = sum(item["elapsed_ms"] for item in evaluations if item["phase"] == phase)
        assert math.isclose(duration, metrics[phase + "_ms"], rel_tol=1e-8, abs_tol=1e-6), phase
    delta_keys = ("io_bytes", "cache_hits", "cache_misses", "evictions")
    for item in evaluations:
        assert math.isfinite(item["elapsed_ms"]) and item["elapsed_ms"] >= 0
        for name in delta_keys:
            assert isinstance(item[name], int) and item[name] >= 0, name
        requests = item["cache_hits"] + item["cache_misses"]
        expected_rate = item["cache_hits"] / requests if requests else 0
        assert math.isclose(item["hit_rate"], expected_rate, rel_tol=1e-8, abs_tol=1e-9)
        for name in ("rss_bytes", "peak_rss_bytes", "process_storage_read_bytes_delta"):
            value = item[name]
            assert value is None or (isinstance(value, int) and value >= 0), name
    for name in delta_keys:
        assert sum(item[name] for item in evaluations) == metrics[name], f"evaluation {name} did not reconcile"
    if paged:
        assert metrics["page_requests"] == metrics["expert_requests"] + metrics["lookup_requests"] > 0
        assert metrics["cache_hits"] + metrics["cache_misses"] == metrics["page_requests"]
        assert metrics["deferred_weight_bytes"] > 0 and metrics["io_bytes"] > 0
        assert metrics["evictions"] > 0, "option test must exercise forced eviction"
        assert metrics["cache_peak_reserved_bytes"] <= metrics["cache_capacity_bytes"]
    else:
        for name in (*delta_keys, "page_requests", "expert_requests", "lookup_requests",
                     "deferred_weight_bytes", "cache_reserved_bytes", "cache_peak_reserved_bytes", "cache_capacity_bytes"):
            assert metrics[name] == 0, f"native reference unexpectedly used PMI storage: {name}"
    return evaluations


def check_ffn_inputs(trace, evaluations, *, width, layers):
    inputs = [item for item in trace["records"] if item["name"].startswith("ffn_input-")]
    expected = {(item["evaluation"], f"ffn_input-{layer}")
                for item in evaluations for layer in range(layers)}
    assert Counter((item["evaluation"], item["name"]) for item in inputs) == Counter(expected), \
        "missing or duplicate per-layer FFN inputs"
    for item in inputs:
        evaluation = evaluations[item["evaluation"]]
        layer = int(item["name"].removeprefix("ffn_input-"))
        # The final layer discards prompt rows that do not produce logits.
        tokens = 1 if layer == layers - 1 else evaluation["input_tokens"]
        assert item["type"] == "f32" and item["shape"] == [width, tokens, 1, 1], item["name"]
        assert item["phase"] == evaluation["phase"]
        assert len(item["values"]) == width * tokens and all(map(math.isfinite, item["values"]))


def option_cases(pmi, root, shards):
    """Exercise new controls separately, preserving the original 22 cases."""
    common = ["--prompt", "hello world", "--max-tokens", 8, "--gpu-layers", 0,
              "--prefill-chunk", 4, "--context", 64]
    metadata, _ = fixture.fixture(expert_type="q8_0", ple_type="iq4_nl")
    values = {name: value for name, _, value in metadata}
    width, layers = values["qwen4exp.embedding_length"], values["qwen4exp.block_count"]
    previous = {
        "reference": json.loads((root / "iq4-ple-reference-chunk4.json").read_text()),
        "paged": json.loads((root / "iq4-ple-paged-chunk4-eviction.json").read_text()),
    }
    traces, paths, reports = {}, {}, []
    cases = [
        ("explicit-default", False, "default", "on", "full", previous["reference"]),
        ("default-plain", False, "default", "off", "full", "explicit-default"),
        ("demand-plain", False, "demand", "off", "full", "default-plain"),
        ("demand-compact", False, "demand", "off", "compact", "demand-plain"),
        ("paged-compact", True, None, "off", "compact", "demand-compact"),
        ("paged-plain", True, None, "off", "full", "demand-plain"),
        ("paged-repack", True, None, "on", "full", previous["paged"]),
        ("demand-forces-plain", False, "demand", "on", "compact", "demand-compact"),
    ]
    for name, paged, reference_mode, repack, trace_mode, baseline in cases:
        label = "options-" + name
        path = root / (label + ".json")
        command = ["run", *reversed(shards), "--cache-bytes", 4352] if paged else ["reference", shards[0]]
        if not paged:
            command += ["--reference-mode", reference_mode]
        invoke(pmi, root, label, [*command, *common, "--cpu-repack", repack,
                                 "--trace-mode", trace_mode, "--trace-json", path])
        actual = json.loads(path.read_text())
        baseline = traces[baseline] if isinstance(baseline, str) else baseline
        comparison = compare(baseline, actual, compact=trace_mode == "compact")
        evaluations = check_evaluation_metrics(actual, paged=paged, repack=repack == "on" and reference_mode != "demand",
                                               reference_mode=reference_mode, trace_mode=trace_mode, chunk=4)
        check_ffn_inputs(actual, evaluations, width=width, layers=layers)
        traces[name], paths[name] = actual, path
        reports.append({"case": name, "variant": "iq4-ple", "comparison": comparison, "metrics": actual["metrics"]})

    for full, compact in (("demand-plain", "demand-compact"), ("paged-plain", "paged-compact")):
        full_trace, compact_trace = traces[full], traces[compact]
        # Every retained record is compared, not merely the logits intersection.
        comparison = compare(full_trace, compact_trace, compact=True)
        payload = lambda trace: 4 * sum(len(item["values"]) for item in trace["records"])
        assert payload(compact_trace) < payload(full_trace), "compact tracing did not reduce logical payload"
        assert len(compact_trace["records"]) < len(full_trace["records"])
        assert paths[compact].stat().st_size < paths[full].stat().st_size, "compact JSON was not smaller"
        report = next(item for item in reports if item["case"] == compact)
        report["full_trace_comparison"] = comparison
        report["full_trace_payload_bytes"] = payload(full_trace)
        report["compact_trace_payload_bytes"] = payload(compact_trace)
    return reports


PROFILE_TIMINGS = (
    "provider_request_ms", "expert_request_ms", "lookup_request_ms", "lookup_decode_ms",
    "activation_copy_ms", "graph_setup_ms", "kernel_ms", "output_copy_ms",
    "graph_teardown_ms", "retention_ms",
)
PROFILE_COUNTS = ("profile_projection_calls", "profile_lookup_rows", "profile_provider_requests",
                  "profile_expert_requests", "profile_lookup_requests")


def check_paging_controls(trace, *, policy, profiling, capacity, quota, bundle_bytes, layers=4):
    metrics = trace["metrics"]
    assert metrics["cache_policy"] == policy
    assert metrics["profile_enabled"] is profiling
    assert metrics["cache_capacity_bytes"] == capacity
    assert metrics["cache_peak_reserved_bytes"] <= capacity
    assert metrics["retention_requested_experts"] == (4 if policy == "previous-top4" else 0)
    assert metrics["retention_effective_experts"] == quota
    assert metrics["retention_protected_bytes"] == layers * quota * bundle_bytes
    assert metrics["retention_peak_protected_bytes"] <= (layers + 1) * quota * bundle_bytes
    assert metrics["retention_peak_protected_bytes"] + metrics["retention_service_reserve_bytes"] <= capacity
    if quota:
        assert metrics["retention_transitions"] == layers * len(metrics["evaluations"])
    if policy == "previous-top4" and quota < 4:
        assert metrics["retention_fallbacks"] > 0
    for item in [metrics, *metrics["evaluations"]]:
        assert item["profile_enabled"] is profiling
        for name in PROFILE_TIMINGS:
            assert isinstance(item[name], (int, float)) and math.isfinite(item[name]) and item[name] >= 0, name
            if not profiling:
                assert item[name] == 0, f"disabled profile collected timing: {name}"
        for name in PROFILE_COUNTS:
            assert isinstance(item[name], int) and item[name] >= 0, name
            if not profiling:
                assert item[name] == 0, f"disabled profile collected counts: {name}"
        if profiling:
            assert item["profile_provider_requests"] == item["cache_hits"] + item["cache_misses"]
            assert item["profile_expert_requests"] == item["profile_projection_calls"]
            assert item["profile_lookup_requests"] == item["profile_lookup_rows"]
            assert item["profile_expert_requests"] + item["profile_lookup_requests"] == item["profile_provider_requests"]
            assert math.isclose(item["provider_request_ms"], item["expert_request_ms"] + item["lookup_request_ms"],
                                rel_tol=1e-8, abs_tol=1e-6)
        assert item["retention_protected_bytes"] <= item["retention_peak_protected_bytes"]
        assert item["retention_peak_protected_bytes"] + item["retention_service_reserve_bytes"] <= capacity
    for name in (*PROFILE_TIMINGS, *PROFILE_COUNTS, "retention_transitions", "retention_fallbacks"):
        assert math.isclose(sum(item[name] for item in metrics["evaluations"]), metrics[name],
                            rel_tol=1e-8, abs_tol=1e-6), f"paging evaluation metric did not reconcile: {name}"
    if profiling:
        assert metrics["profile_projection_calls"] == metrics["expert_requests"] > 0
        assert metrics["profile_lookup_rows"] == metrics["lookup_requests"] > 0
        assert metrics["provider_request_ms"] > 0 and metrics["kernel_ms"] > 0


def check_routing_jsonl(path, trace, *, layers=4):
    raw_lines = path.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in raw_lines]
    assert rows and rows[0]["type"] == "run" and rows[-1]["type"] == "complete"
    assert all(row["schema"] == "pmi.routing.v1" for row in rows)
    assert len({row["run_id"] for row in rows}) == 1 and rows[0]["run_id"]
    assert rows[-1]["complete"] is True and rows[-1]["truncated"] is False
    assert rows[-1]["records"] == len(rows) - 2
    assert rows[-1]["bytes_before_complete"] == sum(map(len, raw_lines[:-1]))
    assert all(row["type"] in ("run", "routing", "metrics", "complete") for row in rows)
    assert trace["metrics"]["routing_observation"] is True

    forbidden = {"prompt", "prompt_text", "text", "tokens", "token_ids", "input_token_ids",
                 "generated_token_ids", "hidden_states", "activations", "logits", "values"}
    def private_fields(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value), "routing-only observation leaked model inputs/activations"
            for child in value.values():
                private_fields(child)
        elif isinstance(value, list):
            for child in value:
                private_fields(child)
    private_fields(rows)
    assert b"hello world" not in b"".join(raw_lines)

    evaluations = trace["metrics"]["evaluations"]
    starts, position = {}, 0
    for evaluation in evaluations:
        starts[evaluation["evaluation"]] = position
        position += evaluation["input_tokens"]
    weights = {(row["evaluation"], int(row["name"].rsplit("-", 1)[1])): row
               for row in trace["records"] if row["name"].startswith("ffn_moe_weights_norm-")}
    expected = {}
    for item in trace["records"]:
        if not item["name"].startswith("ffn_moe_topk-"):
            continue
        evaluation, layer = item["evaluation"], int(item["name"].rsplit("-", 1)[1])
        slots, count, _, _ = item["shape"]
        model_input_count = evaluations[evaluation]["input_tokens"]
        assert count == (1 if layer == layers - 1 else model_input_count)
        probability = weights[evaluation, layer]
        assert probability["shape"] == item["shape"]
        for column in range(count):
            index = model_input_count - 1 if layer == layers - 1 else column
            key = evaluation, layer, index
            assert key not in expected
            expected[key] = (starts[evaluation] + index, item["phase"],
                             item["values"][column * slots:(column + 1) * slots],
                             probability["values"][column * slots:(column + 1) * slots])
    seen = set()
    allowed = {"schema", "type", "run_id", "sequence_id", "eval_id", "phase", "layer",
               "token_index", "token_position", "expert_ids", "routing_weights", "complete", "truncated"}
    routing = [row for row in rows if row["type"] == "routing"]
    for row in routing:
        assert set(row) == allowed, "routing record exceeded its explicit field allowlist"
        key = row["eval_id"], row["layer"], row["token_index"]
        assert key in expected and key not in seen
        seen.add(key)
        position, phase, ids, probabilities = expected[key]
        assert row["sequence_id"] == "0" and row["complete"] is True and row["truncated"] is False
        assert row["token_position"] == position and row["phase"] == phase
        assert row["expert_ids"] == ids and row["routing_weights"] == probabilities
        assert all(math.isfinite(value) and value >= 0 for value in probabilities)
        assert math.isclose(sum(probabilities), 1, rel_tol=1e-6, abs_tol=1e-6)
    assert seen == expected.keys(), "routing-only capture omitted rows from the compact trace"

    profiles = [row for row in rows if row["type"] == "metrics" and row["eval_id"] is not None]
    assert [row["eval_id"] for row in profiles] == list(range(len(evaluations)))
    for row in profiles:
        assert row["per_layer_cache"] is None
        evaluation = evaluations[row["eval_id"]]
        assert row["phase"] == evaluation["phase"]
        for name in ("input_tokens", "elapsed_ms", "io_bytes", "cache_hits", "cache_misses", "evictions",
                     *PROFILE_TIMINGS, *PROFILE_COUNTS, "retention_transitions", "retention_protected_bytes",
                     "retention_peak_protected_bytes", "retention_service_reserve_bytes"):
            assert math.isclose(row["metrics"][name], evaluation[name], rel_tol=1e-8, abs_tol=1e-6), name
        assert row["metrics"]["profile_enabled"] is True
    totals = [row for row in rows if row["type"] == "metrics" and row["eval_id"] is None]
    # Evaluations are already streamed individually; repeating them here can
    # exhaust the record-size cap even when every routing row fits the file.
    assert len(totals) == 1 and "evaluations" not in totals[0]["metrics"]
    assert totals[0]["metrics"] == {k: v for k, v in trace["metrics"].items() if k != "evaluations"}
    return {"routing_rows": len(routing), "evaluations": len(profiles), "bytes": sum(map(len, raw_lines)),
            "completed": True, "matches_compact_trace": True}


def retention_cases(pmi, root, generated):
    """Exercise the opt-in policy against unchanged native full-graph traces."""
    reports = []
    for chunk in (1, 4, 64):
        reference = json.loads((root / f"qsa-long-reference-{chunk}.json").read_text())
        common = ["--prompt", "hello world " * 4, "--max-tokens", 8, "--gpu-layers", 0,
                  "--prefill-chunk", chunk, "--context", 64, "--cpu-repack", "off"]
        off_trace = None
        for policy, profiling in (("lru", False), ("previous-top4", False), ("previous-top4", True)):
            label = f"retention-{policy}-chunk{chunk}-profile{'on' if profiling else 'off'}"
            path = root / (label + ".json")
            observation_path = root / (label + ".routing.jsonl")
            observation_args = []
            if profiling:
                observation_path.unlink(missing_ok=True)  # This test owns its synthetic output.
                observation_args = ["--trace-mode", "compact", "--routing-jsonl", observation_path]
            invoke(pmi, root, label, ["run", *generated["f32"], *common,
                                    "--cache-bytes", 262144, "--cache-policy", policy,
                                    "--profile-paging", "on" if profiling else "off", "--trace-json", path,
                                    *observation_args])
            actual = json.loads(path.read_text())
            comparison = compare(reference, actual, compact=profiling)
            check_paging_controls(actual, policy=policy, profiling=profiling, capacity=262144,
                                  quota=2 if policy == "previous-top4" else 0, bundle_bytes=24576)
            if policy == "previous-top4":
                if profiling:
                    assert off_trace is not None
                    compare(off_trace, actual, compact=True)
                    for key in ("page_requests", "expert_requests", "lookup_requests", "io_bytes",
                                "cache_hits", "cache_misses", "evictions"):
                        assert actual["metrics"][key] == off_trace["metrics"][key], \
                            f"profiling changed deterministic paging behavior: {key}"
                else:
                    off_trace = actual
            reports.append({"case": label, "comparison": comparison, "metrics": actual["metrics"]})
            if profiling:
                reports[-1]["routing_observation"] = check_routing_jsonl(observation_path, actual)
                if chunk == 1:
                    before = observation_path.read_bytes()
                    invoke(pmi, root, "routing-existing-output-rejected",
                           ["run", *generated["f32"], *common, "--routing-jsonl", observation_path], success=False)
                    assert observation_path.read_bytes() == before, "existing routing evidence was replaced"

    # A one-component budget must disable protection without losing the normal
    # eviction/reload execution path or changing any model output.
    label, path = "retention-tiny-fallback", root / "retention-tiny-fallback.json"
    invoke(pmi, root, label, ["run", *generated["f32"], "--prompt", "hello world " * 4,
                            "--max-tokens", 8, "--prefill-chunk", 4, "--context", 64,
                            "--cache-bytes", 8192, "--cache-policy", "previous-top4",
                            "--profile-paging", "on", "--trace-json", path])
    actual = json.loads(path.read_text())
    reference = json.loads((root / "qsa-long-reference-4.json").read_text())
    comparison = compare(reference, actual)
    check_paging_controls(actual, policy="previous-top4", profiling=True, capacity=8192,
                          quota=0, bundle_bytes=24576)
    assert actual["metrics"]["evictions"] > 0
    reports.append({"case": label, "comparison": comparison, "metrics": actual["metrics"]})

    # Quantized expert projections and non-F32 PLE decoding also contribute to
    # the profile; timers must not accidentally cover only the float fixture.
    label, path = "retention-quantized-profile", root / "retention-quantized-profile.json"
    invoke(pmi, root, label, ["run", *generated["iq4-ple"], "--prompt", "hello world",
                            "--max-tokens", 8, "--prefill-chunk", 4, "--context", 64,
                            "--cache-bytes", 262144, "--cache-policy", "previous-top4",
                            "--profile-paging", "on", "--trace-json", path])
    actual = json.loads(path.read_text())
    reference = json.loads((root / "iq4-ple-reference-chunk4.json").read_text())
    comparison = compare(reference, actual)
    check_paging_controls(actual, policy="previous-top4", profiling=True, capacity=262144,
                          quota=2, bundle_bytes=13056)
    assert actual["metrics"]["lookup_decode_ms"] > 0
    reports.append({"case": label, "comparison": comparison, "metrics": actual["metrics"]})

    for option, value in (("--cache-policy", "unknown"), ("--profile-paging", "sometimes")):
        invoke(pmi, root, "retention-invalid-" + option.removeprefix("--"),
               ["run", *generated["f32"], "--prompt", "hello", option, value], success=False)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pmi", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-only", action="store_true", help="explicitly check fixtures before paging integration exists")
    args = parser.parse_args()
    pmi, root = args.pmi.resolve(), args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    variants = [
        ("f32", {"expert_type": "f32"}, 8192),
        ("f16", {"expert_type": "f16", "ple_type": "f16"}, 4096),
        ("q8", {"expert_type": "q8_0"}, 2176),
        ("iq4-ple", {"expert_type": "q8_0", "ple_type": "iq4_nl"}, 4352),
    ]
    reports, untraced_reports, mode_reports, retention_reports, baseline_tokens, generated = [], [], [], [], {}, {}
    for name, kwargs, forced_capacity in variants:
        _, tensor_data = fixture.fixture(**kwargs)
        deferred_bytes = sum(len(data) for tensor_name, _, _, data in tensor_data
                             if "_exps.weight" in tensor_name or tensor_name == "per_layer_token_embd.weight")
        single = fixture.generate(root / (name + ".gguf"), **kwargs)
        split = fixture.generate(root / (name + "-split.gguf"), shards=2, **kwargs)
        generated[name] = split
        # The generated bytes must be independent of destination and repeat.
        duplicate = fixture.generate(root / (name + "-copy.gguf"), **kwargs)
        assert single[0].read_bytes() == duplicate[0].read_bytes(), "nondeterministic fixture generation"
        duplicate[0].unlink()
        duplicate[0].with_suffix(".json").unlink()
        common = ["--prompt", "hello world", "--max-tokens", "8", "--gpu-layers", "0"]
        reference_tokens = invoke(pmi, root, name + "-reference", ["reference", single[0], *common])
        split_tokens = invoke(pmi, root, name + "-split-reference", ["reference", split[0], *common])
        assert reference_tokens == split_tokens, "sharding changed reference generation"
        baseline_tokens[name] = reference_tokens
        if args.reference_only:
            reports.append({"variant": name, "tokens": reference_tokens, "reference_shard_parity": True})
            continue

        for chunk in (1, 4):
            ref_trace_path = root / f"{name}-reference-chunk{chunk}.json"
            invoke(pmi, root, f"{name}-reference-chunk{chunk}",
                   ["reference", split[0], *common, "--prefill-chunk", chunk, "--context", 64,
                    "--trace-json", ref_trace_path])
            reference = json.loads(ref_trace_path.read_text())
            assert reference["tokens"] == reference_tokens, "reference prefill chunking changed greedy generation"
            warm_io = None
            for forced in (False, True):
                capacity = forced_capacity if forced else 4 * 1024 * 1024
                label = f"{name}-paged-chunk{chunk}-{'eviction' if forced else 'warm'}"
                trace_path = root / (label + ".json")
                tokens = invoke(pmi, root, label,
                                ["run", *reversed(split), *common, "--prefill-chunk", chunk,
                                 "--context", 64, "--cache-bytes", capacity, "--trace-json", trace_path])
                actual = json.loads(trace_path.read_text())
                assert actual["tokens"] == tokens
                comparison = compare(reference, actual)
                metrics = actual["metrics"]
                assert metrics["cache_policy"] == "lru" and metrics["profile_enabled"] is False, \
                    "new opt-in controls changed default execution"
                assert metrics["retention_requested_experts"] == metrics["retention_effective_experts"] == 0
                assert metrics["io_bytes"] > 0, "paging reported no reads"
                assert metrics["expert_requests"] > 0 and metrics["lookup_requests"] > 0, "expert or PLE path bypassed paging"
                assert metrics["deferred_weight_bytes"] == deferred_bytes, "expert/PLE deferred byte inventory mismatch"
                assert metrics["cache_capacity_bytes"] == capacity
                assert metrics["cache_peak_reserved_bytes"] <= capacity, "page payload cache exceeded configured capacity"
                assert metrics["cache_misses"] > 0
                if forced:
                    assert metrics["evictions"] > 0, "small cache did not force eviction"
                    assert metrics["io_bytes"] > warm_io, "evicted pages were not reloaded"
                else:
                    assert metrics["cache_hits"] > 0, "decode did not reuse warm pages"
                    assert metrics["io_bytes"] <= deferred_bytes, "large cache unexpectedly reloaded an immutable page"
                    warm_io = metrics["io_bytes"]
                reports.append({"variant": name, "prefill_chunk": chunk, "forced_eviction": forced,
                                "tokens": tokens, "comparison": comparison, "metrics": metrics})

        # The evaluation callback splits scheduler execution to obtain traces.
        # Also verify the ordinary public run without that instrumentation.
        label = name + "-untraced"
        tokens = invoke(pmi, root, label,
                        ["run", *reversed(split), *common, "--prefill-chunk", 4,
                         "--context", 64, "--cache-bytes", forced_capacity])
        assert tokens == reference_tokens, "disabling trace collection changed generation"
        lines = (root / (label + ".stdout")).read_text().splitlines()
        metrics = json.loads(next(line[len("PMI_METRICS "):] for line in lines if line.startswith("PMI_METRICS ")))
        assert metrics["expert_requests"] > 0 and metrics["lookup_requests"] > 0
        assert metrics["evictions"] > 0 and metrics["cache_peak_reserved_bytes"] <= forced_capacity
        untraced_reports.append({"variant": name, "tokens": tokens, "metrics": metrics})

        invoke(pmi, root, name + "-too-small-cache",
               ["run", *split, *common, "--cache-bytes", "1"], success=False)
        invoke(pmi, root, name + "-incomplete-shards", ["run", split[0], *common], success=False)

    if not args.reference_only:
        # Caller-supplied shard paths need not follow llama.cpp's implicit
        # -00001-of-00002 naming convention. Reversing unrelated filenames
        # catches accidental auto-discovery instead of explicit path loading.
        renamed = [root / "weights-arbitrary-a.gguf", root / "weights-unrelated-z.gguf"]
        for source, destination in zip(generated["f32"], renamed):
            shutil.copyfile(source, destination)
        renamed_trace = root / "renamed-shards-paged.json"
        invoke(pmi, root, "renamed-shards-paged",
               ["run", *reversed(renamed), "--prompt", "hello world", "--max-tokens", 8,
                "--gpu-layers", 0, "--prefill-chunk", 4, "--context", 64,
                "--cache-bytes", 8192, "--trace-json", renamed_trace])
        reference = json.loads((root / "f32-reference-chunk4.json").read_text())
        actual = json.loads(renamed_trace.read_text())
        reports.append({"variant": "f32-renamed-shards", "comparison": compare(reference, actual),
                        "metrics": actual["metrics"]})

        # An explicit EOS in the prompt resets PLE's n-gram history. Compare
        # across the chunk boundary as well as the ordinary decode transitions.
        eos_tokens = None
        for chunk in (1, 4):
            common = ["--prompt", "hello</s>world", "--max-tokens", 8, "--gpu-layers", 0,
                      "--prefill-chunk", chunk, "--context", 32]
            reference_path = root / f"eos-reference-{chunk}.json"
            actual_path = root / f"eos-paged-{chunk}.json"
            tokens = invoke(pmi, root, f"eos-reference-{chunk}",
                            ["reference", generated["f32"][0], *common, "--trace-json", reference_path])
            if eos_tokens is None:
                eos_tokens = tokens
            assert tokens == eos_tokens, "EOS-reset reference generation depends on prefill chunking"
            invoke(pmi, root, f"eos-paged-{chunk}",
                   ["run", *generated["f32"], *common, "--cache-bytes", 8192, "--trace-json", actual_path])
            reference, actual = json.loads(reference_path.read_text()), json.loads(actual_path.read_text())
            reports.append({"variant": "f32-eos-reset", "prefill_chunk": chunk,
                            "comparison": compare(reference, actual), "metrics": actual["metrics"]})

        # Far beyond the eight-token QSA budget: many completed two-token pools,
        # transitions between odd/even tails and a prefill/decode boundary. The
        # context deliberately leaves little spare room beyond this prompt.
        long_tokens = None
        for chunk in (1, 4, 64):
            common = ["--prompt", "hello world " * 4, "--max-tokens", 8, "--gpu-layers", 0,
                      "--prefill-chunk", chunk, "--context", 64]
            reference_path = root / f"qsa-long-reference-{chunk}.json"
            actual_path = root / f"qsa-long-paged-{chunk}.json"
            tokens = invoke(pmi, root, f"qsa-long-reference-{chunk}",
                            ["reference", generated["f32"][0], *common, "--trace-json", reference_path])
            if long_tokens is None:
                long_tokens = tokens
            assert tokens == long_tokens, "long-context reference generation depends on prefill chunking"
            invoke(pmi, root, f"qsa-long-paged-{chunk}",
                   ["run", *generated["f32"], *common, "--cache-bytes", 8192, "--trace-json", actual_path])
            reference, actual = json.loads(reference_path.read_text()), json.loads(actual_path.read_text())
            assert any(record["name"].startswith("indexer_top_k") for record in actual["records"]), "QSA selection IDs not traced"
            reports.append({"variant": "f32-qsa-long", "prefill_chunk": chunk,
                            "comparison": compare(reference, actual), "metrics": actual["metrics"]})

        mode_reports = option_cases(pmi, root, generated["iq4-ple"])
        retention_reports = retention_cases(pmi, root, generated)

    # A second deterministic seed prevents a fixture accidentally becoming an
    # all-zero graph or a constant token generator hidden by one golden prompt.
    second = fixture.generate(root / "second-seed.gguf", seed=20261003)
    other = invoke(pmi, root, "second-seed-reference",
                   ["reference", second[0], "--prompt", "hello world", "--max-tokens", 8, "--gpu-layers", 0])
    assert other != baseline_tokens["f32"], "changing all synthetic weights did not affect generation"
    (root / "full-model-results.json").write_text(json.dumps({
        "scope": "synthetic complete Qwen4Exp graph; no trained-model language-quality claim",
        "reference_only": args.reference_only, "reports": reports, "untraced_reports": untraced_reports,
        "mode_reports": mode_reports, "retention_reports": retention_reports,
    }, indent=2) + "\n")
    print(f"Verified {len(variants)} complete Qwen fixtures; {len(reports)} cases; "
          f"{len(mode_reports)} option cases; {len(retention_reports)} retention/profile cases; "
          f"reference_only={args.reference_only}")


if __name__ == "__main__":
    main()
