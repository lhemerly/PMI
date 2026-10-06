#!/usr/bin/env python3
"""Compare opt-in device projections on tiny generated Qwen fixtures.

No trained weights are accessed. The default suite uses CPU devices only;
--vulkan explicitly enables a separate small hardware integration suite.
Native resident execution, the default
PMI CPU path, and the selected device projection path use the same fixture,
prompt, greedy decoder and CPU repacking setting. This establishes graph wiring
and bounded-cache correctness, not broad hardware support or model-quality evidence.
"""

import argparse
from collections import Counter
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pmi_full_model_helpers", SOURCE / "tests/full_model_test.py")
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)

ATOL, RTOL = 2e-5, 1e-4
WARM_DEVICE_BYTES = 4 * 1024 * 1024
GRAPH_BYTES = 1024 * 1024
MAX_REPORT_BYTES = 2 * 1024 * 1024
DEVICE_COUNTS = ("calls", "completed", "failures", "activation_upload_bytes", "output_readback_bytes")
DEVICE_TIMINGS = ("cache_request_ms", "graph_setup_ms", "activation_upload_ms", "kernel_ms",
                  "output_readback_ms", "graph_teardown_ms")
CACHE_COUNTS = ("requests", "requested_bytes", "hits", "misses", "failures", "provider_requests",
                "upload_bytes", "uploads", "evictions")
CACHE_TIMINGS = ("provider_read_ms", "allocation_ms", "upload_ms", "eviction_ms")
CACHE_GAUGES = ("current_bytes", "peak_bytes", "capacity_bytes", "entries", "active_leases")
VULKAN_COMPARISON_POLICY = {
    "name": "Vulkan mixed precision comparison",
    "default_absolute_tolerance": ATOL, "default_relative_tolerance": RTOL,
    "pooled_key_name": "indexer_pool_k_new-<layer>", "pooled_key_vector_width": "shape[0]",
    "pooled_key_nmse_limit": 1e-7, "zero_reference_requires_zero_actual": True,
    "no_averaging_across_vectors_or_evaluations": True,
    "source": "vendor/llama.cpp/tests/test-backend-ops.cpp:nmse,max_nmse_err,test_rms_norm",
    "reason": "one-F32-ULP input differences can cross an FP16 indexer-cache midpoint before pooling/RMS/RoPE",
    "exact_requirements": "tokens, every integer value, and all snapshot signatures including multiplicity",
}


def finite_nonnegative(value, label):
    assert isinstance(value, (int, float)) and not isinstance(value, bool), label
    assert math.isfinite(value) and value >= 0, label


def disabled_projection(metrics):
    expected = None if metrics["path"] == "llama_reference" else "cpu"
    assert metrics["projection_backend"] == expected
    for item in (metrics, *metrics["evaluations"]):
        device = item["device_projection"]
        assert device["enabled"] is False and device["profile_enabled"] is False
        for name in (*DEVICE_COUNTS, *DEVICE_TIMINGS, "graph_current_bytes", "graph_peak_bytes", "graph_capacity_bytes"):
            assert device[name] == 0, f"disabled projection path collected device state: {name}"
        for name in (*CACHE_COUNTS, *CACHE_GAUGES):
            assert device["cache"][name] == 0, f"disabled projection path collected device cache state: {name}"


def check_device_metrics(trace, *, profiling, host_capacity, device_capacity, chunk, backend="device-cpu"):
    metrics = trace["metrics"]
    expected_path = "pmi_paged_qwen_vulkan_projections" if backend == "device-vulkan" else "pmi_paged_qwen_cpu"
    assert metrics["path"] == expected_path and metrics["projection_backend"] == backend
    assert metrics["cpu_repack"] is False and metrics["cache_policy"] == "lru"
    assert metrics["reference_mode"] is None and metrics["profile_enabled"] is profiling
    assert metrics["generated_tokens"] == len(trace["tokens"]) == 8
    assert metrics["cache_capacity_bytes"] == host_capacity
    assert metrics["cache_reserved_bytes"] <= metrics["cache_peak_reserved_bytes"] <= host_capacity
    assert metrics["retention_requested_experts"] == metrics["retention_effective_experts"] == 0
    assert metrics["retention_protected_bytes"] == metrics["retention_peak_protected_bytes"] == 0
    assert metrics["page_requests"] == metrics["expert_requests"] + metrics["lookup_requests"]
    assert metrics["expert_requests"] > 0 and metrics["lookup_requests"] > 0
    assert metrics["io_bytes"] > 0
    evaluations = metrics["evaluations"]
    prefill = metrics["prefill_evaluations"]
    assert prefill > 0 and metrics["decode_evaluations"] == 7
    assert len(evaluations) == prefill + 7
    assert [row["evaluation"] for row in evaluations] == list(range(len(evaluations)))
    assert [row["phase"] for row in evaluations] == ["prefill"] * prefill + ["decode"] * 7
    assert sum(row["input_tokens"] for row in evaluations[:prefill]) == metrics["prompt_tokens"]
    assert all(0 < row["input_tokens"] <= chunk for row in evaluations[:prefill])
    assert all(row["input_tokens"] == 1 for row in evaluations[prefill:])
    for name in ("io_bytes", "cache_hits", "cache_misses", "evictions"):
        assert sum(row[name] for row in evaluations) == metrics[name], f"host evaluation counter mismatch: {name}"
    for phase in ("prefill", "decode"):
        assert math.isclose(sum(row["elapsed_ms"] for row in evaluations if row["phase"] == phase),
                            metrics[phase + "_ms"], rel_tol=1e-8, abs_tol=1e-6)

    for item in (metrics, *evaluations):
        device, cache = item["device_projection"], item["device_projection"]["cache"]
        assert device["enabled"] is True and device["profile_enabled"] is profiling
        assert cache["profile_enabled"] is profiling
        for name in CACHE_TIMINGS:
            finite_nonnegative(cache[name], name)
            if not profiling:
                assert cache[name] == 0, f"disabled cache profiling measured {name}"
        for name in DEVICE_COUNTS:
            assert isinstance(device[name], int) and device[name] >= 0, name
        for name in (*CACHE_COUNTS, *CACHE_GAUGES):
            assert isinstance(cache[name], int) and cache[name] >= 0, name
        for name in DEVICE_TIMINGS:
            finite_nonnegative(device[name], name)
            if not profiling:
                assert device[name] == 0, f"disabled device profiling measured {name}"
        for name in (*full.PROFILE_TIMINGS, *full.PROFILE_COUNTS):
            finite_nonnegative(item[name], name)
            if not profiling:
                assert item[name] == 0, f"disabled graph profiling measured {name}"
        assert device["calls"] == device["completed"] == cache["requests"]
        assert device["failures"] == cache["failures"] == 0
        assert cache["requests"] == cache["hits"] + cache["misses"]
        assert cache["misses"] == cache["uploads"] == cache["provider_requests"]
        assert cache["current_bytes"] <= cache["peak_bytes"] <= cache["capacity_bytes"] == device_capacity
        assert cache["active_leases"] == 0, "completed projection retained a device consumer lease"
        assert device["graph_current_bytes"] == 0, "completed projection retained its temporary graph buffer"
        assert 0 < device["graph_peak_bytes"] <= device["graph_capacity_bytes"] == GRAPH_BYTES
    device, cache = metrics["device_projection"], metrics["device_projection"]["cache"]
    assert device["calls"] == metrics["expert_requests"]
    assert metrics["cache_hits"] + metrics["cache_misses"] == cache["provider_requests"] + metrics["lookup_requests"]
    assert device["activation_upload_bytes"] > 0 and device["output_readback_bytes"] > 0
    assert cache["upload_bytes"] > 0
    for name in (*DEVICE_COUNTS, *DEVICE_TIMINGS):
        assert math.isclose(sum(row["device_projection"][name] for row in evaluations), device[name],
                            rel_tol=1e-8, abs_tol=1e-6), f"device evaluation metric mismatch: {name}"
    for name in CACHE_COUNTS:
        assert sum(row["device_projection"]["cache"][name] for row in evaluations) == cache[name], \
            f"device cache evaluation counter mismatch: {name}"
    if profiling:
        assert device["kernel_ms"] > 0 and device["cache_request_ms"] > 0
    return evaluations


def scalar_metrics(trace):
    return {name: value for name, value in trace["metrics"].items() if name != "evaluations"}


def run_trace(pmi, root, label, arguments):
    path = root / (label + ".json")
    tokens = full.invoke(pmi, root, label, [*arguments, "--trace-json", path])
    trace = json.loads(path.read_text())
    assert trace["tokens"] == tokens, "trace tokens disagree with public command output"
    return trace, path


def compare_all(reference, direct, actual):
    check_router_trace(direct, actual)
    return {
        "native": full.compare(reference, actual, atol=ATOL, rtol=RTOL),
        "direct_pmi_cpu": full.compare(direct, actual, atol=ATOL, rtol=RTOL),
    }


def check_router_trace(reference, actual):
    """Require every routed-layer snapshot, not just intersecting trace keys."""
    labels = ("ffn_moe_topk", "ffn_moe_weights_norm", "ffn_moe_weights_scaled", "ffn_moe_logits")
    def indexed(trace):
        result, seen = {}, Counter()
        for record in trace["records"]:
            if not any(record["name"] == label or record["name"].startswith(label + "-") for label in labels):
                continue
            signature = (record["evaluation"], record["phase"], record["name"], record["type"], tuple(record["shape"]))
            occurrence = seen[signature]
            seen[signature] += 1
            result[signature + (occurrence,)] = record["values"]
        return result
    expected, measured = indexed(reference), indexed(actual)
    assert any(key[2].startswith("ffn_moe_topk") for key in expected), "CPU control omitted routing IDs"
    assert any(key[2].startswith("ffn_moe_weights_norm") for key in expected), "CPU control omitted normalized routing weights"
    assert expected.keys() == measured.keys(), (
        "routing snapshot coverage mismatch: "
        f"missing={sorted(expected.keys() - measured.keys())[:8]}, "
        f"extra={sorted(measured.keys() - expected.keys())[:8]}"
    )
    for key in expected:
        if key[3] == "i32":
            assert expected[key] == measured[key], f"exact routing ID mismatch: {key}"


def negative_cases(pmi, root, shards):
    common = ["--prompt", "hello world", "--max-tokens", 8, "--gpu-layers", 0,
              "--cpu-repack", "off", "--prefill-chunk", 4, "--context", 64]
    budgets = ["--device-cache-bytes", WARM_DEVICE_BYTES, "--device-graph-bytes", GRAPH_BYTES]
    selected = ["--projection-backend", "device-cpu"]
    cases = [
        ("cache-too-small", "run", [*selected, "--device-cache-bytes", 1, "--device-graph-bytes", GRAPH_BYTES]),
        ("graph-too-small", "run", [*selected, "--device-cache-bytes", WARM_DEVICE_BYTES, "--device-graph-bytes", 1]),
        ("unknown-backend", "run", ["--projection-backend", "unknown", *budgets]),
        ("missing-both-budgets", "run", selected),
        ("missing-cache-budget", "run", [*selected, "--device-graph-bytes", GRAPH_BYTES]),
        ("missing-graph-budget", "run", [*selected, "--device-cache-bytes", WARM_DEVICE_BYTES]),
        ("zero-cache-budget", "run", [*selected, "--device-cache-bytes", 0, "--device-graph-bytes", GRAPH_BYTES]),
        ("zero-graph-budget", "run", [*selected, "--device-cache-bytes", WARM_DEVICE_BYTES, "--device-graph-bytes", 0]),
        ("default-with-budgets", "run", budgets),
        ("explicit-cpu-with-budgets", "run", ["--projection-backend", "cpu", *budgets]),
        ("default-cache-budget", "run", ["--device-cache-bytes", WARM_DEVICE_BYTES]),
        ("default-graph-budget", "run", ["--device-graph-bytes", GRAPH_BYTES]),
        ("default-zero-budgets", "run", ["--device-cache-bytes", 0, "--device-graph-bytes", 0]),
        ("reference-device", "reference", [*selected, *budgets]),
        ("reference-explicit-cpu", "reference", ["--projection-backend", "cpu"]),
        ("reference-cache-budget", "reference", ["--device-cache-bytes", WARM_DEVICE_BYTES]),
        ("reference-graph-budget", "reference", ["--device-graph-bytes", GRAPH_BYTES]),
        ("incompatible-retention", "run", [*selected, *budgets, "--cache-policy", "previous-top4"]),
    ]
    reports = []
    for name, command, options in cases:
        paths = shards if command == "run" else shards[:1]
        label = "device-invalid-" + name
        full.invoke(pmi, root, label, [command, *paths, *common, *options], success=False)
        stdout = (root / (label + ".stdout")).read_bytes()
        stderr = (root / (label + ".stderr")).read_bytes()
        assert b"PMI_METRICS " not in stdout, f"{name}: invalid request reported successful generation metrics"
        assert stderr.strip(), f"{name}: rejected request omitted its diagnostic"
        reports.append({"case": name, "rejected": True, "stdout_bytes": len(stdout), "stderr_bytes": len(stderr)})
    return reports


def comparison_diagnostics(reference, actual):
    """Preserve error distribution on strict GPU parity failure; never relax it."""
    def indexed(trace):
        result, seen = {}, Counter()
        for row in trace["records"]:
            key = (row["evaluation"], row["phase"], row["name"], row["type"], tuple(row["shape"]))
            occurrence = seen[key]
            seen[key] += 1
            result[key + (occurrence,)] = row["values"]
        return result
    expected, measured = indexed(reference), indexed(actual)
    groups = {}
    for key in expected.keys() & measured.keys():
        name, kind = key[2], key[3]
        entry = groups.setdefault(name, {"name": name, "values": 0, "nonfinite": 0, "outside_tolerance": 0,
                                        "max_abs_error": 0.0, "sum_abs_error": 0.0})
        for left, right in zip(expected[key], measured[key]):
            entry["values"] += 1
            if not math.isfinite(left) or not math.isfinite(right):
                entry["nonfinite"] += 1
                continue
            error = abs(left - right)
            entry["sum_abs_error"] += error
            entry["max_abs_error"] = max(entry["max_abs_error"], error)
            if (left != right if kind == "i32" else error > ATOL + RTOL * abs(left)):
                entry["outside_tolerance"] += 1
    for entry in groups.values():
        entry["mean_abs_error"] = entry.pop("sum_abs_error") / max(1, entry["values"] - entry["nonfinite"])
    ordered = sorted(groups.values(), key=lambda value: (value["outside_tolerance"], value["max_abs_error"]), reverse=True)
    return {"tokens_equal": reference["tokens"] == actual["tokens"], "absolute_tolerance": ATOL,
            "relative_tolerance": RTOL, "common_records": len(expected.keys() & measured.keys()),
            "reference_only_records": len(expected.keys() - measured.keys()),
            "actual_only_records": len(measured.keys() - expected.keys()), "intermediates": ordered}


def q8_values(payload):
    assert len(payload) % 34 == 0, "Q8_0 payload has an incomplete block"
    values = []
    for offset in range(0, len(payload), 34):
        scale, *codes = struct.unpack_from("<e32b", payload, offset)
        assert math.isfinite(scale)
        values.extend(scale * code for code in codes)
    return values


def make_q8_expert_oracle(output):
    """Expand stored Q8 expert values, retaining every nonexpert tensor byte."""
    metadata, original = full.fixture.fixture(expert_type="q8_0", ple_type="iq4_nl")
    converted, experts, preserved = [], [], []
    largest_page = 0
    for name, shape, kind, payload in original:
        if "_exps.weight" in name:
            assert kind == 8 and len(shape) == 3
            values = q8_values(payload)
            assert len(values) == math.prod(shape)
            expanded = struct.pack("<" + "f" * len(values), *values)
            assert list(struct.unpack("<" + "f" * len(values), expanded)) == values, \
                "stored Q8 scale times code did not round-trip F32 exactly"
            converted.append((name, shape, 0, expanded))
            largest_page = max(largest_page, len(expanded) // shape[2])
            experts.append({"name": name, "shape": list(shape), "source_type": "q8_0", "oracle_type": "f32",
                            "source_bytes": len(payload), "oracle_bytes": len(expanded),
                            "source_sha256": hashlib.sha256(payload).hexdigest(),
                            "oracle_sha256": hashlib.sha256(expanded).hexdigest()})
        else:
            converted.append((name, shape, kind, payload))
            assert converted[-1] == (name, shape, kind, payload)
            preserved.append({"name": name, "shape": list(shape), "ggml_type": kind,
                              "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    assert len(experts) == 12 and len(preserved) == len(original) - 12
    assert largest_page == 16384
    oracle_metadata = [(name, kind, 0 if name == "general.file_type" else value) for name, kind, value in metadata]
    assert [(n, k, v) for n, k, v in oracle_metadata if n != "general.file_type"] == \
           [(n, k, v) for n, k, v in metadata if n != "general.file_type"]
    paths = []
    for shard in range(2):
        path = output.with_name(f"{output.stem}-{shard + 1:05d}-of-00002.gguf")
        shard_meta = oracle_metadata + [("split.no", 2, shard), ("split.count", 2, 2),
                                       ("split.tensors.count", 5, len(converted))]
        full.fixture.write_gguf(path, shard_meta, converted[shard::2])
        paths.append(path)
    audit = {"semantics": "exact stored Q8 weights expanded to F32; CPU uses F32 activations instead of Q8_0 activation rounding",
             "recipe": "decode each little-endian 34-byte block as <e32b, store F32(scale * signed_code)",
             "source_pointer": "vendor/llama.cpp/ggml/src/ggml-quants.c:dequantize_row_q8_0",
             "metadata_change": "general.file_type:7->0; all semantic metadata/tokenizer unchanged",
             "expert_tensors": experts, "byte_identical_nonexpert_tensors": preserved,
             "source_expert_bytes": sum(item["source_bytes"] for item in experts),
             "oracle_expert_bytes": sum(item["oracle_bytes"] for item in experts),
             "largest_oracle_page_bytes": largest_page,
             "files": [{"name": path.name, "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]}
    return paths, largest_page, audit


def check_all_integer_snapshots(reference, actual):
    def indexed(trace):
        result, seen = {}, Counter()
        for row in trace["records"]:
            if row["type"] != "i32":
                continue
            key = (row["evaluation"], row["phase"], row["name"], tuple(row["shape"]))
            occurrence = seen[key]
            seen[key] += 1
            result[key + (occurrence,)] = row["values"]
        return result
    left, right = indexed(reference), indexed(actual)
    assert left and left.keys() == right.keys(), "integer trace coverage mismatch"
    assert left == right, "routing/indexer integer values differ"
    return sum(len(values) for values in left.values())


def compare_vulkan_policy(reference, actual):
    """GPU-only operator criterion; preserve scalar tolerance everywhere else."""
    def pooled(name):
        prefix = "indexer_pool_k_new-"
        return name.startswith(prefix) and name[len(prefix):].isdecimal()
    def indexed(trace):
        result, seen = {}, Counter()
        for row in trace["records"]:
            key = (row["evaluation"], row["phase"], row["name"], row["type"], tuple(row["shape"]))
            occurrence = seen[key]
            seen[key] += 1
            result[key + (occurrence,)] = row
        return result
    left, right = indexed(reference), indexed(actual)
    assert reference["tokens"] == actual["tokens"], "greedy token mismatch"
    assert left.keys() == right.keys(), "Vulkan policy requires identical complete snapshot coverage"
    check_router_trace(reference, actual)
    check_all_integer_snapshots(reference, actual)
    maximum_nmse, maximum_error, vectors, values, records = 0.0, 0.0, 0, 0, 0
    pooled_names = set()
    for key, row in left.items():
        measured = right[key]
        assert len(row["values"]) == len(measured["values"]) == math.prod(row["shape"]), key
        assert all(math.isfinite(value) for value in row["values"] + measured["values"]), key
        if not pooled(row["name"]):
            continue
        assert row["type"] == "f32" and row["shape"][0] > 0, key
        width = row["shape"][0]
        for start in range(0, len(row["values"]), width):
            expected, got = row["values"][start:start + width], measured["values"][start:start + width]
            denominator = sum(value * value for value in expected)
            numerator = sum((a - b) ** 2 for a, b in zip(expected, got))
            if denominator == 0:
                assert all(value == 0 for value in got), f"nonzero pooled key against zero reference: {key} column {start // width}"
                nmse = 0.0
            else:
                nmse = numerator / denominator
            assert math.isfinite(nmse) and nmse <= VULKAN_COMPARISON_POLICY["pooled_key_nmse_limit"], \
                f"pooled key NMSE exceeds declared policy: {key} column {start // width}: {nmse}"
            maximum_nmse = max(maximum_nmse, nmse)
            maximum_error = max(maximum_error, max(abs(a - b) for a, b in zip(expected, got)))
            vectors += 1
            values += width
        records += 1
        pooled_names.add(row["name"])
    assert records and vectors, "Vulkan policy did not exercise QSA pooled keys"
    filtered = lambda trace: {"tokens": trace["tokens"], "records": [row for row in trace["records"] if not pooled(row["name"])]}
    result = full.compare(filtered(reference), filtered(actual), atol=ATOL, rtol=RTOL)
    result["strict_nonpooled_compared_floats"] = result["compared_floats"]
    result["compared_floats"] += values
    result["compared_records"] += records
    result["max_abs_error"] = max(result["max_abs_error"], maximum_error)
    result["trace_names"] = sorted(set(result["trace_names"]) | pooled_names)
    result["scalar_tolerance_scope"] = "all floating values except the explicitly listed pooled key vectors"
    result["pooled_keys"] = {"records": records, "vectors": vectors, "values": values,
                             "max_nmse": maximum_nmse, "nmse_limit": VULKAN_COMPARISON_POLICY["pooled_key_nmse_limit"],
                             "max_abs_error": maximum_error}
    return result


def diagnose_fp16_pool_boundary(trace_root, output):
    """Reconstruct the saved eval-5 QSA pool discrepancy without model execution."""
    trace_root, output = Path(trace_root), Path(output)
    paths = {"cpu": trace_root / "vulkan-iq4-ple-chunk1-f32-oracle-direct.json",
             "gpu": trace_root / "vulkan-iq4-ple-chunk1-eviction.json"}
    traces = {name: json.loads(path.read_text()) for name, path in paths.items()}
    metadata, tensors = full.fixture.fixture(expert_type="q8_0", ple_type="iq4_nl")
    metadata = {name: value for name, _, value in metadata}
    norm_payload = next(payload for name, _, _, payload in tensors if name == "blk.3.indexer.k_norm.weight")
    weights = struct.unpack("<32f", norm_payload)
    f32 = lambda value: struct.unpack("<f", struct.pack("<f", value))[0]
    half = lambda value: struct.unpack("<e", struct.pack("<e", value))[0]
    def record(trace, evaluation, name):
        rows = [row for row in trace["records"] if row["evaluation"] == evaluation and row["name"] == name]
        assert len(rows) == 1
        return rows[0]["values"]
    report = {"scope": "offline reconstruction of eval5 indexer_pool_k_new-3 from eval4/5 raw keys",
              "cause": "one F32 ULP at an FP16 rounding midpoint; pooling/RMS/RoPE propagate different legal cache roundings",
              "source": {"cache_dtype": "vendor/llama.cpp/src/llama-context.cpp:llama_context_default_params",
                         "indexer_cache": "vendor/llama.cpp/src/llama-memory-hybrid-idx.cpp:llama_memory_hybrid_idx",
                         "pool_operator": "vendor/llama.cpp/src/models/qwen4exp.cpp:build_qsa_sel",
                         "rms_rope": "vendor/llama.cpp/ggml/src/ggml-cpu/ops.cpp:ggml_compute_forward_rms_norm_f32,ggml_compute_forward_rope_flt"},
              "trace_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
              "norm_weight_sha256": hashlib.sha256(norm_payload).hexdigest(), "paths": {}}
    means, actual_vectors = {}, {}
    for name, trace in traces.items():
        raw = [[f32(value) for value in record(trace, evaluation, "indexer_k_raw-3")] for evaluation in (4, 5)]
        stored = [[half(value) for value in row] for row in raw]
        mean = [f32(f32(a + b) * .5) for a, b in zip(*stored)]
        means[name] = mean
        squared_mean = f32(sum(f32(value * value) for value in mean) / 32)
        epsilon = f32(metadata["qwen4exp.attention.layer_norm_rms_epsilon"])
        gain = f32(1 / f32(math.sqrt(f32(squared_mean + epsilon))))
        normalized = [f32(f32(value * gain) * weight) for value, weight in zip(mean, weights)]
        rotated = [0.0] * 32
        theta, theta_scale = f32(4), f32(metadata["qwen4exp.rope.freq_base"] ** (-2 / 32))
        for i in range(16):
            cosine, sine = f32(math.cos(theta)), f32(math.sin(theta))
            rotated[i] = f32(f32(normalized[i] * cosine) - f32(normalized[i + 16] * sine))
            rotated[i + 16] = f32(f32(normalized[i] * sine) + f32(normalized[i + 16] * cosine))
            theta = f32(theta * theta_scale)
        actual = record(trace, 5, "indexer_pool_k_new-3")
        actual_vectors[name] = actual
        reconstruction_error = max(abs(a - b) for a, b in zip(rotated, actual))
        assert reconstruction_error < 2e-7, "offline pooling/norm/RoPE reconstruction disagrees"
        report["paths"][name] = {"eval4_raw_coordinate30": raw[0][30],
            "raw_f32_bits": f"0x{struct.unpack('<I', struct.pack('<f', raw[0][30]))[0]:08x}",
            "stored_f16_coordinate30": stored[0][30],
            "stored_f16_bits": f"0x{struct.unpack('<H', struct.pack('<e', raw[0][30]))[0]:04x}",
            "pooled_mean_coordinate30": mean[30], "rms_gain": gain,
            "predicted_output_coordinate30": rotated[30], "actual_output_coordinate30": actual[30],
            "maximum_vector_reconstruction_error": reconstruction_error}
    differences = [(i, b - a) for i, (a, b) in enumerate(zip(means["cpu"], means["gpu"])) if a != b]
    assert differences == [(30, 2 ** -14)], differences
    cpu, gpu = actual_vectors["cpu"], actual_vectors["gpu"]
    report["half_midpoint"] = (report["paths"]["cpu"]["stored_f16_coordinate30"] + report["paths"]["gpu"]["stored_f16_coordinate30"]) / 2
    report["pooled_mean_differences"] = differences
    report["nmse"] = sum((a - b) ** 2 for a, b in zip(cpu, gpu)) / sum(a * a for a in cpu)
    report["relative_l2_error"] = math.sqrt(report["nmse"])
    report["original_scalar_tolerance"] = {"absolute": ATOL, "relative": RTOL,
        "coordinate30_error": abs(cpu[30] - gpu[30]), "coordinate30_limit": ATOL + RTOL * abs(cpu[30]), "passed": False}
    assert report["original_scalar_tolerance"]["coordinate30_error"] > report["original_scalar_tolerance"]["coordinate30_limit"]
    with output.open("x") as target:
        target.write(json.dumps(report, indent=2) + "\n")
    return report


def diagnose_q8_activation(trace_root, output):
    """Offline layer-0 dot products from captured traces; no decoder or device."""
    trace_root, output = Path(trace_root), Path(output)
    paths = {name: trace_root / f"vulkan-iq4-ple-chunk1-{name}.json" for name in ("reference", "direct", "eviction")}
    traces = {name: json.loads(path.read_text()) for name, path in paths.items()}
    cpu, gpu = traces["direct"], traces["eviction"]
    check_router_trace(cpu, gpu)
    assert cpu["tokens"] == gpu["tokens"]
    _, original = full.fixture.fixture(expert_type="q8_0", ple_type="iq4_nl")
    tensors = {name: (shape, kind, payload) for name, shape, kind, payload in original}
    def f32(value):
        return struct.unpack("<f", struct.pack("<f", value))[0]
    def quantized_activation(values):
        result = []
        for start in range(0, len(values), 32):
            block = values[start:start + 32]
            maximum = max(map(abs, block))
            scale = f32(maximum / 127)
            inverse = f32(127 / maximum) if maximum else 0
            stored_scale = struct.unpack("<e", struct.pack("<e", scale))[0]
            result.extend(stored_scale * max(-127, min(127, round(f32(value * inverse)))) for value in block)
        return result
    def record(trace, evaluation, name):
        matches = [row for row in trace["records"] if row["evaluation"] == evaluation and row["name"] == name]
        assert len(matches) == 1
        return matches[0]["values"]
    maxima = {name: 0.0 for name in ("cpu_vs_quantized_activation", "gpu_vs_f32_activation",
                                   "cpu_vs_f32_activation", "gpu_vs_quantized_activation")}
    count = 0
    for evaluation in range(len(cpu["metrics"]["evaluations"])):
        x = [f32(value) for value in record(cpu, evaluation, "ffn_input-0")]
        assert x == [f32(value) for value in record(gpu, evaluation, "ffn_input-0")]
        quantized = quantized_activation(x)
        ids = record(cpu, evaluation, "ffn_moe_topk-0")
        for operation in ("gate", "up"):
            shape, kind, payload = tensors[f"blk.0.ffn_{operation}_exps.weight"]
            assert kind == 8
            weights, direct_dot, quantized_dot = q8_values(payload), [], []
            for expert in ids:
                for row in range(shape[1]):
                    start = (expert * shape[1] + row) * shape[0]
                    weight = weights[start:start + shape[0]]
                    direct_dot.append(sum(a * b for a, b in zip(weight, x)))
                    quantized_dot.append(sum(a * b for a, b in zip(weight, quantized)))
            cpu_values = record(cpu, evaluation, f"ffn_moe_{operation}-0")
            gpu_values = record(gpu, evaluation, f"ffn_moe_{operation}-0")
            for name, left, right in (("cpu_vs_quantized_activation", cpu_values, quantized_dot),
                                     ("gpu_vs_f32_activation", gpu_values, direct_dot),
                                     ("cpu_vs_f32_activation", cpu_values, direct_dot),
                                     ("gpu_vs_quantized_activation", gpu_values, quantized_dot)):
                assert len(left) == len(right)
                maxima[name] = max(maxima[name], max(abs(a - b) for a, b in zip(left, right)))
            count += len(direct_dot)
    assert maxima["cpu_vs_quantized_activation"] < 1e-6 and maxima["gpu_vs_f32_activation"] < 1e-6
    assert maxima["cpu_vs_f32_activation"] > 1e-4 and maxima["gpu_vs_quantized_activation"] > 1e-4
    report = {"scope": "offline, layer0 gate/up only, identical captured input and selected expert IDs",
              "algorithm": "stdlib Q8 halfscale/int8 decode; F32 input rounding; CPU AVX block32 Q8_0 d=F32(max/127), multiplier=F32(127/max), nearest-even code, FP16 stored d; double dot accumulation",
              "cpu_source": "vendor/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c:quantize_row_q8_0",
              "values_per_oracle": count, "max_abs_errors": maxima, "oracle_absolute_tolerance": 1e-6,
              "tokens_equal": True, "all_integer_values_equal": check_all_integer_snapshots(cpu, gpu),
              "trace_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
              "generator_sha256": hashlib.sha256((SOURCE / "tools/make_qwen_fixture.py").read_bytes()).hexdigest(),
              "source_weight_sha256": {name: hashlib.sha256(payload).hexdigest()
                  for name, (_, kind, payload) in tensors.items() if "_exps.weight" in name}}
    manifest = json.loads((trace_root / "vulkan-iq4-ple.json").read_text())
    report["fixture_files"] = []
    for entry in manifest["files"]:
        path = trace_root / Path(entry["path"]).name
        assert path.stat().st_size == entry["bytes"] < 8 * 1024 * 1024
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], "captured synthetic fixture hash changed"
        report["fixture_files"].append({"name": path.name, "bytes": entry["bytes"], "sha256": digest})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as target:
        target.write(json.dumps(report, indent=2) + "\n")
    return report


def run_vulkan_suite(pmi, root, args):
    assert args.expect_description, "--vulkan requires --expect-description for the approved physical GPU"
    assert args.device_index >= 0, "device index must be nonnegative"
    report = {"scope": "tiny generated Qwen fixtures; explicit Vulkan expert projections; CPU dense/PLE graph",
              "success_criterion": "declared Vulkan mixed precision criterion against F32-activation CPU oracle; success does not assert strict paired or original Q8 CPU float parity",
              "comparison_policy": VULKAN_COMPARISON_POLICY,
              "success": False, "expected_device_cases": 8, "device_cases": [], "baselines": [],
              "paired_oracles": [],
              "absolute_tolerance": ATOL, "relative_tolerance": RTOL, "cpu_repack": False,
              "device_index": args.device_index, "expected_description": args.expect_description,
              "expected_device_id": args.expect_device_id,
              "timing_scope": "synthetic fixture only; PMI cache cold/warm, uncontrolled OS/driver caches"}
    path = root / "device-vulkan-full-model-results.json"
    def save():
        encoded = (json.dumps(report, indent=2) + "\n").encode()
        assert len(encoded) <= MAX_REPORT_BYTES, "bounded Vulkan fixture report exceeded its size limit"
        path.write_bytes(encoded)
    save()
    try:
        for variant, options, host_capacity in (
            ("f32", {"expert_type": "f32"}, 8192),
            ("iq4-ple", {"expert_type": "q8_0", "ple_type": "iq4_nl"}, 4352),
        ):
            metadata, _ = full.fixture.fixture(**options)
            metadata = {name: value for name, _, value in metadata}
            width, layers = metadata["qwen4exp.embedding_length"], metadata["qwen4exp.block_count"]
            shards = full.fixture.generate(root / ("vulkan-" + variant + ".gguf"), shards=2, **options)
            oracle_shards = None
            if variant == "iq4-ple":
                oracle_shards, oracle_capacity, oracle_audit = make_q8_expert_oracle(root / "vulkan-iq4-ple-f32-oracle.gguf")
                report["paired_oracles"].append(oracle_audit)
            for chunk in (1, 4):
                stem = f"vulkan-{variant}-chunk{chunk}"
                common = ["--prompt", "hello world", "--max-tokens", 8, "--gpu-layers", 0,
                          "--cpu-repack", "off", "--prefill-chunk", chunk, "--context", 64]
                reference, reference_path = run_trace(pmi, root, stem + "-reference", ["reference", shards[0], *common])
                direct, direct_path = run_trace(pmi, root, stem + "-direct",
                                                ["run", *reversed(shards), *common, "--cache-bytes", host_capacity])
                baseline_comparison = full.compare(reference, direct, atol=ATOL, rtol=RTOL)
                check_router_trace(reference, direct)
                for trace, paged in ((reference, False), (direct, True)):
                    evaluations = full.check_evaluation_metrics(trace, paged=paged, repack=False,
                        reference_mode=None if paged else "default", trace_mode="full", chunk=chunk)
                    disabled_projection(trace["metrics"])
                    full.check_ffn_inputs(trace, evaluations, width=width, layers=layers)
                baseline = {"variant": variant, "prefill_chunk": chunk,
                    "reference_trace": reference_path.name, "direct_trace": direct_path.name, "comparison": baseline_comparison}
                comparison_reference, comparison_direct = reference, direct
                if oracle_shards:
                    comparison_reference, oracle_reference_path = run_trace(pmi, root, stem + "-f32-oracle-reference",
                                                                           ["reference", oracle_shards[0], *common])
                    comparison_direct, oracle_direct_path = run_trace(pmi, root, stem + "-f32-oracle-direct",
                        ["run", *reversed(oracle_shards), *common, "--cache-bytes", oracle_capacity])
                    oracle_comparison = full.compare(comparison_reference, comparison_direct, atol=ATOL, rtol=RTOL)
                    check_router_trace(comparison_reference, comparison_direct)
                    for trace, paged in ((comparison_reference, False), (comparison_direct, True)):
                        evaluations = full.check_evaluation_metrics(trace, paged=paged, repack=False,
                            reference_mode=None if paged else "default", trace_mode="full", chunk=chunk)
                        disabled_projection(trace["metrics"])
                        full.check_ffn_inputs(trace, evaluations, width=width, layers=layers)
                    baseline["paired_f32_oracle"] = {"native_trace": oracle_reference_path.name,
                        "direct_trace": oracle_direct_path.name, "comparison": oracle_comparison,
                        "direct_host_capacity_bytes": oracle_capacity}
                report["baselines"].append(baseline)
                seen = {}
                for mode, capacity in (("eviction", host_capacity * 2), ("warm", WARM_DEVICE_BYTES)):
                    label = stem + "-" + mode
                    actual, actual_path = run_trace(pmi, root, label,
                        ["run", *reversed(shards), *common, "--cache-bytes", host_capacity,
                         "--projection-backend", "device-vulkan", "--projection-device-index", args.device_index,
                         "--device-cache-bytes", capacity, "--device-graph-bytes", GRAPH_BYTES, "--profile-paging", "on"])
                    identity = actual["metrics"]["projection_device"]
                    assert identity["index"] == args.device_index and identity["type"] == "gpu", identity
                    assert "Vulkan" in identity["backend_name"] and args.expect_description in identity["description"], identity
                    if args.expect_device_id:
                        assert identity["device_id"] == args.expect_device_id, identity
                    assert actual["metrics"]["vram_bytes"] is None, "managed cache bytes were reported as physical VRAM"
                    evaluations = check_device_metrics(actual, profiling=True, host_capacity=host_capacity,
                                                       device_capacity=capacity, chunk=chunk, backend="device-vulkan")
                    full.check_ffn_inputs(actual, evaluations, width=width, layers=layers)
                    original_diagnostic = None
                    if oracle_shards:
                        assert reference["tokens"] == direct["tokens"] == actual["tokens"], "original Q8 CPU greedy token mismatch"
                        check_router_trace(reference, actual)
                        check_router_trace(direct, actual)
                        exact_ids = check_all_integer_snapshots(direct, actual)
                        check_all_integer_snapshots(reference, actual)
                        original_error = None
                        try:
                            compare_all(reference, direct, actual)
                        except AssertionError as error:
                            original_error = str(error)
                        original_path = root / (label + ".original-q8-cpu-diagnostic.json")
                        original_diagnostic = {"strict_float_comparison_passed": original_error is None,
                            "error": original_error, "absolute_tolerance": ATOL, "relative_tolerance": RTOL,
                            "tokens_equal": True, "exact_integer_values": exact_ids,
                            "explanation": "original Q8 CPU dot kernel also quantizes F32 activations to Q8_0; tested Vulkan path uses F32 activations",
                            "source_pointer": "vendor/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c:GGML_TYPE_Q8_0 vec_dot_type",
                            "native": comparison_diagnostics(reference, actual),
                            "direct_pmi_cpu": comparison_diagnostics(direct, actual)}
                        original_path.write_text(json.dumps(original_diagnostic, indent=2) + "\n")
                        original_diagnostic = {key: value for key, value in original_diagnostic.items()
                                               if key not in ("native", "direct_pmi_cpu")}
                        original_diagnostic["details"] = original_path.name
                    try:
                        paired_strict = {}
                        for name, control in (("native", comparison_reference), ("direct_pmi_cpu", comparison_direct)):
                            try:
                                full.compare(control, actual, atol=ATOL, rtol=RTOL)
                                paired_strict[name] = {"passed": True, "error": None}
                            except AssertionError as error:
                                paired_strict[name] = {"passed": False, "error": str(error)}
                        if not all(result["passed"] for result in paired_strict.values()):
                            strict_path = root / (label + ".paired-strict-comparison.json")
                            strict_path.write_text(json.dumps({"outcome": paired_strict,
                                "absolute_tolerance": ATOL, "relative_tolerance": RTOL,
                                "native": comparison_diagnostics(comparison_reference, actual),
                                "direct_pmi_cpu": comparison_diagnostics(comparison_direct, actual)}, indent=2) + "\n")
                            paired_strict["details"] = strict_path.name
                        comparison = {"native": compare_vulkan_policy(comparison_reference, actual),
                                      "direct_pmi_cpu": compare_vulkan_policy(comparison_direct, actual)}
                    except AssertionError:
                        diagnostics = {"native": comparison_diagnostics(comparison_reference, actual),
                                       "direct_pmi_cpu": comparison_diagnostics(comparison_direct, actual)}
                        diagnostic_path = root / (label + ".comparison-failure.json")
                        diagnostic_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
                        report["failed_case"] = label
                        report["comparison_diagnostics"] = diagnostic_path.name
                        raise
                    cache = actual["metrics"]["device_projection"]["cache"]
                    if mode == "eviction":
                        assert cache["evictions"] > 0, "small Vulkan weight cache did not evict"
                    else:
                        assert cache["hits"] > 0 and cache["evictions"] == 0, "large Vulkan weight cache did not retain weights"
                    seen[mode] = actual["metrics"]
                    report["device_cases"].append({"case": label, "variant": variant, "prefill_chunk": chunk,
                        "trace": actual_path.name, "comparison": comparison, "metrics": scalar_metrics(actual),
                        "paired_strict_float_comparison": paired_strict,
                        "numerical_reference": "stored-Q8-expanded-to-F32-weights/F32-activations" if oracle_shards else "original-F32-weights/F32-activations",
                        "original_q8_cpu_comparison": original_diagnostic,
                        "validated_evaluations": len(evaluations)})
                    save()
                small, warm = seen["eviction"], seen["warm"]
                assert warm["device_projection"]["cache"]["provider_requests"] < small["device_projection"]["cache"]["provider_requests"]
                assert warm["io_bytes"] < small["io_bytes"], "Vulkan weight cache failed to avoid host storage reads"
        assert len(report["device_cases"]) == report["expected_device_cases"]
        original_comparisons = [case["original_q8_cpu_comparison"] for case in report["device_cases"]
                                if case["original_q8_cpu_comparison"] is not None]
        report["original_q8_cpu_strict_float_passes"] = sum(item["strict_float_comparison_passed"] for item in original_comparisons)
        report["original_q8_cpu_strict_float_comparisons"] = len(original_comparisons)
        report["paired_strict_float_passes"] = sum(case["paired_strict_float_comparison"]["native"]["passed"] and
            case["paired_strict_float_comparison"]["direct_pmi_cpu"]["passed"] for case in report["device_cases"])
        report["paired_strict_float_comparisons"] = len(report["device_cases"])
        report["success"] = True
        save()
        print("Verified 8 Vulkan generated-fixture cases under declared mixed precision policy; "
              "exact original tokens/routes; strict paired and original Q8 CPU float results recorded separately")
    except Exception as error:
        report["error"] = str(error)
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pmi", type=Path)
    parser.add_argument("output", type=Path, help="dedicated build fixture output directory")
    parser.add_argument("--vulkan", action="store_true", help="explicit opt-in to the separately guarded physical GPU suite")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--expect-description", default="")
    parser.add_argument("--expect-device-id", default="")
    args = parser.parse_args()
    pmi, root = args.pmi.resolve(), args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.vulkan:
        run_vulkan_suite(pmi, root, args)
        return
    if args.device_index or args.expect_description or args.expect_device_id:
        parser.error("device identity arguments require --vulkan")
    variants = [
        ("f32", {"expert_type": "f32"}, 8192),
        ("f16", {"expert_type": "f16", "ple_type": "f16"}, 4096),
        ("q8", {"expert_type": "q8_0"}, 2176),
        ("iq4-ple", {"expert_type": "q8_0", "ple_type": "iq4_nl"}, 4352),
    ]
    reports, baselines = [], []
    saved = None
    first_shards = None
    for variant, options, host_capacity in variants:
        metadata, tensor_data = full.fixture.fixture(**options)
        metadata = {name: value for name, _, value in metadata}
        width, layers = metadata["qwen4exp.embedding_length"], metadata["qwen4exp.block_count"]
        expert_bytes = sum(len(data) for name, _, _, data in tensor_data if "_exps.weight" in name)
        shards = full.fixture.generate(root / ("device-" + variant + ".gguf"), shards=2, **options)
        first_shards = first_shards or shards
        previous_tokens = None
        for chunk in (1, 4):
            stem = f"device-{variant}-chunk{chunk}"
            common = ["--prompt", "hello world", "--max-tokens", 8, "--gpu-layers", 0,
                      "--prefill-chunk", chunk, "--context", 64, "--cpu-repack", "off"]
            reference, reference_path = run_trace(pmi, root, stem + "-reference", ["reference", shards[0], *common])
            direct, direct_path = run_trace(pmi, root, stem + "-direct",
                                            ["run", *reversed(shards), *common, "--cache-bytes", host_capacity])
            direct_comparison = full.compare(reference, direct, atol=ATOL, rtol=RTOL)
            for trace, paged in ((reference, False), (direct, True)):
                evaluations = full.check_evaluation_metrics(trace, paged=paged, repack=False,
                    reference_mode=None if paged else "default", trace_mode="full", chunk=chunk)
                full.check_ffn_inputs(trace, evaluations, width=width, layers=layers)
                disabled_projection(trace["metrics"])
            if previous_tokens is not None:
                assert reference["tokens"] == previous_tokens, "native fixture depends on prefill chunking"
            previous_tokens = reference["tokens"]
            baselines.append({"variant": variant, "prefill_chunk": chunk, "tokens": reference["tokens"],
                              "reference_trace": reference_path.name, "direct_trace": direct_path.name,
                              "comparison": direct_comparison})
            traces = {}
            for mode, capacity in (("eviction", host_capacity * 2), ("warm", WARM_DEVICE_BYTES)):
                label = stem + "-" + mode
                arguments = ["run", *reversed(shards), *common, "--cache-bytes", host_capacity,
                             "--projection-backend", "device-cpu", "--device-cache-bytes", capacity,
                             "--device-graph-bytes", GRAPH_BYTES, "--profile-paging", "on"]
                actual, path = run_trace(pmi, root, label, arguments)
                comparison = compare_all(reference, direct, actual)
                evaluations = check_device_metrics(actual, profiling=True, host_capacity=host_capacity,
                                                   device_capacity=capacity, chunk=chunk)
                full.check_ffn_inputs(actual, evaluations, width=width, layers=layers)
                cache = actual["metrics"]["device_projection"]["cache"]
                if mode == "eviction":
                    assert cache["evictions"] > 0, "small device cache did not evict weights"
                else:
                    assert cache["hits"] > 0 and cache["evictions"] == 0, "large device cache did not retain weights"
                    assert cache["upload_bytes"] <= expert_bytes, "large device cache reuploaded immutable expert bytes"
                traces[mode] = actual
                reports.append({"case": label, "variant": variant, "prefill_chunk": chunk,
                                "forced_device_eviction": mode == "eviction", "trace": path.name,
                                "comparison": comparison, "metrics": scalar_metrics(actual),
                                "validated_evaluations": len(evaluations)})
                if variant == "iq4-ple" and chunk == 4 and mode == "warm":
                    saved = (reference, direct, actual, arguments, host_capacity, width, layers)
            small, warm = traces["eviction"]["metrics"], traces["warm"]["metrics"]
            for name in ("page_requests", "expert_requests", "lookup_requests"):
                assert small[name] == warm[name] == direct["metrics"][name], f"projection path changed routing work: {name}"
            for name in DEVICE_COUNTS:
                assert small["device_projection"][name] == warm["device_projection"][name], name
            assert warm["device_projection"]["cache"]["provider_requests"] < small["device_projection"]["cache"]["provider_requests"]
            assert warm["device_projection"]["cache"]["provider_requests"] < direct["metrics"]["expert_requests"]
            assert warm["io_bytes"] < small["io_bytes"] and warm["io_bytes"] < direct["metrics"]["io_bytes"], \
                "retaining device weights did not avoid host storage reads"

    reference, direct, profiled, arguments, host_capacity, width, layers = saved
    off_arguments = arguments[:-1] + ["off"]
    unprofiled, path = run_trace(pmi, root, "device-iq4-ple-profile-off", off_arguments)
    comparison = compare_all(reference, direct, unprofiled)
    full.compare(profiled, unprofiled, atol=ATOL, rtol=RTOL)
    evaluations = check_device_metrics(unprofiled, profiling=False, host_capacity=host_capacity,
                                       device_capacity=WARM_DEVICE_BYTES, chunk=4)
    full.check_ffn_inputs(unprofiled, evaluations, width=width, layers=layers)
    for name in ("page_requests", "expert_requests", "lookup_requests", "io_bytes", "cache_hits", "cache_misses", "evictions"):
        assert profiled["metrics"][name] == unprofiled["metrics"][name], f"profiling changed paging behavior: {name}"
    for name in DEVICE_COUNTS:
        assert profiled["metrics"]["device_projection"][name] == unprofiled["metrics"]["device_projection"][name], name
    # Profiling intentionally changes its enable flag and measured service times.
    # Every remaining field must retain exact behavior, including future fields.
    def cache_behavior(trace):
        return {key: value for key, value in trace["metrics"]["device_projection"]["cache"].items()
                if key not in ("profile_enabled", *CACHE_TIMINGS)}
    assert cache_behavior(profiled) == cache_behavior(unprofiled), "profiling changed cache behavior"
    reports.append({"case": "device-iq4-ple-profile-off", "trace": path.name,
                    "comparison": comparison, "metrics": scalar_metrics(unprofiled),
                    "validated_evaluations": len(evaluations)})

    # Exercise public execution without trace callbacks splitting the scheduler.
    label = "device-iq4-ple-untraced"
    tokens = full.invoke(pmi, root, label, off_arguments)
    assert tokens == reference["tokens"], "untraced device execution changed greedy tokens"
    lines = (root / (label + ".stdout")).read_text().splitlines()
    metrics = json.loads(next(line.removeprefix("PMI_METRICS ") for line in lines if line.startswith("PMI_METRICS ")))
    check_device_metrics({"tokens": tokens, "metrics": metrics}, profiling=False, host_capacity=host_capacity,
                         device_capacity=WARM_DEVICE_BYTES, chunk=4)
    untraced = {"case": label, "tokens": tokens, "metrics": {key: value for key, value in metrics.items() if key != "evaluations"}}
    invalid = negative_cases(pmi, root, first_shards)
    report = {
        "scope": "generated untrained Qwen fixtures; device-CPU projections only; no GPU or trained-model quality claim",
        "absolute_tolerance": ATOL, "relative_tolerance": RTOL,
        "cpu_repack": False, "variants": [name for name, _, _ in variants], "prefill_chunks": [1, 4],
        "baselines": baselines, "device_cases": reports, "untraced": untraced, "negative_cases": invalid,
    }
    encoded = (json.dumps(report, indent=2) + "\n").encode()
    assert len(encoded) <= MAX_REPORT_BYTES, "bounded device fixture report exceeded its size limit"
    (root / "device-full-model-results.json").write_bytes(encoded)
    print(f"Verified {len(variants)} generated Qwen fixtures; {len(reports)} traced device cases; "
          f"one untraced case; {len(invalid)} rejected configurations; no GPU execution")


if __name__ == "__main__":
    main()
