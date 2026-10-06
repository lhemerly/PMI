#!/usr/bin/env python3
"""Audit saved PMI campaign artifacts only. No inference or model-weight reads.

Exit 0 means passed=true; exit 1 means an evidence/correctness/guard failure.
Output is exclusively created and never contains raw commands/generated text.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import routing_dataset as routing

EXPECTED_TOKENS = [11, 353, 2688, 264, 1496, 2812, 3712, 10756,
                   852, 579, 7157, 279, 1483, 35442, 1578, 4313]
LIMIT = 64 * 1024 * 1024


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def extent(values, counter=False):
    values = [v for v in values if number(v)]
    if not values:
        return None
    reset = counter and any(b < a for a, b in zip(values, values[1:]))
    return dict(samples=len(values), first=values[0], last=values[-1], min=min(values), max=max(values),
                delta=None if reset else values[-1] - values[0], counter_reset=reset)


def nested(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def read(path, report, role, required=True):
    if path is None or not Path(path).exists():
        report["sources"][role] = {"present": False}
        if required:
            report["failures"].append(role + "_missing")
        return None
    try:
        data, sha = routing.bounded_read(path, LIMIT)
        report["sources"][role] = {"present": True, "sha256": sha, "bytes": len(data)}
        return data
    except (OSError, routing.DatasetError):
        report["failures"].append(role + "_unreadable_or_oversized")
        return None


def json_lines(data):
    rows = []
    for line in data.splitlines():
        if not line.strip():
            continue
        routing.check(len(line) <= 1024 * 1024 and len(rows) < 100000, "JSONL parser limit")
        row = routing.strict_json(line)
        routing.check(isinstance(row, dict), "JSONL record must be an object")
        rows.append(row)
    return rows


def guard_audit(data, report):
    if data is None:
        return [], []
    rows = json_lines(data)
    starts = [r for r in rows if r.get("event") == "start"]
    finals = [r for r in rows if r.get("event") == "final"]
    if len(starts) != 1 or len(finals) != 1 or not rows or rows[-1].get("event") != "final":
        report["failures"].append("guard_framing_incomplete")
    start, final = (starts[0] if starts else {}), (finals[-1] if finals else {})
    fields = ("status", "reason", "command_started", "command_returncode", "runner_returncode",
              "elapsed_seconds", "sampled_peak_rss_bytes", "sampled_peak_group_swap_bytes",
              "sampled_peak_system_swap_growth_bytes", "sampled_peak_swap_growth_bytes",
              "minimum_mem_available_bytes", "samples", "term_sent", "kill_sent")
    summary = {key: final.get(key) for key in fields}
    summary["monitor_error_count"] = len(final.get("monitor_errors", []))
    summary["remaining_live_pid_count"] = len(final.get("remaining_live_pids", []))
    summary["limits"] = start.get("limits", {})
    summary["stop_reasons"] = [r.get("reason") for r in rows if r.get("event") == "stop"]
    clean = (final.get("status") == "completed" and final.get("reason") is None and
             final.get("command_started") is True and final.get("command_returncode") == 0 and
             final.get("runner_returncode") == 0 and final.get("monitor_errors") == [] and
             final.get("remaining_live_pids") == [] and not final.get("kill_sent") and
             not any(r.get("event") in ("stop", "monitor_error") for r in rows))
    limits = summary["limits"]
    samples = [r for r in rows if r.get("event") == "sample"]
    violations = set()
    comparisons = (("group_rss_bytes", "max_rss_bytes", False),
                   ("group_swap_bytes", "max_swap_growth_bytes", False),
                   ("system_swap_growth_bytes", "max_system_swap_growth_bytes", False),
                   ("mem_available_bytes", "min_available_bytes", True))
    for sample in samples:
        for field, limit, minimum in comparisons:
            bound = limits.get(limit)
            if limit == "max_system_swap_growth_bytes" and bound is None:
                bound = limits.get("max_swap_growth_bytes")
            value = sample.get(field)
            if number(value) and number(bound) and (value < bound if minimum else value > bound):
                violations.add(limit)
    final_bounds = (("sampled_peak_rss_bytes", "max_rss_bytes", False),
                    ("sampled_peak_group_swap_bytes", "max_swap_growth_bytes", False),
                    ("sampled_peak_system_swap_growth_bytes", "max_system_swap_growth_bytes", False),
                    ("minimum_mem_available_bytes", "min_available_bytes", True),
                    ("elapsed_seconds", "wall_seconds", False))
    for field, limit, minimum in final_bounds:
        value, bound = final.get(field), limits.get(limit)
        if number(value) and number(bound) and (value < bound if minimum else value > bound):
            violations.add(limit)
    summary["observed_limit_violations"] = sorted(violations)
    summary["sampled_process_io"] = (samples[-1].get("sampled_io_bytes") if samples else None)
    summary["passed"] = clean and not violations
    report["guard"] = summary
    if not summary["passed"]:
        report["failures"].append("guard_failed")
    return rows, samples


def output_audit(data, report):
    if data is None:
        return None
    lines = data.decode("utf-8", errors="replace").splitlines()
    token_lines = [line[7:].split() for line in lines if line.startswith("TOKENS ")]
    matched = False
    count = None
    if len(token_lines) == 1:
        try:
            tokens = [int(v) for v in token_lines[0]]
            count, matched = len(tokens), tokens == EXPECTED_TOKENS
        except ValueError:
            pass
    report["correctness"] = {"expected_tokens": 16, "observed_token_count": count,
                             "exact_saved_stage3_token_match": matched}
    if not matched:
        report["failures"].append("tokens_missing_or_mismatched")
    metric_lines = [line[len("PMI_METRICS "):] for line in lines if line.startswith("PMI_METRICS ")]
    if len(metric_lines) != 1:
        report["failures"].append("metrics_missing_or_ambiguous")
        return None
    metrics = routing.strict_json(metric_lines[0])
    routing.check(isinstance(metrics, dict), "invalid metrics object")
    evaluations = metrics.get("evaluations", [])
    routing.check(isinstance(evaluations, list), "invalid evaluations")
    report["metrics"] = routing.object_projection(metrics, routing.METRIC_KEYS)
    report["metrics"]["evaluations"] = [{"evaluation": e.get("evaluation"), "phase": e.get("phase"),
                                          **routing.object_projection(e, routing.METRIC_KEYS)} for e in evaluations]
    expected = [(i, "prefill" if i == 0 else "decode", 1) for i in range(16)]
    observed = [(e.get("evaluation"), e.get("phase"), e.get("input_tokens")) for e in evaluations]
    if observed != expected or metrics.get("generated_tokens") != 16:
        report["failures"].append("evaluation_sequence_mismatch")
    for field in ("io_bytes", "cache_hits", "cache_misses", "evictions"):
        values = [e.get(field) for e in evaluations]
        if not values or not all(type(v) is int and v >= 0 for v in values) or sum(values) != metrics.get(field):
            report["failures"].append("counter_reconciliation_" + field)
    hits, misses = metrics.get("cache_hits"), metrics.get("cache_misses")
    hit_rate = hits / (hits + misses) if number(hits) and number(misses) and hits + misses else 0.0
    if not number(metrics.get("hit_rate")) or not math.isclose(hit_rate, metrics["hit_rate"], abs_tol=1e-10):
        report["failures"].append("hit_rate_mismatch")
    peak, capacity = metrics.get("cache_peak_reserved_bytes"), metrics.get("cache_capacity_bytes")
    if number(peak) and number(capacity) and peak > capacity:
        report["failures"].append("cache_capacity_exceeded")
    decode = [e.get("elapsed_ms") for e in evaluations if e.get("phase") == "decode"]
    routing.check(all(number(v) and v >= 0 for v in decode), "invalid evaluation duration")
    physical = [e.get("process_storage_read_bytes_delta") for e in evaluations]
    report["performance"] = dict(decode_evaluations=len(decode), decode_median_ms=statistics.median(decode) if decode else None,
        decode_min_ms=min(decode) if decode else None, decode_max_ms=max(decode) if decode else None,
        decode_sum_ms=sum(decode) if decode else None,
        decode_evaluations_per_second=len(decode) * 1000 / sum(decode) if decode and sum(decode) > 0 else None,
        logical_page_io_bytes=metrics.get("io_bytes"), process_storage_read_bytes=metrics.get("process_storage_read_bytes"),
        evaluation_storage_read_bytes=sum(physical) if physical and all(number(v) for v in physical) else None,
        recomputed_hit_rate=hit_rate,
        io_scope_note="Logical page bytes differ from kernel-accounted storage reads; process totals include setup.")
    profile = {k: v for k, v in report["metrics"].items() if k.startswith("profile_") or k in {
        "provider_request_ms", "expert_request_ms", "lookup_request_ms", "activation_copy_ms",
        "graph_setup_ms", "kernel_ms", "output_copy_ms", "lookup_decode_ms", "graph_teardown_ms", "retention_ms"}}
    report["profile_totals"] = profile or None
    return metrics


def route_map(data, expected_layers=48, metadata=None):
    header, rows, _, complete = routing.parse_jsonl(data, 100000, 1024 * 1024)
    if metadata is not None:
        metadata.update(run_id=header.get("run_id"),
            model=routing.object_projection(header.get("model"), routing.MODEL_KEYS),
            config=routing.object_projection(header.get("config"), routing.CONFIG_KEYS),
            profile=routing.object_projection(header.get("profile"), routing.CONFIG_KEYS))
    routing.check(complete and complete["complete"] is True and complete["truncated"] is False,
                  "routing capture incomplete")
    routing.check(header.get("model", {}).get("layer_count") == expected_layers, "unexpected layer count")
    result = {}
    for row in rows:
        key = (row["evaluation"], row["phase"], row["layer"], row["token_index"], row["token_position"])
        routing.check(key not in result, "duplicate routing row")
        routing.check(row["complete"] is True and row["truncated"] is False and
                      len(row["expert_ids"]) == 10 and row["routing_weights"] is not None,
                      "routing row incomplete")
        routing.check(row["sequence_id"] == header.get("sequence_id"), "routing sequence differs")
        result[key] = (row["expert_ids"], row["routing_weights"])
    expected = {(e, "prefill" if e == 0 else "decode", layer, 0, e)
                for e in range(16) for layer in range(expected_layers)}
    routing.check(set(result) == expected, "routing rows missing or unexpected")
    return result


def telemetry_summary(rows):
    observations, seen = [], set()
    for row in rows:
        for key in ("observations", "last_observations"):
            value = row.get(key)
            if isinstance(value, dict):
                stamp = value.get("started_monotonic_seconds")
                if stamp not in seen or stamp is None:
                    observations.append(value)
                    seen.add(stamp)
    # All saved observations originate on this host's monotonic clock. Sort
    # across preflight/guard/postflight before computing any endpoint delta.
    observations.sort(key=lambda r: r.get("started_monotonic_seconds") or 0)
    vmstat = {field: extent([nested(o, "vmstat", field, "value") for o in observations], True)
              for field in ("pswpin_bytes", "pswpout_bytes", "pgscan_direct_pages", "pgscan_kswapd_pages")}
    psi = {scope: {field: extent([nested(o, "memory_pressure", scope, field) for o in observations])
                   for field in ("avg10_percent", "avg60_percent", "avg300_percent")}
           for scope in ("some", "full")}
    for scope in psi:
        psi[scope]["total_stall_us"] = extent([nested(o, "memory_pressure", scope, "total_stall_us", "value")
                                               for o in observations], True)
    devices = {}
    for observation in observations:
        for device in observation.get("zram") or []:
            devices.setdefault((device.get("name"), device.get("identity")), []).append(device)
    zram = []
    for (name, _), samples in devices.items():
        item = {"name": name, "metrics": {}}
        for field in ("orig_data_size_bytes", "compr_data_size_bytes", "mem_used_total_bytes", "mem_used_max_bytes"):
            item["metrics"][field] = extent([nested(d, "mm_stat", field, "value") for d in samples])
        item["compression_ratio"] = extent([d.get("compression_ratio_uncompressed_to_compressed") for d in samples])
        for field in ("backing_read_bytes", "backing_write_bytes"):
            item["metrics"][field] = extent([nested(d, "bd_stat", field, "value") for d in samples], True)
        zram.append(item)
    return dict(samples=len(observations), vmstat=vmstat, memory_pressure=psi, zram=zram,
                collection_failures=sum(o.get("collection_failed") is True for o in observations),
                unavailable_field_observations=sum(len(o.get("unavailable", [])) for o in observations),
                scope="Host-wide sampled observations; changes are not attributable solely to this process.")


def audit(args):
    report = {"schema": "pmi.cache-campaign.audit.v1", "passed": False, "failures": [], "sources": {},
              "metrics": None, "performance": None, "profile_totals": None, "routing": {"available": False}}
    guard_data = read(args.guard, report, "guard")
    stdout_data = read(args.stdout, report, "stdout")
    guard_rows = []
    try:
        guard_rows, _ = guard_audit(guard_data, report)
    except (ValueError, TypeError, KeyError):
        report["failures"].append("guard_invalid")
    try:
        output_audit(stdout_data, report)
    except (ValueError, TypeError, KeyError):
        report["failures"].append("stdout_metrics_invalid")
    observation_rows = list(guard_rows)
    report["telemetry_guard_window"] = telemetry_summary(guard_rows)
    report["telemetry_additional_windows"] = []
    for i, path in enumerate(args.telemetry):
        data = read(path, report, "telemetry_" + str(i))
        if data is not None:
            try:
                additional = json_lines(data)
                observation_rows.extend(additional)
                report["telemetry_additional_windows"].append(
                    {"source": "telemetry_" + str(i), "summary": telemetry_summary(additional)})
            except ValueError:
                report["failures"].append("telemetry_invalid")
    report["telemetry"] = telemetry_summary(observation_rows)
    current = read(args.routing, report, "routing", required=any(
        value is not None for value in (args.routing, args.baseline_routing, args.validated_prefix)))
    if current is not None:
        try:
            configuration = {}
            routes = route_map(current, metadata=configuration)
            report["configuration"] = configuration
            report["routing"] = {"available": True, "complete": True, "rows": len(routes),
                                  "baseline_exact_match": None, "validated_prefix_exact_match": None}
            if args.baseline_routing:
                data = read(args.baseline_routing, report, "baseline_routing")
                if data is not None:
                    matched = routes == route_map(data)
                    report["routing"]["baseline_exact_match"] = matched
                    if not matched:
                        report["failures"].append("routing_baseline_mismatch")
            if args.validated_prefix:
                data = read(args.validated_prefix, report, "validated_prefix")
                if data is not None:
                    _, prefix, _, _ = routing.parse_legacy(data, hashlib.sha256(data).hexdigest(), "compact", 100000)
                    expected = {(e, "prefill" if e == 0 else "decode", layer, 0, e) for e in range(2) for layer in range(48)}
                    values = {(r["evaluation"], r["phase"], r["layer"], r["token_index"], r["token_position"]):
                              (r["expert_ids"], r["routing_weights"]) for r in prefix}
                    matched = set(values) == expected and all(routes[k] == v for k, v in values.items())
                    report["routing"]["validated_prefix_exact_match"] = matched
                    if not matched:
                        report["failures"].append("validated_prefix_mismatch")
        except (ValueError, TypeError, KeyError):
            report["failures"].append("routing_incomplete_invalid_or_mismatched")
    report["passed"] = not report["failures"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stdout", "guard", "output"):
        parser.add_argument("--" + name, required=True)
    for name in ("routing", "baseline-routing", "validated-prefix"):
        parser.add_argument("--" + name)
    parser.add_argument("--telemetry", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        report = audit(args)
        with Path(args.output).open("x") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
        print(json.dumps({"passed": report["passed"], "failures": report["failures"], "audit": args.output}))
        return 0 if report["passed"] else 1
    except (OSError, ValueError) as error:
        print("campaign audit failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
