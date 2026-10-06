#!/usr/bin/env python3
"""Bounded, offline routing ingestion and replay; never opens model weights."""

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import stat
import sys

SCHEMA = "pmi.routing.v1"
DB_VERSION = 1
MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_RECORDS = 100000
MAX_LINE_BYTES = 1024 * 1024
SPLITS = ("unassigned", "train", "validation", "test", "holdout")
# Explicit projection prevents future trace fields from silently retaining text,
# token IDs, logits, hidden states, arbitrary commands, or filesystem paths.
MODEL_KEYS = {"identity", "architecture", "repository", "revision", "quantization",
              "backend_commit", "implementation_commit", "total_bytes", "complete",
              "full_file_sha256_verified", "official_notice_revision", "verified_at_utc",
              "name", "size", "bytes", "sha256", "files", "shards", "layer_count",
              "expert_count", "experts_used", "tensor_count", "split_count"}
CONFIG_KEYS = {"max_tokens", "gpu_layers", "prefill_chunk", "context_length", "context",
               "cache_bytes", "cache_capacity_bytes", "cpu_repack", "trace_mode",
               "reference_mode", "backend_commit", "implementation_commit", "threads",
               "cpu_threads", "batch_size", "microbatch_size", "seed", "temperature",
               "top_k", "top_p", "min_p", "sampling", "mtp_enabled", "text_only",
               "layer_count", "expert_count", "experts_used", "routing_max_rows",
               "routing_max_bytes", "max_rows", "max_bytes", "policy", "profile",
               "name", "version", "enabled", "capacity_bytes", "service_reserve_bytes",
               "routing_reserve_bytes", "pin_budget_bytes", "pin_ttl", "pin_top_n",
               "predictor", "predictor_version", "read_ahead", "lookahead", "mode",
               "decode_only", "per_layer", "cache_policy", "context_tokens", "layers",
               "implementation_dirty", "backend_patch_sha256", "guard_source_sha256", "stdout_source_sha256"}
CONFIG_KEYS |= {"projection_backend", "device_cache_bytes", "device_graph_bytes"}
CONFIG_KEYS |= {"projection_activation"}
CONFIG_KEYS |= {"projection_device_index", "projection_device", "backend_name",
                "description", "type", "index", "device_id"}
METRIC_KEYS = {"path", "elapsed_ms", "setup_ms", "prefill_ms", "decode_ms", "ttft_ms",
              "first_token_after_setup_ms", "prompt_tokens", "generated_tokens", "input_tokens",
              "trace_mode", "cpu_repack", "reference_mode", "prefill_evaluations",
              "decode_evaluations", "decode_evaluations_per_second", "page_requests",
              "expert_requests", "lookup_requests", "deferred_weight_bytes", "io_bytes",
              "cache_hits", "cache_misses", "hits", "misses", "evictions", "hit_rate",
              "cache_reserved_bytes", "cache_peak_reserved_bytes", "cache_capacity_bytes",
              "rss_bytes", "peak_rss_bytes", "process_storage_read_bytes", "vram_bytes",
              "process_storage_read_bytes_delta", "active_leases", "peak_bytes",
              "capacity_bytes", "reserved_bytes", "read_ms", "compute_ms", "wait_ms",
              "prefetch_requests", "prefetch_hits", "prefetch_bytes", "profile",
              "routed_cache_hits", "routed_cache_misses", "lookup_cache_hits",
              "lookup_cache_misses", "protected_bytes", "protection_hits",
              "protection_admissions", "protection_rejections", "protection_evictions"}
METRIC_KEYS |= {"cache_policy", "routing_observation", "retention_requested_experts",
               "retention_effective_experts", "retention_protected_bytes", "retention_peak_protected_bytes",
               "retention_service_reserve_bytes", "retention_fallbacks", "retention_transitions",
               "profile_enabled", "provider_request_ms", "activation_copy_ms", "graph_setup_ms",
               "kernel_ms", "output_copy_ms", "graph_teardown_ms", "retention_ms",
               "expert_request_ms", "lookup_request_ms", "lookup_decode_ms",
               "profile_projection_calls", "profile_lookup_rows", "profile_provider_requests",
               "profile_expert_requests", "profile_lookup_requests"}
METRIC_KEYS |= {"guard_status", "guard_reason", "guard_elapsed_ms", "guard_peak_group_rss_bytes",
               "guard_peak_group_swap_bytes", "guard_peak_system_swap_growth_bytes",
               "guard_minimum_available_bytes", "guard_command_returncode", "guard_runner_returncode"}
# Device execution remains optional. Retain only recorded operational counters
# and timings, using the same recursive privacy and size limits as CPU metrics.
METRIC_KEYS |= {"projection_backend", "device_projection", "enabled", "calls", "completed", "failures",
               "graph_current_bytes", "graph_peak_bytes", "graph_capacity_bytes",
               "activation_upload_bytes", "output_readback_bytes", "cache_request_ms",
               "activation_upload_ms", "output_readback_ms", "cache", "requests",
               "requested_bytes", "provider_requests", "upload_bytes", "uploads",
               "current_bytes", "entries"}
# Device identity and memory-accounting scope are supplied observations. Physical
# VRAM stays null/absent when unsampled; backend buffer counters never fill it in.
METRIC_KEYS |= {"projection_device", "backend_name", "name", "description", "type", "index",
               "device_id", "vram_bytes_scope", "managed_bytes_scope"}
METRIC_KEYS |= {"projection_activation", "activation_preparation_calls", "activation_preparation_bytes",
               "activation_preparation_peak_bytes", "activation_preparation_ms", "activation_preparation_peak_scope"}
# These observations describe only loader-issued dense tensor payload reads.
# Aligned direct syscall bytes may overlap metadata/deferred ranges; neither
# these counters nor the staging peak measure PMI page reads or process memory.
DENSE_LOAD_KEYS = {"dense_load", "requested", "effective", "scope", "actual_read_scope",
                   "staging_peak_scope", "shards", "shard_index", "direct_io_requested",
                   "direct_io_required", "direct_io_active", "payload_requested_bytes",
                   "payload_completed_bytes", "direct_read_bytes", "buffered_read_bytes",
                   "staging_peak_bytes", "fallback_count"}
CONFIG_KEYS |= DENSE_LOAD_KEYS
METRIC_KEYS |= DENSE_LOAD_KEYS
ID_PATTERN = re.compile(r"[A-Za-z0-9_.:/@+\-]{1,256}\Z")


class DatasetError(ValueError):
    pass


def check(condition, message):
    if not condition:
        raise DatasetError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def identifier(value, field, nullable=False):
    if value is None and nullable:
        return None
    check(isinstance(value, str) and ID_PATTERN.fullmatch(value), "invalid " + field)
    return value


def integer(value, field, nullable=False):
    if value is None and nullable:
        return None
    check(type(value) is int and 0 <= value <= 2**63 - 1, "invalid " + field)
    return value


def flag(value, field):
    check(value is None or type(value) is bool, "invalid " + field)
    return value


def project(value, allowed, depth=0):
    """Keep known operational fields only, including nested profile objects."""
    check(depth <= 12, "metadata nesting limit exceeded")
    if isinstance(value, dict):
        return {k: project(v, DENSE_LOAD_KEYS if k == "dense_load" else allowed, depth + 1)
                for k, v in value.items() if k in allowed}
    if isinstance(value, list):
        check(len(value) <= 4096, "metadata list limit exceeded")
        return [project(v, allowed, depth + 1) for v in value]
    check(value is None or type(value) in (str, int, float, bool), "unsupported metadata value")
    if isinstance(value, str):
        check(len(value) <= 1024, "metadata string limit exceeded")
    if type(value) is float:
        check(math.isfinite(value), "nonfinite metadata value")
    return value


def object_projection(value, allowed):
    if value is None:
        return None
    check(isinstance(value, dict), "expected metadata object")
    return project(value, allowed)


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            check(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def constant(_):
        raise DatasetError("nonfinite JSON constant")

    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise DatasetError("invalid JSON: " + str(error)) from error


def bounded_read(path, limit):
    check(limit > 0, "byte limit must be positive")
    check(stat.S_ISREG(Path(path).stat().st_mode), "source must be a regular file")
    with Path(path).open("rb") as stream:
        import os
        info = os.fstat(stream.fileno())
        check(stat.S_ISREG(info.st_mode), "source must be a regular file")
        check(info.st_size <= limit, "source exceeds byte limit")
        data = stream.read(limit + 1)
    check(len(data) <= limit, "source grew beyond byte limit")
    return data, hashlib.sha256(data).hexdigest()


DDL = """
CREATE TABLE IF NOT EXISTS dataset_meta(version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS sources(
 sha256 TEXT PRIMARY KEY, byte_count INTEGER NOT NULL, format TEXT NOT NULL,
 schema_name TEXT NOT NULL, interpretation_sha256 TEXT NOT NULL, imported_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs(
 run_id TEXT PRIMARY KEY, sequence_id TEXT NOT NULL, prompt_group TEXT,
 metadata_json TEXT NOT NULL, metadata_sha256 TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS run_sources(
 run_id TEXT NOT NULL REFERENCES runs, source_sha256 TEXT NOT NULL REFERENCES sources,
 routing_present INTEGER NOT NULL, completion_seen INTEGER NOT NULL,
 complete INTEGER, truncated INTEGER, reason TEXT, routing_rows INTEGER NOT NULL,
 PRIMARY KEY(run_id,source_sha256));
CREATE TABLE IF NOT EXISTS routes(
 event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs, sequence_id TEXT NOT NULL,
 evaluation INTEGER, phase TEXT, layer INTEGER NOT NULL, token_index INTEGER,
 token_position INTEGER, expert_ids_json TEXT NOT NULL, routing_weights_json TEXT,
 complete INTEGER, truncated INTEGER, payload_sha256 TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS routes_order ON routes(run_id,evaluation,token_index,layer);
CREATE INDEX IF NOT EXISTS routes_layer ON routes(layer,evaluation);
CREATE TABLE IF NOT EXISTS route_sources(
 event_id TEXT NOT NULL REFERENCES routes, source_sha256 TEXT NOT NULL REFERENCES sources,
 PRIMARY KEY(event_id,source_sha256));
CREATE TABLE IF NOT EXISTS measurements(
 event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs, evaluation INTEGER,
 phase TEXT, layer INTEGER, metrics_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS measurement_sources(
 event_id TEXT NOT NULL REFERENCES measurements, source_sha256 TEXT NOT NULL REFERENCES sources,
 PRIMARY KEY(event_id,source_sha256));
CREATE TABLE IF NOT EXISTS splits(
 run_id TEXT PRIMARY KEY REFERENCES runs, split TEXT NOT NULL,
 assigned_at TEXT NOT NULL);
"""


def connect(path, readonly=False):
    db = (sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
          if readonly else sqlite3.connect(path))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if readonly:
        check(version == DB_VERSION, "unsupported SQLite dataset version")
        return db
    check(version in (0, DB_VERSION), "unsupported SQLite dataset version")
    db.executescript(DDL)
    if not db.execute("SELECT version FROM dataset_meta").fetchone():
        db.execute("INSERT INTO dataset_meta VALUES (?)", (DB_VERSION,))
    check(db.execute("SELECT version FROM dataset_meta").fetchone()[0] == DB_VERSION,
          "unsupported dataset schema")
    db.execute("PRAGMA user_version=1")
    db.commit()
    return db


def split_run(db, run_id, split):
    check(split in SPLITS, "invalid split")
    if split == "unassigned":
        return
    run = db.execute("SELECT prompt_group FROM runs WHERE run_id=?", (run_id,)).fetchone()
    check(run is not None, "unknown run")
    old = db.execute("SELECT split FROM splits WHERE run_id=?", (run_id,)).fetchone()
    check(old is None or old[0] == split, "run split is immutable; create a separately governed dataset to change it")
    if run[0] is not None:
        assigned = db.execute("SELECT DISTINCT s.split FROM splits s JOIN runs r USING(run_id) "
                              "WHERE r.prompt_group=?", (run[0],)).fetchall()
        check(all(row[0] == split for row in assigned), "prompt group cannot span dataset splits")
    db.execute("INSERT OR IGNORE INTO splits VALUES (?,?,?)", (run_id, split, now()))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def normalized_route(row, default_sequence):
    ids = row.get("expert_ids")
    check(isinstance(ids, list) and 0 < len(ids) <= 4096, "invalid expert ID list")
    ids = [integer(value, "expert ID") for value in ids]
    check(len(set(ids)) == len(ids), "duplicate selected expert ID")
    weights = row.get("routing_weights")
    if weights is not None:
        check(isinstance(weights, list) and len(weights) == len(ids), "routing weights must match expert IDs")
        check(all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in weights),
              "invalid routing weight")
    return dict(sequence_id=identifier(row.get("sequence_id", default_sequence), "sequence ID"),
                evaluation=integer(row.get("eval_id", row.get("evaluation")), "evaluation", True),
                phase=identifier(row.get("phase"), "phase", True), layer=integer(row.get("layer"), "layer"),
                token_index=integer(row.get("token_index"), "token index", True),
                token_position=integer(row.get("token_position"), "token position", True),
                expert_ids=ids, routing_weights=weights,
                complete=flag(row.get("complete"), "route completeness"),
                truncated=flag(row.get("truncated"), "route truncation"))


def metrics_rows(metrics, evaluation=None, phase=None, layer=None):
    check(isinstance(metrics, dict), "expected metrics object")
    result = [(evaluation, phase, layer, object_projection(metrics, METRIC_KEYS))]
    evaluations = metrics.get("evaluations", [])
    check(isinstance(evaluations, list), "invalid evaluations")
    for row in evaluations:
        check(isinstance(row, dict), "invalid evaluation metrics")
        result.append((integer(row.get("eval_id", row.get("evaluation")), "evaluation", True),
                       identifier(row.get("phase"), "phase", True), None,
                       object_projection(row, METRIC_KEYS)))
    return result


def merge_metrics(left, right):
    """Union recorded partial observations; never sum counters or fill absent data."""
    combined = dict(left)
    for key, value in right.items():
        if key in combined:
            if isinstance(combined[key], dict) and isinstance(value, dict):
                value = merge_metrics(combined[key], value)
            else:
                check(combined[key] == value, "conflicting immutable measurement: " + key)
        combined[key] = value
    return combined


def coalesce_measurements(measurements):
    combined = {}
    for evaluation, phase, layer, values in measurements:
        key = (evaluation, layer)
        if key in combined:
            old_phase, old_values = combined[key]
            check(old_phase is None or phase is None or old_phase == phase,
                  "conflicting evaluation phases")
            combined[key] = (phase or old_phase, merge_metrics(old_values, values))
        else:
            combined[key] = (phase, values)
    return [(evaluation, phase, layer, values)
            for (evaluation, layer), (phase, values) in combined.items()]


def parse_jsonl(data, max_records, max_line_bytes):
    header, routes, measurements, completion = None, [], [], None
    seen = 0
    for line in data.splitlines():
        if not line.strip():
            continue
        seen += 1
        check(seen <= max_records, "record limit exceeded")
        check(len(line) <= max_line_bytes, "JSONL line exceeds byte limit")
        row = strict_json(line)
        check(isinstance(row, dict) and row.get("schema") == SCHEMA, "unsupported routing JSONL schema")
        kind = row.get("type")
        if kind == "run":
            check(header is None and seen == 1, "JSONL must have one leading run record")
            header = row
            identifier(row.get("run_id"), "run ID")
            continue
        check(header is not None, "JSONL run record is missing")
        check(row.get("run_id") == header["run_id"], "JSONL mixes run IDs")
        check(completion is None, "records found after completion")
        if kind == "routing":
            routes.append(normalized_route(row, header.get("sequence_id", "sequence-0")))
        elif kind == "metrics":
            ev = integer(row.get("eval_id", row.get("evaluation")), "evaluation", True)
            phase = identifier(row.get("phase"), "phase", True)
            measurements.extend(metrics_rows(row.get("metrics", {}), ev, phase))
            layer_rows = row.get("per_layer_cache") or []
            check(isinstance(layer_rows, list), "invalid layer cache list")
            for item in layer_rows:
                check(isinstance(item, dict), "invalid layer cache metrics")
                measurements.extend(metrics_rows(item.get("metrics", {}), ev, phase,
                                                 integer(item.get("layer"), "layer")))
        elif kind == "complete":
            completion = dict(complete=flag(row.get("complete"), "completeness"),
                              truncated=flag(row.get("truncated"), "truncation"),
                              reason=identifier(row.get("reason"), "completion reason", True))
        # Unknown record types carry no trusted dataset fields.
    check(header is not None, "JSONL run record is missing")
    check(len(routes) + len(measurements) <= max_records, "expanded record limit exceeded")
    return header, routes, coalesce_measurements(measurements), completion


def legacy_positions(metrics):
    """Derive positions only from a contiguous, unambiguous evaluation history."""
    result, expected, position = {}, 0, 0
    for row in metrics.get("evaluations", []):
        if not isinstance(row, dict) or row.get("evaluation") != expected:
            return {}
        size = row.get("input_tokens")
        if type(size) is not int or not 0 < size <= 2**31:
            return {}
        result[expected] = (position, size)
        expected += 1
        position += size
    return result


def parse_legacy(data, source_hash, kind, max_records):
    source = strict_json(data)
    check(isinstance(source, dict), "expected saved JSON object")
    metrics = source.get("metrics", source if kind == "metrics" else {})
    check(isinstance(metrics, dict), "expected saved metrics")
    rows = source.get("records", []) if kind == "compact" else []
    check(isinstance(rows, list) and len(rows) <= max_records, "record limit exceeded")
    positions = legacy_positions(metrics)
    tensors, routes = {}, []
    for row in rows:
        check(isinstance(row, dict), "invalid compact trace record")
        name = row.get("name", "")
        check(isinstance(name, str), "invalid compact trace tensor name")
        match = re.fullmatch(r"ffn_moe_(topk|weights_norm)-(\d+)", name)
        if not match:
            continue
        evaluation = integer(row.get("evaluation"), "evaluation", True)
        phase = identifier(row.get("phase"), "phase", True)
        shape, values = row.get("shape"), row.get("values")
        check(isinstance(shape, list) and 2 <= len(shape) <= 4 and
              all(type(n) is int and 0 < n <= 4096 for n in shape), "unsupported routing tensor shape")
        check(all(n == 1 for n in shape[2:]), "unsupported routing tensor batch axes")
        check(isinstance(values, list) and len(values) == shape[0] * shape[1], "incomplete routing tensor")
        key = (evaluation, phase, int(match[2]), match[1])
        check(key not in tensors, "duplicate compact routing tensor")
        tensors[key] = (shape, values, row)
    for (evaluation, phase, layer, name), (shape, values, row) in tensors.items():
        if name != "topk":
            continue
        weights = tensors.get((evaluation, phase, layer, "weights_norm"))
        check(weights is None or weights[0] == shape, "routing weight shape mismatch")
        for index in range(shape[1]):
            offset = index * shape[0]
            position = positions.get(evaluation)
            # A whole-evaluation shape is required before adding its token offset.
            absolute = position[0] + index if position and position[1] == shape[1] else None
            routes.append(normalized_route(dict(evaluation=evaluation, phase=phase, layer=layer,
                token_index=index, token_position=absolute, expert_ids=values[offset:offset + shape[0]],
                routing_weights=None if weights is None else weights[1][offset:offset + shape[0]],
                truncated=row.get("truncated")), "sequence-0"))
    measurements = metrics_rows(metrics) if metrics else []
    check(len(routes) + len(measurements) <= max_records, "expanded record limit exceeded")
    header = dict(run_id="legacy:" + source_hash, sequence_id="sequence-0")
    return header, routes, measurements, None


def load_metadata(path, keys):
    if path is None:
        return {}, None
    data, sha = bounded_read(path, 1024 * 1024)
    value = strict_json(data)
    check(isinstance(value, dict), "metadata must be an object")
    if keys == MODEL_KEYS:
        value = value.get("model", value)
    check(isinstance(value, dict), "metadata descriptor must be an object")
    return object_projection(value, keys), sha


def import_source(db, source, kind, *, model_metadata=None, config=None, run_id=None,
                  sequence_id=None, prompt_group=None, split="unassigned", quantization=None,
                  max_bytes=MAX_SOURCE_BYTES, max_records=MAX_RECORDS,
                  max_line_bytes=MAX_LINE_BYTES):
    check(kind in ("jsonl", "compact", "metrics"), "unsupported source format")
    check(max_records > 0 and max_line_bytes > 0, "parser limits must be positive")
    data, source_hash = bounded_read(source, max_bytes)
    header, routes, measurements, completion = (parse_jsonl(data, max_records, max_line_bytes)
        if kind == "jsonl" else parse_legacy(data, source_hash, kind, max_records))
    external_model, model_hash = load_metadata(model_metadata, MODEL_KEYS)
    external_config, config_hash = load_metadata(config, CONFIG_KEYS)
    identifier(run_id, "run ID", True)
    identifier(sequence_id, "sequence ID", True)
    prompt_group = identifier(prompt_group or header.get("prompt_group"), "prompt group", True)
    selected_run = run_id or header["run_id"]
    check(kind != "jsonl" or run_id is None or run_id == header["run_id"], "cannot replace live run ID")
    sequence = sequence_id or header.get("sequence_id", "sequence-0")
    identifier(sequence, "sequence ID")
    for row in routes:
        if kind != "jsonl" or sequence_id is not None:
            row["sequence_id"] = sequence
    model = object_projection(header.get("model") or {}, MODEL_KEYS)
    for key, value in external_model.items():
        check(key not in model or model[key] is None or model[key] == value, "conflicting model metadata: " + key)
        model[key] = value
    if quantization is not None:
        identifier(quantization, "quantization")
        check(model.get("quantization") in (None, quantization), "conflicting model quantization")
        model["quantization"] = quantization
    configuration = object_projection(header.get("config") or {}, CONFIG_KEYS)
    for key, value in external_config.items():
        check(key not in configuration or configuration[key] == value, "conflicting run config: " + key)
        configuration[key] = value
    metadata = {"model": model, "config": configuration,
                **{key: object_projection(header.get(key), CONFIG_KEYS | METRIC_KEYS)
                   for key in ("context", "cache", "policy", "profile")},
                "model_metadata_sha256": model_hash, "config_source_sha256": config_hash}
    interpretation = digest(dict(run_id=selected_run, sequence_id=sequence, prompt_group=prompt_group, metadata=metadata))
    old = db.execute("SELECT interpretation_sha256,format FROM sources WHERE sha256=?", (source_hash,)).fetchone()
    check(old is None or tuple(old) == (interpretation, kind),
          "source already imported with different provenance/configuration/format")
    with db:
        if old is not None:
            split_run(db, selected_run, split)
            return dict(source_sha256=source_hash, run_id=selected_run, already_imported=True,
                        routing_rows=len(routes), measurement_rows=len(measurements))
        prior_run = db.execute("SELECT metadata_sha256,sequence_id,prompt_group FROM runs WHERE run_id=?", (selected_run,)).fetchone()
        check(prior_run is None or tuple(prior_run) == (digest(metadata), sequence, prompt_group),
              "run ID already has different immutable metadata")
        db.execute("INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?)",
                   (selected_run, sequence, prompt_group, canonical(metadata), digest(metadata)))
        db.execute("INSERT INTO sources VALUES (?,?,?,?,?,?)",
                   (source_hash, len(data), kind, SCHEMA if kind == "jsonl" else "pmi.saved." + kind + ".v1", interpretation, now()))
        db.execute("INSERT INTO run_sources VALUES (?,?,?,?,?,?,?,?)",
                   (selected_run, source_hash, bool(routes), completion is not None,
                    completion["complete"] if completion else None, completion["truncated"] if completion else None,
                    completion["reason"] if completion else None, len(routes)))
        for row in routes:
            coordinate = {key: row[key] for key in ("sequence_id", "evaluation", "phase", "layer", "token_index", "token_position")}
            event = digest([selected_run, coordinate])
            payload = digest(row)
            previous = db.execute("SELECT payload_sha256 FROM routes WHERE event_id=?", (event,)).fetchone()
            check(previous is None or previous[0] == payload, "conflicting immutable routing event")
            db.execute("INSERT OR IGNORE INTO routes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (event, selected_run, row["sequence_id"], row["evaluation"], row["phase"], row["layer"],
                        row["token_index"], row["token_position"], canonical(row["expert_ids"]),
                        canonical(row["routing_weights"]) if row["routing_weights"] is not None else None,
                        row["complete"], row["truncated"], payload))
            db.execute("INSERT OR IGNORE INTO route_sources VALUES (?,?)", (event, source_hash))
        for evaluation, phase, layer, values in measurements:
            event = digest([selected_run, evaluation, phase, layer, values])
            payload = digest(values)
            previous = db.execute("""SELECT event_id,phase,metrics_json,payload_sha256 FROM measurements
                WHERE run_id=? AND evaluation IS ? AND layer IS ?""", (selected_run, evaluation, layer)).fetchall()
            for observation in previous:
                check(observation["phase"] is None or phase is None or observation["phase"] == phase,
                      "conflicting evaluation phases")
                merge_metrics(json.loads(observation["metrics_json"]), values)
                if observation["payload_sha256"] == payload and observation["phase"] == phase:
                    event = observation["event_id"]
            db.execute("INSERT OR IGNORE INTO measurements VALUES (?,?,?,?,?,?,?)",
                       (event, selected_run, evaluation, phase, layer, canonical(values), payload))
            db.execute("INSERT OR IGNORE INTO measurement_sources VALUES (?,?)", (event, source_hash))
        split_run(db, selected_run, split if split != "unassigned" else header.get("split", "unassigned"))
    return dict(source_sha256=source_hash, run_id=selected_run, already_imported=False,
                routing_rows=len(routes), measurement_rows=len(measurements),
                routing_present=bool(routes), completion_seen=completion is not None)


def observed_metrics(db, run_id, evaluation, layer, phase=None):
    observations = db.execute("""SELECT phase,metrics_json FROM measurements
        WHERE run_id=? AND evaluation IS ? AND layer IS ?""", (run_id, evaluation, layer)).fetchall()
    values = None
    for observation in observations:
        if phase is not None and observation["phase"] is not None and phase != observation["phase"]:
            continue
        values = merge_metrics(values or {}, json.loads(observation["metrics_json"]))
    return values


def export_rows(db, *, run_id=None, split=None, limit=MAX_RECORDS):
    """Logical expert routing rows, not reconstructed physical page requests."""
    check(limit > 0, "export limit must be positive")
    count = db.execute("""SELECT COUNT(*) FROM routes r LEFT JOIN splits s USING(run_id)
        WHERE (? IS NULL OR r.run_id=?) AND (? IS NULL OR COALESCE(s.split,'unassigned')=?)""",
        (run_id, run_id, split, split)).fetchone()[0]
    check(count <= limit, "export limit exceeded; select a smaller run/split")
    query = """SELECT r.*,u.metadata_json,u.prompt_group,COALESCE(s.split,'unassigned') AS split
               FROM routes r JOIN runs u USING(run_id) LEFT JOIN splits s USING(run_id)
               WHERE (? IS NULL OR r.run_id=?) AND (? IS NULL OR COALESCE(s.split,'unassigned')=?)
               ORDER BY r.run_id,r.evaluation,r.token_index,r.layer,r.event_id LIMIT ?"""
    rows = db.execute(query, (run_id, run_id, split, split, limit + 1))
    for index, row in enumerate(rows):
        check(index < limit, "export limit exceeded; select a smaller run/split")
        source_rows = db.execute("SELECT source_sha256 FROM route_sources WHERE event_id=? ORDER BY source_sha256", (row["event_id"],)).fetchall()
        layer_metrics = observed_metrics(db, row["run_id"], row["evaluation"], row["layer"], row["phase"])
        evaluation_metrics = (observed_metrics(db, row["run_id"], row["evaluation"], None, row["phase"])
                              if row["evaluation"] is not None else None)
        aggregate_metrics = observed_metrics(db, row["run_id"], None, None)
        statuses = [dict(value) for value in db.execute("""SELECT source_sha256,completion_seen,complete,truncated,reason
                    FROM run_sources WHERE run_id=? ORDER BY source_sha256""", (row["run_id"],))]
        yield dict(schema="pmi.routing.replay.v1", event_id=row["event_id"], run_id=row["run_id"],
                   sequence_id=row["sequence_id"], prompt_group=row["prompt_group"], split=row["split"],
                   eval_id=row["evaluation"], phase=row["phase"], layer=row["layer"],
                   token_index=row["token_index"], token_position=row["token_position"],
                   expert_ids=json.loads(row["expert_ids_json"]),
                   routing_weights=None if row["routing_weights_json"] is None else json.loads(row["routing_weights_json"]),
                   complete=None if row["complete"] is None else bool(row["complete"]),
                   truncated=None if row["truncated"] is None else bool(row["truncated"]),
                   per_layer_cache=layer_metrics, evaluation_metrics=evaluation_metrics,
                   aggregate_metrics=aggregate_metrics,
                   source_completeness=statuses,
                   metadata=json.loads(row["metadata_json"]), source_sha256=[r[0] for r in source_rows])


def summary(db):
    return {"schema_version": DB_VERSION,
            **{table: db.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
               for table in ("sources", "runs", "routes", "measurements")},
            "runs_detail": [dict(row) for row in db.execute("""
                SELECT r.run_id,COALESCE(s.split,'unassigned') AS split,
                (SELECT COUNT(*) FROM routes t WHERE t.run_id=r.run_id) AS routing_rows,
                (SELECT COUNT(*) FROM measurements m WHERE m.run_id=r.run_id AND m.layer IS NOT NULL) AS layer_cache_rows,
                EXISTS(SELECT 1 FROM routes t WHERE t.run_id=r.run_id) AS routing_present,
                EXISTS(SELECT 1 FROM run_sources t WHERE t.run_id=r.run_id AND t.completion_seen=1) AS completion_seen
                FROM runs r LEFT JOIN splits s USING(run_id) ORDER BY r.run_id""")]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("import")
    ingest.add_argument("--db", required=True)
    ingest.add_argument("--source", required=True)
    ingest.add_argument("--format", choices=("jsonl", "compact", "metrics"), required=True)
    for name in ("model-metadata", "config", "run-id", "sequence-id", "prompt-group", "quantization"):
        ingest.add_argument("--" + name)
    ingest.add_argument("--split", choices=SPLITS, default="unassigned")
    ingest.add_argument("--max-bytes", type=int, default=MAX_SOURCE_BYTES)
    ingest.add_argument("--max-records", type=int, default=MAX_RECORDS)
    ingest.add_argument("--max-line-bytes", type=int, default=MAX_LINE_BYTES)
    for name in ("export", "replay"):
        export = commands.add_parser(name)
        export.add_argument("--db", required=True)
        export.add_argument("--run-id")
        export.add_argument("--split", choices=SPLITS)
        export.add_argument("--limit", type=int, default=MAX_RECORDS)
    inspect = commands.add_parser("summary")
    inspect.add_argument("--db", required=True)
    split = commands.add_parser("split")
    split.add_argument("--db", required=True)
    split.add_argument("--run-id", required=True)
    split.add_argument("--label", choices=SPLITS[1:], required=True)
    args = parser.parse_args(argv)
    db = None
    try:
        db = connect(args.db, readonly=args.command in ("summary", "export", "replay"))
        if args.command == "import":
            values = vars(args).copy()
            for key in ("command", "db"):
                values.pop(key)
            values["kind"] = values.pop("format")
            print(canonical(import_source(db, **values)))
        elif args.command in ("export", "replay"):
            for row in export_rows(db, run_id=args.run_id, split=args.split, limit=args.limit):
                print(canonical(row))
        elif args.command == "split":
            with db:
                split_run(db, args.run_id, args.label)
            print(canonical({"run_id": args.run_id, "split": args.label}))
        else:
            print(canonical(summary(db)))
        return 0
    except (DatasetError, OSError, sqlite3.Error) as error:
        print("routing dataset: " + str(error), file=sys.stderr)
        return 1
    finally:
        if db is not None:
            db.close()


if __name__ == "__main__":
    sys.exit(main())
