#!/usr/bin/env python3
"""Small, offline contracts for the routing dataset. No weights or inference."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("routing_dataset", Path(__file__).resolve().parents[1] / "tools/routing_dataset.py")
dataset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dataset)


class RoutingDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = dataset.connect(self.root / "routing.sqlite")

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def source(self, rows, name="routing.jsonl"):
        path = self.root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def header(self, run="run-a", **extra):
        return dict(schema=dataset.SCHEMA, type="run", run_id=run, sequence_id="seq-a",
                    model={"identity": "fixture", "revision": "abc", "quantization": "F32"},
                    config={"context_length": 32, "cache_bytes": 4096}, **extra)

    def route(self, run="run-a", **extra):
        return dict(schema=dataset.SCHEMA, type="routing", run_id=run,
                    eval_id=0, phase="prefill", layer=0, token_index=0,
                    expert_ids=[3, 1], **extra)

    def end(self, run="run-a", **extra):
        return dict(schema=dataset.SCHEMA, type="complete", run_id=run,
                    complete=True, truncated=False, **extra)

    def test_live_idempotence_source_hash_and_nulls(self):
        path = self.source([self.header(), self.route(), self.end()])
        first = dataset.import_source(self.db, path, "jsonl")
        again = dataset.import_source(self.db, path, "jsonl")
        self.assertFalse(first["already_imported"])
        self.assertTrue(again["already_imported"])
        self.assertEqual(dataset.summary(self.db)["routes"], 1)
        row = list(dataset.export_rows(self.db))[0]
        self.assertEqual(row["source_sha256"], [first["source_sha256"]])
        for key in ("token_position", "routing_weights", "per_layer_cache", "complete", "truncated"):
            self.assertIsNone(row[key], key)
        status = self.db.execute("SELECT completion_seen,complete,truncated FROM run_sources").fetchone()
        self.assertEqual(tuple(status), (1, 1, 0))

    def test_appended_snapshot_deduplicates_without_mutating_history(self):
        prefix = [self.header(), self.route()]
        path = self.source(prefix)
        one = dataset.import_source(self.db, path, "jsonl")
        path = self.source(prefix + [self.end()])
        two = dataset.import_source(self.db, path, "jsonl")
        self.assertNotEqual(one["source_sha256"], two["source_sha256"])
        self.assertEqual(dataset.summary(self.db)["sources"], 2)
        self.assertEqual(dataset.summary(self.db)["routes"], 1)
        self.assertEqual(len(list(dataset.export_rows(self.db))[0]["source_sha256"]), 2)
        changed = self.route(routing_weights=[0.8, 0.2])
        path = self.source([self.header(), changed, self.end()])
        with self.assertRaisesRegex(dataset.DatasetError, "conflicting immutable routing"):
            dataset.import_source(self.db, path, "jsonl")
        self.assertEqual(dataset.summary(self.db)["sources"], 2)
        self.assertIsNone(list(dataset.export_rows(self.db))[0]["routing_weights"])

    def test_privacy_projection_ignores_unknown_fields(self):
        secret = "DO_NOT_RETAIN_PRIVATE_TEXT"
        header = self.header(prompt=secret, text=secret, token_ids=[9999999], hidden=[secret])
        header["config"].update(prompt=secret, messages=[secret], command=[secret], unknown=secret)
        header["model"]["unknown"] = secret
        header["profile"] = {"elapsed_ms": 1, "prompt": secret, "nested_unknown": {"text": secret}}
        route = self.route(routing_weights=[0.75, 0.25], output=secret, values=[9999999])
        unknown = dict(schema=dataset.SCHEMA, type="future", run_id="run-a", payload=secret)
        path = self.source([header, route, unknown, self.end()])
        dataset.import_source(self.db, path, "jsonl")
        self.assertNotIn(secret, "\n".join(self.db.iterdump()))
        self.assertNotIn("9999999", "\n".join(self.db.iterdump()))
        exported = list(dataset.export_rows(self.db))[0]
        self.assertEqual(exported["metadata"]["profile"], {"elapsed_ms": 1})
        self.assertEqual(exported["routing_weights"], [0.75, 0.25])

    def test_metrics_only_does_not_invent_routes_or_layer_cache(self):
        path = self.root / "stage3.json"
        path.write_text(json.dumps({"tokens": [123], "text": "SECRET", "tracing_enabled": False,
            "metrics": {"io_bytes": 100, "evaluations": [{"evaluation": 0, "phase": "prefill", "input_tokens": 1}]}}))
        result = dataset.import_source(self.db, path, "metrics", run_id="stage3")
        self.assertFalse(result["routing_present"])
        self.assertEqual(list(dataset.export_rows(self.db)), [])
        self.assertEqual(dataset.summary(self.db)["measurements"], 2)
        self.assertNotIn("SECRET", "\n".join(self.db.iterdump()))
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM measurements WHERE layer IS NOT NULL").fetchone()[0], 0)

    def test_layer_cache_present_only_when_recorded(self):
        metrics = dict(schema=dataset.SCHEMA, type="metrics", run_id="run-a", eval_id=0,
            phase="prefill", metrics={"io_bytes": 256},
            per_layer_cache=[{"layer": 0, "metrics": {"cache_hits": 2, "cache_misses": 3, "prompt": "SECRET"}}])
        dataset.import_source(self.db, self.source([self.header(), self.route(), metrics, self.end()]), "jsonl")
        row = list(dataset.export_rows(self.db))[0]
        self.assertEqual(row["per_layer_cache"], {"cache_hits": 2, "cache_misses": 3})
        self.assertNotIn("SECRET", "\n".join(self.db.iterdump()))

    def test_current_runtime_header_and_profile_fields(self):
        header = self.header()
        header["model"] = {"identity": None, "revision": None, "quantization": None,
                           "architecture": "qwen4exp", "layer_count": 48}
        header["config"].update(max_tokens=2, gpu_layers=0, prefill_chunk=1, cpu_repack=False,
            trace_mode="compact", cache_policy="previous-top4", mtp_enabled=False, text_only=True,
            sampling="greedy", routing_max_rows=65536, routing_max_bytes=8388608,
            backend_commit="4ebdf2c", implementation_commit="d58d75e", implementation_dirty=True,
            backend_patch_sha256="a" * 64)
        header["profile"] = {"enabled": True}
        metrics = dict(schema=dataset.SCHEMA, type="metrics", run_id="run-a", eval_id=0,
            phase="prefill", metrics={"cache_policy": "previous-top4", "routing_observation": True,
            "retention_requested_experts": 4, "retention_effective_experts": 4,
            "profile_enabled": True, "provider_request_ms": 12.5, "kernel_ms": 2.5,
            "profile_projection_calls": 144, "profile_lookup_rows": 16})
        dataset.import_source(self.db, self.source([header, self.route(), metrics, self.end()]), "jsonl")
        row = list(dataset.export_rows(self.db))[0]
        self.assertEqual(row["metadata"]["config"], header["config"])
        self.assertEqual(row["evaluation_metrics"], metrics["metrics"])
        self.assertIsNone(row["aggregate_metrics"])
        self.assertIsNone(row["per_layer_cache"])

    def test_device_projection_nested_roundtrip_privacy_and_idempotence(self):
        expected_projection = {
            "enabled": True, "profile_enabled": True, "calls": 12, "completed": 11, "failures": 1,
            "graph_current_bytes": 0, "graph_peak_bytes": 8192, "graph_capacity_bytes": 16384,
            "activation_upload_bytes": 1024, "output_readback_bytes": 512,
            "cache_request_ms": 1.25, "graph_setup_ms": .5, "activation_upload_ms": .125,
            "kernel_ms": 2.5, "output_readback_ms": .0625, "graph_teardown_ms": .25,
            "cache": {"requests": 12, "requested_bytes": 24576, "hits": 3, "misses": 9,
                      "failures": 1, "provider_requests": 8, "upload_bytes": 16384, "uploads": 8,
                      "evictions": 4, "current_bytes": 8192, "peak_bytes": 12288,
                      "capacity_bytes": 16384, "entries": 4, "active_leases": 0}}
        for backend in ("cpu", "device-cpu"):
            with self.subTest(backend=backend):
                run_id = "device-test-" + backend
                header = self.header(run=run_id)
                header["config"].update(projection_backend=backend, device_cache_bytes=16384,
                                        device_graph_bytes=16384)
                projection = json.loads(json.dumps(expected_projection))
                if backend == "cpu":
                    projection["enabled"] = False
                expected = {"projection_backend": backend, "device_projection": json.loads(json.dumps(projection))}
                secret = "PRIVATE_DEVICE_PROJECTION_SENTINEL"
                projection.update(prompt=secret, token_ids=[123456789], hidden_activations=[secret], unknown=secret)
                projection["cache"].update(prompt=secret, token_ids=[123456789], output_text=secret,
                                           unknown={"activation_upload_bytes": 123456789})
                header["config"]["prompt"] = secret
                metrics = dict(schema=dataset.SCHEMA, type="metrics", run_id=run_id,
                    eval_id=0, phase="prefill", metrics={"projection_backend": backend,
                                                         "device_projection": projection})
                source = self.source([header, self.route(run=run_id), metrics, self.end(run=run_id)], run_id + ".jsonl")
                first = dataset.import_source(self.db, source, "jsonl")
                count = dataset.summary(self.db)
                again = dataset.import_source(self.db, source, "jsonl")
                self.assertFalse(first["already_imported"])
                self.assertTrue(again["already_imported"])
                self.assertEqual(count, dataset.summary(self.db))
                row = list(dataset.export_rows(self.db, run_id=run_id))[0]
                self.assertEqual(row["evaluation_metrics"], expected)
                self.assertEqual(row["metadata"]["config"]["projection_backend"], backend)
                self.assertEqual(row["metadata"]["config"]["device_cache_bytes"], 16384)
                self.assertEqual(row["metadata"]["config"]["device_graph_bytes"], 16384)
                self.assertIsNone(row["per_layer_cache"])
                self.assertNotIn(secret, "\n".join(self.db.iterdump()))
                self.assertNotIn("123456789", "\n".join(self.db.iterdump()))

    def test_device_projection_keeps_existing_metadata_bounds(self):
        metrics = {"device_projection": {"cache": {"upload_bytes": float("inf")}}}
        with self.assertRaisesRegex(dataset.DatasetError, "nonfinite"):
            dataset.object_projection(metrics, dataset.METRIC_KEYS)
        for _ in range(13):
            metrics = {"device_projection": metrics}
        with self.assertRaisesRegex(dataset.DatasetError, "nesting limit"):
            dataset.object_projection(metrics, dataset.METRIC_KEYS)
        projected = dataset.object_projection({"cache_hits": 0}, dataset.METRIC_KEYS)
        self.assertEqual(projected, {"cache_hits": 0})  # Optional device data is never fabricated.

    def test_vram_unknown_zero_and_absence_remain_distinct(self):
        for label, reported in (("unknown", {"vram_bytes": None}),
                                ("cpu-zero", {"vram_bytes": 0}), ("absent", {})):
            with self.subTest(label=label):
                header = self.header(run=label)
                # Accounted backend allocations do not establish actual VRAM usage.
                values = {**reported, "device_projection": {"graph_peak_bytes": 8192,
                          "cache": {"current_bytes": 4096, "peak_bytes": 16384}}}
                metrics = dict(schema=dataset.SCHEMA, type="metrics", run_id=label,
                               eval_id=None, metrics=values)
                source = self.source([header, self.route(run=label), metrics, self.end(run=label)], label + ".jsonl")
                dataset.import_source(self.db, source, "jsonl")
                exported = list(dataset.export_rows(self.db, run_id=label))[0]["aggregate_metrics"]
                self.assertEqual(exported, values)
                self.assertEqual("vram_bytes" in exported, "vram_bytes" in reported)
                if reported:
                    self.assertEqual(exported["vram_bytes"], reported["vram_bytes"])

    def test_vulkan_identity_memory_scopes_and_privacy_roundtrip(self):
        device = {"backend_name": "Vulkan", "name": "Vulkan1", "description": "Synthetic GPU fixture",
                  "type": "discrete", "index": 1, "device_id": "0000:04:00.0"}
        expected_config = {"projection_backend": "device-vulkan", "projection_device_index": 1,
                           "projection_device": device, "device_cache_bytes": 16384, "device_graph_bytes": 8192}
        expected_metrics = {
            "path": "pmi_paged_qwen_vulkan_projections", "projection_backend": "device-vulkan",
            "projection_device": device, "vram_bytes": None,
            "vram_bytes_scope": "physical_device_usage_not_sampled",
            "device_projection": {"enabled": True, "graph_peak_bytes": 8192,
                "managed_bytes_scope": "ggml_buffer_reported_allocations_excluding_driver_staging_scratch",
                "cache": {"peak_bytes": 16384, "capacity_bytes": 16384}}}
        header = self.header()
        header["config"].update(json.loads(json.dumps(expected_config)))
        metrics_values = json.loads(json.dumps(expected_metrics))
        secret = "PRIVATE_VULKAN_IDENTITY_SENTINEL"
        for identity in (header["config"]["projection_device"], metrics_values["projection_device"]):
            identity.update(prompt=secret, token_ids=[987654321], uuid=secret, pci_address=secret,
                            unknown={"backend_name": secret})
        metrics_values["device_projection"]["prompt"] = secret
        metrics = dict(schema=dataset.SCHEMA, type="metrics", run_id="run-a", eval_id=None,
                       metrics=metrics_values)
        source = self.source([header, self.route(), metrics, self.end()])
        dataset.import_source(self.db, source, "jsonl")
        before = dataset.summary(self.db)
        again = dataset.import_source(self.db, source, "jsonl")
        self.assertTrue(again["already_imported"])
        self.assertEqual(before, dataset.summary(self.db))
        row = list(dataset.export_rows(self.db))[0]
        self.assertEqual(row["aggregate_metrics"], expected_metrics)
        for key, value in expected_config.items():
            self.assertEqual(row["metadata"]["config"][key], value)
        self.assertNotIn(secret, "\n".join(self.db.iterdump()))
        self.assertNotIn("987654321", "\n".join(self.db.iterdump()))

    def test_cpu_identity_null_and_old_evidence_remain_unchanged(self):
        for label, config, values in (
                ("old-cpu", {"projection_backend": "cpu"}, {"vram_bytes": 0}),
                ("explicit-cpu", {"projection_backend": "cpu", "projection_device_index": None,
                                  "projection_device": None},
                 {"projection_backend": "cpu", "projection_device": None, "vram_bytes": 0,
                  "vram_bytes_scope": "pmi_gpu_allocations_none"})):
            with self.subTest(label=label):
                header = self.header(run=label)
                header["config"].update(config)
                metrics = dict(schema=dataset.SCHEMA, type="metrics", run_id=label, eval_id=None,
                               metrics=values)
                source = self.source([header, self.route(run=label), metrics, self.end(run=label)], label + ".jsonl")
                dataset.import_source(self.db, source, "jsonl")
                row = list(dataset.export_rows(self.db, run_id=label))[0]
                self.assertEqual(row["metadata"]["config"], header["config"])
                self.assertEqual(row["aggregate_metrics"], values)

    def test_failed_startup_preserves_guard_scope_without_fabricating_routes(self):
        header = self.header()
        header["config"]["guard_source_sha256"] = "a" * 64
        metrics = dict(schema=dataset.SCHEMA, type="metrics", run_id="run-a", eval_id=None,
                       metrics={"guard_status": "limit_exceeded", "guard_reason": "max_system_swap_growth_bytes",
                                "guard_peak_group_rss_bytes": 4096, "guard_peak_group_swap_bytes": 0})
        end = dict(schema=dataset.SCHEMA, type="complete", run_id="run-a", complete=False,
                   truncated=False, reason="max_system_swap_growth_bytes")
        dataset.import_source(self.db, self.source([header, metrics, end]), "jsonl")
        self.assertEqual(list(dataset.export_rows(self.db)), [])
        self.assertEqual(dataset.observed_metrics(self.db, "run-a", None, None), metrics["metrics"])
        status = self.db.execute("SELECT complete,truncated FROM run_sources").fetchone()
        self.assertEqual(tuple(status), (0, 0))

    def test_dense_load_shards_scope_privacy_and_idempotence(self):
        for mode in ("default", "direct-required"):
            with self.subTest(mode=mode):
                direct = mode == "direct-required"
                shards = [{"shard_index": index, "direct_io_requested": direct,
                    "direct_io_required": direct, "direct_io_active": direct,
                    "payload_requested_bytes": count, "payload_completed_bytes": count,
                    "direct_read_bytes": 4096 if direct else 0,
                    "buffered_read_bytes": 0 if direct else count,
                    "staging_peak_bytes": 4096 if direct else 0, "fallback_count": 0}
                    for index, count in enumerate((300, 500))]
                dense = {"requested": mode, "effective": mode if direct else "buffered",
                    "scope": "llama_dense_tensor_payload_reads_excludes_gguf_metadata_and_pmi_provider",
                    "actual_read_scope": "read_or_fread_returned_bytes_including_direct_alignment_overlap",
                    "staging_peak_scope": "max_single_aligned_read_allocation",
                    "payload_requested_bytes": 800, "payload_completed_bytes": 800,
                    "direct_read_bytes": 8192 if direct else 0,
                    "buffered_read_bytes": 0 if direct else 800,
                    "staging_peak_bytes": 4096 if direct else 0, "fallback_count": 0, "shards": shards}
                header = self.header(run=mode)
                header["config"]["dense_load"] = json.loads(json.dumps(dense))
                observed = json.loads(json.dumps(dense))
                secret = "PRIVATE_DENSE_LOADER_SENTINEL"
                for value in (header["config"]["dense_load"], observed):
                    value.update(prompt=secret, input_ids=[987654321], unknown=secret, path=secret,
                                 cache_policy=secret, backend_name=secret)
                    value["shards"][0].update(filename=secret, source_path=secret, path=secret,
                                             name=secret, cache_bytes=987654321,
                                             token_ids=[987654321], hidden_activations=[secret])
                metrics = dict(schema=dataset.SCHEMA, type="metrics", run_id=mode,
                               eval_id=None, metrics={"dense_load": observed, "io_bytes": 123})
                path = self.source([header, self.route(run=mode), metrics, self.end(run=mode)], mode + ".jsonl")
                dataset.import_source(self.db, path, "jsonl")
                before = dataset.summary(self.db)
                self.assertTrue(dataset.import_source(self.db, path, "jsonl")["already_imported"])
                self.assertEqual(before, dataset.summary(self.db))
                row = list(dataset.export_rows(self.db, run_id=mode))[0]
                self.assertEqual(row["metadata"]["config"]["dense_load"], dense)
                self.assertEqual(row["aggregate_metrics"], {"dense_load": dense, "io_bytes": 123})
                self.assertNotIn(secret, "\n".join(self.db.iterdump()))
                self.assertNotIn("987654321", "\n".join(self.db.iterdump()))
                self.assertIsNone(row["per_layer_cache"])

    def test_dense_load_absence_null_and_bounds(self):
        for value in ({"io_bytes": 10}, {"io_bytes": 10, "dense_load": None}):
            self.assertEqual(dataset.object_projection(value, dataset.METRIC_KEYS), value)
        with self.assertRaisesRegex(dataset.DatasetError, "list limit"):
            dataset.object_projection({"dense_load": {"shards": [{}] * 4097}}, dataset.METRIC_KEYS)
        with self.assertRaisesRegex(dataset.DatasetError, "nonfinite"):
            dataset.object_projection({"dense_load": {"direct_read_bytes": float("inf")}}, dataset.CONFIG_KEYS)

    def test_projection_activation_preparation_roundtrip_privacy_and_history(self):
        for mode in ("backend-native", "cpu-compatible"):
            with self.subTest(mode=mode):
                prepared = mode == "cpu-compatible"
                header = self.header(run=mode)
                header["config"].update(projection_backend="device-vulkan", projection_activation=mode)
                preparation = {"activation_preparation_calls": 10 if prepared else 0,
                    "activation_preparation_bytes": 102400 if prepared else 0,
                    "activation_preparation_peak_bytes": 13440 if prepared else 0,
                    "activation_preparation_ms": 0.25 if prepared else 0,
                    "activation_preparation_peak_scope":
                        "host_packed_and_expanded_activation_vectors_excludes_allocator_overhead"}
                expected = {"projection_activation": mode, "device_projection": preparation}
                values = json.loads(json.dumps(expected))
                secret = "PRIVATE_PREPARED_ACTIVATION_SENTINEL"
                values["device_projection"].update(prepared_activations=[secret],
                    packed_activations=[987654321], prompt=secret, token_ids=[987654321])
                metrics = dict(schema=dataset.SCHEMA, type="metrics", run_id=mode,
                               eval_id=None, metrics=values)
                path = self.source([header, self.route(run=mode), metrics, self.end(run=mode)], mode + ".jsonl")
                dataset.import_source(self.db, path, "jsonl")
                before = dataset.summary(self.db)
                self.assertTrue(dataset.import_source(self.db, path, "jsonl")["already_imported"])
                self.assertEqual(before, dataset.summary(self.db))
                row = list(dataset.export_rows(self.db, run_id=mode))[0]
                self.assertEqual(row["metadata"]["config"]["projection_activation"], mode)
                self.assertEqual(row["aggregate_metrics"], expected)
                self.assertNotIn(secret, "\n".join(self.db.iterdump()))
                self.assertNotIn("987654321", "\n".join(self.db.iterdump()))
        for historical in ({"projection_backend": "cpu"}, {"projection_activation": None}):
            self.assertEqual(dataset.object_projection(historical, dataset.CONFIG_KEYS), historical)
            self.assertEqual(dataset.object_projection(historical, dataset.METRIC_KEYS), historical)

    def test_partial_live_metrics_and_final_evaluations_combine(self):
        partial = dict(schema=dataset.SCHEMA, type="metrics", run_id="run-a", eval_id=0,
                       metrics={"io_bytes": 128, "elapsed_ms": 2.5})
        aggregate = dict(schema=dataset.SCHEMA, type="metrics", run_id="run-a", eval_id=None,
            metrics={"io_bytes": 128, "evaluations": [{"evaluation": 0, "phase": "prefill",
            "io_bytes": 128, "elapsed_ms": 2.5, "rss_bytes": 4096}]})
        prefix = [self.header(), self.route(), partial]
        dataset.import_source(self.db, self.source(prefix), "jsonl")
        self.assertNotIn("rss_bytes", list(dataset.export_rows(self.db))[0]["evaluation_metrics"])
        dataset.import_source(self.db, self.source(prefix + [aggregate, self.end()]), "jsonl")
        row = list(dataset.export_rows(self.db))[0]
        self.assertEqual(row["evaluation_metrics"], {"io_bytes": 128, "elapsed_ms": 2.5, "rss_bytes": 4096})
        self.assertEqual(row["aggregate_metrics"], {"io_bytes": 128})
        # Earlier observations remain unchanged, while the richer snapshot is separate.
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM measurements WHERE evaluation=0").fetchone()[0], 2)
        bad = json.loads(json.dumps(aggregate))
        bad["metrics"]["evaluations"][0]["io_bytes"] = 129
        with self.assertRaisesRegex(dataset.DatasetError, "conflicting immutable measurement"):
            dataset.import_source(self.db, self.source(prefix + [bad, self.end()]), "jsonl")
        self.assertEqual(dataset.summary(self.db)["sources"], 2)

    def test_export_bound_fails_before_first_row_and_reads_do_not_create(self):
        second = self.route()
        second["layer"] = 1
        dataset.import_source(self.db, self.source([self.header(), self.route(), second]), "jsonl")
        generator = dataset.export_rows(self.db, limit=1)
        with self.assertRaisesRegex(dataset.DatasetError, "export limit"):
            next(generator)
        absent = self.root / "missing.sqlite"
        with self.assertRaises(dataset.sqlite3.Error):
            dataset.connect(absent, readonly=True)
        self.assertFalse(absent.exists())

    def test_split_labels_are_run_level_and_group_disjoint(self):
        first = self.source([self.header(prompt_group="group-one"), self.route()], "one.jsonl")
        second = self.source([self.header(run="run-b", prompt_group="group-one"), self.route(run="run-b")], "two.jsonl")
        dataset.import_source(self.db, first, "jsonl", split="train")
        with self.assertRaisesRegex(dataset.DatasetError, "prompt group"):
            dataset.import_source(self.db, second, "jsonl", split="test")
        self.assertEqual(dataset.summary(self.db)["runs"], 1)
        dataset.import_source(self.db, second, "jsonl", split="train")
        with self.assertRaisesRegex(dataset.DatasetError, "immutable"):
            dataset.split_run(self.db, "run-a", "test")
        self.assertEqual(len(list(dataset.export_rows(self.db, split="train"))), 2)
        self.assertEqual(list(dataset.export_rows(self.db, split="test")), [])

    def test_compact_multi_token_positions_and_weights(self):
        source = {"tokens": [99999], "metrics": {"evaluations": [
            {"evaluation": 0, "phase": "prefill", "input_tokens": 2},
            {"evaluation": 1, "phase": "decode", "input_tokens": 1}]}, "records": [
            {"evaluation": 0, "phase": "prefill", "name": "ffn_moe_topk-2", "shape": [2, 2, 1, 1], "values": [3, 1, 2, 0]},
            {"evaluation": 0, "phase": "prefill", "name": "ffn_moe_weights_norm-2", "shape": [2, 2, 1, 1], "values": [.8, .2, .7, .3]},
            {"evaluation": 1, "phase": "decode", "name": "ffn_moe_topk-2", "shape": [2, 1, 1, 1], "values": [2, 1]},
            {"name": "ffn_input-2", "values": [123.45]},
            {"name": "result_output", "values": [123.45]}]}
        path = self.root / "compact.json"
        path.write_text(json.dumps(source))
        dataset.import_source(self.db, path, "compact")
        rows = list(dataset.export_rows(self.db))
        self.assertEqual([r["token_position"] for r in rows], [0, 1, 2])
        self.assertEqual([r["routing_weights"] for r in rows], [[.8, .2], [.7, .3], None])
        self.assertNotIn("123.45", "\n".join(self.db.iterdump()))
        self.assertNotIn("99999", "\n".join(self.db.iterdump()))

    def test_compact_unknown_position_stays_null(self):
        source = {"records": [{"evaluation": 8, "phase": "decode", "name": "ffn_moe_topk-0", "shape": [1, 1], "values": [2]}]}
        path = self.root / "compact.json"
        path.write_text(json.dumps(source))
        dataset.import_source(self.db, path, "compact")
        self.assertIsNone(list(dataset.export_rows(self.db))[0]["token_position"])

    def test_source_interpretation_cannot_silently_change(self):
        path = self.source([self.header(), self.route()])
        dataset.import_source(self.db, path, "jsonl")
        config = self.root / "config.json"
        config.write_text(json.dumps({"cpu_repack": False}))
        with self.assertRaisesRegex(dataset.DatasetError, "different provenance"):
            dataset.import_source(self.db, path, "jsonl", config=config)
        saved = self.root / "metrics.json"
        saved.write_text('{"metrics":{"io_bytes":0}}')
        dataset.import_source(self.db, saved, "metrics")
        with self.assertRaisesRegex(dataset.DatasetError, "format"):
            dataset.import_source(self.db, saved, "compact")

    def test_parser_limits_and_atomic_failures(self):
        path = self.source([self.header(), self.route(), self.end()])
        for options, message in (({"max_bytes": 8}, "byte limit"),
                                 ({"max_line_bytes": 8}, "line exceeds"),
                                 ({"max_records": 1}, "record limit")):
            with self.assertRaisesRegex(dataset.DatasetError, message):
                dataset.import_source(self.db, path, "jsonl", **options)
        self.assertEqual(dataset.summary(self.db)["sources"], 0)
        path.write_text(json.dumps(self.header()) + '\n{"schema":"pmi.routing.v1","type":"routing"')
        with self.assertRaises(dataset.DatasetError):
            dataset.import_source(self.db, path, "jsonl")
        self.assertEqual(dataset.summary(self.db)["runs"], 0)

    def test_invalid_routing_and_completion_are_rejected(self):
        for extra in ({"routing_weights": [float("nan"), .2]}, {"routing_weights": [.3]},
                      {"token_position": -1}, {"truncated": "false"}):
            with self.assertRaises(dataset.DatasetError):
                dataset.import_source(self.db, self.source([self.header(), self.route(**extra)]), "jsonl")
        with self.assertRaisesRegex(dataset.DatasetError, "after completion"):
            dataset.import_source(self.db, self.source([self.header(), self.end(), self.route()]), "jsonl")
        self.assertEqual(dataset.summary(self.db)["sources"], 0)

    def test_truncation_and_metadata_provenance(self):
        end = self.end()
        end.update(complete=False, truncated=True, reason="row_limit")
        model = self.root / "model.json"
        model.write_text(json.dumps({"model": {"repository": "example/fixture", "files": [{"name": "weight.gguf", "sha256": "a" * 64}]}, "command": ["SECRET"]}))
        dataset.import_source(self.db, self.source([self.header(), self.route(), end]), "jsonl", model_metadata=model)
        row = list(dataset.export_rows(self.db))[0]
        self.assertIsNotNone(row["metadata"]["model_metadata_sha256"])
        self.assertEqual(row["metadata"]["model"]["files"][0]["sha256"], "a" * 64)
        self.assertEqual(tuple(self.db.execute("SELECT complete,truncated,reason FROM run_sources").fetchone()), (0, 1, "row_limit"))
        self.assertNotIn("SECRET", "\n".join(self.db.iterdump()))


if __name__ == "__main__":
    unittest.main()
