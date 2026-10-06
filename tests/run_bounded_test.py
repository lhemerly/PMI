#!/usr/bin/env python3
"""Small Linux supervisor regressions; no models or intentional system pressure.

Run directly with Python, from any working directory. Fixtures expire on their
own within three seconds if supervision regresses. Normal execution takes about
two seconds and the largest actual allocation is 32 MiB.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


RUNNER = Path(__file__).resolve().parents[1] / "tools/run_bounded.py"
sys.path.insert(0, str(RUNNER.parent))
from pressure_policy_test import snapshot


@unittest.skipUnless(sys.platform.startswith("linux") and hasattr(os, "WNOWAIT"),
                     "Linux /proc and waitid(WNOWAIT) are required")
class BoundedRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pmi-bounded-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def arguments(self, name, code, extra=()):
        return [sys.executable, str(RUNNER), "--wall-seconds", "2",
                "--max-rss-bytes", "64MiB", "--min-available-bytes", "0",
                # Do not make unrelated host swap activity a CI test failure.
                "--max-swap-growth-bytes", "1TiB", "--interval-seconds", "0.05",
                "--term-grace-seconds", "0.1", "--kill-wait-seconds", "0.5",
                "--log", str(self.directory / f"{name}.jsonl"),
                "--output", str(self.directory / f"{name}.out"), *extra,
                "--", sys.executable, "-c", code]

    def invoke(self, name, code, expected=0, extra=()):
        result = subprocess.run(self.arguments(name, code, extra), capture_output=True,
                                text=True, timeout=4)
        self.assertEqual(result.returncode, expected, result.stderr)
        records = [json.loads(line) for line in
                   (self.directory / f"{name}.jsonl").read_text().splitlines()]
        self.assertEqual(records[0]["event"], "start")
        final = records[-1]
        self.assertEqual(final["event"], "final")
        self.assertEqual(final["runner_returncode"], expected)
        self.assertFalse(final.get("remaining_live_pids"), final)
        if final["command_started"]:
            self.assertGreaterEqual(final["samples"], 1)
            self.assertIsNotNone(final["command_returncode"])
        return final, (self.directory / f"{name}.out").read_text()

    def test_normal_completion_and_command_output(self):
        # Exit before the default one-second poll: observing completion at the
        # wall deadline must not turn a quick command into a false timeout.
        final, output = self.invoke("normal", 'print("normal output")',
                                    extra=("--wall-seconds", "0.5", "--interval-seconds", "1"))
        self.assertEqual(final["status"], "completed")
        self.assertIsNone(final["reason"])
        self.assertEqual(output, "normal output\n")
        start = json.loads((self.directory / "normal.jsonl").read_text().splitlines()[0])
        self.assertEqual(start["limits"]["max_system_swap_growth_bytes"],
                         start["limits"]["max_swap_growth_bytes"])
        self.assertEqual(final["sampled_peak_swap_growth_bytes"],
                         max(final["sampled_peak_group_swap_bytes"],
                             final["sampled_peak_system_swap_growth_bytes"]))
        self.assertEqual(start["observations"]["schema_version"], 1)
        self.assertIn("unavailable", final["last_observations"])
        self.assertEqual(final["last_sample"]["system_swap_used_bytes"],
                         final["last_sample"]["meminfo_bytes"]["SwapTotal"] -
                         final["last_sample"]["meminfo_bytes"]["SwapFree"])

    def test_explicit_system_swap_limit_is_logged_separately(self):
        self.invoke("separate-swap", "pass", extra=("--max-system-swap-growth-bytes", "2TiB"))
        start = json.loads((self.directory / "separate-swap.jsonl").read_text().splitlines()[0])
        self.assertEqual(start["limits"]["max_swap_growth_bytes"], 1024**4)
        self.assertEqual(start["limits"]["max_system_swap_growth_bytes"], 2 * 1024**4)

    def test_command_exit_code_is_preserved(self):
        final, _ = self.invoke("exit-code", "raise SystemExit(7)", expected=7)
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["command_returncode"], 7)

    def test_wall_limit_kills_only_its_term_resistant_group(self):
        sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
        try:
            worker = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(2)"
            code = ("import json,signal,subprocess,sys,time; "
                    "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                    f"child=subprocess.Popen([sys.executable,'-c',{worker!r}]); "
                    "print(json.dumps({'worker_pid':child.pid}),flush=True); time.sleep(2)")
            final, output = self.invoke("group", code, expected=124,
                                       extra=("--wall-seconds", "0.6"))
            self.assertEqual(final["reason"], "wall_seconds")
            self.assertTrue(final["term_sent"])
            self.assertTrue(final["kill_sent"])
            self.assertIn(sentinel.poll(), (None, 0), "unrelated sentinel received a signal")
            worker_pid = json.loads(output)["worker_pid"]
            try:
                state = Path(f"/proc/{worker_pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
            except FileNotFoundError:
                state = "X"
            self.assertIn(state, ("Z", "X", "x"), "owned worker remained live")
        finally:
            # Popen retains ownership until wait(), avoiding PID reuse here.
            if sentinel.poll() is None:
                sentinel.terminate()
            sentinel.wait(timeout=1)

    def test_rss_guard_stops_small_owned_allocation(self):
        final, _ = self.invoke("rss", "import time; data=bytearray(32*1024*1024); time.sleep(2)",
                               expected=125, extra=("--max-rss-bytes", "16MiB"))
        self.assertEqual(final["reason"], "max_rss_bytes")
        self.assertGreater(final["sampled_peak_rss_bytes"], 16 * 1024**2)

    def test_hard_address_space_limit_rejects_allocation(self):
        # Reserve virtual space without touching pages. Even if the guard
        # regresses, this fixture does not commit a large resident allocation.
        code = ("import errno,mmap\ntry:\n data=mmap.mmap(-1,96*1024*1024)\n"
                "except OSError as error:\n"
                " if error.errno != errno.ENOMEM: raise\n"
                " print('allocation denied')\n"
                "else:\n raise SystemExit('address-space limit did not apply')")
        final, output = self.invoke("address-space", code, extra=("--as-bytes", "64MiB"))
        self.assertEqual(final["status"], "completed")
        self.assertEqual(output, "allocation denied\n")

    def test_existing_logs_are_not_overwritten_and_command_does_not_start(self):
        self.invoke("existing", 'print("preserve this")')
        log, output = self.directory / "existing.jsonl", self.directory / "existing.out"
        original = (log.read_bytes(), output.read_bytes())
        marker = self.directory / "must-not-exist"
        code = f"from pathlib import Path; Path({str(marker)!r}).write_text('started')"
        result = subprocess.run(self.arguments("existing", code), capture_output=True,
                                text=True, timeout=2)
        self.assertEqual(result.returncode, 125)
        self.assertEqual((log.read_bytes(), output.read_bytes()), original)
        self.assertFalse(marker.exists())


@unittest.skipUnless(sys.platform.startswith("linux") and hasattr(os, "WNOWAIT"),
                     "Linux supervisor module required")
class SwapLimitsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Predicate tests use synthetic counters only; no swapping or memory
        # pressure is induced and no model command is executed.
        spec = importlib.util.spec_from_file_location("pmi_run_bounded_test", RUNNER)
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def violations(self, group, system, override=None):
        args = argparse.Namespace(max_rss_bytes=100, min_available_bytes=0,
                                  max_swap_growth_bytes=64,
                                  max_system_swap_growth_bytes=override)
        sample = {"group_rss_bytes": 0, "mem_available_bytes": 100,
                  "group_swap_bytes": group, "system_swap_growth_bytes": system,
                  "swap_growth_bytes": max(group, system)}
        return self.runner.resource_violations(sample, args)

    def test_default_preserves_original_threshold_decisions(self):
        for group in (0, 64, 65):
            for system in (0, 64, 65):
                with self.subTest(group=group, system=system):
                    actual = self.violations(group, system)
                    self.assertEqual(bool(actual), max(group, system) > 64)
                    self.assertEqual("max_swap_growth_bytes" in actual, group > 64)
                    self.assertEqual("max_system_swap_growth_bytes" in actual, system > 64)

    def test_explicit_limits_are_enforced_independently(self):
        self.assertEqual(self.violations(64, 512, 512), [])
        self.assertEqual(self.violations(65, 100, 512), ["max_swap_growth_bytes"])
        self.assertEqual(self.violations(0, 513, 512), ["max_system_swap_growth_bytes"])
        self.assertEqual(self.violations(65, 513, 512),
                         ["max_swap_growth_bytes", "max_system_swap_growth_bytes"])

    def test_zero_system_limit_does_not_fall_back_to_group_limit(self):
        self.assertEqual(self.violations(0, 0, 0), [])
        self.assertEqual(self.violations(0, 1, 0), ["max_system_swap_growth_bytes"])

    def test_unavailable_observations_preserve_completion_and_rss_enforcement(self):
        # Exercise the actual runner with every optional source absent. Neither
        # a successful command nor a required RSS stop depends on those sources.
        with tempfile.TemporaryDirectory(prefix="pmi-missing-observations-") as directory:
            root = Path(directory)
            for code, limit, expected in (("print('ok')", 64 * 1024**2, 0),
                    ("import time; data=bytearray(32*1024*1024); time.sleep(2)",
                     16 * 1024**2, 125)):
                observer = self.runner.LinuxMemoryTelemetry(proc_root=root / "proc",
                                                           sys_root=root / "sys")
                records = []
                args = argparse.Namespace(wall_seconds=2, max_rss_bytes=limit,
                    min_available_bytes=0, max_swap_growth_bytes=1024**4,
                    max_system_swap_growth_bytes=None, as_bytes=None,
                    interval_seconds=0.05, term_grace_seconds=0.1, kill_wait_seconds=0.5,
                    command=[sys.executable, "-c", code])
                with tempfile.TemporaryFile(mode="w+") as output, mock.patch.object(
                        self.runner, "LinuxMemoryTelemetry", return_value=observer):
                    result = self.runner.supervise(args,
                        lambda event, **values: records.append({"event": event, **values}), output)
                self.assertEqual(result, expected)
                self.assertTrue(records[0]["observations"]["unavailable"])
                self.assertIsNone(records[0]["observations"]["vmstat"])
                self.assertFalse(records[-1]["monitor_errors"])
                self.assertFalse(records[-1]["remaining_live_pids"])
                self.assertEqual(records[-1]["reason"], "max_rss_bytes" if expected else None)

    def small_args(self, code):
        return argparse.Namespace(wall_seconds=2, max_rss_bytes=64 * 1024**2,
            min_available_bytes=0, max_swap_growth_bytes=1024**4,
            max_system_swap_growth_bytes=None, as_bytes=None, interval_seconds=0.05,
            term_grace_seconds=0.1, kill_wait_seconds=0.5,
            command=[sys.executable, "-c", code])

    def test_optional_initialization_and_collection_failure_are_not_stops(self):
        for method in ("__init__", "sample"):
            with self.subTest(method=method):
                records = []
                with tempfile.TemporaryFile(mode="w+") as output, mock.patch.object(
                        self.runner.LinuxMemoryTelemetry, method,
                        side_effect=ValueError("optional fixture failure")):
                    result = self.runner.supervise(self.small_args("pass"),
                        lambda event, **values: records.append({"event": event, **values}), output)
                self.assertEqual(result, 0)
                self.assertTrue(records[0]["observations"]["collection_failed"])
                self.assertFalse(records[-1]["monitor_errors"])
                self.assertIsNone(records[-1]["reason"])
                # The full record remains serializable on a failed collector.
                json.dumps(records, allow_nan=False)

    def test_optional_reads_stop_once_cleanup_begins(self):
        records, collection_after_stop = [], []
        stopping = False

        def emit(event, **values):
            nonlocal stopping
            if event in ("stop", "cleanup"):
                stopping = True
            records.append({"event": event, **values})

        def observe():
            collection_after_stop.append(stopping)
            return {"schema_version": 1, "unavailable": []}

        args = self.small_args("import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(2)")
        args.wall_seconds = 0.3
        with tempfile.TemporaryFile(mode="w+") as output, mock.patch.object(
                self.runner.LinuxMemoryTelemetry, "sample", side_effect=observe):
            result = self.runner.supervise(args, emit, output)
        self.assertEqual(result, 124)
        self.assertTrue(records[-1]["kill_sent"])
        self.assertTrue(collection_after_stop)
        self.assertFalse(any(collection_after_stop))
        skipped = [item for item in records if item.get("observations_status") == "skipped_cleanup_or_deadline"]
        self.assertTrue(skipped)
        self.assertTrue(all(item["observations"] is None for item in skipped))


@unittest.skipUnless(sys.platform.startswith("linux") and hasattr(os, "WNOWAIT"),
                     "Linux supervisor module required")
class PressureRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = importlib.import_module("run_bounded")

    small_args = SwapLimitsTest.small_args

    def observer(self, changes=None):
        observer = mock.Mock()
        base = previous = None
        count = 0

        def sample():
            nonlocal base, previous, count
            count += 1
            updates = changes(count) if changes is not None else {}
            item = snapshot(start=10 + count, baseline=base, previous=previous, **updates)
            if base is None:
                base = item
            previous = item
            return item
        observer.sample.side_effect = sample
        return observer

    def supervise(self, args, observer, emit_hook=None):
        records = []

        def emit(event, **values):
            records.append({"event": event, **values})
            if emit_hook:
                emit_hook(event, values)

        with tempfile.TemporaryFile(mode="w+") as output, mock.patch.object(
                self.runner, "LinuxMemoryTelemetry", return_value=observer):
            result = self.runner.supervise(args, emit, output)
        json.dumps(records, allow_nan=False)
        self.assertFalse(records[-1].get("remaining_live_pids"))
        return result, records

    def test_required_preflight_sources_fail_before_spawn(self):
        for limit, source in (("max_zram_growth_bytes", "zram"),
                              ("max_swapout_bytes", "vmstat"),
                              ("max_psi_some_us", "memory_pressure"),
                              ("max_psi_full_us", "memory_pressure")):
            with self.subTest(limit=limit):
                args = self.small_args("raise SystemExit('must not launch')")
                setattr(args, limit, 0)
                observer = mock.Mock()
                item = snapshot()
                item[source] = None
                observer.sample.return_value = item
                with mock.patch.object(self.runner.subprocess, "Popen") as popen:
                    result, records = self.supervise(args, observer)
                popen.assert_not_called()
                self.assertEqual(result, 125)
                self.assertEqual(records[-1]["status"], "refused")
                self.assertEqual(records[-1]["reason"], "pressure_source_error")
                self.assertFalse(records[-1]["command_started"])

    def test_optional_collector_exception_cannot_disable_required_pressure(self):
        for method in ("__init__", "sample"):
            with self.subTest(method=method):
                args, records = self.small_args("pass"), []
                args.max_swapout_bytes = 0
                with tempfile.TemporaryFile(mode="w+") as output, mock.patch.object(
                        self.runner.LinuxMemoryTelemetry, method, side_effect=ValueError("fixture failure")), \
                        mock.patch.object(self.runner.subprocess, "Popen") as popen:
                    result = self.runner.supervise(args,
                        lambda event, **values: records.append({"event": event, **values}), output)
                popen.assert_not_called()
                self.assertEqual(result, 125)
                self.assertEqual(records[-1]["reason"], "pressure_source_error")

    def test_valid_zero_limits_complete_with_new_final_endpoint(self):
        args = self.small_args("pass")
        for name in self.runner.PressurePolicy.LIMITS:
            setattr(args, name, 0)
        result, records = self.supervise(args, self.observer())
        self.assertEqual(result, 0)
        self.assertTrue(records[-1]["pressure_guards"]["enabled"])
        final_samples = [item for item in records if item.get("observations_status") == "required_final"]
        self.assertEqual(len(final_samples), 1)
        self.assertEqual(final_samples[0]["phase"], "leader_exit")
        self.assertGreater(final_samples[0]["observations"]["started_monotonic_seconds"],
                           records[0]["observations"]["started_monotonic_seconds"])

    def test_final_only_breach_or_source_failure_overturns_success(self):
        for final_failure in (False, True):
            with self.subTest(final_failure=final_failure):
                after_exit = False

                def emit(event, values):
                    nonlocal after_exit
                    if event == "cleanup" and values.get("phase") == "leader_exit":
                        after_exit = True

                def changes(_count):
                    if after_exit and final_failure:
                        raise ValueError("required final endpoint unavailable")
                    return {"swapout": 11 if after_exit else 10}

                args = self.small_args("pass")
                args.max_swapout_bytes = 0
                result, records = self.supervise(args, self.observer(changes), emit)
                self.assertEqual(result, 125)
                self.assertEqual(records[-1]["command_returncode"], 0)
                self.assertEqual(records[-1]["reason"],
                                 "pressure_source_error" if final_failure else "max_swapout_bytes")
                self.assertTrue(records[-1]["term_sent"])

    def test_pressure_breach_kills_owned_term_resistant_group_only(self):
        with tempfile.TemporaryDirectory(prefix="pmi-pressure-cleanup-") as directory:
            marker = Path(directory) / "worker-ready"
            worker = ("import os,signal,time; from pathlib import Path; "
                      "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                      f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(2)")
            code = ("import signal,subprocess,sys,time; "
                    "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                    f"subprocess.Popen([sys.executable,'-c',{worker!r}]); time.sleep(2)")
            args = self.small_args(code)
            args.max_psi_some_us = 0
            stopping = False
            late_reads = []

            def emit(event, _values):
                nonlocal stopping
                if event in ("stop", "cleanup"):
                    stopping = True

            def changes(_count):
                late_reads.append(stopping)
                return {"some": 1001 if marker.exists() else 1000}

            sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
            try:
                result, records = self.supervise(args, self.observer(changes), emit)
                self.assertEqual(result, 125)
                final = records[-1]
                self.assertEqual(final["reason"], "max_psi_some_us")
                self.assertTrue(final["term_sent"])
                self.assertTrue(final["kill_sent"])
                self.assertFalse(any(late_reads), "pressure reads delayed failure cleanup")
                self.assertIsNone(sentinel.poll(), "unrelated sentinel was signalled")
                pid = int(marker.read_text())
                try:
                    state = self.runner.identity(pid)["state"]
                except FileNotFoundError:
                    state = "X"
                self.assertIn(state, ("Z", "X", "x"))
            finally:
                if sentinel.poll() is None:
                    sentinel.terminate()
                sentinel.wait(timeout=1)

    def test_runtime_required_source_loss_stops_small_child(self):
        args = self.small_args("import time; time.sleep(2)")
        args.max_psi_full_us = 0

        def changes(count):
            if count > 1:
                raise ValueError("pressure source disappeared")
            return {}

        result, records = self.supervise(args, self.observer(changes))
        self.assertEqual(result, 125)
        self.assertEqual(records[-1]["reason"], "pressure_source_error")
        self.assertTrue(records[-1]["term_sent"])
        self.assertIsNotNone(records[-1]["pressure_guards"]["source_error"])

    def test_admission_refuses_marker_but_is_not_a_runtime_floor(self):
        with tempfile.TemporaryDirectory(prefix="pmi-admission-") as directory:
            marker = Path(directory) / "must-not-launch"
            args = self.small_args(f"from pathlib import Path; Path({str(marker)!r}).write_text('started')")
            args.min_available_bytes, args.admit_available_bytes = 100, 200
            memory = {"mem_available_bytes": 150, "system_swap_used_bytes": 0}
            with mock.patch.object(self.runner, "system_memory", return_value=memory), \
                    mock.patch.object(self.runner.subprocess, "Popen") as popen:
                result, records = self.supervise(args, self.observer())
            popen.assert_not_called()
            self.assertFalse(marker.exists())
            self.assertEqual(result, 125)
            self.assertEqual(records[-1]["reason"], "admit_available_bytes")
            # Equal admission succeeds; later availability below admission
            # but above the original runtime floor must not terminate it.
            args.command = [sys.executable, "-c", "pass"]
            reads = 0

            def system():
                nonlocal reads
                reads += 1
                return {"mem_available_bytes": 200 if reads == 1 else 150,
                        "system_swap_used_bytes": 0}

            with mock.patch.object(self.runner, "system_memory", side_effect=system):
                result, records = self.supervise(args, self.observer())
            self.assertEqual(result, 0)
            self.assertEqual(records[-1]["minimum_mem_available_bytes"], 150)

    def test_admission_cannot_weaken_runtime_floor(self):
        args = self.small_args("pass")
        args.min_available_bytes, args.admit_available_bytes = 100, 99
        with tempfile.TemporaryFile(mode="w+") as output, mock.patch.object(
                self.runner.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(self.runner.MonitorError, "at least"):
                self.runner.supervise(args, lambda *_args, **_kwargs: None, output)
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
