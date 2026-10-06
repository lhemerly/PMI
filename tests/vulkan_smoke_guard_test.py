#!/usr/bin/env python3
"""Tiny CPU-only fake-counter/stub-child regressions; never touches a real GPU."""

import contextlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_vulkan_smoke as guard


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux wrapper required")
class VulkanSmokeGuardTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pmi-vram-stub-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.device = self.directory / "device"
        self.device.mkdir()
        self.counter = self.device / guard.COUNTER_NAME
        self.counter.write_text("0000000100\n")
        self.events = []

    def args(self, code="pass", extra=()):
        return guard.arguments(["--sysfs-dir", str(self.device), "--log",
                                str(self.directory / "guard.jsonl"),
                                "--interval-seconds", "0.002", "--wall-seconds", "0.5",
                                "--term-grace-seconds", "0.01", "--kill-wait-seconds", "0.05",
                                *extra, "--", sys.executable, "-c", code])

    def emit(self, event, **values):
        self.events.append({"event": event, **values})

    def invoke(self, code="pass", expected=0, extra=()):
        result = guard.supervise(self.args(code, extra), self.emit)
        self.assertEqual(result, expected, self.events)
        final = self.events[-1]
        self.assertEqual(final["event"], "final")
        self.assertEqual(final["runner_returncode"], expected)
        self.assertFalse(final["cleanup_incomplete"])
        return final

    def write_from_child(self, value, before=""):
        # Equal-length pwrite avoids an artificial truncate/read race. The
        # integer represents simulated usage; no memory allocation is made.
        return ("import os,signal,time; " + before +
                f"fd=os.open({str(self.counter)!r},os.O_WRONLY); "
                f"os.pwrite(fd,{value!r},0); os.close(fd); time.sleep(0.5)")

    def test_success_logs_peak_and_preserves_group_and_session(self):
        marker = self.directory / "identity.json"
        code = ("import json,os; "
                f"open({str(marker)!r},'w').write(json.dumps([os.getpgrp(),os.getsid(0)]))")
        final = self.invoke(code)
        self.assertEqual(json.loads(marker.read_text()), [os.getpgrp(), os.getsid(0)])
        self.assertEqual(final["sampled_peak_global_vram_bytes"], 100)
        self.assertEqual(final["command_returncode"], 0)
        self.assertTrue(any(event["event"] == "preflight" for event in self.events))

    def test_child_nonzero_is_forwarded(self):
        self.assertEqual(self.invoke("raise SystemExit(7)", 7)["command_returncode"], 7)

    def test_preflight_cap_refuses_without_launch(self):
        self.counter.write_text(str(guard.MAX_VRAM_BYTES + 1))
        with mock.patch.object(guard.subprocess, "Popen") as spawn:
            final = self.invoke(expected=125)
        spawn.assert_not_called()
        self.assertFalse(final["command_started"])
        self.assertEqual(final["reason"], "max_vram_bytes")
        self.assertTrue(any(event["event"] == "stop" for event in self.events))

    def test_missing_counter_refuses_without_launch(self):
        self.counter.unlink()
        with mock.patch.object(guard.subprocess, "Popen") as spawn:
            final = self.invoke(expected=125)
        spawn.assert_not_called()
        self.assertEqual(final["reason"], "source_error")

    def test_invalid_counter_refuses_without_launch(self):
        for raw in (b"", b"-1\n", b"nan\n", b"1 2\n", b"9" * 129, b"18446744073709551616"):
            with self.subTest(raw=raw):
                self.counter.write_bytes(raw)
                with mock.patch.object(guard.subprocess, "Popen") as spawn:
                    self.invoke(expected=125)
                spawn.assert_not_called()

    def test_unreadable_source_refuses_without_launch(self):
        with mock.patch.object(guard.os, "open", side_effect=PermissionError("stub denial")):
            with mock.patch.object(guard.subprocess, "Popen") as spawn:
                final = self.invoke(expected=125)
        spawn.assert_not_called()
        self.assertEqual(final["reason"], "source_error")

    def test_device_replacement_fails_closed(self):
        source = guard.VramCounter(self.device)
        source.sample()
        self.device.rename(self.directory / "old-device")
        self.device.mkdir()
        self.counter.write_text("100\n")
        with self.assertRaisesRegex(guard.MonitorError, "identity changed"):
            source.sample()

    def test_counter_replacement_fails_closed(self):
        source = guard.VramCounter(self.device)
        source.sample()
        replacement = self.device / "replacement"
        replacement.write_text("100\n")
        replacement.replace(self.counter)
        with self.assertRaisesRegex(guard.MonitorError, "identity changed"):
            source.sample()

    def test_symlink_and_nonregular_counter_refused(self):
        self.counter.unlink()
        self.counter.symlink_to(self.directory / "other")
        with self.assertRaises(guard.MonitorError):
            guard.VramCounter(self.device).sample()
        self.counter.unlink()
        os.mkfifo(self.counter)
        with self.assertRaisesRegex(guard.MonitorError, "regular file"):
            guard.VramCounter(self.device).sample()

    def test_runtime_over_cap_stops_child_and_logs_measured_peak(self):
        final = self.invoke(self.write_from_child(b"4294967297\n"), 125)
        self.assertEqual(final["reason"], "max_vram_bytes")
        self.assertEqual(final["sampled_peak_global_vram_bytes"], guard.MAX_VRAM_BYTES + 1)
        self.assertTrue(final["term_sent"])
        self.assertEqual(final["command_returncode"], -signal.SIGTERM)

    def test_runtime_bad_counter_stops_child(self):
        final = self.invoke(self.write_from_child(b"notanumber\n"), 125)
        self.assertEqual(final["reason"], "source_error")
        self.assertTrue(final["term_sent"])

    def test_term_resistant_child_is_killed(self):
        final = self.invoke(self.write_from_child(b"4294967297\n",
                           "signal.signal(signal.SIGTERM,signal.SIG_IGN); "), 125)
        self.assertTrue(final["kill_sent"])
        self.assertEqual(final["command_returncode"], -signal.SIGKILL)

    def test_guard_term_interrupt_cleans_direct_child(self):
        final = self.invoke("import os,signal,time; os.kill(os.getppid(),signal.SIGTERM); time.sleep(0.5)",
                            128 + signal.SIGTERM)
        self.assertEqual(final["status"], "interrupted")
        self.assertTrue(final["term_sent"])

    def test_cli_cannot_raise_caps_or_use_invalid_intervals(self):
        for extra in (("--max-vram-bytes", "5GiB"), ("--max-vram-bytes", "0"),
                      ("--interval-seconds", "0.251"), ("--interval-seconds", "nan"),
                      ("--interval-seconds", "0"), ("--wall-seconds", "601")):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.args(extra=extra)
                self.assertEqual(error.exception.code, 2)
        self.assertEqual(self.args(extra=("--max-vram-bytes", "4GiB")).max_vram_bytes,
                         guard.MAX_VRAM_BYTES)

    def test_exclusive_log_and_real_jsonl_output(self):
        args = self.args()
        argv = ["--sysfs-dir", str(self.device), "--log", str(args.log),
                "--interval-seconds", "0.002", "--", sys.executable, "-c", "pass"]
        self.assertEqual(guard.main(argv), 0)
        original = args.log.read_bytes()
        records = [json.loads(line) for line in original.splitlines()]
        self.assertEqual(records[-1]["sampled_peak_global_vram_bytes"], 100)
        with mock.patch.object(guard.subprocess, "Popen") as spawn:
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(guard.main(argv), 125)
        spawn.assert_not_called()
        self.assertEqual(args.log.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
