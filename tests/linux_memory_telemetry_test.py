#!/usr/bin/env python3
"""Offline Linux telemetry checks against temporary proc/sys/cgroup fixtures."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("linux_memory_telemetry", ROOT / "tools/linux_memory_telemetry.py")
telemetry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = telemetry
spec.loader.exec_module(telemetry)


class Clock:
    def __init__(self): self.value = 100.0
    def __call__(self): return self.value
    def advance(self, seconds=2.0): self.value += seconds


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def escaped_mount_path(path):
    return str(path).replace("\\", "\\134").replace(" ", "\\040").replace("\t", "\\011").replace("\n", "\\012")


def pressure(some=123456, full=1234):
    return (f"some avg10=0.25 avg60=1.50 avg300=2.75 total={some}\n"
            f"full avg10=0.01 avg60=0.02 avg300=0.03 total={full}\n")


class LinuxMemoryTelemetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pmi-memory-telemetry-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.proc = self.root / "proc"
        self.sys = self.root / "sys"
        self.mount = self.root / "cgroup mount"
        self.membership = "/tenant/team one/job"
        self.mount_root = "/tenant/team one"
        self.cgroup = self.mount / "job"
        self.clock = Clock()
        self.vmstat(3, 5)
        write(self.proc / "pressure/memory", pressure())
        write(self.proc / "sys/vm/swappiness", "60\n")
        write(self.proc / "self/cgroup", "0::" + self.membership + "\n")
        write(self.proc / "self/mountinfo",
              "29 23 0:26 " + escaped_mount_path(self.mount_root) + " " + escaped_mount_path(self.mount)
              + " rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n")
        # Deliberately unlike cgroup aggregate usage. The collector must not
        # substitute this process's RSS for the cgroup's shared accounting.
        write(self.proc / "self/status", "Name:\ttest\nPid:\t123\nVmRSS:\t1 kB\n")
        self.zram()
        self.cgroup_files(self.mount, current=262144, limit="1048576")
        self.cgroup_files(self.cgroup, current=81920, limit="max")

    def vmstat(self, incoming, outgoing):
        write(self.proc / "vmstat", f"nr_free_pages 999\npswpin {incoming}\npswpout {outgoing}\npgfault 900\n")

    def zram(self, mm="16384 4096 8192 32768 12288 7 9 4 6", bd="2 3 5"):
        write(self.sys / "block/zram0/mm_stat", mm + "\n")
        write(self.sys / "block/zram0/bd_stat", bd + "\n")
        write(self.sys / "block/zram0/disksize", "65536\n")
        write(self.sys / "block/zram0/comp_algorithm", "lzo [zstd] lz4\n")
        write(self.sys / "block/zram0/backing_dev", "none\n")

    def cgroup_files(self, path, *, current, limit):
        write(path / "memory.current", str(current) + "\n")
        write(path / "memory.swap.current", "24576\n")
        write(path / "memory.max", limit + "\n")
        write(path / "memory.high", "98304\n")
        write(path / "memory.low", "4096\n")
        write(path / "memory.min", "0\n")
        write(path / "memory.swap.max", "40960\n")
        write(path / "memory.swap.high", "32768\n")
        write(path / "memory.stat", "anon 16384\nfile 8192\nkernel 4096\nswapcached 2048\npgfault 100\npgmajfault 4\nworkingset_refault_anon 7\n")
        write(path / "memory.events", "low 1\nhigh 2\nmax 3\noom 4\noom_kill 5\noom_group_kill 6\n")
        write(path / "memory.events.local", "low 0\nhigh 1\nmax 2\noom 3\noom_kill 4\n")
        write(path / "memory.pressure", pressure(9000, 700))
        write(path / "cgroup.procs", "123\n999\n")

    def collector(self):
        return telemetry.LinuxMemoryTelemetry(proc_root=self.proc, sys_root=self.sys,
                                              page_size=8192, clock=self.clock)

    def metric(self, metric, *, value, baseline=0, previous=None, reset=False):
        self.assertEqual(metric["value"], value)
        self.assertEqual(metric["delta_from_baseline"], baseline)
        self.assertEqual(metric["delta_from_previous"], previous)
        self.assertEqual(metric["counter_reset"], reset)

    def test_units_first_sample_and_shared_cgroup_scope(self):
        sample = self.collector().sample()
        self.assertEqual(sample["schema_version"], 1)
        self.assertEqual(sample["page_size_bytes"], 8192)
        self.assertEqual(sample["started_monotonic_seconds"], 100.0)
        self.assertIsNone(sample["interval_seconds"])
        self.assertEqual(sample["duration_seconds"], 0)
        self.metric(sample["vmstat"]["pswpin_pages"], value=3)
        self.metric(sample["vmstat"]["pswpin_bytes"], value=24576)
        self.metric(sample["vmstat"]["pswpout_bytes"], value=40960)
        psi = sample["memory_pressure"]
        self.assertEqual(psi["some"]["avg10_percent"], 0.25)
        self.assertEqual(psi["some"]["avg60_percent"], 1.5)
        self.assertEqual(psi["some"]["avg300_percent"], 2.75)
        self.metric(psi["some"]["total_stall_us"], value=123456)
        self.metric(psi["full"]["total_stall_us"], value=1234)
        zram = sample["zram"][0]
        self.assertEqual(zram["name"], "zram0")
        self.assertEqual(zram["disksize_bytes"], 65536)
        self.assertEqual(sample["swappiness"], 60)
        self.metric(zram["mm_stat"]["orig_data_size_bytes"], value=16384)
        self.metric(zram["mm_stat"]["compr_data_size_bytes"], value=4096)
        self.metric(zram["mm_stat"]["mem_used_total_bytes"], value=8192)
        self.assertEqual(zram["compression_ratio_uncompressed_to_compressed"], 4)
        # bd_stat is specified in 4 KiB blocks, independently of native pages.
        self.metric(zram["bd_stat"]["backing_count_bytes"], value=8192)
        self.metric(zram["bd_stat"]["backing_read_bytes"], value=12288)
        self.metric(zram["bd_stat"]["backing_write_bytes"], value=20480)
        cg = sample["cgroup_v2"]
        self.assertEqual(cg["scope"], "supervisor_cgroup_context")
        self.assertEqual(cg["attribution"], "shared_or_unknown")
        self.metric(cg["usage_bytes"]["memory.current"], value=81920)
        self.metric(cg["usage_bytes"]["memory.swap.current"], value=24576)
        self.assertEqual(cg["limits"]["memory.max"], "max")
        self.metric(cg["stat_bytes"]["anon"], value=16384)
        self.metric(cg["stat_counts"]["pgfault"], value=100)
        self.metric(cg["events"]["oom_kill"], value=5)
        self.metric(cg["events_local"]["oom_kill"], value=4)
        self.metric(cg["memory_pressure"]["some"]["total_stall_us"], value=9000)

    def test_interval_deltas_and_signed_gauges(self):
        collector = self.collector()
        collector.sample()
        self.clock.advance()
        self.vmstat(4, 7)
        self.zram(mm="8192 2048 4096 32768 12288 2 11 3 8", bd="1 5 6")
        write(self.proc / "pressure/memory", pressure(123956, 1334))
        write(self.cgroup / "memory.current", "65536\n")
        sample = collector.sample()
        self.assertEqual(sample["interval_seconds"], 2)
        self.assertEqual(sample["started_monotonic_seconds"], 102)
        self.assertEqual(sample["duration_seconds"], 0)  # Collection cost, not session age.
        self.metric(sample["vmstat"]["pswpin_bytes"], value=32768, baseline=8192, previous=8192)
        self.metric(sample["vmstat"]["pswpout_pages"], value=7, baseline=2, previous=2)
        self.metric(sample["memory_pressure"]["some"]["total_stall_us"], value=123956, baseline=500, previous=500)
        zram = sample["zram"][0]
        self.metric(zram["mm_stat"]["orig_data_size_bytes"], value=8192, baseline=-8192, previous=-8192)
        self.metric(zram["mm_stat"]["same_pages"], value=2, baseline=-5, previous=-5)
        self.metric(zram["mm_stat"]["pages_compacted"], value=11, baseline=2, previous=2)
        self.metric(zram["bd_stat"]["backing_count_bytes"], value=4096, baseline=-4096, previous=-4096)
        self.metric(zram["bd_stat"]["backing_read_bytes"], value=20480, baseline=8192, previous=8192)
        self.metric(sample["cgroup_v2"]["usage_bytes"]["memory.current"], value=65536, baseline=-16384, previous=-16384)

    def test_counter_reset_does_not_regain_a_cumulative_baseline(self):
        self.vmstat(100, 200)
        collector = self.collector()
        collector.sample()
        self.clock.advance()
        self.vmstat(110, 203)
        self.metric(collector.sample()["vmstat"]["pswpin_pages"], value=110, baseline=10, previous=10)
        self.clock.advance()
        self.vmstat(3, 204)
        self.metric(collector.sample()["vmstat"]["pswpin_pages"], value=3, baseline=None, previous=None, reset=True)
        self.clock.advance()
        self.vmstat(115, 205)
        sample = collector.sample()
        self.assertIsNone(sample["vmstat"]["pswpin_pages"]["delta_from_baseline"])
        self.assertEqual(sample["vmstat"]["pswpin_pages"]["delta_from_previous"], 112)
        self.assertIsNone(sample["vmstat"]["pswpin_bytes"]["delta_from_baseline"])
        self.assertEqual(sample["vmstat"]["pswpin_bytes"]["delta_from_previous"], 112 * 8192)
        self.assertEqual(sample["vmstat"]["pswpout_pages"]["delta_from_baseline"], 5)

    def test_late_source_never_gets_a_preflight_baseline(self):
        (self.proc / "vmstat").unlink()
        collector = self.collector()
        first = collector.sample()
        self.assertIsNone(first["vmstat"])
        self.assertTrue(any("vmstat" in entry["source"] for entry in first["unavailable"]))
        self.clock.advance()
        self.vmstat(100, 200)
        self.metric(collector.sample()["vmstat"]["pswpin_pages"], value=100, baseline=None)
        self.clock.advance()
        self.vmstat(105, 202)
        self.metric(collector.sample()["vmstat"]["pswpin_pages"], value=105, baseline=None, previous=5)

    def test_missing_sample_breaks_interval_but_preserves_valid_baseline(self):
        collector = self.collector()
        collector.sample()
        self.clock.advance()
        (self.proc / "vmstat").unlink()
        self.assertIsNone(collector.sample()["vmstat"])
        self.clock.advance()
        self.vmstat(5, 7)
        self.metric(collector.sample()["vmstat"]["pswpin_pages"], value=5, baseline=2)
        self.clock.advance()
        self.vmstat(6, 8)
        self.metric(collector.sample()["vmstat"]["pswpin_pages"], value=6, baseline=3, previous=1)

    def test_counter_reset_across_missing_sample_invalidates_baseline(self):
        self.vmstat(100, 200)
        collector = self.collector()
        collector.sample()
        self.clock.advance()
        self.vmstat(150, 201)
        collector.sample()
        self.clock.advance()
        (self.proc / "vmstat").unlink()
        self.assertIsNone(collector.sample()["vmstat"])
        self.clock.advance()
        self.vmstat(120, 202)
        sample = collector.sample()
        self.metric(sample["vmstat"]["pswpin_pages"], value=120, baseline=None, reset=True)
        self.metric(sample["vmstat"]["pswpin_bytes"], value=120 * 8192, baseline=None, reset=True)
        self.clock.advance()
        self.vmstat(160, 203)
        self.metric(collector.sample()["vmstat"]["pswpin_pages"], value=160, baseline=None, previous=40)

    def test_sampling_cost_is_distinct_from_start_to_start_interval(self):
        ticks = iter((100.0, 100.25, 102.0, 102.5))
        collector = telemetry.LinuxMemoryTelemetry(proc_root=self.proc, sys_root=self.sys,
                                                   page_size=8192, clock=lambda: next(ticks))
        first, second = collector.sample(), collector.sample()
        self.assertEqual(first["started_monotonic_seconds"], 100)
        self.assertEqual(first["duration_seconds"], 0.25)
        self.assertIsNone(first["interval_seconds"])
        self.assertEqual(second["started_monotonic_seconds"], 102)
        self.assertEqual(second["duration_seconds"], 0.5)
        self.assertEqual(second["interval_seconds"], 2)

    def test_cgroup_mount_root_escaped_paths_and_ancestors(self):
        sample = self.collector().sample()
        cg = sample["cgroup_v2"]
        self.assertEqual(cg["membership_path"], self.membership)
        self.assertEqual(cg["mount_point"], str(self.mount))
        self.assertEqual(cg["mount_root"], self.mount_root)
        self.assertTrue(cg["ancestors"], "mounted parent cgroup was not reported")
        self.assertTrue(any(ancestor["limits"]["memory.max"] == 1048576 for ancestor in cg["ancestors"]))
        self.assertEqual([ancestor["path"] for ancestor in cg["ancestors"]], [str(self.mount)])

    def test_cgroup_unmapped_or_invalid_membership_is_explicit(self):
        for member in ("/tenant/team one-other/job", "relative/job", "/tenant/team one/../job"):
            with self.subTest(member=member):
                write(self.proc / "self/cgroup", "0::" + member + "\n")
                sample = self.collector().sample()
                self.assertIsNone(sample["cgroup_v2"])
                self.assertTrue(any(entry["source"] in ("cgroup_v2", str(self.proc / "self/cgroup"))
                                    for entry in sample["unavailable"]))
                self.assertEqual(sample["vmstat"]["pswpin_pages"]["value"], 3)

    def test_cgroup_ambiguous_mounts_are_rejected(self):
        original = (self.proc / "self/mountinfo").read_text()
        other = self.root / "other mount"
        write(self.proc / "self/mountinfo", original +
              "30 23 0:26 " + escaped_mount_path(self.mount_root) + " " + escaped_mount_path(other)
              + " rw - cgroup2 cgroup rw\n")
        sample = self.collector().sample()
        self.assertIsNone(sample["cgroup_v2"])
        self.assertTrue(any(entry["source"] == "cgroup_v2" and "ambiguous" in entry.get("detail", "")
                            for entry in sample["unavailable"]))

    def test_cgroup_most_specific_mount_wins(self):
        original = (self.proc / "self/mountinfo").read_text()
        broad = self.root / "broad mount"
        write(self.proc / "self/mountinfo", "30 23 0:26 / " + escaped_mount_path(broad)
              + " rw - cgroup2 cgroup rw\n" + original)
        cg = self.collector().sample()["cgroup_v2"]
        self.assertEqual(cg["mount_root"], self.mount_root)
        self.assertEqual(cg["mount_point"], str(self.mount))
        self.metric(cg["usage_bytes"]["memory.current"], value=81920)

    def test_cgroup_relocation_never_reuses_another_groups_baseline(self):
        collector = self.collector()
        collector.sample()
        self.clock.advance()
        self.cgroup_files(self.mount / "other", current=163840, limit="max")
        write(self.proc / "self/cgroup", "0::/tenant/team one/other\n")
        sample = collector.sample()
        self.assertEqual(sample["cgroup_v2"]["membership_path"], "/tenant/team one/other")
        self.metric(sample["cgroup_v2"]["usage_bytes"]["memory.current"], value=163840, baseline=None)
        self.clock.advance()
        write(self.mount / "other/memory.current", "163841\n")
        sample = collector.sample()
        self.metric(sample["cgroup_v2"]["usage_bytes"]["memory.current"], value=163841, baseline=None, previous=1)

    def test_replaced_cgroup_and_zram_inodes_start_without_a_baseline(self):
        collector = self.collector()
        first = collector.sample()
        # Retaining the old directories guarantees that their inode numbers
        # cannot be immediately reused for these replacement sources.
        self.cgroup.rename(self.mount / "old-job")
        (self.sys / "block/zram0").rename(self.sys / "block/old-zram")
        self.cgroup_files(self.cgroup, current=163840, limit="max")
        self.zram(mm="32768 8192 16384 32768 16384 14 18 8 12")
        self.clock.advance()
        sample = collector.sample()
        self.assertNotEqual(sample["cgroup_v2"]["identity"], first["cgroup_v2"]["identity"])
        self.assertNotEqual(sample["zram"][0]["identity"], first["zram"][0]["identity"])
        self.metric(sample["cgroup_v2"]["usage_bytes"]["memory.current"], value=163840, baseline=None)
        self.metric(sample["zram"][0]["mm_stat"]["pages_compacted"], value=18, baseline=None)
        self.clock.advance()
        write(self.cgroup / "memory.current", "163841\n")
        self.zram(mm="32768 8192 16384 32768 16384 14 19 8 12")
        sample = collector.sample()
        self.metric(sample["cgroup_v2"]["usage_bytes"]["memory.current"], value=163841, baseline=None, previous=1)
        self.metric(sample["zram"][0]["mm_stat"]["pages_compacted"], value=19, baseline=None, previous=1)

    def test_older_mm_stat_preserves_missing_optional_fields(self):
        self.zram(mm="16384 4096 8192 32768 12288 7 9")
        sample = self.collector().sample()
        mm = sample["zram"][0]["mm_stat"]
        self.metric(mm["pages_compacted"], value=9)
        self.assertIsNone(mm["huge_pages"])
        self.assertIsNone(mm["huge_pages_since"])

    def test_zram_and_cgroup_reset_counters_not_signed_gauges(self):
        collector = self.collector()
        collector.sample()
        self.clock.advance()
        self.zram(mm="8192 2048 4096 32768 12288 2 1 2 1", bd="1 1 1")
        write(self.cgroup / "memory.events", "low 1\nhigh 0\nmax 3\noom 4\noom_kill 5\n")
        sample = collector.sample()
        zram = sample["zram"][0]
        self.metric(zram["mm_stat"]["pages_compacted"], value=1, baseline=None, previous=None, reset=True)
        self.metric(zram["mm_stat"]["huge_pages_since"], value=1, baseline=None, previous=None, reset=True)
        self.metric(zram["mm_stat"]["huge_pages"], value=2, baseline=-2, previous=-2)
        self.metric(zram["bd_stat"]["backing_read_bytes"], value=4096, baseline=None, previous=None, reset=True)
        self.metric(sample["cgroup_v2"]["events"]["high"], value=0, baseline=None, previous=None, reset=True)
        self.clock.advance()
        self.zram(mm="8192 2048 4096 32768 12288 2 20 2 20", bd="1 20 20")
        sample = collector.sample()
        self.assertIsNone(sample["zram"][0]["mm_stat"]["pages_compacted"]["delta_from_baseline"])
        self.assertEqual(sample["zram"][0]["mm_stat"]["pages_compacted"]["delta_from_previous"], 19)

    def test_missing_optional_sources_and_empty_zram_are_explicit(self):
        (self.proc / "pressure/memory").unlink()
        self.zram(mm="0 0 0 0 0 0 0 0 0")
        (self.sys / "block/zram0/bd_stat").unlink()
        sample = self.collector().sample()
        self.assertIsNone(sample["memory_pressure"])
        self.assertIsNone(sample["zram"][0]["bd_stat"])
        self.assertIsNone(sample["zram"][0]["compression_ratio_uncompressed_to_compressed"])
        unavailable = sample["unavailable"]
        self.assertTrue(any("pressure" in entry["source"] for entry in unavailable))
        self.assertTrue(any("bd_stat" in entry["source"] for entry in unavailable))
        self.assertTrue(all(entry["reason"] for entry in unavailable))

    def test_malformed_optional_sources_do_not_abort_other_measurements(self):
        write(self.proc / "pressure/memory", "some avg10=invalid avg60=1 avg300=1 total=10\n")
        write(self.sys / "block/zram0/mm_stat", "not numbers\n")
        sample = self.collector().sample()
        self.assertIsNone(sample["memory_pressure"])
        self.assertIsNone(sample["zram"][0]["mm_stat"])
        self.assertIsNone(sample["zram"][0]["compression_ratio_uncompressed_to_compressed"])
        self.assertEqual(sample["vmstat"]["pswpin_pages"]["value"], 3)
        self.assertEqual(sample["cgroup_v2"]["usage_bytes"]["memory.current"]["value"], 81920)
        self.assertTrue(any("mm_stat" in entry["source"] for entry in sample["unavailable"]))

    def test_psi_rejects_nonfinite_out_of_range_duplicate_or_incomplete_data(self):
        valid = pressure()
        invalid = [valid.replace("avg10=0.25", "avg10=" + value, 1)
                   for value in ("nan", "inf", "-1", "100.01")]
        invalid += [valid.replace("total=123456", "total=-1", 1),
                    valid.replace("avg10=0.25", "avg10=0.25 avg10=0.50", 1),
                    valid + valid.splitlines()[0] + "\n", valid.splitlines()[0] + "\n"]
        for text in invalid:
            with self.subTest(text=text):
                write(self.proc / "pressure/memory", text)
                sample = self.collector().sample()
                self.assertIsNone(sample["memory_pressure"])
                self.assertTrue(any(entry["source"] == str(self.proc / "pressure/memory")
                                    for entry in sample["unavailable"]))
                self.assertEqual(sample["vmstat"]["pswpin_pages"]["value"], 3)

    def test_psi_large_integer_totals_keep_single_microsecond_precision(self):
        total = 2 ** 63 + 17
        write(self.proc / "pressure/memory", pressure(total, total + 100))
        collector = self.collector()
        first = collector.sample()
        self.metric(first["memory_pressure"]["some"]["total_stall_us"], value=total)
        self.clock.advance()
        write(self.proc / "pressure/memory", pressure(total + 1, total + 102))
        sample = collector.sample()
        self.metric(sample["memory_pressure"]["some"]["total_stall_us"], value=total + 1, baseline=1, previous=1)
        self.metric(sample["memory_pressure"]["full"]["total_stall_us"], value=total + 102, baseline=2, previous=2)

    def test_permission_error_is_reported_without_zero_substitution(self):
        denied, original_read = self.cgroup / "memory.current", Path.read_text

        def read(path, *args, **kwargs):
            if path == denied:
                raise PermissionError("fixture permission denied")
            return original_read(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", read):
            sample = self.collector().sample()
        self.assertIsNone(sample["cgroup_v2"]["usage_bytes"].get("memory.current"))
        failures = [entry for entry in sample["unavailable"] if "memory.current" in entry["source"]]
        self.assertTrue(failures)
        self.assertTrue(any("permission" in entry["reason"].lower() for entry in failures))
        self.assertEqual(sample["vmstat"]["pswpout_pages"]["value"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
