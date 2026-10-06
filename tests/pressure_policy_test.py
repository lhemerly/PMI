#!/usr/bin/env python3
"""Pure pressure-policy checks; synthetic counters, no host pressure or workloads."""

import argparse
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from pressure_policy import PressurePolicy, PressureSourceError


def snapshot(*, start=10, zram=(100, 200), swapout=10, some=1000, full=100,
             baseline=None, previous=None):
    values = {"swapout": swapout, "some": some, "full": full,
              **{f"zram{index}": value for index, value in enumerate(zram)}}
    base = values if baseline is None else baseline["fixture_values"]
    old = None if previous is None else previous["fixture_values"]

    def metric(key, multiplier=1):
        return {"value": values[key] * multiplier,
                "delta_from_baseline": (values[key] - base[key]) * multiplier,
                "delta_from_previous": None if old is None else (values[key] - old[key]) * multiplier,
                "counter_reset": False}

    return {"schema_version": 1, "started_monotonic_seconds": start,
            "page_size_bytes": 4096, "fixture_values": values,
            "vmstat": {"pswpout_pages": metric("swapout"), "pswpout_bytes": metric("swapout", 4096)},
            "memory_pressure": {name: {"total_stall_us": metric(name)} for name in ("some", "full")},
            "zram": [{"name": f"zram{index}", "identity": f"/sys/block/zram{index}:1:{index + 20}",
                      "disksize_bytes": 65536, "comp_algorithm": "lzo [zstd]", "backing_dev": "none",
                      "mm_stat": {"mem_used_total_bytes": metric(f"zram{index}")}}
                     for index in range(len(zram))],
            "swap_devices": [{"path": f"/dev/zram{index}", "type": "partition", "size_bytes": 61440,
                              "used_bytes": 4096, "priority": 100} for index in range(len(zram))],
            "unavailable": []}


class PressurePolicyTest(unittest.TestCase):
    def policy(self, **limits):
        return PressurePolicy(argparse.Namespace(**limits))

    def test_disabled_never_requires_optional_sources(self):
        policy = self.policy()
        self.assertIsNone(policy.observe(None))
        self.assertIsNone(policy.observe({"collection_failed": True}))
        self.assertFalse(policy.summary()["enabled"])

    def test_fresh_nonzero_baseline_and_exact_thresholds(self):
        base = snapshot()
        policy = self.policy(max_zram_growth_bytes=50, max_swapout_bytes=8192,
                             max_psi_some_us=20, max_psi_full_us=5)
        self.assertEqual(policy.observe(base)["violations"], [])
        equal = snapshot(start=11, zram=(125, 225), swapout=12, some=1020, full=105,
                         baseline=base, previous=base)
        self.assertEqual(policy.observe(equal)["violations"], [])
        over = snapshot(start=12, zram=(126, 225), swapout=13, some=1021, full=106,
                        baseline=base, previous=equal)
        self.assertEqual(set(policy.observe(over)["violations"]), set(policy.LIMITS))
        self.assertEqual(policy.summary()["sampled_peak_growth"]["swapout_bytes"], 12288)

    def test_zero_is_enabled_and_needs_no_rate_estimation(self):
        policy = self.policy(max_psi_some_us=0)
        base = snapshot()
        policy.observe(base)
        changed = snapshot(start=1_000_010, some=1001, baseline=base, previous=base)
        self.assertEqual(policy.observe(changed)["violations"], ["max_psi_some_us"])

    def test_aggregate_zram_decrease_and_recovery_are_valid(self):
        policy = self.policy(max_zram_growth_bytes=10)
        base = snapshot()
        policy.observe(base)
        # Positive change on one device cannot hide the other's legitimate
        # release: enforce total physical growth, not per-device positive sums.
        lower = snapshot(start=11, zram=(150, 100), baseline=base, previous=base)
        self.assertEqual(policy.observe(lower)["growth"]["zram_growth_bytes"], -50)
        equal = snapshot(start=12, zram=(210, 100), baseline=base, previous=lower)
        self.assertEqual(policy.observe(equal)["violations"], [])
        over = snapshot(start=13, zram=(211, 100), baseline=base, previous=equal)
        self.assertEqual(policy.observe(over)["violations"], ["max_zram_growth_bytes"])
        self.assertEqual(policy.summary()["baseline"], {"zram:zram0": 100, "zram:zram1": 200})

    def test_missing_malformed_and_reset_sources_poison_continuity(self):
        def changed(path, value):
            def mutate(item):
                row = item
                for key in path[:-1]:
                    row = row[key]
                row[path[-1]] = value
            return mutate
        mutations = [
            changed(("vmstat",), None),
            changed(("vmstat", "pswpout_bytes", "value"), "40960"),
            changed(("vmstat", "pswpout_bytes", "value"), True),
            changed(("vmstat", "pswpout_bytes", "value"), -1),
            changed(("vmstat", "pswpout_bytes", "counter_reset"), True),
            changed(("vmstat", "pswpout_bytes", "delta_from_baseline"), None),
            changed(("vmstat", "pswpout_bytes", "delta_from_previous"), None),
            changed(("vmstat", "pswpout_pages", "value"), 999),
            changed(("memory_pressure", "some"), None),
            changed(("memory_pressure", "full", "total_stall_us", "value"), 1.5),
            changed(("zram", 0, "mm_stat"), None),
            changed(("zram", 0, "mm_stat", "mem_used_total_bytes", "value"), None),
            changed(("swap_devices",), None),
            changed(("collection_failed",), True),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                policy = self.policy(**dict.fromkeys(PressurePolicy.LIMITS, 10**9))
                base = snapshot()
                policy.observe(base)
                bad = snapshot(start=11, baseline=base, previous=base)
                mutate(bad)
                with self.assertRaises(PressureSourceError):
                    policy.observe(bad)
                restored = snapshot(start=12, zram=(1000, 2000), swapout=100,
                                    some=10_000, full=1000, baseline=base, previous=base)
                with self.assertRaises(PressureSourceError):
                    policy.observe(restored)
                self.assertIsNotNone(policy.summary()["source_error"])

    def test_regression_above_baseline_is_not_silently_accepted(self):
        for limit, key in (("max_swapout_bytes", "swapout"),
                           ("max_psi_some_us", "some"), ("max_psi_full_us", "full")):
            with self.subTest(limit=limit):
                policy = self.policy(**{limit: 10**9})
                base = snapshot()
                policy.observe(base)
                high = snapshot(start=11, baseline=base, previous=base, **{key: 2000})
                policy.observe(high)
                low = snapshot(start=12, baseline=base, previous=high, **{key: 1500})
                # Even a malformed collector claiming a positive interval
                # delta cannot conceal regression of the absolute counter.
                if key == "swapout":
                    for metric in low["vmstat"].values():
                        metric["delta_from_previous"] = 0
                else:
                    low["memory_pressure"][key]["total_stall_us"]["delta_from_previous"] = 0
                with self.assertRaisesRegex(PressureSourceError, "regressed"):
                    policy.observe(low)

    def test_required_baseline_is_never_rebased(self):
        for kind in ("stale_preflight", "later_rebase"):
            with self.subTest(kind=kind):
                policy = self.policy(max_swapout_bytes=1_000_000)
                base = snapshot()
                if kind == "later_rebase":
                    policy.observe(base)
                    item = snapshot(start=11, swapout=20, baseline=base, previous=base)
                    item["vmstat"]["pswpout_pages"]["delta_from_baseline"] = 0
                    item["vmstat"]["pswpout_bytes"]["delta_from_baseline"] = 0
                else:
                    item = snapshot(start=11, swapout=20, baseline=base, previous=base)
                with self.assertRaises(PressureSourceError):
                    policy.observe(item)

    def test_inventory_order_and_swap_used_are_not_identity_changes(self):
        policy = self.policy(max_zram_growth_bytes=0)
        base = snapshot()
        policy.observe(base)
        next_sample = snapshot(start=11, baseline=base, previous=base)
        next_sample["zram"].reverse()
        next_sample["swap_devices"].reverse()
        next_sample["swap_devices"][0]["used_bytes"] += 4096
        self.assertEqual(policy.observe(next_sample)["violations"], [])

    def test_zram_and_active_swap_identity_changes_fail_closed(self):
        mutations = [
            lambda s: s["zram"].pop(),
            lambda s: s["zram"].append(copy.deepcopy(s["zram"][0])),
            lambda s: s["zram"][0].update(identity="new device"),
            lambda s: s["zram"][0].update(disksize_bytes=131072),
            lambda s: s["zram"][0].update(comp_algorithm="[lzo] zstd"),
            lambda s: s["zram"][0].update(backing_dev="8:1"),
            lambda s: s["swap_devices"].pop(),
            lambda s: s["swap_devices"].append(copy.deepcopy(s["swap_devices"][0])),
            lambda s: s["swap_devices"][0].update(priority=99),
            lambda s: s["swap_devices"][0].update(size_bytes=32768),
            lambda s: s["swap_devices"][0].update(path="/dev/zram3"),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                policy = self.policy(max_zram_growth_bytes=1000)
                base = snapshot()
                policy.observe(base)
                next_sample = snapshot(start=11, baseline=base, previous=base)
                mutate(next_sample)
                with self.assertRaises(PressureSourceError):
                    policy.observe(next_sample)

    def test_preflight_absent_zram_or_unidentified_active_device_refuses(self):
        for changes in ({"zram": [], "swap_devices": []}, {"zram": None},
                        {"swap_devices": []}, {"swap_devices": None}):
            with self.subTest(changes=changes):
                item = snapshot()
                item.update(changes)
                with self.assertRaises(PressureSourceError):
                    self.policy(max_zram_growth_bytes=0).observe(item)

    def test_native_page_size_and_source_clock_remain_stable(self):
        for field, value in (("page_size_bytes", 8192), ("started_monotonic_seconds", 10),
                             ("started_monotonic_seconds", float("nan"))):
            with self.subTest(field=field, value=value):
                policy = self.policy(max_swapout_bytes=10**9)
                base = snapshot()
                policy.observe(base)
                item = snapshot(start=11, baseline=base, previous=base)
                item[field] = value
                if field == "page_size_bytes":
                    for key in ("value", "delta_from_baseline", "delta_from_previous"):
                        item["vmstat"]["pswpout_bytes"][key] *= 2
                with self.assertRaises(PressureSourceError):
                    policy.observe(item)

    def test_unrelated_optional_failure_does_not_invalidate_required_sources(self):
        item = snapshot()
        item["cgroup_v2"] = None
        item["unavailable"] = [{"source": "memory.current", "reason": "PermissionError"}]
        self.assertEqual(self.policy(max_swapout_bytes=0).observe(item)["violations"], [])

    def test_omitted_device_identity_failure_is_not_a_complete_inventory(self):
        for preflight in (True, False):
            with self.subTest(preflight=preflight):
                policy = self.policy(max_zram_growth_bytes=1000)
                base = snapshot(zram=(100,))
                if not preflight:
                    policy.observe(base)
                item = (base if preflight else
                        snapshot(start=11, zram=(100,), baseline=base, previous=base))
                item["unavailable"] = [{"source": "/fixture/sys/block/zram1", "reason": "PermissionError"}]
                with self.assertRaisesRegex(PressureSourceError, "incomplete zram inventory"):
                    policy.observe(item)

    def test_optional_zram_diagnostics_are_not_required_guard_sources(self):
        item = snapshot()
        item["unavailable"] = [{"source": "/sys/block/zram0/bd_stat", "reason": "FileNotFoundError"},
                               {"source": "/sys/block/zram0/io_stat", "reason": "PermissionError"}]
        self.assertEqual(self.policy(max_zram_growth_bytes=0).observe(item)["violations"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
