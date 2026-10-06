"""Opt-in host pressure guards over LinuxMemoryTelemetry schema 1.

The collector remains best-effort. This consumer makes only explicitly enabled
sources mandatory, keeps one preflight baseline, and permanently rejects lost
counter continuity. Host pressure includes other workloads; no background rate
or compression-ratio adjustment is subtracted.
"""

import math
import re


class PressureSourceError(RuntimeError):
    pass


class PressurePolicy:
    LIMITS = ("max_zram_growth_bytes", "max_swapout_bytes",
              "max_psi_some_us", "max_psi_full_us")

    def __init__(self, args):
        self.limits = {name: getattr(args, name, None) for name in self.LIMITS}
        for name, value in self.limits.items():
            if value is not None and (type(value) is not int or value < 0):
                raise PressureSourceError(f"{name} must be a nonnegative integer")
        self.enabled = any(value is not None for value in self.limits.values())
        self.baseline = None
        self.previous = None
        self.identities = None
        self.previous_start = None
        self.latest = None
        self.peak_growth = {}
        self.error = None

    @staticmethod
    def _integer(value, label):
        if type(value) is not int or value < 0:
            raise PressureSourceError(f"missing or invalid {label}")
        return value

    @staticmethod
    def _mapping(value, label):
        if not isinstance(value, dict):
            raise PressureSourceError(f"missing or invalid {label}")
        return value

    def _metric(self, metric, label, *, counter=True):
        metric = self._mapping(metric, label)
        value = self._integer(metric.get("value"), label)
        if metric.get("counter_reset") is not False:
            raise PressureSourceError(f"reset or invalid {label}")
        delta = metric.get("delta_from_baseline")
        if type(delta) is not int or (counter and delta < 0):
            raise PressureSourceError(f"lost baseline for {label}")
        previous_delta = metric.get("delta_from_previous")
        if self.baseline is None:
            if delta != 0 or previous_delta is not None:
                raise PressureSourceError(f"{label} requires a fresh preflight baseline")
        elif type(previous_delta) is not int or (counter and previous_delta < 0):
            raise PressureSourceError(f"lost previous sample for {label}")
        return value, delta

    def _zram(self, observations):
        devices = observations.get("zram")
        swaps = observations.get("swap_devices")
        if not isinstance(devices, list) or not isinstance(swaps, list):
            raise PressureSourceError("zram and active swap inventories are required")
        unavailable = observations.get("unavailable")
        if not isinstance(unavailable, list):
            raise PressureSourceError("zram inventory completeness is unavailable")
        for item in unavailable:
            item = self._mapping(item, "unavailable source")
            source = item.get("source")
            # The collector omits a device when its identity cannot be read.
            # A valid remaining device must not turn that partial inventory
            # into an apparently complete physical-memory measurement.
            if isinstance(source, str) and re.search(r"(?:^|/)block/zram[0-9]+/?$", source):
                raise PressureSourceError(f"incomplete zram inventory: {source}")
        identities, values, deltas = {}, {}, {}
        for device in devices:
            device = self._mapping(device, "zram device")
            name, identity = device.get("name"), device.get("identity")
            if (not isinstance(name, str) or not re.fullmatch(r"zram[0-9]+", name)
                    or name in identities or not isinstance(identity, str) or not identity):
                raise PressureSourceError("invalid or duplicate zram device identity")
            size = self._integer(device.get("disksize_bytes"), f"{name} disksize")
            algorithm, backing = device.get("comp_algorithm"), device.get("backing_dev")
            if not isinstance(algorithm, str) or not algorithm or not isinstance(backing, str) or not backing:
                raise PressureSourceError(f"missing zram configuration for {name}")
            identities[name] = (identity, size, algorithm, backing)
            metric = self._mapping(device.get("mm_stat"), f"{name} mm_stat")
            values[name], deltas[name] = self._metric(metric.get("mem_used_total_bytes"),
                                                     f"{name} mem_used_total", counter=False)
        active, active_zram = {}, set()
        for row in swaps:
            row = self._mapping(row, "active swap entry")
            path, kind, priority = row.get("path"), row.get("type"), row.get("priority")
            if (not isinstance(path, str) or not path.startswith("/") or path in active
                    or kind not in ("partition", "file") or type(priority) is not int):
                raise PressureSourceError("invalid or duplicate active swap identity")
            size = self._integer(row.get("size_bytes"), f"{path} size")
            used = self._integer(row.get("used_bytes"), f"{path} used")
            if size == 0 or used > size:
                raise PressureSourceError(f"invalid active swap size for {path}")
            active[path] = (kind, size, priority)
            match = re.fullmatch(r"/dev/(zram[0-9]+)", path)
            if match:
                name = match[1]
                if name not in identities or identities[name][1] == 0:
                    raise PressureSourceError(f"active swap {path} has no valid zram device")
                active_zram.add(name)
        if not active_zram:
            raise PressureSourceError("zram guard requires an identifiable active /dev/zramN swap device")
        return values, deltas, {"zram": identities, "active_swap": active}

    def _read(self, observations):
        observations = self._mapping(observations, "pressure observation")
        if type(observations.get("schema_version")) is not int or observations["schema_version"] != 1 or observations.get("collection_failed"):
            raise PressureSourceError("required pressure collection failed or has an unknown schema")
        start = observations.get("started_monotonic_seconds")
        if (type(start) not in (int, float) or not math.isfinite(start) or start < 0
                or (self.previous_start is not None and start <= self.previous_start)):
            raise PressureSourceError("invalid or nonadvancing pressure sample timestamp")
        values, deltas, identities, counters = {}, {}, {}, set()
        if self.limits["max_zram_growth_bytes"] is not None:
            zram, changes, identities = self._zram(observations)
            # Preserve per-device baselines, as well as the aggregate, so one
            # silently rebased device cannot be hidden by another's decrease.
            for name, value in zram.items():
                values[f"zram:{name}"] = value
                deltas[f"zram:{name}"] = changes[name]
        if self.limits["max_swapout_bytes"] is not None:
            page_size = self._integer(observations.get("page_size_bytes"), "page size")
            if page_size == 0:
                raise PressureSourceError("page size must be positive")
            identities["page_size_bytes"] = page_size
            vmstat = self._mapping(observations.get("vmstat"), "vmstat")
            pages, page_delta = self._metric(vmstat.get("pswpout_pages"), "pswpout pages")
            value, delta = self._metric(vmstat.get("pswpout_bytes"), "pswpout bytes")
            if value != pages * page_size or delta != page_delta * page_size:
                raise PressureSourceError("inconsistent pswpout page/byte counters")
            values["swapout_bytes"], deltas["swapout_bytes"] = value, delta
            counters.add("swapout_bytes")
        for name in ("some", "full"):
            if self.limits[f"max_psi_{name}_us"] is not None:
                pressure = self._mapping(observations.get("memory_pressure"), "host memory PSI")
                row = self._mapping(pressure.get(name), f"host PSI {name}")
                key = f"psi_{name}_us"
                values[key], deltas[key] = self._metric(row.get("total_stall_us"), key)
                counters.add(key)
        return values, deltas, identities, counters, start

    def observe(self, observations):
        if not self.enabled:
            return None
        if self.error is not None:
            raise PressureSourceError(self.error)
        try:
            values, deltas, identities, counters, start = self._read(observations)
            if self.baseline is None:
                self.baseline, self.identities = dict(values), identities
            else:
                if identities != self.identities or values.keys() != self.baseline.keys():
                    raise PressureSourceError("pressure device/configuration identity changed")
                for key, value in values.items():
                    if key in counters and value < self.previous[key]:
                        raise PressureSourceError(f"{key} regressed since the preceding sample")
                    if deltas[key] != value - self.baseline[key]:
                        raise PressureSourceError(f"{key} baseline changed or became invalid")
            self.previous, self.previous_start = values, start
            growth = {key: values[key] - self.baseline[key] for key in sorted(counters)}
            if self.limits["max_zram_growth_bytes"] is not None:
                growth["zram_growth_bytes"] = sum(value - self.baseline[key]
                    for key, value in values.items() if key.startswith("zram:"))
            for key, value in growth.items():
                self.peak_growth[key] = max(self.peak_growth.get(key, 0), value)
            violations = [limit for limit in self.LIMITS if self.limits[limit] is not None
                          and growth[limit.removeprefix("max_")] > self.limits[limit]]
            self.latest = {"started_monotonic_seconds": start, "growth": growth,
                           "absolute_values": values, "violations": violations}
            return self.latest
        except (PressureSourceError, KeyError, TypeError, ValueError) as error:
            self.error = str(error)
            raise PressureSourceError(self.error) from error

    def summary(self):
        return {"enabled": self.enabled, "scope": "host_including_other_workloads",
                "limits": self.limits, "baseline": self.baseline,
                "identities": self.identities, "last_sample": self.latest,
                "sampled_peak_growth": self.peak_growth, "source_error": self.error}
