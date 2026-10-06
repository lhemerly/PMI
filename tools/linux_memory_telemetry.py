"""Read-only, best-effort Linux memory observations; never a guard policy.

Each sample has its own monotonic start/duration; the files are not an atomic
snapshot. Gauges have signed deltas. Counters have nonnegative deltas or null
after a reset/missing baseline. Missing optional data is null plus an explicit
unavailable record, never a fabricated zero. No sysfs/proc/cgroup file is written.
"""

import math
import os
from pathlib import Path, PurePosixPath
import re
import time


def unsigned(text):
    text = text.strip()
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError("expected an unsigned integer")
    return int(text)


def key_values(text):
    result = {}
    for line in text.splitlines():
        key, value = line.split()
        if key in result:
            raise ValueError("duplicate counter")
        result[key] = unsigned(value)
    return result


def pressure(text):
    result = {}
    for line in text.splitlines():
        name, *items = line.split()
        if name not in ("some", "full") or name in result:
            raise ValueError("invalid or duplicate PSI row")
        values = {}
        for item in items:
            key, value = item.split("=")
            if key in values:
                raise ValueError("duplicate PSI field")
            values[key] = value
        row = {}
        for key in ("avg10", "avg60", "avg300"):
            value = float(values[key])
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError("invalid PSI percentage")
            row[key + "_percent"] = value
        row["total_stall_us"] = unsigned(values["total"])
        result[name] = row
    if set(result) != {"some", "full"}:
        raise ValueError("memory PSI requires some and full rows")
    return result


def unescape_mount(text):
    return re.sub(r"\\(040|011|012|134)", lambda match: chr(int(match[1], 8)), text)


def absolute_path(text):
    path = PurePosixPath(text)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("unresolvable cgroup path")
    return path


def membership(text):
    paths = [line.split(":", 2)[2] for line in text.splitlines() if line.startswith("0::")]
    if len(paths) != 1:
        raise ValueError("one unified cgroup membership is required")
    return str(absolute_path(paths[0]))


def cgroup_location(member, mountinfo):
    candidates = []
    for line in mountinfo.splitlines():
        left, separator, right = line.partition(" - ")
        if not separator or right.split()[0] != "cgroup2":
            continue
        fields = left.split()
        root = absolute_path(unescape_mount(fields[3]))
        mount = absolute_path(unescape_mount(fields[4]))
        try:
            relative = absolute_path(member).relative_to(root)
        except ValueError:
            continue
        candidates.append((len(root.parts), str(root), str(mount), str(mount / relative)))
    if not candidates:
        raise ValueError("membership is not mapped by a visible cgroup2 mount")
    longest = max(item[0] for item in candidates)
    matches = set(item[1:] for item in candidates if item[0] == longest)
    if len(matches) != 1:
        raise ValueError("ambiguous cgroup2 mounts")
    return next(iter(matches))


class LinuxMemoryTelemetry:
    SCHEMA_VERSION = 1
    LIMITS = ("memory.min", "memory.low", "memory.high", "memory.max",
              "memory.swap.high", "memory.swap.max")
    STAT_BYTES = ("anon", "file", "kernel", "shmem", "file_mapped", "file_dirty",
                  "file_writeback", "anon_thp", "file_thp", "shmem_thp", "inactive_anon",
                  "active_anon", "inactive_file", "active_file", "unevictable", "slab",
                  "sock", "pagetables", "swapcached", "zswap", "zswapped")
    STAT_COUNTS = ("pgfault", "pgmajfault", "pgscan", "pgsteal", "pgscan_kswapd",
                   "pgscan_direct", "pgsteal_kswapd", "pgsteal_direct")
    MM_FIELDS = ("orig_data_size_bytes", "compr_data_size_bytes", "mem_used_total_bytes",
                 "mem_limit_bytes", "mem_used_max_bytes", "same_pages", "pages_compacted",
                 "huge_pages", "huge_pages_since")

    def __init__(self, proc_root=Path("/proc"), sys_root=Path("/sys"), page_size=None,
                 clock=time.monotonic):
        self.proc = Path(proc_root)
        self.sys = Path(sys_root)
        self.page_size = os.sysconf("SC_PAGE_SIZE") if page_size is None else page_size
        if not isinstance(self.page_size, int) or self.page_size <= 0:
            raise ValueError("page size must be a positive byte count")
        self.clock = clock
        self.baseline = {}
        self.previous = {}
        self.last_seen = {}
        self.invalid_baselines = set()
        self.first = True
        self.previous_start = None

    def _read(self, path, parse=lambda text: text.strip()):
        try:
            return parse(path.read_text())
        except (OSError, ValueError, KeyError, IndexError) as error:
            self.unavailable.append({"source": str(path), "reason": type(error).__name__,
                                     "detail": str(error)})
            return None

    def _identity(self, path):
        try:
            stat = path.stat()
            return f"{path}:{stat.st_dev}:{stat.st_ino}"
        except OSError as error:
            self.unavailable.append({"source": str(path), "reason": type(error).__name__,
                                     "detail": str(error)})
            return None

    def _metric(self, key, value, counter=False):
        old = self.previous.get(key)
        if self.first:
            self.baseline[key] = value
        base = self.baseline.get(key)
        known = self.last_seen.get(key)
        reset = counter and known is not None and value < known
        if reset:
            self.invalid_baselines.add(key)
        self.current[key] = value
        self.last_seen[key] = value
        return {"value": value,
                "delta_from_baseline": (value - base if base is not None and
                                        key not in self.invalid_baselines else None),
                "delta_from_previous": (value - old if old is not None and
                                        not (counter and value < old) else None),
                "counter_reset": reset}

    def discard_failed_sample(self):
        # A failed collection cannot provide an interval endpoint or introduce
        # a later, silently shifted preflight baseline.
        if self.first:
            self.baseline.clear()
        self.previous = {}
        self.previous_start = None
        self.first = False

    def _pressure(self, path, identity):
        rows = self._read(path, pressure)
        if rows is not None:
            for name, row in rows.items():
                row["total_stall_us"] = self._metric(
                    (identity, name, "pressure"), row["total_stall_us"], counter=True)
        return rows

    def _vmstat(self):
        values = self._read(self.proc / "vmstat", key_values)
        if values is None:
            return None
        result = {}
        for name in ("pswpin", "pswpout", "pgscan_kswapd", "pgscan_direct",
                     "pgsteal_kswapd", "pgsteal_direct"):
            value = values.get(name)
            result[name + "_pages"] = (self._metric(("vmstat", name), value, True)
                                       if value is not None else None)
            if name in ("pswpin", "pswpout"):
                result[name + "_bytes"] = (self._metric(("vmstat_bytes", name),
                                                        value * self.page_size, True)
                                            if value is not None else None)
            if value is None:
                self.unavailable.append({"source": str(self.proc / "vmstat"),
                                         "reason": "missing_field", "field": name})
        return result

    def _swaps(self):
        def parse(text):
            rows = text.splitlines()
            if not rows or rows[0].split() != ["Filename", "Type", "Size", "Used", "Priority"]:
                raise ValueError("unexpected /proc/swaps header")
            result = []
            for row in rows[1:]:
                path, kind, size, used, priority = row.split()
                result.append({"path": unescape_mount(path), "type": kind,
                               "size_bytes": unsigned(size) * 1024,
                               "used_bytes": unsigned(used) * 1024, "priority": int(priority)})
            return result
        return self._read(self.proc / "swaps", parse)

    def _zram(self):
        try:
            devices = sorted(path for path in (self.sys / "block").iterdir()
                             if re.fullmatch(r"zram[0-9]+", path.name))
        except OSError as error:
            self.unavailable.append({"source": str(self.sys / "block"),
                                     "reason": type(error).__name__, "detail": str(error)})
            return None
        result = []
        for path in devices:
            identity = self._identity(path)
            if identity is None:
                continue
            device = {"name": path.name, "identity": identity,
                      "disksize_bytes": self._read(path / "disksize", unsigned),
                      "comp_algorithm": self._read(path / "comp_algorithm"),
                      "backing_dev": self._read(path / "backing_dev")}

            def mm_parse(text):
                values = [unsigned(item) for item in text.split()]
                if len(values) < 7:
                    raise ValueError("zram mm_stat requires at least seven fields")
                return values

            values = self._read(path / "mm_stat", mm_parse)
            device["mm_stat"] = None
            device["compression_ratio_uncompressed_to_compressed"] = None
            if values is not None:
                device["mm_stat"] = {name: self._metric((identity, name), values[i],
                                                       counter=name in ("pages_compacted", "huge_pages_since"))
                                     if i < len(values) else None
                                     for i, name in enumerate(self.MM_FIELDS)}
                if values[1]:
                    device["compression_ratio_uncompressed_to_compressed"] = values[0] / values[1]
                device["allocator_overhead_bytes"] = values[2] - values[1]

            def bd_parse(text):
                values = [unsigned(item) for item in text.split()]
                if len(values) != 3:
                    raise ValueError("zram bd_stat requires three fields")
                return values

            backing = self._read(path / "bd_stat", bd_parse)
            device["bd_stat"] = (None if backing is None else {
                name: self._metric((identity, name), backing[i] * 4096, counter=i > 0)
                for i, name in enumerate(("backing_count_bytes", "backing_read_bytes",
                                          "backing_write_bytes"))})
            def io_parse(text):
                values = [unsigned(item) for item in text.split()]
                if len(values) < 4:
                    raise ValueError("zram io_stat requires four fields")
                return values
            io = self._read(path / "io_stat", io_parse)
            device["io_stat"] = (None if io is None else {
                name: self._metric((identity, name), io[i], True)
                for i, name in enumerate(("failed_reads", "failed_writes", "invalid_io", "notify_free"))})
            result.append(device)
        return result

    def _limits(self, path):
        return {name: self._read(path / name, lambda text: "max" if text.strip() == "max"
                                 else unsigned(text.strip())) for name in self.LIMITS}

    def _events(self, path, identity):
        values = self._read(path, key_values)
        return (None if values is None else
                {key: self._metric((identity, path.name, key), value, True)
                 for key, value in values.items()})

    def _cgroup(self):
        member = self._read(self.proc / "self/cgroup", membership)
        mounts = self._read(self.proc / "self/mountinfo")
        if member is None or mounts is None:
            return None
        try:
            root, mount, directory = cgroup_location(member, mounts)
        except (ValueError, IndexError) as error:
            self.unavailable.append({"source": "cgroup_v2", "reason": "unresolved_membership",
                                     "detail": str(error)})
            return None
        path = Path(directory)
        identity = self._identity(path)
        if identity is None:
            return None
        result = {"scope": "supervisor_cgroup_context", "attribution": "shared_or_unknown",
                  "membership_path": member, "mount_root": root, "mount_point": mount,
                  "identity": identity, "limits": self._limits(path), "usage_bytes": {}}
        for name in ("memory.current", "memory.swap.current"):
            value = self._read(path / name, unsigned)
            result["usage_bytes"][name] = (None if value is None else
                                           self._metric((identity, name), value))
        values = self._read(path / "memory.stat", key_values)
        for field, names, counter in (("stat_bytes", self.STAT_BYTES, False),
                                       ("stat_counts", self.STAT_COUNTS, True)):
            result[field] = (None if values is None else
                             {name: self._metric((identity, name), values[name], counter)
                              if name in values else None for name in names})
        result["events"] = self._events(path / "memory.events", identity)
        result["events_local"] = self._events(path / "memory.events.local", identity)
        result["memory_pressure"] = self._pressure(path / "memory.pressure", identity)
        result["ancestors"] = []
        for parent in path.parents:
            if path == Path(mount) or not parent.is_relative_to(Path(mount)):
                break
            parent_id = self._identity(parent)
            if parent_id is not None:
                result["ancestors"].append({"path": str(parent), "identity": parent_id,
                                             "limits": self._limits(parent),
                                             "events": self._events(parent / "memory.events", parent_id)})
            if parent == Path(mount):
                break
        return result

    def sample(self):
        start = self.clock()
        self.unavailable = []
        self.current = {}
        result = {"schema_version": self.SCHEMA_VERSION, "page_size_bytes": self.page_size,
                  "started_monotonic_seconds": start,
                  "interval_seconds": (None if self.previous_start is None else start - self.previous_start),
                  "vmstat": self._vmstat(),
                  "memory_pressure": self._pressure(self.proc / "pressure/memory", "host"),
                  "zram": self._zram(), "cgroup_v2": self._cgroup(),
                  "swap_devices": self._swaps(),
                  "swappiness": self._read(self.proc / "sys/vm/swappiness", unsigned)}
        result["unavailable"] = self.unavailable
        result["duration_seconds"] = self.clock() - start
        # Only the immediately previous snapshot is eligible for interval
        # deltas; a missing sample never silently widens the interval.
        self.previous = self.current
        self.previous_start = start
        self.first = False
        return result
