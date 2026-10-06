#!/usr/bin/env python3
r"""Run a local command under explicit Linux memory and wall-time guards.

Example (choose limits for the actual machine and command):
  python3 tools/run_bounded.py --wall-seconds 60 --max-rss-bytes 256MiB \
    --min-available-bytes 1GiB --max-swap-growth-bytes 0 --as-bytes 512MiB \
    --log run.jsonl --output command.log -- python3 -c 'print("hello")'

JSONL requires a new regular --log file; command output goes to --output, or
stderr. Files are created exclusively, never overwritten. Use a local filesystem
for the log: regular-file I/O or /proc stalls can still delay enforcement. Pipes
are not supported log sinks because reader backpressure can suspend monitoring.
RSS is summed across the owned process group (shared pages can be counted more
than once). Polling can
miss allocation bursts; optional RLIMIT_AS is a hard per-process virtual-address
limit, including file mappings, not an aggregate RSS limit. /proc I/O totals are
sampled: short-lived workers can be missed, and waited-for child accounting can
overlap parent counters. This is a cooperative command runner,
not a sandbox: descendants that deliberately change their session/group escape
its scope. Only the fresh session/process group created here receives signals.
On leader exit the group also receives TERM before checking leftover workers.
Limits are sampled; cleanup adds the configured TERM grace and KILL wait.
--max-swap-growth-bytes caps owned-group VmSwap. The optional
--max-system-swap-growth-bytes separately caps system-wide swap-used growth from
preflight; it defaults to the same value, preserving the original stop limits.
Linux observations remain optional unless their corresponding pressure limit
is enabled: --max-zram-growth-bytes caps aggregate physical mem_used_total
growth, --max-swapout-bytes caps cumulative host pswpout traffic, and
--max-psi-some-us / --max-psi-full-us cap cumulative host memory PSI stall time.
These include other workloads; no background or compression adjustment is
subtracted. Required sources must be valid before launch and remain continuous.
--admit-available-bytes optionally sets a higher, preflight-only RAM floor.
Additional reads add measurable sampling overhead. All guards remain sampled.
"""

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import stat
import subprocess
import sys
import time

from linux_memory_telemetry import LinuxMemoryTelemetry
from pressure_policy import PressurePolicy, PressureSourceError

OBSERVATION_SCHEMA_VERSION = LinuxMemoryTelemetry.SCHEMA_VERSION


class MonitorError(RuntimeError):
    pass


def byte_count(value):
    match = re.fullmatch(r"(\d+)(KiB|MiB|GiB|TiB)?", value)
    if not match:
        raise argparse.ArgumentTypeError("use bytes or an integer with KiB/MiB/GiB/TiB")
    power = {None: 0, "KiB": 1, "MiB": 2, "GiB": 3, "TiB": 4}[match[2]]
    return int(match[1]) * 1024 ** power


def positive_seconds(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("seconds must be finite and positive")
    return result


def microseconds(value):
    if not re.fullmatch(r"[0-9]+", value):
        raise argparse.ArgumentTypeError("use a nonnegative integer number of microseconds")
    return int(value)


def fields(path):
    result = {}
    for line in path.read_text().splitlines():
        name, separator, rest = line.partition(":")
        if separator:
            parts = rest.split()
            if parts and parts[0].isdigit():
                result[name] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == "kB" else 1)
    return result


def system_memory():
    values = fields(Path("/proc/meminfo"))
    required = ("MemAvailable", "SwapTotal", "SwapFree")
    if any(name not in values for name in required):
        raise MonitorError("required /proc/meminfo counters are unavailable")
    return {"mem_available_bytes": values["MemAvailable"],
            "system_swap_used_bytes": values["SwapTotal"] - values["SwapFree"],
            # Same read as the required guard counters; optional fields can be
            # absent on other kernels without changing guard behavior.
            "meminfo_bytes": {name: values.get(name) for name in (
                "MemTotal", "MemFree", "MemAvailable", "Cached", "Buffers",
                "SReclaimable", "AnonPages", "Shmem", "SwapTotal", "SwapFree", "SwapCached")}}


def identity(pid):
    # The command name in parentheses may itself contain spaces or ')'.
    text = Path(f"/proc/{pid}/stat").read_text()
    tail = text.rsplit(")", 1)[1].split()
    return {"state": tail[0], "group": int(tail[2]), "session": int(tail[3]),
            "start_ticks": int(tail[19])}


class Sampler:
    def __init__(self, group, initial_swap):
        self.group = group
        self.initial_swap = initial_swap
        self.io_seen = {}
        self.peak_rss = 0
        self.peak_swap_growth = 0
        self.peak_group_swap = 0
        self.peak_system_swap_growth = 0
        self.minimum_available = None
        self.samples = 0

    def sample(self):
        memory = system_memory()
        members = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                before = identity(pid)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            if before["group"] != self.group or before["session"] != self.group:
                continue
            try:
                status = fields(entry / "status")
                io = fields(entry / "io")
                after = identity(pid)
            except (FileNotFoundError, ProcessLookupError):
                continue  # A worker exited during this sample.
            except PermissionError as error:
                raise MonitorError(f"cannot read owned process {pid}: {error}") from error
            if (before["start_ticks"] != after["start_ticks"] or
                    after["group"] != self.group or after["session"] != self.group):
                continue
            alive = after["state"] not in ("Z", "X", "x")
            if any(name not in status for name in ("VmRSS", "VmSwap")):
                # exit_mm can clear these fields before stat becomes Z. A main
                # thread may also exit while other threads retain the address
                # space, so try their shared process memory counters as well.
                for _ in range(3):
                    try:
                        for task in (entry / "task").iterdir():
                            try:
                                candidate = fields(task / "status")
                            except (FileNotFoundError, ProcessLookupError):
                                continue
                            if all(name in candidate for name in ("VmRSS", "VmSwap")):
                                status, alive = candidate, True
                                break
                        else:
                            after = identity(pid)
                            alive = after["state"] not in ("Z", "X", "x")
                        if not alive or all(name in status for name in ("VmRSS", "VmSwap")):
                            break
                    except (FileNotFoundError, ProcessLookupError):
                        alive = False
                        break
                    time.sleep(0.001)
                if alive and any(name not in status for name in ("VmRSS", "VmSwap")):
                    raise MonitorError(f"memory counters missing for live owned process {pid}")
            required_io = ("read_bytes", "write_bytes", "rchar", "wchar")
            if any(name not in io for name in required_io):
                raise MonitorError(f"I/O counters missing for owned process {pid}")
            key = (pid, before["start_ticks"])
            old = self.io_seen.setdefault(key, {name: 0 for name in required_io})
            for name in required_io:
                old[name] = max(old[name], io[name])
            members.append({"pid": pid, "alive": alive, "state": after["state"],
                            "rss_bytes": status.get("VmRSS", 0),
                            "swap_bytes": status.get("VmSwap", 0)})
        rss = sum(member["rss_bytes"] for member in members)
        swap = sum(member["swap_bytes"] for member in members)
        system_growth = max(0, memory["system_swap_used_bytes"] - self.initial_swap)
        growth = max(swap, system_growth)
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_swap_growth = max(self.peak_swap_growth, growth)
        self.peak_group_swap = max(self.peak_group_swap, swap)
        self.peak_system_swap_growth = max(self.peak_system_swap_growth, system_growth)
        available = memory["mem_available_bytes"]
        self.minimum_available = available if self.minimum_available is None else min(self.minimum_available, available)
        self.samples += 1
        totals = {name: sum(value[name] for value in self.io_seen.values())
                  for name in ("read_bytes", "write_bytes", "rchar", "wchar")}
        return {**memory, "group_rss_bytes": rss, "group_swap_bytes": swap,
                "system_swap_growth_bytes": system_growth, "swap_growth_bytes": growth,
                "sampled_io_bytes": totals, "members": members}


def leader_exited(process):
    # WNOWAIT leaves a dead leader unreaped, reserving its PID/PGID until the
    # last group signal. Popen.poll()/wait() must not be used before cleanup.
    return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def signal_owned_group(process, sig):
    # Popen(start_new_session=True) makes this exact child its session/group
    # leader. It is still live or unreaped, so this identifier cannot be reused.
    try:
        if os.getpgid(process.pid) != process.pid or os.getsid(process.pid) != process.pid:
            raise MonitorError("owned child no longer leads its original session/group")
        os.killpg(process.pid, sig)
        return True
    except ProcessLookupError:
        return False


def system_swap_limit(args):
    override = getattr(args, "max_system_swap_growth_bytes", None)
    return args.max_swap_growth_bytes if override is None else override


def resource_violations(sample, args):
    violations = []
    if sample["group_rss_bytes"] > args.max_rss_bytes:
        violations.append("max_rss_bytes")
    if sample["mem_available_bytes"] < args.min_available_bytes:
        violations.append("min_available_bytes")
    if sample["group_swap_bytes"] > args.max_swap_growth_bytes:
        violations.append("max_swap_growth_bytes")
    if sample["system_swap_growth_bytes"] > system_swap_limit(args):
        violations.append("max_system_swap_growth_bytes")
    return violations


def observation_error(error, started):
    return {"schema_version": OBSERVATION_SCHEMA_VERSION, "collection_failed": True,
            "started_monotonic_seconds": started,
            "duration_seconds": time.monotonic() - started,
            "unavailable": [{"source": "linux_memory_telemetry", "reason": type(error).__name__,
                             "detail": str(error)}]}


def observe_memory(observer):
    # Optional failures stay diagnostic; PressurePolicy separately rejects an
    # unavailable explicitly required source. Interrupts reach group cleanup.
    started = time.monotonic()
    try:
        return observer.sample()
    except Exception as error:
        observer.discard_failed_sample()
        return observation_error(error, started)


def supervise(args, emit, output):
    pressure_policy = PressurePolicy(args)
    diagnostics = None
    if getattr(args, "loader_diagnostics", False):
        from loader_diagnostics import LoaderDiagnostics
        diagnostics = LoaderDiagnostics()
        diagnostics.sample()  # Optional diagnostic baseline before launch.
    admission = getattr(args, "admit_available_bytes", None)
    if admission is not None and (type(admission) is not int or admission < args.min_available_bytes):
        raise MonitorError("admit_available_bytes must be at least min_available_bytes")
    observation_started = time.monotonic()
    try:
        observer = LinuxMemoryTelemetry()
    except Exception as error:
        observer = None
        observations = observation_error(error, observation_started)
    else:
        observations = observe_memory(observer)
    pressure_error = None
    try:
        pressure_policy.observe(observations)
    except PressureSourceError as error:
        pressure_error = str(error)
    # Both baselines are freshly collected during this invocation, before
    # launch. MemAvailable is sampled last for the admission decision.
    initial = system_memory()
    limits = {"wall_seconds": args.wall_seconds, "max_rss_bytes": args.max_rss_bytes,
              "min_available_bytes": args.min_available_bytes,
              "max_swap_growth_bytes": args.max_swap_growth_bytes,
              "max_system_swap_growth_bytes": system_swap_limit(args),
              "as_bytes": args.as_bytes, "interval_seconds": args.interval_seconds,
              "term_grace_seconds": args.term_grace_seconds,
              "kill_wait_seconds": args.kill_wait_seconds}
    limits.update(pressure_policy.limits, admit_available_bytes=admission)
    emit("start", command=args.command, limits=limits, preflight=initial,
         observations=observations, pressure_guards=pressure_policy.summary())
    if pressure_error is not None:
        emit("final", status="refused", reason="pressure_source_error", command_started=False,
             runner_returncode=125, preflight=initial, monitor_errors=[pressure_error],
             pressure_guards=pressure_policy.summary())
        return 125
    if admission is not None and initial["mem_available_bytes"] < admission:
        emit("final", status="refused", reason="admit_available_bytes", command_started=False,
             runner_returncode=125, preflight=initial, pressure_guards=pressure_policy.summary())
        return 125
    if initial["mem_available_bytes"] < args.min_available_bytes:
        emit("final", status="refused", reason="min_available_bytes", command_started=False,
             runner_returncode=125, preflight=initial)
        return 125

    def address_limit():
        resource.setrlimit(resource.RLIMIT_AS, (args.as_bytes, args.as_bytes))

    caught = []

    def interrupted(signum, _frame):
        caught.append(signum)

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    # An inherited SIGCHLD=SIG_IGN could auto-reap the leader and invalidate the
    # reserved-PID argument. Change only this supervisor's signal disposition.
    previous[signal.SIGCHLD] = signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    process = None
    started = time.monotonic()
    sampler = None
    reason = None
    status = "completed"
    term_at = kill_at = None
    term_sent = kill_sent = False
    last = None
    remaining = []
    command_returncode = None
    monitor_errors = []
    exit_observed = False
    loop_finished = False
    try:
        # This supervisor is single-threaded; the only pre-exec action is the
        # optional resource limit. No shell is inserted around the command.
        process = subprocess.Popen(args.command, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=output, start_new_session=True,
                                   preexec_fn=address_limit if args.as_bytes is not None else None)
        sampler = Sampler(process.pid, initial["system_swap_used_bytes"])
        emit("spawn", pid=process.pid, process_group=process.pid)
        while True:
            now = time.monotonic()
            elapsed = now - started
            measured_violations = []
            observed = None
            pressure_sample = None
            # Collect before the guard snapshot, never between detecting a
            # violation and signaling. Once cleanup/deadline/interruption is
            # pending, diagnostic reads must not delay TERM/KILL checks.
            diagnostic_sample = None
            if diagnostics is not None and not caught and term_at is None and elapsed < args.wall_seconds:
                try:
                    diagnostic_sample = diagnostics.sample(last["members"] if last else (), process.pid)
                except Exception as error:
                    diagnostic_sample = {"error": str(error), "scope": "optional_only"}
            elapsed = time.monotonic() - started
            if observer is not None and not caught and term_at is None and elapsed < args.wall_seconds:
                observed = observe_memory(observer)
                observations = observed
            try:
                elapsed = time.monotonic() - started
                last = sampler.sample()
                if observed is not None:
                    pressure_sample = pressure_policy.observe(observed)
                    if pressure_sample is not None:
                        measured_violations.extend(pressure_sample["violations"])
                measured_violations.extend(resource_violations(last, args))
                remaining = [member["pid"] for member in last["members"] if member["alive"]]
                emit("sample", elapsed_seconds=elapsed, **last, **({"loader_diagnostics": diagnostic_sample} if diagnostics else {}), observations=observed,
                     pressure_guards=pressure_sample,
                     observations_status=("collected" if observed is not None else
                                          "unavailable_collector" if observer is None else
                                          "skipped_cleanup_or_deadline"))
            except (OSError, ValueError, IndexError, MonitorError, PressureSourceError) as error:
                monitor_errors.append(str(error))
                emit("monitor_error", elapsed_seconds=elapsed, message=str(error))
                if reason is None:
                    reason = "pressure_source_error" if isinstance(error, PressureSourceError) else "monitor_error"
                    status = "monitor_error"
                # Missing observability is not permission to let the command run.
                remaining = [process.pid]

            exited = leader_exited(process)
            if exited and not exit_observed:
                exit_observed = True
                if term_at is None:
                    # Close the owned group before the post-exit scan, also
                    # covering ordinary workers forked after an older snapshot.
                    # Signaling an otherwise empty group with a zombie leader
                    # is harmless. No other session/group can receive this.
                    term_sent = signal_owned_group(process, signal.SIGTERM)
                    term_at = time.monotonic()
                    emit("cleanup", phase="leader_exit", signal="SIGTERM", sent=term_sent)
                try:
                    # The leader (and workers it waited for) may have exited
                    # after the first snapshot. Refresh before classifying an
                    # ordinary successful command as leaving live descendants.
                    last = sampler.sample()
                    measured_violations.extend(resource_violations(last, args))
                    remaining = [member["pid"] for member in last["members"]
                                 if member["alive"] and member["pid"] != process.pid]
                    final_observations = final_pressure = None
                    # TERM has already closed the group. Refresh required
                    # sources once before accepting a clean exit, even when a
                    # fast command completed before the next ordinary sample.
                    # Do not perform these reads during failure cleanup or
                    # while surviving workers need prompt TERM/KILL handling.
                    if pressure_policy.enabled and reason is None and not remaining and not measured_violations and not caught:
                        final_observations = (observe_memory(observer) if observer is not None
                                              else observations)
                        observations = final_observations
                        final_pressure = pressure_policy.observe(final_observations)
                        measured_violations.extend(final_pressure["violations"])
                    emit("sample", elapsed_seconds=time.monotonic() - started,
                         phase="leader_exit", **last, observations=final_observations,
                         pressure_guards=final_pressure,
                         observations_status=("required_final" if final_observations is not None
                                              else "skipped_cleanup_or_deadline"))
                except (OSError, ValueError, IndexError, MonitorError, PressureSourceError) as error:
                    monitor_errors.append(str(error))
                    emit("monitor_error", message=str(error))
                    if reason is None:
                        reason = "pressure_source_error" if isinstance(error, PressureSourceError) else "monitor_error"
                        status = "monitor_error"
            violations = []
            if caught:
                violations.append(f"signal_{caught[0]}")
            # Preserve violations observed before exit rather than replacing
            # them with a subsequent zero-RSS zombie sample.
            violations.extend(dict.fromkeys(measured_violations))
            # A quick command may finish between samples. Observing that exit
            # at the deadline does not prove the command exceeded wall time.
            if not (exited and not remaining) and time.monotonic() - started >= args.wall_seconds:
                violations.append("wall_seconds")

            if reason is None and violations:
                reason = violations[0]
                status = "interrupted" if caught else "limit_exceeded"
            if reason is None and exited:
                if remaining:
                    reason, status = "descendants_after_command_exit", "descendants_terminated"
                else:
                    break
            if reason is not None:
                if term_at is None:
                    term_sent = signal_owned_group(process, signal.SIGTERM)
                    term_at = time.monotonic()
                    emit("stop", reason=reason, violations=violations, signal="SIGTERM", sent=term_sent)
                if exited and not remaining:
                    break
                if kill_at is None and time.monotonic() - term_at >= args.term_grace_seconds:
                    kill_sent = signal_owned_group(process, signal.SIGKILL)
                    kill_at = time.monotonic()
                    emit("stop", reason=reason, signal="SIGKILL", sent=kill_sent)
                if kill_at is not None and time.monotonic() - kill_at >= args.kill_wait_seconds:
                    status = "cleanup_incomplete"
                    break  # SIGKILL cannot immediately resolve uninterruptible I/O.

            delay = args.interval_seconds
            if reason is None:
                delay = min(delay, max(0.001, args.wall_seconds - (time.monotonic() - started)))
            elif kill_at is None:
                delay = min(delay, max(0.001, args.term_grace_seconds - (time.monotonic() - term_at)))
            else:
                delay = min(delay, max(0.001, args.kill_wait_seconds - (time.monotonic() - kill_at)))
            time.sleep(delay)
        loop_finished = True
    except BaseException as error:
        # Cleanup must also run for unexpected programming/logging exceptions
        # and allocation failure in the supervisor, not only known I/O errors.
        reason, status = "runner_error", "runner_error"
        try:
            monitor_errors.append(f"{type(error).__name__}: {error}")
            emit("monitor_error", message=monitor_errors[-1])
        except BaseException:
            pass  # A failed log sink must never bypass child cleanup.
    finally:
        if process is not None:
            # An unexpected monitor failure still cleans up only our group. The
            # normal loop has already applied the graceful termination interval.
            if not loop_finished or status == "runner_error":
                try:
                    term_sent = signal_owned_group(process, signal.SIGTERM) or term_sent
                    deadline = time.monotonic() + args.term_grace_seconds
                    for cleanup_signal in (signal.SIGTERM, signal.SIGKILL):
                        while time.monotonic() < deadline:
                            try:
                                last = sampler.sample() if sampler else None
                                remaining = ([member["pid"] for member in last["members"] if member["alive"]]
                                             if last is not None else [process.pid])
                            except BaseException:
                                remaining = [process.pid]  # Cannot verify cleanup.
                            if leader_exited(process) and not remaining:
                                break
                            time.sleep(min(0.05, max(0.001, deadline - time.monotonic())))
                        if leader_exited(process) and not remaining:
                            break
                        if cleanup_signal == signal.SIGTERM:
                            kill_sent = signal_owned_group(process, signal.SIGKILL) or kill_sent
                            deadline = time.monotonic() + args.kill_wait_seconds
                        else:
                            status = "cleanup_incomplete"
                except (OSError, MonitorError) as error:
                    monitor_errors.append(str(error))
            if leader_exited(process):
                # No more signals are sent after this reaps the reserved leader.
                command_returncode = process.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    if status == "completed":
        runner_returncode = command_returncode if command_returncode >= 0 else 128 - command_returncode
    elif status == "interrupted":
        runner_returncode = 128 + caught[0]
    else:
        runner_returncode = 124 if reason == "wall_seconds" and status != "cleanup_incomplete" else 125
    emit("final", status=status, reason=reason, command_started=process is not None,
         command_returncode=command_returncode, runner_returncode=runner_returncode,
         elapsed_seconds=time.monotonic() - started, term_sent=term_sent, kill_sent=kill_sent,
         remaining_live_pids=remaining, monitor_errors=monitor_errors,
         samples=sampler.samples if sampler else 0,
         sampled_peak_rss_bytes=sampler.peak_rss if sampler else None,
         sampled_peak_swap_growth_bytes=sampler.peak_swap_growth if sampler else None,
         sampled_peak_group_swap_bytes=sampler.peak_group_swap if sampler else None,
         sampled_peak_system_swap_growth_bytes=sampler.peak_system_swap_growth if sampler else None,
         minimum_mem_available_bytes=sampler.minimum_available if sampler else None,
         last_sample=last, last_observations=observations, pressure_guards=pressure_policy.summary())
    return runner_returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wall-seconds", required=True, type=positive_seconds)
    parser.add_argument("--max-rss-bytes", required=True, type=byte_count)
    parser.add_argument("--min-available-bytes", required=True, type=byte_count)
    parser.add_argument("--admit-available-bytes", type=byte_count,
                        help="optional preflight-only RAM floor, at least --min-available-bytes")
    parser.add_argument("--max-swap-growth-bytes", required=True, type=byte_count,
                        help="owned-group VmSwap cap; also the default system-growth cap")
    parser.add_argument("--max-system-swap-growth-bytes", type=byte_count,
                        help="system swap-used growth cap from preflight (default: --max-swap-growth-bytes)")
    parser.add_argument("--max-zram-growth-bytes", type=byte_count,
                        help="optional cap on aggregate physical zram mem_used_total growth from preflight")
    parser.add_argument("--max-swapout-bytes", type=byte_count,
                        help="optional cap on cumulative host pswpout traffic bytes from preflight")
    parser.add_argument("--max-psi-some-us", type=microseconds,
                        help="optional cap on cumulative host memory PSI some stall microseconds")
    parser.add_argument("--max-psi-full-us", type=microseconds,
                        help="optional cap on cumulative host memory PSI full stall microseconds")
    parser.add_argument("--loader-diagnostics", action="store_true", help="optional read-only THP/compaction/zone/owned-mapping startup diagnostics; never guard inputs")
    parser.add_argument("--as-bytes", type=byte_count, help="optional hard per-process RLIMIT_AS")
    parser.add_argument("--interval-seconds", type=positive_seconds, default=1.0)
    parser.add_argument("--term-grace-seconds", type=positive_seconds, default=3.0)
    parser.add_argument("--kill-wait-seconds", type=positive_seconds, default=5.0)
    parser.add_argument("--log", type=Path, required=True, help="new regular JSONL file on local storage")
    parser.add_argument("--output", type=Path, help="new combined command-output file (default: stderr)")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.max_rss_bytes == 0 or args.as_bytes == 0:
        parser.error("max RSS and optional address-space limits must be positive")
    if args.admit_available_bytes is not None and args.admit_available_bytes < args.min_available_bytes:
        parser.error("--admit-available-bytes must be at least --min-available-bytes")
    if not 0.05 <= args.interval_seconds <= 5:
        parser.error("interval must be between 0.05 and 5 seconds")
    if not sys.platform.startswith("linux") or not hasattr(os, "WNOWAIT"):
        parser.error("Linux /proc and waitid(WNOWAIT) are required")
    if args.output and args.log.resolve() == args.output.resolve():
        parser.error("JSON log and command output must use different files")

    log = output = None
    try:
        descriptor = os.open(args.log, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                             os.O_CLOEXEC | os.O_NONBLOCK, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise MonitorError("JSON log must be a new regular file")
            log = os.fdopen(descriptor, "w", buffering=1)
        except BaseException:
            os.close(descriptor)
            raise
        output = args.output.open("x", buffering=1) if args.output else sys.stderr

        def emit(event, **values):
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            print(json.dumps({"event": event, "time_utc": timestamp, **values}, allow_nan=False),
                  file=log, flush=True)

        return supervise(args, emit, output)
    except (OSError, MonitorError) as error:
        print(f"run_bounded: {error}", file=sys.stderr)
        return 125
    finally:
        if log is not None:
            log.close()
        if output is not None and output is not sys.stderr:
            output.close()


if __name__ == "__main__":
    sys.exit(main())
