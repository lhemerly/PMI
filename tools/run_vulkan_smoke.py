#!/usr/bin/env python3
"""Sample absolute device VRAM while supervising a cooperative smoke command.

CALLER CONTRACT: invoke this wrapper INSIDE tools/run_bounded.py's owned process
group, with aggregate RSS <=12GiB, RLIMIT_AS <=12GiB, MemAvailable >=4GiB,
owned-group VmSwap <=64MiB, and system swap-used growth <=512MiB. The caller must
budget startup, execution and both supervisors' cleanup within its 600s slot.
The caller must clean the entire owned group on wrapper exit, including normal
exit. This wrapper neither creates a session/group nor signals a process group;
it only terminates its direct child. Descendants remain the caller's job.

Use only a previously reviewed, tiny synthetic GPU correctness command after
"--". This wrapper is not a sandbox and does not constrain arbitrary commands.
It opens regular sysfs counters, never GPU device nodes, and does no GPU probing.
--sysfs-dir exists for explicit device selection and CPU-only fake-file tests;
the production default is /sys/class/drm/card1/device. The one counter measures
ALL VRAM used on that device, including the desktop and other applications, not
just the child. The absolute cap is therefore conservative.

This is a SAMPLED guard, not a hard driver allocation limit: allocation bursts
between reads may exceed the cap, and sampled_peak_global_vram_bytes is not an
instantaneous peak guarantee. The configured polling delay is at most 0.25s;
scheduler, sysfs or local-log stalls can delay actual reads and enforcement.
Source disappearance, unreadability, invalid data, or changed device/counter
identity fail closed. Read errors never authorize a command to continue.

The child inherits stdout/stderr for the outer runner to capture. --log must be
a new regular file on local storage. Final status/exit code reports supervision
only; exit zero does not establish GPU selection or numerical correctness.
"""

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time


MAX_VRAM_BYTES = 4 * 1024**3
MAX_INTERVAL_SECONDS = 0.25
COUNTER_NAME = "mem_info_vram_used"


class MonitorError(RuntimeError):
    pass


def capped_seconds(maximum):
    def parse(value):
        try:
            result = float(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("seconds must be numeric") from error
        if not math.isfinite(result) or not 0 < result <= maximum:
            raise argparse.ArgumentTypeError(f"seconds must be positive and <= {maximum}")
        return result
    return parse


def vram_bytes(value):
    match = re.fullmatch(r"([0-9]+)(KiB|MiB|GiB)?", value)
    if not match:
        raise argparse.ArgumentTypeError("use bytes or an integer with KiB/MiB/GiB")
    result = int(match[1]) * 1024**{None: 0, "KiB": 1, "MiB": 2, "GiB": 3}[match[2]]
    if not 0 < result <= MAX_VRAM_BYTES:
        raise argparse.ArgumentTypeError("VRAM cap must be positive and <= 4GiB")
    return result


class VramCounter:
    """Read only a bounded regular counter and pin its sampled source identity."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.identity = None

    def sample(self):
        directory_fd = counter_fd = None
        try:
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            before = os.fstat(directory_fd)
            canonical = str(self.directory.resolve(strict=True))
            counter_fd = os.open(COUNTER_NAME, os.O_RDONLY | os.O_CLOEXEC |
                                 os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory_fd)
            counter_stat = os.fstat(counter_fd)
            if not stat.S_ISREG(counter_stat.st_mode):
                raise MonitorError("VRAM counter must be a regular file")
            raw = os.read(counter_fd, 129)
            after = self.directory.stat()
            counter_after = os.stat(COUNTER_NAME, dir_fd=directory_fd, follow_symlinks=False)
            current = {"device_path": canonical, "device_dev": before.st_dev,
                       "device_inode": before.st_ino, "counter_dev": counter_stat.st_dev,
                       "counter_inode": counter_stat.st_ino}
            if ((before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) or
                    canonical != str(self.directory.resolve(strict=True)) or
                    (counter_stat.st_dev, counter_stat.st_ino) !=
                    (counter_after.st_dev, counter_after.st_ino)):
                raise MonitorError("VRAM device/counter identity changed during read")
            if self.identity is not None and current != self.identity:
                raise MonitorError("VRAM device/counter identity changed since preflight")
            if len(raw) > 128 or not re.fullmatch(rb"[0-9]+\n?", raw):
                raise MonitorError("VRAM counter is not a valid unsigned byte count")
            value = int(raw)
            if value > 2**64 - 1:
                raise MonitorError("VRAM counter exceeds unsigned 64-bit range")
            self.identity = current
            return value
        except (OSError, ValueError, RuntimeError) as error:
            if isinstance(error, MonitorError):
                raise
            raise MonitorError(f"cannot read VRAM source: {error}") from error
        finally:
            if counter_fd is not None:
                os.close(counter_fd)
            if directory_fd is not None:
                os.close(directory_fd)


def cleanup_child(process, args, observe, emit, errors):
    """Finite cleanup of our unreaped direct child; never use killpg/setsid."""
    sent = {"term_sent": False, "kill_sent": False}
    for sig, grace, key in ((signal.SIGTERM, args.term_grace_seconds, "term_sent"),
                            (signal.SIGKILL, args.kill_wait_seconds, "kill_sent")):
        if process.poll() is not None:
            break
        # Popen checks/reaps its own child before signaling, avoiding signals to
        # an unrelated reused PID. No process other than this child is signaled.
        process.send_signal(sig)
        sent[key] = True
        deadline = time.monotonic() + grace
        try:
            emit("cleanup", signal=signal.Signals(sig).name, pid=process.pid)
        except BaseException as error:
            errors.append(f"cleanup log: {type(error).__name__}: {error}")
        while process.poll() is None and time.monotonic() < deadline:
            try:
                observe("cleanup")
            except BaseException as error:
                message = f"cleanup sample: {type(error).__name__}: {error}"
                if message not in errors:
                    errors.append(message)
            delay = min(args.interval_seconds, max(0, deadline - time.monotonic()))
            if delay:
                time.sleep(delay)
    return {**sent, "cleanup_incomplete": process.poll() is None}


def supervise(args, emit):
    source = VramCounter(args.sysfs_dir)
    started = time.monotonic()
    peak = last = None
    samples = 0
    process = None
    caught = []
    errors = []
    reason = None
    status = "completed"
    cleanup = {"term_sent": False, "kill_sent": False, "cleanup_incomplete": False}

    def interrupted(signum, _frame):
        if not caught:
            caught.append(signum)

    def observe(phase):
        nonlocal peak, last, samples
        last = source.sample()
        peak = last if peak is None else max(peak, last)
        samples += 1
        emit("preflight" if phase == "preflight" else "sample", phase=phase,
             elapsed_seconds=time.monotonic() - started, global_vram_used_bytes=last,
             sampled_peak_global_vram_bytes=peak, source_identity=source.identity)
        return last

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    previous[signal.SIGCHLD] = signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    try:
        emit("start", command=args.command, sysfs_dir=str(args.sysfs_dir),
             limits={"max_vram_bytes": args.max_vram_bytes,
                     "interval_seconds": args.interval_seconds,
                     "wall_seconds": args.wall_seconds,
                     "term_grace_seconds": args.term_grace_seconds,
                     "kill_wait_seconds": args.kill_wait_seconds},
             measurement="absolute whole-device VRAM; sampled, not a hard allocation limit")
        if observe("preflight") > args.max_vram_bytes:
            reason, status = "max_vram_bytes", "refused"
        elif caught:
            reason, status = f"signal_{caught[0]}", "interrupted"
        else:
            process = subprocess.Popen(args.command, stdin=subprocess.DEVNULL,
                                       start_new_session=False)
            emit("spawn", pid=process.pid, process_group=os.getpgrp(), session=os.getsid(0))
            while True:
                if caught:
                    reason, status = f"signal_{caught[0]}", "interrupted"
                    break
                if observe("running") > args.max_vram_bytes:
                    reason, status = "max_vram_bytes", "limit_exceeded"
                    break
                if process.poll() is not None:
                    if observe("child_exit") > args.max_vram_bytes:
                        reason, status = "max_vram_bytes", "limit_exceeded"
                    break
                remaining = args.wall_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    reason, status = "wall_seconds", "limit_exceeded"
                    break
                time.sleep(min(args.interval_seconds, remaining))
    except MonitorError as error:
        reason, status = "source_error", "monitor_error" if process is not None else "refused"
        errors.append(str(error))
        if process is None:
            emit("preflight", status="refused", error=str(error))
    except BaseException as error:
        reason, status = "runner_error", "runner_error"
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        try:
            if reason is not None:
                try:
                    emit("stop", reason=reason, command_started=process is not None,
                         sampled_peak_global_vram_bytes=peak, errors=errors)
                except BaseException as error:
                    errors.append(f"stop log: {type(error).__name__}: {error}")
            if process is not None:
                cleanup = cleanup_child(process, args, observe, emit, errors)
                if cleanup["cleanup_incomplete"]:
                    status = "cleanup_incomplete"
                # A last valid read improves evidence after TERM/KILL; failures
                # stay failures even when the child has already exited.
                try:
                    if observe("final") > args.max_vram_bytes and reason is None:
                        reason, status = "max_vram_bytes", "limit_exceeded"
                except BaseException as error:
                    errors.append(f"final sample: {type(error).__name__}: {error}")
                    if reason is None:
                        reason, status = "source_error", "monitor_error"
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    command_code = process.returncode if process is not None else None
    if status == "completed":
        result = command_code if command_code >= 0 else 128 - command_code
    elif status == "interrupted":
        result = 128 + caught[0]
    else:
        result = 124 if reason == "wall_seconds" and status != "cleanup_incomplete" else 125
    emit("final", status=status, reason=reason, command_started=process is not None,
         command_returncode=command_code, runner_returncode=result, samples=samples,
         sampled_peak_global_vram_bytes=peak, last_global_vram_used_bytes=last,
         elapsed_seconds=time.monotonic() - started, errors=errors, **cleanup)
    return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sysfs-dir", type=Path, default=Path("/sys/class/drm/card1/device"))
    parser.add_argument("--max-vram-bytes", type=vram_bytes, default=MAX_VRAM_BYTES)
    parser.add_argument("--interval-seconds", type=capped_seconds(MAX_INTERVAL_SECONDS), default=0.1)
    parser.add_argument("--wall-seconds", type=capped_seconds(590), default=580.0)
    parser.add_argument("--term-grace-seconds", type=capped_seconds(1), default=1.0)
    parser.add_argument("--kill-wait-seconds", type=capped_seconds(1), default=1.0)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    else:
        parser.error("place command argv after --")
    if not args.command:
        parser.error("a command is required after --")
    if not sys.platform.startswith("linux"):
        parser.error("Linux sysfs is required")
    return args


def main(argv=None):
    args = arguments(argv)
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
        with log:
            def emit(event, **values):
                timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                print(json.dumps({"event": event, "time_utc": timestamp, **values},
                                 allow_nan=False), file=log, flush=True)
            return supervise(args, emit)
    except (OSError, MonitorError) as error:
        print(f"run_vulkan_smoke: {error}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    sys.exit(main())
