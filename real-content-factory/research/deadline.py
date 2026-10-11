"""Reviewed research supervisors with absolute deadlines and retained coordination.

This module grants no launch authority. The public launcher first consumes the
separately reviewed authority and runs exactly one fixed child. The supervisor
must be a dedicated, single-threaded process: becoming a Linux child subreaper
makes every subsequently adopted direct child attributable to this invocation,
even when a native helper changes its session. Signals use verified pidfds.
"""
import array
import ctypes
import errno
import fcntl
import json
import math
import os
from pathlib import Path
import secrets
import selectors
import signal
import socket
import stat
import struct
import subprocess
import sys
import time

OPERATION_SECONDS = 2520
CLEANUP_SECONDS = 180
TOTAL_SECONDS = OPERATION_SECONDS + CLEANUP_SECONDS
OUTPUT_ROOT = Path('/tmp/pmi-model-demo-v4')
DEMO_CHILD_MODULE = 'research.model_demo_entry'
RUNTIME_CHILD_MODULE = 'research.runtime_entry'
CONTROL_FD = 3
STREAM_LIMIT = 1048576
PACKET_LIMIT = 8192
_CONTEXT_KEY = object()


def _process(pid):
    raw = Path('/proc/%d/stat' % pid).read_text()
    prefix, tail = raw.rsplit(')', 1)
    fields = tail.split()
    if int(prefix.split(' ', 1)[0]) != pid or len(fields) < 20:
        raise ValueError('process identity unreadable')
    return (pid, int(fields[19])), int(fields[1])


def _prctl(option, value):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(option, value, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'required process ownership prctl failed')


def _children():
    path = Path('/proc/self/task/%d/children' % os.getpid())
    raw = path.read_text()
    if len(raw) > 65536:
        raise ValueError('owned child inventory too large')
    result = [int(value) for value in raw.split()]
    if len(result) > 4096 or any(pid <= 1 for pid in result):
        raise ValueError('invalid owned child inventory')
    return result


def _encode(row):
    raw = json.dumps(row, separators=(',', ':'), allow_nan=False).encode()
    if len(raw) > PACKET_LIMIT:
        raise ValueError('supervisor packet too large')
    return raw


def _receive(channel, *, credentials=False):
    flags = getattr(socket, 'MSG_CMSG_CLOEXEC', 0)
    raw, ancillary, message_flags, _ = channel.recvmsg(
        PACKET_LIMIT, socket.CMSG_SPACE(8 * array.array('i').itemsize)
        + socket.CMSG_SPACE(struct.calcsize('3i')), flags)
    descriptors, peer = [], None
    try:
        # The kernel installs every SCM_RIGHTS descriptor before returning.
        # Collect all of them before any validation that could raise.
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                values = array.array('i')
                values.frombytes(data[:len(data) - len(data) % values.itemsize])
                descriptors.extend(values)
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                continue
            if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS:
                if peer is not None or len(data) != struct.calcsize('3i'):
                    raise ValueError('ambiguous supervisor peer credentials')
                peer = struct.unpack('3i', data)
            else:
                raise ValueError('unexpected supervisor ancillary data')
        if message_flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise ValueError('truncated supervisor packet')
        if not raw:
            raise EOFError('supervisor control closed')
        row = json.loads(raw)
        if type(row) is not dict or (credentials and peer is None):
            raise ValueError('supervisor packet credentials/object missing')
        for descriptor in descriptors:
            os.set_inheritable(descriptor, False)
        return row, descriptors, peer
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        raise


class SupervisorContext:
    """Process-local context minted only by the authenticated inherited socket."""

    def __init__(self, key, channel, row):
        if key is not _CONTEXT_KEY:
            raise ValueError('inherited supervisor context required')
        self._key, self._channel = key, channel
        self.operation_deadline = row['operation_deadline']
        self.final_deadline = row['final_deadline']
        self.supervisor_identity = tuple(row['supervisor_identity'])
        self.child_pid, self.output_root = row['child_pid'], row['output_root']
        self.child_module = row.get('child_module', DEMO_CHILD_MODULE)
        self.output_root_identity = tuple(row['output_root_identity']) if 'output_root_identity' in row else None
        self.nonce = row['nonce']
        self._identity = _process(os.getpid())[0]
        self._retained = False
        self._finished = False

    def validate(self):
        if (self._key is not _CONTEXT_KEY or self._channel is None
                or self.child_pid != os.getpid()
                or _process(os.getpid())[0] != self._identity
                or os.getppid() != self.supervisor_identity[0]
                or _process(os.getppid())[0] != self.supervisor_identity):
            raise ValueError('live supervisor process ownership lost')
        if time.monotonic() >= self.operation_deadline:
            raise TimeoutError('supervisor operation deadline reached')
        if self.output_root_identity is not None:
            info = _private_root(self.output_root).stat()
            if (info.st_dev, info.st_ino) != self.output_root_identity:
                raise ValueError('authenticated runtime output directory replaced')
        return self

    def retain_coordination(self, owner):
        from .coordination import CoordinationOwner
        self.validate()
        if self._retained or type(owner) is not CoordinationOwner:
            raise ValueError('one trusted coordination owner required')
        descriptors = array.array('i', owner.export_fds())
        self._channel.settimeout(min(5, self.operation_deadline - time.monotonic()))
        self._channel.sendmsg([_encode({'action': 'retain-coordination', 'nonce': self.nonce})],
                              [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptors)])
        row, extra, _ = _receive(self._channel)
        for descriptor in extra:
            os.close(descriptor)
        if extra or row != {'action': 'coordination-retained', 'nonce': self.nonce}:
            raise ValueError('supervisor did not retain both coordination locks')
        self.validate()
        owner.validate()
        self._retained = True

    def finish(self, result_status):
        self.validate()
        if self._finished or not self._retained or result_status not in {
                'finished', 'blocked', 'budget_exhausted', 'cancelled'}:
            raise ValueError('invalid/duplicate child final status')
        self._channel.sendall(_encode({'action': 'finished', 'nonce': self.nonce,
                                     'status': result_status}))
        self._finished = True

    def close(self):
        if self._channel is not None:
            self._channel.close()
            self._channel = None

    def __enter__(self):
        return self.validate()

    def __exit__(self, *_):
        self.close()


def _private_root(value):
    if not isinstance(value, (str, Path)):
        raise ValueError('absolute private supervisor root required')
    root = Path(value)
    if not root.is_absolute() or root.resolve(strict=True) != root or root.is_symlink():
        raise ValueError('canonical private supervisor root required')
    info = root.stat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or info.st_mode & 0o077 or info.st_mode & 0o700 != 0o700):
        raise ValueError('reserved private owned supervisor root required')
    return root


def _validate_context_row(row, parent_identity, now):
    """Validate an already authenticated packet; this alone grants no authority."""
    required = {'schema', 'operation_deadline', 'final_deadline',
                'supervisor_identity', 'child_pid', 'output_root', 'nonce'}
    runtime = row.get('schema') == 'PMI_RUNTIME_SUPERVISOR_V1'
    if runtime:
        required.update(('child_module', 'output_root_identity'))
    if (set(row) != required
            or row.get('schema') not in {'PMI_SUPERVISOR_V1', 'PMI_RUNTIME_SUPERVISOR_V1'}
            or row['supervisor_identity'] != list(parent_identity)
            or type(row['child_pid']) is not int or row['child_pid'] != os.getpid()
            or type(row['output_root']) is not str
            or type(row['nonce']) is not str or len(row['nonce']) != 64
            or any(c not in '0123456789abcdef' for c in row['nonce'])):
        raise ValueError('invalid authenticated supervisor context')
    if runtime:
        if row['child_module'] != RUNTIME_CHILD_MODULE:
            raise ValueError('fixed runtime child module required')
        info = _private_root(row['output_root']).stat()
        identity = row['output_root_identity']
        if (type(identity) is not list or len(identity) != 2
                or any(type(value) is not int or value < 0 for value in identity)
                or identity != [info.st_dev, info.st_ino]):
            raise ValueError('authenticated runtime output directory identity mismatch')
    elif row['output_root'] != str(OUTPUT_ROOT):
        raise ValueError('fixed demonstration output root required')
    operation, final = row['operation_deadline'], row['final_deadline']
    if (any(type(x) not in (int, float) or not math.isfinite(x) for x in (operation, final))
            or not now < operation < final
            or abs(final - operation - CLEANUP_SECONDS) > .001
            or (not runtime and operation > now + OPERATION_SECONDS)):
        raise ValueError('invalid absolute supervisor deadlines')


def read_supervisor_context():
    """Consume fixed inherited FD3; no environment or CLI authority inputs."""
    channel = socket.socket(fileno=CONTROL_FD)
    try:
        os.set_inheritable(CONTROL_FD, False)
        if (channel.family != socket.AF_UNIX
                or channel.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_SEQPACKET):
            raise ValueError('fixed sequenced supervisor socket required')
        peer = struct.unpack('3i', channel.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize('3i')))
        before = _process(os.getppid())[0]
        if peer[:2] != (before[0], os.getuid()):
            raise ValueError('supervisor parent credentials mismatch')
        _prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG, before any admitted work.
        if _process(os.getppid())[0] != before:
            raise ValueError('supervisor parent changed during setup')
        channel.settimeout(5)
        row, descriptors, _ = _receive(channel)
        for descriptor in descriptors:
            os.close(descriptor)
        if descriptors:
            raise ValueError('unexpected descriptors in supervisor context')
        _validate_context_row(row, before, time.monotonic())
        return SupervisorContext(_CONTEXT_KEY, channel, row).validate()
    except BaseException:
        channel.close()
        raise


def _validate_retained(descriptors, child_pid):
    from .coordination import CPU_LOCK, FACTORY_LEASE
    if len(descriptors) != 2:
        raise ValueError('both coordination descriptors required')
    for descriptor, path in zip(descriptors, (CPU_LOCK, FACTORY_LEASE)):
        opened, named = os.fstat(descriptor), path.lstat()
        if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                or not stat.S_ISREG(named.st_mode) or named.st_nlink != 1
                or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
                or not fcntl.fcntl(descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC):
            raise ValueError('received coordination descriptor/path mismatch')
        raw = Path('/proc/self/fdinfo/%d' % descriptor).read_text()
        records = [line.split()[2:] for line in raw.splitlines() if line.startswith('lock:')]
        if (len(records) != 1 or len(records[0]) != 7
                or records[0][:4] != ['FLOCK', 'ADVISORY', 'WRITE', str(child_pid)]
                or records[0][-2:] != ['0', 'EOF']):
            raise ValueError('received descriptor lacks child-owned exclusive flock')


class _OwnedProcesses:
    """Only direct/adopted children of this isolated subreaper may be signaled."""

    def __init__(self, root):
        self.root = root
        self.handles = {}
        self.errors = []

    def refresh(self):
        for pid in _children():
            if pid in self.handles:
                continue
            descriptor = None
            try:
                identity, parent = _process(pid)
                if parent != os.getpid():
                    raise ValueError('child inventory ownership changed')
                descriptor = os.pidfd_open(pid, 0)
                if _process(pid) != (identity, parent):
                    raise ValueError('child identity changed during pidfd acquisition')
                self.handles[pid] = (identity, descriptor)
                descriptor = None
            except (ProcessLookupError, FileNotFoundError):
                pass
            finally:
                if descriptor is not None:
                    os.close(descriptor)

    def signal_all(self, number):
        failures = []
        try:
            self.refresh()
        except (OSError, ValueError) as error:
            failures.append(str(error))
        if self.root.pid not in self.handles and self.root.returncode is None:
            # This Popen is our unreaped direct child, never an arbitrary PID.
            # With one thread and default SIGCHLD no other code can reap/reuse
            # it between Popen's poll and signal. A proc read failure must not
            # prevent stopping this already-proven owned root.
            try:
                self.root.send_signal(number)
            except ProcessLookupError:
                pass
            except OSError as error:
                failures.append(str(error))
        for identity, descriptor in list(self.handles.values()):
            try:
                signal.pidfd_send_signal(descriptor, number)
            except ProcessLookupError:
                pass
            except OSError as error:
                failures.append(str(error))
        if failures:
            raise ValueError('owned signaling uncertainty: ' + '; '.join(failures))

    def reap(self):
        self.root.poll()
        for pid, (_, descriptor) in list(self.handles.items()):
            if pid == self.root.pid:
                done = self.root.returncode is not None
            else:
                try:
                    done = os.waitpid(pid, os.WNOHANG)[0] == pid
                except ChildProcessError:
                    # No other thread/handler is allowed to reap our children.
                    raise ValueError('owned child unexpectedly not waitable')
            if done:
                os.close(descriptor)
                del self.handles[pid]

    def empty(self):
        self.refresh()
        self.reap()
        return self.root.returncode is not None and not self.handles and not _children()

    def close(self):
        for _, descriptor in self.handles.values():
            os.close(descriptor)
        self.handles.clear()


def _cleanup_owned(owned, final_deadline):
    """No per-process timeout can extend the single absolute cleanup deadline."""
    errors = []
    started = time.monotonic()
    while time.monotonic() < final_deadline:
        try:
            owned.signal_all(signal.SIGTERM if time.monotonic() - started < 2 else signal.SIGKILL)
            if owned.empty():
                return True, errors
        except (OSError, ValueError) as error:
            detail = type(error).__name__ + ': ' + str(error)
            if detail not in errors:
                errors.append(detail)
        time.sleep(min(.05, max(0, final_deadline - time.monotonic())))
    return False, errors


def _pump(root, channel, operation_deadline, final_deadline, nonce, stop_requested,
          initial_error=None, capture_dir=None, capture_directory_fd=None):
    """Bounded controller; private injection point for harmless subprocess tests."""
    selector = selectors.DefaultSelector()
    retained, final_status, stop_reason = [], None, None
    counts = {'stdout': 0, 'stderr': 0}
    captures = {name: bytearray() for name in counts}
    errors, cleanup_verified = [], False
    owned = _OwnedProcesses(root)
    stop_started = None
    try:
        if initial_error is not None:
            raise RuntimeError(initial_error)
        channel.setblocking(False)
        selector.register(channel, selectors.EVENT_READ, 'control')
        for name in counts:
            stream = getattr(root, name)
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while time.monotonic() < final_deadline:
            now = time.monotonic()
            root.poll()
            external_stop = stop_requested()
            if stop_reason is None and external_stop:
                stop_reason = external_stop
            if stop_reason is None and now >= operation_deadline:
                stop_reason = 'operation_deadline'
            if stop_reason is None and root.returncode is not None:
                if _children():
                    errors.append('child exited with unexpected owned descendants')
                    stop_reason = 'unexpected_descendants'
                else:
                    stop_reason = 'child_exit'
            if stop_reason is not None:
                if stop_started is None:
                    stop_started = now
                    owned.signal_all(signal.SIGTERM)
                if now - stop_started >= 2:
                    owned.signal_all(signal.SIGKILL)
                if owned.empty():
                    cleanup_verified = True
                    # Drain already queued final control/stream packets below.
            else:
                owned.refresh()
                owned.reap()
            events = selector.select(min(.05, max(0, final_deadline - time.monotonic())))
            for key, _ in events:
                if key.data == 'control':
                    descriptors = []
                    try:
                        row, descriptors, peer = _receive(channel, credentials=True)
                        if peer[:2] != (root.pid, os.getuid()) or row.get('nonce') != nonce:
                            raise ValueError('unauthenticated child control packet')
                        if row.get('action') == 'retain-coordination':
                            if retained or set(row) != {'action', 'nonce'}:
                                raise ValueError('duplicate/invalid coordination transfer')
                            _validate_retained(descriptors, root.pid)
                            retained, descriptors = descriptors, []
                            channel.sendall(_encode({'action': 'coordination-retained', 'nonce': nonce}))
                        elif row.get('action') == 'finished':
                            if (descriptors or not retained or final_status is not None
                                    or set(row) != {'action', 'nonce', 'status'}
                                    or row['status'] not in {'finished', 'blocked', 'budget_exhausted', 'cancelled'}):
                                raise ValueError('invalid child final status')
                            final_status = row['status']
                        else:
                            raise ValueError('unexpected child control action')
                    except BlockingIOError:
                        pass
                    except EOFError:
                        selector.unregister(channel)
                        if final_status is None:
                            stop_reason = stop_reason or 'control_closed_without_final'
                    except (OSError, ValueError, TypeError) as error:
                        errors.append(type(error).__name__ + ': ' + str(error))
                        stop_reason = stop_reason or 'control_failure'
                        selector.unregister(channel)
                    finally:
                        for descriptor in descriptors:
                            os.close(descriptor)
                else:
                    raw = os.read(key.fileobj.fileno(), 65536)
                    if not raw:
                        selector.unregister(key.fileobj)
                    counts[key.data] += len(raw)
                    captures[key.data].extend(raw[:max(0, STREAM_LIMIT - len(captures[key.data]))])
                    if counts[key.data] > STREAM_LIMIT:
                        stop_reason = stop_reason or 'stream_limit'
            if cleanup_verified:
                # All writers gone; drain at most the bounded pipe contents.
                if not events or not selector.get_map():
                    break
    except BaseException as error:
        errors.append(type(error).__name__ + ': ' + str(error))
        stop_reason = stop_reason or 'supervisor_failure'
    finally:
        if not cleanup_verified:
            cleanup_verified, cleanup_errors = _cleanup_owned(owned, final_deadline)
            errors.extend(cleanup_errors)
        if not cleanup_verified:
            errors.append('owned cleanup not proven before absolute final deadline')
        selector.close()
        remaining = [list(row[0]) for row in owned.handles.values()]
        owned.close()
        # Close-only is essential: explicit LOCK_UN would unlock the child's FD.
        for descriptor in retained:
            os.close(descriptor)
        if capture_dir is not None or capture_directory_fd is not None:
            for name, raw in captures.items():
                try:
                    basename = 'controller.' + name + '.log'
                    target = basename if capture_directory_fd is not None else Path(capture_dir) / basename
                    descriptor = os.open(target,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                        0o600, dir_fd=capture_directory_fd)
                    with os.fdopen(descriptor, 'wb') as output:
                        output.write(raw)
                        output.flush()
                        os.fsync(output.fileno())
                except OSError as error:
                    errors.append('capture persistence failed: ' + str(error))
    return {'schema': 'PMI_DEMO_SUPERVISOR_V1', 'operation_deadline': operation_deadline,
                'final_deadline': final_deadline, 'stop_reason': stop_reason,
                'child_pid': root.pid, 'returncode': root.returncode,
                'child_final_status': final_status, 'coordination_retained': bool(retained),
                'coordination_release': 'verified_empty' if cleanup_verified else 'uncertain_at_final_deadline',
                'cleanup_verified': cleanup_verified, 'errors': errors,
                'remaining_owned': remaining,
                'stream_bytes': counts,
                'completed': cleanup_verified and not errors and bool(retained)
                and final_status == 'finished' and root.returncode == 0 and stop_reason == 'child_exit'}


def _child_setup(descriptor):
    if descriptor != CONTROL_FD:
        os.dup2(descriptor, CONTROL_FD, inheritable=True)
        os.close(descriptor)


def _supervisor_scope(authority, started):
    """Snapshot fields only AFTER the caller's process-local authority gate."""
    from .demo_authority import _ResearchDemo
    from .run_authority import ResearchAuthority
    final = authority.deadline
    if type(final) not in (int, float) or not math.isfinite(final):
        raise ValueError('finite authorized supervisor deadline required')
    if type(authority) is ResearchAuthority:
        operation = authority.operation_deadline
        if authority.child_module != RUNTIME_CHILD_MODULE:
            raise ValueError('fixed runtime child module required')
        if (type(operation) not in (int, float) or not math.isfinite(operation)
                or not started < operation < final
                or abs(final - operation - CLEANUP_SECONDS) > .001):
            raise ValueError('finite runtime operation and180s cleanup required')
        return (_private_root(authority.root), RUNTIME_CHILD_MODULE,
                operation, final, 'PMI_RUNTIME_SUPERVISOR_V1')
    if type(authority) is not _ResearchDemo:
        raise ValueError('trusted process-local supervisor authority required')
    # The legacy scope retains its original already-running absolute deadline.
    if final <= started + CLEANUP_SECONDS:
        raise TimeoutError('supervisor operation budget unavailable')
    final = min(started + TOTAL_SECONDS, final)
    return (_private_root(OUTPUT_ROOT), DEMO_CHILD_MODULE, final - CLEANUP_SECONDS,
            final, 'PMI_SUPERVISOR_V1')


def run_demo_supervisor(authority=None):
    """Run the authority's fixed reviewed entry; no configurable command API."""
    from .demo_authority import gate_supervisor
    gate_supervisor(authority)
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Linux pidfd support required')
    if len(list(Path('/proc/self/task').iterdir())) != 1 or _children():
        raise RuntimeError('dedicated single-threaded child-free supervisor required')
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise RuntimeError('default SIGCHLD ownership required')
    started = time.monotonic()
    output_root, child_module, operation_deadline, final_deadline, schema = _supervisor_scope(authority, started)
    for name in ('supervisor-report.json', 'controller.stdout.log', 'controller.stderr.log'):
        target = output_root / name
        if target.exists() or target.is_symlink():
            raise ValueError('supervisor attempt output already exists; use a fresh reviewed attempt')
    parent_identity = _process(os.getppid())[0]
    supervisor_identity = _process(os.getpid())[0]
    stopped = [None]
    prior_handlers = {}
    def requested():
        if stopped[0] is None:
            try:
                if os.getppid() != parent_identity[0] or _process(os.getppid())[0] != parent_identity:
                    stopped[0] = 'supervisor_parent_lost'
            except (OSError, ValueError):
                stopped[0] = 'supervisor_parent_lost'
        return stopped[0]
    def stop(number, _):
        stopped[0] = 'signal_' + str(number)
    channel, child_channel = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    root = None
    output_descriptor = None
    try:
        output_descriptor = os.open(output_root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
        opened, named = os.fstat(output_descriptor), _private_root(output_root).stat()
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise ValueError('supervisor output directory changed during open')
        for number in (signal.SIGTERM, signal.SIGINT):
            prior_handlers[number] = signal.signal(number, stop)
        _prctl(36, 1)  # PR_SET_CHILD_SUBREAPER
        _prctl(1, signal.SIGTERM)
        if requested():
            raise RuntimeError('supervisor parent lost before launch')
        channel.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        nonce = secrets.token_hex(32)
        child_fd = child_channel.fileno()
        root = subprocess.Popen(
            [sys.executable, '-B', '-m', child_module, '--supervised-child'],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            close_fds=True, pass_fds=tuple(sorted({CONTROL_FD, child_fd})),
            preexec_fn=lambda: _child_setup(child_fd), start_new_session=True,
            env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1'},
            cwd=Path(__file__).resolve().parents[1])
        child_channel.close()
        initial_error = None
        try:
            channel.settimeout(min(5, max(.001, operation_deadline - time.monotonic())))
            packet = {'schema': schema,
                'operation_deadline': operation_deadline, 'final_deadline': final_deadline,
                'supervisor_identity': list(supervisor_identity), 'child_pid': root.pid,
                'output_root': str(output_root), 'nonce': nonce}
            if child_module == RUNTIME_CHILD_MODULE:
                packet['child_module'] = child_module
                packet['output_root_identity'] = [opened.st_dev, opened.st_ino]
            channel.sendall(_encode(packet))
        except BaseException as error:
            initial_error = type(error).__name__ + ': ' + str(error)
        report = _pump(root, channel, operation_deadline, final_deadline, nonce, requested,
                       initial_error, capture_directory_fd=output_descriptor)
        report['supervisor_identity'] = list(supervisor_identity)
        report['output_root'] = str(output_root)
        report['child_module'] = child_module
        report['elapsed_seconds'] = time.monotonic() - started
        fd = os.open('supervisor-report.json',
                     os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600, dir_fd=output_descriptor)
        with os.fdopen(fd, 'w') as output:
            json.dump(report, output, sort_keys=True, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.fsync(output_descriptor)
        return report
    finally:
        child_channel.close()
        channel.close()
        if output_descriptor is not None:
            os.close(output_descriptor)
        if root is not None:
            if root.stdout is not None:
                root.stdout.close()
            if root.stderr is not None:
                root.stderr.close()
        for number, handler in prior_handlers.items():
            signal.signal(number, handler)
