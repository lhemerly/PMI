"""One owner for the existing CPU and factory locks; this grants no authority.

Descriptors exported to the fixed supervisor share the same open file
descriptions. Closing this owner's copies must therefore never issue LOCK_UN:
the supervisor retains both locks until its independent cleanup is complete.
"""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat


CPU_LOCK = Path('/tmp/cpu-coordinator-6t3yfB/test.lock')
FACTORY_LEASE = Path('/home/luizh/Documents/Codex/2026-10-08/task-8/content-factory-shared.lease')


class CoordinationError(RuntimeError):
    """Ownership is absent, stale, or inconsistent; do not start more work."""


def _identity(info):
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise CoordinationError('coordination lock must be a regular single-link file')
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid)


def _flock_record(descriptor, owner_pid):
    # Unlike fstat alone, this also rejects a descriptor closed and reopened on
    # the same inode without the original open-file-description lock.
    with open('/proc/self/fdinfo/' + str(descriptor), 'rb') as source:
        raw = source.read(8193)
    if len(raw) > 8192:
        raise CoordinationError('oversized coordination descriptor evidence')
    records = [tuple(line.split()[2:]) for line in raw.splitlines()
               if line.startswith(b'lock:')]
    if (len(records) != 1 or len(records[0]) != 7
            or records[0][:3] != (b'FLOCK', b'ADVISORY', b'WRITE')
            or records[0][3] != str(owner_pid).encode()
            or records[0][-2:] != (b'0', b'EOF')):
        raise CoordinationError('owned exclusive flock evidence missing')
    return records[0]


class _OwnedHandle:
    """Borrowable descriptor view; callers cannot close it through this API."""

    def __init__(self, path, descriptor, identity):
        self.path = path
        self._descriptor = descriptor
        self._identity = identity
        self._owner_pid = os.getpid()
        self._record = _flock_record(descriptor, self._owner_pid)

    @property
    def closed(self):
        return self._descriptor is None

    def fileno(self):
        if self.closed:
            raise CoordinationError('coordination descriptor closed')
        return self._descriptor

    def validate(self):
        try:
            descriptor = self.fileno()
            if (_identity(os.fstat(descriptor)) != self._identity
                    or _identity(self.path.lstat()) != self._identity
                    or not fcntl.fcntl(descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
                    or _flock_record(descriptor, self._owner_pid) != self._record):
                raise CoordinationError('coordination descriptor or path changed')
        except OSError as error:
            raise CoordinationError('coordination descriptor or path unavailable') from error

    def _close_owned(self):
        descriptor = self._descriptor
        self._descriptor = None
        if descriptor is None:
            return
        try:
            current = os.fstat(descriptor)
        except OSError as error:
            raise CoordinationError('owned descriptor disappeared before close') from error
        if ((current.st_dev, current.st_ino) != self._identity[:2]
                or _flock_record(descriptor, self._owner_pid) != self._record):
            # The numeric FD has been reused. Do not close an unrelated handle.
            raise CoordinationError('refusing to close a replaced descriptor')
        os.close(descriptor)


def _acquire(path, *, create=False):
    path = Path(path)
    before = None
    try:
        before = _identity(path.lstat())
    except FileNotFoundError:
        if not create:
            raise
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    if create:
        flags |= os.O_CREAT
    descriptor = os.open(path, flags, 0o600)
    try:
        identity = _identity(os.fstat(descriptor))
        if ((before is not None and before != identity)
                or _identity(path.lstat()) != identity):
            raise CoordinationError('coordination path changed during open')
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle = _OwnedHandle(path, descriptor, identity)
        handle.validate()
        return handle
    except BaseException:
        os.close(descriptor)
        raise


class CoordinationOwner:
    """Single-use context owning the two fixed existing coordination locks."""

    def __init__(self):
        self._entered = False
        self._active = False
        self._pid = None
        self._parent_pid = None
        self._cpu = None
        self._factory = None

    def __enter__(self):
        if self._entered:
            raise CoordinationError('coordination owner cannot be reused')
        self._entered = True
        self._pid, self._parent_pid = os.getpid(), os.getppid()
        try:
            self._cpu = _acquire(CPU_LOCK)
            self._factory = _acquire(FACTORY_LEASE)
            self._active = True
            self._validate()
            return self
        except BaseException:
            self._active = False
            self._close_all()
            raise

    def _validate(self):
        if (not self._active or os.getpid() != self._pid
                or os.getppid() != self._parent_pid):
            raise CoordinationError('active coordination process ownership required')
        if self._cpu.path != CPU_LOCK or self._factory.path != FACTORY_LEASE:
            raise CoordinationError('fixed coordination paths changed')
        self._cpu.validate()
        self._factory.validate()

    def validate(self):
        """Validate current ownership for trusted callers; grant no authority."""
        self._validate()
        return self

    def _close_all(self):
        failure = None
        for handle in (self._factory, self._cpu):
            if handle is not None:
                try:
                    handle._close_owned()
                except BaseException as error:
                    failure = failure or error
        if failure is not None:
            raise failure

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._validate()
        finally:
            self._active = False
            self._close_all()
        return False

    @contextmanager
    def borrow_cpu(self):
        """Validate and lend the held CPU descriptor without acquiring again."""
        self._validate()
        try:
            yield self._cpu
        finally:
            self._validate()

    @contextmanager
    def ownership(self, store, lease):
        """Own one run writer while borrowing the already-held factory lease."""
        self._validate()
        if Path(lease) != FACTORY_LEASE:
            raise CoordinationError('exact fixed factory lease required')
        local = _acquire(Path(store.root) / 'research.lock', create=True)
        try:
            self._validate()
            yield
        finally:
            try:
                self._validate()
                local.validate()
            finally:
                local._close_owned()

    def export_fds(self):
        """Return CPU then factory FDs for the fixed supervisor's SCM_RIGHTS."""
        self._validate()
        return (self._cpu.fileno(), self._factory.fileno())
