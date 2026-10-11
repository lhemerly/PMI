"""Temporary-file lock fixtures only; no subprocess or execution authority."""
import fcntl
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research import coordination


class CoordinationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cpu = self.root / 'cpu.lock'
        self.factory = self.root / 'factory.lease'
        self.cpu.touch(mode=0o600)
        self.factory.touch(mode=0o600)
        self.store = SimpleNamespace(root=self.root / 'run')
        self.store.root.mkdir()
        for name, value in (('CPU_LOCK', self.cpu), ('FACTORY_LEASE', self.factory)):
            replacement = patch.object(coordination, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)

    def assert_locked(self, path):
        with path.open('rb') as contender:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def assert_released(self, path):
        with path.open('rb') as contender:
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_owner_acquires_both_and_borrow_never_reacquires_or_closes(self):
        with coordination.CoordinationOwner() as owner:
            self.assertIs(owner.validate(), owner)
            self.assert_locked(self.cpu)
            self.assert_locked(self.factory)
            with patch.object(coordination.fcntl, 'flock', wraps=fcntl.flock) as flock:
                with owner.borrow_cpu() as first:
                    with owner.borrow_cpu() as second:
                        self.assertIs(first, second)
                        self.assertEqual(first.fileno(), owner.export_fds()[0])
                flock.assert_not_called()
            self.assertFalse(first.closed)
            self.assert_locked(self.cpu)
        self.assertTrue(first.closed)
        self.assert_released(self.cpu)
        self.assert_released(self.factory)

    def test_exported_duplicates_keep_both_locks_after_owner_exit(self):
        duplicates = []
        try:
            with coordination.CoordinationOwner() as owner:
                original = owner.export_fds()
                self.assertTrue(all(not os.get_inheritable(fd) for fd in original))
                duplicates = [os.dup(fd) for fd in original]
            self.assert_locked(self.cpu)
            self.assert_locked(self.factory)
        finally:
            for descriptor in duplicates:
                os.close(descriptor)
        self.assert_released(self.cpu)
        self.assert_released(self.factory)

    def test_context_exit_only_closes_and_never_explicitly_unlocks(self):
        calls = []
        real_flock = fcntl.flock
        def observed(descriptor, operation):
            calls.append(operation)
            return real_flock(descriptor, operation)
        with patch.object(coordination.fcntl, 'flock', side_effect=observed):
            with coordination.CoordinationOwner():
                pass
        self.assertEqual(calls, [fcntl.LOCK_EX | fcntl.LOCK_NB] * 2)

    def test_factory_contention_releases_partial_cpu_acquisition(self):
        owner = coordination.CoordinationOwner()
        with self.factory.open('rb') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                owner.__enter__()
            self.assert_released(self.cpu)
            self.assert_locked(self.factory)
        with self.assertRaises(coordination.CoordinationError):
            owner.__enter__()

    def test_missing_fixed_file_is_not_created(self):
        self.factory.unlink()
        with self.assertRaises(FileNotFoundError):
            with coordination.CoordinationOwner():
                self.fail('missing fixed lease accepted')
        self.assertFalse(self.factory.exists())
        self.assert_released(self.cpu)

    def test_symlink_hardlink_and_nonregular_fixed_files_rejected(self):
        for kind in ('symlink', 'hardlink', 'directory', 'fifo'):
            with self.subTest(kind=kind):
                original = self.root / ('original-' + kind)
                self.cpu.rename(original)
                if kind == 'symlink':
                    self.cpu.symlink_to(original)
                elif kind == 'hardlink':
                    os.link(original, self.cpu)
                elif kind == 'directory':
                    self.cpu.mkdir()
                else:
                    os.mkfifo(self.cpu)
                try:
                    with self.assertRaises(coordination.CoordinationError):
                        with coordination.CoordinationOwner():
                            self.fail('unsafe fixed lock accepted')
                finally:
                    if kind == 'directory':
                        self.cpu.rmdir()
                    else:
                        self.cpu.unlink()
                    original.rename(self.cpu)

    def test_methods_require_active_owner_and_context_is_single_use(self):
        owner = coordination.CoordinationOwner()
        for phase in ('before', 'after'):
            with self.subTest(phase=phase):
                with self.assertRaises(coordination.CoordinationError):
                    owner.validate()
                with self.assertRaises(coordination.CoordinationError):
                    owner.export_fds()
                with self.assertRaises(coordination.CoordinationError):
                    with owner.borrow_cpu():
                        self.fail('inactive borrow accepted')
                with self.assertRaises(coordination.CoordinationError):
                    with owner.ownership(self.store, self.factory):
                        self.fail('inactive run ownership accepted')
            if phase == 'before':
                with owner:
                    with self.assertRaises(coordination.CoordinationError):
                        owner.__enter__()
        with self.assertRaises(coordination.CoordinationError):
            owner.__enter__()

    def test_pid_and_parent_changes_fail_without_releasing_active_locks(self):
        with coordination.CoordinationOwner() as owner:
            for name in ('getpid', 'getppid'):
                with self.subTest(name=name):
                    actual = getattr(os, name)()
                    with patch.object(coordination.os, name, return_value=actual + 100000):
                        with self.assertRaises(coordination.CoordinationError):
                            owner.export_fds()
                self.assert_locked(self.cpu)
                self.assert_locked(self.factory)

    def test_path_inode_replacement_fails_and_original_handle_is_closed(self):
        owner = coordination.CoordinationOwner().__enter__()
        original = self.root / 'retained-cpu'
        self.cpu.rename(original)
        self.cpu.touch(mode=0o600)
        with self.assertRaises(coordination.CoordinationError):
            owner.export_fds()
        with self.assertRaises(coordination.CoordinationError):
            owner.__exit__(None, None, None)
        self.assert_released(original)
        self.assert_released(self.factory)

    def test_fixed_file_replaced_between_lstat_and_open_is_rejected(self):
        original = self.root / 'replaced-during-open'
        real_open = os.open
        def replacing(path, flags, *args, **kwargs):
            if Path(path) == self.cpu:
                self.cpu.rename(original)
                created = real_open(self.cpu, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
                os.close(created)
            return real_open(path, flags, *args, **kwargs)
        with patch.object(coordination.os, 'open', side_effect=replacing):
            with self.assertRaises(coordination.CoordinationError):
                with coordination.CoordinationOwner():
                    self.fail('changed pre-open identity accepted')
        self.assert_released(self.cpu)
        self.assert_released(original)

    def test_changed_constants_or_cloexec_flags_fail_validation(self):
        with coordination.CoordinationOwner() as owner:
            with patch.object(coordination, 'CPU_LOCK', self.root / 'different'):
                with self.assertRaises(coordination.CoordinationError):
                    owner.export_fds()
            descriptor = owner.export_fds()[0]
            os.set_inheritable(descriptor, True)
            try:
                with self.assertRaises(coordination.CoordinationError):
                    owner.export_fds()
            finally:
                os.set_inheritable(descriptor, False)

    def test_same_inode_reopened_descriptor_lacks_owned_flock_evidence(self):
        owner = coordination.CoordinationOwner().__enter__()
        descriptor = owner.export_fds()[0]
        retained = os.dup(descriptor)
        replacement = os.open(self.cpu, os.O_RDWR | os.O_CLOEXEC)
        try:
            os.dup2(replacement, descriptor, inheritable=False)
            os.close(replacement)
            replacement = None
            with self.assertRaises(coordination.CoordinationError):
                owner.export_fds()
            with self.assertRaises(coordination.CoordinationError):
                owner.__exit__(None, None, None)
            self.assertEqual(os.fstat(descriptor).st_ino, self.cpu.stat().st_ino)
            self.assert_locked(self.cpu)
        finally:
            if replacement is not None:
                os.close(replacement)
            os.close(descriptor)
            os.close(retained)
        self.assert_released(self.cpu)
        self.assert_released(self.factory)

    def test_reused_numeric_fd_is_detected_and_unrelated_file_is_not_closed(self):
        owner = coordination.CoordinationOwner().__enter__()
        descriptor = owner.export_fds()[0]
        retained = os.dup(descriptor)
        other = self.root / 'unrelated'
        other.touch(mode=0o600)
        replacement = os.open(other, os.O_RDWR | os.O_CLOEXEC)
        try:
            os.dup2(replacement, descriptor, inheritable=False)
            os.close(replacement)
            replacement = None
            with self.assertRaises(coordination.CoordinationError):
                owner.export_fds()
            with self.assertRaises(coordination.CoordinationError):
                owner.__exit__(None, None, None)
            self.assertEqual(os.fstat(descriptor).st_ino, other.stat().st_ino)
            self.assert_released(self.factory)
        finally:
            if replacement is not None:
                os.close(replacement)
            os.close(descriptor)
            os.close(retained)
        self.assert_released(self.cpu)

    def test_borrow_exit_revalidates_held_lock(self):
        with coordination.CoordinationOwner() as owner:
            with self.assertRaises(coordination.CoordinationError):
                with owner.borrow_cpu() as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            # Restore the fixture so the owner can close normally. Production
            # borrowers have no unlock API and must never perform this mutation.
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_run_ownership_borrows_factory_and_exclusively_locks_local_run(self):
        with coordination.CoordinationOwner() as owner:
            with owner.ownership(self.store, self.factory):
                self.assert_locked(self.store.root / 'research.lock')
                self.assert_locked(self.factory)
                with self.assertRaises(BlockingIOError):
                    with owner.ownership(self.store, self.factory):
                        self.fail('duplicate run writer accepted')
            self.assert_released(self.store.root / 'research.lock')
            self.assert_locked(self.factory)
            with owner.borrow_cpu():
                pass

    def test_run_ownership_rejects_other_lease_and_unsafe_local_lock(self):
        with coordination.CoordinationOwner() as owner:
            with self.assertRaises(coordination.CoordinationError):
                with owner.ownership(self.store, self.root / 'other.lease'):
                    self.fail('alternate factory lease accepted')
            local = self.store.root / 'research.lock'
            self.assertFalse(local.exists())
            local.symlink_to(self.factory)
            with self.assertRaises(coordination.CoordinationError):
                with owner.ownership(self.store, self.factory):
                    self.fail('unsafe local lock accepted')
            self.assert_locked(self.factory)

    def test_body_failure_closes_local_but_keeps_owner_locks(self):
        with coordination.CoordinationOwner() as owner:
            with self.assertRaisesRegex(ValueError, 'fixture'):
                with owner.ownership(self.store, self.factory):
                    raise ValueError('fixture failure')
            self.assert_released(self.store.root / 'research.lock')
            self.assert_locked(self.cpu)
            self.assert_locked(self.factory)

    def test_cpu_admission_borrows_owner_but_samples_fresh_on_every_qualification(self):
        from research.owned_cpu import CPUAdmission
        now = [0.]
        authority = object()
        plan = SimpleNamespace(wall_seconds=10)
        def sample():
            return {'time': now[0], 'available': 17 * 1024 ** 3,
                    'some': 0, 'full': 0, 'swapout': 0, 'swap_used': 0}
        def advance(seconds):
            now[0] += seconds
        with coordination.CoordinationOwner() as owner:
            holder = CPUAdmission(authority=authority, coordination=owner)
            with (patch('research.qualification_authority.gate_cpu') as gate,
                  patch('research.owned_cpu.LOCK') as path,
                  patch('research.owned_cpu.fcntl.flock') as flock,
                  patch('research.owned_cpu.available', side_effect=sample) as available,
                  patch('research.owned_cpu.time.monotonic', side_effect=lambda: now[0]),
                  patch('research.owned_cpu.time.sleep', side_effect=advance)):
                first = holder.qualify(plan, 100)
                borrowed = holder.lock
                self.assertEqual(borrowed.fileno(), owner.export_fds()[0])
                holder.release()
                self.assertIsNone(holder.lock)
                self.assertFalse(borrowed.closed)
                first_reads = available.call_count
                second = holder.qualify(plan, 100)
                self.assertIs(holder.lock, borrowed)
                holder.release()
                self.assertGreater(available.call_count, first_reads)
                self.assertEqual(gate.call_count, 2)
                gate.assert_called_with(authority, 'admission', plan=plan)
                path.open.assert_not_called()
                flock.assert_not_called()
            self.assertTrue(first['approved'] and first['exclusive'])
            self.assertTrue(second['approved'] and second['exclusive'])
            self.assertEqual([row['time'] for row in first['samples']],
                             [index * .25 for index in range(9)])
            self.assertEqual([row['time'] for row in second['samples']],
                             [2 + index * .25 for index in range(9)])
            self.assertEqual((first['expires_at'], second['expires_at']), (3., 5.))
            self.assert_locked(self.cpu)
            self.assert_locked(self.factory)

    def test_cpu_admission_gate_refusal_neither_borrows_nor_samples(self):
        from research.owned_cpu import CPUAdmission
        authority = object()
        plan = SimpleNamespace(wall_seconds=10)
        with coordination.CoordinationOwner() as owner:
            holder = CPUAdmission(authority=authority, coordination=owner)
            with (patch('research.qualification_authority.gate_cpu',
                        side_effect=RuntimeError('fixture authority held')) as gate,
                  patch.object(owner, 'borrow_cpu') as borrow,
                  patch('research.owned_cpu.available') as available,
                  patch('research.owned_cpu.LOCK') as path):
                with self.assertRaisesRegex(RuntimeError, 'fixture authority held'):
                    holder.qualify(plan, 100)
                gate.assert_called_once_with(authority, 'admission', plan=plan)
                borrow.assert_not_called()
                available.assert_not_called()
                path.open.assert_not_called()
            self.assertIsNone(holder.lock)
            self.assertIsNone(holder._borrow)
            self.assert_locked(self.cpu)
            self.assert_locked(self.factory)

    def test_cpu_admission_sample_failure_releases_only_the_borrow(self):
        from research.owned_cpu import CPUAdmission
        now = [0.]
        def sample():
            return {'time': now[0], 'available': 8 * 1024 ** 3,
                    'some': 0, 'full': 0, 'swapout': 0, 'swap_used': 0}
        with coordination.CoordinationOwner() as owner:
            holder = CPUAdmission(coordination=owner)
            with (patch('research.qualification_authority.gate_cpu'),
                  patch('research.owned_cpu.available', side_effect=sample),
                  patch('research.owned_cpu.time.monotonic', side_effect=lambda: now[0]),
                  patch('research.owned_cpu.time.sleep',
                        side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds))):
                with self.assertRaisesRegex(ValueError, 'RAM below'):
                    holder.qualify(SimpleNamespace(wall_seconds=10), 100)
            self.assertIsNone(holder.lock)
            self.assertIsNone(holder._borrow)
            owner.validate()
            self.assert_locked(self.cpu)
            self.assert_locked(self.factory)


if __name__ == '__main__':
    unittest.main()
