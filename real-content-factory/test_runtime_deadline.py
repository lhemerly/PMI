"""Model-free authority/IPC scope checks; public Popen is mocked throughout."""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research import deadline as d
from research.demo_authority import _ResearchDemo


class FixtureAuthority:
    pass


class RuntimeDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.module = patch.dict(sys.modules, {
            'research.run_authority': SimpleNamespace(ResearchAuthority=FixtureAuthority)})
        self.module.start()
        self.addCleanup(self.module.stop)

    def authority(self, **changes):
        value = FixtureAuthority()
        value.root = self.root
        value.child_module = d.RUNTIME_CHILD_MODULE
        value.operation_deadline = 10000
        value.deadline = 10180
        for key, field in changes.items():
            setattr(value, key, field)
        return value

    def row(self, **changes):
        info = self.root.stat()
        value = {'schema': 'PMI_RUNTIME_SUPERVISOR_V1',
                 'operation_deadline': 10000, 'final_deadline': 10180,
                 'supervisor_identity': [101, 202], 'child_pid': os.getpid(),
                 'output_root': str(self.root), 'nonce': 'b' * 64,
                 'child_module': d.RUNTIME_CHILD_MODULE,
                 'output_root_identity': [info.st_dev, info.st_ino]}
        value.update(changes)
        return value

    def test_runtime_uses_exact_authorized_absolute_deadlines(self):
        scope = d._supervisor_scope(self.authority(), 50)
        self.assertEqual(scope, (self.root, d.RUNTIME_CHILD_MODULE, 10000, 10180,
                                 'PMI_RUNTIME_SUPERVISOR_V1'))
        # A later entry cannot reset the operation clock.
        self.assertEqual(d._supervisor_scope(self.authority(), 5000)[2:4], (10000, 10180))

    def test_runtime_expiration_and_mismatched_cleanup_refused(self):
        for operation, final in [(50, 230), (10000, 10179), (10000, 10181), (10180, 10000)]:
            with self.subTest(operation=operation, final=final), self.assertRaises(ValueError):
                d._supervisor_scope(self.authority(operation_deadline=operation, deadline=final), 50)

    def test_deadlines_must_be_finite_numbers_not_booleans(self):
        for field in ('deadline', 'operation_deadline'):
            for bad in (True, False, None, '10000', float('nan'), float('inf')):
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    d._supervisor_scope(self.authority(**{field: bad}), 50)

    def test_module_substitution_is_refused_without_demo_fallback(self):
        for module in (d.DEMO_CHILD_MODULE, 'os', '-c', None):
            with self.subTest(module=module), self.assertRaises(ValueError):
                d._supervisor_scope(self.authority(child_module=module), 50)

    def test_forged_and_subclassed_authority_are_not_runtime_authority(self):
        forged = SimpleNamespace(**vars(self.authority()))
        class Derived(FixtureAuthority):
            pass
        derived = Derived()
        derived.__dict__.update(vars(self.authority()))
        for value in (forged, derived):
            with self.assertRaises(ValueError):
                d._supervisor_scope(value, 50)

    def test_legacy_preserves_original_deadline_and_demo_module(self):
        authority = _ResearchDemo('supervisor', 1000)
        with patch.object(d, 'OUTPUT_ROOT', self.root):
            scope = d._supervisor_scope(authority, 50)
            self.assertEqual(scope, (self.root, d.DEMO_CHILD_MODULE, 820, 1000, 'PMI_SUPERVISOR_V1'))
            authority.deadline = 10000
            self.assertEqual(d._supervisor_scope(authority, 50)[2:4], (2570, 2750))
            authority.deadline = 230
            with self.assertRaises(TimeoutError):
                d._supervisor_scope(authority, 50)

    def test_runtime_packet_allows_larger_finite_budget_and_its_own_root(self):
        d._validate_context_row(self.row(), (101, 202), 50)
        wrong_mode = self.row(schema='PMI_SUPERVISOR_V1')
        with self.assertRaises(ValueError):
            d._validate_context_row(wrong_mode, (101, 202), 50)

    def test_runtime_packet_requires_exact_mode_module_identity_and_fields(self):
        invalid = [self.row(schema='other'), self.row(child_module=d.DEMO_CHILD_MODULE),
                   self.row(output_root_identity=[0, 0]), self.row(extra='ignored'),
                   self.row(supervisor_identity=[101, 999]), self.row(child_pid=True)]
        for row in invalid:
            with self.assertRaises(ValueError):
                d._validate_context_row(row, (101, 202), 50)

    def test_runtime_packet_requires_exact_cleanup_and_finite_future_clock(self):
        for changes in ({'operation_deadline': 50, 'final_deadline': 230},
                        {'operation_deadline': True}, {'final_deadline': float('inf')},
                        {'final_deadline': 10360}):
            with self.assertRaises(ValueError):
                d._validate_context_row(self.row(**changes), (101, 202), 50)

    def test_context_rejects_directory_replacement(self):
        now = time.monotonic()
        row = self.row(supervisor_identity=list(d._process(os.getppid())[0]),
                       operation_deadline=now + 10, final_deadline=now + 190)
        context = d.SupervisorContext(d._CONTEXT_KEY, Mock(), row)
        self.assertIs(context.validate(), context)
        moved = self.root.with_name(self.root.name + '-moved')
        self.root.rename(moved)
        self.root.mkdir(mode=0o700)
        try:
            with self.assertRaisesRegex(ValueError, 'replaced'):
                context.validate()
        finally:
            self.root.rmdir()
            moved.rename(self.root)

    def test_private_root_rejects_relative_symlink_and_shared_permissions(self):
        with self.assertRaises(ValueError):
            d._private_root(Path('relative'))
        link = self.root / 'link'
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            d._private_root(link)
        self.root.chmod(0o755)
        try:
            with self.assertRaises(ValueError):
                d._private_root(self.root)
        finally:
            self.root.chmod(0o700)

    def test_capture_directory_fd_cannot_be_redirected_by_path_replacement(self):
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        moved = self.root.with_name(self.root.name + '-moved')
        self.root.rename(moved)
        self.root.mkdir(mode=0o700)
        owned = Mock(handles={})
        try:
            with patch.object(d, '_OwnedProcesses', return_value=owned), \
                    patch.object(d, '_cleanup_owned', return_value=(True, [])):
                report = d._pump(SimpleNamespace(pid=1, returncode=0), Mock(), 1, 2, 'c' * 64,
                                 lambda: None, initial_error='test-only failure',
                                 capture_directory_fd=descriptor)
            self.assertFalse(report['completed'])
            for name in ('controller.stdout.log', 'controller.stderr.log'):
                self.assertTrue((moved / name).is_file())
                self.assertFalse((self.root / name).exists())
        finally:
            os.close(descriptor)
            self.root.rmdir()
            moved.rename(self.root)

    def test_gate_refusal_precedes_properties_and_spawn(self):
        with patch('research.demo_authority.gate_supervisor', side_effect=ValueError('held')), \
                patch.object(d.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(ValueError, 'held'):
                d.run_demo_supervisor(object())
        popen.assert_not_called()

    def test_public_launcher_has_fixed_argv_authenticated_packet_and_durable_attempt_report(self):
        now = time.monotonic()
        authority = self.authority(operation_deadline=now + 5000, deadline=now + 5180)
        parent, child = Mock(), Mock()
        child.fileno.return_value = 99
        root = Mock(pid=222)
        report = {'completed': True, 'cleanup_verified': True}
        with patch('research.demo_authority.gate_supervisor') as gate, \
                patch.object(d, '_children', return_value=[]), \
                patch.object(d, '_prctl'), patch.object(d.signal, 'signal'), \
                patch.object(d.signal, 'getsignal', return_value=signal.SIG_DFL), \
                patch.object(d.socket, 'socketpair', return_value=(parent, child)), \
                patch.object(d.subprocess, 'Popen', return_value=root) as popen, \
                patch.object(d, '_pump', return_value=report) as pump:
            result = d.run_demo_supervisor(authority)
        gate.assert_called_once_with(authority)
        self.assertEqual(popen.call_args.args[0], [sys.executable, '-B', '-m',
                                                 'research.runtime_entry', '--supervised-child'])
        self.assertNotIn(str(self.root), str(popen.call_args.kwargs['env']))
        packet = json.loads(parent.sendall.call_args.args[0])
        self.assertEqual(packet, self.row(
            operation_deadline=authority.operation_deadline, final_deadline=authority.deadline,
            supervisor_identity=list(d._process(os.getpid())[0]), child_pid=222,
            nonce=packet['nonce']))
        self.assertIsInstance(pump.call_args.kwargs['capture_directory_fd'], int)
        self.assertEqual(json.loads((self.root / 'supervisor-report.json').read_text()), result)
        with patch('research.demo_authority.gate_supervisor'), \
                patch.object(d, '_children', return_value=[]), \
                patch.object(d.subprocess, 'Popen') as repeat:
            with self.assertRaisesRegex(ValueError, 'already exists'):
                d.run_demo_supervisor(authority)
        repeat.assert_not_called()


if __name__ == '__main__':
    unittest.main()
