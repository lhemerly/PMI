"""Model-free tests: fixed harmless fixtures, never the production child entry."""
import array
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research import deadline as d
from research import coordination as c


# This dedicated test subprocess may become a subreaper without changing the
# test runner's process ownership. Its child is a fixed benign fixture below.
FIXTURE = r'''
import json,os,signal,socket,subprocess,sys,time
from pathlib import Path
from research import deadline as d,coordination as c
mode,folder=sys.argv[1:]
c.CPU_LOCK=Path(folder)/'cpu';c.FACTORY_LEASE=Path(folder)/'factory'
for p in (c.CPU_LOCK,c.FACTORY_LEASE):p.touch()
d._prctl(36,1)
channel,child_channel=socket.socketpair(socket.AF_UNIX,socket.SOCK_SEQPACKET)
channel.setsockopt(socket.SOL_SOCKET,socket.SO_PASSCRED,1)
child_source=r"""
import json,os,signal,socket,sys,time
from pathlib import Path
from research import deadline as d,coordination as c
mode,folder=sys.argv[1:]
if mode=='deadline':time.sleep(10)
if mode=='stream':
    while True:os.write(1,b'x'*65536)
if mode=='initial-error':time.sleep(10)
c.CPU_LOCK=Path(folder)/'cpu';c.FACTORY_LEASE=Path(folder)/'factory'
d.OUTPUT_ROOT=Path(folder);d.CLEANUP_SECONDS=3
if mode=='runtime':d.OUTPUT_ROOT=Path('/tmp/unrelated-legacy-demo-root')
print('benign controller diagnostic',file=sys.stderr,flush=True)
with d.read_supervisor_context() as context:
    if mode=='runtime':
        assert context.child_module=='research.runtime_entry'
        assert context.output_root==folder
        print('authenticated reusable runtime context',file=sys.stderr,flush=True)
    with c.CoordinationOwner() as owner:
        context.retain_coordination(owner)
        if mode=='wrong-peer':
            if os.fork()==0:
                context._channel.sendall(d._encode({'action':'finished','nonce':context.nonce,'status':'finished'}))
                os._exit(0)
            time.sleep(10)
        if mode=='orphan':
            if os.fork()==0:
                os.setsid();signal.signal(signal.SIGTERM,signal.SIG_IGN)
                time.sleep(10);os._exit(0)
        context.finish('finished')
"""
descriptor=child_channel.fileno()
root=subprocess.Popen([sys.executable,'-B','-c',child_source,mode,folder],
    stdout=subprocess.PIPE,stderr=subprocess.PIPE,stdin=subprocess.DEVNULL,
    pass_fds=tuple(sorted({3,descriptor})),preexec_fn=lambda:d._child_setup(descriptor),
    start_new_session=True)
child_channel.close()
now=time.monotonic();operation=now+(.15 if mode=='deadline' else 2);final=operation+3
nonce='a'*64
packet={'schema':'PMI_SUPERVISOR_V1','operation_deadline':operation,
 'final_deadline':final,'supervisor_identity':list(d._process(os.getpid())[0]),
 'child_pid':root.pid,'output_root':folder,'nonce':nonce}
if mode=='runtime':
    info=Path(folder).stat()
    packet.update(schema='PMI_RUNTIME_SUPERVISOR_V1',child_module='research.runtime_entry',
                  output_root_identity=[info.st_dev,info.st_ino])
channel.sendall(d._encode(packet))
report=d._pump(root,channel,operation,final,nonce,lambda:None,
              'injected bootstrap failure' if mode=='initial-error' else None,capture_dir=folder)
report['children_after']=d._children();report['elapsed']=time.monotonic()-now
report['captured_stdout_bytes']=(Path(folder)/'controller.stdout.log').stat().st_size
report['captured_stderr']=(Path(folder)/'controller.stderr.log').read_text()
report['capture_mode']=(Path(folder)/'controller.stderr.log').stat().st_mode&0o777
channel.close();root.stdout.close();root.stderr.close()
print(json.dumps(report))
'''


class DeadlineTests(unittest.TestCase):
    def run_fixture(self, mode):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, '-B', '-c', FIXTURE, mode, folder],
                                    capture_output=True, text=True, timeout=9,
                                    cwd=Path(__file__).resolve().parent)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report['cleanup_verified'], report)
        self.assertEqual(report['children_after'], [])
        self.assertEqual(report['remaining_owned'], [])
        self.assertLess(report['elapsed'], 5.5)
        return report

    def test_authenticated_handoff_and_normal_cleanup(self):
        report = self.run_fixture('success')
        self.assertTrue(report['completed'], report)
        self.assertTrue(report['coordination_retained'])
        self.assertEqual(report['child_final_status'], 'finished')
        self.assertEqual(report['coordination_release'], 'verified_empty')
        self.assertIn('benign controller diagnostic', report['captured_stderr'])
        self.assertEqual(report['capture_mode'], 0o600)

    def test_operation_deadline_stops_owned_child(self):
        report = self.run_fixture('deadline')
        self.assertFalse(report['completed'])
        self.assertEqual(report['stop_reason'], 'operation_deadline')
        self.assertLess(report['elapsed'], 1)

    def test_runtime_authenticated_context_uses_its_attempt_directory(self):
        report = self.run_fixture('runtime')
        self.assertTrue(report['completed'], report)
        self.assertIn('authenticated reusable runtime context', report['captured_stderr'])

    def test_separate_session_orphan_forces_failed_cleanup(self):
        report = self.run_fixture('orphan')
        self.assertFalse(report['completed'])
        self.assertEqual(report['stop_reason'], 'unexpected_descendants')
        self.assertTrue(report['coordination_retained'])

    def test_stream_limit_stops_owned_child(self):
        report = self.run_fixture('stream')
        self.assertFalse(report['completed'])
        self.assertEqual(report['stop_reason'], 'stream_limit')
        self.assertGreater(report['stream_bytes']['stdout'], d.STREAM_LIMIT)
        self.assertEqual(report['captured_stdout_bytes'], d.STREAM_LIMIT)

    def test_bootstrap_exception_still_cleans_child(self):
        report = self.run_fixture('initial-error')
        self.assertFalse(report['completed'])
        self.assertEqual(report['stop_reason'], 'supervisor_failure')

    def test_descendant_cannot_impersonate_controller(self):
        report = self.run_fixture('wrong-peer')
        self.assertFalse(report['completed'])
        self.assertEqual(report['stop_reason'], 'control_failure')
        self.assertTrue(any('unauthenticated' in value for value in report['errors']))

    def test_public_launcher_requires_authority_before_any_spawn(self):
        with patch('research.demo_authority.gate_supervisor', side_effect=ValueError('held')), \
                patch.object(d.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(ValueError, 'held'):
                d.run_demo_supervisor()
        popen.assert_not_called()

    def test_context_constructor_rejects_forgery(self):
        with self.assertRaises(ValueError):
            d.SupervisorContext(object(), None, {})

    def test_context_parent_death_or_expiration_is_blocked(self):
        current = d._process(os.getpid())[0]
        parent = d._process(os.getppid())[0]
        channel = Mock()
        row = {'operation_deadline': time.monotonic() + 5,
               'final_deadline': time.monotonic() + 185,
               'supervisor_identity': list(parent), 'child_pid': os.getpid(),
               'output_root': str(d.OUTPUT_ROOT), 'nonce': 'a' * 64}
        context = d.SupervisorContext(d._CONTEXT_KEY, channel, row)
        self.assertIs(context.validate(), context)
        with patch.object(d.os, 'getppid', return_value=1):
            with self.assertRaises(ValueError):
                context.validate()
        with patch.object(d.time, 'monotonic', return_value=row['operation_deadline']):
            with self.assertRaises(TimeoutError):
                context.validate()
        context.close()
        with self.assertRaises(ValueError):
            context.validate()

    def test_receive_closes_all_rights_after_earlier_ancillary_error(self):
        read, write = os.pipe()
        try:
            fake = Mock()
            fake.recvmsg.return_value = (b'{}', [
                (socket.SOL_SOCKET, 99999, b''),
                (socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [read]).tobytes())], 0, None)
            with self.assertRaisesRegex(ValueError, 'ancillary'):
                d._receive(fake)
            with self.assertRaises(OSError):
                os.fstat(read)
        finally:
            os.close(write)

    def test_received_coordination_is_same_locked_description(self):
        with tempfile.TemporaryDirectory() as folder:
            cpu, factory = Path(folder) / 'cpu', Path(folder) / 'factory'
            cpu.touch(); factory.touch()
            with patch.object(c, 'CPU_LOCK', cpu), patch.object(c, 'FACTORY_LEASE', factory):
                with c.CoordinationOwner() as owner:
                    duplicates = [os.dup(fd) for fd in owner.export_fds()]
                    try:
                        d._validate_retained(duplicates, os.getpid())
                        with self.assertRaises(ValueError):
                            d._validate_retained(duplicates, os.getpid() + 1)
                        reopened = os.open(cpu, os.O_RDWR | os.O_CLOEXEC)
                        try:
                            with self.assertRaises(ValueError):
                                d._validate_retained([reopened, duplicates[1]], os.getpid())
                        finally:
                            os.close(reopened)
                    finally:
                        for fd in duplicates:
                            os.close(fd)

    def test_known_pidfds_are_signaled_despite_inventory_failure(self):
        root = SimpleNamespace(pid=123, returncode=None, send_signal=Mock())
        owned = d._OwnedProcesses(root)
        owned.handles = {456: ((456, 8), 11), 789: ((789, 9), 12)}
        with patch.object(owned, 'refresh', side_effect=PermissionError('unreadable')), \
                patch.object(d.signal, 'pidfd_send_signal', side_effect=[PermissionError('denied'), None]) as send:
            with self.assertRaisesRegex(ValueError, 'uncertainty'):
                owned.signal_all(signal.SIGKILL)
        self.assertEqual(send.call_count, 2)
        root.send_signal.assert_called_once_with(signal.SIGKILL)

    def test_cleanup_uses_one_absolute_deadline_and_marks_uncertainty(self):
        owned = Mock()
        owned.signal_all.side_effect = PermissionError('cannot inspect')
        ticks = iter([0, 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
        with patch.object(d.time, 'monotonic', side_effect=lambda: next(ticks)), \
                patch.object(d.time, 'sleep'):
            verified, errors = d._cleanup_owned(owned, .5)
        self.assertFalse(verified)
        self.assertEqual(len(errors), 1)

    def test_child_setup_closes_duplicate_socket_source(self):
        with patch.object(d.os, 'dup2') as duplicate, patch.object(d.os, 'close') as close:
            d._child_setup(17)
        duplicate.assert_called_once_with(17, 3, inheritable=True)
        close.assert_called_once_with(17)


if __name__ == '__main__':
    unittest.main()
