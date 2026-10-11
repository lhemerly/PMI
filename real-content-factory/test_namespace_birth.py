"""Mocked process metadata; no sandbox, process inspection or signals."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research.owned_cpu import namespace_members,process_birth


class NamespaceBirthTests(unittest.TestCase):
    def test_old_stable_process_is_excluded_without_namespace_access(self):
        with patch('research.owned_cpu.Path.iterdir',return_value=[Path('/proc/675')]),patch('research.owned_cpu.process_birth',return_value=(675,1153)) as birth,patch('research.owned_cpu.Path.stat',side_effect=PermissionError('protected')) as stat:
            self.assertEqual(namespace_members(999,43081879),[])
        self.assertEqual(birth.call_count,2);stat.assert_not_called()

    def test_equal_newer_and_unbounded_protected_processes_remain_blocked(self):
        for ticks in (100,101):
            with self.subTest(ticks=ticks),patch('research.owned_cpu.Path.iterdir',return_value=[Path('/proc/675')]),patch('research.owned_cpu.process_birth',return_value=(675,ticks)),patch('research.owned_cpu.Path.stat',side_effect=PermissionError('protected')),self.assertRaises(PermissionError):
                namespace_members(999,100)

    def test_old_identity_change_is_not_an_exclusion(self):
        with patch('research.owned_cpu.Path.iterdir',return_value=[Path('/proc/675')]),patch('research.owned_cpu.process_birth',side_effect=[(675,1),(675,101)]),self.assertRaisesRegex(ValueError,'identity changed'):
            namespace_members(999,100)

    def test_unknown_birth_is_not_an_exclusion(self):
        for error in (PermissionError('unreadable'),ValueError('malformed')):
            with self.subTest(error=type(error).__name__),patch('research.owned_cpu.Path.iterdir',return_value=[Path('/proc/675')]),patch('research.owned_cpu.process_birth',side_effect=error),self.assertRaises(type(error)):
                namespace_members(999,100)

    def test_disappeared_process_is_not_reported_as_member(self):
        for reads in ([FileNotFoundError()],[(675,1),ProcessLookupError()]):
            with self.subTest(reads=reads),patch('research.owned_cpu.Path.iterdir',return_value=[Path('/proc/675')]),patch('research.owned_cpu.process_birth',side_effect=reads):
                self.assertEqual(namespace_members(999,100),[])

    def test_same_tick_namespace_member_is_preserved(self):
        with patch('research.owned_cpu.Path.iterdir',return_value=[Path('/proc/675')]),patch('research.owned_cpu.process_birth',return_value=(675,100)),patch('research.owned_cpu.Path.stat',return_value=SimpleNamespace(st_ino=999,st_uid=0)),patch('research.owned_cpu.fingerprint',return_value=(675,100,'pid:[999]')),patch('research.owned_cpu.Path.read_text',return_value='VmRSS:\t12 kB\nSeccomp:\t2\n'):
            self.assertEqual(namespace_members(999,100),[{'identity':(675,100,'pid:[999]'),'rss':12288,'seccomp':2}])

    def test_recent_member_identity_change_blocks_scan(self):
        with patch('research.owned_cpu.Path.iterdir',return_value=[Path('/proc/675')]),patch('research.owned_cpu.process_birth',return_value=(675,100)),patch('research.owned_cpu.Path.stat',return_value=SimpleNamespace(st_ino=999)),patch('research.owned_cpu.fingerprint',return_value=(675,101,'pid:[999]')),self.assertRaisesRegex(ValueError,'identity changed'):
            namespace_members(999,100)

    def test_member_identity_race_after_status_cannot_prove_empty(self):
        with patch('research.owned_cpu.Path.iterdir',return_value=[Path('/proc/675')]),patch('research.owned_cpu.process_birth',return_value=(675,100)),patch('research.owned_cpu.Path.stat',return_value=SimpleNamespace(st_ino=999)),patch('research.owned_cpu.fingerprint',side_effect=[(675,100,'pid:[999]'),(675,101,'pid:[999]')]),patch('research.owned_cpu.Path.read_text',return_value='VmRSS: 12 kB\nSeccomp: 2\n'),self.assertRaisesRegex(ValueError,'identity changed'):
            namespace_members(999,100)

    def test_invalid_bound_is_refused_before_proc_scan(self):
        for value in (True,-1,1.5,'100'):
            with self.subTest(value=value),patch('research.owned_cpu.Path.iterdir') as scan,self.assertRaises(ValueError):namespace_members(999,value)
            scan.assert_not_called()

    def test_birth_parser_checks_pid_and_handles_parentheses_in_name(self):
        record='675 (fixture (name)) '+' '.join(['S']+['0']*18+['1153'])
        with patch('research.owned_cpu.Path.read_text',return_value=record):
            self.assertEqual(process_birth(675),(675,1153))
            with self.assertRaisesRegex(ValueError,'stat identity'):process_birth(676)


if __name__=='__main__':unittest.main()
