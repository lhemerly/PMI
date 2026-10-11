"""Runtime packaging checks using inert bytes; no worker or process is executed."""
import hashlib
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from research.executor import verify_runtime
from research.runtime_bundle import build_runtime


class RuntimeBundleStaticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.addCleanup(self.thaw)

    def thaw(self):
        for path in self.root.rglob('*'):
            if path.is_dir():
                path.chmod(0o700)

    def inventory(self, worker='interval'):
        sources = self.root / 'sources'
        sources.mkdir()
        fixtures = {
            'usr/bin/python3': b'inert interpreter fixture',
            'lib64/ld-linux-x86-64.so.2': b'\x7fELFinert loader fixture',
            'usr/lib/python3.13/lib-dynload/example.so': b'\x7fELFinert extension fixture',
            'usr/lib/python3.13/example.py': b'# inert standard library fixture\n',
        }
        inventory = {}
        for index, (target, data) in enumerate(fixtures.items()):
            source = sources / str(index)
            source.write_bytes(data)
            # Packaging policy follows target/content, not input permission bits.
            source.chmod(0o400 if data.startswith(b'\x7fELF') else 0o700)
            inventory[target] = {'source': str(source), 'sha256': hashlib.sha256(data).hexdigest()}
        worker_path = Path(__file__).parent / 'research' / f'{worker}_worker.py'
        inventory['worker.py'] = {
            'source': str(worker_path.resolve()),
            'sha256': hashlib.sha256(worker_path.read_bytes()).hexdigest(),
        }
        return inventory

    def package(self, worker='interval'):
        with patch('launch_policy.require_real_execution'), patch(
            'subprocess.Popen', side_effect=AssertionError('process execution forbidden')
        ):
            return build_runtime(self.root / 'bundle', self.inventory(worker), worker=worker)

    def test_required_mountpoints_and_runtime_directories_are_frozen(self):
        receipt = self.package()
        bundle = Path(receipt['root'])
        directories = {path.relative_to(bundle).as_posix() for path in bundle.rglob('*') if path.is_dir()}
        self.assertEqual(directories, {
            'candidate', 'work', 'proc', 'dev', 'tmp', 'lib64', 'usr', 'usr/bin',
            'usr/lib', 'usr/lib/python3.13', 'usr/lib/python3.13/lib-dynload',
        })
        for path in [bundle, *(bundle / name for name in directories)]:
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o500)
        self.assertEqual(verify_runtime(bundle, receipt['files']), receipt['runtime_sha256'])

    def test_elf_loader_and_extension_keep_execute_permission(self):
        receipt = self.package()
        bundle = Path(receipt['root'])
        expected_modes = {
            'usr/bin/python3': 0o500,
            'lib64/ld-linux-x86-64.so.2': 0o500,
            'usr/lib/python3.13/lib-dynload/example.so': 0o500,
            'usr/lib/python3.13/example.py': 0o400,
            'worker.py': 0o400,
        }
        self.assertEqual(set(receipt['files']), set(expected_modes))
        for name, expected in expected_modes.items():
            self.assertEqual(stat.S_IMODE((bundle / name).stat().st_mode), expected)

    def test_missing_or_extra_mount_directory_is_rejected(self):
        receipt = self.package()
        bundle = Path(receipt['root'])
        for name in ('candidate', 'work', 'proc', 'dev', 'tmp'):
            with self.subTest(missing=name):
                bundle.chmod(0o700)
                (bundle / name).rmdir()
                bundle.chmod(0o500)
                with self.assertRaises(ValueError):
                    verify_runtime(bundle, receipt['files'])
                bundle.chmod(0o700)
                (bundle / name).mkdir(mode=0o500)
                bundle.chmod(0o500)
        bundle.chmod(0o700)
        (bundle / 'unexpected').mkdir(mode=0o500)
        bundle.chmod(0o500)
        with self.assertRaises(ValueError):
            verify_runtime(bundle, receipt['files'])

    def test_permission_changes_are_rejected_without_byte_changes(self):
        receipt = self.package()
        bundle = Path(receipt['root'])
        changes = {
            '.': 0o700,
            'candidate': 0o700,
            'usr/lib': 0o700,
            'usr/bin/python3': 0o400,
            'lib64/ld-linux-x86-64.so.2': 0o400,
            'usr/lib/python3.13/lib-dynload/example.so': 0o400,
            'usr/lib/python3.13/example.py': 0o500,
            'worker.py': 0o600,
        }
        for name, mode in changes.items():
            with self.subTest(path=name):
                path = bundle / name
                original = stat.S_IMODE(path.stat().st_mode)
                path.chmod(mode)
                try:
                    with self.assertRaises(ValueError):
                        verify_runtime(bundle, receipt['files'])
                finally:
                    path.chmod(original)

    def test_qualification_worker_is_pinned_to_its_distinct_role(self):
        receipt = self.package(worker='qualification')
        self.assertEqual(
            verify_runtime(receipt['root'], receipt['files'], worker='qualification'),
            receipt['runtime_sha256'],
        )
        with self.assertRaises(ValueError):
            verify_runtime(receipt['root'], receipt['files'], worker='interval')

    def test_mismatched_worker_is_rejected_and_bundle_removed(self):
        inventory = self.inventory('interval')
        with patch('launch_policy.require_real_execution'), patch(
            'subprocess.Popen', side_effect=AssertionError('process execution forbidden')
        ), self.assertRaises(ValueError):
            build_runtime(self.root / 'bundle', inventory, worker='qualification')
        self.assertFalse((self.root / 'bundle').exists())

    def test_unknown_worker_role_is_rejected_before_packaging(self):
        with patch('launch_policy.require_real_execution'), self.assertRaises(ValueError):
            build_runtime(self.root / 'bundle', self.inventory(), worker='unknown')
        self.assertFalse((self.root / 'bundle').exists())


if __name__ == '__main__':
    unittest.main()
