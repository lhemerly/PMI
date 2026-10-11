"""Direct integrity reads of tiny fixtures only; no model/native workloads."""
from contextlib import ExitStack
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import llama_backend as backend


SHA256 = hashlib.sha256
BLOCK = 262144


class DirectHashTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='pmi-direct-hash-unit-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'fixture.bin'

    def fixture(self, size):
        data = (bytes(range(256)) * ((size + 255) // 256))[:size]
        self.path.write_bytes(data)
        return data

    def digest(self):
        return backend.direct_hash_before_deadline(self.path, time.monotonic() + 30)

    def assert_closed(self, descriptor):
        with self.assertRaises(OSError) as raised:
            os.fstat(descriptor)
        self.assertEqual(raised.exception.errno, errno.EBADF)

    def test_real_direct_io_digests_empty_partial_aligned_and_multi_block_files(self):
        real_open, real_preadv = os.open, os.preadv
        for size in (0, 1, 4095, 4096, BLOCK, BLOCK + 37):
            with self.subTest(size=size):
                expected = SHA256(self.fixture(size)).hexdigest()
                descriptors, reads = [], []

                def opened(path, flags):
                    self.assertTrue(flags & os.O_DIRECT)
                    self.assertTrue(flags & os.O_NOFOLLOW)
                    descriptor = real_open(path, flags)
                    descriptors.append(descriptor)
                    return descriptor

                def read(descriptor, buffers, offset):
                    reads.append((offset, len(buffers[0])))
                    return real_preadv(descriptor, buffers, offset)

                with patch.object(backend.os, 'open', side_effect=opened), \
                        patch.object(backend.os, 'preadv', side_effect=read):
                    self.assertEqual(self.digest(), expected)
                self.assertEqual(reads, [(offset, BLOCK) for offset in range(0, size, BLOCK)])
                self.assertEqual(len(descriptors), 1)
                self.assert_closed(descriptors[0])

    def test_short_zero_negative_and_oversized_read_counts_fail_closed(self):
        self.fixture(4096)
        real_open = os.open
        for count in (-1, 0, 4095, 4097, BLOCK + 1):
            with self.subTest(count=count):
                descriptors = []

                def opened(path, flags):
                    descriptor = real_open(path, flags)
                    descriptors.append(descriptor)
                    return descriptor

                with patch.object(backend.os, 'open', side_effect=opened), \
                        patch.object(backend.os, 'preadv', return_value=count), \
                        self.assertRaisesRegex(ValueError, 'short or changed'):
                    self.digest()
                self.assert_closed(descriptors[0])

    def test_path_replacement_after_read_is_rejected(self):
        data = self.fixture(4096)
        replacement = self.root / 'replacement.bin'
        replacement.write_bytes(data)
        real_preadv = os.preadv

        def read(descriptor, buffers, offset):
            count = real_preadv(descriptor, buffers, offset)
            os.replace(replacement, self.path)
            return count

        with patch.object(backend.os, 'preadv', side_effect=read), \
                self.assertRaisesRegex(ValueError, 'model changed'):
            self.digest()

    def test_shrink_after_complete_read_is_rejected(self):
        self.fixture(4096)
        real_preadv = os.preadv

        def read(descriptor, buffers, offset):
            count = real_preadv(descriptor, buffers, offset)
            os.truncate(self.path, 4095)
            return count

        with patch.object(backend.os, 'preadv', side_effect=read), \
                self.assertRaisesRegex(ValueError, 'model changed'):
            self.digest()

    def test_same_size_metadata_change_after_read_is_rejected(self):
        self.fixture(4096)
        original = self.path.stat()
        real_preadv = os.preadv

        def read(descriptor, buffers, offset):
            count = real_preadv(descriptor, buffers, offset)
            os.utime(self.path, ns=(original.st_atime_ns, original.st_mtime_ns + 1000000000))
            return count

        with patch.object(backend.os, 'preadv', side_effect=read), \
                self.assertRaisesRegex(ValueError, 'model changed'):
            self.digest()

    def test_expired_deadline_does_not_open_file(self):
        with patch.object(backend.time, 'monotonic', return_value=10), \
                patch.object(backend.os, 'open') as opened, \
                self.assertRaises(TimeoutError):
            backend.direct_hash_before_deadline(self.path, 10)
        opened.assert_not_called()

    def test_deadline_between_chunks_stops_and_closes_descriptor(self):
        self.fixture(BLOCK + 37)
        real_open, real_preadv = os.open, os.preadv
        descriptors = []

        def opened(path, flags):
            descriptor = real_open(path, flags)
            descriptors.append(descriptor)
            return descriptor

        with patch.object(backend.time, 'monotonic', side_effect=[0, 0, 10]), \
                patch.object(backend.os, 'open', side_effect=opened), \
                patch.object(backend.os, 'preadv', wraps=real_preadv) as read, \
                self.assertRaises(TimeoutError):
            backend.direct_hash_before_deadline(self.path, 10)
        self.assertEqual(read.call_count, 1)
        self.assert_closed(descriptors[0])

    def test_deadline_after_final_chunk_rejects_completed_digest(self):
        self.fixture(1)
        with patch.object(backend.time, 'monotonic', side_effect=[0, 0, 10]), \
                self.assertRaises(TimeoutError):
            backend.direct_hash_before_deadline(self.path, 10)

    def test_unsupported_direct_read_propagates_without_buffered_fallback(self):
        self.fixture(1)
        real_open = os.open
        descriptors = []

        def opened(path, flags):
            descriptor = real_open(path, flags)
            descriptors.append(descriptor)
            return descriptor

        unsupported = OSError(errno.EINVAL, 'fixture unsupported direct read')
        with patch.object(backend.os, 'open', side_effect=opened), \
                patch.object(backend.os, 'preadv', side_effect=unsupported), \
                patch.object(Path, 'open', side_effect=AssertionError('buffered fallback')) as buffered, \
                patch.object(backend.os, 'read', side_effect=AssertionError('buffered fallback')) as read, \
                self.assertRaises(OSError) as raised:
            self.digest()
        self.assertIs(raised.exception, unsupported)
        buffered.assert_not_called()
        read.assert_not_called()
        self.assert_closed(descriptors[0])

    def test_unsupported_direct_open_does_not_retry_without_direct_flag(self):
        unsupported = OSError(errno.EINVAL, 'fixture unsupported direct open')
        with patch.object(backend.os, 'open', side_effect=unsupported) as opened, \
                patch.object(Path, 'open', side_effect=AssertionError('buffered fallback')) as buffered, \
                self.assertRaises(OSError) as raised:
            self.digest()
        self.assertIs(raised.exception, unsupported)
        self.assertEqual(opened.call_count, 1)
        self.assertTrue(opened.call_args.args[1] & os.O_DIRECT)
        buffered.assert_not_called()

    def test_descriptor_without_direct_flag_is_rejected_before_read(self):
        self.fixture(1)
        with patch.object(fcntl, 'fcntl', return_value=os.O_RDONLY), \
                patch.object(backend.os, 'preadv') as read, \
                self.assertRaisesRegex(ValueError, 'stable regular direct-IO'):
            self.digest()
        read.assert_not_called()

    def test_hard_link_and_symbolic_link_are_rejected(self):
        self.fixture(1)
        link = self.root / 'link.bin'
        os.link(self.path, link)
        with self.assertRaisesRegex(ValueError, 'stable regular direct-IO'):
            self.digest()
        link.unlink()
        target = self.root / 'target.bin'
        self.path.rename(target)
        self.path.symlink_to(target)
        with self.assertRaises(OSError) as raised:
            self.digest()
        self.assertEqual(raised.exception.errno, errno.ELOOP)


class ModelHashRoutingTests(unittest.TestCase):
    def test_only_explicit_research_model_hash_uses_direct_reader(self):
        """Exercise the production verification route with exclusively tiny fixtures."""
        with tempfile.TemporaryDirectory(prefix='pmi-model-hash-routing-unit-') as directory:
            root = Path(directory)
            names = ('content_factory.py', 'llama_backend.py', 'native_worker.py',
                     'launch_policy.py', 'prompt_fit.py', 'piper_backend.py',
                     'piper_worker.py', 'load_telemetry.py', 'admission_policy.py')
            for name in names:
                (root / name).write_bytes(b'fixture source')
            model, template, dependency, runtime = [root / name for name in
                ('model.bin', 'template.txt', 'dependency.bin', 'runtime.bin')]
            for path in (model, template, dependency, runtime):
                path.write_bytes(('tiny ' + path.name).encode())
            model_hash = SHA256(model.read_bytes()).hexdigest()
            info = runtime.stat()
            manifest = {'files': {str(runtime): {'SHA256': SHA256(runtime.read_bytes()).hexdigest()}},
                        'identities': {str(runtime): [str(runtime.resolve()), info.st_dev, info.st_ino,
                            info.st_size, info.st_mtime_ns, info.st_ctime_ns]},
                        'directory_inventories': {}}
            freeze = root / 'stock-input-freeze.json'
            freeze.write_text(json.dumps(manifest))
            freeze_bytes = freeze.read_bytes()
            qualification = {'fit_policy': 'fixture-fit', 'layers': 65, 'sha256': model_hash}
            pin = {'schema': 'PMI_REAL_BACKEND_INTEGRITY_V1',
                   'source_sha256': {name: SHA256((root / name).read_bytes()).hexdigest() for name in names},
                   'fit_dependencies_sha256': {str(dependency): SHA256(dependency.read_bytes()).hexdigest()},
                   'models': {'bonsai-stock': qualification}}
            (root / 'backend-input-seal.json').write_text(json.dumps(pin))

            def fixture_sha256(data=b'', *args, **kwargs):
                if data == freeze_bytes:
                    return Mock(hexdigest=Mock(return_value=
                        '9ee5022ff49fbb802b0950a5d9277f1c2b97546d7af8072e2025f4a1fa896591'))
                return SHA256(data, *args, **kwargs)

            for explicit in (False, True):
                with self.subTest(explicit=explicit), ExitStack() as stack:
                    stack.enter_context(patch.object(backend, 'HERE', root))
                    stack.enter_context(patch.object(backend, 'STOCK', root))
                    stack.enter_context(patch.object(backend, 'MODELS', {'bonsai-stock': {
                        'path': model, 'template': template, 'bytes': model.stat().st_size,
                        'sha256': model_hash}}))
                    stack.enter_context(patch.object(backend.hashlib, 'sha256', side_effect=fixture_sha256))
                    stack.enter_context(patch.object(backend, 'verify_bonsai_runtime', return_value={}))
                    stack.enter_context(patch.object(backend, 'fit_prompt', return_value={
                        'policy': 'fixture-fit', 'layers': 65}))
                    stack.enter_context(patch.object(backend, 'require_prompt_fit'))
                    stack.enter_context(patch('research.demo_authority.reviewed_backend_seal', return_value=pin))
                    direct = stack.enter_context(patch.object(backend, 'direct_hash_before_deadline',
                        return_value=model_hash))
                    hashes = stack.enter_context(patch.object(backend, 'hash_before_deadline',
                        wraps=backend.hash_before_deadline))
                    options = {'profile_id': 'bonsai-research-v1', 'role': 'calibration',
                               'authority': object()} if explicit else {}
                    result, fingerprints = backend.verify_runtime_and_model(
                        'bonsai-stock', 'fixture prompt', time.monotonic() + 30, **options)
                    self.assertEqual(result['sha256'], model_hash)
                    self.assertIn(str(model), fingerprints)
                    calls = hashes.call_args_list
                    self.assertEqual(len(calls), 3)
                    model_calls = [call for call in calls if Path(call.args[0]) == model]
                    self.assertEqual(len(model_calls), 1)
                    self.assertIs(model_calls[0].kwargs['direct'], explicit)
                    for call in calls:
                        if Path(call.args[0]) != model:
                            self.assertFalse(call.kwargs.get('direct', False))
                    if explicit:
                        direct.assert_called_once_with(model, model_calls[0].args[1])
                    else:
                        direct.assert_not_called()


if __name__ == '__main__':
    unittest.main()
