"""Deterministic regular-file fixtures; no model, runtime or workload execution."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from content_factory import bounded_read
from llama_backend import BoundedLiveLog, reject_loading_fallback


class LiveLogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'stderr.log'
        self.path.write_bytes(b'first line\n')

    def append(self, value):
        with self.path.open('ab') as file:
            file.write(value)

    def mutate_during_read(self, action):
        original = os.pread
        changed = False
        def read(fd, count, offset):
            nonlocal changed
            data = original(fd, count, offset)
            if not changed:
                changed = True
                action()
            return data
        return patch('llama_backend.os.pread', side_effect=read)

    def test_append_during_read_returns_initial_snapshot(self):
        reader = BoundedLiveLog(self.path, 1048576)
        with self.mutate_during_read(lambda: self.append(b'second line\n')):
            self.assertEqual(reader.read(), b'first line\n')
        self.assertEqual(reader.read(), b'first line\nsecond line\n')

    def test_multichunk_append_preserves_snapshot(self):
        original = b'a' * 65536 + b'b' * 65536 + b'\n'
        self.path.write_bytes(original)
        reader = BoundedLiveLog(self.path, 1048576)
        with self.mutate_during_read(lambda: self.append(b'next\n')):
            self.assertEqual(reader.read(), original)
        self.assertEqual(reader.read(), original + b'next\n')

    def test_partial_line_is_deferred_until_complete(self):
        self.path.write_bytes(b'first\nload_mode = d')
        reader = BoundedLiveLog(self.path, 1048576)
        self.assertEqual(reader.read(), b'first\n')
        reject_loading_fallback(reader.read().decode())
        self.append(b'io\n')
        self.assertEqual(reader.read(), b'first\nload_mode = dio\n')
        reject_loading_fallback(reader.read().decode())

    def test_partial_failure_line_detected_when_complete(self):
        self.path.write_bytes(b'load_mode = buff')
        reader = BoundedLiveLog(self.path, 1048576)
        reject_loading_fallback(reader.read().decode())
        self.append(b'ered\n')
        with self.assertRaises(RuntimeError):
            reject_loading_fallback(reader.read().decode())

    def test_shrink_during_read_rejected(self):
        with self.mutate_during_read(lambda: self.path.write_bytes(b'')):
            with self.assertRaises(ValueError):
                BoundedLiveLog(self.path, 1048576).read()

    def test_shrink_between_polls_rejected(self):
        reader = BoundedLiveLog(self.path, 1048576)
        reader.read()
        self.path.write_bytes(b'short\n')
        with self.assertRaises(ValueError):
            reader.read()

    def test_replacement_during_read_rejected(self):
        def replace():
            replacement = self.path.with_name('replacement')
            replacement.write_bytes(b'first line\n')
            replacement.replace(self.path)
        with self.mutate_during_read(replace), self.assertRaises(ValueError):
            BoundedLiveLog(self.path, 1048576).read()

    def test_replacement_between_polls_rejected(self):
        reader = BoundedLiveLog(self.path, 1048576)
        reader.read()
        replacement = self.path.with_name('replacement')
        replacement.write_bytes(b'first line\n')
        replacement.replace(self.path)
        with self.assertRaises(ValueError):
            reader.read()

    def test_prefix_overwrite_during_read_rejected(self):
        with self.mutate_during_read(lambda: self.path.write_bytes(b'other line\n')):
            with self.assertRaises(ValueError):
                BoundedLiveLog(self.path, 1048576).read()

    def test_prefix_overwrite_between_polls_rejected(self):
        reader = BoundedLiveLog(self.path, 1048576)
        reader.read()
        self.path.write_bytes(b'other line\n')
        with self.assertRaises(ValueError):
            reader.read()

    def test_initial_overflow_rejected_and_exact_cap_accepted(self):
        self.path.write_bytes(b'a' * (1048576 - 1) + b'\n')
        self.assertEqual(len(BoundedLiveLog(self.path, 1048576).read()), 1048576)
        self.append(b'x')
        with self.assertRaises(ValueError):
            BoundedLiveLog(self.path, 1048576).read()

    def test_append_over_cap_during_read_rejected(self):
        with self.mutate_during_read(lambda: self.append(b'x' * 1048576)):
            with self.assertRaises(ValueError):
                BoundedLiveLog(self.path, 1048576).read()

    def test_symlink_rejected(self):
        link = self.path.with_name('link')
        link.symlink_to(self.path)
        with self.assertRaises(OSError):
            BoundedLiveLog(link, 1048576).read()

    def test_nonregular_fifo_rejected_without_blocking(self):
        fifo = self.path.with_name('fifo')
        os.mkfifo(fifo)
        with self.assertRaises(ValueError):
            BoundedLiveLog(fifo, 1048576).read()

    def test_empty_log_can_grow(self):
        self.path.write_bytes(b'')
        reader = BoundedLiveLog(self.path, 1048576)
        self.assertEqual(reader.read(), b'')
        self.append(b'ready\n')
        self.assertEqual(reader.read(), b'ready\n')

    def test_immutable_reader_still_rejects_append(self):
        original = os.fstat
        calls = 0
        def stat_then_append(fd):
            nonlocal calls
            result = original(fd)
            calls += 1
            if calls == 1:
                self.append(b'append\n')
            return result
        with patch('content_factory.os.fstat', side_effect=stat_then_append):
            with self.assertRaisesRegex(ValueError, 'input changed or exceeded bound'):
                bounded_read(self.path, 1048576)


if __name__ == '__main__':
    unittest.main()
