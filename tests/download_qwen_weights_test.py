#!/usr/bin/env python3
"""Offline downloader regression checks; no dependencies, credentials or network."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_qwen_weights", ROOT / "tools/download_qwen_weights.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)

# Independent approval fixture: a code edit must not silently change what the
# downloader considers approved, even if validation and its constants change.
APPROVED_REPOSITORY = "unsloth/Qwen3.8-Flash-Next-GGUF"
APPROVED_REVISION = "38bb39ee97821de2c9009abb7e93950eec396e66"
APPROVED_FILES = (
    {"name": "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf", "size": 10_946_624,
     "sha256": "5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4"},
    {"name": "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf", "size": 49_835_229_856,
     "sha256": "577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7"},
    {"name": "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf", "size": 43_836_407_744,
     "sha256": "d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833"},
)


def approved_metadata():
    return {"id": APPROVED_REPOSITORY, "sha": APPROVED_REVISION,
            "gated": False, "private": False, "disabled": False,
            "siblings": [{"rfilename": "UD-IQ4_XS/" + item["name"], "size": item["size"],
                          "lfs": {"size": item["size"], "sha256": item["sha256"]}} for item in APPROVED_FILES]}


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pmi-downloader-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # Block real subprocesses and Python connections throughout every test.
        # The transfer test replaces Popen with a local in-memory fake explicitly.
        self.process = self.enterContext(mock.patch.object(download.subprocess, "Popen",
                                        side_effect=AssertionError("network/process launch forbidden in offline tests")))
        self.enterContext(mock.patch.object(socket, "create_connection", side_effect=AssertionError("network forbidden")))
        self.enterContext(mock.patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")))
        self.events = self.enterContext(mock.patch.object(download, "emit"))

    def metadata_path(self, value):
        path = self.root / "metadata.json"
        path.write_text(json.dumps(value))
        return path

    def test_exact_approved_metadata_and_rejections(self):
        self.assertEqual(download.REPOSITORY, APPROVED_REPOSITORY)
        self.assertEqual(download.REVISION, APPROVED_REVISION)
        self.assertEqual(download.FILES, APPROVED_FILES)
        self.assertEqual(download.TOTAL_BYTES, 93_682_584_224)
        download.validate_metadata(self.metadata_path(approved_metadata()))
        mutations = {
            "repository": lambda d: d.update(id="someone/another-model"),
            "revision": lambda d: d.update(sha="main"),
            "gated": lambda d: d.update(gated="auto"),
            "private": lambda d: d.update(private=True),
            "disabled": lambda d: d.update(disabled=True),
            "missing": lambda d: d["siblings"].pop(),
            "duplicate": lambda d: d["siblings"].append(copy.deepcopy(d["siblings"][0])),
            "filename": lambda d: d["siblings"][0].update(rfilename="other.gguf"),
            "size": lambda d: d["siblings"][0].update(size=1),
            "lfs_size": lambda d: d["siblings"][0]["lfs"].update(size=1),
            "hash": lambda d: d["siblings"][0]["lfs"].update(sha256="0" * 64),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                value = approved_metadata()
                mutate(value)
                with self.assertRaises(RuntimeError):
                    download.validate_metadata(self.metadata_path(value))

    def test_streaming_hash_and_bad_final_preserved(self):
        self.assertEqual(download.HASH_CHUNK, 8 * 1024**2)
        data = b"x" * (download.HASH_CHUNK + 17)
        path = self.root / "weight.gguf"
        path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        reads, original_open = [], Path.open

        class Reader:
            def __init__(self, file): self.file = file
            def __enter__(self): return self
            def __exit__(self, *args): return self.file.__exit__(*args)
            def read(self, size):
                reads.append(size)
                return self.file.read(size)

        with mock.patch.object(Path, "open", lambda p, *a, **kw: Reader(original_open(p, *a, **kw))):
            self.assertEqual(download.hash_file(path, len(data)), digest)
        self.assertGreaterEqual(len(reads), 3)
        self.assertEqual(set(reads), {8 * 1024**2})
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            download.verify(path, {"size": len(data), "sha256": "0" * 64})
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        with self.assertRaisesRegex(RuntimeError, "size"):
            download.hash_file(path, len(data) - 1)

    def test_bounds_symlinks_and_disk_reserve(self):
        files = ({"name": "a.gguf", "size": 100}, {"name": "b.gguf", "size": 70})
        with mock.patch.object(download, "FILES", files):
            partial = self.root / "a.gguf.part"
            partial.write_bytes(b"a" * 30)
            self.assertEqual(download.bounds(self.root), 140)
            required = 140 + 20 * 1024**3 + 2 * download.NOTICE_LIMIT
            self.assertEqual(download.RESERVE_BYTES, 20 * 1024**3)
            with mock.patch.object(download.shutil, "disk_usage", return_value=SimpleNamespace(free=required)):
                self.assertEqual(download.check_disk(self.root), (required, 140))
            with mock.patch.object(download.shutil, "disk_usage", return_value=SimpleNamespace(free=required - 1)):
                with self.assertRaisesRegex(RuntimeError, "insufficient disk"):
                    download.check_disk(self.root)
            partial.write_bytes(b"a" * 101)
            with self.assertRaisesRegex(RuntimeError, "partial file exceeds"):
                download.bounds(self.root)
            self.assertEqual(partial.stat().st_size, 101)
            partial.unlink()
            final = self.root / "a.gguf"
            final.write_bytes(b"wrong")
            with self.assertRaisesRegex(RuntimeError, "final file has wrong size"):
                download.bounds(self.root)
            self.assertEqual(final.read_bytes(), b"wrong")
            partial.symlink_to(final)
            with self.assertRaisesRegex(RuntimeError, "regular file"):
                download.local_size(partial)
            partial.unlink()
            partial.symlink_to(self.root / "missing")
            with self.assertRaisesRegex(RuntimeError, "regular file"):
                download.local_size(partial)

    def test_atomic_publish_never_replaces_existing_file(self):
        partial, final = self.root / "a.part", self.root / "a.gguf"
        partial.write_bytes(b"verified")
        final.write_bytes(b"preserve existing")
        with self.assertRaises(FileExistsError):
            download.publish(partial, final)
        self.assertEqual(partial.read_bytes(), b"verified")
        self.assertEqual(final.read_bytes(), b"preserve existing")
        final.unlink()
        download.publish(partial, final)
        self.assertFalse(partial.exists())
        self.assertEqual(final.read_bytes(), b"verified")
        download.atomic_json(self.root / "receipt.json", {"complete": True})
        self.assertEqual(json.loads((self.root / "receipt.json").read_text()), {"complete": True})

    def test_complete_partial_bypasses_transfer(self):
        partial = self.root / "ready.gguf.part"
        partial.write_bytes(b"complete but hash must still be checked")
        with mock.patch.object(download, "check_disk", return_value=(10**12, 0)), \
             mock.patch.object(download.shutil, "which", return_value="/not/executed/curl"):
            download.transfer("https://example.invalid/never", partial, partial.stat().st_size, self.root)
        self.process.assert_not_called()

    def test_http_403_stops_without_retry_and_preserves_partial(self):
        partial = self.root / "weight.gguf.part"
        partial.write_bytes(b"valid prefix")
        commands = []

        class Denied:
            def __init__(self, command, **kwargs):
                commands.append(command)
                kwargs["stdout"].write(b"403")
                kwargs["stderr"].write(b"curl: (22) HTTP 403 https://example.invalid/?signed=not-for-output")
            def poll(self): return 22
            def wait(self, timeout=None): return 22

        with mock.patch.object(download.subprocess, "Popen", Denied), \
             mock.patch.object(download.shutil, "which", return_value="/not/executed/curl"), \
             mock.patch.object(download, "check_disk", return_value=(10**12, 100)), \
             mock.patch.object(download.time, "sleep", side_effect=AssertionError("403 must not retry")):
            with self.assertRaisesRegex(RuntimeError, "HTTP 403.*without retry"):
                download.transfer("https://example.invalid/never", partial, 100, self.root)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][1], "--disable")
        self.assertEqual(commands[0][-2:], ["--continue-at", "-"])
        self.assertEqual(partial.read_bytes(), b"valid prefix")
        failures = [call.kwargs for call in self.events.call_args_list if call.args[0] == "transfer_failure"]
        self.assertEqual(failures[0]["http_status"], 403)
        self.assertNotIn("signed=", failures[0]["reason"])
        self.assertIn("HTTP 403", failures[0]["reason"])

    def test_default_main_is_network_free_and_does_not_create_destination(self):
        path = self.metadata_path(approved_metadata())
        destination = self.root / "new-model-folder"
        with mock.patch.object(download, "DESTINATION", destination), \
             mock.patch.object(download.shutil, "disk_usage", return_value=SimpleNamespace(free=10**12)), \
             mock.patch.object(sys, "argv", ["download_qwen_weights.py", "--metadata", str(path), "--shard", "1"]):
            self.assertEqual(download.main(), 0)
        self.assertFalse(destination.exists())
        self.process.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
