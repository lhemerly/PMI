#!/usr/bin/env python3
"""Plan or explicitly download the approved Qwen checkpoint; no network by default.

Transfers resume .gguf.part files. Final names are published only after exact
size and SHA-256 verification. Never run this against a partial file being
written by another downloader. --shard can isolate work from another transfer.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "models/qwen3.8-flash-next-ud-iq4-xs"
REPOSITORY = "unsloth/Qwen3.8-Flash-Next-GGUF"
REVISION = "38bb39ee97821de2c9009abb7e93950eec396e66"
OFFICIAL_REPOSITORY = "Qwen/Qwen3.8-Flash-Next"
OFFICIAL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TOTAL_BYTES = 93_682_584_224
RESERVE_BYTES = 20 * 1024**3
HASH_CHUNK = 8 * 1024**2
PROGRESS_SECONDS = 30
NOTICE_LIMIT = 1024**2
FILES = (
    {"name": "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf", "size": 10_946_624,
     "sha256": "5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4"},
    {"name": "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf", "size": 49_835_229_856,
     "sha256": "577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7"},
    {"name": "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf", "size": 43_836_407_744,
     "sha256": "d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833"},
)


def emit(event, **values):
    print(json.dumps({"event": event, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **values}), flush=True)


def local_size(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return 0
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"expected a regular file, refusing {path}")
    return info.st_size


def validate_metadata(path):
    if path.stat().st_size > 4 * 1024**2:
        raise RuntimeError("metadata exceeds 4 MiB")
    metadata = json.loads(path.read_text())
    if metadata.get("id") != REPOSITORY or metadata.get("sha") != REVISION:
        raise RuntimeError("metadata repository/revision differs from approved checkpoint")
    if metadata.get("gated") or metadata.get("private") or metadata.get("disabled"):
        raise RuntimeError("metadata does not describe the approved public ungated repository")
    for expected in FILES:
        name = "UD-IQ4_XS/" + expected["name"]
        matches = [x for x in metadata.get("siblings", []) if x.get("rfilename") == name]
        if len(matches) != 1:
            raise RuntimeError(f"metadata has missing or duplicate file: {name}")
        entry = matches[0]
        if (entry.get("size") != expected["size"] or entry.get("lfs", {}).get("size") != expected["size"]
                or entry.get("lfs", {}).get("sha256") != expected["sha256"]):
            raise RuntimeError(f"metadata size/hash differs from approved file: {name}")
    if sum(x["size"] for x in FILES) != TOTAL_BYTES:
        raise RuntimeError("internal approved size inventory is inconsistent")


def bounds(directory):
    remaining = 0
    for expected in FILES:
        final = directory / expected["name"]
        partial = directory / (expected["name"] + ".part")
        final_size, partial_size = local_size(final), local_size(partial)
        if final.exists() and final_size != expected["size"]:
            raise RuntimeError(f"existing final file has wrong size; left untouched: {final}")
        if partial_size > expected["size"]:
            raise RuntimeError(f"partial file exceeds approved size; left untouched: {partial}")
        remaining += 0 if final.exists() else expected["size"] - partial_size
    return remaining


def check_disk(directory):
    remaining = bounds(directory)
    existing = directory
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    # Budget every unfinished shard, including any not selected by --shard.
    required = remaining + RESERVE_BYTES + 2 * NOTICE_LIMIT
    if free < required:
        raise RuntimeError(f"insufficient disk: {free} free bytes, {required} required including 20 GiB reserve")
    return free, remaining


def hash_file(path, expected_size=None):
    before = path.stat()
    if local_size(path) != before.st_size or (expected_size is not None and before.st_size != expected_size):
        raise RuntimeError(f"file size changed or differs from expected: {path}")
    digest, count, last = hashlib.sha256(), 0, time.monotonic()
    emit("hash_start", file=path.name, size_bytes=before.st_size)
    with path.open("rb") as stream:
        while data := stream.read(HASH_CHUNK):
            digest.update(data)
            count += len(data)
            if expected_size is not None and count > expected_size:
                raise RuntimeError(f"file grew during verification: {path}")
            if time.monotonic() - last >= PROGRESS_SECONDS:
                emit("hash_progress", file=path.name, hashed_bytes=count, size_bytes=before.st_size)
                last = time.monotonic()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or count != before.st_size:
        raise RuntimeError(f"file changed during verification; stop concurrent writers: {path}")
    return digest.hexdigest()


def verify(path, expected):
    actual = hash_file(path, expected["size"])
    if actual != expected["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch; preserved for inspection and not overwritten: {path}")
    emit("verified", file=path.name, size_bytes=expected["size"], sha256=actual)


def url(repository, revision, filename):
    return f"https://huggingface.co/{repository}/resolve/{revision}/{quote(filename, safe='/')}?download=true"


def transfer(source_url, partial, maximum, directory, *, exact_size=True):
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required; no dependencies will be installed")
    for attempt in range(1, 4):
        check_disk(directory)
        start_bytes = local_size(partial)
        if start_bytes > maximum:
            raise RuntimeError(f"partial exceeds size bound: {partial}")
        if exact_size and start_bytes == maximum:
            return
        if not exact_size and partial.exists():
            # Notice transfers are small. Preserve an incomplete notice rather
            # than silently discarding it to restart a non-range response.
            raise RuntimeError(f"incomplete notice already exists; inspect before retry: {partial}")
        command = [curl, "--disable", "--location", "--fail", "--silent", "--show-error", "--proto", "=https",
                   "--proto-redir", "=https", "--connect-timeout", "30", "--speed-limit", "1024",
                   "--speed-time", "120", "--max-time", "21600", "--retry", "0",
                   "--max-filesize", str(maximum), "--write-out", "%{http_code}",
                   "--output", str(partial), "--url", source_url]
        if exact_size:
            command += ["--continue-at", "-"]
        emit("transfer_start", file=partial.name, attempt=attempt, resume_bytes=start_bytes, maximum_bytes=maximum)
        started = last = time.monotonic()
        with tempfile.TemporaryFile() as errors, tempfile.TemporaryFile() as response:
            # An OS file-size limit bounds the combined resumed file, not only
            # the current HTTP response. Monitoring alone could overshoot by a
            # polling interval. This script targets the approved Linux host.
            def limit_file_size():
                resource.setrlimit(resource.RLIMIT_FSIZE, (maximum, maximum))
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=response, stderr=errors,
                                       preexec_fn=limit_file_size)
            try:
                while process.poll() is None:
                    time.sleep(1)
                    size = local_size(partial)
                    if size > maximum:
                        raise RuntimeError(f"transfer exceeded approved size: {partial}")
                    free, remaining = check_disk(directory)
                    now = time.monotonic()
                    if now - last >= PROGRESS_SECONDS:
                        emit("transfer_progress", file=partial.name, downloaded_bytes=size,
                             maximum_bytes=maximum, bytes_per_second=(size-start_bytes)/(now-started),
                             total_remaining_bytes=remaining, free_bytes=free)
                        last = now
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            status = process.wait()
            response.seek(0)
            code = response.read(32).strip()
            http_status = int(code) if code.isdigit() else None
            errors.seek(0, os.SEEK_END)
            errors.seek(max(0, errors.tell() - 8192))
            reason = errors.read(8192).decode("utf-8", errors="replace").strip()
            reason = re.sub(r"https?://[^\s]+", "[URL omitted]", reason)
        size = local_size(partial)
        if size > maximum:
            raise RuntimeError(f"transfer exceeded approved size: {partial}")
        if status == 0 and (size == maximum if exact_size else 0 < size <= maximum):
            emit("transfer_finished", file=partial.name, size_bytes=size)
            return
        emit("transfer_failure", file=partial.name, attempt=attempt, curl_exit=status,
             http_status=http_status, partial_bytes=size, reason=reason)
        if http_status in (401, 403, 407):
            raise RuntimeError(f"HTTP {http_status} denied access; stopped without retry, partial preserved: {partial}")
        if not exact_size:
            break
        if attempt < 3:
            time.sleep(5 * attempt)
    raise RuntimeError(f"transfer failed after bounded attempts; partial preserved: {partial}")


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def publish(source, destination):
    """Linux atomic rename that refuses to replace any existing destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise RuntimeError("atomic no-replace rename is unavailable; verified partial preserved")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def notices(directory):
    for remote, name in (("LICENSE", "LICENSE"), ("README.md", "Qwen-model-README.md")):
        final, record = directory / name, directory / (name + ".source.json")
        source = url(OFFICIAL_REPOSITORY, OFFICIAL_REVISION, remote)
        if final.exists():
            if not record.exists():
                raise RuntimeError(f"existing notice lacks provenance record; left untouched: {final}")
            saved = json.loads(record.read_text())
            if saved.get("url") != source or local_size(final) > NOTICE_LIMIT or hash_file(final) != saved.get("sha256"):
                raise RuntimeError(f"existing notice failed provenance/hash verification: {final}")
            continue
        partial = directory / (name + ".part")
        transfer(source, partial, NOTICE_LIMIT, directory, exact_size=False)
        raw = partial.read_bytes()
        raw.decode("utf-8")
        if not raw.strip() or raw.lstrip().lower().startswith((b"<!doctype html", b"<html")):
            raise RuntimeError(f"notice is empty or HTML; partial preserved: {partial}")
        digest = hashlib.sha256(raw).hexdigest()
        atomic_json(record, {"url": source, "size_bytes": len(raw), "sha256": digest})
        if final.exists():
            raise RuntimeError(f"notice destination appeared during transfer: {final}")
        publish(partial, final)
        emit("notice_saved", file=name, size_bytes=len(raw), sha256=digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "build-baseline/approved-weight-metadata.json")
    parser.add_argument("--download", action="store_true", help="authorize network transfers for the pinned inventory")
    parser.add_argument("--shard", type=int, choices=(1, 2, 3), action="append", help="process only selected shards (repeatable), plus notices")
    args = parser.parse_args()
    validate_metadata(args.metadata)
    if DESTINATION.resolve() != DESTINATION:
        raise RuntimeError("destination must not redirect through symlinks")
    free, remaining = check_disk(DESTINATION)
    selected = set(args.shard or (1, 2, 3))
    emit("plan", repository=REPOSITORY, revision=REVISION, total_bytes=TOTAL_BYTES,
         remaining_bytes=remaining, free_bytes=free, reserve_bytes=RESERVE_BYTES,
         selected_shards=sorted(selected), destination=str(DESTINATION), download=args.download,
         files=list(FILES), existing_finals_hash_verified=False)
    if not args.download:
        return 0
    DESTINATION.mkdir(parents=True, exist_ok=True)
    notices(DESTINATION)
    verified = set()
    for index, expected in enumerate(FILES, 1):
        if index not in selected:
            continue
        final = DESTINATION / expected["name"]
        partial = DESTINATION / (expected["name"] + ".part")
        if final.exists():
            verify(final, expected)
        else:
            transfer(url(REPOSITORY, REVISION, "UD-IQ4_XS/" + expected["name"]), partial,
                     expected["size"], DESTINATION)
            verify(partial, expected)
            if final.exists():
                raise RuntimeError(f"destination appeared during verification; refusing replacement: {final}")
            publish(partial, final)
            emit("published", file=final.name)
        verified.add(index)
    # A partial-shard invocation never claims completion based on filenames.
    # Hash any other existing finals before writing the completion manifest.
    complete = all((DESTINATION / item["name"]).exists() for item in FILES)
    if complete:
        for index, expected in enumerate(FILES, 1):
            if index not in verified:
                verify(DESTINATION / expected["name"], expected)
                verified.add(index)
        atomic_json(DESTINATION / "all-weights-verified.json", {
            "repository": REPOSITORY, "revision": REVISION, "total_bytes": TOTAL_BYTES,
            "complete": True, "full_file_sha256_verified": True, "files": list(FILES),
            "official_notice_revision": OFFICIAL_REVISION,
            "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    emit("complete" if complete else "selected_shards_complete", all_shards_verified=complete,
         verified_shards=sorted(verified), selected_shards=sorted(selected))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        emit("interrupted", message="Transfer stopped; partial files preserved")
        sys.exit(130)
    except (OSError, ValueError, RuntimeError) as error:
        emit("error", message=str(error))
        sys.exit(1)
