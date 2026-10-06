"""Check fixed decoder inputs against resident execution on a tiny model."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("full_model", ROOT / "tests/full_model_test.py")
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pmi", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--vulkan", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    shards = full.fixture.generate(args.output / "forcing.gguf", shards=2)
    common = ["--prompt", "hello world", "--max-tokens", "8", "--context", "64",
              "--prefill-chunk", "1", "--cpu-repack", "off", "--trace-mode", "compact"]

    def run(name, command, flags):
        trace = args.output / (name + ".json")
        proc = subprocess.run([str(args.pmi.resolve()), command, *map(str, shards if command == "run" else shards[:1]),
                               *common, *flags, "--trace-json", str(trace)],
                              capture_output=True, text=True, timeout=60)
        (args.output / (name + ".stderr")).write_text(proc.stderr)
        assert proc.returncode == 0, proc.stderr[-4000:]
        data = json.loads(trace.read_text())
        return data

    ordinary = run("ordinary", "reference", [])
    baseline = ordinary["tokens"]
    assert len(baseline) == 8
    sequence = args.output / "tokens.txt"
    sequence.write_text(" ".join(map(str, baseline)) + "\n")
    same = run("same", "reference", ["--teacher-forcing-file", str(sequence)])
    assert same["tokens"] == same["predictions"] == baseline
    assert same["metrics"]["teacher_forcing"] and same["metrics"]["prediction_matches"] == 8
    full.compare(ordinary, same, compact=True)

    # Change an input after the common prefill: printing supplied IDs without
    # actually feeding them to the decoder must fail the changed-logits check.
    changed = baseline.copy()
    changed[0] = 270 if baseline[0] != 270 else 271
    sequence.write_text(" ".join(map(str, changed)) + "\n")
    native = run("changed-reference", "reference", ["--teacher-forcing-file", str(sequence)])
    assert native["tokens"] == changed and len(native["predictions"]) == 8
    old_logits = [r for r in ordinary["records"] if r["name"] == "result_output"]
    new_logits = [r for r in native["records"] if r["name"] == "result_output"]
    assert old_logits[0]["values"] == new_logits[0]["values"], "prefill changed before a forced input was fed"
    assert any(a["values"] != b["values"] for a, b in zip(old_logits[1:], new_logits[1:])), "forced inputs were ignored"
    backends = ["cpu", "device-cpu"] + (["device-vulkan"] if args.vulkan else [])
    for backend in backends:
        flags = ["--teacher-forcing-file", str(sequence), "--projection-backend", backend, "--cache-bytes", "8192"]
        if backend != "cpu":
            flags += ["--device-cache-bytes", "16384", "--device-graph-bytes", "65536"]
        if backend == "device-vulkan":
            flags += ["--projection-device-index", "0"]
        actual = run(backend, "run", flags)
        assert actual["tokens"] == changed
        assert actual["metrics"]["prefill_evaluations"] == ordinary["metrics"]["prefill_evaluations"]
        assert actual["metrics"]["decode_evaluations"] == 7
        full.compare(native, actual, compact=True)
        if backend != "device-vulkan":
            assert actual["predictions"] == native["predictions"]

    sentinel = args.output / "must-not-open.gguf"
    for text, diagnostic in [("", "exactly max-tokens"), ("11", "exactly max-tokens"),
                             ("-1 " * 8, "nonnegative integer"), ("2147483648 " * 8, "out of range"),
                             ("1x " * 8, "nonnegative integer")]:
        sequence.write_text(text)
        proc = subprocess.run([str(args.pmi.resolve()), "run", str(sentinel), *common,
                               "--teacher-forcing-file", str(sequence)], capture_output=True, text=True, timeout=10)
        assert proc.returncode and not proc.stdout and diagnostic in proc.stderr, proc.stderr
    sequence.write_text("999999 " * 8)
    proc = subprocess.run([str(args.pmi.resolve()), "run", *map(str, shards), *common,
                           "--teacher-forcing-file", str(sequence)], capture_output=True, text=True, timeout=10)
    assert proc.returncode and not proc.stdout and "outside model vocabulary" in proc.stderr
    print("Teacher forcing verified: changed inputs, exact native/CPU parity, prediction reporting, bounds; backends=" + ",".join(backends))


if __name__ == "__main__":
    main()
