"""Frozen-residency complete-graph coverage; generated F32 weights only."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('full', ROOT / 'tests/full_model_test.py')
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pmi', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--vulkan', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    shards = full.fixture.generate(a.output / 'split.gguf', shards=2)
    common = ['--prompt', 'hello world', '--max-tokens', '8', '--context', '64',
              '--prefill-chunk', '1', '--cpu-repack', 'off', '--trace-mode', 'compact']

    def run(name, command, flags):
        trace = a.output / (name + '.json')
        result = subprocess.run([str(a.pmi.resolve()), command,
                                *map(str, shards if command == 'run' else shards[:1]), *common, *(['--profile-paging','on'] if command == 'run' else []), *flags,
                                '--trace-json', str(trace)], capture_output=True, text=True, timeout=60)
        (a.output / (name + '.stderr')).write_text(result.stderr)
        assert result.returncode == 0, result.stderr[-4000:]
        return json.loads(trace.read_text())

    reference = run('reference', 'reference', [])
    sequence = a.output / 'tokens.txt'
    sequence.write_text(' '.join(map(str, reference['tokens'])) + '\n')
    backends = ['device-cpu'] + (['device-vulkan'] if a.vulkan else [])
    for backend in backends:
        for capacity in (1, 8192, 1048576):
            actual = run(f'{backend}-{capacity}', 'run', [
                '--teacher-forcing-file', str(sequence), '--projection-backend', backend,
                '--projection-dispatch', 'split', '--cache-bytes', '8192',
                '--device-cache-bytes', str(capacity), '--device-graph-bytes', '65536'])
            full.compare(reference, actual, compact=True)
            m = actual['metrics']
            assert m['split_enabled'] and m['split_groups'] > 0
            assert m['split_cpu_projections'] + m['split_device_projections'] == m['expert_requests']
            dec = [e for e in m['evaluations'] if e['phase'] == 'decode']
            assert len(dec) == 7
            # Device leases are probed after evaluation0; misses must never upload.
            assert all(e['device_projection']['cache']['uploads'] == 0 for e in dec)
            assert all(e['device_projection']['cache']['provider_requests'] == 0 for e in dec)
            assert m['device_projection']['cache']['evictions'] == 0
            if capacity == 1:
                assert m['split_device_projections'] == 0
            else:
                assert m['split_device_projections'] > 0
            if capacity == 8192:
                assert m['split_cpu_projections'] > 0
            if backend == 'device-cpu':
                assert actual['predictions'] == reference['tokens']
    print('Split complete-graph fixtures passed: all-CPU/mixed/resident groups, frozen uploads, logits/routing/intermediates/tokens; ' + ','.join(backends))


if __name__ == '__main__':
    main()
