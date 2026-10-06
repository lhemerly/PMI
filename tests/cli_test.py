"""Exercise the public layer command against independently generated golden data."""
import json
from pathlib import Path
import subprocess
import sys

pmi = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
missing = root / 'must-not-open-projection-activation.gguf'
assert not missing.exists(), 'option-validation sentinel must not be a model file'
budgets = ['--device-cache-bytes', '16384', '--device-graph-bytes', '16384']
activation_errors = [
    ('run', ['--projection-backend', 'device-vulkan', '--projection-device-index', '4294967295',
             *budgets, '--projection-activation', 'unknown'],
     'projection-activation must be backend-native or cpu-compatible'),
    ('reference', ['--projection-activation', 'backend-native'],
     'projection-activation is only valid for the run command'),
    ('reference', ['--projection-activation', 'cpu-compatible'],
     'projection-activation is only valid for the run command'),
    ('run', ['--projection-activation', 'cpu-compatible'],
     'cpu-compatible projection-activation requires run with device-vulkan'),
    ('run', ['--projection-backend', 'cpu', '--projection-activation', 'cpu-compatible'],
     'cpu-compatible projection-activation requires run with device-vulkan'),
    ('run', ['--projection-backend', 'device-cpu', *budgets, '--projection-activation', 'cpu-compatible'],
     'cpu-compatible projection-activation requires run with device-vulkan'),
]
for command, flags, diagnostic in activation_errors:
    rejected = subprocess.run([str(pmi), command, str(missing), '--prompt', 'hello', *flags],
                              capture_output=True, text=True, timeout=10)
    assert rejected.returncode != 0 and not rejected.stdout, 'invalid projection activation was accepted'
    assert rejected.stderr.strip() == 'error: ' + diagnostic, rejected.stderr

reports = []
for fixture in sorted(p for p in root.iterdir() if p.is_dir()):
    shards = sorted(fixture.glob('*.gguf'))
    if len(shards) != 2:
        raise AssertionError(f'incomplete fixture: {fixture}')
    args = [str(pmi), 'moe-layer', *map(str, shards), '--input', str(fixture / 'input.txt'),
            '--cache-bytes', '4194304', '--repeat', '2']
    rows = [json.loads(line) for line in subprocess.check_output(args, text=True).splitlines()]
    golden = [float(value) for value in (fixture / 'expected-output.txt').read_text().split()]
    assert len(rows) == 2 and rows[0]['io_bytes'] > 0
    assert rows[1]['io_bytes'] == 0 and rows[1]['hit_rate'] == 1
    for row in rows:
        assert row['expert_ids'] == [1, 2]
        assert row['cache_peak_reserved_bytes'] <= 4194304
        assert len(row['output']) == len(golden)
        assert all(abs(a - b) <= 3e-7 + 3e-5 * abs(b) for a, b in zip(row['output'], golden))
    reversed_args = [str(pmi), 'moe-layer', *map(str, reversed(shards)), '--input', str(fixture / 'input.txt')]
    reversed_row = json.loads(subprocess.check_output(reversed_args, text=True))
    assert reversed_row['expert_ids'] == rows[0]['expert_ids']
    assert reversed_row['output'] == rows[0]['output']
    reports.append({'fixture': fixture.name, 'iterations': rows})
    missing = subprocess.run([str(pmi), 'inspect', str(shards[0])], capture_output=True, text=True)
    assert missing.returncode != 0, 'incomplete shard set was accepted'
    malformed = fixture / 'malformed-input.txt'
    malformed.write_text((fixture / 'input.txt').read_text().strip() + ' 1e')
    rejected = subprocess.run([str(pmi), 'moe-layer', *map(str, shards), '--input', str(malformed)],
                              capture_output=True, text=True)
    assert rejected.returncode != 0 and not rejected.stdout, 'incomplete trailing float was accepted'

(root.parent / 'cli-metrics.json').write_text(json.dumps(reports, indent=2) + '\n')
print(f'CLI verified {len(reports)} formats: golden outputs, shard-order invariance, warm hits, missing-shard errors; '
      f'{len(activation_errors)} projection-activation errors rejected before model access')
