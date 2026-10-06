#!/usr/bin/env python3
"""Exercise the production retention helper using saved routes and metadata only.

The test executable simulates byte-LRU storage and invokes ProtectedPageRetention
for actual pin/rotation decisions. No model payload file is opened. This proves
policy behavior for the saved transition, not numerical or physical-I/O results.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CAPACITY = 512 * 1024**2
SERVICE = 8 * 1024**2


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def bounded_json(path):
    limit = 64 * 1024**2
    check(path.stat().st_size <= limit, 'replay metadata exceeds 64 MiB')
    with path.open('rb') as source:
        data = source.read(limit + 1)
    check(len(data) <= limit, 'replay metadata grew past 64 MiB')
    return json.loads(data)


def prepare(trace, inventory):
    routes, ple = {}, {}
    for row in trace['records']:
        match = re.fullmatch(r'ffn_moe_topk-(\d+)', row['name'])
        if match:
            key = row['evaluation'], int(match[1])
            check(key not in routes and row['type'] == 'i32' and row['shape'] == [10, 1, 1, 1],
                  'expected unique single-token I32 top-ten records')
            values = row['values']
            check(len(values) == 10 and len(set(values)) == 10 and
                  all(type(v) is int and 0 <= v < 512 for v in values), 'invalid target routing IDs')
            routes[key] = values
        elif row['name'] == 'ple_embd':
            check(row['evaluation'] not in ple, 'duplicate PLE record')
            check(row['type'] == 'f32' and row['shape'] == [2560, 1, 1, 1] and
                  all(type(v) in (int, float) and math.isfinite(v) for v in row['values']),
                  'PLE proof requires finite F32 head embeddings')
            ple[row['evaluation']] = row['values']
    check(set(routes) == {(evaluation, layer) for evaluation in range(2) for layer in range(48)},
          'replay requires the frozen two-evaluation, 48-layer trace')
    check(set(ple) == {0, 1} and all(len(v) == 2560 for v in ple.values()), 'missing full PLE head embeddings')
    metadata = inventory['shards'][0]['metadata']
    offsets = metadata['qwen4exp.ple.head_offsets']['first']
    vocab = metadata['qwen4exp.ple.head_vocab_sizes']['first']
    check(len(offsets) == len(vocab) == 16 and
          all(type(o) is int and type(n) is int and o >= 0 and n > 0 and o + n <= 320001536
              for o, n in zip(offsets, vocab)) and
          all(offsets[i] + vocab[i] <= offsets[i + 1] for i in range(15)), 'PLE head ranges are not disjoint')
    check(all(ple[0][160*h:160*(h+1)] != ple[1][160*h:160*(h+1)] for h in range(16)),
          'symbolic unique PLE keys require unequal saved head outputs')
    components = {}
    ple_bytes = None
    for shard_id, shard in enumerate(inventory['shards']):
        check(shard['metadata']['split.no'] == shard_id, 'inventory shard order differs from split.no')
        for tensor in shard['tensors']:
            check(tensor['absolute_offset'] == shard['tensor_data_offset'] + tensor['relative_offset'],
                  'inventory contains inconsistent tensor offsets')
            match = re.fullmatch(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight', tensor['name'])
            if match:
                check(tensor['shape'][-1] == 512 and tensor['bytes'] % 512 == 0, 'unsupported packed expert layout')
                layer, projection = int(match[1]), match[2]
                check((layer, projection) not in components, 'duplicate expert component')
                components[layer, projection] = shard_id, tensor['absolute_offset'], tensor['bytes'] // 512
            elif tensor['name'] == 'per_layer_token_embd.weight':
                check(tensor['shape'] == [160, 320001536] and tensor['bytes'] % tensor['shape'][1] == 0,
                      'unsupported PLE shape/size')
                ple_bytes = tensor['bytes'] // tensor['shape'][1]
    check(ple_bytes == 90, 'saved PLE row size differs from target')
    projections = ('gate', 'up', 'down')

    def expert_range(layer, projection, expert):
        shard, start, length = components[layer, projection]
        return shard, start + expert * length, length

    bundle = {layer: sum(components[layer, p][2] for p in projections) for layer in range(48)}
    reused = [(layer, expert) for layer in range(48) for expert in routes[0, layer][:4]
              if expert in routes[1, layer]]
    expected_hits = 3 * len(reused)
    expected_saved = sum(bundle[layer] for layer, _ in reused)
    metrics = trace['metrics']
    check(metrics['cache_capacity_bytes'] == CAPACITY, 'saved cache capacity is not the frozen 512 MiB')
    counts = [metrics[name] for name in ('cache_hits', 'cache_misses', 'evictions', 'io_bytes',
                                       'cache_reserved_bytes', 'cache_peak_reserved_bytes')]
    events = ['PMI_RETENTION_REPLAY_V1', f'config {CAPACITY} {SERVICE}',
              'expect ' + ' '.join(map(str, counts + [expected_hits, metrics['io_bytes'] - expected_saved]))]
    for evaluation in range(2):
        # No row IDs were captured. These 32 labels are proven pairwise distinct
        # by disjoint head ranges and changed corresponding embeddings. They
        # preserve object identity/size for cache replay, not real PLE offsets.
        for head in range(16):
            events.append(f'service {2**32 - 1} {(evaluation * 16 + head) * 128} {ple_bytes}')
        for layer in range(48):
            selected = routes[evaluation, layer][:4]
            events.append(f'begin {layer} {len(selected) * 3}')
            for expert in selected:
                for projection in projections:
                    events.append('want ' + ' '.join(map(str, expert_range(layer, projection, expert))))
            for projection in projections:
                for expert in routes[evaluation, layer]:
                    events.append('load ' + ' '.join(map(str, expert_range(layer, projection, expert))))
            events.append(f'finish {layer}')
    events.append('end')
    stream = '\n'.join(events) + '\n'
    check(len(stream.encode()) <= 1024 * 1024, 'metadata replay stream exceeds 1 MiB')
    return stream, dict(top_n=4, predicted_projection_hits=expected_hits,
                        predicted_avoided_logical_bytes=expected_saved,
                        predicted_io_bytes=metrics['io_bytes'] - expected_saved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build-baseline/pmi_retention_replay')
    parser.add_argument('--trace', type=Path, default=ROOT / 'build-baseline/pmi-observed-next.trace.json')
    parser.add_argument('--inventory', type=Path, default=ROOT / 'docs/target-header-inventory.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'build-baseline/cache-optimization/retention-replay.json')
    args = parser.parse_args()
    check(not args.output.exists(), 'output already exists; choose a new result path')
    stream, prediction = prepare(bounded_json(args.trace), bounded_json(args.inventory))
    completed = subprocess.run([str(args.binary.resolve())], input=stream, capture_output=True,
                               text=True, timeout=30, check=False)
    check(completed.returncode == 0, f'replay executable failed: {completed.stderr.strip()}')
    result = dict(scope='Saved two-evaluation routing replay: production retention helper, metadata-only LRU provider.',
                  inputs=[dict(path=str(path), sha256=digest(path)) for path in (args.trace, args.inventory)],
                  executable_sha256=digest(args.binary), producer_sha256=digest(Path(__file__)),
                  replay_stream_sha256=hashlib.sha256(stream.encode()).hexdigest(),
                  replay_stream_bytes=len(stream.encode()), prediction=prediction,
                  result=json.loads(completed.stdout),
                  limitations=[
                      'No model payload reads, Rust cache execution, numerical inference, or timing/performance measurement.',
                      'Actual expert byte ranges are reconstructed from the saved header inventory and routing IDs.',
                      'PLE uses 32 symbolic distinct 90-byte ranges: disjoint heads and changed saved head outputs prove distinctness; actual row IDs were not captured.',
                      'Metadata cache uses exact-range byte-LRU with counted pins and temporary leases; all six unmodified baseline counters must match before policy replay.',
                      'Logical read savings apply only to this saved transition; they do not imply physical-I/O or latency savings.'
                  ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps(dict(output=str(args.output), **result['result'])))


if __name__ == '__main__':
    main()
