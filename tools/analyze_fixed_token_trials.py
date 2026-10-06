#!/usr/bin/env python3
"""Read completed guarded trials; produce reproducible timing/cache evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def final(path):
    return json.loads(path.read_text().splitlines()[-1])


def digest(path):
    with path.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': sha}


def trial(directory):
    directory = directory.resolve()
    m = read(directory / 'metrics.json')
    command = read(directory / 'command.json')
    host, vram = final(directory / 'host.jsonl'), final(directory / 'vram.jsonl')
    assert host['status'] == vram['status'] == 'completed'
    assert not host['remaining_live_pids'] and not host['monitor_errors']
    assert host['pressure_guards']['source_error'] is None
    assert read(directory / 'run.json')['returncode'] == 0
    rows = (directory / 'trial.log').read_text().splitlines()
    tokens = [list(map(int, row.split()[1:])) for row in rows if row.startswith('TOKENS ')]
    predictions = [list(map(int, row.split()[1:])) for row in rows if row.startswith('PREDICTIONS ')]
    assert tokens == [command['tokens']] and len(predictions) == 1
    assert m['teacher_forcing'] and m['generated_tokens'] == len(tokens[0]) == len(predictions[0])
    assert m['prediction_matches'] == sum(a == b for a, b in zip(tokens[0], predictions[0]))
    assert not m['cpu_repack'] and m['profile_enabled'] and m['cache_policy'] == 'lru'
    assert m['cache_capacity_bytes'] == 536870912
    assert m['dense_load']['effective'] == 'direct-required' and not m['dense_load']['fallback_count']
    evaluations = m['evaluations']
    dec = [e for e in evaluations if e['phase'] == 'decode']
    assert len(dec) == len(tokens[0]) - 1 and len(evaluations) == len(tokens[0])
    assert all(e['input_tokens'] == 1 for e in evaluations)
    assert math.isclose(sum(e['elapsed_ms'] for e in dec), m['decode_ms'], abs_tol=1e-5)
    for key in ('provider_read_ms', 'allocation_ms', 'upload_ms', 'eviction_ms'):
        assert math.isclose(sum(e['device_projection']['cache'][key] for e in evaluations),
                            m['device_projection']['cache'][key], abs_tol=1e-5)
    storage = [e['process_storage_read_bytes_delta'] for e in evaluations]
    assert all(value is not None and value >= 0 for value in storage)
    d = m['device_projection']
    if m['projection_backend'] == 'device-vulkan':
        assert m['projection_activation'] == 'backend-native' and not d['activation_preparation_calls']
        assert d['calls'] == d['completed'] == 23040 and not d['failures'] and not d['cache']['failures']
        assert d['cache']['profile_enabled'] and not d['cache']['active_leases']
    summary = {'path': str(directory.relative_to(ROOT)), 'backend': m['projection_backend'],
               'device_cache_capacity_bytes': d['cache']['capacity_bytes'],
               'setup_ms': m['setup_ms'], 'prefill_ms': m['prefill_ms'], 'decode_ms': m['decode_ms'],
               'decode_evaluations_per_second': len(dec) * 1000 / m['decode_ms'],
               'decode_latency_median_ms': statistics.median(e['elapsed_ms'] for e in dec),
               'evaluation_storage_read_bytes': sum(storage),
               'decode_storage_read_bytes': sum(e['process_storage_read_bytes_delta'] for e in dec),
               'provider_bytes': m['io_bytes'], 'expert_upload_bytes': d['cache']['upload_bytes'],
               'device_hit_rate': d['cache']['hits'] / d['cache']['requests'] if d['cache']['requests'] else None,
               'process_peak_rss_bytes': m['peak_rss_bytes'],
               'sampled_group_peak_rss_bytes': host['sampled_peak_rss_bytes'],
               'sampled_global_vram_peak_bytes': vram['sampled_peak_global_vram_bytes'],
               'minimum_available_bytes': host['minimum_mem_available_bytes'],
               'pressure_growth': host['pressure_guards']['sampled_peak_growth'],
               'owned_group_peak_swap_bytes': host['sampled_peak_group_swap_bytes'],
               'global_swap_growth_bytes': host['sampled_peak_system_swap_growth_bytes']}
    return {'summary': summary, 'metrics': m, 'tokens': tokens[0], 'predictions': predictions[0],
            'command': command, 'host_final': host, 'vram_final': vram}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('trials', type=Path, nargs='+')
    args = parser.parse_args()
    trials = [trial(p) for p in args.trials]
    assert all(t['tokens'] == trials[0]['tokens'] for t in trials)
    assert len({t['command']['binary']['sha256'] for t in trials}) == 1
    assert len({t['command']['token_file']['sha256'] for t in trials}) == 1
    assert len({t['command']['manifest']['sha256'] for t in trials}) == 1
    groups = {}
    for t in trials:
        s = t['summary']
        name = 'cpu' if s['backend'] == 'cpu' else f"vulkan-{s['device_cache_capacity_bytes']//1024**2}MiB"
        groups.setdefault(name, []).append(t)
    stats = {}
    for name, values in groups.items():
        times = [t['summary']['decode_ms'] for t in values]
        stats[name] = {'runs': len(values), 'decode_ms_mean': statistics.mean(times),
                       'decode_ms_min': min(times), 'decode_ms_max': max(times),
                       'decode_ms_population_stddev': statistics.pstdev(times) if len(times) > 1 else None,
                       'decode_evaluations_per_second_from_mean': 15000 / statistics.mean(times),
                       'all_evaluations_storage_read_bytes': sum(t['summary']['evaluation_storage_read_bytes'] for t in values)}
    gpu = [t for t in trials if t['summary']['backend'] == 'device-vulkan']
    all_warm = all(t['summary']['evaluation_storage_read_bytes'] == 0 for t in trials)
    same_gpu_predictions = all(t['predictions'] == gpu[0]['predictions'] for t in gpu) if gpu else None
    report = {'schema_version': 1, 'scope': 'single-prompt16-token native-precision fixed-input trials',
              'all_evaluation_storage_counters_zero': all_warm,
              'gpu_predictions_match_across_cache_sizes': same_gpu_predictions,
              'quality_claim': None, 'hardware_speedup_claim': None,
              'groups': stats, 'trials': trials,
              'limitations': ['Only one Hello prompt and15decode evaluations perrun; teacher-forced text is not generation quality.',
                              'Identical input tokens do not force identical expert routing across different backend arithmetic.',
                              'OS-warm means zero sampled process read_bytes deltas; no file cache was flushed and no OS-cold baseline was established.',
                              'Repeat spans a user gaming pause; clocks/thermals/desktop load are uncontrolled.',
                              'Global VRAM includes desktop; RAM/VRAM peaks are sampled; storage bytes are process accounting, not unique SSD traffic.',
                              'Cache provider timer includes host cache work; upload is synchronous API wall time including waits, not pure PCIe time.',
                              'Larger cache adds memory; neither256MiB nor1536MiB results imply a general hardware ranking.']}
    if 'vulkan-256MiB' in groups and 'vulkan-1536MiB' in groups:
        a, b = stats['vulkan-256MiB'], stats['vulkan-1536MiB']
        report['observed_decode_time_reduction_fraction'] = 1 - b['decode_ms_mean'] / a['decode_ms_mean']
        baseline_upload = groups['vulkan-256MiB'][0]['summary']['expert_upload_bytes']
        optimized_upload = groups['vulkan-1536MiB'][0]['summary']['expert_upload_bytes']
        report['observed_upload_reduction_fraction'] = 1 - optimized_upload / baseline_upload
    files = [Path(__file__).resolve()]
    for p in args.trials:
        files += [file for file in p.resolve().iterdir() if file.is_file()]
    report['artifacts'] = [digest(file) for file in sorted(set(files))]
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'groups': stats, 'all_warm': all_warm, 'same_gpu_predictions': same_gpu_predictions,
                      'observed_decode_time_reduction_fraction': report.get('observed_decode_time_reduction_fraction')}))


if __name__ == '__main__':
    main()
