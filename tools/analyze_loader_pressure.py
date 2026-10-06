#!/usr/bin/env python3
"""Summarize saved guarded startup traces offline; never start a model."""
import argparse
import json
from pathlib import Path


def metric(group, name, field='delta_from_baseline'):
    item = group.get(name)
    return item.get(field) if isinstance(item, dict) else None


def summarize(path, startup_seconds=4.0):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    final = next((r for r in reversed(rows) if r.get('event') == 'final'), None)
    samples = []
    for row in rows:
        if row.get('event') != 'sample' or row.get('elapsed_seconds', float('inf')) > startup_seconds:
            continue
        # TERM cleanup can free the model. Preserve only samples with a live
        # owned member; do not mistake post-exit free RAM for loading headroom.
        if not any(member.get('alive') for member in row.get('members', [])):
            continue
        obs = row.get('observations') or {}
        vm = obs.get('vmstat') or {}
        cg = obs.get('cgroup_v2') or {}
        pressure = cg.get('memory_pressure') or {}
        guards = row.get('pressure_guards') or {}
        cg_bytes = cg.get('stat_bytes') or {}
        mem = row.get('meminfo_bytes') or {}
        samples.append({
            'elapsed_seconds': row.get('elapsed_seconds'),
            'owned_rss_bytes': row.get('group_rss_bytes'),
            'owned_swap_bytes': row.get('group_swap_bytes'),
            'available_bytes': mem.get('MemAvailable'), 'free_bytes': mem.get('MemFree'),
            'cached_bytes': mem.get('Cached'), 'anonymous_bytes': mem.get('AnonPages'),
            'owned_storage_read_bytes': (row.get('sampled_io_bytes') or {}).get('read_bytes'),
            'host_full_stall_us': (guards.get('growth') or {}).get('psi_full_us'),
            'host_swapout_bytes': (guards.get('growth') or {}).get('swapout_bytes'),
            'host_kswapd_scanned_pages': metric(vm, 'pgscan_kswapd_pages'),
            'host_direct_scanned_pages': metric(vm, 'pgscan_direct_pages'),
            'context_full_stall_us': metric(pressure.get('full') or {}, 'total_stall_us'),
            'context_anonymous_growth_bytes': metric(cg_bytes, 'anon'),
            'context_anon_thp_growth_bytes': metric(cg_bytes, 'anon_thp'),
            'context_swap_growth_bytes': metric(cg.get('usage_bytes') or {}, 'memory.swap.current'),
            'context_major_faults': metric(cg.get('stat_counts') or {}, 'pgmajfault'),
            'context_unevictable_bytes': metric(cg_bytes, 'unevictable', 'value'),
            'context_attribution': cg.get('attribution'),
            'owned_member_states': [m.get('state') for m in row.get('members', []) if m.get('alive')],
        })
    return {'host_log': str(path), 'startup_window_seconds': startup_seconds,
            'sample_count': len(samples), 'samples': samples,
            'initial_live_sample': samples[0] if samples else None,
            'last_live_sample': samples[-1] if samples else None,
            'final': {k: final.get(k) for k in ('status','reason','elapsed_seconds','remaining_live_pids')} if final else None,
            'limits': ['Host counters include other workloads; shared context is not process attribution.',
                       'Counters are sampled, not atomic. Missing values stay null.',
                       'Read bytes and D state cannot distinguish ordinary I/O from reclaim/compaction stalls.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('logs', nargs='+', type=Path)
    p.add_argument('--startup-seconds', type=float, default=4.0)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not 0 < a.startup_seconds <= 60:
        p.error('Startup window must be positive and <=60 seconds')
    with a.output.open('x') as file:
        json.dump([summarize(path, a.startup_seconds) for path in a.logs], file, indent=2, allow_nan=False)
        file.write('\n')


if __name__ == '__main__':
    main()
