#!/usr/bin/env python3
"""Offline replay of existing JSON artifacts; never opens model weights.

Run from an existing validation workspace: python3 -B tools/analyze_saved_cache_trace.py
Requires the saved two-evaluation trace, header inventory and sixteen-token log.
Only stdout is written; no inference, downloads or model payload reads occur.
"""
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / 'build-baseline/pmi-observed-next.trace.json'
INVENTORY = ROOT / 'docs/target-header-inventory.json'
LOG = ROOT / 'build-baseline/validation-20261003/stage3.log'
CAPACITY = 512 * 1024 ** 2
SERVICE = 8 * 1024 ** 2


def read(path):
    return json.loads(path.read_text())


def charge(size):
    return (size + 63) // 64 * 64


def replay(requests, sizes, capacity):
    entries = collections.OrderedDict()
    occupied = peak = io = hits = misses = evictions = 0
    for key in requests:
        if key in entries:
            hits += 1
            entries.move_to_end(key)
            continue
        misses += 1
        io += sizes[key]
        allocation = charge(sizes[key])
        assert allocation <= capacity
        while occupied + allocation > capacity:
            _, n = entries.popitem(last=False)
            occupied -= n
            evictions += 1
        entries[key] = allocation
        occupied += allocation
        peak = max(peak, occupied)
    return dict(capacity_bytes=capacity, hits=hits, misses=misses,
                io_bytes=io, evictions=evictions,
                current_bytes=occupied, peak_bytes=peak)


def main():
    trace, inventory = read(TRACE), read(INVENTORY)
    routes, weights = {}, {}
    for row in trace['records']:
        for prefix, target in [('ffn_moe_topk-', routes),
                               ('ffn_moe_weights_norm-', weights)]:
            if row['name'].startswith(prefix):
                assert row['shape'] == [10, 1, 1, 1]
                target[row['evaluation'], int(row['name'][len(prefix):])] = row['values']
    assert len(routes) == len(weights) == 96
    component = {}
    for shard in inventory['shards']:
        for tensor in shard['tensors']:
            match = re.fullmatch(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight', tensor['name'])
            if match:
                assert tensor['shape'][2] == 512 and tensor['bytes'] % 512 == 0
                component[int(match[1]), match[2]] = tensor['bytes'] // 512
    assert len(component) == 144
    bundles = {layer: sum(component[layer, p] for p in ('gate', 'up', 'down'))
               for layer in range(48)}
    # Each head owns a disjoint interval; unequal embeddings prove that the two
    # evaluations did not read the same row in that head. Actual IDs are absent.
    metadata = inventory['shards'][0]['metadata']
    offsets = metadata['qwen4exp.ple.head_offsets']['first']
    vocab = metadata['qwen4exp.ple.head_vocab_sizes']['first']
    assert all(offsets[i] + vocab[i] <= offsets[i + 1] for i in range(15))
    ple = {r['evaluation']: r['values'] for r in trace['records'] if r['name'] == 'ple_embd'}
    assert all(len(values) == 16 * 160 for values in ple.values())
    unequal_heads = [h for h in range(16)
                     if ple[0][160*h:160*(h+1)] != ple[1][160*h:160*(h+1)]]
    assert len(unequal_heads) == 16
    requests, sizes = [], {}
    for evaluation in range(2):
        for head in range(16):
            key = ('ple', evaluation, head)
            requests.append(key)
            sizes[key] = 90
        for layer in range(48):
            assert len(set(routes[evaluation, layer])) == 10
            for projection in ('gate', 'up', 'down'):
                for expert in routes[evaluation, layer]:
                    key = ('expert', layer, projection, expert)
                    requests.append(key)
                    sizes[key] = component[layer, projection]

    lru = replay(requests, sizes, CAPACITY)
    observed = trace['metrics']
    for simulated, actual in [('hits', 'cache_hits'), ('misses', 'cache_misses'),
                              ('evictions', 'evictions'), ('io_bytes', 'io_bytes'),
                              ('current_bytes', 'cache_reserved_bytes'),
                              ('peak_bytes', 'cache_peak_reserved_bytes')]:
        assert lru[simulated] == observed[actual], (simulated, lru[simulated], observed[actual])

    prior, distances, repeated = {}, [], []
    for i, key in enumerate(requests):
        if key in prior:
            intervening = set(requests[prior[key] + 1:i])
            distances.append({'key': key,
                              'intervening_distinct_objects': len(intervening),
                              'intervening_distinct_allocated_bytes': sum(charge(sizes[k]) for k in intervening),
                              'minimum_lru_capacity_bytes': charge(sizes[key]) + sum(charge(sizes[k]) for k in intervening)})
            repeated.append(key)
        prior[key] = i
    assert len(repeated) == 813 and all(k[0] == 'expert' for k in repeated)
    compulsory = len(set(requests))
    compulsory_bytes = sum(sizes[k] for k in set(requests))
    reusable_bytes = sum(sizes[k] for k in repeated)
    overlap_by_layer = [len(set(routes[0, layer]) & set(routes[1, layer])) for layer in range(48)]

    top_n = []
    for n in range(1, 5):
        retained = sum(n * v for v in bundles.values())
        reused = [(layer, expert) for layer in range(48) for expert in routes[0, layer][:n]
                  if expert in routes[1, layer]]
        saved = sum(bundles[layer] for layer, expert in reused)
        top_n.append({'top_n_per_layer': n, 'retained_bytes': retained,
                      'service_reserve_bytes': SERVICE, 'total_counted_bytes': retained + SERVICE,
                      'fits_existing_capacity': retained + SERVICE <= CAPACITY,
                      'expert_hits_eval1': len(reused), 'projection_hits_eval1': 3 * len(reused),
                      'saved_logical_bytes': saved, 'eval1_read_saving_fraction': saved / (observed['io_bytes']/2),
                      'two_evaluation_read_saving_fraction': saved / observed['io_bytes'],
                      'two_evaluation_reads_after_policy_bytes': observed['io_bytes'] - saved})

    global_candidates = sorted([(weights[0, layer][slot], layer, expert, bundles[layer])
                                for layer in range(48) for slot, expert in enumerate(routes[0, layer])], reverse=True)
    selected, occupied = [], 0
    for score, layer, expert, n in global_candidates:
        if occupied + n <= CAPACITY - SERVICE:
            selected.append((layer, expert, n))
            occupied += n
    global_saved = sum(n for layer, expert, n in selected if expert in routes[1, layer])
    lines = LOG.read_text().splitlines()
    stage3 = json.loads(next(line.removeprefix('PMI_METRICS ') for line in lines if line.startswith('PMI_METRICS ')))
    result = {
        'scope': 'Two existing single-token evaluations, offline metadata/trace replay only; no model weight reads.',
        'inputs': [{'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in (TRACE, INVENTORY, LOG)],
        'request_order': 'PLE first; ascending layers; gate, up, down component sweeps; original routing-slot order.',
        'identity_note': 'Expert keys reconstruct exact distinct shard ranges using layer/component/expert; PLE symbolic keys are proven distinct by disjoint head ranges and unequal head outputs.',
        'lru_replay_matches_all_observed_counters': True,
        'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'lru': lru,
        'expert_bytes_per_evaluation': sum(bundles.values()) * 10,
        'ple_bytes_per_evaluation': 1440,
        'ple_allocation_bytes_per_evaluation': 2048,
        'largest_component_bytes': max(component.values()),
        'overlap_experts': sum(overlap_by_layer),
        'overlap_by_layer': overlap_by_layer,
        'overlap_bytes': reusable_bytes,
        'miss_classification_empty_cache': {
            'compulsory_requests': compulsory, 'compulsory_bytes': compulsory_bytes,
            'capacity_requests': len(repeated), 'capacity_bytes': reusable_bytes,
            'other_requests': 0,
            'note': 'Capacity versus infinite retention for this successful exact-range stream; not hardware-cache conflict misses.'},
        'reuse_distance': {
            'minimum_lru_capacity_bytes_min': min(d['minimum_lru_capacity_bytes'] for d in distances),
            'minimum_lru_capacity_bytes_median': statistics.median(d['minimum_lru_capacity_bytes'] for d in distances),
            'minimum_lru_capacity_bytes_max': max(d['minimum_lru_capacity_bytes'] for d in distances),
            'intervening_distinct_objects_min': min(d['intervening_distinct_objects'] for d in distances),
            'intervening_distinct_objects_max': max(d['intervening_distinct_objects'] for d in distances),
            'all_reuse_distances_exceed_512MiB': all(d['minimum_lru_capacity_bytes'] > CAPACITY for d in distances)},
        'counterfactual_lru_capacities': [replay(requests, sizes, c * 1024**2) for c in (768, 1024, 1536)],
        'causal_per_layer_top_n': top_n,
        'rolling_top4_transition_budget': {
            'prior_protected_bytes': top_n[3]['retained_bytes'],
            'additional_current_layer_bytes': 4 * max(bundles.values()),
            'service_reserve_bytes': SERVICE,
            'conservative_total_bytes': top_n[3]['retained_bytes'] + 4 * max(bundles.values()) + SERVICE,
            'note': 'Retain newly arriving current-top4 pages while previous-top4 remains pinned through the full layer, then retire old pins; otherwise rotation may introduce hidden rereads.'},
        'causal_policy_contract': 'At each eval0 layer router, retain its highest-ranked N experts, all three separate component pages. Nonprotected requests use the globally charged 8MiB transient reserve. Keep prior selection protected through the full current-layer pass; only then rotate it. Do not prematurely replace with current top-N and lose prior experts now ranked N+1..10.',
        'global_score_counterfactual': {
            'label': 'Offline full-eval0 score selection, not an implemented online admission policy; evicted pages cannot be recovered free.',
            'retained_experts': len(selected), 'retained_bytes': occupied,
            'with_service_reserve_bytes': occupied + SERVICE,
            'saved_logical_bytes': global_saved},
        'oracle_upper_bound': {
            'service_reserve_bytes': SERVICE, 'maximum_saved_bytes': min(reusable_bytes, CAPACITY-SERVICE),
            'label': 'Upper bound using unavailable future route knowledge; not an achievable policy result or exact knapsack optimum.'},
        'stage3_scope': {
            'evaluations': len(stage3['evaluations']), 'io_bytes': stage3['io_bytes'],
            'expert_requests': stage3['expert_requests'], 'ple_requests': stage3['lookup_requests'],
            'ple_logical_bytes': stage3['lookup_requests'] * 90,
            'cache_hits': stage3['cache_hits'], 'cache_misses': stage3['cache_misses'],
            'process_storage_read_bytes': stage3['process_storage_read_bytes'],
            'evaluation_storage_read_bytes': sum(e['process_storage_read_bytes_delta'] for e in stage3['evaluations']),
            'nondeferred_physical_tensor_bytes': inventory['summary']['non_expert_dense_weight_bytes'],
            'note': 'No routes or PLE row IDs captured; cannot classify misses or extrapolate top-N reuse from this run.'},
        'limits': ['Payload savings are not timing predictions.',
                   'Logical provider reads differ from OS/process storage reads; provider misses can hit the OS page cache.',
                   'The 8MiB service reserve and protected payload count inside 512MiB; cache metadata/backend buffers remain outside the runtime payload budget.',
                   'Larger-cache replays are counterfactual analysis, not permission or recommendations to raise resource limits.',
                   'PLE has no repeated row in the two-evaluation trace, so its byte-saving ceiling here is zero.',
                   'Single-token grouping creates no within-layer expert reuse; prefill grouping needs a separate multi-token workload.']}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
