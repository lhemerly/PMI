"""Research CLI: trusted operator specification, never model-generated controls."""
import argparse
import json
from pathlib import Path
from .harness import Store, RealAgent, configuration, digest, drive


def summary(state):
    return {'schema': state['schema'], 'status': state['status'], 'reason': state['reason'],
            'phase': state['phase'], 'model_calls': state['calls'],
            'candidate_count': len(state['candidates']), 'elapsed_seconds': state['elapsed_seconds'],
            'cancel_requested': state['cancel_requested'], 'retry_blocked': state.get('retry_blocked', False), 'inflight': state['inflight'],
            'actual_inference': state['actual_inference'], 'backend_mode': state['backend_mode'],
            'evidence_kind': 'RealAdapterEvidence' if state['actual_inference'] is True else 'ContractOnlyOrUnrun',
            'best_candidate': None if state['best_id'] is None else state['candidates'][state['best_id'] - 1],
            'config_sha256': state['config_sha256'],
            'completion_scope': 'declared evaluator checks only',
            'evaluator': state['config']['evaluator'], 'evaluator_sha256': digest(state['config']['evaluator'])}


def main(argv):
    from content_factory import bounded_read, write_new
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['research-plan', 'research-start', 'research-status', 'research-resume', 'research-cancel'])
    parser.add_argument('--spec-file', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--lease', type=Path)
    args = parser.parse_args(argv)
    if args.action in ('research-plan', 'research-start'):
        if args.spec_file is None:
            parser.error('--spec-file required')
        config = configuration(json.loads(bounded_read(args.spec_file, 65536)))
        if args.action == 'research-plan':
            result = {'schema': 1, 'mode': 'research', 'config': config, 'config_sha256': digest(config),
                      'actual_inference': False, 'execution_held': True,
                      'remaining': ['exact workload execution authority', 'new prompt token-fit qualification', 'fresh shared resource admission'],
                      'evaluator_supported': config['evaluator']['id'] in ('text-checks-v1', 'exact-text-v1')}
            if args.out is not None:
                write_new(args.out, result)
            print(json.dumps(result, indent=2)); return 0
        if args.out is None or args.lease is None:
            parser.error('--out and --lease required')
        store = Store(args.out); store.create(config)
        state = drive(store, RealAgent(), args.lease)
    else:
        if args.run_dir is None:
            parser.error('--run-dir required')
        store = Store(args.run_dir)
        if args.action == 'research-status':
            print(json.dumps(summary(store.load()), indent=2)); return 0
        if args.action == 'research-cancel':
            print(json.dumps(summary(store.cancel()), indent=2)); return 0
        if args.lease is None:
            parser.error('--lease required')
        state = drive(store, RealAgent(), args.lease)
    print(json.dumps(summary(state), indent=2))
    return {'complete': 0, 'blocked': 2, 'budget_exhausted': 3, 'cancelled': 130}.get(state['status'], 0)
