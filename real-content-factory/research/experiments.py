"""Typed experiment boundary. No candidate-code executor is enabled."""
from dataclasses import dataclass, asdict
import hashlib
import math
from pathlib import PurePosixPath

EVALUATOR = {'id': 'interval-union-v1', 'contract': 'integer half-open intervals; sorted disjoint union; no input mutation',
             'required_speedup': 1.10, 'correctness': 'all frozen exhaustive, edge and seeded randomized cases',
             'benchmark': 'paired baseline/candidate medians on frozen mixed profiles'}


from pathlib import Path
CORRECTNESS_EVALUATOR = {'oracle_sha256': hashlib.sha256(Path(__file__).with_name('interval_evaluator.py').read_bytes()).hexdigest(),
                         'worker_sha256': hashlib.sha256(Path(__file__).with_name('interval_worker.py').read_bytes()).hexdigest(), 'id': 'interval-union-correctness-v1', 'contract': EVALUATOR['contract'], 'correctness': 'independent endpoint-sweep oracle on frozen bounded cases; no timing claim'}


class ExperimentHeld(RuntimeError):
    pass


@dataclass(frozen=True)
class ExperimentRequest:
    task_id: str
    candidate_id: int
    source_sha256: str
    evaluator_sha256: str
    command_id: str = 'interval-union-check-and-benchmark-v1'
    profile: str = 'mixed-v1'

    def payload(self):
        return asdict(self)


class ExperimentRunner:
    actual_execution = True

    def run(self, request, source, stage, deadline):
        # This interface is deliberately not an argv/shell interface. The host
        # resource runner is not confinement and cannot authorize candidate code.
        raise ExperimentHeld('experiment execution held: reviewed workload authority, immutable evaluator/runtime pins, '
                             'verified confinement and fresh exclusive admission required; no executor installed')


def validate_receipt(receipt, request, actual):
    from .harness import digest
    keys = {'schema', 'request', 'actual_execution', 'confinement_verified', 'admission_verified',
            'cleanup_verified', 'exitcode', 'stdout', 'stderr', 'artifacts', 'measurements'}
    correctness = request.command_id == 'interval-union-correctness-v1' and request.profile == 'correctness-small-v1'
    if correctness: keys.update(('correctness_details','runtime_sha256','primitive_sha256','seccomp_sha256'))
    if not isinstance(receipt, dict) or set(receipt) != keys or receipt['schema'] != 'PMI_EXPERIMENT_RECEIPT_V1':
        raise ValueError('invalid experiment receipt schema')
    if receipt['request'] != request.payload() or receipt['actual_execution'] is not actual:
        raise ValueError('experiment receipt identity/mode changed')
    for flag in ('confinement_verified', 'admission_verified', 'cleanup_verified'):
        if receipt[flag] is not True:
            raise ValueError('experiment ' + flag + ' unproven')
    if type(receipt['exitcode']) is not int:
        raise ValueError('experiment exitcode missing')
    for stream in ('stdout', 'stderr'):
        value = receipt[stream]
        if not isinstance(value, str) or len(value.encode()) > 65536:
            raise ValueError('experiment output bound exceeded')
    artifacts = receipt['artifacts']
    if not isinstance(artifacts, list) or len(artifacts) > 16:
        raise ValueError('artifact count exceeded')
    total = 0; names = set()
    for item in artifacts:
        if not isinstance(item, dict) or set(item) != {'path', 'bytes', 'sha256'}:
            raise ValueError('invalid artifact receipt')
        name = item['path']
        if not isinstance(name, str) or not name or '\\' in name or '\0' in name or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or name in names:
            raise ValueError('artifact escapes scope or repeats')
        names.add(name)
        if type(item['bytes']) is not int or item['bytes'] < 0:
            raise ValueError('invalid artifact size')
        total += item['bytes']
        if not isinstance(item['sha256'], str) or len(item['sha256']) != 64 or any(c not in '0123456789abcdef' for c in item['sha256']):
            raise ValueError('invalid artifact digest')
    if total > 2 * 1048576:
        raise ValueError('artifact byte bound exceeded')
    values = receipt['measurements']
    if not isinstance(values, dict) or set(values) != {'tests_passed', 'tests_total', 'baseline_ns', 'candidate_ns', 'baseline_sha256'}:
        raise ValueError('invalid measured evidence')
    if any(type(values[k]) is not int or values[k] < 0 for k in ('tests_passed', 'tests_total')) or not 0 < values['tests_total'] or values['tests_passed'] > values['tests_total']:
        raise ValueError('invalid correctness counts')
    for key in ('baseline_ns', 'candidate_ns'):
        if correctness:
            if values[key] is not None: raise ValueError('correctness-only receipt cannot claim timing')
            continue
        if type(values[key]) not in (int, float) or not math.isfinite(values[key]) or values[key] <= 0:
            raise ValueError('invalid measured timing')
    if correctness:
        if values['baseline_sha256'] is not None: raise ValueError('correctness receipt cannot claim benchmark baseline')
        for key in ('runtime_sha256','primitive_sha256','seccomp_sha256'):
            if not isinstance(receipt[key],str) or len(receipt[key])!=64 or any(c not in '0123456789abcdef' for c in receipt[key]): raise ValueError('executor identity digest invalid')
    elif not isinstance(values['baseline_sha256'], str) or len(values['baseline_sha256']) != 64 or any(c not in '0123456789abcdef' for c in values['baseline_sha256']):
        raise ValueError('invalid baseline identity')
    correct = receipt['exitcode'] == 0 and values['tests_passed'] == values['tests_total']
    if correctness:
        from .interval_evaluator import cases, case_identity
        details=receipt['correctness_details']
        if not isinstance(details,dict) or set(details)!={'tests_passed','tests_total','failures','cases_sha256','correct','claim_scope'} or details['tests_passed']!=values['tests_passed'] or details['tests_total']!=len(cases()) or details['tests_total']!=values['tests_total'] or details['cases_sha256']!=case_identity(cases()) or details['correct'] is not correct or not isinstance(details['failures'],list) or len(details['failures'])>8:
            raise ValueError('independent correctness receipt changed')
    evaluator = CORRECTNESS_EVALUATOR if correctness else EVALUATOR
    if request.evaluator_sha256 != digest(evaluator): raise ValueError('experiment evaluator identity changed')
    speedup = None if correctness else values['baseline_ns'] / values['candidate_ns']
    return {'evaluator_sha256': digest(evaluator), 'correct': correct, 'speedup': speedup,
            'objective_met': correct if correctness else correct and speedup >= EVALUATOR['required_speedup'],
            'receipt_sha256': digest(receipt)}
