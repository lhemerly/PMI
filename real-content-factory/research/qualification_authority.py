"""Prospective exact CPU exception, requiring review before installation/use.

Source seals are integrity data, never approval. The fixed entry is submitted to
normal platform review separately from installation. No environment variable,
manifest value, model output, or general CLI grants this exception. The general
GPU/model/shell gate is unchanged. This is not a platform authorization service.
"""
import hashlib
import json
from pathlib import Path

CHECKOUT = Path('/home/luizh/Documents/Codex/2026-10-08/task-8/real-content-factory')
ROOT = Path('/tmp/pmi-first-correctness-v2')
INTERVAL_RUNTIME = Path('/tmp/pmi-correctness-runtime-v2')
PROBE_RUNTIME = Path('/tmp/pmi-qualification-runtime-v2')
LEASE = CHECKOUT.parent / 'content-factory-shared.lease'
SEALS_SHA256 = '65e82837bb92570e475965318cd365b90717c6b8d3e0ad7c23599465091e4d37'
INVENTORY_SHA256 = 'ce965c0c1710e7ca0c826019e09566c578a1351db688a4c5e38cd5e68d55da9d'
PRIMITIVE_SHA256 = '8dea6d013ac3a476f1964ffc6d407a3b6223f43cd4a0c43207146a5cefe59882'
CORE_HELPER = Path('/home/luizh/Documents/Codex/2026-10-06/task-3/candidate-qualification/core_limit_exec.py')
CORE_HELPER_SHA256 = '7c5f71ac2e209d06568b061efde9df869b5cc2e22d8249cc911fbcf0ab66b1c9'
PRIOR_REPORT = Path('/tmp/pmi-first-correctness-v1/report.json')
PRIOR_REPORT_SHA256 = 'e6c09e71b3dd465d35fcb8d5b2a80843853e73c4ab9057f69262580b18c43157'
_active = None


def hashed(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources():
    if hashed(PRIOR_REPORT) != PRIOR_REPORT_SHA256:
        raise ValueError('prior failed qualification evidence changed')
    if hashed('/usr/bin/bwrap') != PRIMITIVE_SHA256 or hashed(CORE_HELPER) != CORE_HELPER_SHA256:
        raise ValueError('reviewed Bubblewrap/core helper changed')
    path = CHECKOUT / 'research/FIRST-CPU-SOURCE-SEALS.json'
    if hashed(path) != SEALS_SHA256:
        raise ValueError('exact reviewed CPU source seals missing/changed')
    seals = json.loads(path.read_text())
    for relative, identity in seals.items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('unsafe source seal path')
        if hashed(CHECKOUT / relative) != identity:
            raise ValueError('reviewed CPU source changed: ' + relative)
    path = CHECKOUT / 'research/FIRST-CPU-RUNTIME-INVENTORY.json'
    if hashed(path) != INVENTORY_SHA256:
        raise ValueError('exact reviewed runtime inventory missing/changed')
    return json.loads(path.read_text())


class _FirstCPUQualification:
    def __init__(self):
        self._phase = 'package'
        self.used = set()
        self.inventory = verify_sources()

    @property
    def phase(self):
        return self._phase

    def packaged(self, manifests):
        from .executor import verify_runtime
        if self.phase != 'package' or self.used != {('package','interval'),('package','qualification')}:
            raise ValueError('both exact runtimes must be packaged first')
        for role, destination in (('interval',INTERVAL_RUNTIME),('qualification',PROBE_RUNTIME)):
            record = manifests[role]
            expected = {name:row['sha256'] for name,row in self.inventory[role].items()}
            if record['root'] != str(destination) or record['files'] != expected or record['primitive_sha256'] != PRIMITIVE_SHA256:
                raise ValueError('packaged runtime differs from reviewed inventory')
            if verify_runtime(destination, expected, worker=role) != record['runtime_sha256']:
                raise ValueError('packaged runtime hash mismatch')
        self._phase = 'bootstrap'

    def qualified(self, capture, probe, cleanup):
        from .executor import verify_capture
        from .qualification_worker import EXPECTED_CHECKS
        verify_capture(capture)
        if self.phase != 'bootstrap' or not {('admission','bootstrap'),('transport','bootstrap')} <= self.used:
            raise ValueError('bootstrap must execute before candidate admission')
        if (capture['exitcode'] != 0 or set(probe) != {'schema','probe','candidate_executed','checks','observed'}
                or probe['schema'] != 'PMI_CPU_BOOTSTRAP_V1' or probe['probe'] != 'bootstrap'
                or probe['candidate_executed'] is not False or set(probe['checks']) != set(EXPECTED_CHECKS)
                or any(value is not True for value in probe['checks'].values())
                or cleanup.get('cleanup_verified') is not True):
            raise ValueError('bootstrap evidence/cleanup incomplete')
        self._phase = 'candidate'

    def check(self, operation, **details):
        verify_sources()
        if operation == 'package':
            role = details['worker']
            expected_root = {'interval': INTERVAL_RUNTIME, 'qualification': PROBE_RUNTIME}.get(role)
            if self.phase != 'package' or expected_root is None or Path(details['destination']) != expected_root:
                raise ValueError('CPU packaging scope mismatch')
            if details['inventory'] != self.inventory[role]:
                raise ValueError('CPU packaging inventory mismatch')
            key = ('package', role)
        elif operation in ('admission', 'transport'):
            from .executor import SandboxPlan, build_plan
            from .interval_evaluator import cases
            plan = details['plan']
            if self.phase not in ('bootstrap', 'candidate') or type(plan) is not SandboxPlan:
                raise ValueError('CPU phase/plan mismatch')
            base = ROOT / ('bootstrap' if self.phase == 'bootstrap' else 'correctness/stage')
            runtime = PROBE_RUNTIME if self.phase == 'bootstrap' else INTERVAL_RUNTIME
            values = [{'probe': 'bootstrap'}] if self.phase == 'bootstrap' else cases()
            args = plan.argv
            expected = build_plan(runtime, base/'candidate', base/'artifacts',
                                  int(args[args.index('--seccomp')+1]), int(args[args.index('--json-status-fd')+1]),
                                  values, (plan.block_read_fd, plan.block_write_fd))
            if plan != expected:
                raise ValueError('exact CPU plan/limits changed')
            key = (operation, self.phase)
        elif operation == 'once':
            if self.phase != 'candidate' or details != {
                'source_path': CHECKOUT/'research/interval_candidate.py',
                'manifest_path': ROOT/'interval-manifest.json',
                'out': ROOT/'correctness', 'lease': ROOT/'candidate.lease'}:
                raise ValueError('one-shot CPU scope mismatch')
            key = (operation, self.phase)
        elif operation == 'candidate':
            from .experiments import ExperimentRequest, CORRECTNESS_EVALUATOR
            from .harness import digest
            from .interval_evaluator import validate_source
            source = (CHECKOUT/'research/interval_candidate.py').read_text()
            expected = ExperimentRequest('correctness-demo-001', 1, validate_source(source),
                                         digest(CORRECTNESS_EVALUATOR), 'interval-union-correctness-v1', 'correctness-small-v1')
            if self.phase != 'candidate' or details['request'] != expected or details['source'] != source or Path(details['stage']) != ROOT/'correctness/stage':
                raise ValueError('one authored CPU candidate only')
            key = (operation, self.phase)
        else:
            raise ValueError('unknown scoped CPU operation')
        if key in self.used:
            raise ValueError('scoped CPU operation already reserved; no retries')
        self.used.add(key)


def gate_cpu(authority, operation, **details):
    if authority is None:
        from launch_policy import require_real_execution
        return require_real_execution()
    from .demo_authority import is_demo_authority, gate_cpu as gate_demo_cpu
    if is_demo_authority(authority):
        return gate_demo_cpu(authority, operation, **details)
    if type(authority) is not _FirstCPUQualification or authority is not _active:
        raise ValueError('no trusted scoped CPU authority')
    authority.check(operation, **details)


def activate_for_fixed_entry():
    """Only the fixed, separately reviewed entry uses this process-local scope."""
    global _active
    if _active is not None or ROOT.exists():
        raise ValueError('qualification already reserved; no retry')
    authority = _FirstCPUQualification()
    ROOT.mkdir(mode=0o700)
    _active = authority
    return authority
