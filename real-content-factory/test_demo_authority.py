"""Model-free authority contracts; no native, GPU, sandbox or candidate launches.

Temporary request files and this test process's own sealed memfd are the only
live capability fixtures. Integrity and parent-controller command lines are
mocked explicitly; passing these tests is never authorization to execute.
"""
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import model_profiles
from research import demo_authority as authority
from research.experiments import CORRECTNESS_EVALUATOR, ExperimentRequest
from research.executor import build_plan
from research.interval_evaluator import cases, case_identity, validate_source


SOURCE = 'def merge_intervals(intervals):\n    return []\n'
NOW = 1000.0
MODEL_STAGES = {
    'seed-1': 'generation', 'seed-2': 'generation', 'evolve-1': 'evolution',
    **{'reflection-%d' % i: 'reflection' for i in (1, 2, 3)},
    **{'match-%s' % pair: 'ranking' for pair in ('1-2', '1-3', '2-3')},
    **{'match-%s-debate-%d' % (pair, debate): 'debate'
       for pair in ('1-2', '1-3', '2-3') for debate in (1, 2)},
    'meta-initial': 'meta-review', 'meta-final': 'meta-review',
    'overview': 'overview',
}


class AuthorityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='pmi-authority-unit-')
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / 'demo'
        self.checkout = self.base / 'checkout'
        fixture = self.checkout / 'research/proposed-demo/calibration_seed1_fixture.txt'
        fixture.parent.mkdir(parents=True)
        fixture.write_text('fixed unit-test calibration prompt')
        self.calibration_prompt = fixture.read_text()
        self.integrity = Mock(return_value={'fixture': True})
        # Activation tests below isolate the already-tested admission-only
        # continuation; the native-attempt proof has its own real-file fixtures.
        failed_native = patch.object(authority, 'verify_failed_native_attempt', return_value=None)
        failed_native.start()
        self.addCleanup(failed_native.stop)
        self.prior = self.base / 'prior'
        (self.prior / 'calibration').mkdir(parents=True)
        self.prior_child = {'status': 'blocked', 'model_stages_reserved': 1, 'candidate_stages_reserved': 0,
            'actual_model_completions': 0, 'generated_candidate_receipts': 0,
            'model_completion_receipts': [], 'candidate_receipts': [],
            'workflow': None, 'calibration': None,
            'uncertain_operation': {'kind': 'model', 'model_id': 'bonsai-stock',
                'role': 'calibration', 'prompt_sha256': hashlib.sha256(self.calibration_prompt.encode()).hexdigest(),
                'stage': str(self.prior / 'calibration'), 'deadline': NOW + 100},
            'error': 'ValueError: substantial host paging during admission'}
        self.prior_report = {'cleanup_verified': True, 'remaining_owned': [],
            'coordination_release': 'verified_empty', 'coordination_retained': True,
            'errors': [], 'completed': False, 'returncode': 2,
            'child_final_status': 'blocked', 'stop_reason': 'child_exit',
            'final_deadline': NOW + 2400, 'operation_deadline': NOW + 2220}
        self.original_report = self.base / 'original-supervisor-report.json'
        self.original_report.write_text(json.dumps({'final_deadline': NOW + 2400}))
        (self.prior / 'child-result.json').write_text(json.dumps(self.prior_child))
        (self.prior / 'supervisor-report.json').write_text(json.dumps(self.prior_report))
        prior_request = model_profiles.model_request('bonsai-stock', self.calibration_prompt,
                                                     'bonsai-research-v1', 'calibration')
        (self.prior / 'calibration/request.json').write_text(json.dumps(prior_request))
        # Pure counter fixtures cause the real evaluator to reject paging;
        # neither admission sampling nor model/native processes are invoked.
        self.prior_samples = [{'time': NOW - 60 + index * .5, 'available': 20 * 1024 ** 3,
                               'swapout': index * 1048576, 'PSI_full_us': 0,
                               'PSI_some_us': 0, 'oom_kill': 0} for index in range(61)]
        self.prior_admission = {'reasons': ['passive memory pressure'], 'samples': self.prior_samples}
        (self.prior / 'calibration/admission.json').write_text(json.dumps(self.prior_admission))
        self.prior_telemetry = [{'owned': [], 'unverified_members': [], 'errors': []} for _ in range(61)]
        (self.prior / 'calibration/loading-telemetry.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in self.prior_telemetry))
        for name in ('controller.stdout.log', 'controller.stderr.log'):
            (self.prior / name).write_text('')
        prior_pins = {str(path.relative_to(self.prior)): authority.hashed(path)
                      for path in self.prior.rglob('*') if path.is_file()}
        for name, value in (('ROOT', self.root), ('CHECKOUT', self.checkout),
                            ('RUNTIME', self.base / 'runtime'),
                            ('PRIOR_ROOT', self.prior),
                            ('PRIOR_ARTIFACTS', prior_pins),
                            ('ORIGINAL_REPORT', self.original_report),
                            ('ORIGINAL_REPORT_SHA256', authority.hashed(self.original_report)),
                            ('_active', None), ('verify_integrity', self.integrity)):
            replacement = patch.object(authority, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        monotonic = patch.object(authority.time, 'monotonic', return_value=NOW)
        monotonic.start()
        self.addCleanup(monotonic.stop)
        self.coordination = SimpleNamespace(validate=Mock())
        # Exercise reservation identity separately from laboratory JSON/storage
        # validation, which has its own full fixture-loop test suite.
        self.reservation_patch = patch.object(authority, '_workflow_reservation',
            side_effect=lambda *args: {'model_calls': authority._active.model_count})
        self.reservation = self.reservation_patch.start()
        self.addCleanup(self.reservation_patch.stop)
        self.saved = None
        database = patch('research.laboratory.LaboratoryStore.load',
                         side_effect=lambda: self.saved)
        database.start()
        self.addCleanup(database.stop)

    def child(self, *, calibrated=False):
        child = authority._ResearchDemo('child', NOW + 2700,
                                        SimpleNamespace(validate=Mock()))
        child.calibration_done = calibrated
        child.model_count = 1 if calibrated else 0
        authority._active = child
        return child

    def model_args(self, stage='seed-1', role=None):
        calibration = stage == 'calibration'
        return dict(model_id='bonsai-stock',
                    role=role or ('calibration' if calibration else MODEL_STAGES[stage]),
                    prompt=self.calibration_prompt if calibration else 'bounded fixture prompt',
                    stage=self.root / ('calibration' if calibration else 'workflow/task-' + stage),
                    deadline=NOW + 100, coordination=self.coordination)

    def reserve_model(self, child, **kwargs):
        args = self.model_args(**kwargs)
        authority.gate_model(child, 'agent', **args)
        authority.gate_model(child, 'backend', **args)
        request = model_profiles.model_request(args['model_id'], args['prompt'],
                                                'bonsai-research-v1', args['role'])
        path = args['stage'] / 'request.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(request))
        return args, request, path

    def finish_model(self, child, args):
        authority.gate_model(child, 'complete', **args,
                             capture={'owned_cleanup_empty': True,
                                      'postflight_passed': True, 'returncode': 0},
                             receipt={'actual_inference': True})

    def request(self, identity=1):
        return ExperimentRequest('experiment-%d' % identity, identity,
                                 validate_source(SOURCE), authority.digest(CORRECTNESS_EVALUATOR),
                                 'interval-union-correctness-v1', 'correctness-small-v1')

    def reserve_cpu(self, child, identity=1):
        request = self.request(identity)
        stage = self.root / ('workflow/task-experiment-%d' % identity)
        self.saved = {'mode': 'real', 'inflight': request.task_id,
                      'tasks': [{'id': request.task_id, 'kind': 'experiment', 'status': 'reserved'}],
                      'candidates': [{'source': SOURCE, 'sha256': validate_source(SOURCE)}] * identity,
                      'experiment_calls': child.candidate_count + 1}
        authority.gate_cpu(child, 'candidate', request=request, source=SOURCE, stage=stage)
        plan = build_plan(authority.RUNTIME, stage / 'candidate', stage / 'artifacts',
                          10, 11, cases(), (12, 13))
        return request, plan

    def cpu_receipt(self, request):
        # Construct receipt data only. SOURCE is intentionally not executed and
        # correctness is false; clean bounded failure is still a completed run.
        return {'schema': 'PMI_EXPERIMENT_RECEIPT_V1', 'request': request.payload(),
                'actual_execution': True, 'confinement_verified': True,
                'admission_verified': True, 'cleanup_verified': True,
                'exitcode': 0, 'stdout': '', 'stderr': '', 'artifacts': [],
                'runtime_sha256': '1' * 64, 'primitive_sha256': '2' * 64,
                'seccomp_sha256': '3' * 64,
                'measurements': {'tests_passed': 0, 'tests_total': 884,
                                 'baseline_ns': None, 'candidate_ns': None,
                                 'baseline_sha256': None},
                'correctness_details': {'tests_passed': 0, 'tests_total': 884,
                                       'failures': [], 'cases_sha256': case_identity(cases()),
                                       'correct': False,
                                       'claim_scope': 'unit receipt; no execution'}}

    def test_fixed_entry_is_single_use_and_reserves_private_root(self):
        active = authority.activate_for_fixed_entry()
        self.assertIs(active, authority._active)
        self.assertEqual(active.kind, 'supervisor')
        self.assertEqual(active.deadline, NOW + 2400)
        self.assertEqual(stat.S_IMODE(self.root.stat().st_mode), 0o700)
        self.integrity.assert_called_once_with()
        with self.assertRaises(ValueError):
            authority.activate_for_fixed_entry()

    def test_retry_cannot_reset_original_absolute_deadline(self):
        for deadline in (NOW + 2700, NOW + 2401, NOW + 180, NOW - 1):
            self.prior_report.update(final_deadline=deadline, operation_deadline=deadline - 180)
            self.repin_prior('supervisor-report.json', self.prior_report)
            with self.assertRaises(ValueError):
                authority.activate_for_fixed_entry()
        self.assertFalse(self.root.exists())

    def repin_prior(self, name, value, *, jsonl=False):
        path = self.prior / name
        path.write_text(''.join(json.dumps(row) + '\n' for row in value) if jsonl else json.dumps(value))
        authority.PRIOR_ARTIFACTS[name] = authority.hashed(path)

    def test_retry_rejects_previous_actual_inference_receipt_or_extra_reservation(self):
        for key, value in (('actual_model_completions', 1), ('generated_candidate_receipts', 1),
                           ('model_stages_reserved', 2), ('candidate_stages_reserved', 1),
                           ('model_completion_receipts', ['calibration/result.json']),
                           ('candidate_receipts', ['workflow/task-experiment-1/experiment-receipt.json']),
                           ('calibration', {'actual_inference': True}), ('workflow', {})):
            with self.subTest(key=key):
                self.repin_prior('child-result.json', {**self.prior_child, key: value})
                with self.assertRaisesRegex(ValueError, 'admission-only stop/cleanup'):
                    authority.activate_for_fixed_entry()
        self.assertFalse(self.root.exists())

    def test_retry_rejects_any_native_dispatch_artifact_or_workflow_directory(self):
        for name in ('guarded-command.json', 'native.stdout.log', 'core-limit.json', 'result.json'):
            path = self.prior / 'calibration' / name
            path.write_text('')
            try:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'inventory changed'):
                    authority.activate_for_fixed_entry()
            finally:
                path.unlink()
        (self.prior / 'workflow').mkdir()
        with self.assertRaisesRegex(ValueError, 'inventory changed'):
            authority.activate_for_fixed_entry()
        self.assertFalse(self.root.exists())

    def test_retry_rejects_altered_prior_hash_and_original_deadline_receipt(self):
        path = self.prior / 'controller.stdout.log'
        path.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            authority.activate_for_fixed_entry()
        path.write_text('')
        self.original_report.write_text(json.dumps({'final_deadline': NOW + 2700}))
        with self.assertRaisesRegex(ValueError, 'original deadline receipt changed'):
            authority.activate_for_fixed_entry()
        self.assertFalse(self.root.exists())

    def test_retry_requires_exact_reserved_calibration_request(self):
        operation = {**self.prior_child['uncertain_operation'], 'role': 'generation'}
        self.repin_prior('child-result.json', {**self.prior_child, 'uncertain_operation': operation})
        with self.assertRaisesRegex(ValueError, 'admission-only stop/cleanup'):
            authority.activate_for_fixed_entry()
        self.repin_prior('child-result.json', self.prior_child)
        request = model_profiles.model_request('bonsai-stock', 'different prompt',
                                               'bonsai-research-v1', 'calibration')
        self.repin_prior('calibration/request.json', request)
        with self.assertRaisesRegex(ValueError, 'fixed calibration request changed'):
            authority.activate_for_fixed_entry()
        self.assertFalse(self.root.exists())

    def test_retry_requires_sixty_one_empty_owned_telemetry_rows(self):
        for rows in (self.prior_telemetry[:-1],
                     [{**self.prior_telemetry[0], 'owned': [{'pid': 1}]}] + self.prior_telemetry[1:],
                     [{**self.prior_telemetry[0], 'errors': ['unreadable']}] + self.prior_telemetry[1:],
                     [{**self.prior_telemetry[0], 'unverified_members': [1]}] + self.prior_telemetry[1:]):
            self.repin_prior('calibration/loading-telemetry.jsonl', rows, jsonl=True)
            with self.assertRaisesRegex(ValueError, 'no-native telemetry absent'):
                authority.activate_for_fixed_entry()
        self.assertFalse(self.root.exists())

    def test_retry_requires_reproduced_paging_stop_and_verified_cleanup(self):
        quiet = [{**row, 'swapout': 0} for row in self.prior_samples]
        self.repin_prior('calibration/admission.json', {**self.prior_admission, 'samples': quiet})
        with self.assertRaisesRegex(ValueError, 'paging stop not reproduced'):
            authority.activate_for_fixed_entry()
        self.repin_prior('calibration/admission.json', self.prior_admission)
        for key, value in (('cleanup_verified', False), ('coordination_retained', False),
                           ('coordination_release', 'uncertain_at_final_deadline'),
                           ('errors', ['cleanup uncertain']), ('remaining_owned', [1])):
            self.repin_prior('supervisor-report.json', {**self.prior_report, key: value})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'admission-only stop/cleanup'):
                authority.activate_for_fixed_entry()
        self.assertFalse(self.root.exists())

    def test_existing_root_symlink_and_failed_integrity_do_not_activate(self):
        for kind in ('directory', 'symlink', 'integrity'):
            with self.subTest(kind=kind):
                if kind == 'directory':
                    self.root.mkdir()
                elif kind == 'symlink':
                    self.root.symlink_to(self.base / 'missing')
                else:
                    self.integrity.side_effect = ValueError('reviewed bytes changed')
                with self.assertRaises(ValueError):
                    authority.activate_for_fixed_entry()
                self.assertIsNone(authority._active)
                if self.root.is_symlink():
                    self.root.unlink()
                elif self.root.exists():
                    self.root.rmdir()

    def test_supervisor_invocation_requires_exact_identity_role_and_no_details(self):
        active = authority.activate_for_fixed_entry()
        with self.assertRaises(ValueError):
            authority.gate_supervisor(active, command='arbitrary')
        authority.gate_supervisor(active)
        with self.assertRaises(ValueError):
            authority.gate_supervisor(active)
        authority._active = self.child()
        with self.assertRaises(ValueError):
            authority.gate_supervisor(authority._active)

    def test_authority_rejects_other_object_owner_and_expired_deadline(self):
        active = self.child()
        for value in (None, object(), authority._ResearchDemo('child', NOW + 1)):
            with self.subTest(value=type(value).__name__), self.assertRaises(ValueError):
                authority.reviewed_backend_seal(value)
        active.owner = (os.getpid(), -1)
        with self.assertRaises(ValueError):
            authority.reviewed_backend_seal(active)
        active.owner = authority.identity(os.getpid())[0]
        for deadline in (NOW, NOW - 1, float('inf'), float('nan')):
            active.deadline = deadline
            with self.subTest(deadline=deadline), self.assertRaises(TimeoutError):
                authority.reviewed_backend_seal(active)

    def test_context_invalidated_before_any_new_permission(self):
        active = self.child()
        active.context.validate.side_effect = ValueError('supervisor lost')
        with self.assertRaises(ValueError):
            authority.reviewed_backend_seal(active)
        self.integrity.assert_not_called()

    def test_calibration_requires_exact_prompt_and_precedes_workflow(self):
        active = self.child()
        with self.assertRaises(ValueError):
            authority.gate_model(active, 'agent', **self.model_args())
        args = self.model_args('calibration')
        with self.assertRaises(ValueError):
            authority.gate_model(active, 'agent', **{**args, 'prompt': args['prompt'] + ' changed'})
        authority.gate_model(active, 'agent', **args)
        self.assertEqual(active.model_count, 1)
        with self.assertRaises(ValueError):
            active.calibrated()
        with self.assertRaises(ValueError):
            authority.gate_model(active, 'agent', **args)

    def test_calibration_is_completed_only_after_native_cleanup_and_valid_code(self):
        active = self.child()
        args, request, path = self.reserve_model(active, stage='calibration')
        with authority.native_capability(active, path, request, args['deadline']):
            pass
        self.finish_model(active, args)
        (args['stage'] / 'answer.txt').write_text('not Python')
        with self.assertRaises((SyntaxError, ValueError)):
            active.calibrated()
        (args['stage'] / 'answer.txt').write_text(SOURCE)
        active.calibrated()
        self.assertTrue(active.calibration_done)
        with self.assertRaises(ValueError):
            active.calibrated()
        with self.assertRaises(ValueError):
            authority.gate_model(active, 'agent', **self.model_args('calibration'))

    def test_model_requires_coordination_fixed_profile_and_bounded_deadline(self):
        active = self.child(calibrated=True)
        args = self.model_args()
        for key, value in (('coordination', None), ('model_id', 'qwen3-stock'),
                           ('role', 'semantic-proximity'), ('deadline', True),
                           ('deadline', NOW), ('deadline', NOW + 121),
                           ('deadline', float('inf'))):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                authority.gate_model(active, 'agent', **{**args, key: value})
        self.assertEqual(active.model_count, 1)

    def test_exact_role_task_mapping_has_only_eighteen_workflow_stages(self):
        self.assertEqual(len(MODEL_STAGES), 18)
        for task, role in MODEL_STAGES.items():
            stage = self.root / ('workflow/task-' + task)
            with self.subTest(task=task):
                self.assertEqual(authority._stage(role, stage), stage)
                for other in model_profiles.ROLE_OUTPUT_TOKENS:
                    if other != role:
                        with self.assertRaises(ValueError):
                            authority._stage(other, stage)

    def test_stage_rejects_outside_symlink_and_nonexistent_match(self):
        for name in ('match-1-1', 'match-2-1', 'match-3-3', 'match-1-2-round-2',
                     'seed-3', 'experiment-1', '../task-seed-1', 'unknown'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                authority._stage('ranking', self.root / ('workflow/task-' + name))
        stage = self.root / 'workflow/task-seed-1'
        stage.parent.mkdir(parents=True)
        stage.symlink_to(self.base)
        with self.assertRaises(ValueError):
            authority._stage('generation', stage)
        with self.assertRaises(ValueError):
            authority._stage('generation', self.base / 'task-seed-1')

    def test_model_order_request_identity_and_natural_completion_are_required(self):
        active = self.child(calibrated=True)
        args = self.model_args()
        with self.assertRaises(ValueError):
            authority.gate_model(active, 'backend', **args)
        authority.gate_model(active, 'agent', **args)
        for key, value in (('prompt', 'different'), ('deadline', NOW + 99),
                           ('stage', self.root / 'workflow/task-seed-2')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                authority.gate_model(active, 'backend', **{**args, key: value})
        authority.gate_model(active, 'backend', **args)
        with self.assertRaises(ValueError):
            authority.gate_model(active, 'backend', **args)
        with self.assertRaises(ValueError):
            self.finish_model(active, args)
        active.current['used'].add('native')  # order-only fixture; never launch
        for key, value in (('owned_cleanup_empty', False), ('postflight_passed', False),
                           ('returncode', 1)):
            capture = {'owned_cleanup_empty': True, 'postflight_passed': True, 'returncode': 0}
            capture[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                authority.gate_model(active, 'complete', **args, capture=capture,
                                     receipt={'actual_inference': True})
        self.finish_model(active, args)
        self.assertIsNone(active.current)
        with self.assertRaises(ValueError):
            authority.gate_model(active, 'agent', **args)

    @patch.object(authority, 'PRIOR_NATIVE_ATTEMPTS', 0)
    def test_model_count_cannot_exceed_calibration_plus_eighteen(self):
        active = self.child(calibrated=True)
        for task in MODEL_STAGES:
            args, request, path = self.reserve_model(active, stage=task)
            with authority.native_capability(active, path, request, args['deadline']):
                pass
            self.finish_model(active, args)
        self.assertEqual(active.model_count, 19)
        with self.assertRaises(ValueError):
            authority.gate_model(active, 'agent', **self.model_args())

    @patch.object(authority, 'PRIOR_NATIVE_ATTEMPTS', 1)
    def test_failed_native_attempt_is_charged_to_nineteen_total_cap(self):
        active = self.child(calibrated=True)
        stages = list(MODEL_STAGES)
        for task in stages[:17]:
            args, request, path = self.reserve_model(active, stage=task)
            with authority.native_capability(active, path, request, args['deadline']):
                pass
            self.finish_model(active, args)
        self.assertEqual(active.model_count, 18)
        self.assertEqual(active.model_count + authority.PRIOR_NATIVE_ATTEMPTS, 19)
        with self.assertRaisesRegex(ValueError, '19 total native attempts'):
            authority.gate_model(active, 'agent', **self.model_args(stages[17]))
        self.assertEqual(active.model_count, 18)

    def test_workflow_requires_matching_persisted_reserved_prompt_and_mode(self):
        self.reservation_patch.stop()
        args = self.model_args()
        valid = {'mode': 'real', 'inflight': 'seed-1', 'model_calls': 1,
                 'tasks': [{'id': 'seed-1', 'kind': 'generation', 'status': 'reserved',
                            'model_io': {'prompt_sha256': hashlib.sha256(args['prompt'].encode()).hexdigest()}}]}
        self.saved = valid
        self.assertIs(authority._workflow_reservation(args['role'], args['prompt'], args['stage']), valid)
        mutations = ({'mode': 'fixture'}, {'inflight': None}, {'tasks': []},
                     {'tasks': [{**valid['tasks'][0], 'kind': 'reflection'}]},
                     {'tasks': [{**valid['tasks'][0], 'status': 'done'}]},
                     {'tasks': [{**valid['tasks'][0], 'model_io': {'prompt_sha256': '0' * 64}}]})
        for mutation in mutations:
            self.saved = {**valid, **mutation}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                authority._workflow_reservation(args['role'], args['prompt'], args['stage'])

    def test_model_scope_counter_must_match_persisted_workflow_reservation(self):
        active = self.child(calibrated=True)
        self.reservation.side_effect = None
        self.reservation.return_value = {'model_calls': 2}
        with self.assertRaises(ValueError):
            authority.gate_model(active, 'agent', **self.model_args())
        self.assertEqual(active.model_count, 1)

    def test_native_capability_is_sealed_bound_and_closed_after_context(self):
        active = self.child(calibrated=True)
        args, request, path = self.reserve_model(active)
        with authority.native_capability(active, path, request, args['deadline']) as capability:
            fd = os.open(capability, os.O_RDWR)
            try:
                seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
                self.assertEqual(fcntl.fcntl(fd, fcntl.F_GET_SEALS) & seals, seals)
                payload = json.loads(os.pread(fd, 16384, 0))
                self.assertEqual(payload['request_sha256'], authority.digest(request))
                self.assertEqual(payload['controller'], list(active.owner))
                with self.assertRaises(PermissionError):
                    os.write(fd, b'changed')
            finally:
                os.close(fd)
        self.assertFalse(Path(capability).exists())
        with self.assertRaises(ValueError):
            with authority.native_capability(active, path, request, args['deadline']):
                self.fail('native capability reused')

    def test_native_request_must_match_reserved_model_role_profile_and_prompt(self):
        for key, value in (('role', 'ranking'), ('model_id', 'qwen3-stock'),
                           ('profile_id', 'other'), ('profile_sha256', '0' * 64),
                           ('prompt', 'altered prompt')):
            with self.subTest(key=key):
                active = self.child(calibrated=True)
                args, request, path = self.reserve_model(active)
                request[key] = value
                path.write_text(json.dumps(request))
                with self.assertRaises(ValueError):
                    with authority.native_capability(active, path, request, args['deadline']):
                        self.fail('unreserved native request accepted')

    @contextmanager
    def fixed_controller_cmdline(self):
        read_bytes = Path.read_bytes
        own_cmdline = Path('/proc/%d/cmdline' % os.getpid())
        def read(path):
            if path == own_cmdline:
                return b'python3\0-B\0-m\0research.model_demo_entry\0--supervised-child\0'
            return read_bytes(path)
        with patch.object(Path, 'read_bytes', autospec=True, side_effect=read):
            yield

    def test_native_gate_accepts_only_live_sealed_exact_controller_fixture(self):
        active = self.child(calibrated=True)
        args, request, path = self.reserve_model(active)
        with authority.native_capability(active, path, request, args['deadline']) as capability:
            with self.assertRaisesRegex(ValueError, 'fixed controller'):
                authority.gate_native(path, request, capability_path=capability)
            with self.fixed_controller_cmdline():
                authority.gate_native(path, request, capability_path=capability)
                with self.assertRaises(ValueError):
                    authority.gate_native(path, {**request, 'prompt': 'different'},
                                          capability_path=capability)
                path.write_text(json.dumps({**request, 'prompt': 'changed on disk'}))
                with self.assertRaises(ValueError):
                    authority.gate_native(path, request, capability_path=capability)

    def test_native_gate_rejects_disk_tokens_unsealed_and_expired_memfd(self):
        active = self.child(calibrated=True)
        args, request, path = self.reserve_model(active)
        with self.assertRaises(ValueError):
            authority.gate_native(path, request, capability_path=path)
        fd = os.memfd_create('pmi-research-stage', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        try:
            with self.assertRaises(ValueError):
                authority.gate_native(path, request, capability_path='/proc/%d/fd/%d' % (os.getpid(), fd))
        finally:
            os.close(fd)
        with authority.native_capability(active, path, request, args['deadline']) as capability:
            with patch.object(authority.time, 'monotonic', return_value=args['deadline']):
                with self.assertRaisesRegex(ValueError, 'expired'):
                    authority.gate_native(path, request, capability_path=capability)

    def test_native_gate_rejects_controller_pid_reuse_or_non_descendant(self):
        active = self.child(calibrated=True)
        args, request, path = self.reserve_model(active)
        actual = authority.identity(os.getpid())
        changed = ((actual[0][0], actual[0][1] + 1), actual[1])
        unrelated = ((actual[0][0] + 1, actual[0][1]), 1)
        with authority.native_capability(active, path, request, args['deadline']) as capability:
            with patch.object(authority, 'identity', side_effect=[actual, changed]):
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    authority.gate_native(path, request, capability_path=capability)
            with patch.object(authority, 'identity', side_effect=[actual, actual, unrelated]):
                with self.assertRaisesRegex(ValueError, 'not owned controller descendant'):
                    authority.gate_native(path, request, capability_path=capability)

    def test_regular_request_artifact_must_be_bounded_and_not_symlink(self):
        regular = self.base / 'bounded.json'
        regular.write_bytes(b'{}')
        self.assertEqual(authority._regular(regular, 2), b'{}')
        with self.assertRaises(ValueError):
            authority._regular(regular, 1)
        link = self.base / 'link.json'
        link.symlink_to(regular)
        with self.assertRaises(ValueError):
            authority._regular(link, 2)
        with self.assertRaises(ValueError):
            authority._regular(self.base, 2)

    def test_cpu_requires_calibration_exact_request_source_and_stage(self):
        active = self.child()
        request = self.request()
        stage = self.root / 'workflow/task-experiment-1'
        with self.assertRaises(ValueError):
            authority.gate_cpu(active, 'candidate', request=request, source=SOURCE, stage=stage)
        active.calibration_done = True
        for mutation in ({'request': replace(request, candidate_id=4)},
                         {'request': replace(request, task_id='authored-qualification')},
                         {'request': replace(request, source_sha256='0' * 64)},
                         {'request': replace(request, command_id='shell')},
                         {'request': replace(request, profile='mixed-v1')},
                         {'stage': self.base / 'elsewhere'}, {'source': 'import os'}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                authority.gate_cpu(active, 'candidate', **{'request': request, 'source': SOURCE,
                                                           'stage': stage, **mutation})
        self.assertEqual(active.candidate_count, 0)

    def test_cpu_order_plan_identity_and_receipt_validation(self):
        active = self.child(calibrated=True)
        request, plan = self.reserve_cpu(active)
        with self.assertRaises(ValueError):
            authority.gate_cpu(active, 'transport', plan=plan)
        for key, value in (('wall_seconds', 60), ('cpu_seconds', 50),
                           ('stdin', b'{}'), ('environment', {'PATH': '/tmp'}),
                           ('argv', (*plan.argv, 'arbitrary'))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                authority.gate_cpu(active, 'admission', plan=replace(plan, **{key: value}))
        authority.gate_cpu(active, 'admission', plan=plan)
        with self.assertRaises(ValueError):
            authority.gate_cpu(active, 'admission', plan=plan)
        with self.assertRaises(ValueError):
            active.cpu_complete(self.cpu_receipt(request))
        authority.gate_cpu(active, 'transport', plan=plan)
        receipt = self.cpu_receipt(request)
        with self.assertRaises(ValueError):
            active.cpu_complete({**receipt, 'cleanup_verified': False})
        active.cpu_complete(receipt)
        self.assertIsNone(active.current)
        with self.assertRaises(ValueError):
            active.cpu_complete(receipt)
        with self.assertRaises(ValueError):
            self.reserve_cpu(active)

    def test_three_distinct_cpu_candidates_only_and_no_other_operations(self):
        active = self.child(calibrated=True)
        for identity in (1, 2, 3):
            request, plan = self.reserve_cpu(active, identity)
            authority.gate_cpu(active, 'admission', plan=plan)
            authority.gate_cpu(active, 'transport', plan=plan)
            active.cpu_complete(self.cpu_receipt(request))
        self.assertEqual(active.candidate_count, 3)
        with self.assertRaises(ValueError):
            self.reserve_cpu(active)
        for operation in ('package', 'once', 'shell', 'unknown'):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                authority.gate_cpu(active, operation)


class FailedNativeAttemptTests(unittest.TestCase):
    """Read/copy historical receipts only; no native code or candidates execute."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='pmi-failed-native-unit-')
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.failed = base / 'failed'
        self.prior = base / 'prior'
        self.prior.mkdir()
        self.pins = dict(authority.FAILED_NATIVE_ARTIFACTS)
        for name in self.pins:
            destination = self.failed / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path('/tmp/pmi-model-demo-v3') / name, destination)
        shutil.copyfile('/tmp/pmi-model-demo-v2/supervisor-report.json',
                        self.prior / 'supervisor-report.json')
        for name, value in (('FAILED_NATIVE_ROOT', self.failed),
                            ('FAILED_NATIVE_ARTIFACTS', self.pins), ('PRIOR_ROOT', self.prior)):
            replacement = patch.object(authority, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)

    def test_exact_historical_fitter_failure_has_zero_output_and_verified_cleanup(self):
        self.assertIsNone(authority.verify_failed_native_attempt())
        self.assertEqual((self.failed / 'calibration/native.stdout.log').read_bytes(), b'')
        stderr = (self.failed / 'calibration/native.stderr.log').read_text()
        self.assertIn('fitting params to device memory ...', stderr)
        self.assertNotIn('successfully fit params to free device memory', stderr)

    def test_historical_hash_mutation_refuses_continuation(self):
        with (self.failed / 'calibration/native.stderr.log').open('a') as stream:
            stream.write('\nchanged\n')
        with self.assertRaisesRegex(ValueError, 'failed native artifact changed'):
            authority.verify_failed_native_attempt()

    def test_even_repinned_output_fit_success_or_actual_loading_is_refused(self):
        changes = [('calibration/native.stdout.log', b'output'),
                   ('calibration/native.stderr.log', b'\nsuccessfully fit params to free device memory\n'),
                   ('calibration/native.stderr.log', b'\nload_mode = dio\n')]
        for name, suffix in changes:
            path = self.failed / name
            original = path.read_bytes()
            path.write_bytes(original + suffix)
            self.pins[name] = authority.hashed(path)
            try:
                with self.subTest(name=name, suffix=suffix), self.assertRaisesRegex(ValueError, 'pre-inference fitting scope'):
                    authority.verify_failed_native_attempt()
            finally:
                path.write_bytes(original)
                self.pins[name] = authority.hashed(path)


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='pmi-integrity-unit-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'research/proposed-demo').mkdir(parents=True)
        fixtures = {'controller.py': '# harmless source fixture\n',
                    'launch_policy.py': '# held general gate fixture\n',
                    'test_omitted.py': '# tests are outside runtime inventory\n',
                    'research/demo_authority.py': '# authority fixture\n',
                    'research/model-demo-spec.json': '{}\n',
                    'research/proposed-demo/calibration_seed1_fixture.txt': 'fixed fixture\n',
                    'research/FIRST-QUALIFICATION-VERIFICATION.json': '{"fixture":true}\n'}
        for name, content in fixtures.items():
            (self.root / name).write_text(content)
        self.seal = self.root / 'research/DEMO-INPUT-SEAL.json'
        self.qualification = self.root / 'research/FIRST-QUALIFICATION-VERIFICATION.json'
        self.pin = {'schema': 'PMI_REAL_BACKEND_INTEGRITY_V1', 'scope_id': 'pmi-model-demo-v1',
                    'source_sha256': {name: authority.hashed(self.root / name)
                                      for name in fixtures if name.endswith('.py') and not name.startswith('test_')},
                    'scope_artifacts_sha256': {name: authority.hashed(self.root / name)
                                               for name in fixtures if not name.endswith('.py')},
                    'models': {'bonsai-stock': {'prompt_family': model_profiles.prompt_family_policy()}}}
        self.write_seal()
        for name, value in (('CHECKOUT', self.root), ('SEAL', self.seal),
                            ('QUALIFICATION', self.qualification),
                            ('QUALIFICATION_SHA256', authority.hashed(self.qualification)),
                            ('GENERAL_GATE_SHA256', authority.hashed(self.root / 'launch_policy.py')),
                            ('_active', None)):
            replacement = patch.object(authority, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)

    def write_seal(self):
        self.seal.write_text(json.dumps(self.pin))

    def test_reviewed_integrity_reads_complete_inventory_without_activation(self):
        self.assertEqual(authority.verify_integrity(), self.pin)
        self.assertIsNone(authority._active)
        self.assertFalse((self.root / 'demo').exists())

    def test_source_inventory_must_cover_all_runtime_sources(self):
        (self.root / 'new_entry.py').write_text('# unreviewed source\n')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            authority.verify_integrity()
        (self.root / 'new_entry.py').unlink()
        self.pin['source_sha256']['outside.py'] = '0' * 64
        self.write_seal()
        with self.assertRaisesRegex(ValueError, 'inventory'):
            authority.verify_integrity()

    def test_changed_runtime_source_and_required_artifacts_fail_integrity(self):
        source = self.root / 'controller.py'
        original = source.read_bytes()
        source.write_text('# altered runtime fixture\n')
        with self.assertRaisesRegex(ValueError, 'scoped source changed'):
            authority.verify_integrity()
        source.write_bytes(original)
        self.pin['scope_artifacts_sha256'].pop('research/model-demo-spec.json')
        self.write_seal()
        with self.assertRaisesRegex(ValueError, 'inventory incomplete'):
            authority.verify_integrity()

    def test_scope_artifact_paths_cannot_escape_checkout(self):
        for path in ('../outside.json', '/tmp/outside.json'):
            with self.subTest(path=path):
                self.pin['scope_artifacts_sha256'][path] = '0' * 64
                self.write_seal()
                with self.assertRaisesRegex(ValueError, 'scope artifact changed'):
                    authority.verify_integrity()
                del self.pin['scope_artifacts_sha256'][path]

    def test_general_gate_prior_qualification_and_prompt_family_are_pinned(self):
        for target in (self.qualification, self.root / 'launch_policy.py'):
            original = target.read_bytes()
            target.write_text('changed')
            with self.subTest(target=target.name), self.assertRaisesRegex(ValueError, 'prior qualification'):
                authority.verify_integrity()
            target.write_bytes(original)
        self.pin['models']['bonsai-stock']['prompt_family']['profile']['context_tokens'] = 8192
        self.write_seal()
        with self.assertRaisesRegex(ValueError, 'fixed prompt family'):
            authority.verify_integrity()

    def test_integrity_seal_cannot_be_a_symlink_or_other_scope(self):
        self.pin['scope_id'] = 'some-other-run'
        self.write_seal()
        with self.assertRaisesRegex(ValueError, 'separate reviewed'):
            authority.verify_integrity()
        saved = self.seal.with_suffix('.saved')
        self.seal.rename(saved)
        self.seal.symlink_to(saved)
        with self.assertRaisesRegex(ValueError, 'bounded regular'):
            authority.verify_integrity()


class UnrelatedEvidenceTests(unittest.TestCase):
    def test_general_launch_gate_remains_byte_exact_and_held(self):
        import launch_policy
        path = Path(__file__).parent / 'launch_policy.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), authority.GENERAL_GATE_SHA256)
        with self.assertRaises(launch_policy.PlatformHeld):
            launch_policy.require_real_execution()

    def test_existing_cpu_receipts_are_unchanged_from_original_checkout(self):
        here = Path(__file__).parent
        original = Path('/home/luizh/Documents/Codex/2026-10-08/task-8/real-content-factory')
        for name in ('FIRST-CORRECTNESS-RECEIPT.json', 'FIRST-QUALIFICATION-VERIFICATION.json',
                     'FIRST-QUALIFICATION-ATTEMPT-1.json', 'FIRST-QUALIFICATION-POSTFLIGHT-1.json'):
            with self.subTest(name=name):
                self.assertEqual((here / 'research' / name).read_bytes(),
                                 (original / 'research' / name).read_bytes())


if __name__ == '__main__':
    unittest.main()
