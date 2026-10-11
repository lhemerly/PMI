"""Reusable authority contracts only: temporary records/memfds, no workloads."""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import model_profiles
from research import demo_authority, run_authority as authority
from research.demo_prompts import research_configuration
from research.experiments import CORRECTNESS_EVALUATOR, ExperimentRequest
from research.executor import build_plan
from research.interval_evaluator import cases, validate_source

NOW = 1000.0
SOURCE = 'def merge_intervals(intervals):\n    return []\n'


class RuntimeAuthorityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='pmi-runtime-authority-unit-')
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / 'attempt'
        self.root.mkdir(mode=0o700)
        self.workflow = self.base / 'workflow'
        self.checkout = self.base / 'checkout'
        self.fixture = self.checkout / 'research/proposed-demo/calibration_seed1_fixture.txt'
        self.fixture.parent.mkdir(parents=True)
        self.fixture.write_text('fixed calibration fixture')
        self.seal = self.checkout / 'research/RESEARCH-INPUT-SEAL.json'
        self.seal.write_text('{"fixture":"integrity-only"}')
        self.config = research_configuration('Explore interval unions', initial_candidates=4,
            budget={'model_calls': 80, 'experiment_calls': 8, 'wall_seconds': 3600})
        self.row = {'schema': 'PMI_RESEARCH_LAUNCH_V1', 'root': str(self.root),
                    'workflow_root': str(self.workflow), 'config': self.config,
                    'config_sha256': authority.digest(self.config), 'resume': False}
        self.write_launch()
        self.state = {'config': self.config, 'mode': 'real', 'inflight': None,
                      'tasks': [], 'candidates': [], 'model_calls': 0, 'experiment_calls': 0,
                      'elapsed_seconds': 0., 'status': 'running', 'cancel_requested': False}
        self.integrity = Mock(return_value={'fixture': 'integrity-only'})
        for name, value in (('CHECKOUT', self.checkout), ('SEAL', self.seal), ('_active', None),
                            ('verify_integrity', self.integrity), ('RUNTIME', self.base / 'runtime')):
            replacement = patch.object(authority, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        for replacement in (patch.object(authority.time, 'monotonic', return_value=NOW),
                            patch.object(authority, '_state', side_effect=lambda _: self.state)):
            replacement.start()
            self.addCleanup(replacement.stop)
        self.coordination = SimpleNamespace(validate=Mock())

    def write_launch(self):
        path = self.root / 'launch.json'
        if path.exists():
            path.chmod(0o600)
        path.write_text(json.dumps(self.row))
        path.chmod(0o400)
        self.launch_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    def child(self, *, calibrated=True, count=0):
        child = authority.ResearchAuthority('child', self.row, self.launch_hash,
            NOW + 3780, NOW + 3600, (count, 0, 0., set(), set()), SimpleNamespace(validate=Mock()))
        child.calibration_done = calibrated
        child.calibration_count = int(calibrated)
        authority._active = child
        return child

    def model_args(self, task='seed-4', role='generation'):
        prompt = self.fixture.read_text() if role == 'calibration' else 'bounded research prompt'
        stage = self.root / 'calibration' if role == 'calibration' else self.workflow / ('task-' + task)
        return dict(model_id='bonsai-stock', role=role, prompt=prompt, stage=stage,
                    deadline=NOW + 90, coordination=self.coordination)

    def reserve(self, child, task='seed-4', role='generation'):
        args = self.model_args(task, role)
        if role != 'calibration':
            self.state.update(inflight=task, model_calls=child.model_count + 1,
                tasks=[{'id': task, 'kind': role, 'status': 'reserved', 'targets': [],
                        'model_io': {'prompt_sha256': hashlib.sha256(args['prompt'].encode()).hexdigest()}}])
        authority.gate_model(child, 'agent', **args)
        authority.gate_model(child, 'backend', **args)
        request = model_profiles.model_request('bonsai-stock', args['prompt'], 'bonsai-research-v1', role)
        path = args['stage'] / 'request.json'
        path.parent.mkdir(mode=0o700, parents=True)
        path.write_text(json.dumps(request))
        return args, request, path

    def complete(self, child, args):
        authority.gate_model(child, 'complete', **args,
            capture={'returncode': 0, 'owned_cleanup_empty': True, 'postflight_passed': True},
            receipt={'actual_inference': True})

    def test_activation_uses_configured_budget_and_reserves_attempt_once(self):
        parent = authority.activate_for_entry(self.root, self.config)
        self.assertEqual(parent.operation_deadline, NOW + 3720)
        self.assertEqual(parent.deadline, NOW + 3900)
        self.assertEqual(parent.child_module, 'research.runtime_entry')
        self.assertEqual(parent.model_count, 0)
        self.assertEqual(parent.calibration_count, 0)
        self.assertTrue((self.root / 'authority-reserved.json').is_file())
        with self.assertRaises(ValueError):
            authority.activate_for_entry(self.root, self.config)
        authority._active = None
        with self.assertRaises(FileExistsError):
            authority.activate_for_entry(self.root, self.config)

    def test_launch_requires_immutable_private_exact_config_and_separate_roots(self):
        (self.root / 'launch.json').chmod(0o600)
        with self.assertRaises(ValueError):
            authority.activate_for_entry(self.root, self.config)
        self.write_launch()
        self.root.chmod(0o755)
        with self.assertRaises(ValueError):
            authority.activate_for_entry(self.root, self.config)
        self.root.chmod(0o700)
        self.row['workflow_root'] = str(self.root / 'workflow')
        self.write_launch()
        with self.assertRaises(ValueError):
            authority.activate_for_entry(self.root, self.config)
        self.row['workflow_root'] = str(self.workflow)
        self.row['config_sha256'] = '0' * 64
        self.write_launch()
        with self.assertRaises(ValueError):
            authority.activate_for_entry(self.root, self.config)

    def test_configuration_cannot_expand_model_stage_or_change_evaluator(self):
        for key, value in (('model_id', 'qwen3-stock'), ('evaluator', {}), ('execution_profile', 'arbitrary')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                authority._configuration({**self.config, key: value})
        with self.assertRaises(ValueError):
            authority._configuration({**self.config, 'budget': {**self.config['budget'], 'stage_seconds': 121}})

    def supervisor_context(self, parent):
        from research.deadline import SupervisorContext
        context = SupervisorContext.__new__(SupervisorContext)
        context.output_root = str(self.root)
        context.child_pid = os.getpid()
        context.child_module = 'research.runtime_entry'
        context.operation_deadline = parent.operation_deadline
        context.final_deadline = parent.deadline
        context.supervisor_identity = parent.owner
        return context

    def test_child_activation_binds_exact_context_module_parent_and_deadlines(self):
        from research.deadline import SupervisorContext
        parent = authority.activate_for_entry(self.root, self.config)
        context = self.supervisor_context(parent)
        authority._active = None  # model the new exec process, not a live fork
        with patch.object(SupervisorContext, 'validate', return_value=context):
            child = authority.activate_child(context)
        self.assertEqual(child.kind, 'child')
        self.assertIs(child.context, context)
        self.assertEqual(child.operation_deadline, parent.operation_deadline)
        self.assertEqual(child.deadline, parent.deadline)
        self.assertEqual(child.launch_sha256, parent.launch_sha256)

    def test_child_activation_refuses_fake_cross_mode_expanded_or_wrong_parent_context(self):
        from research.deadline import SupervisorContext
        parent = authority.activate_for_entry(self.root, self.config)
        authority._active = None
        with self.assertRaises(ValueError):
            authority.activate_child(SimpleNamespace(validate=lambda: None))
        for field, value in (('child_module', 'research.model_demo_entry'), ('child_pid', -1),
                             ('operation_deadline', parent.operation_deadline + 1),
                             ('final_deadline', parent.deadline + 1), ('supervisor_identity', (-1, -1))):
            context = self.supervisor_context(parent)
            setattr(context, field, value)
            with patch.object(SupervisorContext, 'validate', return_value=context):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    authority.activate_child(context)
        self.assertIsNone(authority._active)

    def test_resume_budgets_and_completed_task_identities_are_preserved(self):
        self.workflow.mkdir()
        self.row['resume'] = True
        self.write_launch()
        self.state.update(model_calls=25, experiment_calls=4, elapsed_seconds=400.,
            tasks=[{'id': 'seed-1', 'kind': 'generation', 'status': 'done'},
                   {'id': 'experiment-1', 'kind': 'experiment', 'status': 'done'}])
        parent = authority.activate_for_entry(self.root, self.config, resume=True)
        self.assertEqual((parent.model_count, parent.candidate_count), (25, 4))
        self.assertEqual(parent.operation_deadline, NOW + 3320)
        self.assertEqual(parent.used_model_tasks, {'seed-1'})
        self.assertEqual(parent.used_cpu_tasks, {'experiment-1'})

    def test_resume_refuses_uncertain_blocked_fixture_cancelled_or_changed_work(self):
        self.workflow.mkdir()
        self.row['resume'] = True
        self.write_launch()
        base = dict(self.state)
        for changes in ({'inflight': 'seed-1'}, {'status': 'blocked'}, {'mode': 'fixture'},
                        {'cancel_requested': True}, {'status': 'finished'},
                        {'tasks': [{'id': 'seed-1', 'status': 'reserved'}]},
                        {'config': {**self.config, 'goal': 'different'}}):
            self.state = {**base, **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                authority.activate_for_entry(self.root, self.config, resume=True)
        self.assertFalse((self.root / 'authority-reserved.json').exists())

    def test_process_identity_context_and_launch_mutation_are_checked(self):
        child = self.child()
        with self.assertRaises(ValueError):
            authority.reviewed_backend_seal(object())
        child.owner = (os.getpid(), -1)
        with self.assertRaises(ValueError):
            authority.reviewed_backend_seal(child)
        child.owner = authority.identity(os.getpid())[0]
        child.context.validate.side_effect = ValueError('lost supervisor')
        with self.assertRaises(ValueError):
            authority.reviewed_backend_seal(child)
        child.context.validate.side_effect = None
        self.row['config']['goal'] = 'changed'
        self.row['config_sha256'] = authority.digest(self.row['config'])
        self.write_launch()
        with self.assertRaises(ValueError):
            authority.reviewed_backend_seal(child)

    def test_calibration_is_one_separate_call_not_workflow_budget(self):
        child = self.child(calibrated=False, count=25)
        args, request, path = self.reserve(child, role='calibration')
        self.assertEqual((child.model_count, child.calibration_count), (25, 1))
        with authority.native_capability(child, path, request, args['deadline']):
            pass
        self.complete(child, args)
        (args['stage'] / 'answer.txt').write_text(SOURCE)
        child.calibrated()
        with self.assertRaises(ValueError):
            authority.gate_model(child, 'agent', **args)
        self.assertEqual(child.model_count, 25)

    def test_workflow_calls_can_pass_old_demo_limit_but_respect_configured_limit(self):
        child = self.child(count=25)
        args, request, path = self.reserve(child, task='evolve-9', role='evolution')
        self.assertEqual(child.model_count, 26)
        with authority.native_capability(child, path, request, args['deadline']):
            pass
        self.complete(child, args)
        child.model_count = self.config['budget']['model_calls']
        with self.assertRaises(ValueError):
            self.reserve(child, task='evolve-10', role='evolution')

    def test_reservation_requires_matching_prompt_counter_role_and_stage(self):
        child = self.child()
        args = self.model_args()
        task = {'id': 'seed-4', 'kind': 'generation', 'status': 'reserved', 'targets': [],
                'model_io': {'prompt_sha256': hashlib.sha256(args['prompt'].encode()).hexdigest()}}
        self.state.update(inflight='seed-4', model_calls=1, tasks=[task])
        for changes in ({'prompt': 'changed'}, {'role': 'reflection'}, {'stage': self.base / 'task-seed-4'},
                        {'deadline': NOW + 121}, {'deadline': child.operation_deadline + 1}, {'coordination': None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                authority.gate_model(child, 'agent', **{**args, **changes})
        self.state['model_calls'] = 2
        with self.assertRaises(ValueError):
            authority.gate_model(child, 'agent', **args)
        self.assertEqual(child.model_count, 0)

    def test_complete_needs_native_cleanup_and_does_not_accept_retries(self):
        child = self.child()
        args, request, path = self.reserve(child)
        with self.assertRaises(ValueError):
            self.complete(child, args)
        with authority.native_capability(child, path, request, args['deadline']):
            pass
        self.complete(child, args)
        with self.assertRaises(ValueError):
            authority.gate_model(child, 'agent', **args)

    def test_cancel_prevents_activation_and_dispatch_but_allows_completion(self):
        marker = self.base / 'cancel.json'
        marker.write_text('{}')
        with self.assertRaises(ValueError):
            authority.activate_for_entry(self.root, self.config)
        marker.unlink()
        child = self.child()
        args, request, path = self.reserve(child)
        with authority.native_capability(child, path, request, args['deadline']):
            pass
        marker.write_text('{}')
        self.complete(child, args)
        with self.assertRaises(ValueError):
            self.reserve(child, task='seed-5')

    @contextmanager
    def controller_command(self, module='research.runtime_entry'):
        actual_read = Path.read_bytes
        target = Path('/proc/%d/cmdline' % os.getpid())
        def read(path):
            return ('python3\0-B\0-m\0' + module + '\0--supervised-child\0').encode() if path == target else actual_read(path)
        with patch.object(Path, 'read_bytes', autospec=True, side_effect=read):
            yield

    def test_v2_capability_binds_launch_seal_request_and_runtime_controller(self):
        child = self.child()
        args, request, path = self.reserve(child)
        with authority.native_capability(child, path, request, args['deadline']) as capability:
            self.assertEqual(authority.capability_schema(capability), 'PMI_SEALED_NATIVE_REQUEST_V2')
            self.assertTrue(authority.is_runtime_capability_path(capability))
            with self.controller_command():
                authority.gate_native(path, request, capability_path=capability)
                demo_authority.gate_native(path, request, capability_path=capability)
                with self.assertRaises(ValueError):
                    authority.gate_native(path, {**request, 'prompt': 'different'}, capability_path=capability)
                self.seal.write_text('{"different":true}')
                with self.assertRaisesRegex(ValueError, 'source seal changed'):
                    authority.gate_native(path, request, capability_path=capability)
            with self.controller_command('research.model_demo_entry'):
                with self.assertRaisesRegex(ValueError, 'fixed supervised research controller'):
                    authority.gate_native(path, request, capability_path=capability)
        self.assertFalse(Path(capability).exists())

    def test_native_request_mismatch_and_capability_reuse_are_rejected(self):
        child = self.child()
        args, request, path = self.reserve(child)
        wrong = model_profiles.model_request('bonsai-stock', args['prompt'], 'bonsai-research-v1', 'ranking')
        path.write_text(json.dumps(wrong))
        with self.assertRaises(ValueError):
            with authority.native_capability(child, path, wrong, args['deadline']):
                self.fail('unreserved request accepted')
        path.write_text(json.dumps(request))
        with authority.native_capability(child, path, request, args['deadline']):
            pass
        with self.assertRaises(ValueError):
            with authority.native_capability(child, path, request, args['deadline']):
                self.fail('native request reused')

    def test_cpu_candidate_four_uses_same_qualified_plan_and_budget(self):
        child = self.child()
        child.candidate_count = 3
        task_id = 'experiment-4'
        request = ExperimentRequest(task_id, 4, validate_source(SOURCE), authority.digest(CORRECTNESS_EVALUATOR),
                                    'interval-union-correctness-v1', 'correctness-small-v1')
        self.state.update(inflight=task_id, experiment_calls=4,
            tasks=[{'id': task_id, 'kind': 'experiment', 'status': 'reserved', 'targets': [4]}],
            candidates=[{'id': i, 'source': SOURCE, 'sha256': validate_source(SOURCE),
                         'evidence': None, 'duplicate_of': None} for i in range(1, 5)])
        stage = self.workflow / ('task-' + task_id)
        authority.gate_cpu(child, 'candidate', request=request, source=SOURCE, stage=stage)
        self.assertEqual(child.candidate_count, 4)
        plan = build_plan(authority.RUNTIME, stage / 'candidate', stage / 'artifacts', 10, 11, cases(), (12, 13))
        with self.assertRaises(ValueError):
            authority.gate_cpu(child, 'transport', plan=plan)
        with self.assertRaises(ValueError):
            authority.gate_cpu(child, 'admission', plan=replace(plan, wall_seconds=11))
        authority.gate_cpu(child, 'admission', plan=plan)
        authority.gate_cpu(child, 'transport', plan=plan)
        from research import experiments
        with patch.object(experiments, 'validate_receipt', return_value={'validated': True}) as validate:
            child.cpu_complete({'unit': 'receipt fixture'})
            validate.assert_called_once_with({'unit': 'receipt fixture'}, request, True)
        with self.assertRaises(ValueError):
            authority.gate_cpu(child, 'candidate', request=request, source=SOURCE, stage=stage)

    def test_no_packaging_authored_qualification_or_shell_cpu_operation(self):
        child = self.child()
        for operation in ('package', 'once', 'shell'):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                authority.gate_cpu(child, operation)

    def test_exact_type_dispatch_reaches_runtime_without_releasing_general_gate(self):
        child = self.child()
        self.assertTrue(demo_authority.is_demo_authority(child))
        self.assertEqual(demo_authority.reviewed_backend_seal(child), {'fixture': 'integrity-only'})
        from research.qualification_authority import gate_cpu
        with self.assertRaisesRegex(ValueError, 'no arbitrary commands'):
            gate_cpu(child, 'shell')
        import launch_policy
        with self.assertRaises(launch_policy.PlatformHeld):
            gate_cpu(None, 'shell')
        class Impostor(authority.ResearchAuthority):
            pass
        self.assertFalse(authority.is_authority(Impostor.__new__(Impostor)))


class RuntimeIntegrityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='pmi-runtime-integrity-unit-')
        self.addCleanup(temporary.cleanup)
        self.checkout = Path(temporary.name)
        (self.checkout / 'research/proposed-demo').mkdir(parents=True)
        contents = {'launch_policy.py': '# held fixture\n', 'controller.py': '# controller fixture\n',
                    'research/run_authority.py': '# authority fixture\n',
                    'research/FIRST-QUALIFICATION-VERIFICATION.json': '{"qualified":true}',
                    'research/proposed-demo/calibration_seed1_fixture.txt': 'calibration fixture'}
        for name, text in contents.items():
            (self.checkout / name).write_text(text)
        self.qualification = self.checkout / 'research/FIRST-QUALIFICATION-VERIFICATION.json'
        self.manifest = self.checkout / 'runtime-manifest.json'
        self.manifest.write_text('{"runtime":"fixture"}')
        self.seal = self.checkout / 'research/RESEARCH-INPUT-SEAL.json'
        self.pin = {'schema': 'PMI_REAL_BACKEND_INTEGRITY_V1', 'scope_id': 'pmi-research-runtime-v1',
                    'source_sha256': {name: authority.hashed(self.checkout / name) for name in contents if name.endswith('.py')},
                    'scope_artifacts_sha256': {name: authority.hashed(self.checkout / name) for name in contents if not name.endswith('.py')},
                    'external_artifacts_sha256': {str(self.manifest): authority.hashed(self.manifest)},
                    'models': {'bonsai-stock': {'prompt_family': model_profiles.prompt_family_policy()}}}
        self.write_seal()
        for name, value in (('CHECKOUT', self.checkout), ('SEAL', self.seal), ('QUALIFICATION', self.qualification),
                            ('QUALIFICATION_SHA256', authority.hashed(self.qualification)),
                            ('GENERAL_GATE_SHA256', authority.hashed(self.checkout / 'launch_policy.py')),
                            ('RUNTIME_MANIFEST', self.manifest), ('_active', None)):
            replacement = patch.object(authority, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)

    def write_seal(self):
        self.seal.write_text(json.dumps(self.pin))

    def test_full_seal_validates_without_activating_authority(self):
        self.assertEqual(authority.verify_integrity(), self.pin)
        self.assertIsNone(authority._active)

    def test_unsealed_or_changed_source_is_refused(self):
        extra = self.checkout / 'unreviewed.py'
        extra.write_text('# additional executable entry\n')
        with self.assertRaisesRegex(ValueError, 'source inventory'):
            authority.verify_integrity()
        extra.unlink()
        (self.checkout / 'controller.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            authority.verify_integrity()

    def test_qualification_gate_and_runtime_manifest_are_pinned(self):
        for path in (self.qualification, self.checkout / 'launch_policy.py', self.manifest):
            original = path.read_bytes()
            path.write_text('changed')
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                authority.verify_integrity()
            path.write_bytes(original)

    def test_prompt_family_scope_and_artifact_paths_cannot_expand(self):
        self.pin['scope_id'] = 'pmi-model-demo-v1'
        self.write_seal()
        with self.assertRaises(ValueError):
            authority.verify_integrity()
        self.pin['scope_id'] = 'pmi-research-runtime-v1'
        self.pin['scope_artifacts_sha256']['../escape'] = '0' * 64
        self.write_seal()
        with self.assertRaises(ValueError):
            authority.verify_integrity()
        del self.pin['scope_artifacts_sha256']['../escape']
        self.pin['models']['bonsai-stock']['prompt_family']['profile']['context_tokens'] = 8192
        self.write_seal()
        with self.assertRaisesRegex(ValueError, 'prompt family changed'):
            authority.verify_integrity()


if __name__ == '__main__':
    unittest.main()
