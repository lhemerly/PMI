"""Model-free operator contract checks; no physical research execution.

The supervisor and authority boundaries are replaced only where persistence is
tested. A global Popen tripwire prevents model, candidate, or supervisor launch.
Fixture receipts never establish live inference or scientific effectiveness.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research import runtime_entry as entry
from research.demo_prompts import research_configuration


class RuntimePlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.processes = patch('subprocess.Popen', side_effect=AssertionError('physical launch forbidden'))
        self.processes.start()
        self.addCleanup(self.processes.stop)
        self.config = research_configuration('Compare interval algorithms using independent correctness evidence.')

    def test_plan_preserves_valid_configuration_without_launch_authority(self):
        before = copy.deepcopy(self.config)
        with patch('research.run_authority.activate_for_entry', side_effect=AssertionError('plan cannot activate')):
            result = entry.plan(self.config)
        self.assertEqual(self.config, before)
        self.assertEqual(result['config'], before)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_defaults_are_finite_and_protected_stage_caps_are_explicit(self):
        result = entry.plan(self.config)
        saved = result['config']
        self.assertEqual((saved['initial_candidates'], saved['evolutions']), (2, 1))
        self.assertEqual(saved['budget'], {'model_calls': 18, 'experiment_calls': 3,
                                         'wall_seconds': 2400, 'stage_seconds': 120,
                                         'experiment_stage_seconds': 60})

    def test_larger_run_preserves_explicit_budgets_without_automatic_inflation(self):
        config = research_configuration('Test larger finite population.', initial_candidates=4,
                                        evolutions=3, budget={'model_calls': 120,
                                        'experiment_calls': 7, 'wall_seconds': 7200})
        self.assertEqual(entry.plan(config)['config'], config)

    def test_model_evaluator_and_unsupported_fields_are_refused(self):
        invalid = []
        for field, value in [('model_id', 'operator-arbitrary-model'),
                             ('execution_profile', 'unbounded'),
                             ('evaluator', {'id': 'untrusted-evaluator'}),
                             ('shell', 'echo forbidden')]:
            config = copy.deepcopy(self.config)
            config[field] = value
            invalid.append(config)
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                entry.plan(config)

    def test_invalid_and_expanded_stage_limits_never_activate(self):
        for field, values in {'stage_seconds': (121, 0, True),
                              'experiment_stage_seconds': (61, 0, True),
                              'wall_seconds': (0, -1, float('inf'), True),
                              'model_calls': (0, True),
                              'experiment_calls': (0, True)}.items():
            for value in values:
                config = copy.deepcopy(self.config)
                config['budget'][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    entry.plan(config)

    def test_invalid_configuration_is_refused_before_creating_run_or_activating(self):
        bad = copy.deepcopy(self.config)
        bad['budget']['stage_seconds'] = 121
        output = self.root / 'must-not-exist'
        with patch('research.run_authority.activate_for_entry', side_effect=AssertionError('must validate first')):
            with self.assertRaises(ValueError):
                entry.start(bad, output)
        self.assertFalse(output.exists())

    def test_status_missing_run_does_not_create_it(self):
        output = self.root / 'missing'
        with self.assertRaises((ValueError, FileNotFoundError)):
            entry.status(output)
        self.assertFalse(output.exists())

    def test_existing_output_directory_is_not_reused_by_start(self):
        output = self.root / 'occupied'
        output.mkdir()
        note = output / 'retain.txt'
        note.write_text('existing operator data')
        with patch('research.run_authority.activate_for_entry', side_effect=AssertionError('must reject reuse')):
            with self.assertRaises((ValueError, FileExistsError)):
                entry.start(self.config, output)
        self.assertEqual(note.read_text(), 'existing operator data')

    def test_cli_goal_file_plan_uses_file_content(self):
        goal = self.root / 'goal.txt'
        goal.write_text('Investigate touching intervals and empty ranges.')
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            code = entry.main(['plan', '--goal-file', str(goal)])
        self.assertEqual(code, 0)
        result = json.loads(capture.getvalue())
        self.assertEqual(result['config']['goal'], goal.read_text())

    def test_cli_spec_plan_accepts_complete_valid_config(self):
        spec = self.root / 'spec.json'
        spec.write_text(json.dumps(self.config))
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            code = entry.main(['plan', '--spec-file', str(spec)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(capture.getvalue())['config'], self.config)

    def test_cli_rejects_competing_goal_sources(self):
        goal = self.root / 'goal.txt'
        goal.write_text('A goal from a file.')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises((SystemExit, ValueError)):
            entry.main(['plan', '--goal', 'Other goal', '--goal-file', str(goal)])

    def test_cli_flags_change_only_explicit_finite_settings(self):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            code = entry.main(['research-mode-plan', '--goal', 'Larger interval study.',
                               '--initial-candidates', '4', '--evolutions', '3',
                               '--model-calls', '120', '--experiment-calls', '7',
                               '--wall-seconds', '7200'])
        result = json.loads(capture.getvalue())
        self.assertEqual(code, 0)
        config = result['config']
        self.assertEqual((config['initial_candidates'], config['evolutions']), (4, 3))
        self.assertEqual(config['budget'], {'model_calls': 120, 'experiment_calls': 7,
                                          'wall_seconds': 7200, 'stage_seconds': 120,
                                          'experiment_stage_seconds': 60})
        self.assertFalse(result['execution_authorized_by_plan'])

    def test_plan_exposes_demand_when_population_exceeds_default_budget(self):
        result = entry.plan(research_configuration('Explicitly bounded partial study.',
                                                  initial_candidates=4, evolutions=3))
        self.assertGreater(result['expected_full_population_model_calls'], 18)
        self.assertEqual(result['expected_distinct_candidate_checks'], 7)
        self.assertEqual(result['config']['budget']['model_calls'], 18)
        self.assertEqual(result['config']['budget']['experiment_calls'], 3)

    def test_cli_cannot_override_spec_or_existing_run_config(self):
        spec = self.root / 'spec.json'
        spec.write_text(json.dumps(self.config))
        commands = [['plan', '--spec-file', str(spec), '--model-calls', '99'],
                    ['resume', '--run-dir', str(self.root), '--goal', 'Replacement goal'],
                    ['resume', '--run-dir', str(self.root), '--wall-seconds', '999'],
                    ['status', '--run-dir', str(self.root), '--out', str(self.root / 'other')]]
        for arguments in commands:
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises((SystemExit, ValueError)):
                entry.main(arguments)

    def test_content_factory_routes_research_mode_without_legacy_backend(self):
        import content_factory
        arguments = ['research-mode-status', '--run-dir', str(self.root)]
        with patch.object(entry, 'main', return_value=17) as dispatch, \
                patch.object(content_factory, 'require_real_execution',
                             side_effect=AssertionError('legacy backend not expected')):
            self.assertEqual(content_factory.main(arguments), 17)
        dispatch.assert_called_once_with(arguments)


class RuntimePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'run'
        self.config = research_configuration('Persist evidence and never rerun uncertain interval work.')
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('subprocess.Popen', side_effect=AssertionError('physical launch forbidden')))
        self.activations = []
        self.authority = self.stack.enter_context(patch('research.run_authority.activate_for_entry',
                                                       side_effect=self.activate_fixture))
        self.supervisor = self.stack.enter_context(patch('research.deadline.run_demo_supervisor',
                                                        side_effect=self.supervise_fixture))

    def activate_fixture(self, attempt, config, *, resume):
        value = SimpleNamespace(root=Path(attempt), config=copy.deepcopy(config), resume=resume)
        self.activations.append(value)
        return value

    def supervise_fixture(self, *, authority):
        result = {'completed': False, 'cleanup_verified': True,
                  'fixture_label': 'MOCK_ONLY_NO_SUPERVISOR_OR_PHYSICAL_WORK'}
        (authority.root / 'supervisor-report.json').write_text(json.dumps(result))
        return result

    def start(self):
        return entry.start(self.config, self.root)

    def store(self):
        from research.laboratory import LaboratoryStore
        store = LaboratoryStore(self.root / 'workflow')
        store.create(self.config)
        return store

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes()
                for p in self.root.rglob('*') if p.is_file()}

    def test_start_freezes_identity_and_separates_attempt_from_durable_workflow(self):
        result = self.start()
        run = json.loads((self.root / 'run.json').read_text())
        attempt = self.root / 'attempts' / '0001'
        launch = json.loads((attempt / 'launch.json').read_text())
        self.assertEqual(run['config'], self.config)
        self.assertEqual(launch['config'], self.config)
        self.assertEqual(launch['config_sha256'], run['config_sha256'])
        self.assertEqual(launch['root'], str(attempt))
        self.assertEqual(launch['workflow_root'], str(self.root / 'workflow'))
        self.assertFalse(launch['resume'])
        for path in (self.root, self.root / 'attempts', attempt):
            self.assertEqual(path.stat().st_mode & 0o777, 0o700)
        for path in (self.root / 'run.json', attempt / 'launch.json'):
            self.assertEqual(path.stat().st_mode & 0o222, 0)
        self.assertFalse((self.root / 'workflow').exists())
        self.assertFalse(result['supervisor']['completed'])
        self.assertEqual(result['state']['attempt_count'], 1)

    def test_status_is_read_only_and_does_not_claim_live_work_from_mock_receipt(self):
        self.start()
        before = self.snapshot()
        value = entry.status(self.root)
        self.assertEqual(self.snapshot(), before)
        self.assertIsNone(value['workflow'])
        self.assertFalse(value['latest_attempt']['supervisor-report.json']['completed'])
        self.assertEqual(len(self.activations), 1)

    def test_pause_is_idempotent_and_clean_preworkflow_resume_uses_new_attempt(self):
        self.start()
        self.assertTrue(entry.pause(self.root)['pause_requested'])
        pause_record = (self.root / 'pause.json').read_bytes()
        run_record = (self.root / 'run.json').read_bytes()
        first_launch = (self.root / 'attempts/0001/launch.json').read_bytes()
        self.assertTrue(entry.pause(self.root)['pause_requested'])
        self.assertEqual((self.root / 'pause.json').read_bytes(), pause_record)
        result = entry.resume(self.root)
        self.assertEqual(result['state']['attempt_count'], 2)
        self.assertFalse(result['state']['pause_requested'])
        self.assertFalse(self.activations[-1].resume)
        self.assertEqual((self.root / 'run.json').read_bytes(), run_record)
        self.assertEqual((self.root / 'attempts/0001/launch.json').read_bytes(), first_launch)

    def test_ready_workflow_resume_preserves_database_and_remaining_budget(self):
        self.start()
        store = self.store()
        state = store.load()
        state['elapsed_seconds'] = 42.5
        store.save(state)
        before = store.load()
        entry.pause(self.root)
        result = entry.resume(self.root)
        self.assertTrue(self.activations[-1].resume)
        self.assertEqual(self.activations[-1].config, self.config)
        self.assertEqual(store.load(), before)
        self.assertEqual(result['state']['attempt_count'], 2)

    def test_cancel_without_workflow_is_durable_and_cannot_resume(self):
        self.start()
        result = entry.cancel(self.root)
        self.assertTrue(result['cancel_requested'])
        marker = (self.root / 'cancel.json').read_bytes()
        self.assertTrue(entry.cancel(self.root)['cancel_requested'])
        self.assertEqual((self.root / 'cancel.json').read_bytes(), marker)
        with self.assertRaisesRegex(ValueError, 'cancelled'):
            entry.resume(self.root)
        self.assertEqual(len(self.activations), 1)

    def test_cancel_updates_workflow_and_does_not_remove_evidence(self):
        self.start()
        store = self.store()
        before = store.load()
        result = entry.cancel(self.root)
        state = store.load()
        self.assertTrue(state['cancel_requested'])
        self.assertEqual(state['status'], 'cancelled')
        self.assertEqual(state['config'], before['config'])
        self.assertEqual(state['tasks'], before['tasks'])
        self.assertEqual(result['workflow']['status'], 'cancelled')

    def test_missing_or_negative_cleanup_refuses_resume_and_retains_pause(self):
        self.start()
        report = self.root / 'attempts/0001/supervisor-report.json'
        entry.pause(self.root)
        for value in (None, {'cleanup_verified': False}, {'cleanup_verified': 'true'}):
            if report.exists(): report.unlink()
            if value is not None: report.write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'cleanup'):
                entry.resume(self.root)
            self.assertTrue((self.root / 'pause.json').is_file())
            self.assertEqual(len(self.activations), 1)

    def test_terminal_or_blocked_workflow_is_not_a_calibration_only_resume(self):
        self.start()
        store = self.store()
        for terminal in ('blocked', 'budget_exhausted', 'cancelled'):
            state = store.load()
            state.update(status=terminal, reason='MODEL_FREE_CONTROL_FIXTURE')
            store.save(state)
            with self.subTest(status=terminal), self.assertRaisesRegex(ValueError, 'terminal or blocked'):
                entry.resume(self.root)
            self.assertEqual(len(self.activations), 1)

    def test_uncertain_reserved_task_refuses_replay_before_authority(self):
        self.start()
        store = self.store()
        state = store.load()
        state['inflight'] = 'seed-1'
        state['tasks'][0]['status'] = 'reserved'
        # This fixture isolates the entry boundary. Store validation itself is
        # covered separately; no invalid row is written to the real database.
        with patch.object(entry.LaboratoryStore, 'load', return_value=state):
            with self.assertRaisesRegex(ValueError, 'uncertain reserved'):
                entry.resume(self.root)
        self.assertEqual(len(self.activations), 1)

    def test_launch_lock_remains_owned_during_supervisor(self):
        observations = []
        def supervised(*, authority):
            with self.assertRaisesRegex(ValueError, 'active launcher'):
                entry.resume(self.root)
            observations.append('no overlapping attempt')
            return self.supervise_fixture(authority=authority)
        self.supervisor.side_effect = supervised
        self.start()
        self.assertEqual(observations, ['no overlapping attempt'])
        self.assertEqual(len(self.activations), 1)

    def test_authority_failure_is_preserved_without_fabricated_cleanup(self):
        self.authority.side_effect = ValueError('fixture launch boundary held')
        with self.assertRaisesRegex(ValueError, 'boundary held'):
            self.start()
        receipt = json.loads((self.root / 'attempts/0001/launch-error.json').read_text())
        self.assertFalse(receipt['cleanup_claim'])
        self.assertFalse(receipt['automatic_retry'])
        self.supervisor.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'cleanup'):
            entry.resume(self.root)

    def test_manifest_edits_are_detected_without_a_new_attempt(self):
        self.start()
        manifest = self.root / 'run.json'
        row = json.loads(manifest.read_text())
        row['config']['goal'] = 'Changed goal without changing identity'
        manifest.chmod(0o600)
        manifest.write_text(json.dumps(row))
        manifest.chmod(0o400)
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            entry.resume(self.root)
        self.assertEqual(len(self.activations), 1)


class RuntimeChildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.run = self.base / 'run'
        self.attempt = self.run / 'attempts/0001'
        self.attempt.mkdir(parents=True)
        self.checkout = self.base / 'checkout'
        (self.checkout / 'research/proposed-demo').mkdir(parents=True)
        (self.checkout / 'research/proposed-demo/calibration_seed1_fixture.txt').write_text('MODEL_FREE_PROMPT_FIXTURE')
        self.finished = []
        self.events = []
        self.retained = []
        self.workflow_status = 'ready'
        self.coverage = False
        self.calibration_error = None
        test = self

        class Context:
            output_root = test.attempt
            operation_deadline = 1000.0
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def retain_coordination(self, owner): test.retained.append(owner)
            def finish(self, value): test.finished.append(value)

        class Owner:
            def __enter__(self): return self
            def __exit__(self, *args): pass

        self.authority = SimpleNamespace(
            root=self.attempt, workflow_root=self.run / 'workflow',
            config=research_configuration('MODEL_FREE child orchestration fixture.'),
            model_count=0, candidate_count=0, calibration_count=0, current=None,
            calibrated=self.calibrated_fixture)
        self.agent = SimpleNamespace(call=self.calibration_fixture)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('subprocess.Popen', side_effect=AssertionError('physical launch forbidden')))
        self.stack.enter_context(patch('research.deadline.read_supervisor_context', return_value=Context()))
        self.stack.enter_context(patch('research.coordination.CoordinationOwner', Owner))
        self.stack.enter_context(patch('research.run_authority.activate_child', return_value=self.authority))
        self.stack.enter_context(patch('research.demo_authority.CHECKOUT', self.checkout))
        self.stack.enter_context(patch('research.harness.DemoRealAgent', return_value=self.agent))
        self.stack.enter_context(patch.object(entry, 'RuntimeCorrectnessRunner', return_value=object()))
        self.drive = self.stack.enter_context(patch('research.laboratory.drive', side_effect=self.drive_fixture))
        self.stack.enter_context(patch.object(entry, 'summary', side_effect=lambda state: {
            'status': state['status'], 'objective_met': False,
            'demonstration': {'completed_real_demonstration': self.coverage},
            'fixture_label': 'MOCK_ONLY_NO_LIVE_RESEARCH'}))
        self.stack.enter_context(patch.object(entry.time, 'monotonic', return_value=100.0))

    def calibration_fixture(self, role, model, prompt, stage, deadline):
        self.events.append(('calibration', role, model, prompt, deadline))
        self.authority.calibration_count += 1
        return {'text': 'MODEL_FREE_RESPONSE_FIXTURE', 'actual_inference': False,
                'cleanup_verified': True, 'fixture_label': 'MOCK_ONLY_NO_MODEL'}

    def calibrated_fixture(self):
        self.events.append(('calibrated',))
        if self.calibration_error is not None: raise self.calibration_error

    def drive_fixture(self, store, agent, runner, lease, *, coordination, should_pause):
        self.assertEqual(self.events[-1], ('calibrated',))
        self.assertIs(agent, self.agent)
        self.assertIs(coordination, self.retained[0])
        self.assertTrue(store.path.is_file())
        self.events.append(('drive', should_pause()))
        state = store.load()
        state['status'] = self.workflow_status
        return state

    def report(self):
        return json.loads((self.attempt / 'child-result.json').read_text())

    def test_child_calibrates_before_workflow_and_reports_pause_without_live_claim(self):
        self.assertEqual(entry.child_main(), 0)
        self.assertEqual(self.events[0], ('calibration', 'calibration', 'bonsai-stock',
                                         'MODEL_FREE_PROMPT_FIXTURE', 220.0))
        self.assertEqual(self.events[1:], [('calibrated',), ('drive', False)])
        report = self.report()
        self.assertEqual(report['status'], 'paused')
        self.assertFalse(report['calibration']['actual_inference'])
        self.assertFalse(report['workflow']['objective_met'])
        self.assertEqual(report['actual_model_completions'], 0)
        self.assertEqual(report['generated_candidate_receipts'], 0)
        self.assertEqual(self.finished, ['finished'])

    def test_invalid_calibration_keeps_workflow_uncreated(self):
        self.calibration_error = ValueError('fixture response fails profile contract')
        self.assertEqual(entry.child_main(), 2)
        self.drive.assert_not_called()
        self.assertFalse(self.authority.workflow_root.exists())
        self.assertEqual(self.report()['status'], 'blocked')
        self.assertIn('fails profile contract', self.report()['error'])
        self.assertEqual(self.finished, ['blocked'])

    def test_partial_workflow_does_not_claim_full_demonstration(self):
        self.workflow_status = 'finished'
        self.assertEqual(entry.child_main(), 0)
        self.assertEqual(self.report()['status'], 'finished_without_full_coverage')
        self.assertFalse(self.report()['workflow']['objective_met'])
        self.assertEqual(self.report()['actual_model_completions'], 0)

    def test_unexpected_child_flag_arguments_do_not_enter_authenticated_child(self):
        with patch.object(entry, 'child_main', side_effect=AssertionError('no child activation')), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            entry.main(['--supervised-child', '--goal', 'untrusted substitution'])


if __name__ == '__main__':
    unittest.main()
