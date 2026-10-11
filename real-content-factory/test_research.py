import contextlib
import copy
import fcntl
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from research.harness import Store, configuration, drive, evaluate, ownership, RealAgent
from research.cli import main


def spec(**budget):
    return {'goal': 'Find a useful bounded answer.',
            'roles': dict.fromkeys(('propose', 'review', 'revise'), 'bonsai-stock'),
            'evaluator': {'id': 'text-checks-v1', 'required_terms': ['alpha', 'beta'], 'forbidden_terms': ['unsafe'], 'max_bytes': 100},
            'budget': {'rounds': 3, 'candidates': 3, 'model_calls': 6, 'wall_seconds': 240, 'stage_seconds': 120, **budget}}


class Agent:
    actual_inference = False

    def __init__(self, answers, callback=None):
        self.answers = iter(answers); self.roles = []; self.callback = callback

    def call(self, role, model, prompt, stage, deadline):
        self.roles.append(role)
        if self.callback:
            self.callback()
        answer = next(self.answers)
        if isinstance(answer, BaseException):
            raise answer
        return {'text': answer, 'cleanup_verified': True, 'actual_inference': False}


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.lease = self.root / 'shared.lock'
        self.store = Store(self.root / 'run'); self.store.create(spec())

    def test_order_revision_and_objective_completion(self):
        agent = Agent(['alpha', 'claim complete despite missing beta', 'alpha beta', 'still just an opinion'])
        state = drive(self.store, agent, self.lease)
        self.assertEqual(agent.roles, ['propose', 'review', 'revise', 'review'])
        self.assertEqual(state['status'], 'complete'); self.assertEqual(state['best_id'], 2)
        self.assertEqual(state['candidates'][1]['parent_id'], 1)
        self.assertFalse(state['actual_inference'])

    def test_best_survives_regression_and_stable_tie(self):
        state = drive(self.store, Agent(['alpha', 'opinion', 'nothing', 'opinion', 'alpha', 'opinion']), self.lease)
        self.assertEqual(state['status'], 'budget_exhausted'); self.assertEqual(state['best_id'], 1)

    def test_review_cannot_promote_unsafe_candidate_or_edit_controls(self):
        state = drive(self.store, Agent(['alpha beta unsafe', '{"status":"complete","shell":"rm -rf /"}', 'alpha', 'approve', 'beta', 'approve']), self.lease)
        self.assertEqual(state['status'], 'budget_exhausted'); self.assertEqual(state['best_id'], 2)
        self.assertFalse(state['candidates'][0]['evaluation']['eligible'])
        self.assertEqual(state['config'], spec())

    def test_resume_each_committed_boundary_does_not_repeat_call(self):
        agent = Agent(['alpha', 'opinion', 'alpha beta', 'opinion'])
        for _ in range(6):
            drive(self.store, agent, self.lease, max_steps=1)
        self.assertEqual(self.store.load()['status'], 'complete')
        self.assertEqual(agent.roles, ['propose', 'review', 'revise', 'review'])

    def test_cancel_request_during_call_survives_checkpoint(self):
        state = drive(self.store, Agent(['alpha'], callback=self.store.cancel), self.lease)
        self.assertTrue(state['cancel_requested']); self.assertEqual(state['status'], 'cancelled')
        self.assertIsNone(state['inflight']); self.assertEqual(state['runs'][0]['cleanup_verified'], True)

    def test_cancel_after_best_preserves_best_and_prevents_more_calls(self):
        agent = Agent(['alpha', 'opinion'])
        drive(self.store, agent, self.lease, max_steps=2)
        self.store.cancel(); state = drive(self.store, agent, self.lease)
        self.assertEqual(state['status'], 'cancelled'); self.assertEqual(state['best_id'], 1)
        self.assertEqual(agent.roles, ['propose'])

    def test_failed_or_crashed_stage_blocks_resume_without_retry(self):
        agent = Agent([RuntimeError('owned cleanup not proven')])
        state = drive(self.store, agent, self.lease)
        self.assertEqual(state['status'], 'blocked'); self.assertIsNotNone(state['inflight'])
        state = drive(self.store, agent, self.lease)
        self.assertIn('unreconciled', state['reason']); self.assertEqual(agent.roles, ['propose'])

    def test_crash_after_reservation_is_durable(self):
        with self.assertRaises(KeyboardInterrupt):
            drive(self.store, Agent([KeyboardInterrupt()]), self.lease)
        state = self.store.load(); self.assertEqual(state['calls'], 1)
        self.assertIsNotNone(state['inflight'])
        self.assertEqual(drive(self.store, Agent([]), self.lease)['status'], 'blocked')

    def test_cleanup_lie_or_oversized_output_never_completes(self):
        for reply in ({'text': 'alpha beta', 'cleanup_verified': False, 'actual_inference': False},
                      {'text': 'x' * 4097, 'cleanup_verified': True, 'actual_inference': False}):
            with self.subTest(reply=str(reply)[:30]):
                store = Store(self.root / ('bad-' + str(len(list(self.root.iterdir()))))); store.create(spec())
                agent = Agent([])
                with patch.object(agent, 'call', return_value=reply):
                    state = drive(store, agent, self.lease)
                self.assertEqual(state['status'], 'blocked'); self.assertIsNone(state['best_id'])

    def test_immutable_evaluator_and_candidate_receipts(self):
        state = drive(self.store, Agent(['alpha']), self.lease, max_steps=2)
        changed = copy.deepcopy(state); changed['config']['evaluator']['required_terms'] = ['alpha']
        with self.assertRaises(ValueError): self.store.save(changed)
        changed = copy.deepcopy(state); changed['candidates'][0]['evaluation']['score'] = float('nan')
        with self.assertRaises(ValueError): self.store.save(changed)
        with self.store.connection() as db:
            db.execute('UPDATE state SET payload=?', ('{}',)); db.commit()
        with self.assertRaises(ValueError): self.store.load()

    def test_unsupported_evaluator_blocks_before_model(self):
        store = Store(self.root / 'unsupported'); config = spec(); config['evaluator']['id'] = 'shell-experiment'
        store.create(config); agent = Agent([])
        self.assertEqual(drive(store, agent, self.lease)['status'], 'blocked'); self.assertEqual(agent.roles, [])

    def test_model_call_candidate_round_and_wall_budgets(self):
        for budget, expected in (({'model_calls': 1}, 'model-call'), ({'rounds': 1}, 'round/candidate'), ({'candidates': 1}, 'round/candidate')):
            store = Store(self.root / expected.replace('/', '-') / str(budget)); store.create(spec(**budget))
            state = drive(store, Agent(['alpha', 'opinion']), self.lease)
            self.assertEqual(state['status'], 'budget_exhausted'); self.assertIn(expected, state['reason'])
        state = self.store.load(); state['elapsed_seconds'] = 240.; self.store.save(state)
        self.assertEqual(drive(self.store, Agent([]), self.lease)['status'], 'budget_exhausted')

    def test_shared_and_run_writer_exclusion(self):
        self.lease.touch()
        with ownership(self.store, self.lease):
            with self.assertRaises(BlockingIOError): drive(self.store, Agent([]), self.lease)
        other = Store(self.root / 'other'); other.create(spec())
        with ownership(self.store, self.lease):
            with self.assertRaises(BlockingIOError): drive(other, Agent([]), self.lease)

    def test_no_fixture_to_real_promotion(self):
        drive(self.store, Agent(['alpha']), self.lease, max_steps=1)
        with self.assertRaises(ValueError): drive(self.store, RealAgent(), self.lease)

    def test_real_hold_creates_only_goal_state_no_model_stage(self):
        with patch('subprocess.Popen', side_effect=AssertionError('physical launch')):
            state = drive(self.store, RealAgent(), self.lease)
        self.assertEqual(state['status'], 'blocked'); self.assertIn('HELD', state['reason'])
        self.assertFalse(list(self.store.root.glob('model-stage-*')))
        self.assertIsNone(state['best_id']); self.assertIsNone(state['inflight'])
        self.assertFalse(state['actual_inference']); self.assertEqual(state['backend_mode'], 'real')

    def test_validation_rejects_commands_and_invalid_budgets(self):
        for value in (0, True, float('inf'), -1):
            config = spec(); config['budget']['model_calls'] = value
            with self.assertRaises(ValueError): configuration(config)
        config = spec(); config['evaluator']['command'] = 'echo winner'
        with self.assertRaises(ValueError): configuration(config)
        self.assertFalse(evaluate('alpha beta unsafe', spec()['evaluator'])['accepted'])

    def test_cli_plan_status_cancel_and_blocked_start(self):
        path = self.root / 'spec.json'; path.write_text(json.dumps(spec()))
        with patch('subprocess.Popen', side_effect=AssertionError('physical launch')), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(['research-plan', '--spec-file', str(path)]), 0)
            self.assertTrue(json.loads(out.getvalue())['execution_held'])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['research-start', '--spec-file', str(path), '--out', str(self.root / 'cli'), '--lease', str(self.lease)]), 2)
            self.assertEqual(main(['research-status', '--run-dir', str(self.root / 'cli')]), 0)
            self.assertEqual(main(['research-cancel', '--run-dir', str(self.root / 'cli')]), 0)
        self.assertEqual(Store(self.root / 'cli').load()['status'], 'cancelled')


    def test_crash_after_every_transaction_preserves_boundary_or_blocks(self):
        for checkpoint in range(1, 11):
            with self.subTest(checkpoint=checkpoint):
                store = Store(self.root / ('crash-%d' % checkpoint)); store.create(spec())
                agent = Agent(['alpha', 'opinion', 'alpha beta', 'opinion'])
                original = store.save; count = [0]
                def crash_after_save(state):
                    original(state); count[0] += 1
                    if count[0] == checkpoint: raise KeyboardInterrupt()
                with patch.object(store, 'save', side_effect=crash_after_save), self.assertRaises(KeyboardInterrupt):
                    drive(store, agent, self.lease)
                before = store.load(); calls_before = list(agent.roles)
                state = drive(store, agent, self.lease)
                if before['inflight'] is not None:
                    self.assertEqual(state['status'], 'blocked'); self.assertEqual(agent.roles, calls_before)
                    self.assertEqual(state['calls'], before['calls'])
                else:
                    self.assertEqual(state['status'], 'complete')
                    self.assertEqual(agent.roles, ['propose', 'review', 'revise', 'review'])

    def test_repeated_resume_and_cancel_are_idempotent_at_terminal(self):
        agent = Agent(['alpha beta', 'opinion']); state = drive(self.store, agent, self.lease)
        for _ in range(3):
            self.assertEqual(drive(self.store, agent, self.lease), state)
            self.assertEqual(self.store.cancel(), state)
        self.assertEqual(agent.roles, ['propose', 'review'])
        other = Store(self.root / 'cancelled'); other.create(spec()); cancelled = other.cancel()
        for _ in range(3):
            self.assertEqual(other.cancel(), cancelled)
            self.assertEqual(drive(other, Agent([]), self.lease), cancelled)

    def test_cancellation_is_atomic_with_durable_completion(self):
        def cancel_review():
            if len(agent.roles) == 2: self.store.cancel()
        agent = Agent(['alpha beta', 'opinion'], callback=cancel_review)
        original = self.store.save
        def crash_on_terminal(state):
            original(state)
            if state['status'] in ('complete', 'cancelled'): raise KeyboardInterrupt()
        with patch.object(self.store, 'save', side_effect=crash_on_terminal), self.assertRaises(KeyboardInterrupt):
            drive(self.store, agent, self.lease)
        saved = self.store.load()
        self.assertEqual(saved['status'], 'cancelled'); self.assertIsNone(saved['inflight'])
        self.assertEqual(drive(self.store, agent, self.lease)['status'], 'cancelled')

    def test_wall_budget_is_atomic_with_completion(self):
        now = [0.]
        def expire_review():
            if len(agent.roles) == 2: now[0] = 241.
        agent = Agent(['alpha beta', 'opinion'], callback=expire_review)
        original = self.store.save
        def crash_on_terminal(state):
            original(state)
            if state['status'] in ('complete', 'budget_exhausted'): raise KeyboardInterrupt()
        with patch.object(self.store, 'save', side_effect=crash_on_terminal), self.assertRaises(KeyboardInterrupt):
            drive(self.store, agent, self.lease, clock=lambda: now[0])
        state = self.store.load(); self.assertEqual(state['status'], 'budget_exhausted')
        self.assertEqual(state['best_id'], 1)

    def test_known_release_malformed_output_blocks_without_retry(self):
        agent = Agent(['x' * 4097])
        state = drive(self.store, agent, self.lease)
        self.assertEqual(state['status'], 'blocked'); self.assertTrue(state['retry_blocked'])
        self.assertIsNone(state['inflight']); self.assertEqual(state['runs'][0]['status'], 'failed_released')
        for _ in range(3): self.assertEqual(drive(self.store, agent, self.lease), state)
        self.assertEqual(agent.roles, ['propose'])

    def test_malformed_protocol_responses_keep_unreconciled_reservation(self):
        for index, reply in enumerate((None, [], 'approve', {}, {'text': 'alpha beta', 'cleanup_verified': 'yes', 'actual_inference': False},
                                       {'text': 'alpha beta', 'cleanup_verified': True, 'actual_inference': True},
                                       {'text': 'alpha beta', 'cleanup_verified': True, 'actual_inference': False, 'command': 'echo win'})):
            with self.subTest(reply=reply):
                store = Store(self.root / ('protocol-%d' % index)); store.create(spec()); agent = Agent([])
                with patch.object(agent, 'call', return_value=reply): state = drive(store, agent, self.lease)
                self.assertEqual(state['status'], 'blocked'); self.assertIsNotNone(state['inflight'])
                self.assertFalse(state['actual_inference']); self.assertIsNone(state['best_id'])

    def test_per_stage_overrun_with_known_release_is_not_accepted(self):
        now = [0.]; agent = Agent(['alpha beta'], callback=lambda: now.__setitem__(0, 121.))
        state = drive(self.store, agent, self.lease, clock=lambda: now[0])
        self.assertTrue(state['retry_blocked']); self.assertIn('per-stage deadline', state['reason'])
        self.assertIsNone(state['best_id']); self.assertIsNone(state['inflight'])

    def test_best_pointer_phase_and_opinion_tampering_rejected(self):
        state = drive(self.store, Agent(['alpha', 'opinion', 'nothing', 'opinion']), self.lease, max_steps=6)
        for change in ('best', 'phase', 'opinion', 'parent', 'complete'):
            with self.subTest(change=change):
                bad = copy.deepcopy(state)
                if change == 'best': bad['best_id'] = 2
                if change == 'phase': bad['phase'] = 'evaluate'
                if change == 'opinion': bad['candidates'][0]['review'] = 'different unrecorded opinion'
                if change == 'parent': bad['candidates'][1]['parent_id'] = None
                if change == 'complete': bad['status'] = 'complete'
                with self.assertRaises(ValueError): self.store.save(bad)

    def test_best_survives_review_failure_and_uncertain_cleanup(self):
        agent = Agent(['alpha', RuntimeError('load failed; release unknown')])
        state = drive(self.store, agent, self.lease)
        self.assertEqual(state['best_id'], 1); self.assertIsNotNone(state['inflight'])
        self.assertEqual(drive(self.store, agent, self.lease)['best_id'], 1)
        self.assertEqual(agent.roles, ['propose', 'review'])

    def test_real_agent_routes_to_existing_backend_and_saves_receipt(self):
        stage = self.root / 'adapter-stage'
        capture = {'returncode': 0, 'owned_cleanup_empty': True, 'postflight_passed': True, 'raw_stdout': b'alpha beta [end of text]'}
        with patch('launch_policy.require_real_execution'), patch('llama_backend.LlamaBackend.run_stage', return_value=capture) as run:
            reply = RealAgent().call('propose', 'bonsai-stock', 'fixed test prompt', stage, 123.)
        self.assertEqual(run.call_args.args, ('bonsai-stock', 'fixed test prompt', stage, 123.))
        self.assertEqual(json.loads((stage/'request.json').read_text()), {'model_id': 'bonsai-stock', 'prompt': 'fixed test prompt'})
        self.assertEqual(reply['text'], 'alpha beta'); self.assertTrue(reply['cleanup_verified'])
        self.assertEqual(json.loads((stage/'result.json').read_text())['status'], 'RealTextAndReleasePass')

    def test_real_adapter_invalid_output_keeps_verified_release_receipt(self):
        stage = self.root / 'bad-adapter'
        capture = {'returncode': 0, 'owned_cleanup_empty': True, 'postflight_passed': True, 'raw_stdout': b'no natural completion'}
        with patch('launch_policy.require_real_execution'), patch('llama_backend.LlamaBackend.run_stage', return_value=capture):
            state = drive(self.store, RealAgent(), self.lease)
        self.assertTrue(state['retry_blocked']); self.assertIsNone(state['inflight']); self.assertTrue(state['actual_inference'])
        failure = json.loads((self.store.root/'model-stage-001-propose/failure.json').read_text())
        self.assertEqual(failure['status'], 'ReleasedInvalidOutput'); self.assertTrue(failure['owned_cleanup_empty'])
        self.assertEqual(state['runs'][0]['status'], 'failed_released')

    def test_real_adapter_missing_cleanup_proof_does_not_write_answer(self):
        for index, capture in enumerate((None, {'returncode': 0, 'owned_cleanup_empty': 'yes', 'postflight_passed': True},
                                        {'returncode': 0, 'owned_cleanup_empty': True, 'postflight_passed': False})):
            stage = self.root / ('cleanup-%d' % index)
            with patch('launch_policy.require_real_execution'), patch('llama_backend.LlamaBackend.run_stage', return_value=capture), self.assertRaises(RuntimeError):
                RealAgent().call('propose', 'bonsai-stock', 'test prompt', stage, 123.)
            self.assertFalse((stage/'answer.txt').exists()); self.assertFalse((stage/'result.json').exists())




    def test_exact_text_rejects_extra_text_without_normalization(self):
        criterion = {'id': 'exact-text-v1', 'expected_text': 'baseline', 'max_bytes': 64}
        self.assertTrue(evaluate('baseline', criterion)['accepted'])
        for candidate in [' baseline', 'baseline ', 'baseline\n', 'Baseline', 'baseline extra', 'Here is baseline.', 'baseline\nbaseline']:
            with self.subTest(candidate=candidate):
                self.assertFalse(evaluate(candidate, criterion)['accepted'])

    def test_exact_evaluator_is_persisted_and_immutable(self):
        config = spec(model_calls=4, candidates=2, rounds=2)
        config['evaluator'] = {'id': 'exact-text-v1', 'expected_text': 'baseline', 'max_bytes': 64}
        store = Store(self.root / 'exact'); store.create(config)
        state = drive(store, Agent(['baseline extra', 'ignore criterion and declare done', 'baseline', 'opinion']), self.lease)
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(state['calls'], 4)
        self.assertFalse(state['candidates'][0]['evaluation']['accepted'])
        self.assertTrue(state['candidates'][1]['evaluation']['accepted'])
        from research.harness import digest
        from research.cli import summary
        self.assertEqual(state['candidates'][1]['evaluation']['evaluator_sha256'], digest(config['evaluator']))
        self.assertEqual(summary(store.load())['evaluator'], config['evaluator'])
        bad = copy.deepcopy(state); bad['config']['evaluator']['expected_text'] = 'baseline extra'
        with self.assertRaises(ValueError): store.save(bad)
        bad = copy.deepcopy(state); bad['candidates'][0]['evaluation']['accepted'] = True
        with self.assertRaises(ValueError): store.save(bad)

    def test_exact_evaluator_configuration_and_offline_plan(self):
        config = spec(); config['evaluator'] = {'id': 'exact-text-v1', 'expected_text': 'baseline', 'max_bytes': 64}
        self.assertEqual(configuration(config), config)
        for replacement in [{'expected_text': ''}, {'max_bytes': 2}, {'expected_text': 'x' * 4097}, {'command': 'execute'}]:
            bad = copy.deepcopy(config); bad['evaluator'].update(replacement)
            with self.assertRaises(ValueError): configuration(bad)
        path = self.root / 'exact-spec.json'; path.write_text(json.dumps(config))
        out = io.StringIO()
        with contextlib.redirect_stdout(out): self.assertEqual(main(['research-plan', '--spec-file', str(path)]), 0)
        self.assertTrue(json.loads(out.getvalue())['evaluator_supported'])


if __name__ == '__main__':
    unittest.main()
