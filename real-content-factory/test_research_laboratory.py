"""Model-free research-loop contracts; no model, tokenizer or candidate executes."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import demo_prompts as prompts
from research import laboratory as lab
from research.experiments import CORRECTNESS_EVALUATOR, ExperimentRunner
from research.harness import RealAgent, digest
from research.retrieval import evidence
from test_demo_laboratory import FixtureAgent, FixtureRunner, SOURCES


class ResearchFixtureAgent(FixtureAgent):
    """Reuse the demo fixture plumbing with additional inert unique sources."""
    def __init__(self, *, duplicate=False, callback=None):
        super().__init__(duplicate=duplicate)
        self.callback = callback

    def call(self, role, model, prompt, stage, deadline):
        self.calls.append(role)
        self.prompts.append(prompt)
        self.deadlines.append(deadline)
        if role in prompts.CODE_ROLES:
            index = 0 if self.duplicate else self.source_index
            value = SOURCES[index] if index < len(SOURCES) else SOURCES[0] + '# fixture source %d\n' % (index + 1)
            self.source_index += 1
        elif role == 'ranking':
            payload = json.loads(prompt.split('\n')[-1])
            value = json.dumps({'winner': payload['candidates'][0][0],
                                'rationale': 'fixture opinion %d' % len(self.calls)})
        else:
            value = 'fixture opinion %d' % len(self.calls)
        if self.callback is not None:
            self.callback(role)
        return {'text': value, 'cleanup_verified': True, 'actual_inference': False,
                'fixture_label': 'MODEL_FREE_CONTRACT_FIXTURE'}


class ResearchLaboratoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.lease = self.root / 'lease'
        self.no_launch = patch('subprocess.Popen', side_effect=AssertionError('no model/candidate/tokenizer launch'))
        self.no_launch.start()
        self.addCleanup(self.no_launch.stop)

    def store(self, name='run', **options):
        store = lab.LaboratoryStore(self.root / name)
        store.create(lab.research_configuration(**options))
        return store

    def revision_store(self, name='run', **budget):
        return self.store(name, evolutions=3,
                          budget={'model_calls': 45, 'experiment_calls': 5, **budget})

    def assert_fixture(self, state):
        self.assertEqual(state['mode'], 'fixture')
        self.assertFalse(state['actual_inference'])
        self.assertFalse(state['actual_experiments'])
        self.assertIsNone(lab.summary(state)['objective_met'])
        self.assertFalse(lab.summary(state)['demonstration']['completed_real_demonstration'])

    def test_defaults_use_persisted_goal_fixed_profile_and_immutable_evaluator(self):
        config = lab.research_configuration()
        persisted = json.loads(Path(lab.__file__).with_name('RESEARCH-GOAL.json').read_text())
        self.assertEqual(config['goal'], persisted['goal'])
        self.assertEqual(config['execution_profile'], 'bonsai-research-v1')
        self.assertEqual(config['model_id'], 'bonsai-stock')
        self.assertEqual((config['initial_candidates'], config['evolutions']), (2, 1))
        self.assertEqual(config['budget'], {'model_calls': 18, 'experiment_calls': 3,
                                          'wall_seconds': 2400, 'stage_seconds': 120,
                                          'experiment_stage_seconds': 60})
        self.assertEqual(config['evaluator'], CORRECTNESS_EVALUATOR)
        self.assertEqual(lab.configuration(config), config)
        config['evaluator']['id'] = 'unreviewed-evaluator'
        with self.assertRaises(ValueError):
            lab.configuration(config)
        self.assertEqual(lab.research_configuration()['evaluator'], CORRECTNESS_EVALUATOR)

    def test_custom_goal_strategy_and_operator_sources_are_copied(self):
        sources = [evidence('operator-note', 'Retain empty-interval and touching-endpoint checks.', 'operator', None)]
        options = {'generation': 'direct', 'evolution': 'simplify', 'scheduling': 'adaptive',
                   'match_order': 'paper-priority', 'proximity': 'character-trigram',
                   'elo_k': 16, 'reverse_pair_order': True, 'debate_turns': 1,
                   'matches_per_pair': 2, 'evolution_trigger': 'stable', 'stability_threshold': 0}
        budget = {'model_calls': 64, 'experiment_calls': 6, 'wall_seconds': 3600}
        config = lab.research_configuration('Explore interval edge cases.', initial_candidates=3,
                                            evolutions=3, budget=budget, options=options, sources=sources)
        self.assertEqual(config['goal'], 'Explore interval edge cases.')
        self.assertEqual(config['options'], options)
        self.assertEqual(config['sources'], sources)
        self.assertEqual(lab.configuration(config), config)
        sources[0]['excerpt'] = 'mutated input'
        options['elo_k'] = 32
        budget['model_calls'] = 1
        self.assertEqual(config['options']['elo_k'], 16)
        self.assertEqual(config['budget']['model_calls'], 64)
        self.assertNotEqual(config['sources'][0]['excerpt'], sources[0]['excerpt'])

    def test_population_and_budget_boundaries_reject_boolean_coercions(self):
        for seeds, revisions in ((2, 0), (8, 0), (2, 16), (8, 16)):
            with self.subTest(seeds=seeds, revisions=revisions):
                config = lab.research_configuration(initial_candidates=seeds, evolutions=revisions)
                self.assertEqual((config['initial_candidates'], config['evolutions']), (seeds, revisions))
        for key, limits in (('model_calls', (1, 4096)), ('experiment_calls', (1, 24)),
                            ('wall_seconds', (1, 604800))):
            for value in limits:
                with self.subTest(key=key, value=value):
                    self.assertEqual(lab.research_configuration(budget={key: value})['budget'][key], value)
            for value in (0, limits[1] + 1, True, 1.5):
                with self.subTest(key=key, invalid=value), self.assertRaises(ValueError):
                    lab.research_configuration(budget={key: value})
        for key, values in (('initial_candidates', (1, 9, True, 2.0)),
                            ('evolutions', (-1, 17, False, 1.0))):
            for value in values:
                with self.subTest(key=key, invalid=value), self.assertRaises(ValueError):
                    lab.research_configuration(**{key: value})

    def test_fixed_stage_limits_and_configuration_never_grant_real_authority(self):
        for key, fixed in (('stage_seconds', 120), ('experiment_stage_seconds', 60)):
            self.assertEqual(lab.research_configuration(budget={key: fixed})['budget'][key], fixed)
            for value in (fixed - 1, fixed + 1, True):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    lab.research_configuration(budget={key: value})
        for changes in ({'execution_profile': 'unreviewed'}, {'model_id': 'unreviewed'},
                        {'execution_authorized': True}):
            config = lab.research_configuration()
            config.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                lab.configuration(config)
        store = self.revision_store()
        with self.assertRaisesRegex(ValueError, 'reviewed entry'):
            lab.drive(store, RealAgent(), ExperimentRunner(), self.lease)
        self.assertEqual(store.load()['model_calls'], 0)
        self.assertEqual(store.load()['experiment_calls'], 0)

    def test_unreviewed_semantic_strategy_and_tampered_operator_provenance_refused(self):
        with self.assertRaises(ValueError):
            lab.research_configuration(options={'proximity': 'model-semantic'})
        with self.assertRaises(ValueError):
            lab.research_configuration(options={'scheduling': 'unbounded'})
        with self.assertRaises(ValueError):
            lab.research_configuration(options={'unreviewed': True})
        source = evidence('note', 'Operator context.', 'operator', None)
        for field, value in (('sha256', '0' * 64), ('origin', 'retrieved'), ('url', 'https://example.invalid/')):
            changed = copy.deepcopy(source)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                lab.research_configuration(sources=[changed])

    def test_three_revisions_complete_real_sqlite_controller_with_all_pair_coverage(self):
        store = self.revision_store()
        agent = ResearchFixtureAgent()
        runner = FixtureRunner()
        state = lab.drive(store, agent, runner, self.lease)
        self.assertEqual(state['status'], 'finished', state['reason'])
        self.assertEqual((len(agent.calls), len(runner.calls)), (45, 5))
        self.assertEqual((state['model_calls'], state['experiment_calls']), (45, 5))
        self.assertEqual(len({c['source'] for c in state['candidates']}), 5)
        self.assertEqual(len(state['matches']), 10)
        self.assertEqual({tuple(sorted(m['pair'])) for m in state['matches']},
                         set(itertools.combinations(range(1, 6), 2)))
        expected = {'generation': 2, 'evolution': 3, 'reflection': 5, 'debate': 20,
                    'ranking': 10, 'meta-review': 4, 'overview': 1, 'experiment': 5, 'proximity': 4}
        coverage = lab.summary(state)['demonstration']
        self.assertEqual(coverage['completed_by_kind'], expected)
        self.assertEqual(coverage['expected_full_loop_by_kind'], expected)
        self.assertEqual(coverage['expected_full_loop_model_calls'], 45)
        self.assertEqual((coverage['configured_revision_limit'], coverage['completed_revisions']), (3, 3))
        self.assertTrue(coverage['full_comparative_coverage'])
        self.assertEqual([t['id'] for t in state['tasks'] if t['kind'] == 'evolution'],
                         ['evolve-1', 'evolve-2', 'evolve-3'])
        self.assertEqual([t['id'] for t in state['tasks'] if t['kind'] == 'meta-review'],
                         ['meta-initial', 'meta-final', 'meta-revision-2', 'meta-revision-3'])
        self.assertEqual([t['id'] for t in state['tasks'] if t['kind'] == 'proximity'],
                         ['proximity-initial', 'proximity-final', 'proximity-revision-2', 'proximity-revision-3'])
        for child in state['candidates'][2:]:
            self.assertEqual(len(child['parents']), 2)
            self.assertTrue(all(parent < child['id'] for parent in child['parents']))
        self.assertEqual(store.load(), state)
        self.assert_fixture(state)

    def test_resume_reopens_sqlite_and_never_repeats_completed_waves(self):
        store = self.revision_store()
        agent = ResearchFixtureAgent()
        runner = FixtureRunner()
        for _ in range(100):
            store = lab.LaboratoryStore(store.root)
            state = lab.drive(store, agent, runner, self.lease, max_steps=1)
            if state['status'] in ('finished', 'blocked', 'budget_exhausted'):
                break
        self.assertEqual(state['status'], 'finished', state['reason'])
        self.assertEqual((len(agent.calls), len(runner.calls)), (45, 5))
        self.assertEqual(lab.drive(lab.LaboratoryStore(store.root), agent, runner, self.lease), state)
        self.assertEqual((len(agent.calls), len(runner.calls)), (45, 5))
        first = state['tasks'][0]['model_io']
        snapshot = store.prompt_input(first['snapshot_sha256'])
        self.assertEqual(snapshot['snapshot']['model_calls'], 0)
        self.assertEqual(digest(snapshot['snapshot']), first['snapshot_sha256'])

    def test_cancellation_during_later_revision_retains_returned_completion(self):
        store = self.revision_store()
        def cancel_second_revision(role):
            if role == 'evolution' and agent.calls.count('evolution') == 2:
                store.cancel()
        agent = ResearchFixtureAgent(callback=cancel_second_revision)
        runner = FixtureRunner()
        state = lab.drive(store, agent, runner, self.lease)
        self.assertEqual(state['status'], 'cancelled', state['reason'])
        self.assertTrue(state['cancel_requested'])
        self.assertIsNone(state['inflight'])
        revision = next(t for t in state['tasks'] if t['id'] == 'evolve-2')
        self.assertEqual(revision['status'], 'done')
        self.assertIsNotNone(revision['model_io']['completion'])
        self.assertEqual(len(state['candidates']), 4)
        self.assertNotIn('evolve-3', {t['id'] for t in state['tasks']})
        counts = (len(agent.calls), len(runner.calls))
        self.assertEqual(lab.drive(store, agent, runner, self.lease), state)
        self.assertEqual((len(agent.calls), len(runner.calls)), counts)
        self.assertFalse(lab.summary(state)['demonstration']['full_comparative_coverage'])
        self.assert_fixture(state)

    def test_pause_at_committed_boundary_retains_snapshot_then_resumes_without_repetition(self):
        store = self.store()
        agent = ResearchFixtureAgent()
        runner = FixtureRunner()
        state = lab.drive(store, agent, runner, self.lease,
                          should_pause=lambda: len(agent.calls) >= 1)
        self.assertEqual(state['status'], 'ready', state['reason'])
        self.assertEqual(state['reason'], 'paused at committed task boundary')
        self.assertIsNone(state['inflight'])
        self.assertEqual((len(agent.calls), len(runner.calls)), (1, 0))
        self.assertEqual(len(state['candidates']), 1)
        self.assertEqual(state['tasks'][0]['status'], 'done')
        first = copy.deepcopy(state['tasks'][0])
        saved = store.prompt_input(first['model_io']['snapshot_sha256'])
        state = lab.drive(lab.LaboratoryStore(store.root), agent, runner, self.lease)
        self.assertEqual(state['status'], 'finished', state['reason'])
        self.assertEqual((len(agent.calls), len(runner.calls)), (18, 3))
        self.assertEqual(state['tasks'][0], first)
        self.assertEqual(store.prompt_input(first['model_io']['snapshot_sha256']), saved)
        self.assert_fixture(state)
        untouched = self.store('pause-before-start')
        unused_agent = ResearchFixtureAgent()
        unused_runner = FixtureRunner()
        state = lab.drive(untouched, unused_agent, unused_runner, self.lease, should_pause=lambda: True)
        self.assertEqual(state['status'], 'ready')
        self.assertIsNone(state['inflight'])
        self.assertEqual((unused_agent.calls, unused_runner.calls), ([], []))
        self.assertEqual(state['candidates'], [])

    def test_low_call_budgets_stop_without_reserving_or_retrying_an_extra_task(self):
        for budget, expected in (({'model_calls': 1}, (1, 1)),
                                 ({'experiment_calls': 1}, (3, 1))):
            with self.subTest(budget=budget):
                store = self.revision_store(next(iter(budget)), **budget)
                agent = ResearchFixtureAgent()
                runner = FixtureRunner()
                state = lab.drive(store, agent, runner, self.lease)
                self.assertEqual(state['status'], 'budget_exhausted', state['reason'])
                self.assertIsNone(state['inflight'])
                self.assertEqual((len(agent.calls), len(runner.calls)), expected)
                self.assertEqual(lab.drive(store, agent, runner, self.lease), state)
                self.assertEqual((len(agent.calls), len(runner.calls)), expected)
                self.assertFalse(lab.summary(state)['demonstration']['full_comparative_coverage'])

    def test_fixed_model_and_cpu_deadlines_are_clamped_by_remaining_wall_budget(self):
        store = self.revision_store()
        agent = ResearchFixtureAgent()
        runner = FixtureRunner()
        state = lab.drive(store, agent, runner, self.lease, max_steps=2, clock=lambda: 100.)
        self.assertEqual(agent.deadlines, [220.])
        self.assertEqual(runner.deadlines, [160.])
        state['elapsed_seconds'] = state['config']['budget']['wall_seconds'] - 10
        store.save(state)
        lab.drive(store, agent, runner, self.lease, max_steps=1, clock=lambda: 100.)
        self.assertEqual(agent.deadlines[-1], 110.)

    def test_exhausted_wall_budget_stops_before_any_fixture_call(self):
        store = self.revision_store(wall_seconds=1)
        state = store.load()
        state['elapsed_seconds'] = 1
        store.save(state)
        agent = ResearchFixtureAgent()
        runner = FixtureRunner()
        state = lab.drive(store, agent, runner, self.lease, clock=lambda: 100.)
        self.assertEqual(state['status'], 'budget_exhausted')
        self.assertEqual((agent.calls, runner.calls), ([], []))
        self.assertEqual((state['model_calls'], state['experiment_calls']), (0, 0))

    def test_dynamic_coverage_for_zero_revisions_and_changed_match_options(self):
        variants = (({'initial_candidates': 3, 'evolutions': 0},
                     {'generation': 3, 'evolution': 0, 'reflection': 3, 'debate': 6,
                      'ranking': 3, 'meta-review': 1, 'overview': 1, 'experiment': 3, 'proximity': 1}),
                    ({'evolutions': 0, 'options': {'debate_turns': 1, 'matches_per_pair': 2}},
                     {'generation': 2, 'evolution': 0, 'reflection': 2, 'debate': 0,
                      'ranking': 2, 'meta-review': 1, 'overview': 1, 'experiment': 2, 'proximity': 1}))
        for index, (config, expected) in enumerate(variants):
            with self.subTest(config=config):
                store = self.store('variant-%d' % index, **config)
                state = lab.drive(store, ResearchFixtureAgent(), FixtureRunner(), self.lease)
                self.assertEqual(state['status'], 'finished', state['reason'])
                coverage = lab.summary(state)['demonstration']
                self.assertEqual(coverage['completed_by_kind'], expected)
                self.assertEqual(coverage['expected_full_loop_by_kind'], expected)
                self.assertTrue(coverage['full_comparative_coverage'])
                self.assert_fixture(state)

    def test_duplicate_revisions_finish_without_false_comparative_coverage(self):
        store = self.revision_store()
        agent = ResearchFixtureAgent(duplicate=True)
        runner = FixtureRunner()
        state = lab.drive(store, agent, runner, self.lease)
        self.assertEqual(state['status'], 'finished', state['reason'])
        self.assertEqual((len(agent.calls), len(runner.calls)), (15, 1))
        self.assertEqual(state['matches'], [])
        self.assertEqual([c['duplicate_of'] for c in state['candidates']], [None, 1, 1, 1, 1])
        coverage = lab.summary(state)['demonstration']
        self.assertEqual(coverage['distinct_candidate_count'], 1)
        self.assertEqual(coverage['duplicate_candidate_count'], 4)
        self.assertFalse(coverage['full_comparative_coverage'])
        self.assert_fixture(state)

    def assert_fourth_seed_prior_projection(self):
        store = self.store('four-seeds', initial_candidates=4, evolutions=0,
                           budget={'model_calls': 28, 'experiment_calls': 4})
        agent = ResearchFixtureAgent()
        state = lab.drive(store, agent, FixtureRunner(), self.lease, max_steps=10)
        self.assertEqual(len(state['candidates']), 4, state['reason'])
        task = next(t for t in state['tasks'] if t['id'] == 'seed-4')
        self.assertEqual(task['status'], 'done')
        saved = store.prompt_input(task['model_io']['snapshot_sha256'])
        payload = json.loads(saved['prompt'].split('\n')[-1])
        prior = saved['snapshot']['candidates']
        self.assertEqual(payload['prior'], [[c['id'], c['source']] for c in prior[-2:]])
        self.assertEqual(payload['prior_source_refs'], [[c['id'], c['sha256']] for c in prior[:-2]])
        self.assertEqual(payload['prior_projection'],
                         'last-two-full-sources; earlier SHA256 refs resolve in snapshot')
        self.assertEqual(payload['snapshot_sha256'], digest(saved['snapshot']))
        self.assertEqual(payload['seed'][0], 'seed-4')
        self.assertEqual(len({json.loads(prompt.split('\n')[-1])['seed'][1]
                              for role, prompt in zip(agent.calls, agent.prompts) if role == 'generation'}), 4)

    def test_large_population_prompt_snapshots_preserve_full_evidence_without_clipping(self):
        self.assert_fourth_seed_prior_projection()
        store = self.revision_store()
        agent = ResearchFixtureAgent()
        state = lab.drive(store, agent, FixtureRunner(), self.lease)
        self.assertEqual(state['status'], 'finished', state['reason'])
        refs = lab.summary(state)['demonstration']['prompt_snapshot_refs']
        self.assertEqual(len(refs), 45)
        larger = []
        for task in state['tasks']:
            if 'model_io' not in task:
                continue
            model_io = task['model_io']
            saved = store.prompt_input(model_io['snapshot_sha256'])
            payload = json.loads(saved['prompt'].split('\n')[-1])
            self.assertEqual(payload['snapshot_sha256'], digest(saved['snapshot']))
            self.assertEqual(saved['prompt_sha256'], model_io['prompt_sha256'])
            self.assertEqual(payload['mode'], 'fixture')
            self.assertIn(saved['prompt'], agent.prompts)
            self.assertLessEqual(len(saved['prompt'].encode()), prompts.raw_prompt_limit(task['kind']))
            self.assertFalse(model_io['completion']['actual_inference'])
            if len(saved['snapshot']['candidates']) > 3:
                larger.append((task, saved, payload))
                aggregate = task['kind'] in ('meta-review', 'overview')
                durable_candidates = saved['snapshot']['candidates']
                for row in payload['candidates']:
                    candidate = durable_candidates[row[0] - 1]
                    self.assertEqual(row[1], candidate['sha256'] if aggregate else candidate['source'])
                    self.assertEqual(row[3], candidate['reflection'])
                durable_tasks = {t['id']: t for t in saved['snapshot']['tasks']}
                for task_id, pair, result in payload.get('debates', []):
                    self.assertEqual(pair, durable_tasks[task_id]['targets'])
                    self.assertEqual(result, durable_tasks[task_id]['result'])
                for pair, winner, rationale in payload.get('decisions', []):
                    self.assertIn({'pair': pair, 'winner': winner, 'rationale': rationale},
                                  [{key: match[key] for key in ('pair', 'winner', 'rationale')}
                                   for match in saved['snapshot']['matches']])
                if aggregate:
                    projection = payload['projection']
                    selected = [row[0] for row in payload['candidates']]
                    omitted = [identity for identity in task['targets'] if identity not in selected]
                    self.assertEqual(projection['policy'], 'latest-with-parents-v1')
                    self.assertEqual(projection['candidate_count'], len(task['targets']))
                    self.assertEqual(projection['selected_ids'], selected)
                    self.assertEqual(projection['omitted_ids'], omitted)
                    self.assertEqual(len(selected), 3)
                    self.assertIn(max(task['targets']), selected)
                    self.assertTrue(set(durable_candidates[-1]['parents']).issubset(selected))
                    debates = [t for t in durable_tasks.values() if t['kind'] == 'debate' and t['status'] == 'done']
                    self.assertEqual(projection['omitted_debate_count'], len(debates) - len(payload.get('debates', [])))
                    self.assertEqual(projection['omitted_decision_count'],
                                     len(saved['snapshot']['matches']) - len(payload['decisions']))
        self.assertTrue(larger)
        overview, saved, payload = next(item for item in larger if item[0]['kind'] == 'overview')
        self.assertEqual(len(saved['snapshot']['candidates']), 5)
        self.assertEqual(len(saved['snapshot']['matches']), 10)
        self.assertEqual(sum(t['kind'] == 'debate' and t['status'] == 'done'
                             for t in saved['snapshot']['tasks']), 20)
        original = copy.deepcopy(saved['snapshot'])
        pending = next(t for t in original['tasks'] if t['id'] == overview['id'])
        with patch.object(prompts, 'raw_prompt_limit', return_value=1024):
            with self.assertRaisesRegex(ValueError, 'no clipping'):
                prompts.make_prompt(original, pending)
        self.assertEqual(original, saved['snapshot'])


if __name__ == '__main__':
    unittest.main()
