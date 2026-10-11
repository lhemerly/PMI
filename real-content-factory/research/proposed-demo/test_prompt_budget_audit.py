"""No models, tokenizers, candidate execution or production imports."""
import unittest
import copy
import json
from unittest.mock import patch
from prompt_budget_audit import simulate,SOURCES
from next_demo_prompt import make_prompt,validate_response,raw_prompt_limit,serialized_content_bytes


class ProposalFixtureTests(unittest.TestCase):
    def test_unique_candidates_have_exact_eighteen_call_shadow_schedule(self):
        report=simulate()
        self.assertEqual(report['shadow_role_counts'],{'generation':2,'experiment':3,'reflection':3,
                         'proximity':2,'debate':6,'ranking':3,'meta-review':2,'evolution':1,'overview':1})
        self.assertEqual(report['shadow_model_calls'],18)
        self.assertEqual(report['actual_model_calls'],0)
        self.assertEqual(report['actual_candidate_executions'],0)
        self.assertFalse(report['token_fit_run'])

    def test_short_realistic_fixture_falsifies_full_4096_byte_fit(self):
        report=simulate()
        self.assertIsNotNone(report['first_real_controller_byte_block'])
        self.assertGreater(report['max_prompt_bytes'],4096)
        self.assertTrue(any(not row['within_4096'] for row in report['prompts']))

    def test_duplicate_fixture_does_not_claim_a_completed_competing_tournament(self):
        report=simulate(duplicate=True)
        self.assertEqual(report['shadow_model_calls'],9)
        self.assertEqual(report['shadow_experiment_calls'],1)
        self.assertNotIn('ranking',report['shadow_role_counts'])
        self.assertNotIn('debate',report['shadow_role_counts'])

    def test_compact_full_eighteen_call_fixture_at_all_opinion_caps(self):
        snapshots=[]
        def measured(state,task):
            before=copy.deepcopy(state);rendered=make_prompt(state,task)
            self.assertEqual(state,before)
            self.assertLessEqual(len(rendered.encode()),raw_prompt_limit(task['kind']))
            snapshots.append((task['kind'],json.loads(rendered.split('\n')[-1])))
            return rendered
        report=simulate(renderer=measured,response_validator=validate_response,
                        review_bytes=192,debate_bytes=192,meta_bytes=192,rationale_bytes=96)
        self.assertEqual(report['shadow_model_calls'],18)
        self.assertIsNone(report['first_real_controller_byte_block'])
        self.assertFalse(report['token_fit_run'])
        for kind,payload in snapshots:
            self.assertEqual(payload['mode'],'fixture')
            if kind in ('meta-review','overview'):
                self.assertTrue(all(len(row[1])==64 for row in payload['candidates']))
        final=next(payload for kind,payload in reversed(snapshots) if kind=='meta-review')
        self.assertEqual(len(final['debates']),6);self.assertEqual(len(final['decisions']),3)
        self.assertEqual(len(final['latest_meta']),2)

    def test_maximum_escaped_code_caps_fit_complete_fixture(self):
        padded=[]
        for source in SOURCES:
            source=source+'#'
            source+='x'*(640-serialized_content_bytes(source))
            self.assertEqual(serialized_content_bytes(source),640)
            validate_response('generation',source);padded.append(source)
        seen=[]
        def measured(state,task):
            seen.append(task['id']);before=copy.deepcopy(state)
            try:return make_prompt(state,task)
            finally:self.assertEqual(state,before)
        report=simulate(renderer=measured,response_validator=validate_response,candidate_sources=tuple(padded),
                        review_bytes=192,debate_bytes=192,meta_bytes=192,rationale_bytes=96)
        self.assertEqual(report['shadow_model_calls'],18)
        for row in report['prompts']:self.assertLessEqual(row['utf8_bytes'],raw_prompt_limit(row['role']))

    def test_insufficient_prompt_envelope_rejects_without_clipping_or_mutation(self):
        def constrained(state,task):
            before=copy.deepcopy(state)
            try:
                with patch('next_demo_prompt.raw_prompt_limit',return_value=1024):return make_prompt(state,task)
            finally:self.assertEqual(state,before)
        with self.assertRaisesRegex(ValueError,'prompt envelope exceeded; no clipping'):
            simulate(renderer=constrained,response_validator=validate_response,
                     review_bytes=192,debate_bytes=192,meta_bytes=192,rationale_bytes=96)


if __name__=='__main__':unittest.main()
