import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.harness import digest, RealAgent
from research.experiments import EVALUATOR, ExperimentRequest, ExperimentRunner, validate_receipt
from research.laboratory import LaboratoryStore, OPTIONS, configuration, enqueue, drive, summary, prompt, ratings_stable


def spec(**budget):
    return {'goal':'Optimize merge_intervals for integer half-open intervals without changing correctness or mutating inputs.',
            'model_id':'bonsai-stock','initial_candidates':2,'evolutions':1,'evaluator':copy.deepcopy(EVALUATOR),
            'options':copy.deepcopy(OPTIONS),'sources':[],
            'budget':{'model_calls':24,'experiment_calls':12,'wall_seconds':1200,'stage_seconds':120,**budget}}


class Agent:
    actual_inference=False
    def __init__(self, duplicate=False, fail=None, callback=None): self.calls=[]; self.prompts=[]; self.duplicate=duplicate; self.fail=fail; self.callback=callback
    def call(self,role,model,prompt,stage,deadline):
        self.calls.append(role); self.prompts.append(prompt)
        if self.callback: self.callback()
        if self.fail: raise self.fail
        if role in ('generation','evolution'):
            value='def merge_intervals(intervals): return [] # candidate '+str(1 if self.duplicate else len(self.calls))
        elif role=='ranking':
            data=json.loads(prompt.split('\n')[-1]); value=json.dumps({'winner':data['candidates'][-1]['id'],'rationale':'model preference only'})
        else: value='Evidence suggests an improvement; missing external grounding; this is a model opinion.'
        return {'text':value,'cleanup_verified':True,'actual_inference':False}


def receipt(request, *, correct=True, ns=100., actual=False):
    return {'schema':'PMI_EXPERIMENT_RECEIPT_V1','request':request.payload(),'actual_execution':actual,
            'confinement_verified':True,'admission_verified':True,'cleanup_verified':True,'exitcode':0,
            'stdout':'mock output','stderr':'','artifacts':[],
            'measurements':{'tests_passed':100 if correct else 99,'tests_total':100,'baseline_ns':200.,'candidate_ns':ns,'baseline_sha256':'a'*64}}


class Runner:
    actual_execution=False
    def __init__(self, bad=False, callback=None): self.calls=[]; self.bad=bad; self.callback=callback
    def run(self,request,source,stage,deadline):
        self.calls.append(request)
        if self.callback: self.callback()
        value=receipt(request,correct=request.candidate_id!=3,ns=100./request.candidate_id)
        if self.bad: value['cleanup_verified']=False
        return value


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.lease=self.root/'lease';self.store=LaboratoryStore(self.root/'run');self.store.create(spec())

    def test_all_roles_population_lineage_and_measured_incumbent(self):
        agent=Agent(); runner=Runner(); state=drive(self.store,agent,runner,self.lease)
        self.assertEqual(state['status'],'finished');self.assertEqual(len(state['candidates']),3)
        self.assertEqual(len(state['matches']),3);self.assertEqual(len(state['meta_reviews']),2);self.assertEqual(len(state['proximity']),2)
        self.assertEqual(set(agent.calls),{'generation','reflection','ranking','evolution','meta-review','overview','debate'})
        self.assertEqual(state['candidates'][2]['parents'],[2,1]);self.assertEqual(state['incumbent_id'],2)
        self.assertFalse(state['candidates'][2]['evidence']['evaluation']['correct'])
        self.assertGreater(state['candidates'][2]['model_elo'],state['candidates'][1]['model_elo'])
        self.assertFalse(state['actual_inference']);self.assertFalse(state['actual_experiments'])
        self.assertIn('tests_passed',agent.prompts[agent.calls.index('reflection')]);self.assertIn('model_matches',agent.prompts[-1])
        self.assertIsNone(summary(state)['objective_met']);self.assertTrue(summary(state)['contract_objective_met']);self.assertEqual(summary(state)['mode'],'fixture')

    def test_exact_duplicates_skip_experiments_and_matches(self):
        runner=Runner();state=drive(self.store,Agent(duplicate=True),runner,self.lease)
        self.assertEqual(state['status'],'finished');self.assertEqual(len(runner.calls),1)
        self.assertEqual(state['candidates'][1]['duplicate_of'],1);self.assertEqual(state['candidates'][2]['duplicate_of'],1)
        self.assertEqual(state['matches'],[])

    def test_duplicate_task_ids_are_idempotent_and_collisions_refused(self):
        state=self.store.load();self.assertFalse(enqueue(state,'seed-1','generation'))
        with self.assertRaises(ValueError):enqueue(state,'seed-1','overview')
        self.store.save(state);self.assertEqual(len(self.store.load()['tasks']),2)

    def test_resume_single_steps_does_not_repeat_completed_calls(self):
        agent=Agent();runner=Runner()
        for index in range(60):
            state=drive(self.store,agent,runner,self.lease,max_steps=1)
            if state['status']=='finished':break
        calls=len(agent.calls);self.assertEqual(state['status'],'finished')
        drive(self.store,agent,runner,self.lease);self.assertEqual(len(agent.calls),calls)

    def test_crash_after_every_commit_recovers_or_blocks_without_retry(self):
        for checkpoint in range(1,60):
            with self.subTest(checkpoint=checkpoint):
                store=LaboratoryStore(self.root/('crash-'+str(checkpoint)));store.create(spec());agent=Agent();runner=Runner();save=store.save;commits=[]
                def crash(value):
                    save(value);commits.append(value['inflight'])
                    if len(commits)==checkpoint:raise KeyboardInterrupt()
                with patch.object(store,'save',side_effect=crash):
                    try:drive(store,agent,runner,self.lease)
                    except KeyboardInterrupt:pass
                prior=store.load();calls=(len(agent.calls),len(runner.calls));state=drive(store,agent,runner,self.lease)
                if prior['inflight'] is not None:
                    self.assertEqual(state['status'],'blocked');self.assertEqual(calls,(len(agent.calls),len(runner.calls)))
                else:self.assertEqual(state['status'],'finished')

    def test_bad_receipt_preserves_reservation_no_retry(self):
        agent=Agent();runner=Runner(bad=True);state=drive(self.store,agent,runner,self.lease)
        self.assertEqual(state['status'],'blocked');self.assertIsNotNone(state['inflight'])
        self.assertTrue(all(c['evidence'] is None for c in state['candidates']))
        drive(self.store,agent,runner,self.lease);self.assertEqual(len(runner.calls),1)

    def test_model_failure_and_malformed_ranking_block(self):
        state=drive(self.store,Agent(fail=RuntimeError('model release unknown')),Runner(),self.lease)
        self.assertEqual(state['status'],'blocked');self.assertIsNotNone(state['inflight'])
        store=LaboratoryStore(self.root/'malformed');store.create(spec());agent=Agent();call=agent.call
        def malformed(*args):
            reply=call(*args)
            if args[0]=='ranking':reply['text']='{"winner":true,"rationale":"claim done"}'
            return reply
        with patch.object(agent,'call',side_effect=malformed):state=drive(store,agent,Runner(),self.lease)
        self.assertEqual(state['status'],'blocked');self.assertEqual(state['matches'],[])

    def test_cancel_during_task_commits_release_before_stop(self):
        agent=Agent(callback=self.store.cancel);state=drive(self.store,agent,Runner(),self.lease)
        self.assertEqual(state['status'],'cancelled');self.assertIsNone(state['inflight']);self.assertEqual(len(agent.calls),1)
        self.assertEqual(drive(self.store,agent,Runner(),self.lease)['status'],'cancelled')

    def test_immutable_evaluator_completed_task_and_incumbent(self):
        state=drive(self.store,Agent(),Runner(),self.lease)
        for change in ('evaluator','candidate','receipt','incumbent','elo','match','reflection','meta','task'):
            with self.subTest(change=change):
                bad=copy.deepcopy(state)
                if change=='evaluator':bad['config']['evaluator']['required_speedup']=0
                if change=='candidate':bad['candidates'][0]['source']='replacement'
                if change=='receipt':bad['candidates'][0]['evidence']['evaluation']['correct']=False
                if change=='incumbent':bad['incumbent_id']=3
                if change=='elo':bad['candidates'][0]['model_elo']=9999
                if change=='match':bad['matches'][0]['winner']=None
                if change=='reflection':bad['candidates'][0]['reflection']='unrecorded'
                if change=='meta':bad['meta_reviews'][0]['text']='unrecorded'
                if change=='task':bad['tasks'][0]['result']={'candidate_id':2}
                with self.assertRaises(ValueError):self.store.save(bad)

    def test_budgets_and_mode_switch_fail_closed(self):
        store=LaboratoryStore(self.root/'budget');store.create(spec(model_calls=1))
        state=drive(store,Agent(),Runner(),self.lease);self.assertEqual(state['status'],'budget_exhausted')
        with self.assertRaises(ValueError):drive(self.store,Agent(),ExperimentRunner(),self.lease)

    def test_production_gate_and_runner_never_launch(self):
        with patch('subprocess.Popen',side_effect=AssertionError('launch')):
            state=drive(self.store,RealAgent(),ExperimentRunner(),self.lease)
            self.assertEqual(state['status'],'blocked');self.assertIn('HELD',state['reason'])
            with self.assertRaisesRegex(RuntimeError,'confinement'):
                ExperimentRunner().run(ExperimentRequest('test',1,'a'*64,digest(EVALUATOR)),'code',self.root/'never',0)
            self.assertFalse((self.root/'never').exists())

    def test_receipt_scope_and_finite_measurements(self):
        request=ExperimentRequest('experiment-1',1,'a'*64,digest(EVALUATOR));good=receipt(request)
        self.assertTrue(validate_receipt(good,request,False)['correct'])
        for change in ('scope','nan','actual','admission','artifact','output'):
            bad=copy.deepcopy(good)
            if change=='scope':bad['request']['command_id']='shell'
            if change=='nan':bad['measurements']['candidate_ns']=float('nan')
            if change=='actual':bad['actual_execution']=True
            if change=='admission':bad['admission_verified']=False
            if change=='artifact':bad['artifacts']=[{'path':'../control.py','bytes':1,'sha256':'a'*64}]
            if change=='output':bad['stdout']='x'*65537
            with self.assertRaises(ValueError):validate_receipt(bad,request,False)

    def test_configurable_inferred_alternatives_preserve_contract(self):
        for index, options in enumerate(({'elo_k':16,'reverse_pair_order':True,'debate_turns':1}, {'match_order':'round-robin','proximity':'character-trigram','generation':'debate','evolution':'divergent'})):
            config=spec();config['options'].update(options);store=LaboratoryStore(self.root/('alternative-'+str(index)));store.create(config)
            state=drive(store,Agent(),Runner(),self.lease)
            self.assertEqual(state['status'],'finished');self.assertEqual(state['incumbent_id'],2)
            self.assertEqual(state['config']['evaluator'],EVALUATOR)

    def test_prompt_overflow_blocks_without_hidden_clipping(self):
        config=spec();config['sources']=[__import__('research.retrieval',fromlist=['evidence']).evidence(str(i),'x'*512,'operator',None) for i in range(4)]
        store=LaboratoryStore(self.root/'overflow');store.create(config);agent=Agent();base=agent.call
        def large(*args):
            value=base(*args)
            if args[0]=='generation':value['text']='x'*2000
            return value
        with patch.object(agent,'call',side_effect=large):state=drive(store,agent,Runner(),self.lease)
        self.assertEqual(state['status'],'blocked');self.assertIn('4096',state['reason'])

    def test_debate_is_multiple_durable_turns_and_informs_judgment(self):
        agent=Agent();state=drive(self.store,agent,Runner(),self.lease)
        self.assertEqual(agent.calls.count('debate'),6)
        ranking_prompt=agent.prompts[agent.calls.index('ranking')]
        data=json.loads(ranking_prompt.split('\n')[-1]);self.assertEqual(len(data['debate_transcript']),2)
        self.assertEqual(len([t for t in state['tasks'] if t['kind']=='debate']),6)

    def test_seed_requests_differ_and_survive_resume_with_prior_candidates(self):
        state=self.store.load(); seeds=[t for t in state['tasks'] if t['kind']=='generation']
        requests=[json.loads(prompt(state,t).split('\n')[-1]) for t in seeds]
        self.assertNotEqual(requests[0]['generation_request'],requests[1]['generation_request'])
        self.assertEqual([r['generation_request']['task_id'] for r in requests],['seed-1','seed-2'])
        self.assertNotEqual(requests[0]['generation_request']['approach'],requests[1]['generation_request']['approach'])
        self.assertIn('Identify complexity assumptions',prompt(state,seeds[0]))
        agent=Agent();runner=Runner();drive(self.store,agent,runner,self.lease,max_steps=1)
        resumed=self.store.load();second=next(t for t in resumed['tasks'] if t['id']=='seed-2')
        next_prompt=prompt(resumed,second)
        prior=json.loads(next_prompt.split('\n')[-1])['generation_request']['prior_candidates']
        self.assertEqual(prior,[{'id':1,'source':resumed['candidates'][0]['source']}])
        self.assertEqual(next_prompt,prompt(self.store.load(),second))
        state=drive(self.store,agent,runner,self.lease)
        self.assertEqual(state['status'],'finished');self.assertEqual(agent.calls.count('generation'),2)

    def test_adaptive_reflection_waits_for_measurements_including_duplicates(self):
        for duplicate in (False,True):
            config=spec();config['options']['scheduling']='adaptive'
            store=LaboratoryStore(self.root/('adaptive-'+str(duplicate)));store.create(config)
            agent=Agent(duplicate=duplicate);base=agent.call;reviews=[]
            def inspect(role,*args):
                if role=='reflection':
                    data=json.loads(args[1].split('\n')[-1]);record=data['candidates'][0]
                    self.assertIsNotNone(record['measurements']);self.assertIsNotNone(record['checks'])
                    reviews.append(record['id'])
                return base(role,*args)
            with patch.object(agent,'call',side_effect=inspect):state=drive(store,agent,Runner(),self.lease)
            self.assertEqual(state['status'],'finished');self.assertEqual(reviews,[1,2,3])

    def test_meta_reviews_receive_completed_debate_transcripts_with_identity(self):
        agent=Agent();base=agent.call;seen=[]
        def inspect(role,*args):
            if role in ('meta-review','overview'):
                data=json.loads(args[1].split('\n')[-1]);seen.append((role,data['debate_transcript']))
            reply=base(role,*args)
            if role=='debate':reply['text']='Debate-specific concern '+str(len(agent.calls))
            return reply
        with patch.object(agent,'call',side_effect=inspect):state=drive(self.store,agent,Runner(),self.lease)
        self.assertEqual(state['status'],'finished')
        self.assertEqual([(role,len(items)) for role,items in seen],[('meta-review',2),('meta-review',6),('overview',6)])
        durable={t['id']:t for t in state['tasks'] if t['kind']=='debate'}
        for role,items in seen:
            for item in items:
                task=durable[item['task_id']]
                self.assertEqual(item['pair'],task['targets']);self.assertEqual(item['text'],task['result'])

    def test_stability_uses_recent_changes_and_requires_repeated_pair_coverage(self):
        state=self.store.load()
        state['candidates']=[{'id':i,'duplicate_of':None} for i in (1,2)]
        state['matches']=[];self.assertFalse(ratings_stable(state))
        match={'pair':[1,2],'winner':None}
        state['matches']=[match];self.assertFalse(ratings_stable(state))
        state['matches']=[match,match];self.assertTrue(ratings_stable(state))
        # Consistent repeated evidence moves ratings away from 1200 while its
        # latest update shrinks; contradictory evidence does the opposite.
        state['matches']=[{'pair':[1,2],'winner':1},{'pair':[2,1],'winner':1}]
        self.assertTrue(ratings_stable(state))
        state['matches'][1]['winner']=2;self.assertFalse(ratings_stable(state))
        state['matches']=[match,match]
        state['candidates'].append({'id':3,'duplicate_of':None})
        self.assertFalse(ratings_stable(state))
        state['candidates'][-1]['duplicate_of']=1;self.assertTrue(ratings_stable(state))

    def test_stable_evolution_resumes_from_durable_comparisons(self):
        config=spec();config['options'].update(evolution_trigger='stable',matches_per_pair=2,debate_turns=1)
        store=LaboratoryStore(self.root/'stable-history');store.create(config)
        agent=Agent();base=agent.call;runner=Runner()
        def consistent(role,*args):
            reply=base(role,*args)
            if role=='ranking':
                data=json.loads(args[1].split('\n')[-1])
                reply['text']=json.dumps({'winner':min(c['id'] for c in data['candidates']),'rationale':'consistent preference'})
            return reply
        with patch.object(agent,'call',side_effect=consistent):
            for _ in range(70):
                state=drive(store,agent,runner,self.lease,max_steps=1)
                if state['status']=='finished':break
        self.assertEqual(state['status'],'finished');self.assertEqual(agent.calls.count('evolution'),1)
        self.assertEqual(len(state['candidates']),3);self.assertEqual(len(state['matches']),6)

    def test_cli_offline_status_cancel_and_held_start(self):
        from research.laboratory import main
        path=self.root/'lab-spec.json';path.write_text(json.dumps(spec()))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(['research-lab-plan','--spec-file',str(path)]),0)
        self.assertTrue(json.loads(out.getvalue())['execution_held'])
        new=self.root/'held-cli'
        with patch('subprocess.Popen',side_effect=AssertionError('launch')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['research-lab-start','--spec-file',str(path),'--out',str(new),'--lease',str(self.lease)]),2)
            self.assertEqual(main(['research-lab-status','--run-dir',str(new)]),0)
            self.assertEqual(main(['research-lab-cancel','--run-dir',str(new)]),0)
        self.assertFalse((new/'task-seed-1').exists())

    def test_baseline_change_cannot_replace_incumbent(self):
        runner=Runner();base=runner.run
        def changed(*args):
            value=base(*args)
            if args[0].candidate_id==2:value['measurements']['baseline_sha256']='b'*64
            return value
        with patch.object(runner,'run',side_effect=changed):state=drive(self.store,Agent(),runner,self.lease)
        self.assertEqual(state['status'],'blocked');self.assertEqual(state['incumbent_id'],1)
        self.assertIsNone(state['candidates'][1]['evidence'])

    def test_wall_deadline_failure_and_terminal_controls(self):
        now=[0.]
        agent=Agent(callback=lambda:now.__setitem__(0,121.))
        state=drive(self.store,agent,Runner(),self.lease,clock=lambda:now[0])
        self.assertEqual(state['status'],'blocked');self.assertIsNotNone(state['inflight'])
        self.assertEqual(state['candidates'],[])

    def test_shared_writer_lock_excludes_other_driver(self):
        from research.harness import ownership
        with ownership(self.store,self.lease):
            with self.assertRaises(BlockingIOError):drive(self.store,Agent(),Runner(),self.lease)

    def test_fixture_state_never_reports_actual_benchmark_success(self):
        state=drive(self.store,Agent(),Runner(),self.lease)
        view=summary(state)
        self.assertEqual(view['evidence_status'],'ModelFreeContractOrUnrun');self.assertIsNone(view['objective_met'])
        bad=copy.deepcopy(state);bad['actual_experiments']=True
        with self.assertRaises(ValueError):self.store.save(bad)


if __name__=='__main__':unittest.main()
