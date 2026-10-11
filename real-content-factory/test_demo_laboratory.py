"""Model-free integration fixtures; no tokenizer or candidate program executes."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import demo_prompts as prompts
from research.harness import digest
from research.interval_evaluator import cases, case_identity
from research.laboratory import LaboratoryStore, configuration, demo_configuration, drive, main, summary


# These are source strings, never compiled or invoked in this module.
SOURCES=(
'''def merge_intervals(intervals):
    out=[]
    for a,b in sorted(intervals):
        if a==b: continue
        if out and a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)
        else: out.append([a,b])
    return out
''',
'''def merge_intervals(intervals):
    events=[]
    for a,b in intervals:
        if a<b:
            events.append((a,1))
            events.append((b,-1))
    events.sort()
    out=[]
    active=0
    i=0
    while i<len(events):
        point=events[i][0]
        before=active
        while i<len(events) and events[i][0]==point:
            active+=events[i][1]
            i+=1
        if before==0 and active>0: start=point
        if before>0 and active==0: out.append([start,point])
    return out
''',
'''def merge_intervals(intervals):
    result=[]
    for pair in sorted([list(p) for p in intervals if p[0]<p[1]]):
        if result and pair[0]<=result[-1][1]:
            result[-1][1]=max(result[-1][1],pair[1])
        else: result.append(pair)
    return result
''')


def at_cap(value,cap):
    return value+'#'*(cap-prompts.serialized_content_bytes(value))


class FixtureAgent:
    actual_inference=False
    def __init__(self,*,duplicate=False,capped=False,bad=None):
        self.calls=[];self.prompts=[];self.deadlines=[];self.source_index=0
        self.duplicate=duplicate;self.capped=capped;self.bad=bad
    def call(self,role,model,prompt,stage,deadline):
        self.calls.append(role);self.prompts.append(prompt);self.deadlines.append(deadline)
        if self.bad is not None:value=self.bad
        elif role in prompts.CODE_ROLES:
            value=SOURCES[0 if self.duplicate else self.source_index];self.source_index+=1
            if self.capped:value=at_cap(value,prompts.CAPS[role])
        elif role=='ranking':
            payload=json.loads(prompt.split('\n')[-1])
            rationale='Contract fixture opinion; no observed superiority.'
            if self.capped:rationale=at_cap(rationale,prompts.CAPS[role])
            value=json.dumps({'winner':payload['candidates'][0][0],'rationale':rationale})
        else:
            value='Contract fixture opinion; no model or candidate execution.'
            if self.capped:value=at_cap(value,prompts.CAPS[role])
        return {'text':value,'cleanup_verified':True,'actual_inference':False,'fixture_label':'MODEL_FREE_CONTRACT_FIXTURE'}


class FixtureRunner:
    actual_execution=False
    def __init__(self):self.calls=[];self.deadlines=[]
    def run(self,request,source,stage,deadline):
        self.calls.append(request);self.deadlines.append(deadline)
        # Fabricated results exercise receipt plumbing only; the frozen case list
        # is identified without evaluating source or claiming real correctness.
        correct=request.candidate_id!=3;total=len(cases());passed=total if correct else total-1
        return {'schema':'PMI_EXPERIMENT_RECEIPT_V1','request':request.payload(),'actual_execution':False,
                'confinement_verified':True,'admission_verified':True,'cleanup_verified':True,'exitcode':0,
                'stdout':'MODEL_FREE_CONTRACT_FIXTURE: fabricated counts, source never executed','stderr':'','artifacts':[],
                'measurements':{'tests_passed':passed,'tests_total':total,'baseline_ns':None,'candidate_ns':None,'baseline_sha256':None},
                'correctness_details':{'tests_passed':passed,'tests_total':total,'failures':[],
                                       'cases_sha256':case_identity(cases()),'correct':correct,
                                       'claim_scope':'fabricated model-free fixture only'},
                'runtime_sha256':'a'*64,'primitive_sha256':'b'*64,'seccomp_sha256':'c'*64}


class DemoLaboratoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.lease=self.root/'lease'
        self.store=LaboratoryStore(self.root/'run');self.store.create(demo_configuration())
        self.no_launch=patch('subprocess.Popen',side_effect=AssertionError('no model/candidate/tokenizer launch'))
        self.no_launch.start();self.addCleanup(self.no_launch.stop)

    def test_exact_profile_and_legacy_budget_unchanged(self):
        from test_laboratory import spec
        self.assertEqual(configuration(demo_configuration())['budget'],
                         {'model_calls':18,'experiment_calls':3,'wall_seconds':2400,'stage_seconds':120,'experiment_stage_seconds':60})
        self.assertEqual(configuration(spec())['budget']['wall_seconds'],1200)
        bad=spec(wall_seconds=2400)
        with self.assertRaises(ValueError):configuration(bad)
        for field,value in [('execution_profile','other'),('goal',''),('model_id','other'),('evolutions',True)]:
            bad=demo_configuration();bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):configuration(bad)
        bad=demo_configuration();bad['budget']['experiment_stage_seconds']=61
        with self.assertRaises(ValueError):configuration(bad)

    def test_complete_real_controller_sqlite_loop_is_explicitly_fixture(self):
        agent=FixtureAgent();runner=FixtureRunner()
        state=drive(self.store,agent,runner,self.lease)
        self.assertEqual(state['status'],'finished');self.assertEqual(len(agent.calls),18);self.assertEqual(len(runner.calls),3)
        report=summary(state);coverage=report['demonstration']
        self.assertTrue(coverage['full_comparative_coverage']);self.assertFalse(coverage['completed_real_demonstration'])
        self.assertFalse(state['actual_inference']);self.assertFalse(state['actual_experiments']);self.assertIsNone(report['objective_met'])
        self.assertFalse(state['candidates'][2]['evidence']['evaluation']['correct'])
        self.assertEqual(len(coverage['prompt_snapshot_refs']),18)
        for task in state['tasks']:
            if 'model_io' not in task:continue
            evidence=task['model_io'];saved=self.store.prompt_input(evidence['snapshot_sha256'])
            self.assertEqual(saved['prompt_sha256'],evidence['prompt_sha256'])
            self.assertIn(saved['prompt'],agent.prompts)
            payload=json.loads(saved['prompt'].split('\n')[-1])
            self.assertEqual(payload['snapshot_sha256'],digest(saved['snapshot']))
            self.assertFalse(evidence['completion']['actual_inference'])
            self.assertEqual(payload['mode'],'fixture')
        final=json.loads(agent.prompts[-1].split('\n')[-1])
        self.assertEqual(len(final['debates']),6);self.assertEqual(len(final['decisions']),3)
        self.assertEqual(len(final['candidates'][0][1]),64)
        self.assertEqual(self.store.load(),state)

    def test_all_response_caps_full_loop_rejects_no_prompts(self):
        agent=FixtureAgent(capped=True)
        state=drive(self.store,agent,FixtureRunner(),self.lease)
        self.assertEqual(state['status'],'finished',state['reason']);self.assertEqual(len(agent.calls),18)
        # A failed check encodes false (one byte longer than the prior all-pass
        # planner fixture); it must still fit without dropping evidence.
        self.assertEqual(max(len(p.encode()) for p in agent.prompts),3798)
        for role,value in zip(agent.calls,agent.prompts):self.assertLessEqual(len(value.encode()),prompts.raw_prompt_limit(role))
        for candidate in state['candidates']:self.assertEqual(prompts.serialized_content_bytes(candidate['source']),640)

    def test_duplicate_finished_is_ineligible_for_comparative_demonstration(self):
        state=drive(self.store,FixtureAgent(duplicate=True),FixtureRunner(),self.lease)
        report=summary(state)['demonstration']
        self.assertEqual(state['status'],'finished');self.assertEqual(state['model_calls'],9);self.assertEqual(state['experiment_calls'],1)
        self.assertEqual(report['distinct_candidate_count'],1);self.assertFalse(report['full_comparative_coverage'])
        self.assertFalse(report['completed_real_demonstration'])

    def test_resume_does_not_repeat_and_resolves_old_snapshots(self):
        agent=FixtureAgent();runner=FixtureRunner()
        for _ in range(60):
            state=drive(self.store,agent,runner,self.lease,max_steps=1)
            if state['status']=='finished':break
        self.assertEqual(state['status'],'finished');self.assertEqual(len(agent.calls),18)
        self.assertEqual(drive(self.store,agent,runner,self.lease),state);self.assertEqual(len(agent.calls),18)
        self.assertEqual(self.store.prompt_input(state['tasks'][0]['model_io']['snapshot_sha256'])['snapshot']['model_calls'],0)

    def test_reservation_and_snapshot_commit_atomically_before_agent_call(self):
        class CrashAgent(FixtureAgent):
            def call(inner,*args):
                saved=self.store.load();task=next(t for t in saved['tasks'] if t['id']==saved['inflight'])
                self.assertEqual(task['status'],'reserved');self.store.prompt_input(task['model_io']['snapshot_sha256'])
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):drive(self.store,CrashAgent(),FixtureRunner(),self.lease)
        agent=FixtureAgent();state=drive(self.store,agent,FixtureRunner(),self.lease)
        self.assertEqual(state['status'],'blocked');self.assertEqual(agent.calls,[])

    def test_invalid_source_keeps_returned_completion_and_blocks_without_retry(self):
        agent=FixtureAgent(bad='def merge_intervals(intervals):\n    return intervals.get(0)')
        state=drive(self.store,agent,FixtureRunner(),self.lease)
        self.assertEqual(state['status'],'blocked');self.assertEqual(state['candidates'],[])
        self.assertEqual(state['tasks'][0]['model_io']['completion']['text'],agent.bad)
        self.assertIsNotNone(state['inflight']);drive(self.store,agent,FixtureRunner(),self.lease);self.assertEqual(len(agent.calls),1)

    def test_model_and_cpu_deadlines_follow_role_and_global_remaining(self):
        agent=FixtureAgent();runner=FixtureRunner()
        state=drive(self.store,agent,runner,self.lease,max_steps=2,clock=lambda:100.)
        self.assertEqual(agent.deadlines,[220.]);self.assertEqual(runner.deadlines,[160.])
        state['elapsed_seconds']=2390.;self.store.save(state)
        drive(self.store,agent,runner,self.lease,max_steps=1,clock=lambda:100.)
        self.assertEqual(agent.deadlines[-1],110.)

    def test_snapshot_or_completion_tampering_refused(self):
        state=drive(self.store,FixtureAgent(),FixtureRunner(),self.lease,max_steps=1)
        changed=copy.deepcopy(state);changed['tasks'][0]['model_io']['completion']['text']='different'
        with self.assertRaises(ValueError):self.store.save(changed)
        with self.store.connection() as db:
            db.execute("UPDATE prompt_inputs SET prompt='changed'");db.commit()
        with self.assertRaisesRegex(ValueError,'prompt changed'):self.store.load()

    def test_cli_plan_allowed_start_and_resume_profile_refused(self):
        path=self.root/'spec.json';path.write_text(json.dumps(demo_configuration()))
        with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(main(['research-lab-plan','--spec-file',str(path)]),0)
        target=self.root/'cannot-start'
        with self.assertRaisesRegex(ValueError,'research-mode-start'):
            main(['research-lab-start','--spec-file',str(path),'--out',str(target),'--lease',str(self.lease)])
        self.assertFalse(target.exists())
        with self.assertRaisesRegex(ValueError,'research-mode-resume'):
            main(['research-lab-resume','--run-dir',str(self.store.root),'--lease',str(self.lease)])

    def test_coordination_owner_is_borrowed_and_legacy_lock_not_reopened(self):
        borrowed=[]
        class OwnerFixture:
            @contextlib.contextmanager
            def ownership(inner,store,lease):
                borrowed.append((store.root,lease));yield
        with patch('research.laboratory.ownership',side_effect=AssertionError('nested factory acquisition')):
            state=drive(self.store,FixtureAgent(),FixtureRunner(),self.lease,max_steps=1,coordination=OwnerFixture())
        self.assertEqual(borrowed,[(self.store.root,self.lease)]);self.assertEqual(state['model_calls'],1)

    def test_profile_data_does_not_allow_generic_real_adapter(self):
        from research.harness import RealAgent
        from research.experiments import ExperimentRunner
        with self.assertRaisesRegex(ValueError,'reviewed entry'):
            drive(self.store,RealAgent(),ExperimentRunner(),self.lease)
        self.assertEqual(self.store.load()['model_calls'],0)

    def test_serialized_escaping_caps_and_explicit_overflow(self):
        self.assertEqual(prompts.serialized_content_bytes('"'*96),192)
        self.assertEqual(prompts.bounded_text('reflection','"'*96),'"'*96)
        with self.assertRaises(ValueError):prompts.bounded_text('reflection','"'*97)
        self.assertEqual(prompts.serialized_content_bytes('\U0001f600'),12)
        state=self.store.load();state['mode']='fixture';original=copy.deepcopy(state)
        with patch.object(prompts,'raw_prompt_limit',return_value=1024),self.assertRaisesRegex(ValueError,'no clipping'):
            prompts.make_prompt(state,state['tasks'][0])
        self.assertEqual(state,original)

    def test_strict_ranking_and_narrow_source_subset(self):
        for reply in ('{"winner":1,"winner":2,"rationale":"x"}','{"winner":true,"rationale":"x"}',
                      '{"winner":1,"rationale":"x","extra":0}','{"winner":NaN,"rationale":"x"}'):
            with self.subTest(reply=reply),self.assertRaises(ValueError):prompts.validate_response('ranking',reply,[1,2])
        with self.assertRaises(ValueError):prompts.validate_response('generation','def merge_intervals(intervals):\n    x: int=1\n    return []')
        for source in SOURCES:self.assertEqual(prompts.validate_response('generation',source),source)


if __name__=='__main__':unittest.main()
