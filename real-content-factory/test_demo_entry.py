"""Model-free orchestration only: all real-shaped receipts below are mocks.

No tokenizer, model, generated candidate, subprocess, supervisor, or real runtime
is invoked. Laboratory integration and runtime receipt validation have their own
tests; these fixtures check only the fixed entry's orchestration and reporting.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import model_demo_entry as entry
from research.demo_prompts import validate_response


SOURCE='def merge_intervals(intervals):\n    return []\n'


class ContextFixture:
    operation_deadline=1000.
    def __init__(self):self.retained=[];self.finished=[]
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def retain_coordination(self,owner):self.retained.append(owner)
    def finish(self,status):
        if status not in ('finished','blocked','budget_exhausted','cancelled'):
            raise ValueError('invalid/duplicate child final status')
        self.finished.append(status)


class OwnerFixture:
    def __enter__(self):return self
    def __exit__(self,*args):pass


class AuthorityFixture:
    def __init__(self,root):
        self.root=root;self.model_count=0;self.candidate_count=0;self.current=None;self.calibration_done=False
    def calibrated(self):
        validate_response('generation',(self.root/'calibration/answer.txt').read_text())
        self.calibration_done=True


class StoreFixture:
    def __init__(self,root):self.root=Path(root);self.spec=None
    def create(self,spec):self.root.mkdir();self.spec=copy.deepcopy(spec)


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.root=self.base/'output';self.root.mkdir()
        self.checkout=self.base/'checkout';(self.checkout/'research/proposed-demo').mkdir(parents=True)
        self.prompt='MODEL_FREE_CALIBRATION_PROMPT_FIXTURE'
        (self.checkout/'research/proposed-demo/calibration_seed1_fixture.txt').write_text(self.prompt)
        self.spec={'fixture_only':'fixed specification'}
        (self.checkout/'research/model-demo-spec.json').write_text(json.dumps(self.spec))
        self.context=ContextFixture();self.authority=AuthorityFixture(self.root)
        self.calls=[];self.stores=[];self.drive_calls=[];self.agent_response=SOURCE
        self.workflow_status='finished';self.full=True;self.objective=True;self.pending=False
        self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
        for target,value in [('research.model_demo_entry.ROOT',self.root),
                             ('research.model_demo_entry.CHECKOUT',self.checkout)]:
            self.stack.enter_context(patch(target,value))
        self.stack.enter_context(patch('subprocess.Popen',side_effect=AssertionError('no real process allowed')))
        self.stack.enter_context(patch('research.deadline.read_supervisor_context',return_value=self.context))
        self.stack.enter_context(patch('research.coordination.CoordinationOwner',OwnerFixture))
        self.stack.enter_context(patch('research.model_demo_entry.activate_child',return_value=self.authority))
        self.stack.enter_context(patch('research.model_demo_entry.DemoCorrectnessRunner',return_value=object()))
        self.stack.enter_context(patch('research.harness.DemoRealAgent',side_effect=self.make_agent))
        self.stack.enter_context(patch('research.laboratory.LaboratoryStore',side_effect=self.make_store))
        self.stack.enter_context(patch('research.laboratory.demo_configuration',return_value=self.spec))
        self.stack.enter_context(patch('research.laboratory.drive',side_effect=self.fake_drive))
        self.stack.enter_context(patch('research.laboratory.summary',side_effect=lambda s:s['summary']))
        self.stack.enter_context(patch('research.model_demo_entry.time.monotonic',return_value=100.))

    def make_store(self,path):
        value=StoreFixture(path);self.stores.append(value);return value

    def write_completion_fixture(self,stage,role):
        stage.mkdir(exist_ok=True)
        receipt={'actual_inference':True,'status':'RealTextAndReleasePass',
                 'fixture_label':'MOCK_ONLY_NO_RUNTIME_PROOF','role':role,
                 'fit':{'fixture_label':'MOCK_ONLY_NO_TOKENIZER_CALL'}}
        (stage/'result.json').write_text(json.dumps(receipt))
        return receipt

    def make_agent(self,*,authority,coordination):
        test=self
        class AgentFixture:
            def call(inner,role,model,prompt,stage,deadline):
                test.calls.append((role,model,prompt,stage,deadline))
                authority.model_count+=1
                test.write_completion_fixture(stage,role)
                (stage/'answer.txt').write_text(test.agent_response)
                return {'text':test.agent_response,'actual_inference':True,'cleanup_verified':True,
                        'profile_id':'bonsai-research-v1','role':role,
                        'fit':{'fixture_label':'MOCK_ONLY_NO_TOKENIZER_CALL'},
                        'fixture_label':'MOCK_ONLY_NO_RUNTIME_PROOF'}
        return AgentFixture()

    def fake_drive(self,store,agent,runner,lease,*,coordination):
        self.assertTrue(self.authority.calibration_done)
        self.assertIs(coordination,self.context.retained[0])
        self.drive_calls.append(store.root)
        count=18 if self.full else 9
        tasks=[]
        for number in range(count):
            self.write_completion_fixture(store.root/('task-fixture-'+str(number)),'fixture-role')
            tasks.append({'model_io':{'completion':{'actual_inference':True,'fixture_label':'MOCK_ONLY'}}})
        if self.pending:tasks.append({'model_io':{'completion':None}})
        experiments=3 if self.full else 1
        for number in range(experiments):
            stage=store.root/('experiment-fixture-'+str(number));stage.mkdir()
            (stage/'experiment-receipt.json').write_text(json.dumps({'actual_execution':True,'cleanup_verified':True,'fixture_label':'MOCK_ONLY_NO_CANDIDATE_RUN'}))
        self.authority.model_count+=count;self.authority.candidate_count=experiments
        return {'tasks':tasks,'candidates':[{'evidence':{'fixture_label':'MOCK_ONLY'}}]*experiments,
                'status':self.workflow_status,
                'summary':{'status':self.workflow_status,'objective_met':self.objective,
                           'demonstration':{'completed_real_demonstration':self.full}}}

    def report(self):return json.loads((self.root/'child-result.json').read_text())

    def test_success_uses_calibration_then_fixed_loop_and_finished_protocol(self):
        self.assertEqual(entry.child_main(),0)
        self.assertEqual(self.calls,[('calibration','bonsai-stock',self.prompt,self.root/'calibration',220.)])
        self.assertEqual(self.stores[0].spec,self.spec);self.assertEqual(self.drive_calls,[self.root/'workflow'])
        self.assertEqual(self.context.finished,['finished'])
        report=self.report();self.assertEqual(report['status'],'completed')
        self.assertEqual(report['actual_model_completions'],19);self.assertEqual(report['generated_candidate_receipts'],3)
        self.assertIn('MOCK_ONLY',report['calibration']['fit']['fixture_label'])

    def test_invalid_calibration_is_retained_and_never_starts_workflow(self):
        self.agent_response='def merge_intervals(intervals):\n    return intervals.get(0)'
        self.assertEqual(entry.child_main(),2);self.assertEqual(self.drive_calls,[]);self.assertEqual(self.stores,[])
        report=self.report();self.assertEqual(report['status'],'blocked');self.assertEqual(report['actual_model_completions'],1)
        self.assertEqual(report['generated_candidate_receipts'],0);self.assertEqual(self.context.finished,['blocked'])

    def test_incomplete_comparative_loop_is_not_reported_success(self):
        self.full=False
        self.assertEqual(entry.child_main(),2)
        self.assertEqual(self.report()['status'],'finished_without_full_demonstration')
        self.assertEqual(self.context.finished,['blocked'])

    def test_failed_objective_is_not_reported_success(self):
        self.objective=False
        self.assertEqual(entry.child_main(),2)
        self.assertEqual(self.report()['status'],'finished_without_full_demonstration')
        self.assertEqual(self.context.finished,['blocked'])

    def test_blocked_reserved_completion_can_be_none_without_reporting_error(self):
        self.full=False;self.workflow_status='blocked';self.pending=True
        self.assertEqual(entry.child_main(),2)
        report=self.report();self.assertEqual(report['status'],'blocked')
        self.assertNotIn('error',report)
        self.assertEqual(report['actual_model_completions'],10)

    def test_raw_arguments_refused_before_authority_activation(self):
        with patch.object(entry,'activate_for_fixed_entry',side_effect=AssertionError('no activation')):
            with self.assertRaises(ValueError):entry.main(['--arbitrary','anything'])

    def test_supervisor_completion_boolean_controls_root_exit(self):
        for result,expected in (({'completed':True},0),({'completed':False},2),({'status':'completed'},2)):
            with self.subTest(result=result),patch.object(entry,'activate_for_fixed_entry',return_value=self.authority),\
                    patch('research.deadline.run_demo_supervisor',return_value=result) as supervisor,\
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entry.main([]),expected);supervisor.assert_called_once_with(authority=self.authority)

    def test_supervised_child_flag_routes_only_to_authenticated_child_function(self):
        with patch.object(entry,'child_main',return_value=2) as child,patch.object(entry,'activate_for_fixed_entry',side_effect=AssertionError('no parent activation')):
            self.assertEqual(entry.main(['--supervised-child']),2);child.assert_called_once_with()


class DirectRunnerWiringTests(unittest.TestCase):
    def test_constructor_uses_real_admission_transport_and_runner_classes(self):
        from research.executor import BubblewrapCorrectnessRunner
        from research.owned_cpu import CPUAdmission, CPUOwnedTransport
        from research.demo_authority import hashed
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);runtime=root/'runtime';runtime.mkdir()
            manifest=root/'interval-manifest.json'
            files={'worker.py':'a'*64};runtime_sha='b'*64;primitive_sha='c'*64
            manifest.write_text(json.dumps({'root':str(runtime),'files':files,
                'runtime_sha256':runtime_sha,'primitive_sha256':primitive_sha,
                'packaging':{'fixture_label':'MODEL_FREE_CONSTRUCTOR_ONLY'}}))
            pin={'external_artifacts_sha256':{str(manifest):hashed(manifest)}}
            authority=object();coordination=object()
            with patch.object(entry,'RUNTIME',runtime),patch.object(entry,'RUNTIME_MANIFEST',manifest),\
                    patch('research.executor.verify_runtime',return_value=runtime_sha) as verify_runtime,\
                    patch('research.demo_authority.verify_integrity',return_value=pin) as verify_integrity,\
                    patch('subprocess.Popen',side_effect=AssertionError('no launch during constructor test')):
                constructed=entry.DemoCorrectnessRunner(authority,coordination)
            verify_runtime.assert_called_once_with(runtime,files)
            verify_integrity.assert_called_once_with()
            self.assertIs(type(constructed.runner),BubblewrapCorrectnessRunner)
            self.assertIs(type(constructed.runner.admission),CPUAdmission)
            self.assertIs(type(constructed.runner.transport),CPUOwnedTransport)
            self.assertIs(constructed.authority,authority)
            self.assertIs(constructed.runner.authority,authority)
            self.assertIs(constructed.runner.admission.authority,authority)
            self.assertIs(constructed.runner.admission.coordination,coordination)
            self.assertIs(constructed.runner.transport.authority,authority)


if __name__=='__main__':unittest.main()
