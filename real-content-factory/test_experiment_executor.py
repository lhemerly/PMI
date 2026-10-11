import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import time
import unittest
from unittest.mock import patch

from research.harness import digest
from research.experiments import CORRECTNESS_EVALUATOR, ExperimentRequest, ExperimentHeld, validate_receipt
from research.executor import BubblewrapCorrectnessRunner, build_plan, collect_artifacts, seccomp_x86_64, verify_capture, verify_runtime
from research.interval_evaluator import reference, cases, evaluate_outputs, validate_source
from research.retrieval import RetrievalProvider, OperatorProvider, retrieve, evidence

SOURCE='def merge_intervals(intervals):\n    return sorted(intervals)\n'


class Admission:
    def qualify(self,plan,deadline):return {'approved':True,'exclusive':True,'expires_at':deadline}


class Transport:
    def __init__(self,correct=True,corrupt=None):self.plans=[];self.correct=correct;self.corrupt=corrupt
    def run(self,plan,deadline):
        self.plans.append(plan);values=json.loads(plan.stdin)['cases']
        outputs=[reference(value) for value in values] if self.correct else [[] for value in values]
        value={'exitcode':0,'stdout':json.dumps({'outputs':outputs},separators=(',',':')).encode(),'stderr':b'captured mock stderr',
               'namespace_isolation':True,'seccomp_verified':True,'resource_limits_verified':True,'remaining_pids':[],'cleanup_verified':True,'timed_out':False}
        if self.corrupt:self.corrupt(value)
        return value


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.runtime=self.root/'runtime';(self.runtime/'usr/bin').mkdir(parents=True)
        (self.runtime/'usr/bin/python3').write_bytes(b'not-an-executable-test-fixture')
        worker=Path(__file__).parent/'research/interval_worker.py';(self.runtime/'worker.py').write_bytes(worker.read_bytes())
        self.manifest={p.relative_to(self.runtime).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in self.runtime.rglob('*') if p.is_file()}
        for name in ('candidate','work','proc','dev','tmp'):(self.runtime/name).mkdir()
        for path in self.runtime.rglob('*'):
            path.chmod(0o500 if path.is_dir() or path.name=='python3' else 0o400)
        self.runtime.chmod(0o500)
        self.addCleanup(self.thaw_runtime)
        self.primitive=hashlib.sha256(Path('/usr/bin/bwrap').read_bytes()).hexdigest()
        self.request=ExperimentRequest('experiment-1',1,validate_source(SOURCE),digest(CORRECTNESS_EVALUATOR),'interval-union-correctness-v1','correctness-small-v1')

    def thaw_runtime(self):
        for path in self.runtime.rglob('*'):
            if path.is_dir() and not path.is_symlink():path.chmod(0o700)
        self.runtime.chmod(0o700)

    def run_mock(self,transport=None,admission=None):
        runner=BubblewrapCorrectnessRunner(self.runtime,self.manifest,self.primitive,admission or Admission(),transport or Transport())
        with patch('launch_policy.require_real_execution'),patch('subprocess.Popen',side_effect=AssertionError('no candidate execution allowed')):
            return runner.run(self.request,SOURCE,self.root/'stage',time.monotonic()+30)

    def test_bubblewrap_plan_has_strict_isolation_and_no_host_home(self):
        plan=build_plan('/runtime-bundle','/candidate-snapshot','/candidate-artifacts',5,6,cases())
        for flag in ('--unshare-user','--unshare-pid','--unshare-net','--unshare-ipc','--unshare-uts','--unshare-cgroup','--clearenv','--cap-drop','--die-with-parent','--seccomp','--new-session'):self.assertIn(flag,plan.argv)
        for unsafe in ('--share-net','--unshare-user-try','--unshare-all','--not-a-security-boundary','--dev-bind','--as-pid-1','--disable-userns'):self.assertNotIn(unsafe,plan.argv)
        self.assertNotIn('/home',plan.argv);self.assertNotIn('/sys',plan.argv);self.assertNotIn('/etc',plan.argv)
        self.assertEqual(plan.environment,{'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','LC_ALL':'C.UTF-8'})
        self.assertEqual(plan.wall_seconds,10);self.assertEqual(plan.cpu_seconds,5)
        self.assertEqual(plan.argv[-4:],('/usr/bin/python3','-I','-B','/worker.py'))

    def test_mount_and_descriptor_collisions_refused(self):
        for paths in [('/runtime','/runtime/secret','/work'),('relative','/candidate','/work')]:
            with self.assertRaises(ValueError):build_plan(*paths,5,6,[])
        with self.assertRaises(ValueError):build_plan('/runtime','/candidate','/work',5,5,[])

    def test_seccomp_instructions_reject_wrong_and_x32_abi_and_process_creation(self):
        program=seccomp_x86_64();self.assertEqual(len(program)%8,0)
        rows=[struct.unpack('HBBI',program[i:i+8]) for i in range(0,len(program),8)]
        self.assertIn((0x15,1,0,0xc000003e),rows);self.assertIn((0x35,0,1,0x40000000),rows)
        for syscall in (41,53,56,57,58,272,308,435):self.assertIn((0x15,0,1,syscall),rows)
        self.assertEqual(rows[-1],(0x06,0,0,0x7fff0000))

    def test_independent_oracle_and_frozen_case_identities(self):
        self.assertEqual(reference([[3,4],[1,3],[2,2],[1,2]]),[[1,4]])
        self.assertEqual(reference([[0,0]]),[])
        values=cases();self.assertEqual(values,cases());self.assertEqual(len(values),884)
        outputs=[reference(value) for value in values]
        self.assertTrue(evaluate_outputs(values,outputs)['correct'])
        outputs[1]=[['bad',1]];self.assertFalse(evaluate_outputs(values,outputs)['correct'])
        with self.assertRaises(ValueError):evaluate_outputs(values,[])

    def test_source_subset_refuses_evaluator_spoof_and_host_operations(self):
        validate_source(SOURCE)
        for source in ['import os\ndef merge_intervals(intervals): return []','def merge_intervals(intervals): return eval("1")','def merge_intervals(intervals): return intervals.__class__','def merge_intervals(intervals):\n global tests_passed\n return []','def other(intervals): return []']:
            with self.assertRaises(ValueError):validate_source(source)

    def test_runtime_manifest_rejects_extra_files_links_and_changed_worker(self):
        verify_runtime(self.runtime,self.manifest)
        self.runtime.chmod(0o700)
        (self.runtime/'credentials').write_text('fixture')
        self.runtime.chmod(0o500)
        with self.assertRaises(ValueError):verify_runtime(self.runtime,self.manifest)
        self.runtime.chmod(0o700);(self.runtime/'credentials').unlink();self.runtime.chmod(0o500)
        (self.runtime/'usr').chmod(0o700);(self.runtime/'usr/link').symlink_to('/etc');(self.runtime/'usr').chmod(0o500)
        with self.assertRaises(ValueError):verify_runtime(self.runtime,self.manifest)
        (self.runtime/'usr').chmod(0o700);(self.runtime/'usr/link').unlink();(self.runtime/'usr').chmod(0o500)
        (self.runtime/'worker.py').chmod(0o600);(self.runtime/'worker.py').write_text('changed');(self.runtime/'worker.py').chmod(0o400)
        with self.assertRaises(ValueError):verify_runtime(self.runtime,self.manifest)

    def test_mock_executor_oracle_receipt_has_no_performance_claim(self):
        transport=Transport();result=self.run_mock(transport)
        self.assertEqual(len(transport.plans),1)
        self.assertEqual(result['stderr'],'captured mock stderr')
        self.assertTrue(result['correctness_details']['correct'])
        self.assertIsNone(result['measurements']['candidate_ns'])
        check=validate_receipt(result,self.request,True);self.assertTrue(check['correct']);self.assertIsNone(check['speedup'])
        self.assertTrue((self.root/'stage/experiment-receipt.json').exists())

    def test_runner_status_descriptor_supports_write_and_pread(self):
        transport=Transport();original=transport.run
        def capture(plan,deadline):
            descriptor=int(plan.argv[plan.argv.index('--json-status-fd')+1])
            evidence=b'{"child-pid":42}\n'
            os.write(descriptor,evidence)
            self.assertEqual(os.pread(descriptor,len(evidence),0),evidence)
            return original(plan,deadline)
        transport.run=capture
        self.run_mock(transport)

    def test_wrong_algorithm_outputs_fail_even_with_zero_exit(self):
        result=self.run_mock(Transport(correct=False))
        self.assertEqual(result['exitcode'],0);self.assertFalse(validate_receipt(result,self.request,True)['correct'])

    def test_default_authority_denies_before_stage_or_process(self):
        with patch('subprocess.Popen',side_effect=AssertionError('launch')),self.assertRaises(RuntimeError):
            BubblewrapCorrectnessRunner().run(self.request,SOURCE,self.root/'never',time.monotonic()+30)
        self.assertFalse((self.root/'never').exists())

    def test_missing_admission_denies_before_transport(self):
        transport=Transport();runner=BubblewrapCorrectnessRunner(self.runtime,self.manifest,self.primitive,transport=transport)
        with patch('launch_policy.require_real_execution'),patch.object(runner.admission,'qualify',side_effect=ExperimentHeld('admission refused')),self.assertRaises(ExperimentHeld):runner.run(self.request,SOURCE,self.root/'stage',time.monotonic()+30)
        self.assertEqual(transport.plans,[]);self.assertFalse((self.root/'stage/experiment-receipt.json').exists())

    def test_namespace_resource_and_descendant_failures_do_not_produce_receipt(self):
        for index,changes in enumerate([{'namespace_isolation':False},{'seccomp_verified':False},{'resource_limits_verified':False},{'remaining_pids':[10]},{'timed_out':True}]):
            with self.subTest(changes=changes):
                runner=BubblewrapCorrectnessRunner(self.runtime,self.manifest,self.primitive,Admission(),Transport(corrupt=lambda value:value.update(changes)))
                with patch('launch_policy.require_real_execution'),self.assertRaises(ValueError):runner.run(self.request,SOURCE,self.root/str(index),time.monotonic()+30)
                self.assertFalse((self.root/str(index)/'experiment-receipt.json').exists())

    def test_artifact_paths_symlinks_hardlinks_and_size_bounds(self):
        work=self.root/'work';work.mkdir();(work/'data.txt').write_text('result')
        self.assertEqual(collect_artifacts(work)[0]['sha256'],hashlib.sha256(b'result').hexdigest())
        (work/'escape').symlink_to(self.root/'runtime')
        with self.assertRaises(ValueError):collect_artifacts(work)
        (work/'escape').unlink();__import__('os').link(work/'data.txt',work/'hardlink')
        with self.assertRaises(ValueError):collect_artifacts(work)
        (work/'hardlink').unlink();(work/'large').write_bytes(b'x'*(2*1048576+1))
        with self.assertRaises(ValueError):collect_artifacts(work)

    def test_retrieval_provenance_and_default_denial(self):
        operator=OperatorProvider([{'id':'contract','excerpt':'Half-open intervals exclude their right endpoint.'}])
        records=retrieve(operator,'interval union');self.assertEqual(records[0]['origin'],'operator');self.assertIsNone(records[0]['url'])
        with self.assertRaises(ExperimentHeld):retrieve(RetrievalProvider(),'interval union')
        class Mock:
            origin='mock'
            def search(self,query,limit):return [evidence('test','Mock search result','mock',None)]
        self.assertEqual(retrieve(Mock(),'union')[0]['origin'],'mock')
        class Retrieved:
            origin='retrieved'
            def search(self,query,limit):return [evidence('source','Provider excerpt','retrieved','https://example.org/source')]
        # Mocked provider carrying retrieved-origin protocol, not a network call.
        self.assertEqual(retrieve(Retrieved(),'union')[0]['url'],'https://example.org/source')
        with self.assertRaises(ValueError):evidence('mock','excerpt','mock','https://example.org')

    def test_retrieval_provider_cannot_relabel_or_tamper(self):
        class Bad:
            origin='mock'
            def search(self,query,limit):return [evidence('operator','text','operator',None)]
        with self.assertRaises(ValueError):retrieve(Bad(),'union')
        with self.assertRaises(ValueError):retrieve(Bad(),'x'*513)

    def test_correctness_driver_preserves_durable_reservation_on_unproven_cleanup(self):
        from research.laboratory import LaboratoryStore,drive
        from test_laboratory import Agent,spec
        config=spec();config['evaluator']=CORRECTNESS_EVALUATOR
        store=LaboratoryStore(self.root/'lab');store.create(config)
        class MockRunner:
            actual_execution=False
            def run(inner,request,source,stage,deadline):
                runner=BubblewrapCorrectnessRunner(self.runtime,self.manifest,self.primitive,Admission(),Transport(corrupt=lambda value:value.update(cleanup_verified=False)))
                with patch('launch_policy.require_real_execution'):return runner.run(request,source,stage,deadline)
        agent=Agent();state=drive(store,agent,MockRunner(),self.root/'lease')
        self.assertEqual(state['status'],'blocked');self.assertIsNotNone(state['inflight'])
        before=len(agent.calls);drive(store,agent,MockRunner(),self.root/'lease');self.assertEqual(len(agent.calls),before)

    def test_correctness_receipt_does_not_accept_timing_or_forged_counts(self):
        result=self.run_mock()
        for field in ('timing','counts','case_identity'):
            bad=copy.deepcopy(result)
            if field=='timing':bad['measurements']['candidate_ns']=1
            if field=='counts':bad['correctness_details']['tests_passed']=0
            if field=='case_identity':bad['correctness_details']['cases_sha256']='0'*64
            with self.assertRaises(ValueError):validate_receipt(bad,self.request,True)

    def test_artifact_root_symlink_and_retrieved_config_relabelling_refused(self):
        link=self.root/'link';link.symlink_to(self.runtime,target_is_directory=True)
        with self.assertRaises(ValueError):collect_artifacts(link)
        from research.laboratory import configuration
        from test_laboratory import spec
        config=spec();config['sources']=[evidence('provided','Text','retrieved','https://example.org')]
        with self.assertRaises(ValueError):configuration(config)

    def test_correctness_receipts_integrate_without_benchmark_ranking(self):
        from research.laboratory import LaboratoryStore,drive,summary
        from test_laboratory import Agent,spec
        config=spec();config['evaluator']=CORRECTNESS_EVALUATOR;config['evolutions']=0
        store=LaboratoryStore(self.root/'correctness-lab');store.create(config)
        class MockRunner:
            actual_execution=False
            def run(inner,request,source,stage,deadline):
                runner=BubblewrapCorrectnessRunner(self.runtime,self.manifest,self.primitive,Admission(),Transport())
                with patch('launch_policy.require_real_execution'),patch('subprocess.Popen',side_effect=AssertionError('launch')):
                    result=runner.run(request,source,stage,deadline)
                result['actual_execution']=False;return result
        state=drive(store,Agent(),MockRunner(),self.root/'lease')
        self.assertEqual(state['status'],'finished');self.assertEqual(state['incumbent_id'],1)
        self.assertIsNone(summary(state)['objective_met']);self.assertTrue(summary(state)['contract_objective_met'])
        self.assertIsNone(state['candidates'][0]['evidence']['evaluation']['speedup'])


if __name__=='__main__':unittest.main()
