import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.executor import build_plan,seccomp_x86_64
from research.owned_cpu import CPUAdmission,CPUOwnedTransport,check_namespace,require_samples,NAMESPACES
from research.runtime_bundle import build_runtime
from research.semantic_proximity import parse_semantic
from research.laboratory import LaboratoryStore,drive,OPTIONS,pairs
from test_laboratory import Agent,Runner,spec


class FakeProcess:
    def __init__(self):
        read,write=os.pipe();self.stdin=os.fdopen(write,'wb');self.input_reader=read
        self.stdout=self.output(b'{"outputs":[]}');self.stderr=self.output(b'separate stderr');self.returncode=None;self.pid=123456
    def output(self,value):
        read,write=os.pipe();os.write(write,value);os.close(write);return os.fdopen(read,'rb')
    def poll(self):self.returncode=0;return 0
    def wait(self,timeout=None):self.returncode=0;return 0
    def kill(self):self.returncode=-9
    def close(self):
        for file in (self.stdin,self.stdout,self.stderr):
            if not file.closed:file.close()
        os.close(self.input_reader)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def test_production_entries_hold_before_any_process_or_runtime_copy(self):
        plan=build_plan('/runtime','/candidate','/work',5,6,[])
        with patch('subprocess.Popen',side_effect=AssertionError('launch')):
            for call in (lambda:CPUAdmission().qualify(plan,100),lambda:CPUOwnedTransport().run(plan,100),lambda:build_runtime(self.root/'bundle',{})):
                with self.assertRaises(RuntimeError):call()
        self.assertFalse((self.root/'bundle').exists())

    def test_admission_memory_pressure_and_continuity(self):
        samples=[{'time':float(i),'available':17*1024**3,'some':i,'full':0,'swapout':0,'swap_used':0} for i in range(3)];require_samples(samples)
        for change in ('memory','pressure','gap','reset'):
            bad=copy.deepcopy(samples)
            if change=='memory':bad[1]['available']=1
            if change=='pressure':bad[1]['full']=999999
            if change=='gap':bad[1]['time']=0.
            if change=='reset':bad[2]['some']=0
            with self.assertRaises(ValueError):require_samples(bad)

    def test_namespace_capture_requires_all_distinct_ids(self):
        parent={name:10 for name in NAMESPACES};row={'child-pid':99,**{name+'-namespace':20 for name in NAMESPACES}}
        self.assertEqual(check_namespace(row,parent),20)
        for name in NAMESPACES:
            if name=='user':continue
            bad=copy.deepcopy(row);bad[name+'-namespace']=10
            with self.assertRaises(ValueError):check_namespace(bad,parent)

    def test_owned_transport_invokes_only_fixed_argv_and_separate_pipes(self):
        filter_path=self.root/'filter';filter_path.write_bytes(seccomp_x86_64())
        status_path=self.root/'status';status_path.write_text('')
        fake=FakeProcess();self.addCleanup(fake.close)
        row={'child-pid':99,**{name+'-namespace':999999 for name in NAMESPACES}}
        transport=CPUOwnedTransport()
        with filter_path.open('rb') as filt,status_path.open('rb') as status:
            plan=build_plan('/runtime','/candidate',self.root.resolve(),filt.fileno(),status.fileno(),[])
            with patch('launch_policy.require_real_execution'),patch('research.owned_cpu.subprocess.Popen',return_value=fake) as popen,patch('research.owned_cpu.namespace_rows',return_value=[row,{'exit-code':0}]),patch('research.owned_cpu.pin_namespace',return_value=(999999,None)),patch('research.owned_cpu.namespace_members',return_value=[]) as members,patch('research.owned_cpu.available',return_value={'available':17*1024**3,'some':0,'full':0,'swapout':0,'swap_used':0}),patch('research.owned_cpu.fingerprint',side_effect=lambda pid:(pid,2,'pid:[3]')):
                result=transport.run(plan,__import__('time').monotonic()+10)
        self.assertEqual(result['stdout'],b'{"outputs":[]}');self.assertEqual(result['stderr'],b'separate stderr')
        self.assertEqual(popen.call_args.args,(plan.argv,));self.assertIs(popen.call_args.kwargs['shell'],False)
        expected={int(plan.argv[plan.argv.index('--seccomp')+1]),int(plan.argv[plan.argv.index('--json-status-fd')+1])}
        self.assertEqual(set(popen.call_args.kwargs['pass_fds']),expected)
        self.assertTrue(result['cleanup_verified'])
        self.assertTrue(members.call_args_list)
        self.assertTrue(all(call.args==(999999,2) for call in members.call_args_list))
        self.assertEqual(transport.last_cleanup,{'launcher_reaped':True,'namespace_verified':True,'remaining_pids':[],'cleanup_verified':True})

    def test_owned_transport_setup_failure_blocks_no_successful_capture(self):
        filt_path=self.root/'filter';filt_path.write_bytes(seccomp_x86_64());status_path=self.root/'status';status_path.write_text('')
        fake=FakeProcess();self.addCleanup(fake.close)
        with filt_path.open('rb') as filt,status_path.open('rb') as status:
            plan=build_plan('/runtime','/candidate',self.root.resolve(),filt.fileno(),status.fileno(),[])
            with patch('launch_policy.require_real_execution'),patch('research.owned_cpu.subprocess.Popen',return_value=fake),patch('research.owned_cpu.namespace_rows',return_value=[]),patch('research.owned_cpu.available',return_value={'available':17*1024**3,'some':0,'full':0,'swapout':0,'swap_used':0}),patch('research.owned_cpu.fingerprint',side_effect=lambda pid:(pid,2,'pid:[3]')),self.assertRaises(ValueError):CPUOwnedTransport().run(plan,__import__('time').monotonic()+10)

    def test_runtime_builder_copies_only_explicit_pinned_fixture_files(self):
        python=self.root/'python-fixture';python.write_bytes(b'not executable')
        worker=Path(__file__).parent/'research/interval_worker.py'
        inventory={name:{'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in [('usr/bin/python3',python),('worker.py',worker)]}
        with patch('launch_policy.require_real_execution'):
            receipt=build_runtime(self.root/'runtime',inventory)
        self.assertEqual(set(receipt['files']),set(inventory));self.assertEqual((self.root/'runtime/usr/bin/python3').read_bytes(),b'not executable')
        (self.root/'runtime').chmod(0o700);(self.root/'runtime/usr').chmod(0o700);(self.root/'runtime/usr/bin').chmod(0o700)
        bad=copy.deepcopy(inventory);bad['../secrets']=bad.pop('usr/bin/python3')
        with patch('launch_policy.require_real_execution'),self.assertRaises(ValueError):build_runtime(self.root/'bad',bad)
        self.assertFalse((self.root/'bad').exists())

    def test_semantic_backend_uses_existing_agent_and_schedules_after_graph(self):
        config=spec();config['options']['proximity']='model-semantic';store=LaboratoryStore(self.root/'semantic');store.create(config)
        agent=Agent();base=agent.call
        def semantic(role,model,prompt,stage,deadline):
            if role!='semantic-proximity':return base(role,model,prompt,stage,deadline)
            data=json.loads(prompt.split('\n')[-1]);ids=[c['id'] for c in data['candidates']]
            edges=[{'pair':list(pair),'similarity':.8,'rationale':'same algorithm family'} for pair in __import__('itertools').combinations(ids,2)]
            agent.calls.append(role);return {'text':json.dumps({'edges':edges}),'cleanup_verified':True,'actual_inference':False}
        with patch.object(agent,'call',side_effect=semantic):state=drive(store,agent,Runner(),self.root/'lease')
        self.assertEqual(state['status'],'finished');self.assertEqual(agent.calls.count('semantic-proximity'),2)
        self.assertIn('model-semantic',state['proximity'][0]['method']);self.assertEqual(state['incumbent_id'],2)

    def test_semantic_invalid_or_incomplete_results_do_not_fallback(self):
        for answer in ['{"edges":[]}',json.dumps({'edges':[{'pair':[1,2],'similarity':float('nan'),'rationale':'claim'}]}),json.dumps({'edges':[{'pair':[1,2],'similarity':True,'rationale':'claim'}]})]:
            with self.assertRaises(ValueError):parse_semantic(answer,[1,2])

    def test_tournament_repeated_comparisons_and_configurable_stability(self):
        config=spec();config['budget']['model_calls']=24;config['options'].update(matches_per_pair=2,debate_turns=1,scheduling='adaptive',match_order='paper-priority')
        store=LaboratoryStore(self.root/'repeat');store.create(config);state=drive(store,Agent(),Runner(),self.root/'lease')
        self.assertEqual(state['status'],'finished');self.assertEqual(len(state['matches']),6)
        first,second=state['matches'][:2];self.assertEqual(first['pair'],list(reversed(second['pair'])))
        config=spec();config['options'].update(evolution_trigger='stable',stability_threshold=0)
        store=LaboratoryStore(self.root/'stable');store.create(config);state=drive(store,Agent(),Runner(),self.root/'lease')
        self.assertEqual(state['status'],'finished');self.assertEqual(len(state['candidates']),2)

    def test_admission_lock_is_released_on_rejection(self):
        from research.owned_cpu import CPUAdmission
        from research.executor import build_plan
        holder=CPUAdmission();lock=io.StringIO()
        plan=build_plan('/runtime','/candidate','/work',5,6,[])
        with patch('launch_policy.require_real_execution'),patch('research.owned_cpu.LOCK') as path,patch('research.owned_cpu.fcntl.flock',side_effect=BlockingIOError):
            path.open.return_value=lock
            with self.assertRaises(BlockingIOError):holder.qualify(plan,100)
        self.assertTrue(lock.closed);self.assertIsNone(holder.lock)

    def test_transport_bounds_cleanup_and_model_failure_no_fallback(self):
        filt_path=self.root/'filter';filt_path.write_bytes(seccomp_x86_64());status_path=self.root/'status';status_path.write_text('')
        fake=FakeProcess();self.addCleanup(fake.close)
        row={'child-pid':99,**{name+'-namespace':999999 for name in NAMESPACES}}
        member={'identity':(42,100,'pid:[999999]'),'rss':512*1048576,'seccomp':2}
        transport=CPUOwnedTransport()
        with filt_path.open('rb') as filt,status_path.open('rb') as status:
            plan=build_plan('/runtime','/candidate',self.root.resolve(),filt.fileno(),status.fileno(),[])
            with patch('launch_policy.require_real_execution'),patch('research.owned_cpu.subprocess.Popen',return_value=fake),patch('research.owned_cpu.namespace_rows',return_value=[row]),patch('research.owned_cpu.pin_namespace',return_value=(999999,None)),patch('research.owned_cpu.namespace_members',side_effect=[[member],[member],[],[]]),patch('research.owned_cpu.kill_member') as kill,patch('research.owned_cpu.available',return_value={'available':17*1024**3,'some':0,'full':0,'swapout':0,'swap_used':0}),patch('research.owned_cpu.fingerprint',side_effect=lambda pid:(pid,2,'pid:[3]')),self.assertRaises(MemoryError):transport.run(plan,__import__('time').monotonic()+10)
        kill.assert_called_once_with(member)
        self.assertTrue(transport.last_cleanup['cleanup_verified']);self.assertEqual(transport.last_cleanup['remaining_pids'],[])

    def test_semantic_driver_failure_stays_reserved_without_lexical_fallback(self):
        config=spec();config['options']['proximity']='model-semantic';store=LaboratoryStore(self.root/'bad-semantic');store.create(config)
        agent=Agent();base=agent.call
        def invalid(*args):
            if args[0]=='semantic-proximity':return {'text':'{"edges":[]}','cleanup_verified':True,'actual_inference':False}
            return base(*args)
        with patch.object(agent,'call',side_effect=invalid):state=drive(store,agent,Runner(),self.root/'lease')
        self.assertEqual(state['status'],'blocked');self.assertIsNotNone(state['inflight']);self.assertEqual(state['proximity'],[])

    def test_admission_fresh_virtual_samples_with_owned_lock(self):
        clock=[0.];holder=CPUAdmission();lock=io.StringIO();plan=build_plan('/runtime','/candidate','/work',5,6,[])
        def sample():return {'time':clock[0],'available':17*1024**3,'some':0,'full':0,'swapout':0,'swap_used':0}
        with patch('launch_policy.require_real_execution'),patch('research.owned_cpu.LOCK') as path,patch('research.owned_cpu.fcntl.flock') as flock,patch('research.owned_cpu.time.monotonic',side_effect=lambda:clock[0]),patch('research.owned_cpu.time.sleep',side_effect=lambda value:clock.__setitem__(0,clock[0]+value)),patch('research.owned_cpu.available',side_effect=sample),patch('research.owned_cpu.fingerprint',side_effect=lambda pid:(pid,2,'pid:[3]')):
            path.open.return_value=lock;admission=holder.qualify(plan,100);holder.release()
        self.assertTrue(admission['exclusive']);self.assertEqual(admission['expires_at'],3);self.assertTrue(lock.closed);flock.assert_called_once()

    def test_correctness_cli_plan_and_held_start_create_no_runtime(self):
        from research.executor import main
        from test_experiment_executor import SOURCE
        source=self.root/'source.py';source.write_text(SOURCE)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(['research-experiment-plan','--source-file',str(source)]),0)
        self.assertEqual(json.loads(out.getvalue())['model_calls'],0)
        with patch('subprocess.Popen',side_effect=AssertionError('launch')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['research-experiment-start','--source-file',str(source),'--runtime-manifest',str(self.root/'absent'),'--out',str(self.root/'never'),'--lease',str(self.root/'lease')]),2)
        self.assertFalse((self.root/'never').exists())

    def test_one_shot_reservation_blocks_crash_and_disallows_duplicate_run(self):
        from research.executor import run_once
        from test_experiment_executor import SOURCE
        import sqlite3
        source=self.root/'source.py';source.write_text(SOURCE)
        fake=unittest.mock.Mock();fake.run.side_effect=KeyboardInterrupt()
        with patch('launch_policy.require_real_execution'),patch('research.executor.BubblewrapCorrectnessRunner.from_manifest',return_value=fake),self.assertRaises(KeyboardInterrupt):run_once(source,self.root/'manifest',self.root/'once',self.root/'lease')
        with contextlib.closing(sqlite3.connect(self.root/'once/experiment.sqlite3')) as db:row=db.execute('SELECT status,result FROM run').fetchone()
        self.assertEqual(row[0],'blocked');self.assertFalse(json.loads(row[1])['retry_allowed'])
        with patch('launch_policy.require_real_execution'),patch('research.executor.BubblewrapCorrectnessRunner.from_manifest',return_value=fake),self.assertRaises(FileExistsError):run_once(source,self.root/'manifest',self.root/'once',self.root/'lease')
        self.assertEqual(fake.run.call_count,1)

    def test_one_shot_valid_mock_receipt_is_durably_recorded(self):
        from research.executor import run_once
        from research.interval_evaluator import cases,reference,evaluate_outputs
        from test_experiment_executor import SOURCE
        import sqlite3
        source=self.root/'source.py';source.write_text(SOURCE)
        def result(request,*unused):
            values=cases();details=evaluate_outputs(values,[reference(value) for value in values])
            return {'schema':'PMI_EXPERIMENT_RECEIPT_V1','request':request.payload(),'actual_execution':True,'confinement_verified':True,'admission_verified':True,'cleanup_verified':True,'exitcode':0,'stdout':'mock protocol','stderr':'','artifacts':[],
                    'measurements':{'tests_passed':884,'tests_total':884,'baseline_ns':None,'candidate_ns':None,'baseline_sha256':None},'correctness_details':details,'runtime_sha256':'a'*64,'primitive_sha256':'b'*64,'seccomp_sha256':'c'*64}
        fake=unittest.mock.Mock();fake.run.side_effect=result
        with patch('launch_policy.require_real_execution'),patch('research.executor.BubblewrapCorrectnessRunner.from_manifest',return_value=fake),patch('subprocess.Popen',side_effect=AssertionError('launch')):saved=run_once(source,self.root/'manifest',self.root/'success',self.root/'lease')
        self.assertTrue(saved['evaluation']['correct'])
        with contextlib.closing(sqlite3.connect(self.root/'success/experiment.sqlite3')) as db:self.assertEqual(db.execute('SELECT status FROM run').fetchone()[0],'finished')

    def test_namespace_handle_is_pinned_until_cleanup_and_identity_checked(self):
        from research.owned_cpu import pin_namespace
        parent={name:10 for name in NAMESPACES};row={'child-pid':99,**{name+'-namespace':20 for name in NAMESPACES}}
        with patch('research.owned_cpu.fingerprint',return_value=(99,7,'pid:[20]')),patch('research.owned_cpu.os.open',return_value=100) as opened,patch('research.owned_cpu.os.fstat',return_value=type('Stat',(),{'st_ino':20})()),patch('research.owned_cpu.Path.stat',return_value=type('Stat',(),{'st_ino':30})()):
            self.assertEqual(pin_namespace(row,parent),(20,100))
        self.assertEqual(opened.call_args.args[0],'/proc/99/ns/pid')
        with patch('research.owned_cpu.fingerprint',return_value=(99,7,'pid:[20]')),patch('research.owned_cpu.os.open',return_value=100),patch('research.owned_cpu.os.fstat',return_value=type('Stat',(),{'st_ino':21})()),patch('research.owned_cpu.os.close') as close,self.assertRaises(ValueError):pin_namespace(row,parent)
        close.assert_called_once_with(100)

    def test_bwrap_status_user_namespace_is_checked_from_live_proc(self):
        from research.owned_cpu import pin_namespace
        parent={name:10 for name in NAMESPACES};row={'child-pid':99,**{name+'-namespace':20 for name in NAMESPACES if name!='user'}}
        self.assertEqual(check_namespace(row,parent),20)
        with patch('research.owned_cpu.fingerprint',return_value=(99,7,'pid:[20]')),patch('research.owned_cpu.os.open',return_value=100),patch('research.owned_cpu.os.fstat',return_value=type('Stat',(),{'st_ino':20})()),patch('research.owned_cpu.Path.stat',return_value=type('Stat',(),{'st_ino':10})()),patch('research.owned_cpu.os.close') as close,self.assertRaisesRegex(ValueError,'user namespace'):
            pin_namespace(row,parent)
        close.assert_called_once_with(100)

    def test_namespace_status_file_is_readable_and_partial_write_safe(self):
        from research.owned_cpu import namespace_rows
        path=self.root/'namespace-status'
        with path.open('x+b') as file:
            os.write(file.fileno(),b'{ "child-pid": 42 }\n{ "exit-')
            self.assertEqual(namespace_rows(file.fileno()),[{'child-pid':42}])
            os.write(file.fileno(),b'code": 0 }\n')
            self.assertEqual(namespace_rows(file.fileno()),[{'child-pid':42},{'exit-code':0}])

    def test_handshake_blocks_candidate_until_namespace_handle_is_pinned(self):
        read,write=os.pipe();self.addCleanup(os.close,read);self.addCleanup(os.close,write)
        plan=build_plan('/runtime','/candidate','/work',100,101,[],(read,write))
        self.assertIn('--block-fd',plan.argv);self.assertEqual(plan.block_write_fd,write)
        with self.assertRaises(ValueError):build_plan('/runtime','/candidate','/work',read,101,[],(read,write))

    def test_namespace_parser_ignores_uncommitted_nested_object(self):
        from research.owned_cpu import namespace_rows
        with (self.root/'partial').open('x+b') as file:
            os.write(file.fileno(),b'{"child-pid":42,"extra":{}')
            self.assertEqual(namespace_rows(file.fileno()),[])
            os.write(file.fileno(),b'}\n')
            self.assertEqual(namespace_rows(file.fileno()),[{'child-pid':42,'extra':{}}])
        with (self.root/'malformed').open('x+b') as file:
            os.write(file.fileno(),b'[]\n')
            with self.assertRaises(ValueError):namespace_rows(file.fileno())

    def test_namespace_pid_reuse_closes_pinned_handle(self):
        from research.owned_cpu import pin_namespace
        parent={name:10 for name in NAMESPACES};row={'child-pid':99,**{name+'-namespace':20 for name in NAMESPACES if name!='user'}}
        with patch('research.owned_cpu.fingerprint',side_effect=[(99,7,'pid:[20]'),(99,8,'pid:[20]')]),patch('research.owned_cpu.os.open',return_value=100),patch('research.owned_cpu.os.fstat',return_value=type('Stat',(),{'st_ino':20})()),patch('research.owned_cpu.Path.stat',return_value=type('Stat',(),{'st_ino':30})()),patch('research.owned_cpu.os.close') as close,self.assertRaisesRegex(ValueError,'child identity'):
            pin_namespace(row,parent)
        close.assert_called_once_with(100)

    def test_inherited_low_descriptor_hard_limit_refuses_before_popen(self):
        plan=build_plan('/runtime','/candidate','/work',5,6,[]);transport=CPUOwnedTransport()
        with patch('launch_policy.require_real_execution'),patch('research.owned_cpu.resource.getrlimit',return_value=(32,32)),patch('research.owned_cpu.subprocess.Popen') as popen,self.assertRaisesRegex(ValueError,'inherited descriptor'):
            transport.run(plan,100)
        popen.assert_not_called();self.assertIsNone(transport.last_cleanup)

    def test_child_limits_leave_worker_fd_headroom_without_touching_parent(self):
        from research.owned_cpu import child_limits,LAUNCHER_NOFILE
        import resource
        module=unittest.mock.Mock();plan=build_plan('/runtime','/candidate','/work',5,6,[])
        with patch('importlib.util.spec_from_file_location'),patch('importlib.util.module_from_spec',return_value=module),patch('research.owned_cpu.resource.setrlimit') as limit,patch('research.owned_cpu.os.sched_getaffinity',return_value={2,3}),patch('research.owned_cpu.os.sched_setaffinity') as affinity:
            child_limits(plan)
        self.assertIn(unittest.mock.call(resource.RLIMIT_NOFILE,(LAUNCHER_NOFILE,LAUNCHER_NOFILE)),limit.call_args_list)
        affinity.assert_called_once_with(0,{2});module.configure.assert_called_once()

    def test_selector_setup_error_still_reaps_owned_launcher(self):
        filt_path=self.root/'filter';filt_path.write_bytes(seccomp_x86_64());status_path=self.root/'status';status_path.write_text('')
        fake=FakeProcess();self.addCleanup(fake.close);transport=CPUOwnedTransport()
        with filt_path.open('rb') as filt,status_path.open('r+b') as status:
            plan=build_plan('/runtime','/candidate',self.root.resolve(),filt.fileno(),status.fileno(),[])
            with patch('launch_policy.require_real_execution'),patch('research.owned_cpu.subprocess.Popen',return_value=fake),patch('research.owned_cpu.available',return_value={}),patch('research.owned_cpu.fingerprint',side_effect=lambda pid:(pid,2,'pid:[3]')),patch('research.owned_cpu.selectors.DefaultSelector',side_effect=OSError('selector failure')),patch.object(fake,'wait',wraps=fake.wait) as waited,self.assertRaisesRegex(OSError,'selector failure'):
                transport.run(plan,__import__('time').monotonic()+10)
        waited.assert_called_once();self.assertTrue(transport.last_cleanup['launcher_reaped']);self.assertFalse(transport.last_cleanup['cleanup_verified'])


if __name__=='__main__':unittest.main()
