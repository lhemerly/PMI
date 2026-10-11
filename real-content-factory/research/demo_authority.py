"""Exact first research demonstration scope; source review is not launch authority.

The fixed entry is submitted separately to normal platform review. This scope
implements the explicitly approved calibration + 18 role calls + 3 correctness
runs only. No CLI flag, model output, JSON permission, or environment variable
releases the general gate. Same-user host compromise is outside this application
capability boundary; generated programs never receive host capabilities.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

CHECKOUT=Path('/home/luizh/Documents/Codex/2026-10-08/task-8/real-content-factory')
ROOT=Path('/tmp/pmi-model-demo-v4')
PRIOR_ROOT=Path('/tmp/pmi-model-demo-v2')
PRIOR_ARTIFACTS={
    'child-result.json':'632c43affe9ef1c32b307711d3464a3b4a639f28865bc884d0ddf9b2ab828262',
    'supervisor-report.json':'7bc33c5b71d6b564bae5136146807eee66ab1608cb1ad7faabd4e20a19db6e21',
    'calibration/admission.json':'585352660b7717fe86de3adc4ecd50aae092ccc8c45cc219e3fef806ee741adb',
    'calibration/request.json':'66c458ec0bfaa9208a5949020ae638ff5cf3ba3b557d4b09f9c2962abce11238',
    'calibration/loading-telemetry.jsonl':'39ffa84f48c69345e7b6669ff737210c107f4d9053bacd32969298b60cea8bfb',
    'controller.stdout.log':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'controller.stderr.log':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
}
FAILED_NATIVE_ROOT=Path('/tmp/pmi-model-demo-v3')
FAILED_NATIVE_ARTIFACTS={'calibration/admission.json': 'a0b090e0888bb63ebc6fe547193c7cb4df96ab0561ff8222fddb4c5f94551f78', 'calibration/admission.json.policy.json': 'd692806081efe7514857444a218e8306f81bb1a3a2867992ac8b359dee656c43', 'calibration/core-limit.json': '39172af13527930348bd2d4cd0b3aef8f2ff1377c9b97b63fc5b4f6ee19c1238', 'calibration/guarded-command.json': '4a76945871ba1e32c2f87aeb59c236ea601de5c568dad03f1892f20180ab99a2', 'calibration/host.jsonl': '56b365ec91ead67b9de86303b84b573ac0243d5a6f218bf90c239f5e04a17f07', 'calibration/loading-telemetry.jsonl': '89f166e954f16914f51fe35b43a5bb759cbf6a1758e0b6898e5ce862ca8a3546', 'calibration/native.stderr.log': 'd77cd9df3202bafd55ff30c9995dce2ab5d17a7d5bc2f784b7c6744c22c9ccea', 'calibration/native.stdout.log': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'calibration/request.json': '66c458ec0bfaa9208a5949020ae638ff5cf3ba3b557d4b09f9c2962abce11238', 'calibration/supervisor.log': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'calibration/trial.log': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'calibration/vram.jsonl': '083f14b736ed9fe34e6003f3ad06f38738163fb90077e6fcfc0d534fe321176d', 'child-result.json': '9f134c42a2dfaf8c581ca67d537e45e07e2ca686691b761a1241ca0a29730763', 'controller.stderr.log': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'controller.stdout.log': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'supervisor-report.json': 'f84739ecda9b04b3c5513a1b53969baca332deae66e6d485f9c131c59b4fe0bd'}
PRIOR_NATIVE_ATTEMPTS=1
ORIGINAL_REPORT=Path('/tmp/pmi-model-demo-v1/supervisor-report.json')
ORIGINAL_REPORT_SHA256='be1a81736d6225ae452bb6cc22be3b880612209aefeee0b8e6d80341958e2ff7'
RUNTIME=Path('/tmp/pmi-correctness-runtime-v2')
RUNTIME_MANIFEST=Path('/tmp/pmi-first-correctness-v2/interval-manifest.json')
SEAL=CHECKOUT/'research/DEMO-INPUT-SEAL.json'
QUALIFICATION=CHECKOUT/'research/FIRST-QUALIFICATION-VERIFICATION.json'
QUALIFICATION_SHA256='823d6f708dceab39fe409fcfac0c139e6a0dca9e5017a2237385f1d5b9151388'
GENERAL_GATE_SHA256='3dacd214fb5679ec602da28ee7ae357a7d29b9c448268af3c9319fcb6d779f34'
APPROVED_SCOPE='Sentinel_4571dbb9c3188191a27e94f450e67075'
_active=None


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def hashed(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    raw=Path('/proc/%d/stat'%pid).read_text();end=raw.rfind(')');fields=raw[end+2:].split()
    if end<0 or int(raw.split(' ',1)[0])!=pid or len(fields)<20:raise ValueError('process identity unreadable')
    return (pid,int(fields[19])),int(fields[1])


def _regular(path,max_bytes):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>max_bytes:raise ValueError('bounded regular artifact required')
    before=path.stat();raw=path.read_bytes();after=path.stat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('artifact changed while reading')
    return raw


def verify_integrity():
    """Integrity alone never grants activation. No model/tensor/runtime loading."""
    if hashed(QUALIFICATION)!=QUALIFICATION_SHA256 or hashed(CHECKOUT/'launch_policy.py')!=GENERAL_GATE_SHA256:
        raise ValueError('prior qualification or unrelated general gate changed')
    pin=json.loads(_regular(SEAL,1048576))
    if pin.get('schema')!='PMI_REAL_BACKEND_INTEGRITY_V1' or pin.get('scope_id')!='pmi-model-demo-v1':raise ValueError('separate reviewed demo integrity seal required')
    expected={str(p.relative_to(CHECKOUT)) for p in CHECKOUT.glob('*.py') if not p.name.startswith('test_')}
    expected|={str(p.relative_to(CHECKOUT)) for p in (CHECKOUT/'research').glob('*.py')}
    if set(pin.get('source_sha256',{}))!=expected:raise ValueError('complete scoped source inventory required')
    for name,value in pin['source_sha256'].items():
        if hashed(CHECKOUT/name)!=value:raise ValueError('scoped source changed: '+name)
    for name,value in pin.get('scope_artifacts_sha256',{}).items():
        if Path(name).is_absolute() or '..' in Path(name).parts or hashed(CHECKOUT/name)!=value:raise ValueError('scope artifact changed')
    required={'research/model-demo-spec.json','research/proposed-demo/calibration_seed1_fixture.txt','research/FIRST-QUALIFICATION-VERIFICATION.json'}
    if not required<=set(pin.get('scope_artifacts_sha256',{})):raise ValueError('scope artifact inventory incomplete')
    from model_profiles import prompt_family_policy
    if pin.get('models',{}).get('bonsai-stock',{}).get('prompt_family')!=prompt_family_policy():raise ValueError('fixed prompt family changed')
    return pin


def _valid(authority,kind=None):
    if type(authority) is not _ResearchDemo or authority is not _active or authority.owner!=identity(os.getpid())[0]:raise ValueError('no active process-local demonstration authority')
    if kind is not None and authority.kind!=kind:raise ValueError('wrong demonstration authority role')
    if authority.context is not None:authority.context.validate()
    if not math.isfinite(authority.deadline) or time.monotonic()>=authority.deadline:raise TimeoutError('demonstration operation deadline')
    return authority


class _ResearchDemo:
    def __init__(self,kind,deadline,context=None):
        self.kind=kind;self.deadline=deadline;self.context=context;self.owner=identity(os.getpid())[0]
        self.model_count=0;self.candidate_count=0;self.calibration_done=False
        self.current=None;self.supervisor_started=False
        self.used_model_stages=set();self.used_candidate_ids=set()

    def calibrated(self):
        _valid(self,'child')
        if self.model_count!=1 or self.current is not None or self.calibration_done:raise ValueError('calibration completion absent')
        from .demo_prompts import validate_response
        validate_response('generation',(ROOT/'calibration/answer.txt').read_text())
        self.calibration_done=True

    def cpu_complete(self,receipt):
        _valid(self,'child')
        if not self.current or self.current['kind']!='cpu' or self.current['used']!={'candidate','admission','transport'}:raise ValueError('CPU request incomplete')
        from .experiments import validate_receipt
        validate_receipt(receipt,self.current['request'],True)
        self.current=None


def verify_failed_native_attempt():
    """Preserve the one failed native dispatch and charge it to the total cap."""
    entries=list(FAILED_NATIVE_ROOT.rglob('*'))
    if (FAILED_NATIVE_ROOT.is_symlink() or any(p.is_symlink() for p in entries)
            or {str(p.relative_to(FAILED_NATIVE_ROOT)) for p in entries}!=set(FAILED_NATIVE_ARTIFACTS)|{'calibration'}):
        raise ValueError('failed native artifact inventory changed')
    for name,value in FAILED_NATIVE_ARTIFACTS.items():
        if hashlib.sha256(_regular(FAILED_NATIVE_ROOT/name,1048576)).hexdigest()!=value:raise ValueError('failed native artifact changed')
    child=json.loads(_regular(FAILED_NATIVE_ROOT/'child-result.json',65536))
    outer=json.loads(_regular(FAILED_NATIVE_ROOT/'supervisor-report.json',65536))
    host=json.loads(_regular(FAILED_NATIVE_ROOT/'calibration/host.jsonl',1048576).splitlines()[-1])
    gpu=json.loads(_regular(FAILED_NATIVE_ROOT/'calibration/vram.jsonl',1048576).splitlines()[-1])
    if (child.get('error')!='ValueError: input changed or exceeded bound'
            or child.get('model_stages_reserved')!=1 or child.get('actual_model_completions')!=0
            or child.get('candidate_stages_reserved')!=0 or child.get('generated_candidate_receipts')!=0
            or outer.get('cleanup_verified') is not True or outer.get('remaining_owned')!=[] or outer.get('errors')!=[]
            or outer.get('coordination_release')!='verified_empty'
            or host.get('remaining_live_pids')!=[] or host.get('monitor_errors')!=[]
            or host.get('status')!='interrupted' or host.get('reason')!='signal_15'
            or gpu.get('cleanup_incomplete') is not False or gpu.get('errors')!=[] or gpu.get('status')!='interrupted'):
        raise ValueError('failed native cleanup/completion evidence changed')
    prior=json.loads(_regular(PRIOR_ROOT/'supervisor-report.json',65536))
    if outer['final_deadline']!=prior['final_deadline'] or outer['operation_deadline']!=prior['operation_deadline']:
        raise ValueError('failed native deadline changed')
    stderr=_regular(FAILED_NATIVE_ROOT/'calibration/native.stderr.log',1048576).decode()
    if (_regular(FAILED_NATIVE_ROOT/'calibration/native.stdout.log',1048576)
            or 'fitting params to device memory ...' not in stderr
            or 'successfully fit params to free device memory' in stderr
            or any(mode!='none' for mode in re.findall(r'load_mode = (\w+)',stderr))):
        raise ValueError('failed native pre-inference fitting scope unproven')


def activate_for_fixed_entry():
    """Called only by the fixed no-argument entry, separately reviewed for launch."""
    global _active
    if _active is not None or ROOT.exists() or ROOT.is_symlink():raise ValueError('demo already reserved; no automatic retry')
    verify_integrity()
    # Explicitly requested, separately reviewed continuation after the exact
    # admission-only stop. Never reuse a reservation with native dispatch.
    entries=list(PRIOR_ROOT.rglob('*'))
    expected=set(PRIOR_ARTIFACTS)|{'calibration'}
    if {str(p.relative_to(PRIOR_ROOT)) for p in entries}!=expected or any(p.is_symlink() for p in entries) or PRIOR_ROOT.is_symlink():
        raise ValueError('prior admission-only artifact inventory changed')
    for name,sha256 in PRIOR_ARTIFACTS.items():
        if hashlib.sha256(_regular(PRIOR_ROOT/name,1048576)).hexdigest()!=sha256:
            raise ValueError('prior admission-only artifact changed')
    if hashed(ORIGINAL_REPORT)!=ORIGINAL_REPORT_SHA256:raise ValueError('original deadline receipt changed')
    child=json.loads(_regular(PRIOR_ROOT/'child-result.json',65536))
    prior=json.loads(_regular(PRIOR_ROOT/'supervisor-report.json',65536))
    original=json.loads(_regular(ORIGINAL_REPORT,65536))
    from model_profiles import model_request
    prompt=(CHECKOUT/'research/proposed-demo/calibration_seed1_fixture.txt').read_text()
    request=json.loads(_regular(PRIOR_ROOT/'calibration/request.json',65536))
    if request!=model_request('bonsai-stock',prompt,'bonsai-research-v1','calibration'):
        raise ValueError('prior fixed calibration request changed')
    operation=child.get('uncertain_operation')
    if (child.get('status')!='blocked' or child.get('model_stages_reserved')!=1 or child.get('candidate_stages_reserved')!=0
            or child.get('actual_model_completions')!=0 or child.get('generated_candidate_receipts')!=0
            or child.get('model_completion_receipts')!=[] or child.get('candidate_receipts')!=[]
            or child.get('workflow') is not None or child.get('calibration') is not None
            or child.get('error')!='ValueError: substantial host paging during admission'
            or not isinstance(operation,dict) or set(operation)!={'kind','model_id','role','prompt_sha256','stage','deadline'}
            or operation.get('kind')!='model' or operation.get('model_id')!='bonsai-stock' or operation.get('role')!='calibration'
            or operation.get('prompt_sha256')!=hashlib.sha256(prompt.encode()).hexdigest()
            or operation.get('stage')!=str(PRIOR_ROOT/'calibration')
            or type(operation.get('deadline')) not in (int,float) or not math.isfinite(operation['deadline'])
            or operation['deadline']>prior.get('operation_deadline',0)
            or prior.get('cleanup_verified') is not True or prior.get('remaining_owned')!=[]
            or prior.get('coordination_release')!='verified_empty' or prior.get('coordination_retained') is not True
            or prior.get('errors')!=[] or prior.get('completed') is not False or prior.get('returncode')!=2
            or prior.get('child_final_status')!='blocked' or prior.get('stop_reason')!='child_exit'):
        raise ValueError('prior admission-only stop/cleanup not proven')
    rows=[json.loads(line) for line in _regular(PRIOR_ROOT/'calibration/loading-telemetry.jsonl',1048576).splitlines()]
    if len(rows)!=61 or any(row.get('owned')!=[] or row.get('unverified_members')!=[] or row.get('errors')!=[] for row in rows):
        raise ValueError('prior no-native telemetry absent')
    admission=json.loads(_regular(PRIOR_ROOT/'calibration/admission.json',1048576))
    if admission.get('reasons')!=['passive memory pressure']:raise ValueError('prior admission reason differs')
    from admission_policy import evaluate
    try:evaluate(admission['samples'])
    except ValueError as error:
        if str(error)!='substantial host paging during admission':raise ValueError('prior paging stop differs') from error
    else:raise ValueError('prior paging stop not reproduced from counters')
    final_deadline=prior['final_deadline']
    now=time.monotonic()
    if (type(final_deadline) not in (int,float) or not math.isfinite(final_deadline)
            or final_deadline!=original['final_deadline'] or prior['operation_deadline']!=final_deadline-180
            or not now+180<final_deadline<=now+2700):
        raise ValueError('original absolute scope deadline expired/invalid')
    verify_failed_native_attempt()
    ROOT.mkdir(mode=0o700)
    _active=_ResearchDemo('supervisor',final_deadline)
    return _active


def gate_supervisor(authority,**details):
    from . import run_authority
    if run_authority.is_authority(authority):return run_authority.gate_supervisor(authority,**details)
    value=_valid(authority,'supervisor');verify_integrity()
    if details or value.supervisor_started:raise ValueError('one fixed supervisor invocation only')
    value.supervisor_started=True


def activate_child(context):
    global _active
    from .deadline import SupervisorContext
    if type(context) is not SupervisorContext or _active is not None or getattr(context,'child_module',None)!='research.model_demo_entry':raise ValueError('trusted supervisor context required')
    context.validate();verify_integrity()
    if Path(context.output_root)!=ROOT or context.child_pid!=os.getpid():raise ValueError('fixed child output/identity required')
    _active=_ResearchDemo('child',context.operation_deadline,context)
    return _active


def reviewed_backend_seal(authority):
    from . import run_authority
    if run_authority.is_authority(authority):return run_authority.reviewed_backend_seal(authority)
    _valid(authority,'child');return verify_integrity()


def _stage(role,stage):
    stage=Path(stage)
    permitted={'generation':{'seed-1','seed-2'},'evolution':{'evolve-1'},
               'reflection':{'reflection-1','reflection-2','reflection-3'},
               'ranking':{'match-1-2','match-1-3','match-2-3'},
               'debate':{'match-%d-%d-debate-%d'%(a,b,t) for a,b in ((1,2),(1,3),(2,3)) for t in (1,2)},
               'meta-review':{'meta-initial','meta-final'},'overview':{'overview'}}
    if role=='calibration':
        if stage!=ROOT/'calibration':raise ValueError('fixed calibration stage required')
    elif role not in permitted or stage.parent!=ROOT/'workflow' or stage.name not in {'task-'+name for name in permitted[role]}:
        raise ValueError('fixed role and workflow task identity required')
    if stage.is_symlink():raise ValueError('stage symlink refused')
    return stage


def _workflow_reservation(role,prompt,stage):
    from .laboratory import LaboratoryStore
    state=LaboratoryStore(ROOT/'workflow').load()
    task_id=Path(stage).name.removeprefix('task-')
    task=next((t for t in state['tasks'] if t['id']==task_id),None)
    if (state['mode']!='real' or state['inflight']!=task_id or task is None or task['kind']!=role
            or task['status']!='reserved' or task.get('model_io',{}).get('prompt_sha256')!=hashlib.sha256(prompt.encode()).hexdigest()):
        raise ValueError('persisted exact model task/prompt reservation required')
    return state


def gate_model(authority,operation,*,model_id,role,prompt,stage,deadline,coordination,**details):
    from . import run_authority
    if run_authority.is_authority(authority):return run_authority.gate_model(authority,operation,model_id=model_id,role=role,prompt=prompt,stage=stage,deadline=deadline,coordination=coordination,**details)
    a=_valid(authority,'child')
    from model_profiles import profile_settings
    # Descriptor validates exact model and role, no caller-chosen context/argv.
    profile_settings('bonsai-research-v1',model_id,role)
    if coordination is None:raise ValueError('whole-window resource ownership required')
    coordination.validate()
    stage=_stage(role,stage)
    if type(deadline) not in (int,float) or not math.isfinite(deadline) or not time.monotonic()<deadline<=a.deadline:raise ValueError('bounded model deadline required')
    key={'kind':'model','model_id':model_id,'role':role,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'stage':str(stage),'deadline':deadline}
    if operation=='agent':
        verify_integrity()
        if a.current is not None or details or str(stage) in a.used_model_stages:raise ValueError('no overlapping/retried model stage')
        if role=='calibration':
            expected=(CHECKOUT/'research/proposed-demo/calibration_seed1_fixture.txt').read_text()
            if a.model_count or a.calibration_done or prompt!=expected:raise ValueError('one exact separate calibration only')
        else:
            if not a.calibration_done or a.model_count+PRIOR_NATIVE_ATTEMPTS>=19:raise ValueError('calibration first;19 total native attempts including prior failures')
            persisted=_workflow_reservation(role,prompt,stage)
            if persisted['model_calls']!=a.model_count:raise ValueError('persisted model count differs from scope')
        if deadline>time.monotonic()+120.01:raise ValueError('model stage exceeds120s')
        a.model_count+=1;a.used_model_stages.add(str(stage));a.current={**key,'used':{'agent'}}
    elif operation in ('backend','complete'):
        if a.current is None or any(a.current.get(k)!=v for k,v in key.items()) or operation in a.current['used']:raise ValueError('reserved model request identity changed')
        if operation=='backend':
            if details or a.current['used']!={'agent'}:raise ValueError('backend order changed')
            a.current['used'].add(operation)
        else:
            capture=details.get('capture',{});receipt=details.get('receipt',{})
            if set(details)!={'capture','receipt'} or a.current['used']!={'agent','backend','native'} or capture.get('owned_cleanup_empty') is not True or capture.get('postflight_passed') is not True or capture.get('returncode')!=0 or receipt.get('actual_inference') is not True:raise ValueError('natural model completion/release evidence absent')
            a.current=None
    else:raise ValueError('unknown model operation')


@contextmanager
def native_capability(authority,request_path,request,deadline):
    from . import run_authority
    if run_authority.is_authority(authority):
        with run_authority.native_capability(authority,request_path,request,deadline) as capability:yield capability
        return
    a=_valid(authority,'child');path=Path(request_path)
    if not a.current or a.current['kind']!='model' or a.current['used']!={'agent','backend'} or path!=Path(a.current['stage'])/'request.json' or deadline!=a.current['deadline']:raise ValueError('native request unreserved')
    from model_profiles import validate_model_request
    validate_model_request(request)
    if request['role']!=a.current['role'] or request['model_id']!=a.current['model_id']:raise ValueError('native model/role changed')
    if json.loads(_regular(path,32768))!=request or hashlib.sha256(request['prompt'].encode()).hexdigest()!=a.current['prompt_sha256']:raise ValueError('native request changed')
    payload={'schema':'PMI_SEALED_NATIVE_REQUEST_V1','controller':list(a.owner),'request_path':str(path),'request_sha256':digest(request),'deadline':deadline,'scope':'pmi-model-demo-v1'}
    fd=os.memfd_create('pmi-research-stage',os.MFD_CLOEXEC|os.MFD_ALLOW_SEALING)
    try:
        os.write(fd,json.dumps(payload,separators=(',',':')).encode());os.lseek(fd,0,os.SEEK_SET)
        seals=fcntl.F_SEAL_WRITE|fcntl.F_SEAL_GROW|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_SEAL
        fcntl.fcntl(fd,fcntl.F_ADD_SEALS,seals)
        a.current['used'].add('native')
        yield '/proc/%d/fd/%d'%(os.getpid(),fd)
    finally:os.close(fd)


def gate_native(request_path,request,*,capability_path):
    """Reopen a controller-owned sealed memfd through proc; never a disk token."""
    from . import run_authority
    if run_authority.is_runtime_capability_path(capability_path):return run_authority.gate_native(request_path,request,capability_path=capability_path)
    from model_profiles import validate_model_request
    validate_model_request(request)
    match=re.fullmatch(r'/proc/([1-9][0-9]*)/fd/([0-9]+)',str(capability_path))
    if match is None:raise ValueError('live controller capability required')
    controller=int(match[1]);before=identity(controller)[0]
    fd=os.open(capability_path,os.O_RDONLY|os.O_CLOEXEC)
    try:
        expected=fcntl.F_SEAL_WRITE|fcntl.F_SEAL_GROW|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_SEAL
        if fcntl.fcntl(fd,fcntl.F_GET_SEALS)&expected!=expected or not os.readlink('/proc/self/fd/%d'%fd).startswith('/memfd:pmi-research-stage'):raise ValueError('sealed native capability required')
        payload=json.loads(os.pread(fd,16385,0))
    finally:os.close(fd)
    if identity(controller)[0]!=before or payload!={'schema':'PMI_SEALED_NATIVE_REQUEST_V1','controller':list(before),'request_path':str(request_path),'request_sha256':digest(request),'deadline':payload.get('deadline'),'scope':'pmi-model-demo-v1'}:raise ValueError('native capability identity mismatch')
    deadline=payload['deadline']
    if type(deadline) not in (int,float) or not math.isfinite(deadline) or not time.monotonic()<deadline<=time.monotonic()+120:raise ValueError('expired/unbounded native capability')
    current=os.getpid();found=False
    for _ in range(32):
        process,parent=identity(current)
        if process==before:found=True;break
        if parent<=1:break
        current=parent
    if not found:raise ValueError('native process is not owned controller descendant')
    command=Path('/proc/%d/cmdline'%controller).read_bytes().split(b'\0')
    if b'research.model_demo_entry' not in command or b'--supervised-child' not in command:raise ValueError('fixed controller entry required')
    _stage(request['role'],Path(request_path).parent)
    if json.loads(_regular(request_path,32768))!=request:raise ValueError('native request file changed')
    verify_integrity()


def is_demo_authority(authority):
    from .run_authority import is_authority
    return type(authority) is _ResearchDemo or is_authority(authority)


def gate_cpu(authority,operation,**details):
    from . import run_authority
    if run_authority.is_authority(authority):return run_authority.gate_cpu(authority,operation,**details)
    a=_valid(authority,'child')
    if not a.calibration_done:raise ValueError('calibration required before generated CPU work')
    from .experiments import ExperimentRequest,CORRECTNESS_EVALUATOR
    from .executor import SandboxPlan,build_plan
    from .interval_evaluator import cases,validate_source
    if operation=='candidate':
        request=details['request'];source=details['source'];stage=Path(details['stage'])
        if a.current is not None or a.candidate_count>=3 or type(request) is not ExperimentRequest or request.candidate_id not in (1,2,3) or request.candidate_id in a.used_candidate_ids:raise ValueError('three nonoverlapping generated requests only')
        expected=ExperimentRequest('experiment-%d'%request.candidate_id,request.candidate_id,validate_source(source),digest(CORRECTNESS_EVALUATOR),'interval-union-correctness-v1','correctness-small-v1')
        if request!=expected or stage!=ROOT/'workflow'/('task-'+request.task_id):raise ValueError('generated correctness request changed')
        from .demo_prompts import validate_response
        validate_response('generation',source)
        from .laboratory import LaboratoryStore
        saved=LaboratoryStore(ROOT/'workflow').load()
        pending=next((t for t in saved['tasks'] if t['id']==request.task_id),None)
        candidate=saved['candidates'][request.candidate_id-1]
        if saved['mode']!='real' or saved['inflight']!=request.task_id or pending is None or pending['status']!='reserved' or pending['kind']!='experiment' or candidate['source']!=source or candidate['sha256']!=request.source_sha256 or saved['experiment_calls']!=a.candidate_count+1:raise ValueError('persisted exact candidate reservation required')
        a.candidate_count+=1;a.used_candidate_ids.add(request.candidate_id);a.current={'kind':'cpu','stage':str(stage),'request':request,'used':{'candidate'}}
    elif operation in ('admission','transport'):
        if not a.current or a.current['kind']!='cpu' or operation in a.current['used']:raise ValueError('CPU reservation/order missing')
        if operation=='transport' and 'admission' not in a.current['used']:raise ValueError('fresh admission required')
        plan=details['plan'];stage=Path(a.current['stage']);args=plan.argv
        if type(plan) is not SandboxPlan:raise ValueError('fixed CPU sandbox plan required')
        expected=build_plan(RUNTIME,stage/'candidate',stage/'artifacts',int(args[args.index('--seccomp')+1]),int(args[args.index('--json-status-fd')+1]),cases(),(plan.block_read_fd,plan.block_write_fd))
        if plan!=expected:raise ValueError('candidate sandbox/limits changed')
        a.current['used'].add(operation)
    else:raise ValueError('no packaging, authored-qualification reuse or arbitrary commands in demo scope')
