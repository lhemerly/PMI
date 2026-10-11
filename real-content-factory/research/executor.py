"""Disabled Bubblewrap correctness executor; plans and protocols are testable.

Production plumbing exists, but execution authority remains held. Generic execution
cannot be enabled by a configuration flag or candidate output.
"""
from dataclasses import dataclass
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import time

from .experiments import ExperimentHeld
from .interval_evaluator import validate_source, cases, evaluate_outputs, case_identity

COMMAND='interval-union-correctness-v1'
PROFILE='correctness-small-v1'
RUNTIME_MOUNTPOINTS=frozenset(('candidate','work','proc','dev','tmp'))


def seccomp_x86_64():
    # Linux x86_64 cBPF: kill wrong architecture, EPERM fork/clone/namespaces,
    # networking, ptrace, privileged mounts, kernel interfaces and io_uring.
    # This is an explicit inferred policy requiring reviewed ABI/compat testing.
    deny=(41,53,56,57,58,101,155,165,166,169,246,248,249,250,272,298,304,308,310,311,320,321,323,425,426,427,435)
    rows=[(0x20,0,0,4),(0x15,1,0,0xc000003e),(0x06,0,0,0x80000000),(0x20,0,0,0)]
    # x32 syscall ABI uses the same audit architecture: reject it explicitly.
    rows.extend([(0x35,0,1,0x40000000),(0x06,0,0,0x80000000)])
    for number in deny:rows.extend([(0x15,0,1,number),(0x06,0,0,0x00050001)])
    rows.append((0x06,0,0,0x7fff0000))
    return b''.join(struct.pack('HBBI',*row) for row in rows)


@dataclass(frozen=True)
class SandboxPlan:
    argv: tuple
    environment: dict
    cwd: str
    stdin: bytes
    stdout_limit: int=65536
    stderr_limit: int=65536
    wall_seconds: int=10
    cpu_seconds: int=5
    rss_bytes: int=256*1048576
    as_bytes: int=512*1048576
    artifact_bytes: int=2*1048576
    block_read_fd: int|None=None
    block_write_fd: int|None=None


def build_plan(runtime_root,candidate_dir,artifact_dir,seccomp_fd,status_fd,values,handshake=None):
    paths=[Path(value) for value in (runtime_root,candidate_dir,artifact_dir)]
    if any(not p.is_absolute() or '..' in p.parts for p in paths) or len(set(paths))!=3:
        raise ValueError('distinct absolute scoped mounts required')
    if any(a==b or a in b.parents or b in a.parents for a,b in __import__('itertools').combinations(paths,2)):
        raise ValueError('sandbox mounts must not overlap')
    if any(type(fd) is not int or fd<3 for fd in (seccomp_fd,status_fd)) or seccomp_fd==status_fd:raise ValueError('dedicated filter/status descriptors required')
    payload=json.dumps({'cases':values},separators=(',',':')).encode()
    if len(payload)>262144:raise ValueError('case input bound')
    argv=('/usr/bin/bwrap','--unshare-user','--unshare-pid','--unshare-net','--unshare-ipc','--unshare-uts',
          '--unshare-cgroup','--cap-drop','ALL','--new-session','--die-with-parent',
          '--clearenv','--setenv','LANG','C.UTF-8','--setenv','LC_ALL','C.UTF-8',
          '--ro-bind',str(paths[0]),'/','--ro-bind',str(paths[1]),'/candidate',
          '--bind',str(paths[2]),'/work','--proc','/proc','--dev','/dev',
          '--size',str(2*1048576),'--tmpfs','/tmp','--chdir','/work',
          '--seccomp',str(seccomp_fd),'--json-status-fd',str(status_fd),'--',
          '/usr/bin/python3','-I','-B','/worker.py')
    if handshake is not None:
        if len(handshake)!=2 or any(type(fd) is not int or fd<3 for fd in handshake) or len(set((*handshake,seccomp_fd,status_fd)))!=4:raise ValueError('dedicated handshake descriptors required')
        index=argv.index('--');argv=argv[:index]+('--block-fd',str(handshake[0]))+argv[index:]
    return SandboxPlan(argv,{'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','LC_ALL':'C.UTF-8'},str(paths[2]),payload,block_read_fd=None if handshake is None else handshake[0],block_write_fd=None if handshake is None else handshake[1])


def verify_runtime(root,manifest,worker="interval"):
    if worker not in ("interval","qualification"):raise ValueError("unknown fixed worker role")
    root=Path(root)
    if root.is_symlink() or not root.is_absolute() or not root.is_dir():raise ValueError('frozen runtime root required')
    if not isinstance(manifest,dict) or not manifest:raise ValueError('runtime inventory missing')
    if stat.S_IMODE(root.stat().st_mode)!=0o500:raise ValueError('runtime root permissions changed')
    expected_dirs=set(RUNTIME_MOUNTPOINTS)
    for name in manifest:
        if not isinstance(name,str) or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('invalid runtime inventory path')
        expected_dirs.update(parent.as_posix() for parent in Path(name).parents if parent!=Path('.'))
    found={};directories=set()
    for path in root.rglob('*'):
        info=path.lstat();relative=path.relative_to(root).as_posix()
        if stat.S_ISDIR(info.st_mode):
            if stat.S_IMODE(info.st_mode)!=0o500:raise ValueError('runtime directory permissions changed')
            directories.add(relative);continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:raise ValueError('runtime special/symlink/hardlink refused')
        if relative.split('/')[0] not in ('usr','lib','lib64') and relative!='worker.py':raise ValueError('unexpected runtime content')
        data=path.read_bytes()
        expected_mode=0o500 if data.startswith(b'\x7fELF') or relative=='usr/bin/python3' else 0o400
        if stat.S_IMODE(info.st_mode)!=expected_mode:raise ValueError('runtime file permissions changed')
        found[relative]=hashlib.sha256(data).hexdigest()
    if directories!=expected_dirs:raise ValueError('runtime directory inventory changed')
    if found!=manifest or not {'usr/bin/python3','worker.py'}<=set(found):raise ValueError('runtime inventory changed')
    if found['worker.py']!=hashlib.sha256(Path(__file__).with_name('interval_worker.py' if worker=='interval' else 'qualification_worker.py').read_bytes()).hexdigest():raise ValueError('worker identity changed')
    return hashlib.sha256(json.dumps(found,sort_keys=True).encode()).hexdigest()


def collect_artifacts(root):
    root=Path(root)
    if root.is_symlink() or not root.is_dir(): raise ValueError('artifact root must be an owned regular directory')
    result=[];total=0
    for path in sorted(root.rglob('*')):
        info=path.lstat()
        if stat.S_ISDIR(info.st_mode):continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:raise ValueError('artifact special/symlink/hardlink refused')
        name=path.relative_to(root).as_posix()
        descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(descriptor,'rb') as file:
            before=os.fstat(file.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1:raise ValueError('artifact identity changed')
            total+=before.st_size
            if total>2*1048576 or len(result)>=16:raise ValueError('artifact bounds exceeded')
            data=file.read(before.st_size+1);after=os.fstat(file.fileno())
            if len(data)!=before.st_size or (before.st_ino,before.st_dev,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_dev,after.st_size,after.st_mtime_ns):raise ValueError('artifact changed during collection')
        result.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    return result


from .owned_cpu import CPUAdmission as AdmissionProvider, CPUOwnedTransport as OwnedTransport


def verify_capture(capture):
    keys={'exitcode','stdout','stderr','namespace_isolation','seccomp_verified','resource_limits_verified','remaining_pids','cleanup_verified','timed_out'}
    if not isinstance(capture,dict) or set(capture)!=keys or type(capture['exitcode']) is not int:raise ValueError('invalid sandbox capture')
    if any(capture[flag] is not True for flag in ('namespace_isolation','seccomp_verified','resource_limits_verified','cleanup_verified')) or capture['remaining_pids']!=[] or capture['timed_out'] is not False:
        raise ValueError('sandbox confinement/resources/cleanup unproven')
    for stream in ('stdout','stderr'):
        if not isinstance(capture[stream],bytes) or len(capture[stream])>65536:raise ValueError('sandbox stream bound')
    return capture


class BubblewrapCorrectnessRunner:
    actual_execution=True

    def __init__(self,runtime_root=None,runtime_manifest=None,primitive_sha256=None,admission=None,transport=None,*,authority=None):
        self.runtime_root=runtime_root;self.runtime_manifest=runtime_manifest;self.primitive_sha256=primitive_sha256
        self.authority=authority
        self.admission=admission or AdmissionProvider(authority=authority);self.transport=transport or OwnedTransport(authority=authority)

    @classmethod
    def from_manifest(cls,path,*,authority=None):
        from content_factory import bounded_read
        saved=json.loads(bounded_read(Path(path),1048576))
        if not isinstance(saved,dict) or set(saved)!={'root','files','runtime_sha256','primitive_sha256','packaging'}:raise ValueError('explicit runtime manifest schema required')
        if verify_runtime(saved['root'],saved['files'])!=saved['runtime_sha256']:raise ValueError('runtime manifest identity changed')
        return cls(saved['root'],saved['files'],saved['primitive_sha256'],authority=authority)

    def run(self,request,source,stage,deadline):
        from .qualification_authority import gate_cpu
        gate_cpu(self.authority,'candidate',request=request,source=source,stage=stage)
        if request.command_id!=COMMAND or request.profile!=PROFILE:raise ValueError('correctness command/profile only; benchmarks held')
        from .harness import digest
        from .experiments import CORRECTNESS_EVALUATOR
        if request.evaluator_sha256!=digest(CORRECTNESS_EVALUATOR) or validate_source(source)!=request.source_sha256:raise ValueError('source/evaluator identity changed')
        runtime_sha=verify_runtime(self.runtime_root,self.runtime_manifest)
        if hashlib.sha256(Path('/usr/bin/bwrap').read_bytes()).hexdigest()!=self.primitive_sha256:raise ValueError('Bubblewrap primitive pin missing/changed')
        if time.monotonic()>=deadline:raise TimeoutError('correctness deadline exhausted')
        stage=Path(stage);stage.mkdir(mode=0o700)
        candidate=stage/'candidate';candidate.mkdir(mode=0o700)
        (candidate/'source.py').write_text(source);(candidate/'source.py').chmod(0o400)
        artifacts=stage/'artifacts';artifacts.mkdir(mode=0o700)
        filter_path=stage/'seccomp.bpf';filter_path.write_bytes(seccomp_x86_64())
        status_path=stage/'namespace-status.jsonl'
        values=cases()
        # The future qualified supervisor owns FD lifetime and closes all other
        # inherited handles; no candidate-controlled descriptors/argv exist.
        with filter_path.open('rb') as seccomp, status_path.open('x+b') as status:
            handshake=os.pipe()
            try:
                plan=build_plan(Path(self.runtime_root),candidate.resolve(),artifacts.resolve(),seccomp.fileno(),status.fileno(),values,handshake)
                try:
                    admission=self.admission.qualify(plan,deadline)
                    if not isinstance(admission,dict) or admission.get('approved') is not True or admission.get('exclusive') is not True or type(admission.get('expires_at')) not in (int,float) or not time.monotonic()<admission['expires_at']<=deadline:raise ValueError('fresh CPU admission refused')
                    capture=verify_capture(self.transport.run(plan,deadline))
                finally:
                    release=getattr(self.admission,'release',None)
                    if release is not None:release()
            finally:
                for descriptor in handshake:os.close(descriptor)
        stdout=capture['stdout'].decode('utf-8',errors='strict');stderr=capture['stderr'].decode('utf-8',errors='strict')
        outputs=json.loads(stdout) if capture['exitcode']==0 else {'outputs':[]}
        if not isinstance(outputs,dict) or set(outputs)!={'outputs'}:raise ValueError('candidate protocol malformed')
        evaluation=evaluate_outputs(values,outputs['outputs']) if capture['exitcode']==0 else {'tests_passed':0,'tests_total':len(values),'failures':[{'exitcode':capture['exitcode']}],'cases_sha256':case_identity(values),'correct':False,'claim_scope':'observed bounded cases; no proof for all inputs'}
        receipt={'schema':'PMI_EXPERIMENT_RECEIPT_V1','request':request.payload(),'actual_execution':True,
                 'confinement_verified':True,'admission_verified':True,'cleanup_verified':True,'exitcode':capture['exitcode'],
                 'stdout':stdout,'stderr':stderr,'artifacts':collect_artifacts(artifacts),
                 'runtime_sha256':runtime_sha,'primitive_sha256':self.primitive_sha256,'seccomp_sha256':hashlib.sha256(seccomp_x86_64()).hexdigest(),
                 'measurements':{'tests_passed':evaluation['tests_passed'],'tests_total':evaluation['tests_total'],'baseline_ns':None,'candidate_ns':None,'baseline_sha256':None},
                 'correctness_details':evaluation}
        (stage/'experiment-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return receipt


def run_once(source_path,manifest_path,out,lease,*,authority=None):
    from .qualification_authority import gate_cpu
    gate_cpu(authority,'once',source_path=source_path,manifest_path=manifest_path,out=out,lease=lease)
    from content_factory import bounded_read
    from .harness import digest,ownership
    from .experiments import CORRECTNESS_EVALUATOR,ExperimentRequest,validate_receipt
    import sqlite3
    source=bounded_read(Path(source_path),4096).decode('utf-8');identity=validate_source(source)
    runner=BubblewrapCorrectnessRunner.from_manifest(manifest_path,authority=authority)
    root=Path(out);root.mkdir(mode=0o700)
    holder=type('RunRoot',(),{'root':root})()
    request=ExperimentRequest('correctness-demo-001',1,identity,digest(CORRECTNESS_EVALUATOR),COMMAND,PROFILE)
    with ownership(holder,lease),closing(sqlite3.connect(root/'experiment.sqlite3')) as db:
        db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE run (id INTEGER PRIMARY KEY CHECK(id=1), request TEXT NOT NULL, status TEXT NOT NULL, result TEXT)')
        db.execute('INSERT INTO run VALUES (1,?, ?,NULL)',(json.dumps(request.payload(),sort_keys=True),'reserved'));db.commit()
        try:
            result=runner.run(request,source,root/'stage',time.monotonic()+120)
            evaluation=validate_receipt(result,request,True)
            saved={'receipt':result,'evaluation':evaluation}
            db.execute('UPDATE run SET status=?,result=? WHERE id=1',('finished',json.dumps(saved,allow_nan=False)));db.commit()
            return saved
        except BaseException as error:
            db.execute('UPDATE run SET status=?,result=? WHERE id=1',('blocked',json.dumps({'error':type(error).__name__+': '+str(error),'retry_allowed':False})));db.commit()
            raise


def main(argv):
    import argparse
    parser=argparse.ArgumentParser(description='Gated correctness-only experiment; no model or benchmark')
    parser.add_argument('action',choices=['research-experiment-plan','research-experiment-start','research-experiment-status'])
    parser.add_argument('--source-file',type=Path);parser.add_argument('--runtime-manifest',type=Path);parser.add_argument('--out',type=Path);parser.add_argument('--lease',type=Path);parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args(argv)
    from content_factory import bounded_read
    if args.action=='research-experiment-status':
        if args.run_dir is None:parser.error('--run-dir required')
        import sqlite3
        path=args.run_dir/'experiment.sqlite3'
        if path.is_symlink() or not path.is_file():raise ValueError('experiment database missing/unsafe')
        with closing(sqlite3.connect('file:'+str(path.resolve())+'?mode=ro',uri=True)) as db:row=db.execute('SELECT request,status,result FROM run WHERE id=1').fetchone()
        print(json.dumps({'request':json.loads(row[0]),'status':row[1],'result':None if row[2] is None else json.loads(row[2])},indent=2));return 0
    if args.source_file is None:parser.error('--source-file required')
    if args.action=='research-experiment-plan':
        identity=validate_source(bounded_read(args.source_file,4096).decode('utf-8'))
        print(json.dumps({'command_id':COMMAND,'profile':PROFILE,'source_sha256':identity,'cases':len(cases()),'cases_sha256':case_identity(cases()),'model_calls':0,'performance_measurement':False,'execution_held':True},indent=2));return 0
    if any(value is None for value in (args.runtime_manifest,args.out,args.lease)):parser.error('--runtime-manifest, --out and --lease required')
    from launch_policy import PlatformHeld
    try:result=run_once(args.source_file,args.runtime_manifest,args.out,args.lease)
    except (PlatformHeld,ExperimentHeld) as error:print(json.dumps({'status':'blocked','reason':str(error)}));return 2
    print(json.dumps(result,indent=2));return 0 if result['evaluation']['correct'] else 3
