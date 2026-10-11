"""Owned CPU transport and admission. Every production entry retains the gate."""
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import selectors
import signal
import subprocess
import time

LOCK=Path('/tmp/cpu-coordinator-6t3yfB/test.lock')
NAMESPACES=('user','pid','net','mnt','ipc','uts','cgroup')
LAUNCHER_NOFILE=64


def process_birth(pid):
    raw=(Path('/proc')/str(pid)/'stat').read_text()
    prefix,tail=raw.rsplit(')',1);fields=tail.split()
    if int(prefix.split(' ',1)[0])!=pid:raise ValueError('process stat identity changed')
    ticks=int(fields[19])
    if ticks<0:raise ValueError('invalid process birth time')
    return pid,ticks


def fingerprint(pid):
    return (*process_birth(pid),os.readlink(Path('/proc')/str(pid)/'ns/pid'))


def available():
    fields={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    memory=int(fields['MemAvailable'].split()[0])*1024
    pressure={line.split()[0]:dict(part.split('=') for part in line.split()[1:]) for line in Path('/proc/pressure/memory').read_text().splitlines()}
    counters=dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    swap_used=(int(fields['SwapTotal'].split()[0])-int(fields['SwapFree'].split()[0]))*1024
    return {'swap_used':swap_used,'swapout':int(counters['pswpout'])*os.sysconf('SC_PAGE_SIZE'),'time':time.monotonic(),'available':memory,'some':int(pressure['some']['total']),'full':int(pressure['full']['total'])}


def require_samples(samples):
    if len(samples)<3 or samples[-1]['time']-samples[0]['time']<2:raise ValueError('CPU admission coverage missing')
    for row in samples:
        if type(row['time']) not in (int,float) or not math.isfinite(row['time']) or any(type(row[key]) is not int or row[key]<0 for key in ('available','some','full','swap_used','swapout')):raise ValueError('invalid CPU admission source')
        if row['available']<16*1024**3:raise ValueError('CPU admission RAM below16GiB')
    for a,b in zip(samples,samples[1:]):
        delta=b['time']-a['time']
        if not 0<delta<=1 or b['some']<a['some'] or b['full']<a['full']:raise ValueError('CPU admission continuity failed')
        if b['swapout']<a['swapout'] or b['swapout']-a['swapout']>delta*1048576:raise ValueError('CPU admission paging')
        if b['some']-a['some']>delta*250000 or b['full']-a['full']>delta*125000:raise ValueError('CPU admission sustained memory pressure')


class CPUAdmission:
    def __init__(self,*,authority=None,coordination=None):
        self.lock=None;self.authority=authority;self.coordination=coordination;self._borrow=None
    def qualify(self,plan,deadline):
        from .qualification_authority import gate_cpu
        gate_cpu(self.authority,'admission',plan=plan)
        if self.lock is not None or self._borrow is not None:raise RuntimeError('CPU admission already active')
        if self.coordination is None:self.lock=LOCK.open('r+')
        else:
            from .coordination import CoordinationOwner
            if type(self.coordination) is not CoordinationOwner:raise ValueError('trusted coordination owner required')
            borrowed=self.coordination.borrow_cpu()
            self.lock=borrowed.__enter__();self._borrow=borrowed
        try:
            if self.coordination is None:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            samples=[];start=time.monotonic()
            while True:
                if time.monotonic()+plan.wall_seconds+3>=deadline:raise TimeoutError('CPU admission deadline')
                samples.append(available())
                if samples[-1]['time']-start>=2:break
                time.sleep(.25)
            require_samples(samples)
            return {'approved':True,'exclusive':True,'expires_at':min(deadline,time.monotonic()+1),'samples':samples,'parent':fingerprint(os.getpid())}
        except BaseException:self.release();raise
    def release(self):
        if self._borrow is not None:
            borrowed=self._borrow;self._borrow=None;self.lock=None;borrowed.__exit__(None,None,None)
        elif self.lock is not None:self.lock.close();self.lock=None


def namespace_rows(fd):
    size=os.fstat(fd).st_size
    if size>65536:raise ValueError('namespace status bound exceeded')
    raw=os.pread(fd,size,0)
    # --json-status-fd uses JSONL. Only newline-terminated records are committed;
    # the final partial write may even end with a nested object's closing brace.
    lines=raw.split(b'\n')[:-1]
    rows=[json.loads(line) for line in lines if line.strip()]
    if any(not isinstance(row,dict) for row in rows):raise ValueError('namespace status object required')
    return rows


def namespace_members(identity,birth_floor=None):
    if birth_floor is not None and (type(birth_floor) is not int or birth_floor<0):raise ValueError('invalid namespace birth bound')
    result=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            birth=None
            if birth_floor is not None:
                birth=process_birth(int(path.name))
                if birth[1]<birth_floor:
                    # PID namespace membership cannot change after creation.
                    # Strictly older tasks predate the verified owned launcher
                    # that creates this fresh namespace; equal ticks stay in scope.
                    if process_birth(int(path.name))!=birth:raise ValueError('process identity changed during namespace exclusion')
                    continue
            if (path/'ns/pid').stat().st_ino!=identity:continue
            stamp=fingerprint(int(path.name))
            if birth is not None and stamp[:2]!=birth:raise ValueError('process identity changed during namespace scan')
            if stamp[2]!='pid:['+str(identity)+']':continue
            status=(path/'status').read_text()
            if fingerprint(int(path.name))!=stamp:raise ValueError('sandbox member identity changed during status read')
            rss=next((int(line.split()[1])*1024 for line in status.splitlines() if line.startswith('VmRSS:')),0)
            seccomp=next((int(line.split()[1]) for line in status.splitlines() if line.startswith('Seccomp:')),0)
            result.append({'identity':stamp,'rss':rss,'seccomp':seccomp})
        except FileNotFoundError:continue
        except ProcessLookupError:continue
        except PermissionError:
            # A dumpability change can root-own an owned task's /proc entry.
            # For the bounded scan, only proven older birth is an exclusion.
            if birth_floor is not None:raise
            try:owner=path.stat().st_uid
            except (FileNotFoundError,ProcessLookupError):continue
            if owner==os.getuid():raise
            continue
    return result


def check_namespace(row,parent_ids):
    if type(row.get('child-pid')) is not int or row['child-pid']<=0:raise ValueError('sandbox child identity missing')
    # Bubblewrap 0.12 deliberately omits user namespace from its JSON status.
    # pin_namespace independently checks the live child user namespace.
    for name in NAMESPACES:
        if name=='user':continue
        value=row.get(name+'-namespace')
        if type(value) is not int or value<=0 or value==parent_ids[name]:raise ValueError('sandbox namespace missing/shared: '+name)
    return row['pid-namespace']


def pin_namespace(row,parent_ids):
    identity=check_namespace(row,parent_ids)
    before=fingerprint(row['child-pid'])
    if before[2]!='pid:['+str(identity)+']':raise ValueError('sandbox child namespace changed')
    descriptor=os.open('/proc/'+str(row['child-pid'])+'/ns/pid',os.O_RDONLY|os.O_CLOEXEC)
    try:
        if os.fstat(descriptor).st_ino!=identity:raise ValueError('sandbox namespace identity changed')
        user_identity=(Path('/proc')/str(row['child-pid'])/'ns/user').stat().st_ino
        if user_identity<=0 or user_identity==parent_ids['user']:raise ValueError('sandbox user namespace missing/shared')
        if fingerprint(row['child-pid'])!=before:raise ValueError('sandbox child identity changed')
    except BaseException:os.close(descriptor);raise
    # Retain this handle through cleanup to prevent namespace inode reuse.
    return identity,descriptor


def kill_member(member):
    identity=member['identity']
    try:
        if fingerprint(identity[0])==identity:os.kill(identity[0],signal.SIGKILL)
    except (FileNotFoundError,ProcessLookupError):pass


def child_limits(plan):
    # The existing kernel/core policy is preserved for this owned child only.
    from content_factory import STOCK
    import importlib.util
    path=STOCK.parent/'candidate-qualification/core_limit_exec.py'
    spec=importlib.util.spec_from_file_location('owned_core_limits',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from llama_backend import CORE_PATTERN
    module.configure(CORE_PATTERN,'7.2.6-1-cachyos')
    resource.setrlimit(resource.RLIMIT_AS,(plan.as_bytes,plan.as_bytes))
    resource.setrlimit(resource.RLIMIT_CPU,(plan.cpu_seconds,plan.cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE,(65536,65536))
    # Bubblewrap setup needs its own descriptor headroom. The fixed worker
    # lowers this to 32 after Python has loaded; never change controller limits.
    resource.setrlimit(resource.RLIMIT_NOFILE,(LAUNCHER_NOFILE,LAUNCHER_NOFILE))
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})


class CPUOwnedTransport:
    def __init__(self,*,authority=None):self.last_cleanup=None;self.authority=authority
    def run(self,plan,deadline):
        self.last_cleanup=None
        from .qualification_authority import gate_cpu
        gate_cpu(self.authority,'transport',plan=plan)
        from .executor import seccomp_x86_64
        seccomp_fd=int(plan.argv[plan.argv.index('--seccomp')+1]);status_fd=int(plan.argv[plan.argv.index('--json-status-fd')+1])
        inherited_hard=resource.getrlimit(resource.RLIMIT_NOFILE)[1]
        if inherited_hard!=resource.RLIM_INFINITY and inherited_hard<LAUNCHER_NOFILE:raise ValueError('inherited descriptor hard limit below launcher requirement')
        passed=(seccomp_fd,status_fd)+(() if plan.block_read_fd is None else (plan.block_read_fd,))
        if any(fd>=LAUNCHER_NOFILE for fd in passed):raise ValueError('launcher descriptor outside bounded range')
        if '--unshare-pid' not in plan.argv or '--pidns' in plan.argv:raise ValueError('fresh owned PID namespace required')
        if os.pread(seccomp_fd,4096,0)!=seccomp_x86_64():raise ValueError('filter FD identity changed')
        parent_ids={name:(Path('/proc/self/ns')/name).stat().st_ino for name in NAMESPACES}
        ancestor=fingerprint(os.getppid());cutoff=min(deadline,time.monotonic()+plan.wall_seconds)
        baseline=available()
        process=subprocess.Popen(plan.argv,cwd=plan.cwd,env=plan.environment,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                 shell=False,close_fds=True,pass_fds=passed,start_new_session=True,preexec_fn=lambda:child_limits(plan))
        streams={'stdout':bytearray(),'stderr':bytearray()};selector=None;written=0;ns=None;namespace_fd=None;birth_floor=None;error=None;timed_out=False;exitcode=None
        try:
            launcher=fingerprint(process.pid)
            if launcher[0]!=process.pid or type(launcher[1]) is not int or launcher[1]<0 or fingerprint(process.pid)!=launcher:raise ValueError('owned launcher identity changed')
            birth_floor=launcher[1]
            selector=selectors.DefaultSelector()
            for stream,name,events in ((process.stdout,'stdout',selectors.EVENT_READ),(process.stderr,'stderr',selectors.EVENT_READ),(process.stdin,'stdin',selectors.EVENT_WRITE)):
                os.set_blocking(stream.fileno(),False);selector.register(stream,events,name)
            while process.poll() is None or selector.get_map():
                if time.monotonic()>=cutoff:timed_out=True;raise TimeoutError('owned CPU wall deadline')
                if fingerprint(ancestor[0])!=ancestor:raise RuntimeError('owned parent identity lost')
                rows=namespace_rows(status_fd)
                start=next((row for row in rows if 'child-pid' in row),None)
                if start is not None and ns is None:
                    ns,namespace_fd=pin_namespace(start,parent_ids)
                    if plan.block_write_fd is not None:os.write(plan.block_write_fd,b'1')
                if ns is not None:
                    members=namespace_members(ns,birth_floor)
                    if sum(member['rss'] for member in members)>plan.rss_bytes:raise MemoryError('owned CPU RSS limit')
                sample=available()
                if sample['available']<12*1024**3:raise MemoryError('owned CPU RAM floor below12GiB')
                if any(sample[key]<baseline[key] for key in ('some','full','swapout')):raise ValueError('owned CPU pressure counters regressed')
                if sample['swap_used']-baseline['swap_used']>512*1048576 or sample['swapout']-baseline['swapout']>256*1048576 or sample['some']-baseline['some']>500000 or sample['full']-baseline['full']>250000:raise MemoryError('owned CPU host pressure limit')
                for key,events in selector.select(.025):
                    stream=key.fileobj;name=key.data
                    if name=='stdin':
                        try:amount=os.write(stream.fileno(),plan.stdin[written:written+8192]);written+=amount
                        except BlockingIOError:continue
                        except BrokenPipeError:written=len(plan.stdin)
                        if written>=len(plan.stdin):selector.unregister(stream);stream.close()
                    else:
                        value=os.read(stream.fileno(),8192)
                        if not value:selector.unregister(stream);stream.close()
                        else:
                            streams[name].extend(value)
                            if len(streams[name])>getattr(plan,name+'_limit'):raise ValueError(name+' bound exceeded')
            exitcode=process.wait(timeout=max(.001,cutoff-time.monotonic()))
            rows=namespace_rows(status_fd)
            start=next((row for row in rows if 'child-pid' in row),None)
            if start is None:raise ValueError('sandbox setup/namespace evidence absent')
            if ns is None:ns,namespace_fd=pin_namespace(start,parent_ids)
            if check_namespace(start,parent_ids)!=ns:raise ValueError('sandbox namespace changed')
            ends=[row['exit-code'] for row in rows if 'exit-code' in row]
            if ends!=[exitcode]:raise ValueError('sandbox exit evidence inconsistent')
        except BaseException as failure:error=failure
        finally:
            self.last_cleanup={'launcher_reaped':False,'namespace_verified':ns is not None,'remaining_pids':None,'cleanup_verified':False}
            try:
                if selector is not None:selector.close()
                for stream in (process.stdin,process.stdout,process.stderr):
                    if not stream.closed:stream.close()
                # Signal only unreaped owned launcher and fingerprinted members
                # of the pinned namespace; never a global PID selector.
                if process.poll() is None:process.kill()
                try:process.wait(timeout=2);self.last_cleanup['launcher_reaped']=True
                except subprocess.TimeoutExpired:error=error or RuntimeError('owned launcher cleanup unproven')
                if ns is not None:
                    for member in namespace_members(ns,birth_floor):kill_member(member)
                    stop=time.monotonic()+2
                    while namespace_members(ns,birth_floor) and time.monotonic()<stop:time.sleep(.025)
                    remaining=namespace_members(ns,birth_floor)
                    self.last_cleanup['remaining_pids']=[member['identity'][0] for member in remaining]
                    if remaining:error=error or RuntimeError('sandbox descendant cleanup incomplete')
                    self.last_cleanup['cleanup_verified']=not remaining and self.last_cleanup['launcher_reaped']
            except BaseException as failure:error=error or failure
            finally:
                if namespace_fd is not None:os.close(namespace_fd)
        if error is not None:raise error
        return {'exitcode':exitcode,'stdout':bytes(streams['stdout']),'stderr':bytes(streams['stderr']),
                'namespace_isolation':True,'seccomp_verified':True,'resource_limits_verified':True,
                'remaining_pids':[],'cleanup_verified':True,'timed_out':timed_out}
