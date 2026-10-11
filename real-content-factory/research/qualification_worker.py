"""Fixed benign bootstrap evidence. No candidate files are read or executed."""
import errno
import json
import os
import resource
import socket
import sys

SCHEMA='PMI_CPU_BOOTSTRAP_V1'
EXPECTED_CHECKS=('fixed_request','resource_limits','core_limit','one_core_affinity',
                 'stdio_descriptors_only','seccomp_filter','no_new_privileges','no_capabilities',
                 'mount_permissions','private_filesystems','tmp_bound','scoped_root',
                 'work_directory','namespace_ids','socket_denied','fork_denied')
NAMESPACES=('user','pid','net','mnt','ipc','uts','cgroup')
LIMITS={'address_space':(resource.RLIMIT_AS,512*1048576),
        'cpu':(resource.RLIMIT_CPU,5),'file_size':(resource.RLIMIT_FSIZE,65536),
        'descriptors':(resource.RLIMIT_NOFILE,32)}


def denied_socket():
    try:probe=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    except OSError as error:return error.errno==errno.EPERM
    probe.close();return False


def denied_fork():
    try:pid=os.fork()
    except OSError as error:return error.errno==errno.EPERM
    # A broken filter must not leave a descendant, even on a failed probe.
    if pid==0:os._exit(99)
    os.waitpid(pid,0);return False


def collect_evidence():
    for kind,limit in LIMITS.values():resource.setrlimit(kind,(limit,limit))
    request=sys.stdin.buffer.read(4097)
    if len(request)>4096 or json.loads(request)!={'cases':[{'probe':'bootstrap'}]}:
        raise ValueError('fixed bootstrap request required')
    limits={name:list(resource.getrlimit(kind)) for name,(kind,limit) in LIMITS.items()}
    limits['core']=list(resource.getrlimit(resource.RLIMIT_CORE))
    with open('/proc/self/status') as file:
        status={key:value.strip() for key,value in (line.split(':',1) for line in file if ':' in line)}
    mounts={}
    with open('/proc/self/mountinfo') as file:
        for line in file:
            fields=line.split();separator=fields.index('-')
            mounts[fields[4]]={'flags':fields[5].split(','),'type':fields[separator+1]}
    required={'/':'ro','/candidate':'ro','/work':'rw','/proc':'rw','/dev':'rw','/tmp':'rw'}
    mount_evidence={path:mounts.get(path) for path in required}
    descriptors=[]
    for name in os.listdir('/proc/self/fd'):
        try:os.fstat(int(name))
        except OSError as error:
            if error.errno==errno.EBADF:continue
            raise
        descriptors.append(int(name))
    descriptors.sort()
    affinity=sorted(os.sched_getaffinity(0))
    namespaces={name:os.stat('/proc/self/ns/'+name).st_ino for name in NAMESPACES}
    tmp=os.statvfs('/tmp');tmp_bytes=tmp.f_blocks*tmp.f_frsize
    root_entries=sorted(os.listdir('/'))
    checks={
        'fixed_request':True,
        'resource_limits':all(limits[name]==[limit,limit] for name,(kind,limit) in LIMITS.items()),
        'core_limit':limits['core']==[1,1],
        'one_core_affinity':len(affinity)==1,
        'stdio_descriptors_only':descriptors==[0,1,2],
        'seccomp_filter':status.get('Seccomp')=='2',
        'no_new_privileges':status.get('NoNewPrivs')=='1',
        'no_capabilities':all(int(status.get(name,'-1'),16)==0 for name in ('CapInh','CapPrm','CapEff','CapBnd','CapAmb')),
        'mount_permissions':all(path in mounts and flag in mounts[path]['flags'] for path,flag in required.items()),
        'private_filesystems':all(mounts.get(path,{}).get('type')==kind for path,kind in (('/proc','proc'),('/dev','tmpfs'),('/tmp','tmpfs'))),
        'tmp_bound':0<tmp_bytes<=2*1048576,
        'scoped_root':set(root_entries)<=set(('usr','lib','lib64','worker.py','candidate','work','proc','dev','tmp')),
        'work_directory':os.getcwd()=='/work',
        'namespace_ids':all(value>0 for value in namespaces.values()),
        'socket_denied':denied_socket(),
        'fork_denied':denied_fork(),
    }
    if set(checks)!=set(EXPECTED_CHECKS):raise ValueError('bootstrap evidence schema changed')
    return {'schema':SCHEMA,'probe':'bootstrap','candidate_executed':False,
            'checks':checks,'observed':{'limits':limits,'descriptors':descriptors,'affinity':affinity,
            'namespaces':namespaces,'mounts':mount_evidence,'tmp_bytes':tmp_bytes,'root_entries':root_entries}}


def main():
    try:
        result=collect_evidence();code=0 if all(result['checks'].values()) else 1
    except Exception as error:
        result={'schema':SCHEMA,'probe':'bootstrap','candidate_executed':False,
                'checks':{'bootstrap_completed':False},'error':type(error).__name__};code=1
    encoded=json.dumps(result,sort_keys=True,separators=(',',':')).encode()
    if len(encoded)>65536:raise ValueError('bootstrap evidence bound')
    sys.stdout.buffer.write(encoded+b'\n')
    return code


if __name__=='__main__':sys.exit(main())
