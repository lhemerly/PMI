"""Optional read-only startup diagnostics. Never a guard policy or model runner."""
import os
from pathlib import Path
import re
import time
from linux_memory_telemetry import pressure, key_values, cgroup_location, membership


def zones(text):
    result = []
    for line in text.splitlines():
        match = re.match(r'Node (\d+), zone\s+(\S+)', line)
        if match:
            result.append({'node': int(match[1]), 'zone': match[2]})
            continue
        fields = line.split()
        if not result:
            continue
        if fields[:2] == ['pages','free'] and len(fields) == 3:
            result[-1]['free_pages'] = int(fields[2])
        elif len(fields) == 2 and fields[0] in ('min','low','high','boost','managed'):
            result[-1][fields[0] + '_pages'] = int(fields[1])
    return result


class LoaderDiagnostics:
    def __init__(self, proc=Path('/proc'), sys=Path('/sys'), clock=time.monotonic):
        self.proc, self.sys, self.clock = proc, sys, clock
        self.baseline, self.previous = {}, {}
        self.page_size = os.sysconf('SC_PAGE_SIZE')
        self.ticks_per_second = os.sysconf('SC_CLK_TCK')

    def sample(self, members=(), group=None):
        started = self.clock()
        unavailable = []

        def read(path, parse=lambda x: x.strip()):
            try:
                with path.open() as f:
                    text = f.read(262145)
                if len(text) > 262144:
                    raise ValueError('diagnostic source exceeds 256KiB')
                return parse(text)
            except (OSError,ValueError,KeyError,IndexError) as error:
                unavailable.append({'source':str(path),'error':type(error).__name__})
                return None

        def counter(key, value):
            if value is None:
                return {'value':None,'delta_from_baseline':None,'delta_from_previous':None,'reset':False}
            first = self.baseline.setdefault(key,value)
            prior = self.previous.get(key,value)
            reset = value < prior
            self.previous[key] = value
            if reset:
                self.baseline[key] = value
            return {'value':value,'delta_from_baseline':None if reset else value-first,
                    'delta_from_previous':None if reset else value-prior,'reset':reset}

        def psi(path):
            data = read(path,pressure)
            if data is not None:
                for name,row in data.items():
                    row['total_stall_us'] = counter((str(path),name),row['total_stall_us'])
            return data

        vm = read(self.proc/'vmstat',key_values)
        required = {'compact_stall','compact_success','compact_fail','compact_daemon_wake',
                    'thp_fault_alloc','thp_fault_fallback','thp_fault_fallback_charge',
                    'thp_collapse_alloc','thp_collapse_alloc_failed','allocstall_normal',
                    'pgscan_direct_throttle','pgfault','pgmajfault'}
        keys = required | {k for k in (vm or {}) if k.startswith(('compact_','thp_','allocstall_'))}
        counters = {k:counter(('vm',k),(vm or {}).get(k)) for k in sorted(keys)}
        owned = []
        mountinfo = read(self.proc/'self/mountinfo')
        threads_seen = 0
        for member in members:
            if not member.get('alive'):
                continue
            if self.clock()-started > 0.05:
                unavailable.append({'source':'owned processes','error':'50ms diagnostic budget'});break
            pid = member['pid'];directory = self.proc/str(pid)
            stat = read(directory/'stat',lambda x:x.rsplit(')',1)[1].split())
            if stat is None or len(stat)<20:
                continue
            if group is not None and (int(stat[2]) != group or int(stat[3]) != group):
                continue
            start_ticks = int(stat[19]);identity=(pid,start_ticks)
            row={'pid':pid,'start_ticks':start_ticks,'cgroup_attribution':'shared_or_unknown'}
            status = read(directory/'status',lambda x:dict(line.split(':',1) for line in x.splitlines() if ':' in line)) or {}
            row['status_bytes']={k:int(status[k].split()[0])*1024 if k in status else None for k in ('VmLck','VmRSS','RssAnon','RssFile','RssShmem','VmSwap')}
            maps = read(directory/'smaps_rollup',lambda x:dict(line.split(':',1) for line in x.splitlines() if ':' in line)) or {}
            row['mapping_bytes']={k:int(maps[k].split()[0])*1024 if k in maps else None for k in ('Rss','Anonymous','AnonHugePages','Locked')}
            member_path = read(directory/'cgroup',membership)
            row['cgroup_membership'] = member_path
            if member_path is not None and mountinfo is not None:
                try:
                    _,_,path=cgroup_location(member_path,mountinfo); cg=Path(path)
                    row['cgroup_memory_pressure']=psi(cg/'memory.pressure')
                    row['cgroup_io_pressure']=psi(cg/'io.pressure')
                    row['cgroup_events']=read(cg/'memory.events',key_values)
                except ValueError as e:
                    unavailable.append({'source':str(directory/'cgroup'),'error':str(e)})
            row['minor_faults'] = counter((identity,'minor'),int(stat[7]))
            row['major_faults'] = counter((identity,'major'),int(stat[9]))
            row['threads']=[]
            try:
                tasks=sorted((directory/'task').iterdir(),key=lambda p:int(p.name))
            except OSError:
                tasks=[]
            for task in tasks:
                if threads_seen >= 64 or self.clock()-started > 0.05:
                    unavailable.append({'source':'owned threads','error':'entry/time budget'});break
                ts=read(task/'stat',lambda x:x.rsplit(')',1)[1].split())
                if ts is None or len(ts)<20:continue
                threads_seen += 1
                row['threads'].append({'tid':int(task.name),'state':ts[0],
                    'wait_channel':read(task/'wchan'),
                    'block_io_delay_ticks':counter((identity,int(task.name),'blkio'),int(ts[39])) if len(ts)>39 else None})
            after=read(directory/'stat',lambda x:x.rsplit(')',1)[1].split())
            if after is not None and len(after)>19 and int(after[19])==start_ticks:
                owned.append(row)
        result = {'scope':'read_only_optional_diagnostics_not_guard_inputs','started_monotonic_seconds':started,
                'duration_seconds':self.clock()-started,'page_size':self.page_size,'clock_ticks_per_second':self.ticks_per_second,
                'vmstat':counters,'host_io_pressure':psi(self.proc/'pressure/io'),
                'zone_watermarks':read(self.proc/'zoneinfo',zones),'buddyinfo':read(self.proc/'buddyinfo'),
                'thp_policy':{n:read(self.sys/'kernel/mm/transparent_hugepage'/n) for n in ('enabled','defrag','hpage_pmd_size')},
                'delay_accounting':read(self.proc/'sys/kernel/task_delayacct'),
                'owned':owned,'unavailable':unavailable}
        result['duration_seconds'] = self.clock()-started
        return result
