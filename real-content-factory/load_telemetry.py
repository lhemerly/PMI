"""Diagnostic snapshots only: existing supervisors decide all stops and cleanup."""
import json
import os
from pathlib import Path
import time


def keyed(text):
    return {line.split()[0].rstrip(':'): int(line.split()[1])
            for line in text.splitlines() if len(line.split()) >= 2}


def pressure(text):
    return {line.split()[0]: {k: float(v) if k != 'total' else int(v)
                             for k, v in (word.split('=') for word in line.split()[1:])}
            for line in text.splitlines()}


class LoadTelemetry:
    def __init__(self, stage, proc=Path('/proc'), model_path=None):
        self.stage, self.proc = Path(stage), Path(proc)
        self.model_path = None if model_path is None else Path(model_path)
        self.direct_io_evidence = []
        self.last_time = None
        self.baseline = None
        self.owned_seen = {}
        self.output = (self.stage/'loading-telemetry.jsonl').open('x')

    def verified_descendant(self, pid, start_ticks, spawn):
        try:
            current = pid
            for _ in range(32):
                raw = (self.proc/str(current)/'stat').read_text()
                fields = raw[raw.rfind(')')+2:].split()
                if current == pid and int(fields[19]) != start_ticks:
                    return False
                if current == spawn['pid']:
                    return (int(fields[19]) == spawn['start_ticks'] and
                            os.readlink(self.proc/str(current)/'ns/pid') == spawn['pid_namespace'])
                current = int(fields[1])
                if current <= 1:
                    return False
        except (OSError, ValueError, IndexError, KeyError):
            pass
        return False

    def sample(self, force=False):
        now = time.monotonic()
        if not force and self.last_time is not None and now-self.last_time < .25:
            return
        row = {'monotonic_seconds':now, 'wall_ns':time.time_ns(),
               'interval_seconds':None if self.last_time is None else now-self.last_time,
               'scope':'host pressure includes other work; owned members require ancestry to original supervisor PID/start/namespace',
               'errors':[], 'owned':[], 'unverified_members':[]}
        def read(relative, parser):
            try:
                return parser((self.proc/relative).read_text())
            except (OSError, ValueError, IndexError) as error:
                row['errors'].append({'source':relative, 'error':type(error).__name__})
                return None
        row['PSI'] = {name:read('pressure/'+name, pressure) for name in ('cpu','io','memory')}
        vm = read('vmstat', keyed)
        row['host_counters'] = None if vm is None else {k:vm.get(k) for k in
            ('pgfault','pgmajfault','oom_kill','pswpin','pswpout','pgscan_direct','pgsteal_direct')}
        row['meminfo'] = read('meminfo', keyed)
        row['page_size_bytes'] = os.sysconf('SC_PAGE_SIZE')
        # Membership is diagnostic, never used to signal or kill processes.
        host = self.stage/'host.jsonl'
        if host.exists():
            try:
                with host.open('rb') as file:
                    file.seek(max(0,host.stat().st_size-262144))
                    lines = file.read(262144).splitlines()
                with host.open('rb') as file:
                    first_lines = file.read(262144).splitlines()
                spawn = next(json.loads(line) for line in first_lines if b'"event": "spawn"' in line)
                latest = next(json.loads(line) for line in reversed(lines)
                              if b'"event": "sample"' in line)
                row['supervisor_elapsed_seconds'] = latest['elapsed_seconds']
                for member in latest['members']:
                    if not member.get('alive'):
                        continue
                    pid = member['pid']
                    raw = (self.proc/str(pid)/'stat').read_text()
                    fields = raw[raw.rfind(')')+2:].split()
                    identity = (pid, int(fields[19]))
                    if not self.verified_descendant(pid, identity[1], spawn):
                        row['unverified_members'].append({'pid':pid,'observed_start_ticks':identity[1],
                            'reason':'cannot bind live ancestry to original supervisor PID/start/namespace'})
                        continue
                    counters = {'minor_faults':int(fields[7]), 'major_faults':int(fields[9]),
                                'CPU_ticks':int(fields[11])+int(fields[12])}
                    counters['io'] = read(str(pid)+'/io', keyed)
                    if self.model_path is not None:
                        try:
                            model_stat = self.model_path.stat()
                            for fd in (self.proc/str(pid)/'fd').iterdir():
                                if fd.resolve() != self.model_path.resolve():
                                    continue
                                file_stat = fd.stat()
                                info = (self.proc/str(pid)/'fdinfo'/fd.name).read_text()
                                flag_text = next(line.split()[1] for line in info.splitlines() if line.startswith('flags:'))
                                flags = int(flag_text, 8)
                                after = (self.proc/str(pid)/'stat').read_text()
                                after_fields = after[after.rfind(')')+2:].split()
                                if int(after_fields[19]) != identity[1]:
                                    raise ValueError('PID identity changed during fd observation')
                                if self.verified_descendant(pid,identity[1],spawn) and flags & os.O_DIRECT and (file_stat.st_dev,file_stat.st_ino) == (model_stat.st_dev,model_stat.st_ino):
                                    evidence = {'pid':pid,'start_ticks':identity[1], 'fd':fd.name,
                                                'flags_octal':flag_text,'O_DIRECT':True,
                                                'model_device':file_stat.st_dev,'model_inode':file_stat.st_ino,
                                                'monotonic_seconds':now}
                                    self.direct_io_evidence.append(evidence)
                                    row.setdefault('model_direct_io',[]).append(evidence)
                        except (OSError, ValueError, StopIteration) as error:
                            row['errors'].append({'source':'model_fd_flags','error':type(error).__name__})
                    if not self.verified_descendant(pid,identity[1],spawn):
                        row['unverified_members'].append({'pid':pid,'observed_start_ticks':identity[1],
                            'reason':'ancestry changed during observation'})
                        continue
                    base = self.owned_seen.setdefault(identity, counters)
                    row['owned'].append({'pid':pid,'start_ticks':identity[1], 'state':fields[0],
                        'counters':counters, 'delta_since_first_observation':
                        {k:counters[k]-base[k] for k in ('minor_faults','major_faults','CPU_ticks')}})
            except (OSError, ValueError, IndexError, KeyError, StopIteration) as error:
                row['errors'].append({'source':'supervisor_members','error':type(error).__name__})
        if self.baseline is None:
            self.baseline = row['host_counters']
        if row['host_counters'] is not None and self.baseline is not None:
            row['host_delta_from_baseline'] = {k:None if v is None or self.baseline[k] is None
                else v-self.baseline[k] for k,v in row['host_counters'].items()}
        self.output.write(json.dumps(row,separators=(',',':'))+'\n')
        self.output.flush()
        self.last_time = now
