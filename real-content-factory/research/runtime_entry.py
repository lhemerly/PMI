"""Reusable, reviewed local research runtime; configuration grants no authority.

Each launch owns an immutable attempt directory and an outer cleanup supervisor.
Completed workflow evidence survives a pause/resume. Uncertain reservations are
never replayed automatically. The supported executor remains interval correctness.
"""
from contextlib import contextmanager
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import sys
import time

from .harness import digest
from .laboratory import LaboratoryStore, configuration, research_configuration, summary

RUN_SCHEMA = 'PMI_RESEARCH_RUN_V1'
LAUNCH_SCHEMA = 'PMI_RESEARCH_LAUNCH_V1'


class _BoundaryStop(Exception):
    pass


def _read(path, maximum=1048576):
    from content_factory import bounded_read
    return json.loads(bounded_read(Path(path), maximum))


def _write(path, value, *, readonly=False):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        if readonly: os.fchmod(stream.fileno(), 0o400)
    parent = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try: os.fsync(parent)
    finally: os.close(parent)


def _directory(path):
    path = Path(path).absolute()
    if path.resolve() != path: raise ValueError('canonical run path required; symlinks refused')
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError('private current-user run directory required')
    return path


def _run(path):
    root = _directory(path)
    record = _read(root/'run.json', 65536)
    if set(record) != {'schema', 'config', 'config_sha256', 'created_utc'} or record['schema'] != RUN_SCHEMA:
        raise ValueError('research run manifest required')
    config = configuration(record['config'])
    if config.get('execution_profile') != 'bonsai-research-v1' or digest(config) != record['config_sha256']:
        raise ValueError('research run configuration identity changed')
    if (root/'run.json').stat().st_mode & 0o222: raise ValueError('immutable run manifest required')
    _directory(root/'attempts')
    return root, config


@contextmanager
def _launch_lock(root):
    path = root/'launch.lock'
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid():
            raise ValueError('invalid launch lock')
        try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise ValueError('research run already has an active launcher') from error
        current = path.lstat()
        if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino): raise ValueError('launch lock replaced')
        yield
    finally: os.close(fd)


def _attempts(root):
    values = sorted((root/'attempts').iterdir())
    if any(not p.name.isdigit() or len(p.name) != 4 for p in values): raise ValueError('unexpected attempt entry')
    for i, path in enumerate(values, 1):
        if path.name != f'{i:04d}': raise ValueError('attempt sequence changed')
        _directory(path)
    return values


def plan(config):
    config = configuration(config)
    if config.get('execution_profile') != 'bonsai-research-v1': raise ValueError('research execution profile required')
    from .laboratory import MODEL_ROLES
    population = config['initial_candidates'] + config['evolutions']
    pairs = population*(population-1)//2*config['options']['matches_per_pair']
    role_counts = {'generation':config['initial_candidates'], 'reflection':population,
                  'debate':pairs*(config['options']['debate_turns']-1), 'ranking':pairs,
                  'evolution':config['evolutions'], 'meta-review':config['evolutions']+1, 'overview':1}
    return {'mode':'offline research plan', 'config':config, 'config_sha256':digest(config),
            'expected_full_population_model_calls':sum(role_counts[k] for k in MODEL_ROLES if k in role_counts),
            'expected_distinct_candidate_checks':population, 'role_counts':role_counts,
            'calibration':'one separate model call per launch attempt;120 seconds',
            'execution_authorized_by_plan':False, 'objective':'884 frozen interval correctness cases',
            'claim_scope':'engineering discovery loop; no paper reproduction or research effectiveness claim'}


def status(run_dir):
    root, config = _run(run_dir)
    attempts = _attempts(root)
    value = {'run_dir':str(root), 'config_sha256':digest(config), 'attempt_count':len(attempts),
             'cancel_requested':(root/'cancel.json').exists(), 'pause_requested':(root/'pause.json').exists(),
             'workflow':None, 'latest_attempt':None}
    store = LaboratoryStore(root/'workflow')
    if store.path.exists(): value['workflow'] = summary(store.load())
    if attempts:
        latest = attempts[-1]
        value['latest_attempt'] = {'root':str(latest)}
        for name in ('supervisor-report.json', 'child-result.json', 'launch-error.json'):
            if (latest/name).exists(): value['latest_attempt'][name] = _read(latest/name)
    return value


def _request(run_dir, action):
    root, _ = _run(run_dir)
    path = root/(action+'.json')
    if not path.exists():
        try: _write(path, {'action':action, 'requested_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        except FileExistsError: pass
    if action == 'cancel':
        store = LaboratoryStore(root/'workflow')
        if store.path.is_file(): store.cancel()
    return status(root)


def cancel(run_dir): return _request(run_dir, 'cancel')
def pause(run_dir): return _request(run_dir, 'pause')


def _launch(root, config, *, resuming):
    from .run_authority import activate_for_entry
    from .deadline import run_demo_supervisor
    with _launch_lock(root):
        attempts = _attempts(root)
        if resuming:
            if not attempts: raise ValueError('no previous attempt to resume')
            previous = attempts[-1]
            report_path = previous/'supervisor-report.json'
            if not report_path.is_file() or _read(report_path).get('cleanup_verified') is not True:
                raise ValueError('previous independent supervisor cleanup is unproven')
        elif attempts: raise ValueError('start requires a fresh run')
        if (root/'cancel.json').exists(): raise ValueError('cancelled run cannot resume')
        store = LaboratoryStore(root/'workflow')
        existing = store.path.is_file()
        if existing:
            state = store.load()
            if state['config'] != config: raise ValueError('persisted workflow configuration changed')
            if state['inflight'] is not None or any(t['status']=='reserved' for t in state['tasks']):
                raise ValueError('uncertain reserved task; no automatic replay')
            if state['status'] in ('finished','cancelled','budget_exhausted','blocked'):
                raise ValueError('workflow is terminal or blocked; retained for inspection')
        if resuming and (root/'pause.json').exists(): (root/'pause.json').unlink()
        if len(attempts) >= 9999: raise ValueError('attempt sequence exhausted')
        attempt = root/'attempts'/f'{len(attempts)+1:04d}'
        attempt.mkdir(mode=0o700)
        launch = {'schema':LAUNCH_SCHEMA, 'root':str(attempt), 'workflow_root':str(root/'workflow'),
                  'config':config, 'config_sha256':digest(config), 'resume':existing}
        _write(attempt/'launch.json', launch, readonly=True)
        try:
            authority = activate_for_entry(attempt, config, resume=existing)
            outcome = run_demo_supervisor(authority=authority)
        except Exception as error:
            _write(attempt/'launch-error.json', {'error':type(error).__name__+': '+str(error),
                    'cleanup_claim':False, 'automatic_retry':False})
            raise
        return {'run_dir':str(root), 'attempt_dir':str(attempt), 'supervisor':outcome, 'state':status(root)}


def start(config, out):
    config = configuration(config)
    plan(config)
    root = Path(out).absolute()
    if root.resolve() != root or not root.parent.is_dir(): raise ValueError('existing canonical output parent required')
    root.mkdir(mode=0o700)
    (root/'attempts').mkdir(mode=0o700)
    _write(root/'run.json', {'schema':RUN_SCHEMA,'config':config,'config_sha256':digest(config),
                           'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, readonly=True)
    return _launch(root, config, resuming=False)


def resume(run_dir):
    root, config = _run(run_dir)
    return _launch(root, config, resuming=True)


class RuntimeCorrectnessRunner:
    actual_execution = True

    def __init__(self, authority, coordination):
        from .executor import BubblewrapCorrectnessRunner, verify_runtime
        from .owned_cpu import CPUAdmission, CPUOwnedTransport
        from .demo_authority import RUNTIME, RUNTIME_MANIFEST, hashed
        from .run_authority import verify_integrity
        saved = _read(RUNTIME_MANIFEST)
        if set(saved) != {'root','files','runtime_sha256','primitive_sha256','packaging'} or Path(saved['root']) != RUNTIME:
            raise ValueError('previous qualified minimal runtime manifest required')
        if verify_runtime(RUNTIME, saved['files']) != saved['runtime_sha256']:
            raise ValueError('qualified minimal runtime changed')
        if verify_integrity().get('external_artifacts_sha256',{}).get(str(RUNTIME_MANIFEST)) != hashed(RUNTIME_MANIFEST):
            raise ValueError('qualified runtime manifest pin changed')
        self.authority = authority
        self.runner = BubblewrapCorrectnessRunner(RUNTIME, saved['files'], saved['primitive_sha256'],
            admission=CPUAdmission(authority=authority,coordination=coordination),
            transport=CPUOwnedTransport(authority=authority),authority=authority)

    def run(self, request, source, stage, deadline):
        now = time.monotonic()
        if (type(deadline) not in (int,float) or not math.isfinite(deadline)
                or not now < deadline <= min(self.authority.operation_deadline, now+60)):
            raise ValueError('CPU correctness stage requires bounded operation deadline')
        receipt = self.runner.run(request, source, stage, deadline)
        self.authority.cpu_complete(receipt)
        return receipt


def _completed_receipts(roots):
    completed = []; experiments = []
    for root in roots:
        if not root.is_dir(): continue
        for path in root.glob('**/result.json'):
            row = _read(path)
            if row.get('actual_inference') is True and row.get('status') == 'RealTextAndReleasePass': completed.append(str(path))
        for path in root.glob('**/experiment-receipt.json'):
            row = _read(path)
            if row.get('actual_execution') is True and row.get('cleanup_verified') is True: experiments.append(str(path))
    return completed, experiments


def child_main():
    from .deadline import read_supervisor_context
    from .coordination import CoordinationOwner, FACTORY_LEASE
    from .run_authority import activate_child
    from .demo_authority import CHECKOUT
    from .harness import DemoRealAgent
    from .laboratory import drive
    report = {'scope':'pmi-research-runtime-v1', 'status':'blocked', 'calibration':None, 'workflow':None,
              'actual_model_completions':0, 'generated_candidate_receipts':0, 'no_performance_measurement':True,
              'claim_scope':'engineering discovery loop; no paper reproduction or research effectiveness claim'}
    authority = None
    with read_supervisor_context() as context:
        attempt = Path(context.output_root)
        try:
            with CoordinationOwner() as coordination:
                context.retain_coordination(coordination)
                authority = activate_child(context)
                if (authority.workflow_root.parent/'pause.json').exists():
                    report['status'] = 'paused'
                    raise _BoundaryStop()
                agent = DemoRealAgent(authority=authority, coordination=coordination)
                runner = RuntimeCorrectnessRunner(authority, coordination)
                prompt = (CHECKOUT/'research/proposed-demo/calibration_seed1_fixture.txt').read_text()
                result = agent.call('calibration', 'bonsai-stock', prompt, attempt/'calibration',
                                    min(context.operation_deadline, time.monotonic()+120))
                authority.calibrated()
                report['calibration'] = {k:v for k,v in result.items() if k != 'text'}
                store = LaboratoryStore(authority.workflow_root)
                if not store.path.exists(): store.create(authority.config)
                root = authority.workflow_root.parent
                def should_pause():
                    if (root/'cancel.json').exists():
                        store.cancel()
                        return True
                    return (root/'pause.json').exists()
                state = drive(store, agent, runner, FACTORY_LEASE, coordination=coordination, should_pause=should_pause)
                report['workflow'] = summary(state)
                report['status'] = state['status']
                if state['status'] == 'ready': report['status'] = 'paused'
                elif state['status'] == 'finished':
                    report['status'] = 'completed' if report['workflow'].get('demonstration',{}).get('completed_real_demonstration') is True else 'finished_without_full_coverage'
        except _BoundaryStop:
            if authority is not None and LaboratoryStore(authority.workflow_root).path.is_file():
                report['workflow'] = summary(LaboratoryStore(authority.workflow_root).load())
        except BaseException as error:
            report['error'] = type(error).__name__+': '+str(error)
            if authority is not None and LaboratoryStore(authority.workflow_root).path.is_file():
                try: report['workflow'] = summary(LaboratoryStore(authority.workflow_root).load())
                except Exception as read_error: report['state_read_error'] = type(read_error).__name__+': '+str(read_error)
        finally:
            if authority is not None:
                try:
                    completed, experiments = _completed_receipts([attempt, authority.workflow_root])
                    report.update(actual_model_completions=len(completed), generated_candidate_receipts=len(experiments),
                                  model_completion_receipts=completed, candidate_receipts=experiments,
                                  workflow_model_stages_reserved=authority.model_count,
                                  candidate_stages_reserved=authority.candidate_count,
                                  calibration_stages_reserved=authority.calibration_count,
                                  uncertain_operation=None if authority.current is None else {
                                      k:v for k,v in authority.current.items() if k not in ('request','used')})
                except Exception as error: report['receipt_scan_error'] = type(error).__name__+': '+str(error)
            _write(attempt/'child-result.json', report)
            context.finish('finished' if report['status'] in ('completed','paused','finished_without_full_coverage') else
                           report['status'] if report['status'] in ('blocked','budget_exhausted','cancelled') else 'blocked')
    return 0 if report['status'] in ('completed','paused','finished_without_full_coverage') else 2


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    if values == ['--supervised-child']: return child_main()
    if values and values[0].startswith('research-mode-'): values[0] = values[0][len('research-mode-'):]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan','start','status','resume','cancel','pause'))
    goal = parser.add_mutually_exclusive_group()
    goal.add_argument('--goal'); goal.add_argument('--goal-file', type=Path); goal.add_argument('--spec-file', type=Path)
    parser.add_argument('--out', type=Path); parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--initial-candidates', type=int); parser.add_argument('--evolutions', type=int)
    parser.add_argument('--model-calls', type=int); parser.add_argument('--experiment-calls', type=int)
    parser.add_argument('--wall-seconds', type=int)
    args = parser.parse_args(values)
    settings = ('initial_candidates','evolutions','model_calls','experiment_calls','wall_seconds')
    if args.action in ('plan','start'):
        if args.run_dir is not None: parser.error('--run-dir applies to existing runs')
        if args.action == 'plan' and args.out is not None: parser.error('plan prints to stdout; --out applies to start')
        if args.spec_file is not None:
            if any(getattr(args,k) is not None for k in settings): parser.error('spec-file is immutable; do not combine configuration overrides')
            config = configuration(_read(args.spec_file,65536))
        else:
            from content_factory import bounded_read
            goal_text = bounded_read(args.goal_file,8192).decode() if args.goal_file is not None else args.goal
            budget = {k:getattr(args,k) for k in ('model_calls','experiment_calls','wall_seconds') if getattr(args,k) is not None}
            values = {k:getattr(args,k) for k in ('initial_candidates','evolutions') if getattr(args,k) is not None}
            config = research_configuration(goal_text,budget=budget,**values)
        if args.action == 'plan': result = plan(config)
        else:
            if args.out is None: parser.error('--out required')
            result = start(config,args.out)
    else:
        if args.run_dir is None: parser.error('--run-dir required')
        if args.out is not None or args.goal is not None or args.goal_file is not None or args.spec_file is not None or any(getattr(args,k) is not None for k in settings):
            parser.error('existing run configuration is immutable')
        result = {'status':status,'resume':resume,'cancel':cancel,'pause':pause}[args.action](args.run_dir)
    print(json.dumps(result,indent=2,allow_nan=False))
    if args.action in ('start','resume'): return 0 if result['supervisor'].get('completed') is True else 2
    return 0


if __name__ == '__main__': raise SystemExit(main())
