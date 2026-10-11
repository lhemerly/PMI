"""Process-local authority for the reviewed reusable research runtime.

The seal and launch record establish integrity and request identity, never
platform approval. Every launch still goes through normal platform review.
Only the fixed Bonsai profile and qualified interval correctness runner are
supported. This boundary does not protect against arbitrary same-user host code.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time

from .demo_authority import identity, digest, hashed

CHECKOUT = Path('/home/luizh/Documents/Codex/2026-10-08/task-8/real-content-factory')
SEAL = CHECKOUT / 'research/RESEARCH-INPUT-SEAL.json'
QUALIFICATION = CHECKOUT / 'research/FIRST-QUALIFICATION-VERIFICATION.json'
QUALIFICATION_SHA256 = '823d6f708dceab39fe409fcfac0c139e6a0dca9e5017a2237385f1d5b9151388'
GENERAL_GATE_SHA256 = '3dacd214fb5679ec602da28ee7ae357a7d29b9c448268af3c9319fcb6d779f34'
RUNTIME = Path('/tmp/pmi-correctness-runtime-v2')
RUNTIME_MANIFEST = Path('/tmp/pmi-first-correctness-v2/interval-manifest.json')
SCOPE = 'pmi-research-runtime-v1'
CHILD_MODULE = 'research.runtime_entry'
CALIBRATION_SECONDS = 120
CLEANUP_SECONDS = 180
_active = None


def _regular(path, limit, *, immutable=False):
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size > limit or (immutable and before.st_mode & 0o222)):
            raise ValueError('bounded regular immutable artifact required')
        raw = bytearray()
        while len(raw) <= limit:
            chunk = os.read(descriptor, min(65536, limit + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        stamp = lambda value: (value.st_dev, value.st_ino, value.st_size,
                               value.st_mtime_ns, value.st_ctime_ns)
        if len(raw) > limit or stamp(before) != stamp(after) or len(raw) != before.st_size:
            raise ValueError('artifact changed while reading')
        return bytes(raw)
    finally:
        os.close(descriptor)


def _path(value):
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts or path.resolve() != path:
        raise ValueError('canonical absolute path without symlinks required')
    return path


def verify_integrity():
    if (hashed(QUALIFICATION) != QUALIFICATION_SHA256
            or hashed(CHECKOUT / 'launch_policy.py') != GENERAL_GATE_SHA256):
        raise ValueError('prior CPU qualification or general launch gate changed')
    pin = json.loads(_regular(SEAL, 1048576))
    if pin.get('schema') != 'PMI_REAL_BACKEND_INTEGRITY_V1' or pin.get('scope_id') != SCOPE:
        raise ValueError('separately reviewed research integrity seal required')
    expected = {str(path.relative_to(CHECKOUT)) for path in CHECKOUT.glob('*.py')
                if not path.name.startswith('test_')}
    expected |= {str(path.relative_to(CHECKOUT)) for path in (CHECKOUT / 'research').glob('*.py')}
    if set(pin.get('source_sha256', {})) != expected:
        raise ValueError('complete research runtime source inventory required')
    for name, expected_hash in pin['source_sha256'].items():
        if hashlib.sha256(_regular(CHECKOUT / name, 2 * 1048576)).hexdigest() != expected_hash:
            raise ValueError('reviewed research source changed: ' + name)
    artifacts = pin.get('scope_artifacts_sha256', {})
    required = {'research/FIRST-QUALIFICATION-VERIFICATION.json',
                'research/proposed-demo/calibration_seed1_fixture.txt'}
    if not required <= set(artifacts):
        raise ValueError('required research artifacts missing from seal')
    for name, expected_hash in artifacts.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('invalid sealed artifact path')
        if hashlib.sha256(_regular(CHECKOUT / name, 2 * 1048576)).hexdigest() != expected_hash:
            raise ValueError('reviewed research artifact changed')
    if pin.get('external_artifacts_sha256', {}).get(str(RUNTIME_MANIFEST)) != hashed(RUNTIME_MANIFEST):
        raise ValueError('qualified CPU runtime manifest changed')
    from model_profiles import prompt_family_policy
    if pin.get('models', {}).get('bonsai-stock', {}).get('prompt_family') != prompt_family_policy():
        raise ValueError('reviewed model prompt family changed')
    return pin


def _configuration(value):
    from .laboratory import configuration
    from .experiments import CORRECTNESS_EVALUATOR
    validated = configuration(value)
    if (digest(validated) != digest(value) or validated.get('execution_profile') != 'bonsai-research-v1'
            or validated.get('model_id') != 'bonsai-stock'
            or validated.get('evaluator') != CORRECTNESS_EVALUATOR):
        raise ValueError('reviewed Bonsai correctness research configuration required')
    budget = validated['budget']
    for key in ('model_calls', 'experiment_calls', 'wall_seconds', 'stage_seconds', 'experiment_stage_seconds'):
        if type(budget.get(key)) is not int or budget[key] <= 0:
            raise ValueError('finite positive research budgets required')
    if budget['stage_seconds'] > 120 or budget['experiment_stage_seconds'] > 60:
        raise ValueError('fixed model and CPU stage limits cannot expand')
    return validated


def _launch(root):
    root = _path(root)
    info = root.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('private owned attempt directory required')
    raw = _regular(root / 'launch.json', 1048576, immutable=True)
    row = json.loads(raw)
    if not isinstance(row, dict) or set(row) != {
            'schema', 'root', 'workflow_root', 'config', 'config_sha256', 'resume'}:
        raise ValueError('exact research launch record required')
    workflow = _path(row['workflow_root'])
    if (row['schema'] != 'PMI_RESEARCH_LAUNCH_V1' or row['root'] != str(root)
            or type(row['resume']) is not bool or workflow == root
            or workflow in root.parents or root in workflow.parents):
        raise ValueError('separate attempt/workflow roots required')
    config = _configuration(row['config'])
    if row['config_sha256'] != digest(config):
        raise ValueError('launch configuration digest changed')
    return row, hashlib.sha256(raw).hexdigest()


def _state(workflow):
    from .laboratory import LaboratoryStore
    return LaboratoryStore(workflow).load()


def _require_not_cancelled(workflow):
    marker = Path(workflow).parent / 'cancel.json'
    if marker.exists() or marker.is_symlink():
        raise ValueError('research run cancellation requested')


def _baseline(row):
    workflow = Path(row['workflow_root'])
    _require_not_cancelled(workflow)
    if not workflow.exists():
        if row['resume']:
            raise ValueError('resume requires an existing validated workflow')
        return 0, 0, 0.0, set(), set()
    if not row['resume']:
        raise ValueError('existing workflow requires explicit resume')
    state = _state(workflow)
    if (digest(state['config']) != row['config_sha256'] or state['mode'] not in (None, 'real')
            or state['inflight'] is not None or any(task['status'] == 'reserved' for task in state['tasks'])
            or state['status'] in ('blocked', 'finished', 'cancelled') or state['cancel_requested']):
        raise ValueError('resume cannot change configuration or retry uncertain/completed work')
    return (state['model_calls'], state['experiment_calls'], state['elapsed_seconds'],
            {task['id'] for task in state['tasks'] if task['kind'] != 'experiment' and task['status'] == 'done'},
            {task['id'] for task in state['tasks'] if task['kind'] == 'experiment' and task['status'] == 'done'})


class ResearchAuthority:
    def __init__(self, kind, row, launch_sha256, final_deadline, operation_deadline, baseline, context=None):
        self.kind = kind
        self.root = Path(row['root'])
        self.workflow_root = Path(row['workflow_root'])
        self.config = row['config']
        self.launch_sha256 = launch_sha256
        self.deadline = final_deadline
        self.operation_deadline = operation_deadline
        self.child_module = CHILD_MODULE
        self.owner = identity(os.getpid())[0]
        self.context = context
        self.model_count, self.candidate_count, self.elapsed_seconds, used_models, used_cpu = baseline
        self.used_model_tasks = set(used_models)
        self.used_cpu_tasks = set(used_cpu)
        self.calibration_count = 0
        self.calibration_done = False
        self.current = None
        self.supervisor_started = False

    def calibrated(self):
        _valid(self, 'child')
        if self.calibration_count != 1 or self.current is not None or self.calibration_done:
            raise ValueError('one completed calibration required')
        from .demo_prompts import validate_response
        validate_response('generation', _regular(self.root / 'calibration/answer.txt', 4096).decode())
        self.calibration_done = True

    def cpu_complete(self, receipt):
        _valid(self, 'child')
        if not self.current or self.current['kind'] != 'cpu' or self.current['used'] != {'candidate', 'admission', 'transport'}:
            raise ValueError('complete owned CPU request required')
        from .experiments import validate_receipt
        validate_receipt(receipt, self.current['request'], True)
        self.current = None


def is_authority(value):
    return type(value) is ResearchAuthority


def _valid(value, kind=None):
    if not is_authority(value) or value is not _active or value.owner != identity(os.getpid())[0]:
        raise ValueError('active process-local research authority required')
    if kind is not None and value.kind != kind:
        raise ValueError('wrong research authority role')
    if value.context is not None:
        value.context.validate()
    if not math.isfinite(value.operation_deadline) or time.monotonic() >= value.operation_deadline:
        raise TimeoutError('research operation deadline exhausted')
    row, launch_hash = _launch(value.root)
    if (launch_hash != value.launch_sha256 or row['config'] != value.config
            or Path(row['workflow_root']) != value.workflow_root):
        raise ValueError('immutable launch identity changed')
    return value


def activate_for_entry(root, config, *, resume=False):
    global _active
    if _active is not None:
        raise ValueError('one process-local research launch only')
    verify_integrity()
    row, launch_hash = _launch(root)
    if type(resume) is not bool or row['resume'] is not resume or digest(_configuration(config)) != row['config_sha256']:
        raise ValueError('requested launch identity differs from immutable record')
    baseline = _baseline(row)
    remaining = row['config']['budget']['wall_seconds'] - baseline[2]
    if remaining <= 0 or baseline[0] >= row['config']['budget']['model_calls']:
        raise ValueError('persisted research budget exhausted')
    # A process crash cannot reuse the same attempt directory and calibration.
    with (Path(root) / 'authority-reserved.json').open('x') as stream:
        json.dump({'launch_sha256': launch_hash, 'owner': list(identity(os.getpid())[0])}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    operation_deadline = time.monotonic() + remaining + CALIBRATION_SECONDS
    _active = ResearchAuthority('supervisor', row, launch_hash,
                               operation_deadline + CLEANUP_SECONDS, operation_deadline, baseline)
    return _active


def activate_child(context):
    global _active
    from .deadline import SupervisorContext
    if type(context) is not SupervisorContext or _active is not None:
        raise ValueError('trusted unconsumed research supervisor context required')
    context.validate()
    verify_integrity()
    row, launch_hash = _launch(context.output_root)
    if context.child_pid != os.getpid() or context.child_module != CHILD_MODULE:
        raise ValueError('supervised child identity differs')
    baseline = _baseline(row)
    remaining = row['config']['budget']['wall_seconds'] - baseline[2]
    now = time.monotonic()
    if (not now < context.operation_deadline <= now + remaining + CALIBRATION_SECONDS
            or context.final_deadline != context.operation_deadline + CLEANUP_SECONDS):
        raise ValueError('bounded research supervisor deadlines required')
    reservation = json.loads(_regular(Path(context.output_root) / 'authority-reserved.json', 4096))
    if (set(reservation) != {'launch_sha256', 'owner'} or reservation.get('launch_sha256') != launch_hash
            or reservation['owner'] != list(context.supervisor_identity)):
        raise ValueError('parent launch reservation changed')
    _active = ResearchAuthority('child', row, launch_hash, context.final_deadline,
                               context.operation_deadline, baseline, context)
    return _active


def gate_supervisor(authority, **details):
    value = _valid(authority, 'supervisor')
    _require_not_cancelled(value.workflow_root)
    verify_integrity()
    if details or value.supervisor_started:
        raise ValueError('one fixed research supervisor invocation only')
    value.supervisor_started = True


def reviewed_backend_seal(authority):
    _valid(authority, 'child')
    return verify_integrity()


def _stage(root, workflow, role, stage):
    stage = _path(stage)
    if role == 'calibration':
        if stage != root / 'calibration':
            raise ValueError('fixed per-attempt calibration stage required')
    elif (stage.parent != workflow or not stage.name.startswith('task-')
          or not re.fullmatch(r'[a-z][a-z0-9-]{0,100}', stage.name[5:])):
        raise ValueError('persistent workflow task stage required')
    return stage


def _reservation(value, stage, kind, prompt=None):
    state = _state(value.workflow_root)
    task_id = stage.name[5:]
    task = next((task for task in state['tasks'] if task['id'] == task_id), None)
    if (digest(state['config']) != digest(value.config) or state['mode'] != 'real'
            or state['inflight'] != task_id or task is None or task['kind'] != kind
            or task['status'] != 'reserved'):
        raise ValueError('exact persisted research task reservation required')
    if prompt is not None and task.get('model_io', {}).get('prompt_sha256') != hashlib.sha256(prompt.encode()).hexdigest():
        raise ValueError('reserved research prompt changed')
    return state, task


def gate_model(authority, operation, *, model_id, role, prompt, stage, deadline, coordination, **details):
    value = _valid(authority, 'child')
    if operation != 'complete':
        _require_not_cancelled(value.workflow_root)
    from model_profiles import model_request
    request = model_request(model_id, prompt, 'bonsai-research-v1', role)
    if coordination is None:
        raise ValueError('whole-window resource coordination required')
    coordination.validate()
    stage = _stage(value.root, value.workflow_root, role, stage)
    if (type(deadline) not in (int, float) or not math.isfinite(deadline)
            or not time.monotonic() < deadline <= value.operation_deadline
            or deadline > time.monotonic() + min(120, value.config['budget']['stage_seconds']) + .01):
        raise ValueError('bounded research model deadline required')
    key = {'kind': 'model', 'request_sha256': digest(request), 'stage': str(stage), 'deadline': deadline}
    if operation == 'agent':
        verify_integrity()
        if value.current is not None or details:
            raise ValueError('overlapping or unrecognized model reservation')
        if role == 'calibration':
            if (value.calibration_count or prompt != _regular(CHECKOUT / 'research/proposed-demo/calibration_seed1_fixture.txt', 32768).decode()
                    or stage.exists()):
                raise ValueError('one exact new calibration per attempt required')
            value.calibration_count = 1
        else:
            state, task = _reservation(value, stage, role, prompt)
            if (not value.calibration_done or task['id'] in value.used_model_tasks
                    or state['model_calls'] != value.model_count + 1
                    or state['model_calls'] > value.config['budget']['model_calls'] or stage.exists()):
                raise ValueError('workflow model budget/order/retry refused')
            value.model_count = state['model_calls']
            value.used_model_tasks.add(task['id'])
        value.current = {**key, 'used': {'agent'}}
    elif operation == 'backend':
        if (details or value.current is None or any(value.current.get(k) != v for k, v in key.items())
                or value.current['used'] != {'agent'}):
            raise ValueError('reserved model request/backend order changed')
        value.current['used'].add('backend')
    elif operation == 'complete':
        capture, receipt = details.get('capture', {}), details.get('receipt', {})
        if (set(details) != {'capture', 'receipt'} or value.current is None
                or any(value.current.get(k) != v for k, v in key.items())
                or value.current['used'] != {'agent', 'backend', 'native'}
                or capture.get('owned_cleanup_empty') is not True or capture.get('postflight_passed') is not True
                or type(capture.get('returncode')) is not int or capture['returncode'] != 0
                or receipt.get('actual_inference') is not True):
            raise ValueError('natural model completion and release required')
        value.current = None
    else:
        raise ValueError('unknown research model operation')


@contextmanager
def native_capability(authority, request_path, request, deadline):
    value = _valid(authority, 'child')
    _require_not_cancelled(value.workflow_root)
    from model_profiles import validate_model_request
    validate_model_request(request)
    path = _path(request_path)
    if (not value.current or value.current['kind'] != 'model' or value.current['used'] != {'agent', 'backend'}
            or path != Path(value.current['stage']) / 'request.json' or deadline != value.current['deadline']
            or digest(request) != value.current['request_sha256']
            or json.loads(_regular(path, 32768)) != request):
        raise ValueError('exact reserved native research request required')
    payload = {'schema': 'PMI_SEALED_NATIVE_REQUEST_V2', 'scope': SCOPE,
               'controller': list(value.owner), 'root': str(value.root),
               'workflow_root': str(value.workflow_root), 'launch_sha256': value.launch_sha256,
               'source_seal_sha256': hashlib.sha256(_regular(SEAL, 1048576)).hexdigest(),
               'request_path': str(path), 'request_sha256': digest(request),
               'deadline': deadline, 'operation_deadline': value.operation_deadline}
    fd = os.memfd_create('pmi-research-runtime-stage', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        raw = json.dumps(payload, separators=(',', ':')).encode()
        if os.write(fd, raw) != len(raw):
            raise OSError('short native capability write')
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
        value.current['used'].add('native')
        yield '/proc/%d/fd/%d' % (os.getpid(), fd)
    finally:
        os.close(fd)


def _read_capability(path):
    match = re.fullmatch(r'/proc/([1-9][0-9]*)/fd/([0-9]+)', str(path))
    if match is None:
        raise ValueError('live controller memfd capability required')
    controller = identity(int(match[1]))[0]
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        expected = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
        if (fcntl.fcntl(fd, fcntl.F_GET_SEALS) & expected != expected
                or not os.readlink('/proc/self/fd/%d' % fd).startswith('/memfd:pmi-research-')):
            raise ValueError('sealed research memfd required')
        payload = json.loads(os.pread(fd, 16385, 0))
    finally:
        os.close(fd)
    if identity(controller[0])[0] != controller:
        raise ValueError('native capability controller identity changed')
    return controller, payload


def capability_schema(path):
    return _read_capability(path)[1].get('schema')


def is_runtime_capability_path(path):
    """Route by memfd name only; gate_native still verifies all authority."""
    if re.fullmatch(r'/proc/([1-9][0-9]*)/fd/([0-9]+)', str(path)) is None:
        return False
    try:
        return os.readlink(path).startswith('/memfd:pmi-research-runtime-stage ')
    except OSError:
        return False


def gate_native(request_path, request, *, capability_path):
    from model_profiles import validate_model_request
    validate_model_request(request)
    controller, payload = _read_capability(capability_path)
    keys = {'schema', 'scope', 'controller', 'root', 'workflow_root', 'launch_sha256',
            'source_seal_sha256', 'request_path', 'request_sha256', 'deadline', 'operation_deadline'}
    if (not isinstance(payload, dict) or set(payload) != keys
            or payload['schema'] != 'PMI_SEALED_NATIVE_REQUEST_V2' or payload['scope'] != SCOPE
            or payload['controller'] != list(controller) or payload['request_path'] != str(request_path)
            or payload['request_sha256'] != digest(request)):
        raise ValueError('exact native research capability identity required')
    deadline, operation_deadline = payload['deadline'], payload['operation_deadline']
    if (any(type(value) not in (int, float) or not math.isfinite(value) for value in (deadline, operation_deadline))
            or not time.monotonic() < deadline <= min(operation_deadline, time.monotonic() + 120)):
        raise ValueError('expired or unbounded native research capability')
    current = os.getpid()
    for _ in range(32):
        process, parent = identity(current)
        if process == controller:
            break
        if parent <= 1:
            raise ValueError('native process is not a controller descendant')
        current = parent
    else:
        raise ValueError('native controller ancestry not proven')
    command = Path('/proc/%d/cmdline' % controller[0]).read_bytes().split(b'\0')
    if [b'-m', CHILD_MODULE.encode(), b'--supervised-child'] not in [command[i:i+3] for i in range(len(command)-2)]:
        raise ValueError('fixed supervised research controller required')
    row, launch_hash = _launch(payload['root'])
    _require_not_cancelled(Path(row['workflow_root']))
    if launch_hash != payload['launch_sha256'] or row['workflow_root'] != payload['workflow_root']:
        raise ValueError('native launch root/record changed')
    stage = _stage(Path(row['root']), Path(row['workflow_root']), request['role'], Path(request_path).parent)
    if request['role'] == 'calibration':
        if request['prompt'] != _regular(CHECKOUT / 'research/proposed-demo/calibration_seed1_fixture.txt', 32768).decode():
            raise ValueError('native calibration prompt changed')
    else:
        _reservation(type('LaunchIdentity', (), {'workflow_root': Path(row['workflow_root']), 'config': row['config']})(),
                     stage, request['role'], request['prompt'])
    if json.loads(_regular(request_path, 32768)) != request:
        raise ValueError('native research request file changed')
    verify_integrity()
    if hashlib.sha256(_regular(SEAL, 1048576)).hexdigest() != payload['source_seal_sha256']:
        raise ValueError('native reviewed source seal changed')


def gate_cpu(authority, operation, **details):
    value = _valid(authority, 'child')
    _require_not_cancelled(value.workflow_root)
    if not value.calibration_done:
        raise ValueError('calibration required before generated CPU work')
    from .experiments import ExperimentRequest, CORRECTNESS_EVALUATOR
    from .executor import SandboxPlan, build_plan
    from .interval_evaluator import cases, validate_source
    if operation == 'candidate':
        request, source = details['request'], details['source']
        stage = _stage(value.root, value.workflow_root, 'experiment', details['stage'])
        if (type(request) is not ExperimentRequest or type(request.candidate_id) is not int
                or request.candidate_id < 1 or value.current is not None):
            raise ValueError('one typed research CPU request required')
        state, task = _reservation(value, stage, 'experiment')
        if (request.candidate_id > len(state['candidates']) or task['targets'] != [request.candidate_id]
                or task['id'] != request.task_id or task['id'] in value.used_cpu_tasks
                or state['experiment_calls'] != value.candidate_count + 1
                or state['experiment_calls'] > value.config['budget']['experiment_calls']):
            raise ValueError('research CPU reservation identity/budget changed')
        expected = ExperimentRequest(task['id'], request.candidate_id, validate_source(source),
                                     digest(CORRECTNESS_EVALUATOR), 'interval-union-correctness-v1', 'correctness-small-v1')
        candidate = state['candidates'][request.candidate_id - 1]
        if (request != expected or candidate['source'] != source or candidate['sha256'] != request.source_sha256
                or candidate['id'] != request.candidate_id or candidate['evidence'] is not None
                or candidate['duplicate_of'] is not None or stage.exists()):
            raise ValueError('exact new generated correctness request required')
        from .demo_prompts import validate_response
        validate_response('generation', source)
        value.candidate_count = state['experiment_calls']
        value.used_cpu_tasks.add(task['id'])
        value.current = {'kind': 'cpu', 'stage': str(stage), 'request': request, 'used': {'candidate'}}
    elif operation in ('admission', 'transport'):
        if (not value.current or value.current['kind'] != 'cpu' or operation in value.current['used']
                or (operation == 'transport' and value.current['used'] != {'candidate', 'admission'})):
            raise ValueError('CPU reservation/admission order required')
        plan = details['plan']
        if type(plan) is not SandboxPlan:
            raise ValueError('fixed CPU sandbox plan required')
        stage, args = Path(value.current['stage']), plan.argv
        expected = build_plan(RUNTIME, stage / 'candidate', stage / 'artifacts',
                              int(args[args.index('--seccomp')+1]), int(args[args.index('--json-status-fd')+1]),
                              cases(), (plan.block_read_fd, plan.block_write_fd))
        if plan != expected:
            raise ValueError('qualified CPU sandbox/limits changed')
        value.current['used'].add(operation)
    else:
        raise ValueError('no arbitrary commands, packaging or authored qualification reuse')
