"""Real subprocess adapter source; gated, never invoked during CPU contracts.

Reuse existing core/host/VRAM supervisors. A fresh single-turn process per stage
is the load/unload boundary; next stage cannot begin before cleanup/postflight.
"""
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time
from contextlib import nullcontext

from launch_policy import require_real_execution
from content_factory import HERE, STOCK, MODELS, ENVIRONMENT, bounded_read, automatic_fit_profile
from prompt_fit import fit_prompt, prompt_context
from admission_policy import demo_postflight
from model_profiles import (model_request, profile_settings, prompt_family_policy,
                            require_profile_fit, canonical_sha256)

QUAL = STOCK.parent / 'upstream-v060/gpu-qualification'
CORE = STOCK.parent / 'candidate-qualification/core_limit_exec.py'
CORE_PATTERN = '|/usr/lib/systemd/systemd-coredump %P %u %g %s %t %c %h %d %F %I'
IDLE_VRAM = 751619276


class BoundedLiveLog:
    """Read an append-only log snapshot; never weaken immutable artifact reads.

    Each poll verifies the previously observed prefix, rereads this snapshot in
    bounded chunks, and checks the descriptor and pathname throughout. Complete
    lines alone reach parsers, since a producer may be midway through a write.
    """
    def __init__(self, path, cap):
        if type(cap) is not int or cap <= 0:
            raise ValueError('positive live-log byte cap required')
        self.path, self.cap = Path(path), cap
        self.identity, self.observed_size, self.prefix = None, 0, b''

    def read(self):
        fd = os.open(self.path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            first = os.fstat(fd)
            identity = (first.st_dev, first.st_ino)
            high_water = self.observed_size
            if self.identity is not None and identity != self.identity:
                raise ValueError('live log replaced')

            def check():
                nonlocal high_water
                for info in (os.fstat(fd), self.path.lstat()):
                    if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != identity:
                        raise ValueError('live log is not the original regular file')
                    if info.st_size < high_water:
                        raise ValueError('live log shrank')
                    if info.st_size > self.cap:
                        raise ValueError('live log exceeded byte cap')
                    high_water = info.st_size

            check()
            size = first.st_size
            if not 0 <= size <= self.cap or size < self.observed_size:
                raise ValueError('invalid live log snapshot size')
            chunks = []
            offset = 0
            while offset < size:
                check()
                chunk = os.pread(fd, min(65536, size - offset), offset)
                if not chunk:
                    raise ValueError('live log shortened during snapshot')
                chunks.append(chunk)
                offset += len(chunk)
                check()
            data = b''.join(chunks)
            if data[:len(self.prefix)] != self.prefix:
                raise ValueError('live log prior prefix changed')
            for offset in range(0, size, 65536):
                expected = data[offset:offset + 65536]
                if os.pread(fd, len(expected), offset) != expected:
                    raise ValueError('live log snapshot prefix changed')
                check()
            check()
            self.identity, self.observed_size, self.prefix = identity, high_water, data
            return data[:data.rfind(b'\n') + 1]
        finally:
            os.close(fd)


def file_identity(path):
    info = Path(path).stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
            info.st_mode, info.st_uid, info.st_gid)


def guarded_argv(stage, scope, host_cutoff, *, demo_capability_path=None):
    command = ['/usr/bin/python3', '-B', str(CORE), '--expected-pattern', CORE_PATTERN,
            '--expected-kernel', '7.2.6-1-cachyos', '--receipt', str(stage / 'core-limit.json'), '--',
            '/usr/bin/python3', '-B', str(QUAL / 'run_bounded_parent.py'),
            '--parent-pid', str(scope['pid']), '--parent-start-ticks', str(scope['start_ticks']),
            '--parent-pid-namespace', scope['pid_namespace'], '--absolute-deadline', str(host_cutoff),
            '--wall-seconds', '60', '--max-rss-bytes', '12GiB', '--as-bytes', '16GiB',
            '--admit-available-bytes', '16GiB', '--min-available-bytes', '12GiB',
            '--max-swap-growth-bytes', '64MiB', '--max-system-swap-growth-bytes', '512MiB',
            '--max-zram-growth-bytes', '128MiB', '--max-swapout-bytes', '256MiB',
            '--max-psi-some-us', '500000', '--max-psi-full-us', '250000', '--interval-seconds', '.25',
            '--term-grace-seconds', '3', '--kill-wait-seconds', '5', '--log', str(stage / 'host.jsonl'),
            '--output', str(stage / 'trial.log'), '--',
            '/usr/bin/python3', '-B', str(STOCK / 'gpu_guard_7g.py'), '--sysfs-dir', '/sys/class/drm/card1/device',
            '--log', str(stage / 'vram.jsonl'), '--max-vram-bytes', '7GiB', '--wall-seconds', '50',
            '--interval-seconds', '.1', '--term-grace-seconds', '1', '--kill-wait-seconds', '1', '--',
            '/usr/bin/nice', '-n', '10', '/usr/bin/python3', '-B', str(HERE / 'native_worker.py'),
            '--request', str(stage / 'request.json')]
    if demo_capability_path is not None:
        command += ['--demo-capability-path', demo_capability_path]
    return command


def final_row(path):
    # Guard-generated logs only. Oversized/malformed evidence never proves release.
    rows = bounded_read(path, 8 * 1048576).splitlines()
    if not rows:
        raise ValueError('empty supervisor log')
    return json.loads(rows[-1])


def require_cleanup(stage):
    host = final_row(stage / 'host.jsonl')
    gpu = final_row(stage / 'vram.jsonl')
    if (host.get('event') != 'final' or host.get('remaining_live_pids') != []
            or host.get('monitor_errors') != [] or host.get('command_started') is not True
            or gpu.get('event') != 'final' or gpu.get('cleanup_incomplete') is not False
            or gpu.get('errors') != []):
        raise RuntimeError('owned cleanup evidence missing or failed')
    return host, gpu


def hash_before_deadline(path, deadline, *, direct=False):
    if direct:
        return direct_hash_before_deadline(path, deadline)
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('integrity work exhausted assigned stage deadline')
            chunk = file.read(262144)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def direct_hash_before_deadline(path, deadline):
    """Hash every byte without an application fallback to buffered model IO.

    The reviewed Linux filesystem accepts page-aligned4096/256KiB reads. The
    final request remains aligned even when EOF is not; only returned bytes
    enter SHA256. Unsupported IO/alignment fails closed. No cache eviction or
    machine setting is requested, and this does not attribute host pressure.
    """
    import fcntl
    path = Path(path)
    if time.monotonic() >= deadline: raise TimeoutError('integrity work exhausted assigned stage deadline')
    flags = os.O_RDONLY | os.O_DIRECT | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    fd = os.open(path, flags)
    def identity(value):
        return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns,value.st_mode,value.st_nlink)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or identity(path.lstat()) != identity(before)
                or not fcntl.fcntl(fd,fcntl.F_GETFL) & os.O_DIRECT):
            raise ValueError('stable regular direct-IO model required')
        digest = hashlib.sha256(); offset = 0; block = 262144
        with mmap.mmap(-1,block) as buffer:
            with memoryview(buffer) as view:
                while offset < before.st_size:
                    if time.monotonic() >= deadline: raise TimeoutError('integrity work exhausted assigned stage deadline')
                    count = os.preadv(fd,[view],offset)
                    if count != min(block,before.st_size-offset): raise ValueError('short or changed direct model read')
                    digest.update(view[:count]); offset += count
        if (time.monotonic() >= deadline): raise TimeoutError('integrity work exhausted assigned stage deadline')
        if identity(os.fstat(fd)) != identity(before) or identity(path.lstat()) != identity(before):
            raise ValueError('model changed during direct integrity read')
        return digest.hexdigest()
    finally:
        os.close(fd)


def integrity_observation():
    """Diagnostic host counters around integrity work; never an admission."""
    from load_telemetry import keyed, pressure
    result = {'monotonic_seconds':time.monotonic(), 'errors':[]}
    for key, path, parser, fields in (
            ('meminfo_kB','/proc/meminfo',keyed,('MemAvailable','Cached','SwapFree')),
            ('vmstat','/proc/vmstat',keyed,('pswpin','pswpout','oom_kill')),
            ('memory_PSI','/proc/pressure/memory',pressure,('some','full'))):
        try:
            values = parser(Path(path).read_text())
            result[key] = {field:values[field] for field in fields}
        except (OSError,ValueError,KeyError) as error:
            result['errors'].append({'source':path,'error':type(error).__name__})
    return result


def verify_runtime_and_model(key, prompt, deadline, *, profile_id=None, role=None, authority=None):
    """Integrity is separate from approval. New controller needs a new seal."""
    if profile_id is None:
        if role is not None or authority is not None:
            raise ValueError('scoped fields require explicit execution profile')
        pin = json.loads(bounded_read(HERE / 'backend-input-seal.json', 1048576))
    else:
        profile_settings(profile_id, key, role)
        from research.demo_authority import reviewed_backend_seal
        pin = reviewed_backend_seal(authority)
    # Deliberately absent until exact new source/model/template/token-fit review.
    if pin.get('schema') != 'PMI_REAL_BACKEND_INTEGRITY_V1':
        raise ValueError('new backend integrity seal required')
    sources = (['content_factory.py', 'llama_backend.py', 'native_worker.py', 'launch_policy.py', 'prompt_fit.py', 'piper_backend.py', 'piper_worker.py', 'load_telemetry.py', 'admission_policy.py']
               if profile_id is None else pin['source_sha256'])
    for name in sources:
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != pin['source_sha256'][name]:
            raise ValueError('backend source changed')
    fit_dependencies = pin.get('fit_dependencies_sha256')
    if not isinstance(fit_dependencies, dict) or not fit_dependencies:
        raise ValueError('pinned prompt-renderer dependencies required')
    fit_frozen = {}
    for name, expected in fit_dependencies.items():
        before = file_identity(name)
        if hash_before_deadline(name, deadline) != expected or file_identity(name) != before:
            raise ValueError('prompt renderer dependency changed')
        fit_frozen[name] = before
    stock_freeze = STOCK / 'stock-input-freeze.json'
    if hashlib.sha256(stock_freeze.read_bytes()).hexdigest() != '9ee5022ff49fbb802b0950a5d9277f1c2b97546d7af8072e2025f4a1fa896591':
        raise ValueError('installed runtime seal changed')
    manifest = json.loads(stock_freeze.read_text())
    for name, row in manifest['files'].items():
        before = file_identity(name)
        info = Path(name).stat()
        original = [str(Path(name).resolve()), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
        if original != manifest['identities'][name]:
            raise ValueError('installed runtime metadata no longer matches original freeze')
        if hash_before_deadline(name, deadline) != row['SHA256'] or file_identity(name) != before:
            raise ValueError('runtime/helper bytes changed: ' + name)
    for name, row in manifest['directory_inventories'].items():
        if Path(name).exists() != row['exists'] or sorted(str(p) for p in Path(name).glob(row['glob'])) != row['paths']:
            raise ValueError('runtime discovery changed')
    qualification = pin['models'][key]
    if key == 'bonsai-stock':
        fingerprints_runtime = verify_bonsai_runtime(deadline)
    else:
        fingerprints_runtime = {}
    if key not in ('qwen3-stock', 'bonsai-stock'):
        raise ValueError('modified profile remains unqualified; do not silently enable it')
    fit_kwargs = {} if profile_id is None else {'profile_id': profile_id, 'role': role}
    fit = fit_prompt(prompt, MODELS[key]['path'], MODELS[key]['template'], deadline,
                     profile='bonsai-stock' if key == 'bonsai-stock' else 'qwen3', **fit_kwargs)
    require_prompt_fit(qualification, fit, model_id=key, prompt=prompt,
                       profile_id=profile_id, role=role)
    if qualification.get('fit_policy') != fit['policy'] or qualification.get('layers') != fit['layers']:
        raise ValueError('sealed model tokenizer/layer qualification changed')
    model = Path(MODELS[key]['path'])
    before = file_identity(model)
    if before[2] != MODELS[key]['bytes']:
        raise ValueError('model size mismatch')
    expected = MODELS[key]['sha256'] or qualification['sha256']
    actual = hash_before_deadline(model, deadline, direct=profile_id is not None)
    after = file_identity(model)
    if actual != expected or before != after:
        raise ValueError('model bytes changed')
    watched = [model, stock_freeze, *map(Path, manifest['files']), *HERE.glob('*.py')]
    fingerprints = {str(path): file_identity(path) for path in watched}
    fingerprints.update(fingerprints_runtime)
    fingerprints.update(fit_frozen)
    fingerprints[str(MODELS[key]['template'])] = file_identity(MODELS[key]['template'])
    if profile_id is not None:
        fingerprints.update({str(HERE / name): file_identity(HERE / name) for name in sources})
        qualification = {**qualification, 'actual_prompt_fit': fit}
    return qualification, fingerprints


def verify_bonsai_runtime(deadline):
    path = HERE / 'bonsai-runtime-inventory.json'
    if hashlib.sha256(bounded_read(path, 1048576)).hexdigest() != MODELS['bonsai-stock']['runtime_inventory_sha256']:
        raise ValueError('compiled Bonsai runtime inventory changed')
    manifest = json.loads(bounded_read(path, 1048576))
    binary = MODELS['bonsai-stock']['native']
    if manifest['files_sha256'].get(binary) != MODELS['bonsai-stock']['native_sha256']:
        raise ValueError('compiled Bonsai executable identity differs')
    directory = HERE / '.runtime/prism-vulkan/bin'
    actual = sorted(str(p) for p in directory.iterdir() if p.is_file())
    if actual != manifest['bin_inventory']:
        raise ValueError('compiled Bonsai runtime discovery changed')
    frozen = {str(path): file_identity(path), str(directory): file_identity(directory)}
    for name, expected in manifest['files_sha256'].items():
        before = file_identity(name)
        if hash_before_deadline(name, deadline) != expected or file_identity(name) != before:
            raise ValueError('compiled Bonsai runtime bytes changed')
        frozen[name] = before
    return frozen


def require_prompt_fit(qualification, fit, *, model_id=None, prompt=None, profile_id=None, role=None):
    if profile_id is not None:
        if qualification.get('prompt_family') != prompt_family_policy():
            raise ValueError('reviewed bounded prompt family absent or changed')
        require_profile_fit(fit, model_id, prompt, profile_id, role)
        return
    expected = qualification.get('prompt_fits', {}).get(fit['prompt_sha256'])
    if expected != fit:
        raise ValueError('exact prompt/template/context fit evidence absent or changed')



def reject_fit_failure(stderr):
    if re.search(r'failed to fit params|encountered an error while trying to fit params|device .* did not report memory', stderr):
        raise RuntimeError('runtime fitter failed or lacked a device memory budget; stop without retry')


def reject_loading_fallback(stderr):
    if 'Falling back to buffered I/O' in stderr or 'Falling back to buffered IO due to' in stderr:
        raise RuntimeError('DirectIO candidate fell back to buffered I/O; stop without retry')
    probing = False
    measuring = False
    fitted = False
    actual_lines = []
    for line in stderr.splitlines():
        if 'fitting params to device memory ...' in line:
            probing, measuring = True, False
        if probing and 'getting device memory data for initial parameters:' in line:
            measuring = True
        if 'successfully fit params to free device memory' in line:
            probing, measuring, fitted = False, False, True
        mode = re.search(r'load_mode = (\w+)', line)
        if mode:
            if mode[1] == 'none' and probing and measuring:
                # Pinned common/fit.cpp uses no_alloc=true/load_mode=NONE.
                continue
            if mode[1] != 'dio':
                raise RuntimeError('actual model load is not DirectIO; stop without retry')
        if fitted and not probing:
            actual_lines.append(line)
    return '\n'.join(actual_lines)


def validate_direct_loading(stderr, direct_io_evidence):
    actual_log = reject_loading_fallback(stderr)
    if not re.search(r'load_mode = dio\b', actual_log):
        raise RuntimeError('DirectIO load mode not proven')
    if not direct_io_evidence or not all(row.get('O_DIRECT') is True for row in direct_io_evidence):
        raise RuntimeError('actual O_DIRECT model descriptor not observed; candidate unqualified')
    if not re.search(r'using async uploads for device Vulkan0, buffer type Vulkan0, backend Vulkan0', actual_log):
        raise RuntimeError('bounded Vulkan async upload not proven; candidate unqualified')
    return {'mode':'dio', 'direct_io_descriptors':direct_io_evidence,
            'Vulkan_async_upload':True,'staging_buffers':4,
            'staging_base_bytes':4*64*1048576,
            'staging_formula':'4*(64MiB+2*A) async buffers + max concurrent rounded CPU tensor temporary; CPU tensor storage additional. Failed async can retain partial buffers plus GPU tensor read_buf and rounded temporary.'}


def validate_placement(stderr, expected_layers, context, fitted):
    if fitted:
        reject_fit_failure(stderr)
        if 'successfully fit params to free device memory' not in stderr:
            raise RuntimeError('runtime fit success not proven')
    layers = re.findall(r'offloaded (\d+)/(\d+) layers to GPU', stderr)
    if not layers or any(int(total) != expected_layers or not 0 <= int(count) <= expected_layers
                         or (not fitted and int(count) != expected_layers) for count, total in layers):
        raise RuntimeError('selected-model placement not proven')
    if not re.search(rf'n_ctx(?:_per_seq)?\s*=\s*{context}\b', stderr):
        raise RuntimeError('actual context not proven')
    buffers = re.findall(r'(\S+) model buffer size\s*=\s*([\d.]+) MiB', stderr)
    if not buffers or any(not 0 <= float(size) < 12288 for _, size in buffers):
        raise RuntimeError('model buffer evidence invalid')
    if int(layers[-1][0]) > 0 and not any('Vulkan' in name and float(size) > 0 for name, size in buffers):
        raise RuntimeError('GPU model buffers unqualified')
    if not fitted and any('CPU' in name and float(size) > 0 for name, size in buffers):
        raise RuntimeError('unexpected CPU model buffer')
    return {'policy':'prism-auto-fit-1536MiB-margin-v1' if fitted else 'full-GPU-v1',
            'GPU_layers':int(layers[-1][0]), 'total_layers':expected_layers,
            'context':context, 'model_buffers_MiB':buffers, 'runtime_fit_success':fitted}

class LlamaBackend:
    actual_inference = True

    def __init__(self, *, authority=None, coordination=None):
        self.authority = authority
        self.coordination = coordination

    def run_stage(self, key, prompt, stage, deadline, *, profile_id=None, role=None):
        settings = None
        if profile_id is None:
            require_real_execution()  # no direct adapter/native-worker bypass entry point
            if role is not None or self.authority is not None:
                raise ValueError('scoped fields require explicit execution profile')
            expected_request = {'model_id': key, 'prompt': prompt}
        else:
            settings = profile_settings(profile_id, key, role)
            from research.demo_authority import gate_model
            gate_model(self.authority, 'backend', model_id=key, role=role, prompt=prompt,
                       stage=stage, deadline=deadline, coordination=self.coordination)
            expected_request = model_request(key, prompt, profile_id, role)
        fitted = automatic_fit_profile(key, prompt, profile_id=profile_id, role=role)
        context = prompt_context(prompt, key) if settings is None else settings['context_tokens']
        request_path = stage / 'request.json'
        request_before = file_identity(request_path)
        if json.loads(bounded_read(request_path, 32768)) != expected_request or file_identity(request_path) != request_before:
            raise ValueError('native request differs from selected prompt/model')
        verify_kwargs = {} if settings is None else {'profile_id': profile_id, 'role': role, 'authority': self.authority}
        if settings is None:
            qualification, fingerprints = verify_runtime_and_model(key, prompt, deadline, **verify_kwargs)
        else:
            integrity_record = {'scope':'host counters; causal attribution unproven',
                                'model_read_mode':'O_DIRECT/256KiB/aligned/no-buffered-fallback',
                                'before':integrity_observation(), 'verified':False}
            try:
                qualification, fingerprints = verify_runtime_and_model(key, prompt, deadline, **verify_kwargs)
                integrity_record['verified'] = True
            finally:
                integrity_record['after'] = integrity_observation()
                with (stage/'integrity-io.json').open('x') as stream:
                    json.dump(integrity_record,stream,indent=2)
        fingerprints[str(request_path)] = request_before
        sys.path.insert(0, str(QUAL))
        import run_gpu_window as owned
        from window_common import alive, fingerprint, passive, require_scope
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, owned.interrupted)
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        ancestor = fingerprint(os.getppid())
        scope = require_scope()

        from load_telemetry import LoadTelemetry
        telemetry = LoadTelemetry(stage, model_path=MODELS[key]['path'])
        native_log = stage / 'native.stderr.log'
        live_reader = BoundedLiveLog(native_log, 1048576)

        def check():
            telemetry.sample()
            if fitted and (native_log.exists() or native_log.is_symlink() or live_reader.identity is not None):
                live_log = live_reader.read().decode('utf-8', errors='replace')
                reject_fit_failure(live_log)
                reject_loading_fallback(live_log)
            if owned.STOP or not alive(ancestor) or not alive(scope) or time.monotonic() >= deadline:
                raise RuntimeError('signal/ancestor/slot deadline stop')
            if any(file_identity(path) != frozen for path, frozen in fingerprints.items()):
                raise RuntimeError('model/runtime/controller metadata changed during stage')

        check()
        # Same passive source/thermal/pressure checks; stronger original stock admission floors.
        admission = passive
        if fitted:
            from admission_policy import admit
            admission = admit
        admission(stage / 'admission.json', 30, deadline - 55, check)
        samples = json.loads((stage / 'admission.json').read_text())['samples']
        if any(s['available'] < 16 * 1024**3 or
               (not fitted and s['vram'] > IDLE_VRAM) for s in samples):
            raise RuntimeError('original RAM/idle-VRAM admission refused')
        if not 0 <= time.monotonic() - samples[-1]['time'] <= 1:
            raise RuntimeError('stale admission')
        cutoff = min(time.monotonic() + 60, deadline - 25)
        capability = nullcontext(None)
        if settings is not None:
            from research.demo_authority import native_capability
            capability = native_capability(self.authority, request_path, expected_request, deadline)
        try:
            with capability as capability_path:
                command = guarded_argv(stage, scope, cutoff, demo_capability_path=capability_path)
                owned.event(stage / 'guarded-command.json', command)
                check()
                if not 0 <= time.monotonic() - samples[-1]['time'] <= 1:
                    raise RuntimeError('stale admission before dispatch')
                rc = owned.run_owned(command, stage / 'supervisor.log', stage / 'host.jsonl', cutoff, check, ENVIRONMENT)
        finally:
            telemetry.sample(force=True)
        host, gpu = require_cleanup(stage)
        if rc or any(row.get('status') != 'completed' or row.get('command_returncode') != 0 for row in [host, gpu]):
            raise RuntimeError('native/host/VRAM failure; no next model')
        postflight = passive if settings is None else demo_postflight
        postflight(stage / 'postflight.json', 20, deadline, check)
        raw = bounded_read(stage / 'native.stdout.log', 1048576)
        stderr = bounded_read(stage / 'native.stderr.log', 1048576).decode('utf-8', errors='strict')
        if not re.search(r'using device Vulkan0 \([^\n]*RX 6600 XT[^\n]*\) \(0000:04:00\.0\)', stderr):
            raise RuntimeError('physical Vulkan device not proven')
        placement = validate_placement(stderr, qualification['layers'], context, fitted)
        if fitted:
            placement['loading'] = validate_direct_loading(stderr, telemetry.direct_io_evidence)
        from content_factory import write_new
        write_new(stage / 'placement.json', placement)
        capture = {'returncode': 0, 'raw_stdout': raw, 'owned_cleanup_empty': True, 'postflight_passed': True}
        if settings is not None:
            fit = qualification['actual_prompt_fit']
            require_profile_fit(fit, key, prompt, profile_id, role)
            capture.update({**settings, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                            'fit': fit, 'fit_sha256': canonical_sha256(fit)})
        return capture
