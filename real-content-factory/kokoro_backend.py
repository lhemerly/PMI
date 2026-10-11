"""Kokoro-82M CPU speech adapter; implementation present, execution still held.

This module launches an isolated, bounded worker using the official
``kokoro==0.9.4`` API and verified local assets. Imports remain lazy; the worker
loads local config, weights, and voice paths with Hugging Face offline mode.
``kokoro_plan()`` remains a metadata-only report and does not qualify the runtime.

Isolation rules this file keeps:

* The preserved Piper baseline (`piper_backend.py`, `piper_worker.py`) is not
  imported, modified or replaced. Kokoro's 24 kHz contract and Piper's 22,050 Hz
  contract are different, and each validator refuses the other's rate.
* Factory integration requires explicit `--speech-model kokoro-82m`; missing or
  unverified assets stop before text generation, with no Piper fallback.
* Piper remains the default route for backwards compatibility and historical
  receipts. Its worker and 22,050 Hz validator are unchanged.
* Every physical entry point is platform-held, exactly like the other adapters.
  Configuring a voice never grants execution.
"""
import hashlib
import io
import json
import os
import signal
import shutil
from pathlib import Path
import sys
import time
import wave

from launch_policy import require_real_execution, require_approved_kokoro_sample
from content_factory import HERE, bounded_read
from model_intent import (KOKORO_DEFAULT_VOICE, KOKORO_PYTHON_SPEC,
                          KOKORO_REPOSITORY, KOKORO_REVISION,
                          KOKORO_LICENSE, KOKORO_SAMPLE_RATE, KOKORO_SHA256, KOKORO_VOICES,
                          KOKORO_CONFIG_FILE, KOKORO_RUNTIME_VERSION, KOKORO_WEIGHT_FILE,
                          PIPER_ADAPTER_ID, PIPER_SAMPLE_RATE)

DEFAULT_VOICE = KOKORO_DEFAULT_VOICE

ADAPTER_ID = 'kokoro-cpu'
MODEL_ID = 'kokoro-82m'
RUNTIME_VERSION = KOKORO_RUNTIME_VERSION
RECEIPT_SCHEMA = 'PMI_KOKORO_CPU_SPEECH_V1'
INPUT_SEAL_SCHEMA = 'PMI_KOKORO_CPU_INTEGRITY_V1'
VOICE_ENVIRONMENT = 'PMI_KOKORO_VOICE'
SAMPLE_RATE = KOKORO_SAMPLE_RATE
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
MAX_TEXT_BYTES = 4096
MAX_SECONDS = 30
MAX_FRAMES = SAMPLE_RATE * MAX_SECONDS
MAX_WAV = SAMPLE_RATE * SAMPLE_WIDTH_BYTES * MAX_SECONDS + 4096
STAGE_SECONDS = 120
ASSET_DIRECTORY = HERE / '.kokoro-assets'
MODEL_FILE = ASSET_DIRECTORY / KOKORO_WEIGHT_FILE
MODEL_CONFIG_FILE = ASSET_DIRECTORY / KOKORO_CONFIG_FILE
VOICE_DIRECTORY = ASSET_DIRECTORY / 'voices'
RUNTIME_DIRECTORY = HERE / '.kokoro-venv'
INPUT_SEAL = HERE / 'kokoro-input-seal.json'
RUNTIME_QUALIFIED = False
SYNTHESIS_IMPLEMENTED = True

# Voice is configuration. Each entry is (BCP-47 language, language name).
VOICES = {voice_id: {'voice_id': voice_id, 'language': language, 'language_name': name,
                     'prefix': voice_id.split('_', 1)[0], 'configurable': True}
          for voice_id, (language, name) in KOKORO_VOICES.items()}


class KokoroUnavailable(RuntimeError):
    """Weights, runtime or reviewed input evidence for Kokoro are absent here."""


def check_text(text):
    """Same bound the Piper baseline uses; the two contracts stay separate."""
    if (not isinstance(text, str) or not text.strip() or '\0' in text
            or len(text.encode('utf-8')) > MAX_TEXT_BYTES):
        raise ValueError('kokoro text must be nonempty UTF-8, no NUL, at most 4096 bytes')


def check_voice(voice_id):
    """Only a configured voice id is accepted; nothing is resolved by prefix."""
    if not isinstance(voice_id, str) or voice_id not in VOICES:
        raise ValueError('kokoro voice must be one of the configured voices: '
                         + ', '.join(sorted(VOICES)))
    return voice_id


def resolve_voice(cli_value=None, environ=None):
    """Configuration only: CLI beats PMI_KOKORO_VOICE beats the default."""
    environment = os.environ if environ is None else environ
    if cli_value is not None:
        return check_voice(cli_value)
    configured = environment.get(VOICE_ENVIRONMENT)
    if configured is not None:
        return check_voice(configured)
    return check_voice(DEFAULT_VOICE)


def voice_config(voice_id):
    return dict(VOICES[check_voice(voice_id)])


def languages():
    return sorted({row['language'] for row in VOICES.values()})


def voice_pack(voice_id):
    """Where the selected voice pack would live; this function never creates it."""
    return VOICE_DIRECTORY / (check_voice(voice_id) + '.pt')


def _file_presence(path):
    """Return only existence/size metadata. Never read a model or voice pack."""
    try:
        if not path.is_file():
            return False, None
        size = path.stat().st_size
        return size > 0, size
    except OSError:
        return False, None


def _runtime_presence():
    """Check the pinned package files without importing or launching the runtime."""
    interpreter = RUNTIME_DIRECTORY / 'bin/python'
    try:
        package_files = list((RUNTIME_DIRECTORY / 'lib').glob(
            f'python*/site-packages/kokoro-{RUNTIME_VERSION}.dist-info/METADATA'))
        init_files = list((RUNTIME_DIRECTORY / 'lib').glob(
            'python*/site-packages/kokoro/__init__.py'))
        return interpreter.is_file() and any(path.is_file() for path in package_files) \
            and any(path.is_file() for path in init_files)
    except OSError:
        return False


def _espeak_path():
    """Locate the sealed project eSpeak-NG, then the system binary."""
    local = HERE / '.runtime/kokoro/espeak/bin/espeak-ng'
    if local.is_file() and os.access(local, os.X_OK):
        return local
    value = shutil.which('espeak-ng', path='/usr/bin:/bin')
    return Path(value) if value else None


def missing_artifacts(voice_id):
    """Inventory required files by path/stat only; hashes and seals stay unverified."""
    voice = check_voice(voice_id)
    model_present, model_bytes = _file_presence(MODEL_FILE)
    config_present, config_bytes = _file_presence(MODEL_CONFIG_FILE)
    voice_path = voice_pack(voice)
    voice_present, voice_bytes = _file_presence(voice_path)
    runtime_present = _runtime_presence()
    seal_present, seal_bytes = _file_presence(INPUT_SEAL)
    espeak_path = _espeak_path()
    espeak_present, espeak_bytes = _file_presence(espeak_path) if espeak_path else (False, None)
    return [
        {'role': 'model_weights', 'repository': KOKORO_REPOSITORY,
         'revision': KOKORO_REVISION, 'file': str(MODEL_FILE),
         'sha256': KOKORO_SHA256, 'sha256_verified_here': False,
         'bytes': model_bytes, 'present': model_present},
        {'role': 'model_config', 'repository': KOKORO_REPOSITORY,
         'revision': KOKORO_REVISION, 'file': str(MODEL_CONFIG_FILE),
         'sha256': None, 'sha256_verified_here': False,
         'bytes': config_bytes, 'present': config_present},
        {'role': 'voice_pack', 'voice_id': voice, 'file': str(voice_path),
         'sha256': None, 'sha256_verified_here': False,
         'bytes': voice_bytes, 'present': voice_present},
        {'role': 'synthesis_runtime', 'package': 'kokoro', 'version': RUNTIME_VERSION,
         'device': 'CPU', 'environment': str(RUNTIME_DIRECTORY), 'installed': runtime_present,
         'present': runtime_present},
        {'role': 'phonemizer', 'package': 'espeak-ng',
         'file': str(espeak_path) if espeak_path else None,
         'bytes': espeak_bytes, 'present': espeak_present},
        {'role': 'input_seal', 'file': str(INPUT_SEAL), 'bytes': seal_bytes,
         'present': seal_present, 'verified': False},
    ]


def readiness(voice_id):
    """Report selectable-voice prerequisites without opening assets or qualifying runtime."""
    voice = check_voice(voice_id)
    rows = missing_artifacts(voice)
    by_role = {row['role']: row for row in rows}
    absent = [row['role'] for row in rows if not row['present']]
    unverified = [role for role in ('model_weights', 'model_config', 'voice_pack')
                  if not by_role[role]['present'] or not by_role[role]['sha256_verified_here']]
    if by_role['input_seal']['present'] and not by_role['input_seal']['verified']:
        unverified.append('input_seal')
    elif not by_role['input_seal']['present']:
        unverified.append('input_seal')
    if not RUNTIME_QUALIFIED:
        unverified.append('runtime_qualification')
    if not SYNTHESIS_IMPLEMENTED:
        unverified.append('synthesis_implementation')
    return {'model_id': MODEL_ID, 'voice_id': voice,
            'assets_present': (by_role['model_weights']['present']
                              and by_role['model_config']['present']
                              and by_role['voice_pack']['present']),
            'identity_verified': (by_role['model_weights']['sha256_verified_here']
                                  and by_role['model_config']['sha256_verified_here']
                                 and by_role['voice_pack']['sha256_verified_here']),
            'runtime_installed': by_role['synthesis_runtime']['installed'],
            'phonemizer_installed': by_role['phonemizer']['present'],
            'runtime_qualified': RUNTIME_QUALIFIED,
            'synthesis_implemented': SYNTHESIS_IMPLEMENTED,
            'runtime_code_status': 'implemented-but-unrun',
            'ready': not absent and not unverified,
            'missing': absent, 'unverified': unverified,
            'artifacts': rows}


def require_ready(voice_id):
    report = readiness(voice_id)
    # Presence is enough to proceed to full seal verification in run_stage;
    # the plan still labels identities and qualification as unverified.
    if report['missing']:
        raise KokoroUnavailable('kokoro prerequisites are absent: ' + ', '.join(report['missing']))
    try:
        seal = json.loads(bounded_read(INPUT_SEAL, 16 * 1048576))
        files = seal.get('files_sha256')
        inventory = sorted(str(path) for path in RUNTIME_DIRECTORY.rglob('*')
                           if path.is_file())
        sources = {str(path) for path in HERE.glob('*.py') if path.is_file()}
        phonemizer = _espeak_path()
        required = {str(MODEL_FILE), str(MODEL_CONFIG_FILE),
                    str(voice_pack('af_heart')), str(voice_pack('pf_dora')),
                    *sources, *inventory}
        if phonemizer is None:
            raise ValueError('eSpeak-NG executable is unavailable')
        required.add(str(phonemizer))
        if (seal.get('schema') != INPUT_SEAL_SCHEMA
                or seal.get('model_repository') != KOKORO_REPOSITORY
                or seal.get('model_revision') != KOKORO_REVISION
                or seal.get('model_sha256') != KOKORO_SHA256
                or seal.get('runtime_package') != 'kokoro'
                or seal.get('runtime_version') != RUNTIME_VERSION
                or seal.get('runtime_inventory') != inventory
                or not isinstance(files, dict) or not required.issubset(files)
                or files.get(str(MODEL_FILE)) != KOKORO_SHA256):
            raise ValueError('Kokoro integrity seal is incomplete or does not match pinned inputs')
    except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError) as error:
        raise KokoroUnavailable('Kokoro local integrity seal is not ready: ' + str(error)) from error
    return report


def verify_kokoro_inputs(voice_id, deadline):
    """Verify reviewed local assets, runtime inventory, and source before launching.

    The seal is a reviewed input, not an authority to execute. It must bind the
    exact model identity, runtime API version, both local voice packs, the source
    files, and every file in the private runtime environment.
    """
    from llama_backend import file_identity, hash_before_deadline

    voice = check_voice(voice_id)
    seal_identity = file_identity(INPUT_SEAL)
    seal_bytes = bounded_read(INPUT_SEAL, 16 * 1048576)
    seal = json.loads(seal_bytes)
    if file_identity(INPUT_SEAL) != seal_identity:
        raise ValueError('kokoro input seal changed while it was read')
    files = seal.get('files_sha256')
    if (seal.get('schema') != INPUT_SEAL_SCHEMA
            or seal.get('model_repository') != KOKORO_REPOSITORY
            or seal.get('model_revision') != KOKORO_REVISION
            or seal.get('model_sha256') != KOKORO_SHA256
            or seal.get('runtime_package') != 'kokoro'
            or seal.get('runtime_version') != RUNTIME_VERSION
            or not isinstance(files, dict) or not files):
        raise ValueError('reviewed Kokoro model/runtime integrity seal required')

    runtime_inventory = sorted(str(path) for path in RUNTIME_DIRECTORY.rglob('*')
                               if path.is_file())
    if (not runtime_inventory
            or runtime_inventory != seal.get('runtime_inventory')):
        raise ValueError('Kokoro runtime file inventory changed')
    source_files = sorted(str(path) for path in HERE.glob('*.py') if path.is_file())
    required = {str(MODEL_FILE), str(MODEL_CONFIG_FILE),
                str(voice_pack('af_heart')), str(voice_pack('pf_dora')),
                *source_files, *runtime_inventory}
    phonemizer = _espeak_path()
    if phonemizer is None or not phonemizer.is_file() or not os.access(phonemizer, os.X_OK):
        raise ValueError('eSpeak-NG executable required for Kokoro language G2P')
    required.add(str(phonemizer))
    if not required.issubset(files):
        raise ValueError('Kokoro integrity seal omits a required asset/runtime/source file')
    if files.get(str(MODEL_FILE)) != KOKORO_SHA256:
        raise ValueError('Kokoro weights do not match the pinned SHA-256')

    allowed_roots = (HERE, RUNTIME_DIRECTORY)
    frozen = {str(INPUT_SEAL): seal_identity}
    for name, expected in files.items():
        path = Path(name)
        if path != phonemizer and not any(path.is_relative_to(root) for root in allowed_roots):
            raise ValueError('Kokoro integrity seal references a path outside the approved roots')
        before = file_identity(path)
        if (hash_before_deadline(path, deadline) != expected
                or file_identity(path) != before):
            raise ValueError('Kokoro asset/runtime/source bytes changed: ' + name)
        frozen[str(path)] = before
    return {'file_identities': frozen,
            'model_sha256': files[str(MODEL_FILE)],
            'model_config_sha256': files[str(MODEL_CONFIG_FILE)],
            'voice_pack_sha256': files[str(voice_pack(voice))],
            'input_seal_sha256': hashlib.sha256(seal_bytes).hexdigest()}


def kokoro_plan(voice_id=None):
    """Offline plan: metadata checks only; no import, weight read, service or socket."""
    voice = check_voice(DEFAULT_VOICE if voice_id is None else voice_id)
    report = readiness(voice)
    return {
        'schema': RECEIPT_SCHEMA, 'status': 'PlatformHeld', 'adapter': ADAPTER_ID,
        'model_id': MODEL_ID, 'device': 'CPU', 'isolated': True, 'wired_into_pipeline': True,
        'voice_id': voice, 'default_voice': DEFAULT_VOICE,
        'voice_environment': VOICE_ENVIRONMENT,
        'voices': {key: dict(row) for key, row in VOICES.items()},
        'languages': languages(),
        'sample_rate': SAMPLE_RATE, 'channels': CHANNELS,
        'sample_width_bytes': SAMPLE_WIDTH_BYTES,
        'max_text_bytes': MAX_TEXT_BYTES, 'max_audio_seconds': MAX_SECONDS,
        'max_wav_bytes': MAX_WAV,
        'model': {'repository': KOKORO_REPOSITORY, 'revision': KOKORO_REVISION,
                  'file': str(MODEL_FILE),
                  'sha256': KOKORO_SHA256, 'sha256_verified_here': False,
                  'bytes': report['artifacts'][0]['bytes'], 'license': KOKORO_LICENSE,
                  'license_note': 'Apache-2.0 per the pinned official model card; this does not '
                                  'establish local asset identity or runtime qualification'},
        'model_config': {'file': str(MODEL_CONFIG_FILE),
                         'repository': KOKORO_REPOSITORY,
                         'revision': KOKORO_REVISION,
                         'sha256_verified_here': False},
        'model_present': report['artifacts'][0]['present'],
        'assets_present': report['assets_present'], 'readiness': report,
        'runtime': {'package': 'kokoro', 'version': RUNTIME_VERSION,
                    'python_version': KOKORO_PYTHON_SPEC,
                    'required_packages': ['huggingface-hub', 'loguru',
                                          'misaki[en]>=0.9.4', 'numpy',
                                          'torch (CPU build)', 'transformers'],
                    'environment': str(RUNTIME_DIRECTORY), 'device': 'CPU'},
        'runtime_installed': report['runtime_installed'],
        'runtime_qualified': False,
        'runtime_code_status': 'implemented-but-unrun',
        'synthesis_implemented': SYNTHESIS_IMPLEMENTED,
        'execution_held': True,
        'actual_inference': False, 'assets_downloaded': False, 'weights_opened': False,
        'preserved_baseline': {'adapter': PIPER_ADAPTER_ID,
                               'sample_rate': PIPER_SAMPLE_RATE,
                               'unchanged': True,
                               'note': 'Piper remains the approved auxiliary CPU adapter; '
                                       'this adapter does not replace, wrap or modify it'},
        'missing_artifacts': report['artifacts'],
        'unavailable_modalities': ['video'],
        'release_rule': 'a synthesized WAV is accepted only after real PCM validation and '
                        'proven owned cleanup, under the same shared lease as the rest of '
                        'the sequence',
        'remaining': ['local config.json, pinned kokoro-v1_0.pth, and af_heart.pt / '
                      'pf_dora.pt files with a reviewed kokoro-input-seal.json',
                      'an isolated Python runtime with kokoro==' + RUNTIME_VERSION
                      + ', its dependencies, and system espeak-ng, sealed by exact inventory',
                      'a bounded first real CPU run to assess this exact host/workload; '
                      'the runtime code itself has not been run here',
                      'normal platform review for each physical execution'],
    }


def validate_wav(path):
    """Validate real 24 kHz mono PCM16 bytes. Never synthesizes, never plays."""
    raw = bounded_read(path, MAX_WAV)
    with wave.open(io.BytesIO(raw), 'rb') as wav:
        frames = wav.getnframes()
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(),
                wav.getcomptype()) != (SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH_BYTES, 'NONE'):
            raise ValueError('kokoro audio must be 24000 Hz mono 16-bit PCM')
        if (not 0 < frames <= SAMPLE_RATE * MAX_SECONDS
                or len(wav.readframes(frames)) != frames * SAMPLE_WIDTH_BYTES):
            raise ValueError('empty, truncated or over-duration kokoro WAV')
    return {'adapter': ADAPTER_ID, 'wav_sha256': hashlib.sha256(raw).hexdigest(),
            'frames': frames, 'sample_rate': SAMPLE_RATE, 'channels': CHANNELS,
            'sample_width_bytes': SAMPLE_WIDTH_BYTES,
            'duration_seconds': frames / SAMPLE_RATE}


def kokoro_guarded_argv(stage, scope, cutoff):
    """Reuse the project's owned host/VRAM supervisors for the isolated worker."""
    from llama_backend import guarded_argv

    argv = guarded_argv(stage, scope, cutoff)
    for index, value in enumerate(argv[:-1]):
        if value == '--wall-seconds':
            argv[index + 1] = str(STAGE_SECONDS)
    worker_start = argv.index('/usr/bin/nice')
    return argv[:worker_start] + [
        '/usr/bin/nice', '-n', '10', str(RUNTIME_DIRECTORY / 'bin/python'), '-B',
        str(HERE / 'kokoro_worker.py'), '--request', str(Path(stage) / 'request.json')]


class KokoroBackend:
    """Explicitly selected CPU adapter; lazy runtime imports and no Piper fallback."""

    actual_inference = True
    model_id = MODEL_ID
    runtime_version = RUNTIME_VERSION

    def __init__(self, voice_id=None, environ=None):
        self.voice_id = resolve_voice(voice_id, environ)

    def plan(self):
        return kokoro_plan(self.voice_id)

    def readiness(self):
        return readiness(self.voice_id)

    def ensure_ready(self):
        return require_ready(self.voice_id)

    def run_stage(self, text, stage, deadline):
        require_approved_kokoro_sample(text, self.voice_id)  # exact reviewed CPU sample only
        check_text(text)
        self.ensure_ready()
        from llama_backend import QUAL, file_identity, require_cleanup

        stage = Path(stage)
        request_path = stage / 'request.json'
        request_before = file_identity(request_path)
        request = json.loads(bounded_read(request_path, 32768))
        if (set(request) != {'model_id', 'text', 'voice_id', 'text_sha256'}
                or request.get('model_id') != self.model_id
                or request.get('text') != text or request.get('voice_id') != self.voice_id
                or request.get('text_sha256') != hashlib.sha256(text.encode()).hexdigest()
                or file_identity(request_path) != request_before):
            raise ValueError('kokoro request differs from the completed text or the voice')
        verified = verify_kokoro_inputs(self.voice_id, deadline - 5)
        frozen = verified['file_identities']
        frozen[str(request_path)] = request_before
        sys.path.insert(0, str(QUAL))
        import run_gpu_window as owned
        from window_common import alive, fingerprint, passive, require_scope

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, owned.interrupted)
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        ancestor = fingerprint(os.getppid())
        scope = require_scope()

        def check():
            if (owned.STOP or not alive(ancestor) or not alive(scope)
                    or time.monotonic() >= deadline):
                raise RuntimeError('signal/owner/deadline stop')
            for name, identity in frozen.items():
                if file_identity(name) != identity:
                    raise RuntimeError('Kokoro input changed during owned speech run')

        check()
        from admission_policy import speech_admit
        admission = speech_admit  # elapsed coverage includes full input-identity sweeps
        from content_factory import ENVIRONMENT, automatic_fit_profile
        job_prompt = bounded_read(stage.parent / 'prompt.txt', 4096).decode('utf-8')
        plan = json.loads(bounded_read(stage.parent / 'plan.json', 65536))
        if (automatic_fit_profile('bonsai-stock', job_prompt)
                and [row['model_id'] for row in plan.get('stages', [])] == ['bonsai-stock']):
            from admission_policy import speech_admit
            admission = speech_admit
        admission(stage / 'admission.json', 30, deadline - 55, check)
        samples = json.loads(bounded_read(stage / 'admission.json', 2 * 1048576))['samples']
        if not samples or any(row.get('available', 0) < 16 * 1024**3 for row in samples):
            raise RuntimeError('Kokoro speech admission refused')
        cutoff = min(time.monotonic() + STAGE_SECONDS, deadline - 25)
        if cutoff <= time.monotonic() + 5:
            raise TimeoutError('insufficient bounded Kokoro worker time')
        command = kokoro_guarded_argv(stage, scope, cutoff)
        owned.event(stage / 'guarded-command.json', command)
        check()
        if not 0 <= time.monotonic() - samples[-1]['time'] <= 1:
            raise RuntimeError('stale Kokoro speech admission')
        environment = {**ENVIRONMENT,
                       'PHONEMIZER_ESPEAK_LIBRARY': str(HERE / '.runtime/kokoro/espeak/lib/libespeak-ng.so'),
                       'ESPEAK_DATA_PATH': str(HERE / '.runtime/kokoro/espeak/share/espeak-ng-data'),
                       'OMP_NUM_THREADS': str(4),
                       'OPENBLAS_NUM_THREADS': str(4),
                       'MKL_NUM_THREADS': str(4),
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'HF_HUB_DISABLE_TELEMETRY': '1'}
        rc = owned.run_owned(command, stage / 'supervisor.log', stage / 'host.jsonl',
                             cutoff, check, environment)
        host, gpu = require_cleanup(stage)
        if rc or any(row.get('status') != 'completed' or row.get('command_returncode') != 0
                     for row in (host, gpu)):
            raise RuntimeError('Kokoro worker/supervisor failure')
        from admission_policy import speech_postflight
        speech_postflight(stage / 'postflight.json', 20, deadline, check)
        audio = validate_wav(stage / 'answer.wav')
        return {'returncode': 0, 'owned_cleanup_empty': True,
                'postflight_passed': True, 'runtime_version': RUNTIME_VERSION,
                'runtime_code_status': 'executed-this-receipt',
                'runtime_qualification': 'unqualified',
                'model_repository': KOKORO_REPOSITORY,
                'model_revision': KOKORO_REVISION,
                'model_sha256': verified['model_sha256'],
                'model_sha256_verified': True,
                'model_config_sha256': verified['model_config_sha256'],
                'voice_pack_sha256': verified['voice_pack_sha256'],
                'input_seal_sha256': verified['input_seal_sha256'],
                'voice_id': self.voice_id, 'voice_language': VOICES[self.voice_id]['language'],
                **audio}
