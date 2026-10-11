"""Sequential CPU speech candidate; every physical entry remains platform-held."""
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import time
import wave

from launch_policy import require_real_execution
from content_factory import HERE, ENVIRONMENT, bounded_read
from llama_backend import (QUAL, file_identity, guarded_argv,
                           hash_before_deadline, require_cleanup)

VENV = HERE / '.piper-venv'
VOICE = HERE / 'voices/en_US-ljspeech-medium.onnx'
VOICE_BYTES = 63531379
VOICE_SHA256 = '6f52a751e2349abe7a76735eb09dc1875298c77ea2342ffd2fef79ff81b87f22'
MAX_WAV = 22050 * 2 * 30 + 4096


def check_speech_text(text):
    if not isinstance(text, str) or not text.strip() or '\0' in text or len(text.encode('utf-8')) > 4096:
        raise ValueError('speech text must be nonempty UTF-8, no NUL, at most 4096 bytes')


def speech_plan():
    return {'status': 'PlatformHeld', 'voice_id': 'en_US-ljspeech-medium', 'device': 'CPU',
            'package': 'piper-tts==1.8.0', 'package_license': 'GPL-3.0-or-later',
            'voice_bytes': VOICE_BYTES, 'voice_sha256': VOICE_SHA256,
            'voice_file': str(VOICE), 'max_text_bytes': 4096, 'max_audio_seconds': 30,
            'max_wav_bytes': MAX_WAV, 'host_wall_seconds': 60, 'stage_seconds': 120,
            'release_rule': 'text cleanup/postflight before Piper load; same lease through audio cleanup',
            'remaining': ['normal platform review for each physical execution']}


def validate_wav(path):
    raw = bounded_read(path, MAX_WAV)
    with wave.open(io.BytesIO(raw), 'rb') as wav:
        frames = wav.getnframes()
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (22050, 1, 2, 'NONE'):
            raise ValueError('unexpected PCM WAV format')
        if not 0 < frames <= 22050 * 30 or len(wav.readframes(frames)) != frames * 2:
            raise ValueError('empty, truncated or over-duration WAV')
    return {'wav_sha256': hashlib.sha256(raw).hexdigest(), 'frames': frames,
            'sample_rate': 22050, 'duration_seconds': frames / 22050}


def verify_speech_inputs(deadline):
    """Review pins dependencies and full venv inventory; never generates authority."""
    seal_path = HERE / 'speech-input-seal.json'
    seal = json.loads(bounded_read(seal_path, 2 * 1048576))
    if seal.get('schema') != 'PMI_PIPER_CPU_INTEGRITY_V1':
        raise ValueError('reviewed speech integrity seal required')
    files = seal['files_sha256']
    required = {str(VOICE), str(VOICE) + '.json', str(VENV / 'bin/python'),
                *map(str, HERE.glob('*.py'))}
    inventory = sorted(str(p) for p in VENV.rglob('*') if p.is_file())
    if inventory != seal['venv_inventory'] or not inventory or not required.issubset(files) or not set(inventory).issubset(files):
        raise ValueError('speech source/runtime inventory incomplete or changed')
    if VOICE.stat().st_size != VOICE_BYTES or files[str(VOICE)] != VOICE_SHA256:
        raise ValueError('unexpected voice identity')
    frozen = {str(seal_path): file_identity(seal_path)}
    for name, digest in files.items():
        before = file_identity(name)
        if hash_before_deadline(name, deadline) != digest or file_identity(name) != before:
            raise ValueError('speech dependency bytes changed')
        frozen[name] = before
    return frozen


def speech_guarded_argv(stage, scope, cutoff):
    # Retain original core/host/VRAM ceilings even though Piper is CPU-only.
    argv = guarded_argv(stage, scope, cutoff)
    index = argv.index('/usr/bin/nice')
    return argv[:index] + ['/usr/bin/nice', '-n', '10', str(VENV / 'bin/python'), '-B',
                          str(HERE / 'piper_worker.py'), '--request', str(stage / 'request.json')]


class PiperBackend:
    actual_inference = True

    def run_stage(self, text, stage, deadline):
        require_real_execution()
        check_speech_text(text)
        frozen = verify_speech_inputs(deadline)
        request_path = stage / 'request.json'
        request_before = file_identity(request_path)
        if json.loads(bounded_read(request_path, 32768)) != {'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest()} or file_identity(request_path) != request_before:
            raise ValueError('speech request differs from completed text')
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
            if owned.STOP or not alive(ancestor) or not alive(scope) or time.monotonic() >= deadline:
                raise RuntimeError('signal/owner/deadline stop')
            if any(file_identity(name) != value for name, value in frozen.items()):
                raise RuntimeError('speech request/runtime changed')

        check()
        admission = passive
        from content_factory import automatic_fit_profile
        job_prompt = bounded_read(stage.parent / 'prompt.txt', 4096).decode('utf-8')
        plan = json.loads(bounded_read(stage.parent / 'plan.json', 65536))
        if (automatic_fit_profile('bonsai-stock', job_prompt) and
                [row['model_id'] for row in plan['stages']] == ['bonsai-stock']):
            from admission_policy import speech_admit
            admission = speech_admit
        admission(stage / 'admission.json', 30, deadline - 55, check)
        samples = json.loads((stage / 'admission.json').read_text())['samples']
        if any(s['available'] < 16 * 1024**3 for s in samples):
            raise RuntimeError('speech admission refused')
        cutoff = min(time.monotonic() + 60, deadline - 25)
        command = speech_guarded_argv(stage, scope, cutoff)
        owned.event(stage / 'guarded-command.json', command)
        check()
        if not 0 <= time.monotonic() - samples[-1]['time'] <= 1:
            raise RuntimeError('stale speech admission')
        environment = {**ENVIRONMENT, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
        rc = owned.run_owned(command, stage / 'supervisor.log', stage / 'host.jsonl', cutoff, check, environment)
        host, gpu = require_cleanup(stage)
        if rc or any(row.get('status') != 'completed' or row.get('command_returncode') != 0 for row in (host, gpu)):
            raise RuntimeError('speech supervisor failure')
        from admission_policy import speech_postflight
        speech_postflight(stage / 'postflight.json', 20, deadline, check)
        validate_wav(stage / 'answer.wav')
        return {'returncode': 0, 'owned_cleanup_empty': True, 'postflight_passed': True}


def verified_resume_text(source_job, deadline):
    """Only the separately sealed successful Bonsai answer may resume speech."""
    from content_factory import decode_answer
    static_manifest = HERE / 'resume-speech-provenance.json'
    candidate = source_job / 'resume-speech-provenance.json'
    seal = json.loads(bounded_read(HERE / 'speech-input-seal.json', 2 * 1048576))
    raw = bounded_read(candidate if candidate.is_file() else static_manifest, 65536)
    manifest = json.loads(raw)
    files = seal.get('files_sha256', {})
    dynamic = candidate.is_file()
    if dynamic:
        ready = json.loads(bounded_read(source_job/'text-ready.json',65536))
        prior = json.loads(bounded_read(source_job/'result.json',65536))
        completed = prior.get('completed_stages',[])
        matching = any(row.get('model_id')=='bonsai-stock' and row.get('status')=='RealTextAndReleasePass'
                       and row.get('answer_file')==str(source_job/'stage-0-bonsai-stock/answer.txt')
                       for row in completed)
        if (ready.get('schema')!='PMI_TEXT_RESUME_READY_V1'
                or ready.get('status')!='RealTextAndReleasePass'
                or hashlib.sha256(raw).hexdigest()!=ready.get('provenance_sha256')
                or prior.get('status')!='StoppedNoRetry' or not matching):
            raise ValueError('failed job has no verified completed Bonsai text stage')
    elif hashlib.sha256(raw).hexdigest()!=files.get(str(static_manifest)):
        raise ValueError('resume provenance absent or unsealed')
    from content_factory import MODELS
    if (seal.get('schema')!='PMI_PIPER_CPU_INTEGRITY_V1'
            or manifest.get('schema')!='PMI_VERIFIED_TEXT_RESUME_V1'
            or str(source_job.resolve())!=manifest.get('source_job')
            or manifest.get('model_id')!='bonsai-stock'
            or manifest.get('model_sha256')!=MODELS['bonsai-stock']['sha256']):
        raise ValueError('resume provenance absent, unsealed or source differs')
    for name, digest in manifest['files_sha256'].items():
        if ((not dynamic and files.get(name) != digest) or not Path(name).is_relative_to(source_job.resolve())):
            raise ValueError('resume evidence is outside reviewed source or unsealed')
        before = file_identity(name)
        if hash_before_deadline(name, deadline) != digest or file_identity(name) != before:
            raise ValueError('resume source evidence changed')
    stage = source_job / 'stage-0-bonsai-stock'
    required = [source_job/'prompt.txt', source_job/'plan.json', stage/'request.json',
                stage/'answer.txt', stage/'native.stdout.log', stage/'native.stderr.log',
                stage/'result.json', stage/'placement.json', stage/'host.jsonl',
                stage/'vram.jsonl', stage/'postflight.json']
    if set(map(str,required)) != set(manifest['files_sha256']):
        raise ValueError('resume provenance evidence set incomplete')
    prompt=bounded_read(source_job/'prompt.txt',4096).decode('utf-8')
    request=json.loads(bounded_read(stage/'request.json',32768))
    plan=json.loads(bounded_read(source_job/'plan.json',65536))
    if (request!={'model_id':'bonsai-stock','prompt':prompt} or
            not plan.get('stages') or plan['stages'][0].get('model_id')!='bonsai-stock' or
            manifest.get('prompt_sha256')!=hashlib.sha256(prompt.encode()).hexdigest()):
        raise ValueError('resume prompt/model request differs')
    result = json.loads(bounded_read(stage/'result.json', 65536))
    if (result.get('status') != 'RealTextAndReleasePass' or result.get('actual_inference') is not True
            or result.get('model_id') != 'bonsai-stock' or result.get('owned_cleanup_empty') is not True
            or result.get('postflight_passed') is not True):
        raise ValueError('resume text completion/release unproven')
    host, gpu = require_cleanup(stage)
    if any(row.get('status') != 'completed' or row.get('command_returncode') != 0 for row in (host,gpu)):
        raise ValueError('resume source supervisors did not complete')
    if json.loads(bounded_read(stage/'postflight.json', 65536)).get('reasons') != []:
        raise ValueError('resume source postflight failed')
    raw_answer = bounded_read(stage/'answer.txt', 4096)
    text = raw_answer.decode('utf-8').rstrip('\n')
    from content_factory import automatic_fit_profile
    if (not automatic_fit_profile('bonsai-stock',prompt) or
            hashlib.sha256(text.encode()).hexdigest() != result.get('answer_sha256') or
            hashlib.sha256(raw_answer).hexdigest() != manifest.get('answer_file_sha256') or
            decode_answer(bounded_read(stage/'native.stdout.log', 1048576)) != text):
        raise ValueError('resume answer/prompt/natural finish mismatch')
    check_speech_text(text)
    return text, prompt, manifest
