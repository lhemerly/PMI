#!/usr/bin/env python3
"""Prompt-driven serial real-text backend source; physical execution remains Held."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
from prompt_fit import SYSTEM_PROMPT, prompt_context
from model_intent import (DEFAULT_SPEECH_MODEL_ID, KOKORO_REPOSITORY,
                          KOKORO_REVISION, KOKORO_ROLE_ID, KOKORO_RUNTIME_VERSION,
                          KOKORO_SHA256, PIPER_ROLE_ID, PIPER_VOICE_ID,
                          SPEECH_MODEL_IDS)

from launch_policy import require_real_execution

HERE = Path(__file__).resolve().parent
STOCK = Path('/home/luizh/Documents/Codex/2026-10-06/task-3/smaller-conversation-prep-v1')
NATIVE = '/usr/bin/llama-completion'
MAX_PROMPT = 4096
MAX_OUTPUT = 1048576
ENVIRONMENT = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'LD_LIBRARY_PATH': '/usr/lib'}
PRISM_SOURCE = STOCK / 'source/prism/PrismML-Eng-llama.cpp-adfffbe'
PRISM_NATIVE = str(HERE / '.runtime/prism-vulkan/bin/llama-completion')
DEFAULT_MODEL = 'bonsai-stock'
MODELS = {
    'bonsai-stock': {
        'catalog_id': 'Bonsai2-27B-PTQ1_0',
        'repository': 'prism-ml/Ternary-Bonsai-2-27B-gguf',
        'revision': 'b072e1d3b35a0a630cece372c2127528e0994386',
        'path': str(STOCK / 'models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'),
        'bytes': 5946648928,
        'sha256': '53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3',
        'track': 'ordinary-stock',
        'template': str(STOCK / 'templates/bonsai2-27b-official.jinja'),
        'native': PRISM_NATIVE,
        'native_sha256': 'c1b84e353ab86f3444603f59c944283b1fe1043359585b8ddb843bd2a10f5686',
        'runtime_inventory_sha256': '89ea8bb1223e1063d8858dc61bccce05b8476a5d33bec6e5cd49fcf0058a9d98',
        'runtime_commit': 'adfffbe41b2cabcd51fff326ab045662265062bb',
        'qualification': 'one real exact welcome-prompt 512-token-context run completed with DIO Vulkan placement, bounded natural completion and verified unload/postflight; other prompts/contexts remain unqualified; not the Heretic derivative',
    },
    'qwen3-stock': {
        'path': str(STOCK / 'models/Qwen3-8B-Q4_K_M.gguf'),
        'bytes': 5027783488,
        'sha256': 'd98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785',
        'track': 'ordinary-stock',
        'template': str(STOCK / 'templates/qwen3-8b-nonthinking-setting.jinja'),
        'qualification': 'stock frozen baseline; custom prompt/controller still unqualified',
    },
    'qwen25-modified': {
        'path': '/home/luizh/models/Qwen2.5-7B-Instruct-abliterated-v2.Q4_K_M.gguf',
        'bytes': 4683073920,
        'sha256': None,
        'track': 'optional-modified',
        'template': None,
        'qualification': 'file present; hash/license/embedded template/native compatibility unqualified',
    },
}


def bounded_read(path, cap):
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as file:
        before = os.fstat(file.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > cap:
            raise ValueError('expected bounded regular file')
        data = file.read(cap + 1)
        after = os.fstat(file.fileno())
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if len(data) > cap or len(data) != before.st_size or any(getattr(before, k) != getattr(after, k) for k in fields):
            raise ValueError('input changed or exceeded bound')
        return data


def check_prompt(prompt):
    if not isinstance(prompt, str) or not prompt.strip() or '\0' in prompt or len(prompt.encode('utf-8')) > MAX_PROMPT:
        raise ValueError('prompt must be nonempty UTF-8, no NUL, at most 4096 bytes')


def select_models(ids):
    if not 1 <= len(ids) <= 2 or len(set(ids)) != len(ids) or any(key not in MODELS for key in ids):
        raise ValueError('select one or two distinct supported model IDs; no arbitrary paths or commands')
    return ids


def create_speech_backend(model_id=DEFAULT_SPEECH_MODEL_ID, voice_id=None, environ=None):
    """Select exactly one speech adapter; an unsupported or mismatched voice is an error."""
    if model_id not in SPEECH_MODEL_IDS:
        raise ValueError('select one supported speech model: ' + ', '.join(SPEECH_MODEL_IDS))
    if model_id == PIPER_ROLE_ID:
        if voice_id is not None:
            raise ValueError('--speech-voice is supported only with --speech-model kokoro-82m')
        from piper_backend import PiperBackend
        return PiperBackend()
    if model_id == KOKORO_ROLE_ID:
        from kokoro_backend import KokoroBackend
        return KokoroBackend(voice_id=voice_id, environ=environ)
    raise ValueError('unsupported speech model: ' + str(model_id))


def _speech_contract(speech):
    """Return the validator/identity for the selected adapter without fallback."""
    model_id = getattr(speech, 'model_id', PIPER_ROLE_ID)
    if model_id == PIPER_ROLE_ID:
        from piper_backend import check_speech_text, validate_wav
        return {'model_id': PIPER_ROLE_ID, 'stage': 'speech:piper-cpu',
                'voice_id': PIPER_VOICE_ID, 'language': 'en-US',
                'check_text': check_speech_text, 'validate_wav': validate_wav}
    if model_id == KOKORO_ROLE_ID:
        from kokoro_backend import VOICES, check_text, validate_wav
        voice_id = getattr(speech, 'voice_id', None)
        if voice_id not in VOICES:
            raise ValueError('selected Kokoro backend has no recognized voice identity')
        voice = VOICES[voice_id]
        return {'model_id': KOKORO_ROLE_ID, 'stage': 'speech:kokoro-82m',
                'voice_id': voice_id, 'language': voice['language'],
                'check_text': check_text, 'validate_wav': validate_wav}
    raise ValueError('speech backend has unsupported model identity: ' + str(model_id))


def _kokoro_capture_metadata(capture, actual_inference):
    """Copy verified runtime/asset identity into receipts without upgrading mocks."""
    keys = ('runtime_version', 'runtime_code_status', 'runtime_qualification',
            'model_repository', 'model_revision', 'model_sha256',
            'model_sha256_verified', 'model_config_sha256', 'voice_pack_sha256',
            'input_seal_sha256')
    if any(key not in capture for key in keys):
        raise RuntimeError('Kokoro capture is missing runtime or input identity fields')
    if (capture['runtime_version'] != KOKORO_RUNTIME_VERSION
            or capture['model_repository'] != KOKORO_REPOSITORY
            or capture['model_revision'] != KOKORO_REVISION):
        raise RuntimeError('Kokoro capture does not match the pinned runtime/model identity')
    if not isinstance(capture['model_sha256_verified'], bool):
        raise RuntimeError('Kokoro model identity verification status is invalid')
    if actual_inference:
        hashes = (capture['model_sha256'], capture['model_config_sha256'],
                  capture['voice_pack_sha256'], capture['input_seal_sha256'])
        if (not capture['model_sha256_verified']
                or capture['model_sha256'] != KOKORO_SHA256
                or any(not isinstance(value, str) or len(value) != 64
                       or any(char not in '0123456789abcdef' for char in value)
                       for value in hashes)):
            raise RuntimeError('Kokoro inference receipt lacks verified pinned input hashes')
    return {**{key: capture[key] for key in keys},
            'asset_identity_status': ('verified-before-launch'
                                      if capture['model_sha256_verified']
                                      else 'mock-unverified')}


def automatic_fit_profile(model_id, prompt, *, profile_id=None, role=None):
    if profile_id is not None:
        from model_profiles import profile_settings
        profile_settings(profile_id, model_id, role)
        return True
    if role is not None:
        raise ValueError('role requires explicit execution profile')
    return model_id == 'bonsai-stock' and prompt_context(prompt, model_id) == 512


def native_argv(model_id, prompt, *, profile_id=None, role=None):
    check_prompt(prompt)
    from model_profiles import profile_settings
    settings = None if profile_id is None else profile_settings(profile_id, model_id, role)
    fitted = automatic_fit_profile(model_id, prompt, profile_id=profile_id, role=role)
    context = prompt_context(prompt, model_id) if settings is None else settings['context_tokens']
    output = 64 if settings is None else settings['output_tokens']
    spec = MODELS[model_id]
    argv = [spec.get('native', NATIVE), '--model', spec['path'], '--ctx-size', str(context), '--gpu-layers', '99',
            '--device', 'Vulkan0', '--override-tensor', '.*=Vulkan0', '--fit', 'off',
            '--flash-attn', 'on', '--cache-type-k', 'f16', '--cache-type-v', 'f16',
            '--threads', '4', '--threads-batch', '4', '--batch-size', '256', '--ubatch-size', '128',
            '--jinja', '--conversation', '--single-turn', '--simple-io', '--no-display-prompt',
            '--color', 'off', '--no-context-shift', '--no-warmup', '--perf', '--log-colors', 'off',
            '--temp', '0.7', '--top-p', '0.8', '--top-k', '20', '--min-p', '0',
            '--presence-penalty', '1.5', '--repeat-penalty', '1', '--seed', '1234', '--predict', str(output)]
    if fitted:
        for flag in ('--gpu-layers', '--override-tensor', '--fit'):
            position = argv.index(flag)
            del argv[position:position+2]
        argv += ['--fit', 'on', '--fit-target', '1536', '--verbose', '--load-mode', 'dio']
    if model_id == 'bonsai-stock':
        argv += ['--reasoning', 'off']
    if spec['template']:
        argv += ['--chat-template-file', spec['template']]
    return argv + ['--system-prompt', SYSTEM_PROMPT, '--prompt', prompt]


def write_new(path, value):
    data = value if isinstance(value, bytes) else (json.dumps(value, indent=2) + '\n').encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    with os.fdopen(descriptor, 'wb') as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())


def review_plan(prompt, ids):
    check_prompt(prompt)
    select_models(ids)
    return {
        'schema': 'PMI_SERIAL_REAL_TEXT_CANDIDATE_V1', 'status': 'HELD',
        'actual_inference': False, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'input_mode': 'same operator prompt for each selected model; comparison outputs',
        'stages': [{'model_id': key, 'track': MODELS[key]['track'], 'kind': 'text',
                    'model': MODELS[key], 'native_argv': native_argv(key, prompt)} for key in ids],
        'release_rule': 'process exits, host/GPU owned cleanup and postflight pass BEFORE next load',
        'bounds': {'whole_job_seconds': 240, 'per_stage_seconds_max': 120, 'RAM_admit_GiB': 16,
                   'RAM_floor_GiB': 12, 'RSS_GiB': 12, 'AS_GiB': 16, 'global_VRAM_GiB': 7,
                   'native_output_bytes': MAX_OUTPUT, 'threads': 4, 'nice': 10},
        'remaining': ['normal platform review for each physical execution',
                      'reviewed exact-prompt fit evidence for any prompt beyond the pinned welcome smoke',
                      'separate model qualification for other selected profiles'],
        'unavailable_modalities': ['image', 'video', 'audio'],
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py'))},
    }


def decode_answer(raw):
    if len(raw) > MAX_OUTPUT:
        raise ValueError('native output exceeds cap')
    text = raw.decode('utf-8', errors='strict').rstrip()
    suffix = ' [end of text]'
    if not text.endswith(suffix):
        raise ValueError('native response incomplete: natural completion suffix absent')
    answer = text[:-len(suffix)].strip()
    if not answer:
        raise ValueError('empty native answer')
    return answer



def _progress(root, stage, status, **details):
    """Persist and print short user-facing stage updates."""
    record = {'stage':stage, 'status':status, **details}
    with (root/'progress.jsonl').open('ab') as events:
        events.write((json.dumps(record,ensure_ascii=False)+'\n').encode())
        events.flush();os.fsync(events.fileno())
    detail = ' '.join(f'{key}={value}' for key,value in details.items())
    print(f"[{stage}] {status}" + (f" {detail}" if detail else ''), flush=True)


def _write_resume_provenance(root, prompt, stage, answer):
    """Pin completed text-stage evidence so speech can resume without text launch."""
    files=[root/'prompt.txt',root/'plan.json',stage/'request.json',stage/'answer.txt',
           stage/'native.stdout.log',stage/'native.stderr.log',stage/'result.json',
           stage/'placement.json',stage/'host.jsonl',stage/'vram.jsonl',stage/'postflight.json']
    if any(not path.is_file() for path in files):
        return None
    record={'schema':'PMI_VERIFIED_TEXT_RESUME_V1','source_job':str(root.resolve()),
            'model_id':'bonsai-stock','model_sha256':MODELS['bonsai-stock']['sha256'],
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'answer':answer,'answer_sha256':hashlib.sha256(answer.encode()).hexdigest(),
            'answer_file_sha256':hashlib.sha256((stage/'answer.txt').read_bytes()).hexdigest(),
            'native_stdout_sha256':hashlib.sha256((stage/'native.stdout.log').read_bytes()).hexdigest(),
            'files_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
            'text_completion':'RealTextAndReleasePass','source_owned_cleanup_empty':True,
            'source_postflight_passed':True,'text_model_launch_on_resume':False}
    manifest=root/'resume-speech-provenance.json';write_new(manifest,record)
    write_new(root/'text-ready.json',{'schema':'PMI_TEXT_RESUME_READY_V1',
             'provenance_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
             'answer_sha256':record['answer_sha256'],'status':'RealTextAndReleasePass'})
    return str(manifest)


def _progress(root, stage, status, **details):
    """Persist and print short user-facing stage updates."""
    record = {'stage':stage, 'status':status, **details}
    with (root/'progress.jsonl').open('ab') as events:
        events.write((json.dumps(record,ensure_ascii=False)+'\n').encode())
        events.flush();os.fsync(events.fileno())
    detail = ' '.join(f'{key}={value}' for key,value in details.items())
    print(f"[{stage}] {status}" + (f" {detail}" if detail else ''), flush=True)


def _write_resume_provenance(root, prompt, stage, answer):
    """Pin completed text-stage evidence so speech can resume without text launch."""
    files=[root/'prompt.txt',root/'plan.json',stage/'request.json',stage/'answer.txt',
           stage/'native.stdout.log',stage/'native.stderr.log',stage/'result.json',
           stage/'placement.json',stage/'host.jsonl',stage/'vram.jsonl',stage/'postflight.json']
    if any(not path.is_file() for path in files):
        return None
    record={'schema':'PMI_VERIFIED_TEXT_RESUME_V1','source_job':str(root.resolve()),
            'model_id':'bonsai-stock','model_sha256':MODELS['bonsai-stock']['sha256'],
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'answer':answer,'answer_sha256':hashlib.sha256(answer.encode()).hexdigest(),
            'answer_file_sha256':hashlib.sha256((stage/'answer.txt').read_bytes()).hexdigest(),
            'native_stdout_sha256':hashlib.sha256((stage/'native.stdout.log').read_bytes()).hexdigest(),
            'files_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
            'text_completion':'RealTextAndReleasePass','source_owned_cleanup_empty':True,
            'source_postflight_passed':True,'text_model_launch_on_resume':False}
    manifest=root/'resume-speech-provenance.json';write_new(manifest,record)
    write_new(root/'text-ready.json',{'schema':'PMI_TEXT_RESUME_READY_V1',
             'provenance_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
             'answer_sha256':record['answer_sha256'],'status':'RealTextAndReleasePass'})
    return str(manifest)

def _run_sequence(prompt, ids, root, lease, backend, speech=None, image=None):
    """Private orchestration seam. CPU fixtures are labeled ContractOnly, never real success."""
    review_plan(prompt, ids)
    if (backend.actual_inference or (speech is not None and speech.actual_inference)
            or (image is not None and image.actual_inference)):
        require_real_execution()
    if speech is not None and len(ids) != 1:
        raise ValueError('speech requires exactly one selected text model')
    if image is not None and speech is None:
        raise ValueError('image stage requires the verified text and speech sequence')
    speech_contract = _speech_contract(speech) if speech is not None else None
    if speech is not None and hasattr(speech, 'ensure_ready'):
        # Fail before creating a job or running text when the explicitly selected
        # audio model lacks its weights, voice pack, seal or reviewed runtime.
        speech.ensure_ready()
    root.mkdir(mode=0o700, exist_ok=False)
    descriptor = os.open(lease, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    deadline = time.monotonic() + 240
    with os.fdopen(descriptor, 'rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_new(root / 'prompt.txt', prompt.encode())
        write_new(root / 'plan.json', review_plan(prompt, ids))
        write_new(root / 'progress.jsonl', b'')
        rows = []
        active_stage = 'text'
        try:
            for number, key in enumerate(ids):
                if time.monotonic() >= deadline - 20:
                    raise TimeoutError('whole-job final reserve reached')
                stage = root / f'stage-{number}-{key}'
                stage.mkdir(mode=0o700)
                active_stage = f'text:{key}'
                _progress(root, active_stage, 'running')
                write_new(stage / 'request.json', {'model_id': key, 'prompt': prompt})
                # Synchronous backend must prove unload before returning. No servers/resident workers.
                capture = backend.run_stage(key, prompt, stage, min(deadline - 20, time.monotonic() + 120))
                if capture['returncode'] != 0 or not capture['owned_cleanup_empty'] or not capture['postflight_passed']:
                    raise RuntimeError('native failure or release unproven; all later models held')
                answer = decode_answer(capture['raw_stdout'])
                write_new(stage / 'answer.txt', (answer + '\n').encode())
                write_new(stage / 'result.json', {
                    'model_id': key, 'track': MODELS[key]['track'], 'actual_inference': backend.actual_inference,
                    'status': 'RealTextAndReleasePass' if backend.actual_inference else 'ContractOnly',
                    'raw_sha256': hashlib.sha256(capture['raw_stdout']).hexdigest(),
                    'answer_sha256': hashlib.sha256(answer.encode()).hexdigest(),
                    'owned_cleanup_empty': True, 'postflight_passed': True,
                })
                resume_provenance = None
                if key == 'bonsai-stock' and backend.actual_inference:
                    resume_provenance = _write_resume_provenance(root,prompt,stage,answer)
                rows.append({'model_id':key,'status':'RealTextAndReleasePass' if backend.actual_inference else 'ContractOnly',
                             'answer_file':str(stage/'answer.txt'),'resume_provenance':resume_provenance})
                _progress(root,active_stage,'complete',answer_file=str(stage/'answer.txt'))
            if speech is not None:
                if time.monotonic() >= deadline - 120:
                    raise TimeoutError('insufficient whole-job reserve for speech')
                spoken = bounded_read(Path(rows[0]['answer_file']), 4096).decode('utf-8').strip()
                speech_contract['check_text'](spoken)
                audio_stage = root / 'speech'
                audio_stage.mkdir(mode=0o700)
                active_stage = speech_contract['stage']
                _progress(root,active_stage,'running',text_source=str(rows[0]['answer_file']))
                speech_request = {'text': spoken,
                                  'text_sha256': hashlib.sha256(spoken.encode()).hexdigest()}
                if speech_contract['model_id'] == KOKORO_ROLE_ID:
                    speech_request.update({'model_id': KOKORO_ROLE_ID,
                                           'voice_id': speech_contract['voice_id']})
                write_new(audio_stage / 'request.json', speech_request)
                capture = speech.run_stage(spoken, audio_stage, min(deadline - 20, time.monotonic() + 120))
                if capture['returncode'] != 0 or not capture['owned_cleanup_empty'] or not capture['postflight_passed']:
                    raise RuntimeError('speech failure or release unproven')
                audio = speech_contract['validate_wav'](audio_stage / 'answer.wav')
                speech_result = {
                    'status': 'RealSpeechAndReleasePass' if speech.actual_inference else 'ContractOnly',
                    'actual_inference': speech.actual_inference, **audio}
                speech_row = {'voice_id': speech_contract['voice_id'],
                              'status': speech_result['status'],
                              'audio_file': str(audio_stage / 'answer.wav')}
                if speech_contract['model_id'] == KOKORO_ROLE_ID:
                    kokoro_metadata = _kokoro_capture_metadata(capture, speech.actual_inference)
                    speech_result.update({'model_id': KOKORO_ROLE_ID,
                                          'voice_id': speech_contract['voice_id'],
                                          'voice_language': speech_contract['language'],
                                          **kokoro_metadata})
                    speech_row.update({'model_id': KOKORO_ROLE_ID,
                                       'voice_language': speech_contract['language'],
                                       'sample_rate': audio['sample_rate'],
                                       'adapter': audio['adapter'],
                                       **kokoro_metadata})
                write_new(audio_stage / 'result.json', speech_result)
                rows.append(speech_row)
                _progress(root,active_stage,'complete',audio_file=str(audio_stage/'answer.wav'))
            if image is not None:
                # Last stage: text and speech receipts stay valid if the image stop.
                if time.monotonic() >= deadline - 60:
                    raise TimeoutError('insufficient whole-job reserve for image')
                from comfy_backend import check_prompt as check_image_prompt, validate_artifact
                image_stage = root / 'image'
                image_stage.mkdir(mode=0o700)
                active_stage = 'image:comfyui-local'
                source_text = bounded_read(Path(rows[0]['answer_file']), 4096).decode('utf-8').strip()
                check_image_prompt(source_text)
                endpoint = image.endpoint
                _progress(root,active_stage,'running',text_source=str(rows[0]['answer_file']),endpoint=endpoint)
                write_new(image_stage / 'request.json', {
                    'prompt': source_text, 'text_source': str(rows[0]['answer_file']),
                    'prompt_sha256': hashlib.sha256(source_text.encode()).hexdigest(),
                    'endpoint': endpoint})
                receipt = image.run_stage(source_text, image_stage, min(deadline - 20, time.monotonic() + 120))
                if not receipt.get('release_proven'):
                    raise RuntimeError('image failure or ComfyUI unload unproven')
                artifact = validate_artifact(bounded_read(Path(receipt['artifact_file']), 32 * 1048576))
                if artifact != receipt['artifact']:
                    raise RuntimeError('saved image differs from the validated artifact receipt')
                image_status = 'RealImageAndReleasePass' if image.actual_inference else 'ContractOnly'
                write_new(image_stage / 'result.json', {
                    'status': image_status, 'actual_inference': image.actual_inference,
                    'endpoint': endpoint, 'prompt_id': receipt.get('prompt_id'),
                    'release_proven': True, 'automatic_retry': False, **artifact})
                rows.append({'modality':'image','endpoint':endpoint,'status':image_status,
                             'image_file': receipt['artifact_file'],
                             'image_sha256': artifact['sha256']})
                _progress(root,active_stage,'complete',image_file=receipt['artifact_file'])
            result = {'status':'CompletedRealSpeechAndImage' if image is not None and backend.actual_inference
                          else 'CompletedRealSpeech' if speech is not None and backend.actual_inference
                          else 'CompletedRealText' if backend.actual_inference else 'ContractOnly',
                      'actual_inference':backend.actual_inference,'stages':rows,'automatic_retry':False,
                      'paths':{'prompt':str(root/'prompt.txt'),'result':str(root/'result.json'),'progress':str(root/'progress.jsonl'),
                               'answer':rows[0].get('answer_file') if rows else None,
                               'wav':next((row['audio_file'] for row in rows if 'audio_file' in row),None),
                               'image':next((row['image_file'] for row in rows if 'image_file' in row),None),
                               'image_receipt':str(root/'image'/'result.json') if (root/'image'/'result.json').exists() else None}}
        except Exception as error:
            _progress(root,active_stage,'failed',error=str(error))
            result = {'status': 'StoppedNoRetry', 'actual_inference': backend.actual_inference,
                      'completed_stages':rows,'error':str(error),'later_models_held':True,
                      'automatic_retry':False,
                      'paths':{'result':str(root/'result.json'),'progress':str(root/'progress.jsonl'),
                               'answer':rows[0].get('answer_file') if rows else None,
                               'image':next((row['image_file'] for row in rows if 'image_file' in row),None),
                               'resume_provenance':str(root/'resume-speech-provenance.json') if (root/'resume-speech-provenance.json').exists() else None}}
            write_new(root / 'result.json', result)
            raise
        write_new(root / 'result.json', result)
        return result



def _resume_speech(source_job, root, lease, speech):
    """Fresh lease/admission and selected CPU speech only; never creates a text backend."""
    if speech.actual_inference:
        require_real_execution()
    from piper_backend import verified_resume_text
    speech_contract = _speech_contract(speech)
    if hasattr(speech, 'ensure_ready'):
        speech.ensure_ready()
    deadline = time.monotonic() + 240
    descriptor = os.open(lease, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        text, prompt, provenance = verified_resume_text(source_job, min(deadline-140,time.monotonic()+60))
        root.mkdir(mode=0o700, exist_ok=False)
        write_new(root/'prompt.txt', prompt.encode())
        write_new(root/'progress.jsonl',b'')
        _progress(root,'text:bonsai-stock','reused',source_answer=str(source_job/'stage-0-bonsai-stock/answer.txt'))
        plan_record = {'status':'VerifiedTextResume',
                       'stages':[{'model_id':'bonsai-stock','source_job':str(source_job)}],
                       'provenance':provenance,'text_model_launch':False}
        if speech_contract['model_id'] == KOKORO_ROLE_ID:
            plan_record['speech'] = {'model_id': KOKORO_ROLE_ID,
                                     'voice_id': speech_contract['voice_id'],
                                     'voice_language': speech_contract['language']}
        write_new(root/'plan.json', plan_record)
        stage = root/'speech'
        stage.mkdir(mode=0o700)
        active_stage = speech_contract['stage']
        _progress(root,active_stage,'running',text_source=str(source_job/'stage-0-bonsai-stock/answer.txt'))
        request = {'text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest()}
        if speech_contract['model_id'] == KOKORO_ROLE_ID:
            request.update({'model_id': KOKORO_ROLE_ID,
                            'voice_id': speech_contract['voice_id']})
        write_new(stage/'request.json', request)
        try:
            capture = speech.run_stage(text,stage,min(deadline-20,time.monotonic()+120))
            if capture['returncode'] != 0 or not capture['owned_cleanup_empty'] or not capture['postflight_passed']:
                raise RuntimeError('speech failure or release unproven')
            audio = speech_contract['validate_wav'](stage/'answer.wav')
            result = {'status':'RealSpeechAndReleasePass' if speech.actual_inference else 'ContractOnly',
                      'actual_inference':speech.actual_inference,'text_model_launch':False,
                      'source_job':str(source_job),'automatic_retry':False,**audio,
                      'paths':{'answer':str(source_job/'stage-0-bonsai-stock/answer.txt'),
                               'wav':str(stage/'answer.wav'),'result':str(root/'result.json'),
                               'speech_result':str(stage/'result.json'),'progress':str(root/'progress.jsonl')}}
            if speech_contract['model_id'] == KOKORO_ROLE_ID:
                kokoro_metadata = _kokoro_capture_metadata(capture, speech.actual_inference)
                result.update({'speech_model': KOKORO_ROLE_ID,
                               'voice_id': speech_contract['voice_id'],
                               'voice_language': speech_contract['language'],
                               **kokoro_metadata})
            write_new(stage/'result.json',result)
            _progress(root,active_stage,'complete',audio_file=str(stage/'answer.wav'))
        except Exception as error:
            _progress(root,active_stage,'failed',error=str(error))
            write_new(root/'result.json',{'status':'StoppedNoRetry','error':str(error),'automatic_retry':False,'text_model_launch':False})
            raise
        write_new(root/'result.json',result)
        return result

def main(argv=None):
    values = sys.argv[1:] if argv is None else argv
    if values and values[0].startswith('research-mode-'):
        from research.runtime_entry import main as runtime_main
        return runtime_main(values)
    if values and values[0].startswith('research-experiment-'):
        from research.executor import main as experiment_main
        return experiment_main(values)
    if values and values[0].startswith('research-lab-'):
        from research.laboratory import main as laboratory_main
        return laboratory_main(values)
    if values and values[0].startswith('research-'):
        from research.cli import main as research_main
        return research_main(values)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['models', 'plan', 'run', 'speak', 'speak-image',
                                           'speech-plan', 'image-plan', 'resume-speech'])
    parser.add_argument('--prompt-file', type=Path)
    parser.add_argument('--models', default=DEFAULT_MODEL)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--lease', type=Path)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--comfy-endpoint', default=None)
    parser.add_argument('--speech-model', default=None,
                        help='speech backend: piper-cpu (default) or kokoro-82m')
    parser.add_argument('--speech-voice', default=None,
                        help='Kokoro voice id, for example af_heart or pf_dora')
    args = parser.parse_args(argv)
    if args.action == 'models':
        print(json.dumps(MODELS, indent=2))
        return 0
    # Preserve denial BEFORE any output directory, scope probe, helper import or launch.
    if args.action in ('run', 'speak', 'speak-image', 'resume-speech'):
        require_real_execution()
    if args.action == 'resume-speech':
        if args.resume_from is None or args.out is None or args.lease is None:
            parser.error('--resume-from, --out and --lease required')
        speech = create_speech_backend(args.speech_model if args.speech_model is not None
                                       else DEFAULT_SPEECH_MODEL_ID,
                                       args.speech_voice)
        print(json.dumps(_resume_speech(args.resume_from,args.out,args.lease,speech),indent=2))
        return 0
    speech_actions = ('speech-plan', 'speak', 'speak-image')
    if args.action not in speech_actions and (args.speech_model is not None or args.speech_voice is not None):
        parser.error('--speech-model and --speech-voice require speech-plan, speak, speak-image or resume-speech')
    if args.prompt_file is None or args.out is None:
        parser.error('--prompt-file and --out required')
    prompt = bounded_read(args.prompt_file, MAX_PROMPT).decode('utf-8', errors='strict')
    ids = select_models(args.models.split(','))
    if args.action in ('plan', 'speech-plan', 'image-plan'):
        plan = review_plan(prompt, ids)
        if args.action in ('speech-plan', 'image-plan') and len(ids) != 1:
            raise ValueError('speech and image require exactly one selected text model')
        if args.action == 'speech-plan':
            selected_speech_model = (args.speech_model if args.speech_model is not None
                                     else DEFAULT_SPEECH_MODEL_ID)
            speech = create_speech_backend(selected_speech_model, args.speech_voice)
            if selected_speech_model == KOKORO_ROLE_ID:
                speech_plan_record = speech.plan()
            else:
                from piper_backend import speech_plan
                speech_plan_record = speech_plan()
            speech_plan_record['model_id'] = selected_speech_model
            plan['speech'] = speech_plan_record
            plan['unavailable_modalities'] = ['image', 'video']
        if args.action == 'image-plan':
            from comfy_backend import image_plan, resolve_endpoint
            plan['image'] = image_plan(resolve_endpoint(args.comfy_endpoint), prompt)
            plan['unavailable_modalities'] = ['video', 'audio']
        write_new(args.out, plan)
        print(str(args.out))
        return 0
    if args.lease is None:
        parser.error('--lease required')
    from llama_backend import LlamaBackend
    speech = None
    if args.action in ('speak', 'speak-image'):
        speech = create_speech_backend(args.speech_model if args.speech_model is not None
                                       else DEFAULT_SPEECH_MODEL_ID,
                                       args.speech_voice)
    image = None
    if args.action == 'speak-image':
        from comfy_backend import ComfyBackend
        image = ComfyBackend(args.comfy_endpoint)
    result = _run_sequence(prompt, ids, args.out, args.lease, LlamaBackend(), speech, image)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(125)
