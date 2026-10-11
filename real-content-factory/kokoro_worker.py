"""Isolated local Kokoro worker. The caller enforces the physical execution hold."""
import argparse
from array import array
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import wave
import io

from content_factory import bounded_read, write_new
from launch_policy import require_real_execution, require_approved_kokoro_sample
import kokoro_backend as adapter
from model_intent import KOKORO_REPOSITORY


THREADS = 4


def load_runtime():
    """Import only in the owned worker, and accept only the reviewed API version."""
    if not (sys.version_info[:2] >= (3, 10) and sys.version_info[:2] < (3, 13)):
        raise RuntimeError('Kokoro 0.9.4 requires Python >=3.10,<3.13')
    from importlib import metadata

    version = metadata.version('kokoro')
    if version != adapter.RUNTIME_VERSION:
        raise RuntimeError(f'Kokoro API version mismatch: expected {adapter.RUNTIME_VERSION}, got {version}')
    import torch
    from kokoro import KModel, KPipeline
    return type('KokoroRuntime', (), {
        'version': version, 'torch': torch, 'KModel': KModel, 'KPipeline': KPipeline,
    })


load_runtime.actual_inference = True


def _audio_values(audio, remaining_frames):
    """Convert one official KPipeline audio tensor to bounded mono float values."""
    if audio is None:
        raise ValueError('KPipeline returned no audio for a nonempty text chunk')
    if hasattr(audio, 'detach'):
        audio = audio.detach()
    if hasattr(audio, 'cpu'):
        audio = audio.cpu()
    shape = getattr(audio, 'shape', None)
    if shape is not None:
        shape = tuple(int(axis) for axis in shape)
        if len(shape) == 2 and 1 in shape:
            if shape[0] == 1:
                shape = (shape[1],)
            elif shape[1] == 1:
                shape = (shape[0],)
        if len(shape) != 1:
            raise ValueError('KPipeline audio must be one-dimensional mono samples')
        if shape[0] > remaining_frames:
            raise ValueError('Kokoro output exceeds the 30-second WAV limit')
    if hasattr(audio, 'numel') and int(audio.numel()) > remaining_frames:
        raise ValueError('Kokoro output exceeds the 30-second WAV limit')
    if hasattr(audio, 'tolist'):
        audio = audio.tolist()
    if isinstance(audio, (str, bytes, bytearray)):
        raise ValueError('KPipeline audio must be numeric samples')
    try:
        values = list(audio)
    except TypeError as error:
        raise ValueError('KPipeline audio must be an iterable of samples') from error
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ValueError('KPipeline audio must be mono')
        values = list(values[0])
    if not values or len(values) > remaining_frames:
        raise ValueError('Kokoro output is empty or exceeds the 30-second WAV limit')
    samples = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError('KPipeline audio contains a nonnumeric sample')
        try:
            sample = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError('KPipeline audio contains a nonnumeric sample') from error
        if not math.isfinite(sample):
            raise ValueError('KPipeline audio contains a nonfinite sample')
        sample = min(1.0, max(-1.0, sample))
        samples.append(int(round(sample * (32768 if sample < 0 else 32767))))
    pcm = array('h', samples)
    if sys.byteorder != 'little':
        pcm.byteswap()
    return pcm.tobytes(), len(samples)


def _pipeline_audio(item):
    """Read the documented ``(graphemes, phonemes, audio)`` result tuple."""
    try:
        values = tuple(item)
    except TypeError as error:
        raise ValueError('KPipeline yielded an unsupported result') from error
    if len(values) != 3:
        raise ValueError('KPipeline result must contain graphemes, phonemes, and audio')
    return values[2]


def _wav_bytes(pcm):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(adapter.CHANNELS)
        wav.setsampwidth(adapter.SAMPLE_WIDTH_BYTES)
        wav.setframerate(adapter.SAMPLE_RATE)
        wav.writeframes(pcm)
    data = buffer.getvalue()
    if len(data) > adapter.MAX_WAV:
        raise ValueError('Kokoro WAV exceeds its bounded output size')
    return data


def synthesize_wav_bytes(text, voice_id, *, model_file=None, config_file=None,
                         voice_file=None, runtime_loader=load_runtime):
    """Use the pinned official KModel/KPipeline API on explicit local files only.

    ``runtime_loader`` is an injection seam for model-free contract tests. The
    production worker passes the real lazy loader after the launch gate.
    """
    adapter.check_text(text)
    voice_id = adapter.check_voice(voice_id)
    model_file = Path(model_file or adapter.MODEL_FILE)
    config_file = Path(config_file or adapter.MODEL_CONFIG_FILE)
    voice_file = Path(voice_file or adapter.voice_pack(voice_id))
    language_code = {'af_heart': 'a', 'pf_dora': 'p'}[voice_id]

    runtime = runtime_loader()
    if runtime.version != adapter.RUNTIME_VERSION:
        raise RuntimeError(f'Kokoro API version mismatch: expected {adapter.RUNTIME_VERSION}, got {runtime.version}')
    torch = runtime.torch
    model = pipeline = generator = None
    try:
        torch.set_num_threads(THREADS)
        torch.set_num_interop_threads(1)
        model = runtime.KModel(repo_id=KOKORO_REPOSITORY,
                               config=str(config_file), model=str(model_file))
        model = model.to('cpu').eval()
        pipeline = runtime.KPipeline(lang_code=language_code,
                                     repo_id=KOKORO_REPOSITORY,
                                     model=model, device='cpu')
        pcm = bytearray()
        frames = 0
        with torch.inference_mode():
            generator = pipeline(text, voice=str(voice_file), speed=1.0,
                                 split_pattern=r'\n+')
            for item in generator:
                part, count = _audio_values(_pipeline_audio(item),
                                            adapter.MAX_FRAMES - frames)
                pcm.extend(part)
                frames += count
                if len(pcm) > adapter.MAX_FRAMES * adapter.SAMPLE_WIDTH_BYTES:
                    raise ValueError('Kokoro output exceeds the 30-second WAV limit')
        if frames == 0:
            raise ValueError('KPipeline produced no audio chunks')
        return _wav_bytes(pcm)
    finally:
        try:
            if generator is not None:
                close = getattr(generator, 'close', None)
                if close is not None:
                    close()
        finally:
            del generator
            del pipeline
            del model
            gc.collect()


def process_request(request, stage, *, runtime_loader=load_runtime):
    """Synthesize and validate before atomically creating the worker WAV."""
    stage = Path(stage)
    if (not isinstance(request, dict)
            or set(request) != {'model_id', 'text', 'voice_id', 'text_sha256'}
            or request.get('model_id') != adapter.MODEL_ID):
        raise ValueError('unexpected Kokoro request fields or model id')
    text = request['text']
    voice_id = adapter.check_voice(request['voice_id'])
    adapter.check_text(text)
    if request['text_sha256'] != hashlib.sha256(text.encode('utf-8')).hexdigest():
        raise ValueError('Kokoro request text identity mismatch')
    wav_data = synthesize_wav_bytes(text, voice_id, runtime_loader=runtime_loader)
    output_path = stage / 'answer.wav'
    created_identity = None
    try:
        write_new(output_path, wav_data)
        from llama_backend import file_identity
        created_identity = file_identity(output_path)
        receipt = adapter.validate_wav(output_path)
        if receipt['wav_sha256'] != hashlib.sha256(wav_data).hexdigest():
            raise ValueError('saved Kokoro WAV differs from synthesized bytes')
        actual_inference = bool(getattr(runtime_loader, 'actual_inference', True))
        return {'model_id': adapter.MODEL_ID, 'adapter': adapter.ADAPTER_ID,
                'runtime_version': adapter.RUNTIME_VERSION,
                'runtime_code_status': ('executed-this-receipt' if actual_inference
                                        else 'mocked-runtime-boundary'),
                'runtime_qualification': 'unqualified',
                'voice_id': voice_id, 'voice_language': adapter.VOICES[voice_id]['language'],
                'actual_inference': actual_inference, **receipt}
    except Exception:
        if created_identity is not None:
            try:
                from llama_backend import file_identity
                if file_identity(output_path) == created_identity:
                    output_path.unlink()
            except OSError:
                pass
        raise


def main(argv=None, *, runtime_loader=load_runtime):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args(argv)
    stage = args.request.parent
    raw = bounded_read(args.request, 32768)
    request = json.loads(raw)
    if (not isinstance(request, dict)
            or set(request) != {'model_id', 'text', 'voice_id', 'text_sha256'}
            or request.get('model_id') != adapter.MODEL_ID):
        raise ValueError('unexpected Kokoro request fields or model id')
    adapter.check_text(request.get('text'))
    adapter.check_voice(request.get('voice_id'))
    if request.get('text_sha256') != hashlib.sha256(request['text'].encode('utf-8')).hexdigest():
        raise ValueError('Kokoro request text identity mismatch')
    require_approved_kokoro_sample(request['text'], request['voice_id'])
    adapter.verify_kokoro_inputs(request['voice_id'],
                                 time.monotonic() + adapter.STAGE_SECONDS - 20)

    resource.setrlimit(resource.RLIMIT_FSIZE, (adapter.MAX_WAV, adapter.MAX_WAV))
    for descriptor_number, name in ((1, 'native.stdout.log'), (2, 'native.stderr.log')):
        descriptor = os.open(stage / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.dup2(descriptor, descriptor_number)
        os.close(descriptor)
    print(json.dumps(process_request(request, stage, runtime_loader=runtime_loader),
                     sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    main()
