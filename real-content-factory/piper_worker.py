"""Fixed CPU-only speech worker; no model or dependency downloading."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import wave

from launch_policy import require_real_execution
from content_factory import bounded_read
from piper_backend import VOICE, MAX_WAV, check_speech_text, verify_speech_inputs
import time



def validate_voice_config(config):
    """Accept only the language identifiers in the pinned U.S. LJSpeech config."""
    if (not isinstance(config, dict) or config.get('phoneme_type') != 'espeak'
            or not isinstance(config.get('espeak'), dict)
            or config['espeak'].get('voice') != 'en'
            or not isinstance(config.get('language'), dict)
            or config['language'].get('code') != 'en_US'):
        raise ValueError('pinned English (en/espeak, en_US) voice configuration required')
    return True

def main():
    require_real_execution()  # before imports of synthesis runtime or any output
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    raw = bounded_read(args.request, 32768)
    request = json.loads(raw)
    if set(request) != {'text', 'text_sha256'}:
        raise ValueError('unexpected speech request fields')
    text = request['text']
    check_speech_text(text)
    if hashlib.sha256(text.encode()).hexdigest() != request['text_sha256']:
        raise ValueError('speech text identity mismatch')
    verify_speech_inputs(time.monotonic() + 10)
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_WAV, MAX_WAV))
    for number, name in ((1, 'native.stdout.log'), (2, 'native.stderr.log')):
        fd = os.open(args.request.parent / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.dup2(fd, number)
        os.close(fd)
    from piper import PiperVoice
    from piper.config import PiperConfig
    import onnxruntime
    config = json.loads(bounded_read(str(VOICE) + '.json', 65536))
    validate_voice_config(config)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    voice = PiperVoice(config=PiperConfig.from_dict(config),
                       session=onnxruntime.InferenceSession(str(VOICE), sess_options=options,
                                                          providers=['CPUExecutionProvider']))
    if voice.session.get_providers() != ['CPUExecutionProvider']:
        raise RuntimeError('CPU-only ONNX provider required')
    # The WAV is created exclusively; duration and truncation are checked by parent.
    with (args.request.parent / 'answer.wav').open('xb') as output:
        with wave.open(output, 'wb') as wav:
            voice.synthesize_wav(text, wav)


if __name__ == '__main__':
    main()
