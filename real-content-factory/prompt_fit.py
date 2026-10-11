"""Bounded metadata-only GGUF inspection and conservative GPT-2 byte-BPE fit."""
import hashlib
import struct
from pathlib import Path
import time

TEMPLATE_SHA256 = '9268aed773b364879d44189da349474a1c4d56271fc0e7c67bef81575828a4ca'
BONSAI_TEMPLATE_SHA256 = 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041'
SYSTEM_PROMPT = 'Answer concisely in plain text.'
CONTEXT = 4096
OUTPUT_TOKENS = 64
SMOKE_PROMPT_SHA256 = 'c6b2c373d9f19a22ae271d53ec61e7f8ef97c722e3876d6506ef586447a41ea6'


def prompt_context(prompt, profile):
    return 512 if profile == 'bonsai-stock' and hashlib.sha256(prompt.encode('utf-8')).hexdigest() == SMOKE_PROMPT_SHA256 else CONTEXT

FORMATS = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}


def gguf_metadata(path, deadline=None):
    """Read only the key/value prefix, never tensor metadata or tensor bytes."""
    deadline = deadline or time.monotonic() + 10
    cap = 32 * 1048576
    with Path(path).open('rb') as file:
        def read(count):
            if time.monotonic() >= deadline or count < 0 or file.tell() + count > cap:
                raise ValueError('GGUF metadata deadline/byte cap')
            raw = file.read(count)
            if len(raw) != count:
                raise ValueError('truncated GGUF metadata')
            return raw
        def integer(fmt):
            return struct.unpack('<' + fmt, read(struct.calcsize(fmt)))[0]
        def string():
            count = integer('Q')
            if count > 1048576:
                raise ValueError('GGUF string bound')
            return read(count).decode('utf-8', errors='strict')
        def value(kind, retain):
            if kind == 8:
                return string()
            if kind == 9:
                item = integer('I'); count = integer('Q')
                if item == 9 or count > 1000000:
                    raise ValueError('GGUF array bound')
                result = [] if retain else None
                for _ in range(count):
                    current = value(item, False)
                    if retain:
                        result.append(current)
                return result
            if kind not in FORMATS:
                raise ValueError('unknown GGUF metadata type')
            return integer(FORMATS[kind])
        if read(4) != b'GGUF' or integer('I') != 3:
            raise ValueError('expected GGUF v3')
        tensors = integer('Q'); count = integer('Q')
        if count > 4096 or tensors > 100000:
            raise ValueError('GGUF header bound')
        result = {}
        for _ in range(count):
            key = string(); kind = integer('I')
            if key in result:
                raise ValueError('duplicate GGUF metadata key')
            result[key] = value(kind, key == 'tokenizer.ggml.tokens')
        return result, file.tell()


def byte_alphabet():
    values = list(range(33,127)) + list(range(161,173)) + list(range(174,256))
    codepoints = values[:]
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte); codepoints.append(256 + extra); extra += 1
    return {chr(codepoint) for codepoint in codepoints}


def fit_prompt(prompt, model_path, template_path, deadline=None, profile='qwen3', *,
               profile_id=None, role=None):
    if not prompt.strip() or '\0' in prompt or len(prompt.encode('utf-8')) > 4096:
        raise ValueError('invalid bounded prompt')
    if profile not in ('qwen3', 'bonsai-stock'):
        raise ValueError('unsupported tokenizer fit profile')
    settings = None
    if profile_id is not None:
        from model_profiles import profile_settings
        settings = profile_settings(profile_id, profile, role)
    elif role is not None:
        raise ValueError('role requires explicit execution profile')
    template_hash = TEMPLATE_SHA256 if profile == 'qwen3' else BONSAI_TEMPLATE_SHA256
    architecture, pre = ('qwen3', 'qwen2') if profile == 'qwen3' else ('qwen35', 'qwen35')
    raw = Path(template_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != template_hash:
        raise ValueError('Qwen template differs from pinned stock source')
    metadata, prefix_bytes = gguf_metadata(model_path, deadline)
    if metadata.get('general.architecture') != architecture or metadata.get('tokenizer.ggml.model') != 'gpt2' or metadata.get('tokenizer.ggml.pre') != pre:
        raise ValueError('byte-BPE fit proof applies only to this stock tokenizer')
    if profile == 'bonsai-stock' and raw.decode() != metadata.get('tokenizer.chat_template'):
        raise ValueError('Bonsai official/embedded template mismatch')
    if not byte_alphabet().issubset(set(metadata.get('tokenizer.ggml.tokens', []))):
        raise ValueError('complete byte token alphabet unproven')
    # Stock Qwen GPT-2 BPE splits/merges byte symbols; it does not expand input
    # with unigram normalization. Count every UTF-8 byte conservatively plus
    # two optional boundary tokens. This is an upper bound, not an exact count.
    from jinja2 import Environment, StrictUndefined
    env = Environment(undefined=StrictUndefined)
    rendered = env.from_string(raw.decode()).render(
        messages=[{'role':'system','content':SYSTEM_PROMPT}, {'role':'user','content':prompt}],
        tools=None, add_generation_prompt=True, enable_thinking=False)
    upper = len(rendered.encode('utf-8')) + 2
    context = prompt_context(prompt, profile) if settings is None else settings['context_tokens']
    output = OUTPUT_TOKENS if settings is None else settings['output_tokens']
    margin = 0 if settings is None else settings['context_margin_tokens']
    if upper + output + margin > context:
        raise ValueError('prompt/template conservative token bound exceeds context reserve')
    result = {'policy':('pinned-qwen3-gpt2-complete-byte-alphabet-upper-bound-v1' if profile == 'qwen3' else 'pinned-bonsai-qwen35-gpt2-complete-byte-alphabet-upper-bound-v1'),
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
            'template_sha256':template_hash, 'formatted_utf8_bytes':len(rendered.encode()),
            'input_token_upper_bound':upper, 'output_tokens_reserved':output,
            'context':context, 'metadata_bytes_read':prefix_bytes,
            'layers':metadata[architecture + '.block_count'] + 1, 'exact_token_count':None,
            'tensor_bytes_read':0}
    if settings is not None:
        result.update({key: settings[key] for key in
                       ('profile_id', 'role', 'profile_sha256', 'context_margin_tokens')})
    return result
