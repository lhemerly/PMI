"""Closed, source-reviewed model settings. These values grant no execution authority."""
import hashlib
import json

PROFILE_ID = 'bonsai-research-v1'
MODEL_ID = 'bonsai-stock'
ROLE_OUTPUT_TOKENS = {
    'calibration': 512, 'generation': 512, 'evolution': 512,
    'reflection': 128, 'debate': 128, 'ranking': 128,
    'meta-review': 128, 'overview': 128,
}
FIT_POLICY = 'pinned-bonsai-qwen35-gpt2-complete-byte-alphabet-upper-bound-v1'
TEMPLATE_SHA256 = 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041'


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def profile_descriptor():
    return {
        'profile_id': PROFILE_ID, 'model_id': MODEL_ID,
        'catalog_id': 'Bonsai2-27B-PTQ1_0',
        'model_sha256': '53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3',
        'native_sha256': 'c1b84e353ab86f3444603f59c944283b1fe1043359585b8ddb843bd2a10f5686',
        'runtime_inventory_sha256': '89ea8bb1223e1063d8858dc61bccce05b8476a5d33bec6e5cd49fcf0058a9d98',
        'template_sha256': TEMPLATE_SHA256, 'tokenizer_family': 'bonsai-stock',
        'context_tokens': 4096, 'role_output_tokens': dict(ROLE_OUTPUT_TOKENS),
        'context_margin_tokens': 4, 'max_prompt_utf8_bytes': 4096,
        'automatic_fit': True, 'fit_target_MiB': 1536, 'load_mode': 'dio',
        'reasoning': 'off', 'threads': 4, 'threads_batch': 4,
        'batch_size': 256, 'ubatch_size': 128, 'seed': 1234,
        'sampling': {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20,
                     'min_p': 0, 'presence_penalty': 1.5, 'repeat_penalty': 1},
        'resources': {'VRAM_GiB': 7, 'RSS_GiB': 12, 'AS_GiB': 16,
                      'RAM_admit_GiB': 16, 'RAM_floor_GiB': 12,
                      'stage_seconds': 120, 'host_seconds': 60, 'GPU_seconds': 50},
    }


def profile_settings(profile_id, model_id, role):
    if (profile_id != PROFILE_ID or model_id != MODEL_ID
            or not isinstance(role, str) or role not in ROLE_OUTPUT_TOKENS):
        raise ValueError('unsupported explicit model execution profile/model/role')
    descriptor = profile_descriptor()
    return {'profile_id': PROFILE_ID, 'role': role,
            'profile_sha256': canonical_sha256(descriptor),
            'context_tokens': descriptor['context_tokens'],
            'output_tokens': ROLE_OUTPUT_TOKENS[role],
            'context_margin_tokens': descriptor['context_margin_tokens']}


def prompt_family_policy():
    """Must match a separately verified scoped seal; a JSON match is not approval."""
    return {'schema': 'PMI_REVIEWED_PROMPT_FAMILY_V1',
            'profile': profile_descriptor(),
            'profile_sha256': canonical_sha256(profile_descriptor()),
            'fit_policy': FIT_POLICY, 'every_actual_prompt_requires_fit': True,
            'upper_bound': 'rendered_utf8_bytes_plus_two_boundary_tokens',
            'trusted_source_scope': ['model_profiles.py', 'prompt_fit.py',
                                    'research/demo_prompts.py', 'research/laboratory.py',
                                    'research/harness.py', 'research/demo_authority.py'],
            'loop_model_calls': 18, 'separate_calibration_calls': 1,
            'experiment_calls': 3, 'loop_wall_seconds': 2400,
            'calibration_wall_seconds': 120}


def model_request(model_id, prompt, profile_id, role):
    settings = profile_settings(profile_id, model_id, role)
    if (not isinstance(prompt, str) or not prompt.strip() or '\0' in prompt
            or len(prompt.encode()) > 4096):
        raise ValueError('invalid bounded model prompt')
    return {'model_id': model_id, 'prompt': prompt, 'profile_id': profile_id,
            'role': role, 'profile_sha256': settings['profile_sha256']}


def validate_model_request(request):
    if not isinstance(request, dict) or set(request) != {
            'model_id', 'prompt', 'profile_id', 'role', 'profile_sha256'}:
        raise ValueError('unexpected explicit model request keys')
    if request != model_request(request['model_id'], request['prompt'],
                                request['profile_id'], request['role']):
        raise ValueError('explicit model request profile identity differs')
    return profile_settings(request['profile_id'], request['model_id'], request['role'])


def require_profile_fit(fit, model_id, prompt, profile_id, role):
    """Validate a newly computed fit receipt against this exact call, never authorize."""
    settings = profile_settings(profile_id, model_id, role)
    expected = {'profile_id': profile_id, 'role': role,
                'profile_sha256': settings['profile_sha256'],
                'policy': FIT_POLICY, 'template_sha256': TEMPLATE_SHA256,
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'context': settings['context_tokens'],
                'output_tokens_reserved': settings['output_tokens'],
                'context_margin_tokens': settings['context_margin_tokens'],
                'tensor_bytes_read': 0, 'exact_token_count': None}
    if not isinstance(fit, dict) or any(fit.get(k) != v for k, v in expected.items()):
        raise ValueError('actual prompt fit/profile/role identity differs')
    for key in ('formatted_utf8_bytes', 'input_token_upper_bound', 'metadata_bytes_read', 'layers'):
        if type(fit.get(key)) is not int or fit[key] < 0:
            raise ValueError('invalid conservative fit bound')
    if (fit['input_token_upper_bound'] != fit['formatted_utf8_bytes'] + 2
            or not 0 < fit['metadata_bytes_read'] <= 32 * 1048576
            or fit['layers'] != 65
            or fit['input_token_upper_bound'] + settings['output_tokens']
               + settings['context_margin_tokens'] > settings['context_tokens']):
        raise ValueError('actual prompt does not fit fixed context/output reserve')
    return settings
