"""Optional local ComfyUI image adapter for the serial content factory.

The submitted graph is the selected FLUX.2 Klein 4B distilled ComfyUI route, flattened
from the pinned official workflow snapshot in `workflows/`, using that model's real
node classes and model roles rather than a generic checkpoint graph.

The project-local ComfyUI/PyTorch environment and three Klein model files are
installed per `.runtime/comfyui/SETUP-RECEIPT.md`. No server, device workload,
inference, provider call or image publication is performed by this adapter's
offline planning path. Its asset inventory checks only paths and byte counts; it
does not read model weights or recompute their hashes. The adapter remains inert
until the normal platform hold is lifted. Standard library only. Text-to-Piper
ordering, shared lease, admission and unload sequencing are unchanged; the image
stage is appended after proven text release and speech.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import threading
import time
import uuid
from urllib.parse import quote, urlsplit
import urllib.error
import urllib.request
import zlib

from launch_policy import require_real_execution
from content_factory import bounded_read, write_new
from model_intent import IMAGE_ASSET_IDENTITIES, IMAGE_ASSET_REPOSITORY

DEFAULT_ENDPOINT = 'http://127.0.0.1:8188'
ENDPOINT_ENVIRONMENT = 'PMI_COMFY_ENDPOINT'
LOOPBACK_HOSTS = ('127.0.0.1', 'localhost', '::1')
RECEIPT_SCHEMA = 'PMI_COMFY_IMAGE_RECEIPT_V1'

MAX_PROMPT = 4096
MAX_WORKFLOW_BYTES = 262144
MAX_JSON_BYTES = 4 * 1048576
MAX_ARTIFACT_BYTES = 32 * 1048576
REQUEST_TIMEOUT_SECONDS = 30
POLL_INTERVAL_SECONDS = 2.0
IMAGE_STAGE_SECONDS = 120
RELEASE_POLL_SECONDS = 1.0
RELEASE_TIMEOUT_SECONDS = 15
RELEASE_VRAM_TOLERANCE_BYTES = 64 * 1048576
SAMPLER_ERROR = 'comfy node reported execution error'

SELECTED_HARNESS = 'ComfyUI'
SELECTED_MODEL = 'FLUX.2 Klein 4B distilled'
COMFYUI_ROOT = Path(__file__).resolve().parent / '.runtime/comfyui/app'
SETUP_RECEIPT = '.runtime/comfyui/SETUP-RECEIPT.md'

# Pinned official Comfy-Org workflow snapshot kept as JSON source data only.
SOURCE_WORKFLOW_FILE = 'workflows/official_flux2_klein_text_to_image.json'
SOURCE_WORKFLOW_BYTES = 69872
SOURCE_WORKFLOW_SHA256 = '8bb879856e54765b6c3ed219181e855981d8d10cfd19acff836603d9b56e838f'
SOURCE_UPSTREAM_REPOSITORY = 'Comfy-Org/workflow_templates'
SOURCE_UPSTREAM_PATH = 'templates/image_flux2_klein_text_to_image.json'
SOURCE_UPSTREAM_REVISION = 'e7cd011d4ded3411c2f481200544f0be6fdc962e'
SOURCE_UPSTREAM_COMMIT = 'chore(templates): add model file sizes to download notes'
SOURCE_CATALOG = ('https://comfy.org/workflows/'
                  'image_flux2_klein_text_to_image-814fd547d86e/')
SOURCE_WORKFLOW_ID = '92112d97-bb64-4b44-86f2-ea5691ef8f6e'
SOURCE_WORKFLOW_REVISION = 0
SOURCE_WORKFLOW_VERSION = 0.4
SELECTED_SUBGRAPH_ID = 'a67caa28-5f85-4917-8396-36004960dd30'
SELECTED_SUBGRAPH_NAME = 'Text to Image (Flux.2 Klein 4B Distilled)'
EXCLUDED_SUBGRAPH_ID = '7b34ab90-36f9-45ba-a665-71d418f0df18'
EXCLUDED_SUBGRAPH_NAME = 'Text to Image (Flux.2 Klein 4B)'

# Model roles named by the pinned workflow's own loaders and embedded download note.
TEXT_ENCODER_NODE = '2'
TEXT_ENCODER_FILE = 'qwen_3_4b.safetensors'
TEXT_ENCODER_FOLDER = 'models/text_encoders'
TEXT_ENCODER_TYPE = 'flux2'
TEXT_ENCODER_SOURCE_BYTES = 7.49e9
DIFFUSION_NODE = '1'
DIFFUSION_MODEL = 'flux-2-klein-4b.safetensors'
DIFFUSION_FOLDER = 'models/diffusion_models'
DIFFUSION_SOURCE_BYTES = 7.22e9
UNSELECTED_DIFFUSION_MODEL = 'flux-2-klein-base-4b.safetensors'
DOCUMENTED_ALTERNATE_DIFFUSION_MODEL = 'flux-2-klein-4b-fp8.safetensors'
WEIGHT_DTYPE = 'default'
VAE_NODE = '3'
VAE_FILE = 'flux2-vae.safetensors'
VAE_FOLDER = 'models/vae'
VAE_SOURCE_BYTES = 320.6e6
CLIP_DEVICE = 'default'

# PMI image bound. The upstream UI snapshot uses 1024x1024; 512x512 is a PMI
# bound applied identically to the latent and the scheduler, not an upstream setting.
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
BATCH_SIZE = 1
STEPS = 4
CFG = 1.0
SAMPLER_NAME = 'euler'
SEED_CONTROL = 'fixed'
PROMPT_NODE = '4'
SEED_NODE = '7'
SEED_INPUT = 'noise_seed'
FILENAME_PREFIX = 'PMI/answer'
SAVE_NODE = '13'

# Node IDs of the flattened API graph, in execution order.
GRAPH_ORDER = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13')
GUIDER_NODE = '6'
SAMPLER_SELECT_NODE = '8'
SCHEDULER_NODE = '9'
LATENT_NODE = '10'
SAMPLE_NODE = '11'
DECODE_NODE = '12'

# Pinned FLUX.2 Klein 4B distilled graph. The UI workflow binds both width and
# height through PrimitiveInt helpers; the API graph binds the same equal values
# directly on the latent and the scheduler. Only the positive CLIP text and the
# deterministic RandomNoise seed are injected; everything else is fixed here.
RECIPE_NODES = {
    '1': {'class_type': 'UNETLoader',
          'inputs': {'unet_name': DIFFUSION_MODEL, 'weight_dtype': WEIGHT_DTYPE}},
    '2': {'class_type': 'CLIPLoader',
          'inputs': {'clip_name': TEXT_ENCODER_FILE, 'type': TEXT_ENCODER_TYPE,
                     'device': CLIP_DEVICE}},
    '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': VAE_FILE}},
    '4': {'class_type': 'CLIPTextEncode', 'inputs': {'text': '', 'clip': ['2', 0]}},
    '5': {'class_type': 'ConditioningZeroOut', 'inputs': {'conditioning': ['4', 0]}},
    '6': {'class_type': 'CFGGuider',
          'inputs': {'model': ['1', 0], 'positive': ['4', 0], 'negative': ['5', 0], 'cfg': CFG}},
    '7': {'class_type': 'RandomNoise',
          'inputs': {SEED_INPUT: 0, 'control_after_generate': SEED_CONTROL}},
    '8': {'class_type': 'KSamplerSelect', 'inputs': {'sampler_name': SAMPLER_NAME}},
    '9': {'class_type': 'Flux2Scheduler',
          'inputs': {'width': IMAGE_WIDTH, 'height': IMAGE_HEIGHT, 'steps': STEPS}},
    '10': {'class_type': 'EmptyFlux2LatentImage',
           'inputs': {'width': IMAGE_WIDTH, 'height': IMAGE_HEIGHT, 'batch_size': BATCH_SIZE}},
    '11': {'class_type': 'SamplerCustomAdvanced',
           'inputs': {'noise': ['7', 0], 'guider': ['6', 0], 'sampler': ['8', 0],
                      'sigmas': ['9', 0], 'latent_image': ['10', 0]}},
    '12': {'class_type': 'VAEDecode', 'inputs': {'samples': ['11', 0], 'vae': ['3', 0]}},
    '13': {'class_type': 'SaveImage',
           'inputs': {'images': ['12', 0], 'filename_prefix': FILENAME_PREFIX}},
}

# The only links the pinned graph may contain: (from node, to node, input name).
PINNED_EDGES = (
    ('1', '6', 'model'),
    ('2', '4', 'clip'),
    ('4', '5', 'conditioning'),
    ('4', '6', 'positive'),
    ('5', '6', 'negative'),
    ('6', '11', 'guider'),
    ('7', '11', 'noise'),
    ('8', '11', 'sampler'),
    ('9', '11', 'sigmas'),
    ('10', '11', 'latent_image'),
    ('11', '12', 'samples'),
    ('3', '12', 'vae'),
    ('12', '13', 'images'),
)

# Graph shapes that cannot express the selected distilled route are refused by
# name as well as by node set, so a swapped backend or model is never submitted.
INCOMPATIBLE_CLASSES = (
    'CheckpointLoaderSimple', 'CheckpointLoader', 'CheckpointLoaderGGUF', 'KSampler',
    'KSamplerAdvanced', 'KSamplerSelectAdvanced', 'EmptyLatentImage', 'EmptySD3LatentImage',
    'LatentUpscale', 'FluxGuidance', 'LoraLoader', 'ModelSamplingFlux', 'CLIPSetLastLayer',
    'DualCLIPLoader', 'TripleCLIPLoader', 'CLIPLoaderGGUF', 'VAELoaderGGUF',
)


class ComfyFailure(RuntimeError):
    """Endpoint refused the graph, the queue, the artifact or the unload proof."""


class ComfyCancelled(RuntimeError):
    """Owner signal or operator cancellation stopped the stage; no retry."""


def check_endpoint(endpoint):
    """Accept only an explicit loopback HTTP origin; never a credentialed or remote URL."""
    if not isinstance(endpoint, str) or not endpoint or len(endpoint) > 256 or '\0' in endpoint:
        raise ValueError('comfy endpoint must be a short loopback http origin')
    parts = urlsplit(endpoint)
    if (parts.scheme != 'http' or parts.username or parts.password or parts.query
            or parts.fragment or parts.path not in ('', '/')):
        raise ValueError('comfy endpoint must be plain http loopback without path or credentials')
    if parts.hostname not in LOOPBACK_HOSTS or parts.port is None or not 1 <= parts.port <= 65535:
        raise ValueError('comfy endpoint must name a loopback host and explicit port')
    host = f'[{parts.hostname}]' if ':' in parts.hostname else parts.hostname
    return f'http://{host}:{parts.port}'


def resolve_endpoint(cli_value=None, environ=None):
    """Configuration only. This never grants execution; the platform hold still decides."""
    environment = os.environ if environ is None else environ
    return check_endpoint(cli_value or environment.get(ENDPOINT_ENVIRONMENT) or DEFAULT_ENDPOINT)


def check_prompt(text):
    if not isinstance(text, str) or not text.strip() or '\0' in text or len(text.encode('utf-8')) > MAX_PROMPT:
        raise ValueError('image prompt must be nonempty UTF-8, no NUL, at most 4096 bytes')


def prompt_seed(text):
    """Deterministic per-prompt seed so an identical request reproduces identically."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % (2 ** 32)


def recipe_dimensions():
    """The PMI image bound, read back out of the pinned latent and scheduler nodes."""
    latent = RECIPE_NODES[LATENT_NODE]['inputs']
    scheduler = RECIPE_NODES[SCHEDULER_NODE]['inputs']
    if (latent['width'], latent['height']) != (scheduler['width'], scheduler['height']):
        raise ValueError('pinned latent and scheduler dimensions differ')
    if not (1 <= latent['width'] <= 4096 and 1 <= latent['height'] <= 4096):
        raise ValueError('pinned image dimensions are outside the reviewed envelope')
    return latent['width'], latent['height']


IMAGE_BOUNDS = recipe_dimensions()


def read_source_workflow():
    """Load the pinned official snapshot as plain JSON source data; never import or run it."""
    path = Path(__file__).resolve().parent / SOURCE_WORKFLOW_FILE
    data = bounded_read(path, MAX_WORKFLOW_BYTES)
    if len(data) != SOURCE_WORKFLOW_BYTES or hashlib.sha256(data).hexdigest() != SOURCE_WORKFLOW_SHA256:
        raise ValueError('pinned official workflow snapshot bytes or SHA-256 differ from the source pin')
    return json.loads(data)


def model_roles():
    """Report installed-file presence and receipt identities without reading weights."""
    definitions = {
        'text_encoder': {'node': TEXT_ENCODER_NODE, 'class_type': 'CLIPLoader',
                         'input': 'clip_name', 'type': TEXT_ENCODER_TYPE, 'device': CLIP_DEVICE,
                         'file': TEXT_ENCODER_FILE, 'folder': TEXT_ENCODER_FOLDER,
                         'source_size_bytes': TEXT_ENCODER_SOURCE_BYTES},
        'diffusion_model': {'node': DIFFUSION_NODE, 'class_type': 'UNETLoader',
                            'input': 'unet_name', 'weight_dtype': WEIGHT_DTYPE,
                            'file': DIFFUSION_MODEL, 'folder': DIFFUSION_FOLDER,
                            'source_size_bytes': DIFFUSION_SOURCE_BYTES,
                            'unselected_base_file': UNSELECTED_DIFFUSION_MODEL,
                            'documented_alternate_file': DOCUMENTED_ALTERNATE_DIFFUSION_MODEL},
        'vae': {'node': VAE_NODE, 'class_type': 'VAELoader', 'input': 'vae_name',
                'file': VAE_FILE, 'folder': VAE_FOLDER, 'source_size_bytes': VAE_SOURCE_BYTES},
    }
    result = {}
    for role_id, definition in definitions.items():
        identity = IMAGE_ASSET_IDENTITIES[role_id]
        path = COMFYUI_ROOT / definition['folder'] / definition['file']
        try:
            is_file = path.is_file()
            actual_bytes = path.stat().st_size if is_file else None
        except OSError:
            is_file, actual_bytes = False, None
        role = dict(definition)
        role.update({
            'installation': 'present-unverified' if is_file else 'absent',
            'present': is_file,
            'installed_path': str(path),
            'bytes': actual_bytes,
            'receipt_bytes': identity['bytes'],
            'size_matches_receipt': is_file and actual_bytes == identity['bytes'],
            'repository': IMAGE_ASSET_REPOSITORY,
            'revision': identity['revision'],
            'sha256': identity['sha256'],
            'sha256_verified_here': False,
            'setup_receipt': SETUP_RECEIPT,
            'note': 'presence and byte count use filesystem metadata only; SHA-256 is recorded '
                    'from the setup receipt and was not recomputed by this plan',
        })
        result[role_id] = role
    result['diffusion_model']['note'] += (
        '; the distilled 4B file is selected, and the base/FP8 filenames are not substitutes')
    return result


def recipe():
    """The reviewed image recipe: selected graph, model roles and open host questions."""
    roles = model_roles()
    return {
        'schema': 'PMI_COMFY_IMAGE_RECIPE_V2', 'status': 'PlatformHeld',
        'selected_harness': SELECTED_HARNESS, 'selected_model': SELECTED_MODEL,
        'workflow_compatible': True,
        'host_runtime_verified': False,
        'qualification': 'Unresolved',
        'source': {
            'file': SOURCE_WORKFLOW_FILE, 'bytes': SOURCE_WORKFLOW_BYTES,
            'sha256': SOURCE_WORKFLOW_SHA256, 'repository': SOURCE_UPSTREAM_REPOSITORY,
            'path': SOURCE_UPSTREAM_PATH, 'revision': SOURCE_UPSTREAM_REVISION,
            'revision_note': SOURCE_UPSTREAM_COMMIT, 'catalog': SOURCE_CATALOG,
            'workflow_id': SOURCE_WORKFLOW_ID, 'workflow_revision': SOURCE_WORKFLOW_REVISION,
            'workflow_version': SOURCE_WORKFLOW_VERSION,
            'selected_subgraph': {'id': SELECTED_SUBGRAPH_ID, 'name': SELECTED_SUBGRAPH_NAME},
            'excluded_subgraph': {'id': EXCLUDED_SUBGRAPH_ID, 'name': EXCLUDED_SUBGRAPH_NAME,
                                  'reason': 'non-distilled base branch of the same root UI graph'},
            'ui_only_nodes': ['PrimitiveInt', 'PrimitiveStringMultiline', 'MarkdownNote'],
            'note': 'JSON source data only; never imported into ComfyUI or executed here',
        },
        'endpoint': {'default': DEFAULT_ENDPOINT, 'environment': ENDPOINT_ENVIRONMENT,
                     'loopback_only': True, 'scheme': 'http'},
        'model_roles': roles,
        'model_assets_present': all(role['present'] and role['size_matches_receipt']
                                    for role in roles.values()),
        'graph': {'nodes': {key: dict(value) for key, value in RECIPE_NODES.items()},
                  'order': list(GRAPH_ORDER), 'api_format': True,
                  'edges': [{'from': source, 'to': target, 'input': name}
                            for source, target, name in PINNED_EDGES],
                  'output_nodes': [SAVE_NODE]},
        'sampler': {'steps': STEPS, 'cfg': CFG, 'sampler_name': SAMPLER_NAME,
                    'scheduler': 'Flux2Scheduler', 'batch_size': BATCH_SIZE,
                    'seed_node': SEED_NODE, 'seed_input': SEED_INPUT,
                    'seed_control': SEED_CONTROL,
                    'seed': 'sha256(prompt)[0:8] mod 2**32'},
        'latent': {'node': LATENT_NODE, 'class_type': 'EmptyFlux2LatentImage',
                   'width': IMAGE_WIDTH, 'height': IMAGE_HEIGHT, 'batch_size': BATCH_SIZE,
                   'dimension_source': 'PMI project bound applied to both the latent and the '
                                       'Flux2Scheduler; the upstream UI snapshot uses 1024x1024'},
        'conditioning': {'positive_node': PROMPT_NODE, 'negative': 'ConditioningZeroOut',
                         'negative_prompt_text': None,
                         'note': 'the distilled branch zeroes its negative conditioning; there is no '
                                 'second CLIPTextEncode and no negative prompt text'},
        'artifact': {'node': SAVE_NODE, 'kind': 'output', 'format': 'png',
                     'max_bytes': MAX_ARTIFACT_BYTES, 'bounds': [IMAGE_BOUNDS[0], IMAGE_BOUNDS[1]],
                     'prefix': FILENAME_PREFIX},
        'release_rule': 'POST /free acknowledges an asynchronous request only; measured /system_stats '
                        'VRAM recovery to the pre-inference baseline is required before a pass',
        'cancellation': 'owner signal or deadline targets only this prompt_id for interrupt and queue deletion',
        'bounds': {'stage_seconds': IMAGE_STAGE_SECONDS, 'poll_interval_seconds': POLL_INTERVAL_SECONDS,
                   'release_timeout_seconds': RELEASE_TIMEOUT_SECONDS,
                   'release_vram_tolerance_bytes': RELEASE_VRAM_TOLERANCE_BYTES,
                   'request_timeout_seconds': REQUEST_TIMEOUT_SECONDS,
                   'workflow_bytes': MAX_WORKFLOW_BYTES, 'artifact_bytes': MAX_ARTIFACT_BYTES},
        'actual_inference': False, 'endpoint_reachable': False,
        'remaining': ['normal platform review for each physical execution',
                      'reconfirm the installed model-file hashes against the retained setup receipt '
                      'before any image workload',
                      'separate qualification of the host PyTorch/ROCm runtime and VRAM fit',
                      'exact prompt/seed image evidence before any claimed image success'],
    }


def image_plan(endpoint=DEFAULT_ENDPOINT, prompt=None):
    """Offline plan; checks local asset paths and sizes only, with no endpoint/device probe."""
    plan = recipe()
    plan['endpoint']['configured'] = check_endpoint(endpoint)
    plan['stages'] = ['text:bonsai-stock', 'speech:piper-cpu', 'image:comfyui-local']
    plan['release_rule'] = ('text unload/postflight, then CPU speech, then one ComfyUI submission; '
                            'measured post-/free VRAM recovery to baseline is required before completion')
    if prompt is not None:
        check_prompt(prompt)
        plan['prompt_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
        plan['seed'] = prompt_seed(prompt)
        plan['workflow_sha256'] = hashlib.sha256(
            json.dumps(build_workflow(prompt), sort_keys=True).encode()).hexdigest()
    plan['actual_inference'] = False
    plan['image_execution_held'] = True
    plan['unavailable_modalities'] = ['video', 'audio']
    return plan


def build_workflow(prompt, seed=None):
    """Build the pinned API-format graph. Prompts can never add, drop or rewire nodes."""
    check_prompt(prompt)
    workflow = json.loads(json.dumps(RECIPE_NODES))
    workflow[PROMPT_NODE]['inputs']['text'] = prompt
    workflow[SEED_NODE]['inputs'][SEED_INPUT] = prompt_seed(prompt) if seed is None else seed
    return validate_workflow(workflow, prompt)


def validate_workflow(workflow, prompt=None):
    """Only the pinned node set, classes, links, settings, prompt and seed may reach the endpoint."""
    if not isinstance(workflow, dict) or set(workflow) != set(RECIPE_NODES):
        raise ValueError('image workflow node set differs from the pinned FLUX.2 Klein 4B distilled recipe')
    for node, expected in RECIPE_NODES.items():
        got = workflow[node]
        if not isinstance(got, dict) or set(got) != {'class_type', 'inputs'}:
            raise ValueError('image workflow node is not an API node record: ' + node)
        if got['class_type'] != expected['class_type']:
            if got['class_type'] in INCOMPATIBLE_CLASSES:
                raise ValueError('image workflow uses a node that cannot express the selected '
                                 'FLUX.2 Klein 4B distilled graph: ' + str(got['class_type']))
            raise ValueError('image workflow class_type changed: ' + node)
        inputs = got['inputs']
        if not isinstance(inputs, dict) or set(inputs) != set(expected['inputs']):
            raise ValueError('image workflow input set changed: ' + node)
    classes = [workflow[node]['class_type'] for node in sorted(workflow)]
    for banned in sorted(set(classes) & set(INCOMPATIBLE_CLASSES)):
        raise ValueError('image workflow uses a node that cannot express the selected graph: ' + banned)
    if classes.count('CLIPTextEncode') != 1:
        raise ValueError('pinned distilled graph encodes exactly one CLIPTextEncode; '
                         'negative conditioning is zeroed by ConditioningZeroOut')
    serialized = json.dumps(workflow, sort_keys=True)
    for unselected in (UNSELECTED_DIFFUSION_MODEL, DOCUMENTED_ALTERNATE_DIFFUSION_MODEL):
        if unselected in serialized:
            raise ValueError('image workflow names a model file that is not the selected one: ' + unselected)
    edges = []
    for node in sorted(workflow):
        for key, value in sorted(workflow[node]['inputs'].items()):
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                edges.append((value[0], node, key))
                if value[0] not in workflow:
                    raise ValueError('image workflow link leaves the pinned node set: ' + node + '/' + key)
                if not isinstance(value[1], int) or isinstance(value[1], bool) or value[1] != 0:
                    raise ValueError('image workflow link uses a foreign output slot: ' + node + '/' + key)
    if tuple(sorted(edges)) != tuple(sorted(PINNED_EDGES)):
        raise ValueError('image workflow link topology differs from the pinned distilled graph')
    for node, expected in RECIPE_NODES.items():
        for key, value in expected['inputs'].items():
            if node == PROMPT_NODE and key == 'text':
                continue
            if node == SEED_NODE and key == SEED_INPUT:
                continue
            if workflow[node]['inputs'][key] != value:
                raise ValueError('image workflow pinned input changed: ' + node + '/' + key)
    if (workflow[SCHEDULER_NODE]['inputs']['width'] != workflow[LATENT_NODE]['inputs']['width']
            or workflow[SCHEDULER_NODE]['inputs']['height'] != workflow[LATENT_NODE]['inputs']['height']):
        raise ValueError('image workflow scheduler and latent dimensions differ')
    if workflow[SEED_NODE]['inputs']['control_after_generate'] != SEED_CONTROL:
        raise ValueError('image workflow seed control is not the pinned fixed value')
    text = workflow[PROMPT_NODE]['inputs']['text']
    check_prompt(text)
    if prompt is not None and text != prompt:
        raise ValueError('image workflow text is not the requested prompt')
    seed = workflow[SEED_NODE]['inputs'][SEED_INPUT]
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2 ** 32:
        raise ValueError('image workflow seed is not a bounded 32-bit integer')
    outputs = [node for node, value in workflow.items()
               if value['class_type'] in ('SaveImage', 'PreviewImage', 'SaveAnimatedPNG', 'SaveAnimatedWEBP')]
    if outputs != [SAVE_NODE]:
        raise ValueError('image workflow must have exactly one pinned SaveImage output node')
    encoded = json.dumps(workflow).encode('utf-8')
    if len(encoded) > MAX_WORKFLOW_BYTES:
        raise ValueError('image workflow exceeds byte bound')
    return workflow


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so loopback responses cannot be replayed elsewhere."""

    def redirect_request(self, *args, **kwargs):
        return None


class LocalTransport:
    """Loopback-only JSON/binary client. No proxy, no redirect, no credentials."""

    def __init__(self, endpoint=DEFAULT_ENDPOINT):
        self.endpoint = check_endpoint(endpoint)

    def call(self, path, payload=None, method='GET', limit=MAX_JSON_BYTES):
        # Path is an internal API route built here, never operator or response text.
        if (not isinstance(path, str) or not path.startswith('/') or path.startswith('//')
                or '://' in path or any(ord(c) < 33 or ord(c) == 127 for c in path)):
            raise ValueError('comfy path must be a relative API path')
        data = None
        if payload is not None:
            data = json.dumps(payload).encode('utf-8')
            if len(data) > MAX_WORKFLOW_BYTES:
                raise ValueError('comfy request body exceeds bound')
        request = urllib.request.Request(
            self.endpoint + path, data=data, method=method,
            headers={'Content-Type': 'application/json'} if data is not None else {})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        try:
            with opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                if getattr(response, 'status', 200) != 200:
                    raise ComfyFailure(f'{method} {path} HTTP {response.status}')
                body = response.read(limit + 1)
        except urllib.error.HTTPError as error:
            raise ComfyFailure(f'{method} {path} refused with HTTP {error.code}') from None
        except urllib.error.URLError as error:
            raise ComfyFailure(f'{method} {path} unreachable: {error.reason}') from None
        if len(body) > limit:
            raise ComfyFailure(f'{method} {path} response exceeds bound')
        return body

    def call_json(self, path, payload=None, method='GET', limit=MAX_JSON_BYTES):
        body = self.call(path, payload, method, limit)
        try:
            return json.loads(body)
        except ValueError:
            raise ComfyFailure(f'{method} {path} returned invalid JSON') from None


def submit(client, workflow, client_id):
    """POST /prompt; the endpoint validates the graph before it is ever queued."""
    validate_workflow(workflow, workflow[PROMPT_NODE]['inputs']['text'])
    answer = client.call_json('/prompt', {'prompt': workflow, 'client_id': client_id}, 'POST')
    if not isinstance(answer, dict):
        raise ComfyFailure('comfy prompt response is not an object')
    node_errors = answer.get('node_errors') or {}
    if node_errors:
        raise ComfyFailure('comfy refused the graph: ' + json.dumps(node_errors, sort_keys=True)[:512])
    prompt_id = answer.get('prompt_id')
    try:
        canonical_prompt_id = str(uuid.UUID(prompt_id))
    except (ValueError, TypeError, AttributeError):
        canonical_prompt_id = None
    if canonical_prompt_id != prompt_id:
        raise ComfyFailure('comfy prompt response has no canonical UUID prompt_id')
    return {'prompt_id': prompt_id, 'client_id': client_id,
            'queue_number': answer.get('number'), 'node_errors': {}}


def history(client, prompt_id):
    """GET /history/{id}; an empty object simply means the prompt is still queued."""
    try:
        canonical_prompt_id = str(uuid.UUID(prompt_id))
    except (ValueError, TypeError, AttributeError):
        canonical_prompt_id = None
    if canonical_prompt_id != prompt_id:
        raise ValueError('unexpected prompt_id')
    answer = client.call_json('/history/' + prompt_id)
    if not isinstance(answer, dict):
        raise ComfyFailure('comfy history response is not an object')
    return answer.get(prompt_id) or {}


def await_completion(client, prompt_id, deadline, check=None, clock=time.monotonic, sleeper=time.sleep):
    """Poll until success or a proven failure; cancellation and deadlines stop cleanly."""
    polls = 0
    while True:
        if check is not None:
            check()
        entry = history(client, prompt_id)
        polls += 1
        status = entry.get('status') if isinstance(entry, dict) else None
        if isinstance(status, dict) and status.get('status_str') in ('success', 'error'):
            return entry, polls
        if clock() >= deadline:
            raise TimeoutError('comfy image stage deadline reached before completion')
        if polls > IMAGE_STAGE_SECONDS / POLL_INTERVAL_SECONDS + 8:
            raise TimeoutError('comfy image poll count exceeded the stage bound')
        sleeper(POLL_INTERVAL_SECONDS)


def artifact_reference(entry):
    """Exactly one output image from the pinned SaveImage node is accepted."""
    status = entry.get('status') or {}
    if status.get('status_str') != 'success' or status.get('completed') is not True:
        messages = status.get('messages') or ['queue status: ' + str(status.get('status_str'))]
        raise ComfyFailure(SAMPLER_ERROR + ': ' + json.dumps(messages, sort_keys=True)[:512])
    outputs = entry.get('outputs') or {}
    images = (outputs.get(SAVE_NODE) or {}).get('images')
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise ComfyFailure('comfy history has no single pinned SaveImage artifact')
    image = images[0]
    name, subfolder, kind = image.get('filename'), image.get('subfolder', ''), image.get('type', 'output')
    if (not isinstance(name, str) or not name or len(name) > 255 or '/' in name or '\\' in name
            or name in ('.', '..') or not isinstance(subfolder, str) or len(subfolder) > 255
            or kind != 'output'):
        raise ComfyFailure('comfy artifact reference is not a plain output filename')
    if subfolder != 'PMI' or not name.startswith('answer_'):
        raise ComfyFailure('comfy artifact is outside the pinned PMI/answer output prefix')
    return {'filename': name, 'subfolder': subfolder, 'type': kind}


def fetch_artifact(client, reference):
    """GET /view for the one pinned output image, bounded by the artifact cap."""
    query = ('filename=' + quote(reference['filename'], safe='')
             + '&subfolder=' + quote(reference['subfolder'], safe='')
             + '&type=' + quote(reference['type'], safe=''))
    data = client.call('/view?' + query, None, 'GET', MAX_ARTIFACT_BYTES)
    if not data:
        raise ComfyFailure('comfy artifact body is empty')
    return data


def validate_artifact(data, expected=IMAGE_BOUNDS):
    """Validate real image bytes before anything is written; return a content receipt."""
    if not isinstance(data, (bytes, bytearray)) or not data or len(data) > MAX_ARTIFACT_BYTES:
        raise ValueError('image artifact missing or above byte bound')
    data = bytes(data)
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return _validate_png(data, expected)
    if data[:3] == b'\xff\xd8\xff':
        return _validate_jpeg(data, expected)
    raise ValueError('image artifact is neither PNG nor JPEG')


def _validate_png(data, expected):
    if len(data) < 45:
        raise ValueError('PNG artifact is truncated')
    offset, header, seen = 8, None, False
    while offset + 12 <= len(data):
        length = struct.unpack('>I', data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        end = offset + 12 + length
        if length > len(data) or end > len(data):
            raise ValueError('PNG chunk exceeds artifact bound')
        payload = data[offset + 8:offset + 8 + length]
        if struct.unpack('>I', data[end - 4:end])[0] != zlib.crc32(kind + payload) & 0xffffffff:
            raise ValueError('PNG chunk checksum mismatch')
        if kind == b'IHDR':
            if header is not None:
                raise ValueError('PNG has more than one header')
            header = struct.unpack('>IIBBBBB', payload[:13])
        if kind == b'IEND':
            seen = True
            if end != len(data):
                raise ValueError('PNG has trailing bytes after IEND')
            break
        offset = end
    if header is None or not seen:
        raise ValueError('PNG is incomplete: header or IEND missing')
    width, height, depth, color = header[0], header[1], header[2], header[3]
    if width * height > 4096 * 4096 or depth != 8 or color not in (0, 2, 4, 6):
        raise ValueError('PNG format outside the reviewed envelope')
    return _image_receipt('png', data, width, height, expected)


def _validate_jpeg(data, expected):
    offset, size = 2, None
    while offset + 4 <= len(data):
        if data[offset] != 0xff:
            offset += 1
            continue
        marker = data[offset + 1]
        offset += 2
        if marker in (0xd8, 0x01) or 0xd0 <= marker <= 0xd7:
            continue
        if marker == 0xd9:
            break
        if offset + 2 > len(data):
            raise ValueError('JPEG segment header truncated')
        length = int.from_bytes(data[offset:offset + 2], 'big')
        if length < 2 or offset + length > len(data):
            raise ValueError('JPEG segment exceeds artifact bound')
        if marker in (0xc0, 0xc1, 0xc2):
            payload = data[offset + 2:offset + length]
            if len(payload) < 5:
                raise ValueError('JPEG frame header truncated')
            size = (int.from_bytes(payload[1:3], 'big'), int.from_bytes(payload[3:5], 'big'))
            break
        offset += length
    if size is None:
        raise ValueError('JPEG frame header missing')
    if not data.rstrip(b'\0').endswith(b'\xff\xd9'):
        raise ValueError('JPEG has no end-of-image marker')
    return _image_receipt('jpeg', data, size[1], size[0], expected)


def _image_receipt(kind, data, width, height, expected):
    if not 1 <= width <= 4096 or not 1 <= height <= 4096:
        raise ValueError('image dimensions outside the reviewed envelope')
    if expected and (width, height) != tuple(expected):
        raise ValueError('image dimensions differ from the pinned latent size')
    return {'format': kind, 'width': width, 'height': height, 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def save_artifact(stage, data, receipt):
    """Create the image file exclusively so a receipt can never name foreign bytes."""
    path = stage / ('answer.png' if receipt['format'] == 'png' else 'answer.jpg')
    write_new(path, data)
    return str(path)


def cancel(client, prompt_id):
    """Best-effort targeted stop; never issue ComfyUI's global interrupt."""
    outcome = {}
    if not prompt_id:
        return outcome
    for name, path, payload, method in (
            ('interrupt', '/interrupt', {'prompt_id': prompt_id}, 'POST'),
            ('queue_delete', '/queue', {'delete': [prompt_id]}, 'POST')):
        try:
            client.call(path, payload, method, 65536)
            outcome[name] = 'ok'
        except Exception as error:  # best effort; the stage error is what matters
            outcome[name] = str(error)
    return outcome


def _vram_sample(client):
    """Return validated primary-device global free VRAM bytes from /system_stats."""
    stats = client.call_json('/system_stats', limit=65536)
    devices = stats.get('devices') if isinstance(stats, dict) else None
    if not isinstance(devices, list) or not devices or not isinstance(devices[0], dict):
        raise ComfyFailure('ComfyUI system_stats has no primary device memory telemetry')
    device = devices[0]
    free, total = device.get('vram_free'), device.get('vram_total')
    if (not isinstance(free, int) or isinstance(free, bool) or free < 0
            or not isinstance(total, int) or isinstance(total, bool) or total <= 0 or free > total):
        raise ComfyFailure('ComfyUI system_stats primary-device VRAM counters are invalid')
    return {'name': device.get('name'), 'type': device.get('type'),
            'vram_total_bytes': total, 'vram_free_bytes': free}


def release(client, baseline, deadline, check=None, clock=time.monotonic, sleeper=time.sleep):
    """Request async release, then prove VRAM recovers near its pre-run baseline."""
    client.call('/free', {'unload_models': True, 'free_memory': True}, 'POST', 65536)
    requested_at = clock()
    samples = []
    while True:
        if check is not None:
            check()
        sample = _vram_sample(client)
        samples.append(sample)
        recovered = sample['vram_free_bytes'] >= baseline['vram_free_bytes'] - RELEASE_VRAM_TOLERANCE_BYTES
        if recovered:
            return {'endpoint': '/free', 'request_acknowledged': True,
                    'unload_models': True, 'free_memory': True,
                    'proof': 'post-/system_stats primary-device vram_free returned to pre-inference baseline tolerance',
                    'baseline': baseline, 'postflight': sample, 'samples': samples,
                    'request_to_recovery_seconds': max(0, clock() - requested_at), 'unload_proven': True}
        if clock() >= min(deadline, requested_at + RELEASE_TIMEOUT_SECONDS):
            raise ComfyFailure('ComfyUI unload remains unproven: VRAM did not recover to baseline')
        sleeper(RELEASE_POLL_SECONDS)


def run_image(client, prompt, stage, deadline, check=None, clock=time.monotonic,
              sleeper=time.sleep, endpoint=None):
    """Submit, poll, validate, save and unload one image. Any stop is recorded."""
    stage = Path(stage)
    check_prompt(prompt)
    endpoint = check_endpoint(endpoint or getattr(client, 'endpoint', DEFAULT_ENDPOINT))
    workflow = build_workflow(prompt)
    client_id = hashlib.sha256((endpoint + '\0' + prompt).encode()).hexdigest()[:32]
    write_new(stage / 'workflow.json', workflow)
    receipt = {'schema': RECEIPT_SCHEMA, 'endpoint': endpoint, 'client_id': client_id,
               'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
               'seed': workflow[SEED_NODE]['inputs'][SEED_INPUT],
               'seed_control': workflow[SEED_NODE]['inputs']['control_after_generate'],
               'workflow_sha256': hashlib.sha256(json.dumps(workflow, sort_keys=True).encode()).hexdigest(),
               'recipe_sha256': hashlib.sha256(json.dumps(RECIPE_NODES, sort_keys=True).encode()).hexdigest(),
               'source_workflow_sha256': SOURCE_WORKFLOW_SHA256,
               'selected_model': SELECTED_MODEL, 'selected_harness': SELECTED_HARNESS,
               'image_bounds': [IMAGE_BOUNDS[0], IMAGE_BOUNDS[1]],
               'automatic_retry': False}
    prompt_id = None
    try:
        baseline = _vram_sample(client)
        submission = submit(client, workflow, client_id)
        prompt_id = submission['prompt_id']
        write_new(stage / 'submission.json', submission)
        entry, polls = await_completion(client, prompt_id, deadline, check, clock, sleeper)
        write_new(stage / 'history.json', entry if isinstance(entry, dict) else {})
        reference = artifact_reference(entry)
        data = fetch_artifact(client, reference)
        artifact = validate_artifact(data)
        path = save_artifact(stage, data, artifact)
    except BaseException as error:
        _record_stop(stage, error, prompt_id, cancel(client, prompt_id))
        raise
    receipt.update({'prompt_id': submission['prompt_id'], 'queue_status': 'success', 'polls': polls,
                    'reference': reference, 'artifact': artifact, 'artifact_file': path})
    try:
        receipt['release'] = release(client, baseline, deadline, check=check, clock=clock, sleeper=sleeper)
    except ComfyFailure as error:
        # Artifact exists but the server still holds the checkpoint: stop without retry.
        receipt['release'] = {'unload_proven': False, 'error': str(error)}
        receipt['release_proven'] = False
        receipt['status'] = 'StoppedNoRetry'
        receipt['cancel'] = cancel(client, prompt_id)
        write_new(stage / 'image-receipt.json', receipt)
        _record_stop(stage, RuntimeError('ComfyUI model unload unproven; stop without retry'),
                     prompt_id, receipt)
        raise RuntimeError('ComfyUI model unload unproven; stop without retry') from None
    receipt['release_proven'] = receipt['release']['unload_proven'] is True
    receipt['status'] = 'ImageAndReleasePass' if receipt['release_proven'] else 'StoppedNoRetry'
    write_new(stage / 'image-receipt.json', receipt)
    if not receipt['release_proven']:
        raise ComfyFailure('ComfyUI model unload unproven; stop without retry')
    return receipt


def _record_stop(stage, error, prompt_id, stopped):
    """Best effort only: a stop receipt must never mask the failure it explains."""
    try:
        write_new(stage / 'image-cancel.json',
                  {'reason': str(error), 'error_type': type(error).__name__, 'prompt_id': prompt_id,
                   'cancel': stopped, 'automatic_retry': False})
    except (OSError, ValueError):
        pass


class ComfyCancellation:
    """Turn owner SIGINT/SIGTERM into a cooperative stop the stage can prove."""

    def __init__(self, deadline):
        self.deadline = deadline
        self.event = threading.Event()
        self.previous = {}

    def install(self):
        if threading.current_thread() is not threading.main_thread():
            return self
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self.previous[sig] = signal.signal(sig, self.event.set)
            except (ValueError, OSError):
                pass
        return self

    def restore(self):
        for sig, handler in self.previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
        self.previous.clear()

    def check(self):
        if self.event.is_set():
            raise ComfyCancelled('owner signal cancelled the image stage; stop without retry')
        if time.monotonic() >= self.deadline:
            raise TimeoutError('comfy image stage deadline reached')


class ComfyBackend:
    """Optional image backend. Every physical entry stays platform-held."""
    actual_inference = True

    def __init__(self, endpoint=None, environ=None):
        self.endpoint = resolve_endpoint(endpoint, environ)

    def run_stage(self, prompt, stage, deadline):
        require_real_execution()  # before any endpoint probe, import or output
        raise ComfyFailure(
            'ComfyUI image submission disabled: the FLUX.2 Klein 4B distilled graph is pinned and '
            'source-confirmed, but this host execution is platform-held and the host PyTorch/ROCm '
            'runtime and VRAM fit remain unverified'
        )
        check_prompt(prompt)
        stage = Path(stage)
        request = json.loads(bounded_read(stage / 'request.json', 65536))
        if (request.get('prompt') != prompt or not isinstance(request.get('text_source'), str)
                or request.get('prompt_sha256') != hashlib.sha256(prompt.encode()).hexdigest()
                or request.get('endpoint') != self.endpoint):
            raise ValueError('image request differs from the completed text and configured endpoint')
        client = LocalTransport(self.endpoint)
        cancellation = ComfyCancellation(deadline).install()
        try:
            receipt = run_image(client, prompt, stage, deadline, check=cancellation.check,
                                endpoint=self.endpoint)
        finally:
            cancellation.restore()
        return {'status': 'RealImageAndReleasePass', 'prompt_id': receipt['prompt_id'],
                'artifact': receipt['artifact'], 'artifact_file': receipt['artifact_file'],
                'release_proven': receipt['release_proven']}
