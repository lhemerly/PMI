"""Typed reconciliation of PMI's intended model lineup with this checkout.

Two questions are deliberately kept apart for every model role, because merging
them is how a stock artifact quietly inherits somebody else's qualification:

* ``installation`` -- is the artifact present here, and has anybody established
  its identity.
* ``runtime_qualification`` -- has the runtime that would load it been qualified
  for an exact workload on this host.

This module is a declaration plus the invariants the rest of the project must
keep satisfying. It fetches nothing, opens no weight, installs no package,
starts no service and runs no model. Unpinned identities are recorded as
``None``, which always means *unknown*; it never means *verified*.

The intended text model is the Heretic derivative named below. It is absent
here and no hash of it is known, so it carries neither installation nor runtime
evidence. The installed stock Bonsai weights keep their preserved receipts and
their own scoped qualification; that evidence is historical baseline and is not
transferred to Heretic.
"""
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping, Optional, Tuple

SCHEMA = 'PMI_MODEL_INTENT_V1'

# The executable text default is unchanged by this declaration. It stays the
# preserved stock baseline; switching it to the intended Heretic artifact, to
# Qwen3-8B or to qwen25-modified is a separate reviewed change, not a side
# effect of recording intent.
DEFAULT_TEXT_MODEL_ID = 'bonsai-stock'


class Modality(str, Enum):
    TEXT = 'text'
    SPEECH = 'speech'
    IMAGE = 'image'
    VIDEO = 'video'


class Selection(str, Enum):
    INTENDED = 'intended'
    PRESERVED_BASELINE = 'preserved-baseline'
    NOT_SELECTED = 'not-selected'
    UNRESOLVED = 'unresolved'


class Installation(str, Enum):
    """Answering the installation question only; this never implies runtime support."""
    ABSENT = 'absent'
    PRESENT_UNVERIFIED = 'present-unverified'


class RuntimeQualification(str, Enum):
    """Answering the runtime question only."""
    UNQUALIFIED = 'unqualified'
    SCOPED_RUN_PROVEN = 'scoped-run-proven'


@dataclass(frozen=True)
class ModelRole:
    """One model in one modality, with one honest answer per question."""

    modality: Modality
    role_id: str
    selection: Selection
    installation: Installation
    runtime_qualification: RuntimeQualification
    primary: bool = False
    harness: Optional[str] = None
    runtime: Optional[str] = None
    loader: Optional[str] = None
    folder: Optional[str] = None
    file_name: Optional[str] = None
    repository: Optional[str] = None
    revision: Optional[str] = None
    revision_kind: Optional[str] = None
    revision_verified_here: bool = False
    quantization: Optional[str] = None
    sha256: Optional[str] = None
    sha256_verified_here: bool = False
    bytes: Optional[int] = None
    qualification_scope: Optional[str] = None
    substitutes_intended: bool = False
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ModalityIntent:
    """Route-level facts that are not properties of any single model file."""

    modality: Modality
    selected: bool
    resolved: bool
    harness: Optional[str]
    stages: Tuple[str, ...] = ()
    source_compatibility_proven: Optional[bool] = None
    runtime_qualified: bool = False
    notes: Tuple[str, ...] = ()


# --- Declared identities -----------------------------------------------------

HERETIC_REPOSITORY = 'OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-v2-GGUF'
HERETIC_REVISION = '6db27547a8f25667cdb43a28d15cc5d33937b7dc'
HERETIC_QUANTIZATION = 'PTQ1_0'
HERETIC_ROLE_ID = 'heretic-ptq1_0'

STOCK_BONSAI_ROLE_ID = 'bonsai-stock'
STOCK_BONSAI_REPOSITORY = 'prism-ml/Ternary-Bonsai-2-27B-gguf'
STOCK_BONSAI_REVISION = 'b072e1d3b35a0a630cece372c2127528e0994386'
STOCK_BONSAI_SHA256 = '53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3'
STOCK_BONSAI_BYTES = 5946648928

KOKORO_ROLE_ID = 'kokoro-82m'
KOKORO_REPOSITORY = 'hexgrad/Kokoro-82M'
KOKORO_REVISION = 'f3ff3571791e39611d31c381e3a41a3af07b4987'
KOKORO_SHA256 = '496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4'
KOKORO_WEIGHT_FILE = 'kokoro-v1_0.pth'
KOKORO_CONFIG_FILE = 'config.json'
KOKORO_RUNTIME_VERSION = '0.9.4'
KOKORO_PYTHON_SPEC = '>=3.10,<3.13'
KOKORO_LICENSE = 'Apache-2.0'
KOKORO_SAMPLE_RATE = 24000
KOKORO_DEFAULT_VOICE = 'af_heart'
KOKORO_VOICES = {'af_heart': ('en-US', 'English'),
                 'pf_dora': ('pt-BR', 'Brazilian Portuguese')}

PIPER_ROLE_ID = PIPER_ADAPTER_ID = 'piper-cpu'
PIPER_SAMPLE_RATE = 22050
PIPER_VOICE_ID = 'en_US-ljspeech-medium'
PIPER_VOICE_FILE = 'en_US-ljspeech-medium.onnx'
PIPER_VOICE_BYTES = 63531379
PIPER_VOICE_SHA256 = '6f52a751e2349abe7a76735eb09dc1875298c77ea2342ffd2fef79ff81b87f22'
PIPER_PACKAGE = 'piper-tts'
PIPER_PACKAGE_VERSION = '1.8.0'
DEFAULT_SPEECH_MODEL_ID = PIPER_ROLE_ID
SPEECH_MODEL_IDS = (PIPER_ROLE_ID, KOKORO_ROLE_ID)

IMAGE_HARNESS = 'ComfyUI'
IMAGE_MODEL = 'FLUX.2 Klein 4B distilled'
IMAGE_SOURCE_REPOSITORY = 'Comfy-Org/workflow_templates'
IMAGE_SOURCE_REVISION = 'e7cd011d4ded3411c2f481200544f0be6fdc962e'
IMAGE_SOURCE_SHA256 = '8bb879856e54765b6c3ed219181e855981d8d10cfd19acff836603d9b56e838f'
IMAGE_SOURCE_BYTES = 69872
# Role id, ComfyUI loader node class, pinned file name, ComfyUI folder.
IMAGE_ROLES = (
    ('text_encoder', 'CLIPLoader', 'qwen_3_4b.safetensors', 'models/text_encoders'),
    ('diffusion_model', 'UNETLoader', 'flux-2-klein-4b.safetensors', 'models/diffusion_models'),
    ('vae', 'VAELoader', 'flux2-vae.safetensors', 'models/vae'),
)
IMAGE_ASSET_REPOSITORY = 'Comfy-Org/vae-text-encorder-for-flux-klein-4b'
IMAGE_ASSET_IDENTITIES = {
    'text_encoder': {'revision': '8556e4d870cda7c53c7942b190bfeea5be9bd411',
                     'bytes': 8044982048,
                     'sha256': '6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a'},
    'diffusion_model': {'revision': 'adabe6c3c8bdcc9cb1f27b25729b70924961c223',
                        'bytes': 7751105712,
                        'sha256': 'ec3d4e733a771f61c052fb4856c48b336c55eaf2c65487c2a1faeb9bbda7a343'},
    'vae': {'revision': 'a9e4ca87c16db4c4e1a16406a9ddb300ab0ae246',
            'bytes': 336211292,
            'sha256': '868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3'},
}

UNAVAILABLE_MODALITIES = ('video',)

RULES = (
    'no role substitutes another: substitutes_intended is False for every declared role',
    'installation and runtime_qualification are independent answers; neither implies the other',
    'a role whose installation is absent keeps sha256 None or sha256_verified_here False; '
    'None means unknown, never verified',
    'stock Bonsai receipts and tests are a preserved historical baseline and are not '
    'Heretic qualification',
    'qwen3-stock and qwen25-modified are not selected and are never silent substitutes',
    'piper-cpu stays an approved auxiliary CPU baseline, unchanged by any newer adapter',
    'video has no selected model, route, plan or artifact',
    'no modality hardcodes a verified runtime; runtime_qualification is read from declared '
    'evidence only',
)

IMAGE_NOTES = (
    'ComfyUI plus FLUX.2 Klein 4B distilled stays the selected route and the pinned '
    'workflow snapshot is unchanged',
    'graph compatibility is a property of the pinned source snapshot, not of this host',
    'the three model roles are installed under .runtime/comfyui/app per the retained setup '
    'receipt; the source-review plan does not rehash them',
    'ComfyUI v0.39.0 and PyTorch 2.14.1+rocm7.2 are installed per that receipt; no device '
    'workload, host support or VRAM fit is qualified',
    'host PyTorch/ROCm support and VRAM fit remain unverified on the RX 6600 XT and the '
    'execution hold is unchanged',
)


TEXT_ROLES = (
    ModelRole(
        modality=Modality.TEXT, role_id=HERETIC_ROLE_ID, selection=Selection.INTENDED,
        primary=True, installation=Installation.ABSENT,
        runtime_qualification=RuntimeQualification.UNQUALIFIED,
        runtime='llama-completion (Prism PTQ1_0 build)',
        repository=HERETIC_REPOSITORY, revision=HERETIC_REVISION,
        revision_kind='observed-upstream', quantization=HERETIC_QUANTIZATION,
        notes=('intended catalog text target; not present in this checkout',
               'the upstream revision is an observation of the source tree, not a '
               'downloaded and hashed artifact',
               'no weight hash is known; sha256 None means unknown, not unverified',
               'the stock Bonsai GGUF is a different artifact; its receipts, template fit '
               'and scoped run do not qualify this derivative',
               'requires the Prism specialized runtime plus its own admission, placement '
               'and unload evidence before any dispatch')),
    ModelRole(
        modality=Modality.TEXT, role_id=STOCK_BONSAI_ROLE_ID,
        selection=Selection.PRESERVED_BASELINE,
        installation=Installation.PRESENT_UNVERIFIED,
        runtime_qualification=RuntimeQualification.SCOPED_RUN_PROVEN,
        runtime='llama-completion (Prism Vulkan build)',
        repository=STOCK_BONSAI_REPOSITORY, revision=STOCK_BONSAI_REVISION,
        revision_kind='acquisition-receipt', sha256=STOCK_BONSAI_SHA256, bytes=STOCK_BONSAI_BYTES,
        qualification_scope='one real exact welcome-prompt 512-token-context run with proven '
                            'placement, bounded natural completion and verified unload and '
                            'postflight; other prompts and contexts remain unqualified',
        notes=('installed stock artifact; its identity is recorded, not re-verified here',
               'preserved historical baseline for text; the adapter default is unchanged',
               'explicitly not the Heretic derivative and never a stand-in for it',
               'Apache-2.0 per retained metadata; LICENSE and NOTICE.txt preserved at '
               'qualification')),
    ModelRole(
        modality=Modality.TEXT, role_id='qwen3-stock', selection=Selection.NOT_SELECTED,
        installation=Installation.ABSENT,
        runtime_qualification=RuntimeQualification.UNQUALIFIED,
        notes=('owner withdrew Qwen3-8B as the first run; frozen stock baseline only',
               'never substituted for the intended text role',
               'identity stays in content_factory.MODELS; nothing is claimed here')),
    ModelRole(
        modality=Modality.TEXT, role_id='qwen25-modified', selection=Selection.NOT_SELECTED,
        installation=Installation.ABSENT,
        runtime_qualification=RuntimeQualification.UNQUALIFIED,
        notes=('optional modified profile; hash, license and native compatibility are '
               'unqualified',
               'never silently substituted when the intended model is unavailable',
               'identity stays in content_factory.MODELS; nothing is claimed here')),
)

SPEECH_ROLES = (
    ModelRole(
        modality=Modality.SPEECH, role_id=KOKORO_ROLE_ID, selection=Selection.INTENDED,
        primary=True, installation=Installation.ABSENT,
        runtime_qualification=RuntimeQualification.UNQUALIFIED,
        runtime='kokoro==' + KOKORO_RUNTIME_VERSION + ' (Python, CPU only)',
        repository=KOKORO_REPOSITORY, revision=KOKORO_REVISION,
        revision_kind='pinned-by-operator', file_name=KOKORO_WEIGHT_FILE,
        sha256=KOKORO_SHA256,
        notes=('intended speech model; weights not fetched, opened or hashed here',
               'the declared SHA-256 is the operator-supplied model identity, unverified here',
               'the pinned official model card lists Apache-2.0; that does not verify local assets',
               'voice is configuration, not a constant: af_heart for English (en-US) and '
               'pf_dora for Brazilian Portuguese (pt-BR)',
               '24 kHz output is a different audio contract from the 22,050 Hz Piper baseline',
               'the source adapter targets the official kokoro==' + KOKORO_RUNTIME_VERSION
               + ' API, but the runtime is absent and implementation is unrun here')),
    ModelRole(
        modality=Modality.SPEECH, role_id=PIPER_ROLE_ID,
        selection=Selection.PRESERVED_BASELINE,
        installation=Installation.PRESENT_UNVERIFIED,
        runtime_qualification=RuntimeQualification.SCOPED_RUN_PROVEN,
        runtime=PIPER_PACKAGE + ' ' + PIPER_PACKAGE_VERSION + ' (CPU only)',
        quantization='espeak phonemes', file_name=PIPER_VOICE_FILE,
        sha256=PIPER_VOICE_SHA256, bytes=PIPER_VOICE_BYTES,
        revision_kind='voice-acquisition-receipt',
        qualification_scope='sealed CPU-only resume from the preserved Bonsai answer, with WAV '
                            'validation and measured unload and postflight; every physical '
                            'execution still needs its own platform review',
        notes=('approved auxiliary CPU adapter; preserved unchanged by any newer adapter',
               'mono 16-bit PCM at 22,050 Hz, at most 30 seconds',
               'voice ' + PIPER_VOICE_ID + ' pinned by hash in piper_backend.py; the '
               'identity recorded here is the preserved one, not a fresh verification',
               'GPL-3.0-or-later package; voice model card MIT, dataset public domain')),
)

IMAGE_MODEL_ROLES = tuple(
    ModelRole(
        modality=Modality.IMAGE, role_id=role_id, selection=Selection.INTENDED,
        primary=(role_id == 'diffusion_model'), installation=Installation.PRESENT_UNVERIFIED,
        runtime_qualification=RuntimeQualification.UNQUALIFIED,
        harness=IMAGE_HARNESS,
        runtime='ComfyUI v0.39.0 with PyTorch 2.14.1+rocm7.2 installed; host workload unqualified',
        loader=loader, folder=folder, file_name=file_name,
        repository=IMAGE_ASSET_REPOSITORY,
        revision=IMAGE_ASSET_IDENTITIES[role_id]['revision'],
        revision_kind='setup-receipt',
        sha256=IMAGE_ASSET_IDENTITIES[role_id]['sha256'],
        bytes=IMAGE_ASSET_IDENTITIES[role_id]['bytes'],
        notes=('installed ComfyUI asset for the ' + role_id.replace('_', ' ')
               + ' role, loaded by ' + loader + ' from ' + folder,
               'the retained setup receipt records this revision, byte count and SHA-256; '
               'this source review did not rehash the file',
               'installation, host runtime and runtime qualification are separate; '
               'receipt-backed files do not establish host inference support')
        + IMAGE_NOTES)
    for role_id, loader, file_name, folder in IMAGE_ROLES)

VIDEO_MODEL_ROLES = ()

MODALITIES = {
    Modality.TEXT: ModalityIntent(
        Modality.TEXT, selected=True, resolved=False, harness=None,
        stages=('text:bonsai-stock',),
        notes=('the intended Heretic artifact is unavailable, so the executable default '
               'remains the preserved stock baseline',
               'no Heretic dispatch, no Heretic weight hash and no stock-Heretic call exist here',
               'the Qwen baselines stay out of selection and out of substitution')),
    Modality.SPEECH: ModalityIntent(
        Modality.SPEECH, selected=True, resolved=False, harness=None,
        stages=('speech:piper-cpu',),
        notes=('Piper remains the approved auxiliary CPU adapter and keeps serving the '
               'default route for backwards compatibility and historical receipts',
               'Kokoro is explicitly selectable with --speech-model kokoro-82m, but its '
               'assets/runtime are not ready and it is not qualified')),
    Modality.IMAGE: ModalityIntent(
        Modality.IMAGE, selected=True, resolved=False, harness=IMAGE_HARNESS,
        stages=('text:bonsai-stock', 'speech:piper-cpu', 'image:comfyui-local'),
        source_compatibility_proven=True, notes=IMAGE_NOTES),
    Modality.VIDEO: ModalityIntent(
        Modality.VIDEO, selected=False, resolved=False, harness=None,
        notes=('no video model, harness, route or artifact is selected or declared',
               'video stays unresolved and unavailable')),
}

ROLES = {Modality.TEXT: TEXT_ROLES, Modality.SPEECH: SPEECH_ROLES,
         Modality.IMAGE: IMAGE_MODEL_ROLES, Modality.VIDEO: VIDEO_MODEL_ROLES}

# Roles somebody named upstream but whose weight file nobody here has opened.
DECLARED_IDENTITY_ROLES = tuple(
    role for role in TEXT_ROLES + SPEECH_ROLES + IMAGE_MODEL_ROLES
    if role.repository and role.bytes is None)


def validate():
    """Machine-checkable invariants. Raise rather than let a status drift."""
    if DEFAULT_TEXT_MODEL_ID != STOCK_BONSAI_ROLE_ID:
        raise ValueError('this declaration does not change the executable text default')
    for modality, entry in MODALITIES.items():
        if entry.runtime_qualified:
            raise ValueError('no modality may hardcode a verified runtime: ' + modality.value)
        roles = ROLES[modality]
        for role in roles:
            if role.substitutes_intended:
                raise ValueError('no role may substitute the intended model: ' + role.role_id)
            if role.installation is Installation.ABSENT and role.sha256_verified_here:
                raise ValueError('an absent role cannot have a verified hash: ' + role.role_id)
            if (role.installation is Installation.PRESENT_UNVERIFIED
                    and (role.sha256 is None or role.bytes is None or role.bytes <= 0)):
                raise ValueError('a present-unverified role must record its receipt identity and '
                                 'positive byte count: ' + role.role_id)
            if (role.runtime_qualification is RuntimeQualification.SCOPED_RUN_PROVEN
                    and role.selection is not Selection.PRESERVED_BASELINE):
                raise ValueError('only a preserved baseline may carry scoped run evidence: '
                                 + role.role_id)
            if (role.runtime_qualification is RuntimeQualification.UNQUALIFIED
                    and role.qualification_scope is not None):
                raise ValueError('an unqualified role cannot carry a qualification scope: '
                                 + role.role_id)
            if role.primary and role.selection is not Selection.INTENDED:
                raise ValueError('the primary role must be the intended one: ' + role.role_id)
        if modality is Modality.VIDEO:
            if roles or entry.selected:
                raise ValueError('video has no selected model, route or artifact')
            continue
        primaries = [role for role in roles if role.primary]
        if len(primaries) != 1 or primaries[0].runtime_qualification is not RuntimeQualification.UNQUALIFIED:
            raise ValueError('each implemented modality names exactly one primary intended '
                             'role, and it is unqualified: ' + modality.value)
    if 'video' not in UNAVAILABLE_MODALITIES:
        raise ValueError('video must stay in the unavailable modality list')
    return True


def _jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _modality_document(modality):
    entry = MODALITIES[modality]
    return {'selected': entry.selected, 'resolved': entry.resolved, 'harness': entry.harness,
            'stages': list(entry.stages),
            'source_compatibility_proven': entry.source_compatibility_proven,
            'runtime_qualified': entry.runtime_qualified, 'qualification': 'Unresolved',
            'declared_roles': [role.role_id for role in ROLES[modality]],
            'roles': [asdict(role) for role in ROLES[modality]],
            'notes': list(entry.notes)}


def intent():
    """The JSON-ready declaration. Deterministic: no clock, no random, no filesystem."""
    validate()
    return _jsonable({
        'schema': SCHEMA,
        'default_text_model_id': DEFAULT_TEXT_MODEL_ID,
        'modality_order': [modality.value for modality in MODALITIES],
        'modalities': {modality.value: _modality_document(modality) for modality in MODALITIES},
        'unavailable_modalities': list(UNAVAILABLE_MODALITIES),
        'declared_identity_roles': [role.role_id for role in DECLARED_IDENTITY_ROLES],
        'rules': list(RULES),
        'actual_inference': False,
        'assets_downloaded': False,
        'weights_opened': False,
        'models_run': False,
    })
