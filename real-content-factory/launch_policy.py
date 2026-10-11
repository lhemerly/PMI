"""Preserve the unresolved platform denial. Data/flags cannot lift this hold."""

class PlatformHeld(RuntimeError):
    pass


def require_real_execution():
    raise PlatformHeld(
        "HELD: prior GPU/model launch rejected before process creation. "
        "Supported platform must recognize direct trusted approval of the exact "
        "new prompt/model-stage workload; parent must coordinate exclusive fresh admission. "
        "No local approval file, environment variable or CLI option grants execution."
    )


def require_approved_kokoro_sample(text, voice_id):
    """Exact CPU speech exception reviewed separately from held Qwen GPU action.

    Original denial: thread 01a1123d-22ca-7786-9ee9-3475dbf7e034,
    turn 01a11be5-db30-75ab-85f4-2d8513714692. All general execution
    entry points retain require_real_execution(). No GPU approval is granted.
    """
    if text != 'O enxame é o instrumento da minha vontade.' or voice_id != 'pf_dora':
        require_real_execution()
