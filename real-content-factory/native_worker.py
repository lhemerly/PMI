"""Fixed native exec bridge inside the original host/core/VRAM supervisors."""
import argparse
import os
from pathlib import Path
import json
import resource
import sys

from launch_policy import require_real_execution
from content_factory import bounded_read, native_argv, ENVIRONMENT


def main():
    # The legacy entry is held before reading its request. A demo argument grants
    # nothing: the separate gate must validate the controller's sealed capability.
    if '--demo-capability-path' not in sys.argv[1:]:
        require_real_execution()
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--demo-capability-path')
    args = parser.parse_args()
    request = json.loads(bounded_read(args.request, 32768))
    if args.demo_capability_path is not None:
        from model_profiles import validate_model_request
        validate_model_request(request)
        from research.demo_authority import gate_native
        gate_native(args.request, request, capability_path=args.demo_capability_path)
        argv = native_argv(request['model_id'], request['prompt'],
                           profile_id=request['profile_id'], role=request['role'])
    else:
        if set(request) != {'model_id', 'prompt'}:
            raise ValueError('unexpected native request keys')
        argv = native_argv(request['model_id'], request['prompt'])
    stage = args.request.parent
    for number, name in [(1, 'native.stdout.log'), (2, 'native.stderr.log')]:
        descriptor = os.open(stage / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.dup2(descriptor, number)
        os.close(descriptor)
    stdin = os.open('/dev/null', os.O_RDONLY)
    os.dup2(stdin, 0)
    os.close(stdin)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
    environment = dict(ENVIRONMENT)
    if request['model_id'] == 'bonsai-stock':
        environment['LD_LIBRARY_PATH'] = str(Path(argv[0]).parent) + ':/usr/lib'
    os.execve(argv[0], argv, environment)


if __name__ == '__main__':
    main()
