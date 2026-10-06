#!/usr/bin/env python3
"""Run one guarded PMI replay; never retry, flush caches or download weights.

Use sequentially after passive host admission. Each process starts with empty
PMI caches. OS cache state is observed through process read_bytes, not assumed.
The supplied stream fixes decoder inputs; arithmetic may still change routing.
"""
import argparse
import ctypes as C
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def artifact(path):
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def gpu_exec():
    # Runs inside BOTH supervisors, not in the launcher process.
    library_dir, pci, sysfs, output, *command = sys.argv[2:]
    class Caps(C.Structure):
        _fields_ = [(name, C.c_bool) for name in
                    ('async_', 'host_buffer', 'buffer_from_host_ptr', 'events', 'mmap_support')]
    class Props(C.Structure):
        _fields_ = [('name', C.c_char_p), ('description', C.c_char_p),
                    ('memory_free', C.c_size_t), ('memory_total', C.c_size_t),
                    ('type', C.c_int), ('device_id', C.c_char_p), ('caps', Caps)]
    base = C.CDLL(str(Path(library_dir) / 'libggml-base.so.0.25.3'))
    vk = C.CDLL(str(Path(library_dir) / 'libggml-vulkan.so.0.25.3'))
    vk.ggml_backend_vk_reg.restype = C.c_void_p
    base.ggml_backend_reg_dev_count.argtypes = [C.c_void_p]
    base.ggml_backend_reg_dev_count.restype = C.c_size_t
    base.ggml_backend_reg_dev_get.argtypes = [C.c_void_p, C.c_size_t]
    base.ggml_backend_reg_dev_get.restype = C.c_void_p
    base.ggml_backend_dev_get_props.argtypes = [C.c_void_p, C.POINTER(Props)]
    reg = vk.ggml_backend_vk_reg()
    if not reg or not base.ggml_backend_reg_dev_count(reg):
        raise RuntimeError('No Vulkan device; no fallback')
    props = Props()
    base.ggml_backend_dev_get_props(base.ggml_backend_reg_dev_get(reg, 0), C.byref(props))
    decode = lambda value: value.decode() if value else None
    identity = {name: decode(getattr(props, name)) for name in ('name', 'description', 'device_id')}
    identity.update(index=0, type=props.type, reported_total_bytes=props.memory_total,
                    reported_free_bytes=props.memory_free, utc=utc(), sysfs_path=str(Path(sysfs).resolve()))
    if props.type != 1 or identity['device_id'] != pci or Path(sysfs).resolve().name != pci:
        raise RuntimeError('Vulkan index 0 and guarded physical GPU differ')
    write(Path(output), identity)
    os.execv(command[0], command)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pmi', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--expected-revision', required=True)
    p.add_argument('--tokens-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New exclusive run directory')
    p.add_argument('--backend', choices=['cpu', 'device-vulkan'], required=True)
    p.add_argument('--dense-pages', choices=['native','base'], default='native')
    p.add_argument('--loader-diagnostics', action='store_true')
    p.add_argument('--dispatch', choices=['all', 'split'], default='all')
    p.add_argument('--device-upload-host-policy', choices=['retained','transient'], default='retained')
    p.add_argument('--device-upload-host-io', choices=['buffered','direct-preferred','direct-required'], default='buffered')
    p.add_argument('--device-cache-bytes', type=int, default=268435456)
    p.add_argument('--host-cache-bytes', type=int, default=536870912)
    p.add_argument('--libraries', type=Path, required=True)
    p.add_argument('--pci-device', default='0000:04:00.0')
    p.add_argument('--sysfs-dir', type=Path, default=Path('/sys/class/drm/card1/device'))
    p.add_argument('--prompt', default='Hello')
    p.add_argument('--context', type=int, default=256)
    args = p.parse_args()
    if args.dispatch == 'split' and args.backend == 'cpu':
        p.error('Split dispatch requires the explicit device backend')
    if args.device_upload_host_io != 'buffered' and args.backend == 'cpu':
        p.error('Direct upload reads require the explicit device backend')
    if not args.tokens_file.is_file() or args.tokens_file.stat().st_size > 65536:
        p.error('Fixed token file must be regular and <=65536 bytes')
    with args.tokens_file.open('rb') as file:
        raw = file.read(65537)
    if len(raw) > 65536:
        p.error('Fixed token file grew beyond byte limit')
    try:
        tokens = [int(x) for x in raw.split()]
    except ValueError:
        p.error('Fixed token stream contains a malformed integer')
    if not 0 < len(tokens) <= 65536 or any(x < 0 or x > 2147483647 for x in tokens):
        p.error('Invalid fixed token stream')
    manifest_path = args.model_dir / 'all-weights-verified.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest['revision'] != args.expected_revision or not manifest['complete']
            or not manifest['full_file_sha256_verified']):
        p.error('Expected complete, previously SHA-verified checkpoint revision')
    weights = []
    for entry in manifest['files']:
        file = (args.model_dir / entry['name']).resolve()
        if file.parent != args.model_dir.resolve() or file.stat().st_size != entry['size']:
            p.error('Shard path or size differs from verified manifest')
        weights.append(str(file))
    if args.device_cache_bytes <= 0 or args.device_cache_bytes > 2 * 1024**3:
        p.error('Device cache must be positive and <=2GiB for this guarded campaign')
    if args.host_cache_bytes <= 0 or args.host_cache_bytes > 1024**3:
        p.error('Host cache must be positive and <=1GiB for this guarded campaign')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    command = [str(args.pmi.resolve()), 'run', *weights, '--prompt', args.prompt,
               '--max-tokens', str(len(tokens)), '--context', str(args.context),
               '--prefill-chunk', '1', '--dense-load', 'direct-required', '--cpu-repack', 'off',
               '--cache-policy', 'lru', '--cache-bytes', str(args.host_cache_bytes),
               '--teacher-forcing-file', str(args.tokens_file.resolve()),
               '--projection-backend', args.backend, '--projection-dispatch', args.dispatch, '--profile-paging', 'on', '--dense-pages', args.dense_pages,
               '--progress-jsonl', str(out / 'progress.jsonl'),
               '--device-upload-host-policy', args.device_upload_host_policy]
    # Preserve the default command's compatibility with earlier saved binaries.
    if args.device_upload_host_io != 'buffered':
        command += ['--device-upload-host-io', args.device_upload_host_io]
    if args.loader_diagnostics or args.dense_pages == 'base':
        command += ['--load-trace-jsonl', str(out / 'loader.jsonl')]
    if args.backend == 'device-vulkan':
        command += ['--projection-activation', 'backend-native', '--projection-device-index', '0',
                    '--device-cache-bytes', str(args.device_cache_bytes), '--device-graph-bytes', '4194304']
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--gpu-exec',
                   str(args.libraries.resolve()), args.pci_device, str(args.sysfs_dir),
                   str(out / 'gpu-identity.json'), *command]
    guarded = [sys.executable, '-B', str(ROOT / 'tools/run_bounded.py'),
               '--wall-seconds', '120', '--max-rss-bytes', '8GiB', '--as-bytes', '12GiB',
               '--admit-available-bytes', '14GiB', '--min-available-bytes', '8GiB',
               '--max-swap-growth-bytes', '64MiB', '--max-system-swap-growth-bytes', '512MiB',
               '--max-zram-growth-bytes', '128MiB', '--max-swapout-bytes', '256MiB',
               '--max-psi-some-us', '500000', '--max-psi-full-us', '250000',
               '--interval-seconds', '0.25', '--term-grace-seconds', '3', '--kill-wait-seconds', '5',
               *(['--loader-diagnostics'] if args.loader_diagnostics else []),
               '--log', str(out / 'host.jsonl'), '--output', str(out / 'trial.log'), '--',
               sys.executable, '-B', str(ROOT / 'tools/run_vulkan_smoke.py'),
               '--sysfs-dir', str(args.sysfs_dir), '--log', str(out / 'vram.jsonl'),
               '--max-vram-bytes', '4GiB', '--wall-seconds', '110', '--interval-seconds', '0.1',
               '--term-grace-seconds', '1', '--kill-wait-seconds', '1', '--', *command]
    write(out / 'command.json', {'started_utc': utc(), 'command': guarded, 'tokens': tokens,
          'checkpoint': manifest, 'manifest': artifact(manifest_path),
          'binary': artifact(args.pmi.resolve()), 'token_file': artifact(args.tokens_file.resolve()),
          'runner': artifact(Path(__file__).resolve()), 'automatic_retries': 0,
          'dispatch': args.dispatch, 'device_upload_host_policy': args.device_upload_host_policy,
          'device_upload_host_io': args.device_upload_host_io, 'host_cache_bytes': args.host_cache_bytes,
          'dense_pages': args.dense_pages, 'loader_diagnostics': args.loader_diagnostics, 'cache_state': 'empty PMI caches at process start; OS cache uncontrolled/observed, never flushed'})
    # Quota includes retained payload, active upload leases and direct staging.
    # These are shared/temporary ownership of allocations, not additive caches.
    result = subprocess.run(guarded, cwd=ROOT)
    final = {'finished_utc': utc(), 'returncode': result.returncode, 'automatic_retries': 0}
    write(out / 'run.json', final)
    if result.returncode:
        print(json.dumps(final), flush=True)
        return result.returncode
    lines = (out / 'trial.log').read_text().splitlines()
    emitted = [[int(x) for x in row.split()[1:]] for row in lines if row.startswith('TOKENS ')]
    predictions = [[int(x) for x in row.split()[1:]] for row in lines if row.startswith('PREDICTIONS ')]
    metrics = [json.loads(row.split(' ', 1)[1]) for row in lines if row.startswith('PMI_METRICS ')]
    if emitted != [tokens] or len(predictions) != 1 or len(predictions[0]) != len(tokens) or len(metrics) != 1:
        raise RuntimeError('Replay output or prediction count differs from fixed stream')
    m = metrics[0]
    if m['device_upload_host_policy'] != args.device_upload_host_policy:
        raise RuntimeError('Requested upload host policy not reflected in metrics')
    if m['cache_capacity_bytes'] != args.host_cache_bytes:
        raise RuntimeError('Requested host cache quota not reflected in metrics')
    if m.get('device_upload_host_io','buffered') != args.device_upload_host_io:
        raise RuntimeError('Requested upload read policy not reflected in metrics')
    if args.device_upload_host_io == 'direct-required':
        io=m['provider_io']
        if not io['direct_completed'] or io['fallback_count']:
            raise RuntimeError('Required direct upload reads did not complete without runtime fallback')
    if (not m['teacher_forcing'] or m['decode_evaluations'] != len(tokens) - 1
            or m['cpu_repack'] or m['dense_load']['fallback_count']
            or m['dense_load']['effective'] != 'direct-required'):
        raise RuntimeError('Replay did not use expected evaluation/loading policy')
    if args.dense_pages == 'base' and (not m['dense_pages']['base_requested'] or
            not m['dense_pages']['observed'] or not m['dense_pages']['advised_buffers'] or
            not m['dense_pages']['advised_bytes']):
        raise RuntimeError('Requested owned-buffer base-page policy was not applied')
    for file in ('host.jsonl', 'vram.jsonl'):
        guard = json.loads((out / file).read_text().splitlines()[-1])
        if guard['status'] != 'completed' or guard.get('remaining_live_pids'):
            raise RuntimeError('Guard completion/cleanup failed')
    write(out / 'metrics.json', m)
    dec = [e for e in m['evaluations'] if e['phase'] == 'decode']
    summary = dict(final, backend=args.backend, dispatch=args.dispatch, dense_pages=m['dense_pages'], setup_ms=m['setup_ms'], prefill_ms=m['prefill_ms'],
                   decode_ms=m['decode_ms'], decode_evaluations_per_second=len(dec)*1000/m['decode_ms'],
                   decode_storage_read_bytes=sum(e['process_storage_read_bytes_delta'] for e in dec),
                   provider_bytes=m['io_bytes'], prediction_matches=m['prediction_matches'],
                   device_cache=m['device_projection']['cache'] if args.backend != 'cpu' else None)
    summary['split'] = {key: m[key] for key in ('split_enabled', 'split_groups',
        'split_cpu_projections', 'split_device_projections', 'split_critical_ms',
        'split_cpu_ms', 'split_device_ms', 'split_join_ms', 'split_overlap_ms')}
    if args.dispatch == 'split':
        if not m['split_enabled'] or any(e['device_projection']['cache']['uploads'] or
                e['device_projection']['cache']['provider_requests'] for e in dec):
            raise RuntimeError('Split decode uploaded or fetched device misses')
    write(out / 'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--gpu-exec':
        gpu_exec()
    else:
        sys.exit(main())
