"""CPU-only contract tests, no actual model, supervisor or device invocation."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import content_factory as factory
from launch_policy import PlatformHeld
from llama_backend import LlamaBackend, guarded_argv, require_cleanup, hash_before_deadline, validate_placement, reject_fit_failure, validate_direct_loading, reject_loading_fallback


class ContractBackend:
    actual_inference = False

    def __init__(self, release=True, incomplete=False):
        self.release = release
        self.incomplete = incomplete
        self.events = []
        self.live = False

    def run_stage(self, key, prompt, stage, deadline):
        if self.live:
            raise AssertionError('overlapping model loads')
        self.live = True
        self.events.append(('load', key))
        self.events.append(('exit-and-release', key))
        self.live = False
        return {'returncode': 0, 'owned_cleanup_empty': self.release,
                'postflight_passed': self.release,
                'raw_stdout': b'contract text' if self.incomplete else b'contract text [end of text]\n'}


def png_bytes(width=512, height=512):
    """Minimal but real PNG fixture. Bytes only; no model, device or encoder run."""
    import struct
    import zlib

    def chunk(kind, payload):
        return (struct.pack('>I', len(payload)) + kind + payload
                + struct.pack('>I', zlib.crc32(kind + payload) & 0xffffffff))

    header = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    raw = b''.join(b'\0' + b'\x40' * (width * 3) for _ in range(height))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header)
            + chunk(b'IDAT', zlib.compress(raw, 1)) + chunk(b'IEND', b''))


def jpeg_bytes(width=512, height=512):
    """SOI + APP0 + SOF0 + EOI fixture: enough for structural frame validation."""
    return (b'\xff\xd8'
            + b'\xff\xe0' + (16).to_bytes(2, 'big') + b'JFIF\0' + b'\x01\x02\0\0\x01\0\x01\0\0'
            + b'\xff\xc0' + (11).to_bytes(2, 'big') + b'\x08'
            + height.to_bytes(2, 'big') + width.to_bytes(2, 'big') + b'\x01\x01\x11\0'
            + b'\xff\xd9')


class _SpeechOnly:
    """Marker speech fixture used only to prove image ordering/validation refusals."""
    actual_inference = False

    def run_stage(self, text, stage, deadline):
        raise AssertionError('two-model text selection is refused before any speech stage')


SUCCESS_HISTORY = {'status': {'status_str': 'success', 'completed': True},
                   'outputs': {'13': {'images': [{'filename': 'answer_00001_.png',
                                                  'subfolder': 'PMI', 'type': 'output'}]}}}
FAILED_HISTORY = {'status': {'status_str': 'error', 'completed': False,
                             'messages': [['execution_error', {'exception_message': 'out of memory'}]]}}


class FakeComfy:
    """Scripted loopback stand-in for ComfyUI. No socket, server, checkpoint or GPU."""
    actual_inference = False

    def __init__(self, artifact=None, node_errors=None, history=None, unload_fails=False,
                 prompt_id='12345678-1234-5678-1234-567812345678', vram_samples=None):
        self.endpoint = 'http://127.0.0.1:8188'
        self.calls = []
        # b'' is a distinct "endpoint returned nothing" case from the default fixture.
        self.artifact = png_bytes() if artifact is None else artifact
        self.node_errors = node_errors
        self.history = [SUCCESS_HISTORY] if history is None else list(history)
        self.unload_fails = unload_fails
        self.prompt_id = prompt_id
        self.vram_samples = list(vram_samples or [7 * 1024**3, 7 * 1024**3])

    def run_stage(self, prompt, stage, deadline):
        """Backend-shaped wrapper so sequencing tests use the same receipt contract."""
        import comfy_backend as comfy
        return comfy.run_image(self, prompt, stage, deadline, sleeper=lambda seconds: None)

    def call(self, path, payload=None, method='GET', limit=None):
        import comfy_backend as comfy
        self.calls.append((method, path.split('?')[0], payload))
        if path.startswith('/view?'):
            if self.artifact is None:
                raise comfy.ComfyFailure('GET /view? response body is empty')
            return self.artifact
        if path == '/free' and self.unload_fails:
            raise comfy.ComfyFailure('POST /free refused with HTTP 404')
        return b'{}'

    def call_json(self, path, payload=None, method='GET', limit=None):
        self.calls.append((method, path.split('?')[0], payload))
        if path == '/prompt':
            if self.node_errors:
                return {'prompt_id': self.prompt_id, 'number': 1, 'node_errors': self.node_errors}
            return {'prompt_id': self.prompt_id, 'number': 1, 'node_errors': {}}
        if path == '/system_stats':
            free = self.vram_samples.pop(0) if len(self.vram_samples) > 1 else self.vram_samples[0]
            return {'devices': [{'name': 'mock GPU', 'type': 'cuda',
                                 'vram_free': free, 'vram_total': 8 * 1024**3}]}
        if path.startswith('/history/'):
            entry = self.history.pop(0) if self.history else {}
            return {self.prompt_id: entry} if entry else {}
        return {}


class Tests(unittest.TestCase):
    def test_admission_allows_isolated_host_page_and_rejects_pressure_bad_counters_OOM(self):
        from admission_policy import evaluate
        samples = [{'time':i*.5, 'available':22*1024**3, 'swapout':4096 if i else 0,
                    'PSI_full_us':i*35,'PSI_some_us':i*36,'oom_kill':0} for i in range(61)]
        self.assertEqual(evaluate(samples)['swapout_growth_bytes'],4096)
        for kind in ('swap','full','some','missing','reset','oom','gap','RAM'):
            rows = [dict(s) for s in samples]
            if kind=='swap':
                for i,r in enumerate(rows):r['swapout']=int(i*.5*2*1048576)
            elif kind in ('full','some'):
                for i,r in enumerate(rows):r['PSI_'+kind+'_us']=int(i*.5*400000)
            elif kind=='missing':del rows[10]['PSI_full_us']
            elif kind=='reset':rows[-1]['swapout']=0
            elif kind=='oom':rows[-1]['oom_kill']=1
            elif kind=='gap':rows[10]['time']=rows[9]['time']
            elif kind=='RAM':rows[-1]['available']=15*1024**3
            with self.subTest(kind=kind), self.assertRaises(ValueError):evaluate(rows)

    def test_loading_telemetry_preserves_host_and_owned_identity_and_marks_missing_sources(self):
        from load_telemetry import LoadTelemetry
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); proc = base/'proc'; proc.mkdir(); (proc/'pressure').mkdir()
            for name in ('cpu','io','memory'):
                (proc/'pressure'/name).write_text('some avg10=1.00 avg60=0.00 avg300=0.00 total=100\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=10\n')
            (proc/'vmstat').write_text('pgfault 100\npgmajfault 2\noom_kill 0\npswpin 0\npswpout 4\n')
            (proc/'meminfo').write_text('MemAvailable: 20000000 kB\n')
            process = proc/'123'; process.mkdir()
            fields = ['0']*30; fields[0]='R'; fields[7]='50'; fields[9]='3'; fields[11]='7'; fields[19]='999'
            (process/'stat').write_text('123 (name with spaces) '+' '.join(fields))
            (process/'io').write_text('read_bytes: 4096\nwrite_bytes: 0\n')
            (process/'ns').mkdir(); (process/'ns'/'pid').symlink_to('pid:[1]')
            (base/'host.jsonl').write_text(json.dumps({'event':'spawn','pid':123,'start_ticks':999,'pid_namespace':'pid:[1]'})+'\n'+json.dumps({'event':'sample','elapsed_seconds':1,'members':[{'pid':123,'alive':True}]})+'\n')
            model = base/'model.gguf'; model.write_bytes(b'fixture')
            (process/'fd').mkdir(); (process/'fdinfo').mkdir()
            (process/'fd'/'7').symlink_to(model)
            import os
            (process/'fdinfo'/'7').write_text('flags: '+oct(os.O_RDONLY|os.O_DIRECT)[2:]+'\n')
            telemetry = LoadTelemetry(base, proc, model); telemetry.sample(force=True)
            self.assertTrue(telemetry.verified_descendant(123,999,{'pid':123,'start_ticks':999,'pid_namespace':'pid:[1]'}))
            self.assertFalse(telemetry.verified_descendant(123,999,{'pid':123,'start_ticks':998,'pid_namespace':'pid:[1]'}))
            self.assertTrue(telemetry.direct_io_evidence[0]['O_DIRECT'])
            self.assertEqual(telemetry.direct_io_evidence[0]['model_inode'],model.stat().st_ino)
            fields[7]='55'; (process/'stat').write_text('123 (name with spaces) '+' '.join(fields))
            (proc/'pressure/io').unlink(); telemetry.sample(force=True); telemetry.output.close()
            rows = [json.loads(line) for line in (base/'loading-telemetry.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['owned'][0]['start_ticks'],999)
            self.assertEqual(rows[-1]['owned'][0]['delta_since_first_observation']['minor_faults'],5)
            self.assertEqual(rows[-1]['host_delta_from_baseline']['oom_kill'],0)
            self.assertIsNone(rows[-1]['PSI']['io'])
            self.assertTrue(rows[-1]['errors'])

    def test_direct_loading_requires_real_fd_and_async_receipts_and_rejects_fallback(self):
        log = ('common_fit_params: successfully fit params to free device memory\n'
               'load_tensors: loading model tensors (load_mode = dio)\n'
               'load_all_data: using async uploads for device Vulkan0, buffer type Vulkan0, backend Vulkan0\n')
        evidence = [{'O_DIRECT':True, 'pid':123,'start_ticks':999}]
        proof = validate_direct_loading(log, evidence)
        self.assertEqual(proof['staging_base_bytes'],256*1048576)
        for bad, descriptors in [(log,[]), (log,[{'O_DIRECT':False}]),
                                 (log.replace('dio','mmap'), evidence),
                                 (log.splitlines()[0],evidence),
                                 (log+'Falling back to buffered I/O',evidence),
                                 (log+'Falling back to buffered IO due to Invalid argument',evidence)]:
            with self.assertRaises(RuntimeError):
                validate_direct_loading(bad, descriptors)
        with self.assertRaises(RuntimeError):
            reject_loading_fallback('Failed to open file. Falling back to buffered I/O')

    def test_recorded_fitter_none_probe_is_not_actual_loading_fallback(self):
        probe = ('0.00.020.574 I cmn  common_init_: fitting params to device memory ...\n'
                 '0.00.020.590 I common_params_fit_impl: getting device memory data for initial parameters:\n'
                 '0.00.413.137 I load_tensors: loading model tensors, this can take a while... (load_mode = none)\n'
                 '0.00.493.023 I common_fit_params: successfully fit params to free device memory\n')
        reject_loading_fallback(probe)
        actual = ('load_tensors: loading model tensors (load_mode = dio)\n'
                  'load_all_data: using async uploads for device Vulkan0, buffer type Vulkan0, backend Vulkan0\n')
        self.assertTrue(validate_direct_loading(probe+actual,[{'O_DIRECT':True}])['Vulkan_async_upload'])
        for log in [probe, probe+'load_mode = none', probe+'load_mode = mmap',
                    'load_mode = none', probe+actual+'Falling back to buffered I/O',
                    probe+actual+'Falling back to buffered IO due to Invalid argument']:
            with self.assertRaises(RuntimeError):
                validate_direct_loading(log,[{'O_DIRECT':True}])

    def test_fitter_accepts_measured_partial_placement_and_rejects_failed_fit(self):
        log = ('common_fit_params: successfully fit params to free device memory\n'
               'offloaded 42/65 layers to GPU\nn_ctx = 512\n'
               'Vulkan0 model buffer size = 3900.00 MiB\nCPU_Mapped model buffer size = 1700.00 MiB\n')
        result = validate_placement(log, 65, 512, True)
        self.assertEqual(result['GPU_layers'], 42)
        for bad in [log.replace('42/65','42/64'), log.replace('512','4096'),
                    log.replace('successfully fit params to free device memory','failed to fit params'),
                    log.replace('successfully fit params to free device memory','')]:
            with self.assertRaises(RuntimeError):
                validate_placement(bad, 65, 512, True)
        with self.assertRaises(RuntimeError):
            validate_placement(log, 65, 512, False)
        with self.assertRaises(RuntimeError):
            reject_fit_failure('encountered an error while trying to fit params')

    def test_fitter_is_only_enabled_for_exact_smoke_and_keeps_bounds(self):
        prompt = (factory.HERE/'first-run-prompt.txt').read_text()
        argv = factory.native_argv('bonsai-stock', prompt)
        self.assertEqual(argv[argv.index('--fit')+1], 'on')
        self.assertEqual(argv[argv.index('--fit-target')+1], '1536')
        self.assertEqual(argv[argv.index('--load-mode')+1], 'dio')
        self.assertNotIn('--gpu-layers', argv)
        self.assertNotIn('--override-tensor', argv)
        self.assertEqual(argv[argv.index('--ctx-size')+1], '512')
        self.assertEqual(argv[argv.index('--predict')+1], '64')
        other = factory.native_argv('bonsai-stock', prompt+'!')
        self.assertEqual(other[other.index('--fit')+1], 'off')

    def test_two_selected_models_are_serial_and_outputs_are_not_inference_claims(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            backend = ContractBackend()
            result = factory._run_sequence('Write a short story', ['qwen3-stock', 'qwen25-modified'],
                                           base / 'job', base / 'lease', backend)
            self.assertEqual(result['status'], 'ContractOnly')
            self.assertFalse(result['actual_inference'])
            self.assertEqual(backend.events, [('load', 'qwen3-stock'), ('exit-and-release', 'qwen3-stock'),
                                              ('load', 'qwen25-modified'), ('exit-and-release', 'qwen25-modified')])
            for row in result['stages']:
                self.assertEqual(Path(row['answer_file']).read_text(), 'contract text\n')

    def test_release_failure_or_incomplete_answer_holds_second_model(self):
        for backend in [ContractBackend(release=False), ContractBackend(incomplete=True)]:
            with tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                with self.assertRaises((ValueError, RuntimeError)):
                    factory._run_sequence('Prompt', ['qwen3-stock', 'qwen25-modified'],
                                          base / 'job', base / 'lease', backend)
                self.assertEqual(len(backend.events), 2)
                saved = json.loads((base / 'job/result.json').read_text())
                self.assertEqual(saved['status'], 'StoppedNoRetry')
                self.assertFalse(saved['actual_inference'])
                self.assertTrue(saved['later_models_held'])

    def test_prompt_and_model_ids_cannot_supply_commands(self):
        prompt = '$(touch /tmp/never) `echo shell`; --model fake é'
        argv = factory.native_argv('qwen3-stock', prompt)
        self.assertEqual(argv[0], '/usr/bin/llama-completion')
        self.assertEqual(argv[-2:], ['--prompt', prompt])
        for invalid in ['', ' ', '\0', 'é' * 2049]:
            with self.assertRaises(ValueError):
                factory.check_prompt(invalid)
        for ids in [[], ['qwen3-stock'] * 2, ['anything'], ['qwen3-stock', 'qwen25-modified', 'anything']]:
            with self.assertRaises(ValueError):
                factory.select_models(ids)

    def test_all_physical_entry_points_are_held_before_outputs_or_helpers(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'must-not-exist'
            with patch('subprocess.Popen', side_effect=AssertionError('physical process attempted')):
                with self.assertRaises(PlatformHeld):
                    factory.main(['run', '--out', str(output)])
                with self.assertRaises(PlatformHeld):
                    LlamaBackend().run_stage('qwen3-stock', 'hello', output, 0)
                with self.assertRaises(PlatformHeld):
                    factory._run_sequence('hello', ['qwen3-stock'], output, output / 'lease', LlamaBackend())
            self.assertFalse(output.exists())

    def test_bounded_input_plan_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            prompt = base / 'prompt.txt'
            prompt.write_text('A useful operator prompt')
            output = base / 'plan.json'
            args = ['plan', '--prompt-file', str(prompt), '--models', 'qwen3-stock,qwen25-modified', '--out', str(output)]
            self.assertEqual(factory.main(args), 0)
            document = json.loads(output.read_text())
            self.assertEqual(document['status'], 'HELD')
            self.assertEqual(len(document['stages']), 2)
            self.assertFalse(document['actual_inference'])
            with self.assertRaises(FileExistsError):
                factory.main(args)
            prompt.write_bytes(b'x' * 4097)
            with self.assertRaises(ValueError):
                factory.bounded_read(prompt, 4096)

    def test_guard_command_preserves_resource_bounds_and_owned_parent(self):
        argv = guarded_argv(Path('/operator/stage'), {'pid': 123, 'start_ticks': 456, 'pid_namespace': 'pid:[789]'}, 10)
        for key, expected in [('--max-rss-bytes', '12GiB'), ('--as-bytes', '16GiB'),
                              ('--min-available-bytes', '12GiB'), ('--max-vram-bytes', '7GiB'),
                              ('--parent-pid', '123'), ('--wall-seconds', '60')]:
            self.assertEqual(argv[argv.index(key) + 1], expected)
        self.assertNotIn('shell', argv)
        self.assertEqual(argv[-2:], ['--request', '/operator/stage/request.json'])

    def test_release_evidence_must_have_explicit_final_empty_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            host = {'event': 'final', 'remaining_live_pids': [], 'monitor_errors': [], 'command_started': True}
            gpu = {'event': 'final', 'cleanup_incomplete': False, 'errors': []}
            (base / 'host.jsonl').write_text(json.dumps(host) + '\n')
            (base / 'vram.jsonl').write_text(json.dumps(gpu) + '\n')
            require_cleanup(base)
            del host['remaining_live_pids']
            (base / 'host.jsonl').write_text(json.dumps(host) + '\n')
            with self.assertRaises(RuntimeError):
                require_cleanup(base)

    def test_integrity_hash_has_an_absolute_deadline(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'tiny-input'
            path.write_bytes(b'abc')
            self.assertEqual(hash_before_deadline(path, time.monotonic() + 1),
                             'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
            with self.assertRaises(TimeoutError):
                hash_before_deadline(path, time.monotonic() - 1)

    def test_image_endpoint_is_loopback_only_and_never_remote_or_credentialed(self):
        import comfy_backend as comfy
        self.assertEqual(comfy.DEFAULT_ENDPOINT, 'http://127.0.0.1:8188')
        for good in ['http://127.0.0.1:8188', 'http://127.0.0.1:8188/', 'http://localhost:8188',
                     'http://[::1]:8188']:
            self.assertEqual(comfy.check_endpoint(good), good.rstrip('/'))
        for bad in ['https://127.0.0.1:8188', 'http://127.0.0.1', 'http://10.0.0.5:8188',
                    'http://example.com:8188', 'http://user:pass@127.0.0.1:8188',
                    'http://127.0.0.1:8188/api', 'http://127.0.0.1:8188?x=1',
                    'http://127.0.0.1:8188#f', 'http://127.0.0.1:99999', 'unix:///run/comfy.sock',
                    '', 'http://0.0.0.0:8188', 'http://127.0.0.1:8188\0']:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                comfy.check_endpoint(bad)
        for bad in [None, 8188, ['http://127.0.0.1:8188'], 'http://127.0.0.1:8188' * 40]:
            with self.assertRaises(ValueError):
                comfy.check_endpoint(bad)
        # CLI beats environment beats default; configuration never grants execution.
        self.assertEqual(comfy.resolve_endpoint(None, {}), 'http://127.0.0.1:8188')
        self.assertEqual(comfy.resolve_endpoint(None, {'PMI_COMFY_ENDPOINT': 'http://127.0.0.1:9000'}),
                         'http://127.0.0.1:9000')
        self.assertEqual(comfy.resolve_endpoint('http://127.0.0.1:8189',
                                                {'PMI_COMFY_ENDPOINT': 'http://127.0.0.1:9000'}),
                         'http://127.0.0.1:8189')
        with self.assertRaises(ValueError):
            comfy.resolve_endpoint(None, {'PMI_COMFY_ENDPOINT': 'https://example.com'})

    def test_image_workflow_source_snapshot_is_the_pinned_official_distilled_template(self):
        import comfy_backend as comfy
        self.assertEqual(comfy.SOURCE_WORKFLOW_BYTES, 69872)
        self.assertEqual(comfy.SOURCE_WORKFLOW_SHA256,
                         '8bb879856e54765b6c3ed219181e855981d8d10cfd19acff836603d9b56e838f')
        document = comfy.read_source_workflow()
        self.assertEqual((document['id'], document['revision'], str(document['version'])),
                         ('92112d97-bb64-4b44-86f2-ea5691ef8f6e', 0, '0.4'))
        subgraphs = {sub['id']: sub for sub in document['definitions']['subgraphs']}
        self.assertEqual(sorted(subgraphs),
                         sorted(['a67caa28-5f85-4917-8396-36004960dd30',
                                 '7b34ab90-36f9-45ba-a665-71d418f0df18']))
        selected, base = (subgraphs['a67caa28-5f85-4917-8396-36004960dd30'],
                          subgraphs['7b34ab90-36f9-45ba-a665-71d418f0df18'])
        self.assertEqual(selected['name'], 'Text to Image (Flux.2 Klein 4B Distilled)')
        self.assertEqual(base['name'], 'Text to Image (Flux.2 Klein 4B)')

        def widgets(branch, node_type):
            return next(node['widgets_values'] for node in branch['nodes'] if node['type'] == node_type)

        def types(branch):
            return {node['type'] for node in branch['nodes']}

        # The selected branch is the distilled 4B role; the sibling is the base branch.
        self.assertEqual(widgets(selected, 'UNETLoader')[0], 'flux-2-klein-4b.safetensors')
        self.assertEqual(widgets(base, 'UNETLoader')[0], 'flux-2-klein-base-4b.safetensors')
        self.assertEqual(widgets(selected, 'CLIPLoader')[:2], ['qwen_3_4b.safetensors', 'flux2'])
        self.assertEqual(widgets(selected, 'VAELoader'), ['flux2-vae.safetensors'])
        self.assertIn('ConditioningZeroOut', types(selected))
        self.assertNotIn('ConditioningZeroOut', types(base))
        # Pinned settings are read from the distilled branch, not from the base branch.
        self.assertEqual(widgets(selected, 'Flux2Scheduler')[0], 4)
        self.assertEqual(widgets(selected, 'CFGGuider'), [1])
        self.assertEqual(widgets(selected, 'KSamplerSelect'), ['euler'])
        self.assertEqual(widgets(selected, 'EmptyFlux2LatentImage')[2], 1)
        self.assertEqual(widgets(base, 'Flux2Scheduler')[0], 20)
        self.assertEqual(widgets(base, 'CFGGuider'), [5])
        self.assertEqual(widgets(selected, 'RandomNoise')[1], 'randomize')
        # PrimitiveInt is only a UI dimension helper; the API graph binds numbers directly.
        self.assertEqual(sum(node['type'] == 'PrimitiveInt' for node in selected['nodes']), 2)
        api_classes = {record['class_type'] for record in comfy.RECIPE_NODES.values()}
        self.assertNotIn('PrimitiveInt', api_classes)
        # SaveImage is the root output node attached to the subgraph, not a node inside it.
        self.assertNotIn('SaveImage', types(selected))
        self.assertEqual(api_classes | {'PrimitiveInt'}, types(selected) | {'SaveImage'})
        # The distilled root instance and its SaveImage output are mode 4 in the UI snapshot.
        root = {node['type']: node for node in document['nodes']}
        self.assertEqual(root['a67caa28-5f85-4917-8396-36004960dd30']['mode'], 4)
        self.assertEqual(root['7b34ab90-36f9-45ba-a665-71d418f0df18']['mode'], 0)
        self.assertEqual(sorted(node['mode'] for node in document['nodes'] if node['type'] == 'SaveImage'), [0, 4])
        # The submitted API graph carries no UI subgraph records at all.
        self.assertNotIn('definitions', json.dumps(comfy.build_workflow('A quiet harbour at dawn')))
        self.assertNotIn('a67caa28-5f85-4917-8396-36004960dd30',
                         json.dumps(comfy.build_workflow('A quiet harbour at dawn')))
        self.assertEqual(document['id'], comfy.SOURCE_WORKFLOW_ID)

    def test_image_workflow_is_the_pinned_distilled_recipe_and_prompts_supply_only_text_and_seed(self):
        import comfy_backend as comfy
        prompt = 'A quiet harbour at dawn'
        workflow = comfy.build_workflow(prompt)
        self.assertEqual(list(workflow), list(comfy.GRAPH_ORDER))
        self.assertEqual([node['class_type'] for node in workflow.values()],
                         ['UNETLoader', 'CLIPLoader', 'VAELoader', 'CLIPTextEncode',
                          'ConditioningZeroOut', 'CFGGuider', 'RandomNoise', 'KSamplerSelect',
                          'Flux2Scheduler', 'EmptyFlux2LatentImage', 'SamplerCustomAdvanced',
                          'VAEDecode', 'SaveImage'])
        for node, record in workflow.items():
            self.assertEqual(set(record), {'class_type', 'inputs'}, node)
        # Only the positive text and the derived seed differ from the pinned recipe.
        expected = json.loads(json.dumps(comfy.RECIPE_NODES))
        expected['4']['inputs']['text'] = prompt
        expected['7']['inputs']['noise_seed'] = comfy.prompt_seed(prompt)
        self.assertEqual(workflow, expected)
        self.assertEqual(comfy.RECIPE_NODES['4']['inputs']['text'], '')
        self.assertEqual(comfy.RECIPE_NODES['7']['inputs']['noise_seed'], 0)
        # Distilled model roles, never the base branch and never a checkpoint loader.
        self.assertEqual(workflow['1']['inputs'], {'unet_name': 'flux-2-klein-4b.safetensors',
                                                  'weight_dtype': 'default'})
        self.assertEqual(workflow['2']['inputs'], {'clip_name': 'qwen_3_4b.safetensors',
                                                  'type': 'flux2', 'device': 'default'})
        self.assertEqual(workflow['3']['inputs'], {'vae_name': 'flux2-vae.safetensors'})
        self.assertNotIn('flux-2-klein-base-4b.safetensors', json.dumps(workflow))
        self.assertNotIn('flux-2-klein-4b-fp8.safetensors', json.dumps(workflow))
        # Exact pinned topology: 13 links and no UI helper nodes.
        edges = sorted((value[0], node, key) for node in workflow
                       for key, value in workflow[node]['inputs'].items()
                       if isinstance(value, list) and len(value) == 2)
        self.assertEqual(tuple(edges), tuple(sorted(comfy.PINNED_EDGES)))
        self.assertEqual(len(edges), 13)
        # Pinned settings and exactly one output node.
        self.assertEqual(workflow['6']['inputs']['cfg'], 1.0)
        self.assertEqual(workflow['8']['inputs']['sampler_name'], 'euler')
        self.assertEqual(workflow['9']['inputs']['steps'], 4)
        self.assertEqual(workflow['10']['inputs']['batch_size'], 1)
        self.assertEqual(comfy.recipe_dimensions(), (512, 512))
        self.assertEqual(comfy.IMAGE_BOUNDS, (comfy.IMAGE_WIDTH, comfy.IMAGE_HEIGHT))
        self.assertEqual([node for node, record in workflow.items()
                          if record['class_type'] == 'SaveImage'], [comfy.SAVE_NODE])
        self.assertEqual(workflow[comfy.SAVE_NODE]['inputs']['filename_prefix'], 'PMI/answer')
        # Deterministic seed contract: derived from the prompt, injected, and pinned to fixed.
        self.assertEqual(workflow['7']['inputs']['noise_seed'], comfy.prompt_seed(prompt))
        self.assertEqual(workflow['7']['inputs']['control_after_generate'], 'fixed')
        self.assertEqual(comfy.build_workflow(prompt)['7']['inputs']['noise_seed'],
                         workflow['7']['inputs']['noise_seed'])
        # PMI's seed contract stays inside 32 bits, narrower than the upstream snapshot's own
        # RandomNoise value, and seed=None keeps meaning "derive it from the prompt".
        self.assertEqual(comfy.build_workflow(prompt, seed=432262096)['7']['inputs']['noise_seed'],
                         432262096)
        self.assertEqual(comfy.build_workflow(prompt, seed=None)['7']['inputs']['noise_seed'],
                         comfy.prompt_seed(prompt))
        self.assertNotEqual(comfy.prompt_seed(prompt), comfy.prompt_seed(prompt + '!'))
        for bad in ['', ' ', '\0', 'é' * 2049, None, 7]:
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                comfy.check_prompt(bad)
        self.assertEqual(comfy.validate_workflow(workflow, prompt), workflow)

    def test_image_workflow_rejects_tampered_nodes_edges_models_and_settings(self):
        import comfy_backend as comfy
        prompt = 'A quiet harbour at dawn'
        workflow = comfy.build_workflow(prompt)
        tampered = (
            # Node set and node record shape.
            ('dropped output node', lambda w: w.pop('13')),
            ('dropped zero-out node', lambda w: w.pop('5')),
            ('added node', lambda w: w.__setitem__('14', w['13'])),
            ('renamed node', lambda w: w.__setitem__('40', w.pop('13'))),
            ('ui metadata added', lambda w: w['13'].__setitem__('_meta', {'title': 'x'})),
            ('ui definitions added', lambda w: w.__setitem__('definitions', {'subgraphs': []})),
            ('node record not an object', lambda w: w.__setitem__('13', ['SaveImage'])),
            # Class mutation, including the whole incompatible-checkpoint family.
            ('loader swapped', lambda w: w['1'].__setitem__('class_type', 'CheckpointLoaderSimple')),
            ('generic sampler', lambda w: w.__setitem__('11', {'class_type': 'KSampler',
                                                               'inputs': workflow['11']['inputs']})),
            ('generic latent', lambda w: w.__setitem__('10', {'class_type': 'EmptyLatentImage',
                                                              'inputs': workflow['10']['inputs']})),
            ('lora injected', lambda w: w.__setitem__('6', {'class_type': 'LoraLoader',
                                                            'inputs': workflow['6']['inputs']})),
            ('scheduler swapped', lambda w: w['9'].__setitem__('class_type', 'KSampler')),
            ('decode swapped', lambda w: w['12'].__setitem__('class_type', 'PreviewImage')),
            ('vae loader swapped', lambda w: w['3'].__setitem__('class_type', 'VAELoaderGGUF')),
            # Model roles and filenames.
            ('base branch diffusion', lambda w: w['1']['inputs'].__setitem__(
                'unet_name', 'flux-2-klein-base-4b.safetensors')),
            ('documented fp8 variant', lambda w: w['1']['inputs'].__setitem__(
                'unet_name', 'flux-2-klein-4b-fp8.safetensors')),
            ('clip file swapped', lambda w: w['2']['inputs'].__setitem__('clip_name', 'qwen3_8b.safetensors')),
            ('clip type changed', lambda w: w['2']['inputs'].__setitem__('type', 'sd3')),
            ('clip device changed', lambda w: w['2']['inputs'].__setitem__('device', 'cpu')),
            ('weight dtype changed', lambda w: w['1']['inputs'].__setitem__('weight_dtype', 'fp8')),
            ('vae file swapped', lambda w: w['3']['inputs'].__setitem__('vae_name', 'flux-vae.safetensors')),
            # Input set drift.
            ('extra input', lambda w: w['4']['inputs'].__setitem__('extra', 1)),
            ('missing input', lambda w: w['13']['inputs'].pop('filename_prefix')),
            ('prefix changed', lambda w: w['13']['inputs'].__setitem__('filename_prefix', '../../escape')),
            ('prefix dropped to a bare folder', lambda w: w['13']['inputs'].__setitem__('filename_prefix', 'PMI')),
            # Link redirection and the zeroed-negative path.
            ('negative path redirected directly', lambda w: w['6']['inputs'].__setitem__('negative', ['4', 0])),
            ('zero-out rewired', lambda w: w['5']['inputs'].__setitem__('conditioning', ['2', 0])),
            ('positive from zero-out', lambda w: w['6']['inputs'].__setitem__('positive', ['5', 0])),
            ('guider from noise', lambda w: w['11']['inputs'].__setitem__('guider', ['7', 0])),
            ('model from guider', lambda w: w['6']['inputs'].__setitem__('model', ['2', 0])),
            ('vae from clip', lambda w: w['12']['inputs'].__setitem__('vae', ['2', 0])),
            ('samples from latent', lambda w: w['12']['inputs'].__setitem__('samples', ['10', 0])),
            ('decode from sampler select', lambda w: w['12']['inputs'].__setitem__('samples', ['8', 0])),
            ('foreign slot', lambda w: w['4']['inputs'].__setitem__('clip', ['2', 1])),
            ('unlinked positive text', lambda w: w['4']['inputs'].__setitem__('clip', '2:0')),
            ('link to a missing node', lambda w: w['6']['inputs'].__setitem__('model', ['99', 0])),
            # Dimensions, batch, steps, cfg and sampler.
            ('latent width', lambda w: w['10']['inputs'].__setitem__('width', 1024)),
            ('scheduler height', lambda w: w['9']['inputs'].__setitem__('height', 768)),
            ('latent height mismatch', lambda w: w['10']['inputs'].__setitem__('height', 256)),
            ('scheduler width mismatch', lambda w: w['9']['inputs'].__setitem__('width', 1024)),
            ('batch size', lambda w: w['10']['inputs'].__setitem__('batch_size', 4)),
            ('steps', lambda w: w['9']['inputs'].__setitem__('steps', 20)),
            ('cfg', lambda w: w['6']['inputs'].__setitem__('cfg', 7.0)),
            ('sampler name', lambda w: w['8']['inputs'].__setitem__('sampler_name', 'ddim')),
            # Seed and prompt contract.
            ('seed control randomized', lambda w: w['7']['inputs'].__setitem__(
                'control_after_generate', 'randomize')),
            ('negative seed', lambda w: w['7']['inputs'].__setitem__('noise_seed', -1)),
            ('seed above 2**32', lambda w: w['7']['inputs'].__setitem__('noise_seed', 2 ** 32)),
            ('boolean seed', lambda w: w['7']['inputs'].__setitem__('noise_seed', True)),
            ('string seed', lambda w: w['7']['inputs'].__setitem__('noise_seed', '7')),
            ('float seed', lambda w: w['7']['inputs'].__setitem__('noise_seed', 7.5)),
            ('prompt changed', lambda w: w['4']['inputs'].__setitem__('text', 'a different prompt')),
            ('prompt emptied', lambda w: w['4']['inputs'].__setitem__('text', '')),
        )
        for why, mutate in tampered:
            with self.subTest(why=why):
                broken = json.loads(json.dumps(workflow))
                mutate(broken)
                with self.assertRaises(ValueError):
                    comfy.validate_workflow(broken, prompt)
        # The exact pinned graph is the only positive case.
        self.assertEqual(comfy.validate_workflow(workflow, prompt), workflow)
        self.assertEqual(comfy.build_workflow(prompt, seed=0)['7']['inputs']['noise_seed'], 0)
        self.assertEqual(comfy.build_workflow(prompt, seed=2 ** 32 - 1)['7']['inputs']['noise_seed'],
                         2 ** 32 - 1)
        for bad in [-1, 2 ** 32, True, '7', 7.5, [7]]:
            with self.subTest(seed=repr(bad)), self.assertRaises(ValueError):
                comfy.build_workflow(prompt, seed=bad)
        for broken in ['', '\0']:
            with self.subTest(text=repr(broken)), self.assertRaises(ValueError):
                broken_workflow = json.loads(json.dumps(workflow))
                broken_workflow['4']['inputs']['text'] = broken
                comfy.validate_workflow(broken_workflow)
        with self.assertRaises(ValueError):
            comfy.validate_workflow(workflow, 'a different prompt')
        for bad in [None, [], 'graph', {'4': {}}]:
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                comfy.validate_workflow(bad, prompt)

    def test_image_artifact_receipt_needs_real_png_or_jpeg_bytes(self):
        import comfy_backend as comfy
        png = png_bytes(512, 512)
        receipt = comfy.validate_artifact(png)
        self.assertEqual(receipt['format'], 'png')
        self.assertEqual((receipt['width'], receipt['height'], receipt['bytes']), (512, 512, len(png)))
        self.assertEqual(receipt['sha256'], hashlib.sha256(png).hexdigest())
        self.assertEqual(comfy.validate_artifact(jpeg_bytes(512, 512))['format'], 'jpeg')
        # Same bytes must produce the same receipt; no wall-clock or random input.
        self.assertEqual(comfy.validate_artifact(png), receipt)
        for bad, why in [(b'', 'empty'), (b'GIF89a' + png[6:], 'other format'),
                         (png[:-12], 'no IEND'), (png[:40], 'truncated'),
                         (png + b'trailing', 'trailing bytes'),
                         (png_bytes(256, 256), 'dimensions differ'),
                         (png_bytes(8192, 8192), 'above envelope'),
                         (b'\x89PNG\r\n\x1a\n' + b'\x00' * 60, 'garbage chunks')]:
            with self.subTest(why=why), self.assertRaises(ValueError):
                comfy.validate_artifact(bad)
        corrupt = bytearray(png_bytes(64, 64))
        corrupt[30] ^= 0xff  # inside the IHDR payload, so its CRC no longer matches
        with self.assertRaises(ValueError):
            comfy.validate_artifact(bytes(corrupt))
        self.assertEqual(comfy.validate_artifact(png_bytes(64, 64), expected=(64, 64))['width'], 64)
        with self.assertRaises(ValueError):
            comfy.validate_artifact(png_bytes(64, 64), expected=(512, 512))
        with self.assertRaises(ValueError):
            comfy.validate_artifact(b'\x89PNG\r\n\x1a\n' + b'\x00' * (comfy.MAX_ARTIFACT_BYTES + 1))

    def test_image_stage_submits_polls_validates_and_unloads_with_a_receipt(self):
        import comfy_backend as comfy
        prompt = 'A quiet harbour at dawn'
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); stage = base / 'image'; stage.mkdir()
            client = FakeComfy(history=[{}, {}, SUCCESS_HISTORY])
            slept = []
            receipt = comfy.run_image(client, prompt, stage, time.monotonic() + 60,
                                      sleeper=slept.append)
            self.assertEqual(receipt['schema'], 'PMI_COMFY_IMAGE_RECEIPT_V1')
            self.assertEqual(receipt['status'], 'ImageAndReleasePass')
            self.assertTrue(receipt['release_proven'])
            self.assertTrue(receipt['release']['unload_models'])
            self.assertEqual(receipt['polls'], 3)
            self.assertEqual(slept, [comfy.POLL_INTERVAL_SECONDS, comfy.POLL_INTERVAL_SECONDS])
            self.assertEqual(receipt['prompt_id'], '12345678-1234-5678-1234-567812345678')
            # Submitted graph is exactly the pinned distilled recipe with only text and seed injected.
            method, path, payload = client.calls[1]
            self.assertEqual((method, path), ('POST', '/prompt'))
            self.assertEqual(payload['prompt'], comfy.build_workflow(prompt))
            self.assertEqual(list(payload['prompt']), list(comfy.GRAPH_ORDER))
            self.assertEqual(payload['prompt']['4']['inputs']['text'], prompt)
            self.assertEqual(payload['prompt']['7']['inputs']['noise_seed'], comfy.prompt_seed(prompt))
            self.assertEqual(payload['client_id'], receipt['client_id'])
            self.assertEqual([call[1] for call in client.calls],
                             ['/system_stats', '/prompt', '/history/12345678-1234-5678-1234-567812345678',
                              '/history/12345678-1234-5678-1234-567812345678',
                              '/history/12345678-1234-5678-1234-567812345678', '/view', '/free', '/system_stats'])
            self.assertEqual(client.calls[-2][0], 'POST')
            self.assertTrue(receipt['release']['request_acknowledged'])
            self.assertGreaterEqual(receipt['release']['postflight']['vram_free_bytes'],
                                    receipt['release']['baseline']['vram_free_bytes'] - comfy.RELEASE_VRAM_TOLERANCE_BYTES)
            # Output history is keyed by the single selected SaveImage node id.
            self.assertEqual(sorted(SUCCESS_HISTORY['outputs']), [comfy.SAVE_NODE])
            self.assertEqual(receipt['reference'],
                             {'filename': 'answer_00001_.png', 'subfolder': 'PMI', 'type': 'output'})
            self.assertEqual(len(SUCCESS_HISTORY['outputs'][comfy.SAVE_NODE]['images']), 1)
            self.assertTrue(SUCCESS_HISTORY['outputs'][comfy.SAVE_NODE]['images'][0]['filename']
                            .startswith('answer_'))
            # The synthetic PNG dimensions match the pinned recipe bound.
            self.assertEqual((receipt['artifact']['width'], receipt['artifact']['height']),
                             comfy.IMAGE_BOUNDS)
            self.assertEqual((receipt['artifact']['width'], receipt['artifact']['height']),
                             (comfy.RECIPE_NODES['10']['inputs']['width'],
                              comfy.RECIPE_NODES['9']['inputs']['height']))
            # Receipt carries the graph, prompt and seed hashes.
            self.assertEqual(receipt['prompt_sha256'], hashlib.sha256(prompt.encode()).hexdigest())
            self.assertEqual(receipt['seed'], comfy.prompt_seed(prompt))
            self.assertEqual(receipt['seed_control'], 'fixed')
            self.assertEqual(receipt['workflow_sha256'],
                             hashlib.sha256(json.dumps(payload['prompt'], sort_keys=True).encode()).hexdigest())
            self.assertEqual(receipt['recipe_sha256'],
                             hashlib.sha256(json.dumps(comfy.RECIPE_NODES, sort_keys=True).encode()).hexdigest())
            self.assertEqual(receipt['source_workflow_sha256'], comfy.SOURCE_WORKFLOW_SHA256)
            self.assertEqual(receipt['selected_model'], 'FLUX.2 Klein 4B distilled')
            self.assertEqual(receipt['selected_harness'], 'ComfyUI')
            # Saved artifact matches the validated receipt byte for byte.
            saved = Path(receipt['artifact_file'])
            self.assertEqual(saved.name, 'answer.png')
            self.assertEqual(saved.read_bytes(), client.artifact)
            self.assertEqual(hashlib.sha256(saved.read_bytes()).hexdigest(), receipt['artifact']['sha256'])
            for name in ['workflow.json', 'submission.json', 'history.json', 'image-receipt.json']:
                self.assertTrue((stage / name).is_file(), name)
            self.assertFalse((stage / 'image-cancel.json').exists())
            stored = json.loads((stage / 'image-receipt.json').read_text())
            self.assertEqual(stored['artifact']['sha256'], receipt['artifact']['sha256'])
            self.assertEqual(stored['prompt_sha256'], hashlib.sha256(prompt.encode()).hexdigest())
            self.assertFalse(stored['automatic_retry'])
            # The saved workflow.json is the submitted graph, not a UI subgraph record.
            self.assertEqual(json.loads((stage / 'workflow.json').read_text()), payload['prompt'])

    def test_image_stage_failures_cancel_without_retry_and_never_claim_a_pass(self):
        import comfy_backend as comfy
        cases = {
            # Refused at submission: no prompt_id exists, so no interrupt is attempted.
            'refused graph': (dict(node_errors={'1': {'errors': [['required_input_missing', 'ckpt_name']]}}), False),
            'missing prompt id': (dict(prompt_id=''), False),
            # Accepted and queued: the prompt must also be deleted from the queue.
            'node failure': (dict(history=[FAILED_HISTORY]), True),
            'no artifact': (dict(history=[{'status': {'status_str': 'success', 'completed': True},
                                           'outputs': {}}]), True),
            'not an image': (dict(artifact=b'this is not an image'), True),
            'empty body': (dict(artifact=b''), True),
            'wrong size': (dict(artifact=png_bytes(256, 256)), True),
        }
        for why, (options, queued) in cases.items():
            with self.subTest(why=why), tempfile.TemporaryDirectory() as temporary:
                stage = Path(temporary) / 'image'; stage.mkdir()
                client = FakeComfy(**options)
                with self.assertRaises((comfy.ComfyFailure, ValueError)):
                    comfy.run_image(client, 'A quiet harbour at dawn', stage, time.monotonic() + 60,
                                    sleeper=lambda seconds: None)
                self.assertFalse((stage / 'answer.png').exists())
                self.assertFalse((stage / 'image-receipt.json').exists())
                stop = json.loads((stage / 'image-cancel.json').read_text())
                self.assertFalse(stop['automatic_retry'])
                self.assertIn(stop['error_type'], ('ComfyFailure', 'ValueError'))
                # Cancellation is targeted to our own valid prompt and never global.
                self.assertNotIn(('POST', '/interrupt', None), client.calls)
                targeted = ('POST', '/interrupt', {'prompt_id': '12345678-1234-5678-1234-567812345678'}) in client.calls
                self.assertEqual(targeted, queued)
                deleted = ('POST', '/queue', {'delete': ['12345678-1234-5678-1234-567812345678']}) in client.calls
                self.assertEqual(deleted, queued)
                self.assertEqual(stop['prompt_id'] is not None, queued)

    def test_image_stage_deadline_and_owner_cancel_stop_promptly(self):
        import comfy_backend as comfy
        ticks = {'n': 0}

        def clock():
            ticks['n'] += 1
            return 0 if ticks['n'] < 3 else 10 ** 9

        def owner():
            raise comfy.ComfyCancelled('owner signal cancelled the image stage; stop without retry')

        for why, check, expected in [('deadline', None, 'TimeoutError'),
                                     ('owner signal', owner, 'ComfyCancelled')]:
            with self.subTest(why=why), tempfile.TemporaryDirectory() as temporary:
                ticks['n'] = 0
                stage = Path(temporary) / 'image'; stage.mkdir()
                client = FakeComfy(history=[{}] * 6)
                with self.assertRaises((TimeoutError, comfy.ComfyCancelled)):
                    comfy.run_image(client, 'A quiet harbour at dawn', stage, 5, check=check,
                                    clock=clock, sleeper=lambda seconds: None)
                stop = json.loads((stage / 'image-cancel.json').read_text())
                self.assertFalse(stop['automatic_retry'])
                self.assertEqual(stop['error_type'], expected)
                self.assertEqual(stop['prompt_id'], '12345678-1234-5678-1234-567812345678')
                self.assertIn(('POST', '/interrupt', {'prompt_id': stop['prompt_id']}), client.calls)
                self.assertEqual(client.calls[-1][:2], ('POST', '/queue'))
                self.assertFalse((stage / 'image-receipt.json').exists())

    def test_image_unload_must_be_proven_before_the_stage_is_a_pass(self):
        import comfy_backend as comfy
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / 'image'; stage.mkdir()
            client = FakeComfy(vram_samples=[7 * 1024**3, 5 * 1024**3])
            with self.assertRaisesRegex(RuntimeError, 'unload unproven'):
                comfy.run_image(client, 'A quiet harbour at dawn', stage, time.monotonic() + 60,
                                    clock=iter(range(100)).__next__, sleeper=lambda seconds: None)
            receipt = json.loads((stage / 'image-receipt.json').read_text())
            self.assertFalse(receipt['release_proven'])
            self.assertEqual(receipt['status'], 'StoppedNoRetry')
            self.assertTrue((stage / 'answer.png').is_file())
            stop = json.loads((stage / 'image-cancel.json').read_text())
            self.assertIn('unload unproven', stop['reason'])
            self.assertIn(('POST', '/interrupt', {'prompt_id': '12345678-1234-5678-1234-567812345678'}), client.calls)
        # The in-memory fake shows that an async release acknowledgment is not a pass.
        with self.assertRaises(comfy.ComfyFailure):
            comfy.release(FakeComfy(vram_samples=[5 * 1024**3]), {'vram_free_bytes': 7 * 1024**3}, 100,
                          clock=iter(range(100)).__next__, sleeper=lambda seconds: None)

    def test_image_transport_refuses_redirects_remote_bodies_and_proxy_use(self):
        import comfy_backend as comfy
        with patch('urllib.request.build_opener') as opener:
            opener.side_effect = AssertionError('socket attempted')
            with self.assertRaises(AssertionError):
                comfy.LocalTransport('http://127.0.0.1:8188').call_json('/prompt', {}, 'POST')
        for bad in ['prompt', '//127.0.0.1:8188/prompt', 'http://127.0.0.1:8188/prompt', '/prompt\nx']:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                comfy.LocalTransport().call(bad)
        self.assertIsNone(comfy._NoRedirect().redirect_request(None, None, 302, '', {}, 'http://x'))
        transport = comfy.LocalTransport()
        self.assertEqual(transport.endpoint, 'http://127.0.0.1:8188')
        with patch('urllib.request.build_opener', side_effect=AssertionError('built')) as built:
            self.assertIsNotNone(transport.endpoint)  # construction never builds an opener
            built.assert_not_called()

    def test_image_backend_and_speak_image_remain_platform_held(self):
        import comfy_backend as comfy
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'absent'
            with patch('subprocess.Popen', side_effect=AssertionError('process attempted')), \
                 patch('socket.socket', side_effect=AssertionError('socket attempted')):
                with self.assertRaises(PlatformHeld):
                    comfy.ComfyBackend().run_stage('hello', output, 0)
                with self.assertRaises(PlatformHeld):
                    factory.main(['speak-image', '--out', str(output)])
            self.assertFalse(output.exists())
            self.assertEqual(list(output.iterdir()) if output.exists() else [], [])

    def test_image_plan_is_offline_and_reports_held_with_separate_model_assets(self):
        import comfy_backend as comfy
        with tempfile.TemporaryDirectory() as temporary, \
             patch('socket.socket', side_effect=AssertionError('socket attempted')), \
             patch('urllib.request.build_opener', side_effect=AssertionError('opener built')) as opener:
            base = Path(temporary)
            prompt = base / 'prompt.txt'
            prompt.write_text('A quiet harbour at dawn')
            output = base / 'plan.json'
            self.assertEqual(factory.main(['image-plan', '--prompt-file', str(prompt),
                                           '--models', 'bonsai-stock', '--out', str(output),
                                           '--comfy-endpoint', 'http://127.0.0.1:8188']), 0)
            opener.assert_not_called()
            plan = json.loads(output.read_text())
            self.assertEqual(plan['status'], 'HELD')
            self.assertFalse(plan['actual_inference'])
            image = plan['image']
            self.assertEqual(image['schema'], 'PMI_COMFY_IMAGE_RECIPE_V2')
            self.assertEqual(image['status'], 'PlatformHeld')
            self.assertFalse(image['actual_inference'])
            self.assertTrue(image['image_execution_held'])
            self.assertFalse(image['endpoint_reachable'])
            # ComfyUI plus distilled Klein stay selected; the plan never switches backend or model.
            self.assertEqual(image['selected_harness'], 'ComfyUI')
            self.assertEqual(image['selected_model'], 'FLUX.2 Klein 4B distilled')
            self.assertTrue(image['workflow_compatible'])
            self.assertFalse(image['host_runtime_verified'])
            self.assertEqual(image['qualification'], 'Unresolved')
            self.assertNotIn('stable-diffusion.cpp', json.dumps(image))
            # The pinned source and its digest are reported, not recomputed from the network.
            self.assertEqual(image['source']['sha256'],
                             '8bb879856e54765b6c3ed219181e855981d8d10cfd19acff836603d9b56e838f')
            self.assertEqual(image['source']['bytes'], 69872)
            self.assertEqual(image['source']['selected_subgraph'],
                             {'id': 'a67caa28-5f85-4917-8396-36004960dd30',
                              'name': 'Text to Image (Flux.2 Klein 4B Distilled)'})
            # The receipt-backed files are present and match recorded sizes. The
            # plan carries their SHA values as receipt identities without rehashing.
            self.assertTrue(image['model_assets_present'])
            self.assertEqual(sorted(image['model_roles']), ['diffusion_model', 'text_encoder', 'vae'])
            expected_assets = {
                'text_encoder': (8044982048, '6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a'),
                'diffusion_model': (7751105712, 'ec3d4e733a771f61c052fb4856c48b336c55eaf2c65487c2a1faeb9bbda7a343'),
                'vae': (336211292, '868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3'),
            }
            for role, spec in image['model_roles'].items():
                with self.subTest(role=role):
                    self.assertTrue(spec['present'])
                    self.assertEqual(spec['installation'], 'present-unverified')
                    self.assertTrue(spec['size_matches_receipt'])
                    self.assertEqual(spec['bytes'], expected_assets[role][0])
                    self.assertEqual(spec['receipt_bytes'], expected_assets[role][0])
                    self.assertEqual(spec['sha256'], expected_assets[role][1])
                    self.assertFalse(spec['sha256_verified_here'])
            self.assertEqual(image['model_roles']['diffusion_model']['file'],
                             'flux-2-klein-4b.safetensors')
            self.assertEqual(image['model_roles']['diffusion_model']['unselected_base_file'],
                             'flux-2-klein-base-4b.safetensors')
            self.assertEqual(image['model_roles']['text_encoder']['file'], 'qwen_3_4b.safetensors')
            self.assertEqual(image['model_roles']['text_encoder']['type'], 'flux2')
            self.assertEqual(image['model_roles']['vae']['file'], 'flux2-vae.safetensors')
            # The graph in the plan is the pinned distilled graph, not the old checkpoint recipe.
            self.assertEqual(list(image['graph']['nodes']), list(comfy.GRAPH_ORDER))
            self.assertEqual(image['graph']['nodes']['1']['class_type'], 'UNETLoader')
            self.assertEqual(image['graph']['output_nodes'], [comfy.SAVE_NODE])
            self.assertEqual(len(image['graph']['edges']), 13)
            self.assertEqual(image['conditioning']['negative'], 'ConditioningZeroOut')
            self.assertIsNone(image['conditioning']['negative_prompt_text'])
            self.assertEqual(image['stages'], ['text:bonsai-stock', 'speech:piper-cpu', 'image:comfyui-local'])
            self.assertEqual(image['seed'], comfy.prompt_seed('A quiet harbour at dawn'))
            self.assertEqual(image['prompt_sha256'],
                             hashlib.sha256('A quiet harbour at dawn'.encode()).hexdigest())
            self.assertEqual(image['workflow_sha256'],
                             hashlib.sha256(json.dumps(comfy.build_workflow('A quiet harbour at dawn'),
                                                       sort_keys=True).encode()).hexdigest())
            self.assertEqual(image['artifact']['bounds'], [512, 512])
            self.assertEqual(plan['unavailable_modalities'], ['video', 'audio'])
            self.assertIn('image', plan)
            self.assertNotIn('speech', plan)
            self.assertTrue(any('reconfirm the installed model-file hashes' in item
                                for item in image['remaining']))
            self.assertTrue(any('PyTorch/ROCm' in item for item in image['remaining']))
            with self.assertRaises(ValueError):
                factory.main(['image-plan', '--prompt-file', str(prompt), '--out', str(output),
                              '--comfy-endpoint', 'https://example.com'])
            with self.assertRaises(ValueError):
                factory.main(['image-plan', '--prompt-file', str(prompt), '--out', str(base / 'x.json'),
                              '--models', 'qwen3-stock,qwen25-modified'])
            self.assertFalse((base / 'x.json').exists())

    def test_model_intent_names_heretic_with_an_unknown_hash_and_an_absent_artifact(self):
        import model_intent as intent
        self.assertTrue(intent.validate())
        document = intent.intent()
        self.assertEqual(document['schema'], 'PMI_MODEL_INTENT_V1')
        self.assertEqual(document['modality_order'], ['text', 'speech', 'image', 'video'])
        self.assertFalse(document['actual_inference'])
        self.assertFalse(document['assets_downloaded'])
        self.assertFalse(document['weights_opened'])
        self.assertFalse(document['models_run'])
        roles = {role['role_id']: role for role in document['modalities']['text']['roles']}
        heretic = roles['heretic-ptq1_0']
        self.assertEqual(heretic['selection'], 'intended')
        self.assertTrue(heretic['primary'])
        self.assertEqual(heretic['repository'],
                         'OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-v2-GGUF')
        self.assertEqual(heretic['revision'], '6db27547a8f25667cdb43a28d15cc5d33937b7dc')
        self.assertEqual(heretic['revision_kind'], 'observed-upstream')
        self.assertFalse(heretic['revision_verified_here'])
        self.assertEqual(heretic['quantization'], 'PTQ1_0')
        # Explicitly unknown and explicitly unavailable; never a placeholder value.
        self.assertIsNone(heretic['sha256'])
        self.assertFalse(heretic['sha256_verified_here'])
        self.assertIsNone(heretic['bytes'])
        self.assertEqual(heretic['installation'], 'absent')
        self.assertEqual(heretic['runtime_qualification'], 'unqualified')
        self.assertIsNone(heretic['qualification_scope'])
        self.assertFalse(heretic['substitutes_intended'])
        # Stock Bonsai stays a preserved baseline and never stands in for Heretic.
        stock = roles['bonsai-stock']
        self.assertEqual(stock['selection'], 'preserved-baseline')
        self.assertEqual(stock['repository'], 'prism-ml/Ternary-Bonsai-2-27B-gguf')
        self.assertEqual(stock['revision'], 'b072e1d3b35a0a630cece372c2127528e0994386')
        self.assertEqual(stock['sha256'],
                         '53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3')
        self.assertEqual(stock['installation'], 'present-unverified')
        self.assertEqual(stock['runtime_qualification'], 'scoped-run-proven')
        self.assertNotEqual(stock['repository'], heretic['repository'])
        self.assertNotEqual(stock['revision'], heretic['revision'])
        # The executable default and the preserved receipts are untouched.
        self.assertEqual(document['default_text_model_id'], 'bonsai-stock')
        self.assertEqual(factory.DEFAULT_MODEL, 'bonsai-stock')
        self.assertEqual(factory.MODELS['bonsai-stock']['sha256'], stock['sha256'])
        self.assertIn('not the Heretic derivative', factory.MODELS['bonsai-stock']['qualification'])
        # The stock catalog names no Heretic artifact, revision or hash.
        serialized = json.dumps(factory.MODELS)
        self.assertNotIn('6db27547', serialized)
        self.assertNotIn('Uncensored-Heretic', serialized)
        self.assertNotIn('heretic-ptq1_0', serialized)

    def test_model_intent_offers_no_qwen_substitution_and_leaves_video_unavailable(self):
        import model_intent as intent
        document = intent.intent()
        text = {role['role_id']: role for role in document['modalities']['text']['roles']}
        for role_id in ('qwen3-stock', 'qwen25-modified'):
            with self.subTest(role_id=role_id):
                role = text[role_id]
                self.assertEqual(role['selection'], 'not-selected')
                self.assertFalse(role['primary'])
                self.assertFalse(role['substitutes_intended'])
                self.assertEqual(role['runtime_qualification'], 'unqualified')
                self.assertIsNone(role['qualification_scope'])
                self.assertIsNone(role['sha256'])
        # The intended role is not executable here: the absent artifact cannot be selected.
        with self.assertRaises(ValueError):
            factory.select_models(['heretic-ptq1_0'])
        self.assertEqual(sorted(factory.MODELS), ['bonsai-stock', 'qwen25-modified', 'qwen3-stock'])
        self.assertIsNone(factory.MODELS['qwen25-modified']['sha256'])
        video = document['modalities']['video']
        self.assertEqual(video['roles'], [])
        self.assertEqual(video['declared_roles'], [])
        self.assertEqual(video['stages'], [])
        self.assertFalse(video['selected'])
        self.assertFalse(video['resolved'])
        self.assertIsNone(video['harness'])
        self.assertFalse(video['runtime_qualified'])
        self.assertIsNone(video['source_compatibility_proven'])
        self.assertEqual(document['unavailable_modalities'], ['video'])
        self.assertEqual(intent.ROLES[intent.Modality.VIDEO], ())

    def test_model_intent_image_roles_separate_installation_from_runtime_qualification(self):
        import comfy_backend as comfy
        import model_intent as intent
        document = intent.intent()
        image = document['modalities']['image']
        # The selected route is unchanged.
        self.assertEqual(image['harness'], 'ComfyUI')
        self.assertTrue(image['selected'])
        self.assertFalse(image['resolved'])
        self.assertEqual(comfy.SELECTED_HARNESS, image['harness'])
        self.assertEqual(comfy.SELECTED_MODEL, 'FLUX.2 Klein 4B distilled')
        # Source graph compatibility is a source property, never a host claim.
        self.assertTrue(image['source_compatibility_proven'])
        self.assertFalse(image['runtime_qualified'])
        self.assertEqual(image['qualification'], 'Unresolved')
        self.assertTrue(comfy.recipe()['workflow_compatible'])
        self.assertFalse(comfy.recipe()['host_runtime_verified'])
        self.assertTrue(comfy.recipe()['model_assets_present'])
        roles = {role['role_id']: role for role in image['roles']}
        self.assertEqual(sorted(roles), ['diffusion_model', 'text_encoder', 'vae'])
        for role_id, spec in sorted(comfy.model_roles().items()):
            with self.subTest(role_id=role_id):
                role = roles[role_id]
                self.assertEqual(role['loader'], spec['class_type'])
                self.assertEqual(role['file_name'], spec['file'])
                self.assertEqual(role['folder'], spec['folder'])
                self.assertEqual(role['harness'], 'ComfyUI')
                # Installed files carry receipt identities, while runtime qualification
                # remains separate and the plan does not claim a fresh hash check.
                self.assertEqual(role['installation'], 'present-unverified')
                self.assertEqual(role['runtime_qualification'], 'unqualified')
                self.assertIsNone(role['qualification_scope'])
                self.assertEqual(role['sha256'], spec['sha256'])
                self.assertEqual(role['bytes'], spec['bytes'])
                self.assertEqual(role['revision'], spec['revision'])
                self.assertFalse(role['sha256_verified_here'])
                self.assertFalse(role['substitutes_intended'])
        self.assertTrue(roles['diffusion_model']['primary'])
        self.assertFalse(roles['text_encoder']['primary'])
        self.assertFalse(roles['vae']['primary'])
        # The pinned workflow snapshot stays exactly as committed.
        self.assertEqual(intent.IMAGE_SOURCE_REVISION, comfy.SOURCE_UPSTREAM_REVISION)
        self.assertEqual(intent.IMAGE_SOURCE_SHA256, comfy.SOURCE_WORKFLOW_SHA256)
        self.assertEqual(intent.IMAGE_SOURCE_BYTES, comfy.SOURCE_WORKFLOW_BYTES)
        # No declared role or modality anywhere hardcodes a verified status. Preserved
        # baselines keep their installed identity and their scoped evidence; nothing
        # claims a verified hash, a verified revision, a qualified runtime or a substitution.
        serialized = json.dumps(document)
        for forbidden in ('"runtime_qualified": true', '"sha256_verified_here": true',
                          '"revision_verified_here": true', '"substitutes_intended": true'):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, serialized)

    def test_model_intent_pins_kokoro_and_preserves_the_piper_audio_baseline(self):
        import model_intent as intent
        document = intent.intent()
        speech = {role['role_id']: role for role in document['modalities']['speech']['roles']}
        kokoro = speech['kokoro-82m']
        self.assertEqual(kokoro['selection'], 'intended')
        self.assertTrue(kokoro['primary'])
        self.assertEqual(kokoro['repository'], 'hexgrad/Kokoro-82M')
        self.assertEqual(kokoro['revision'], 'f3ff3571791e39611d31c381e3a41a3af07b4987')
        self.assertEqual(kokoro['sha256'],
                         '496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4')
        self.assertFalse(kokoro['sha256_verified_here'])
        self.assertIsNone(kokoro['bytes'])
        self.assertEqual(kokoro['installation'], 'absent')
        self.assertEqual(kokoro['runtime_qualification'], 'unqualified')
        self.assertFalse(kokoro['substitutes_intended'])
        # The declared audio contract: two configurable voices, 24 kHz, CPU only.
        self.assertEqual(intent.KOKORO_SAMPLE_RATE, 24000)
        self.assertEqual(intent.KOKORO_WEIGHT_FILE, 'kokoro-v1_0.pth')
        self.assertEqual(intent.KOKORO_CONFIG_FILE, 'config.json')
        self.assertEqual(intent.KOKORO_RUNTIME_VERSION, '0.9.4')
        self.assertEqual(intent.KOKORO_PYTHON_SPEC, '>=3.10,<3.13')
        self.assertEqual(intent.KOKORO_LICENSE, 'Apache-2.0')
        self.assertEqual(intent.DEFAULT_SPEECH_MODEL_ID, 'piper-cpu')
        self.assertEqual(intent.SPEECH_MODEL_IDS, ('piper-cpu', 'kokoro-82m'))
        self.assertEqual(intent.KOKORO_DEFAULT_VOICE, 'af_heart')
        self.assertEqual(intent.KOKORO_VOICES,
                         {'af_heart': ('en-US', 'English'),
                          'pf_dora': ('pt-BR', 'Brazilian Portuguese')})
        # Piper is preserved, not replaced, and never takes the intended role's place.
        piper = speech['piper-cpu']
        self.assertEqual(piper['selection'], 'preserved-baseline')
        self.assertEqual(piper['runtime_qualification'], 'scoped-run-proven')
        self.assertTrue(piper['qualification_scope'])
        self.assertFalse(piper['primary'])
        self.assertEqual(piper['installation'], 'present-unverified')
        self.assertEqual([role['role_id'] for role in document['modalities']['speech']['roles']
                          if role['selection'] == 'intended'], ['kokoro-82m'])
        self.assertEqual(intent.PIPER_SAMPLE_RATE, 22050)
        self.assertEqual(intent.PIPER_VOICE_ID, 'en_US-ljspeech-medium')
        self.assertEqual(document['modalities']['speech']['stages'], ['speech:piper-cpu'])
        # The declaration is deterministic: no clock, no random, no filesystem.
        self.assertEqual(json.dumps(intent.intent(), sort_keys=True),
                         json.dumps(intent.intent(), sort_keys=True))
        self.assertEqual(len(intent.RULES), len(set(intent.RULES)))
        # Heretic is declared with an identity nobody here has opened a file for.
        declared = {role.role_id: role for role in intent.DECLARED_IDENTITY_ROLES}
        self.assertIn('heretic-ptq1_0', declared)
        self.assertIsNone(declared['heretic-ptq1_0'].bytes)
        self.assertIsNone(declared['heretic-ptq1_0'].sha256)
        self.assertIn('kokoro-82m', declared)
        self.assertIsNone(declared['kokoro-82m'].bytes)

    def test_image_requires_the_verified_text_and_speech_sequence(self):
        import comfy_backend as comfy
        from test_speech import SpeechFixture
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            text = ContractBackend()
            # A real image backend is refused before any directory, exactly like text/speech.
            with self.assertRaises(PlatformHeld):
                factory._run_sequence('Prompt', ['qwen3-stock'], base / 'held', base / 'held-lease',
                                      text, SpeechFixture(text), comfy.ComfyBackend())
            self.assertFalse((base / 'held').exists())
            self.assertEqual(text.events, [])
            with self.assertRaisesRegex(ValueError, 'image stage requires'):
                factory._run_sequence('Prompt', ['qwen3-stock'], base / 'job', base / 'lease',
                                      ContractBackend(), None, FakeComfy())
            with self.assertRaisesRegex(ValueError, 'exactly one selected text model'):
                factory._run_sequence('Prompt', ['qwen3-stock', 'qwen25-modified'], base / 'other',
                                      base / 'lease2', ContractBackend(), _SpeechOnly(), FakeComfy())
            self.assertFalse((base / 'job').exists())
            self.assertFalse((base / 'other').exists())


if __name__ == '__main__':
    unittest.main()
