"""Model-free fixtures only: no real metadata, tokenizer, asset or process probes."""
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch

import content_factory as factory
import llama_backend as backend
import model_profiles as profiles
import native_worker
import prompt_fit
from research.harness import DemoRealAgent, RealAgent, ReleasedStageError, digest


WELCOME = ('Write one friendly sentence welcoming a listener to a local content factory. '
           'Use at most twenty words, plain English, no headings.')


def fixture_fit(prompt='fixture', role='generation', rendered_bytes=200):
    settings = profiles.profile_settings(profiles.PROFILE_ID, profiles.MODEL_ID, role)
    return {'policy': profiles.FIT_POLICY, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'template_sha256': profiles.TEMPLATE_SHA256, 'formatted_utf8_bytes': rendered_bytes,
            'input_token_upper_bound': rendered_bytes + 2,
            'output_tokens_reserved': settings['output_tokens'], 'context': 4096,
            'metadata_bytes_read': 100, 'layers': 65, 'exact_token_count': None,
            'tensor_bytes_read': 0, **{key: settings[key] for key in
            ('profile_id', 'role', 'profile_sha256', 'context_margin_tokens')}}


def fixture_capture(prompt='fixture', role='generation'):
    settings = profiles.profile_settings(profiles.PROFILE_ID, profiles.MODEL_ID, role)
    fit = fixture_fit(prompt, role)
    return {'returncode': 0, 'raw_stdout': b'fixture answer [end of text]',
            'owned_cleanup_empty': True, 'postflight_passed': True, **settings,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'fit': fit, 'fit_sha256': digest(fit)}


class ProfileTests(unittest.TestCase):
    def test_closed_roles_and_model(self):
        for role, output in profiles.ROLE_OUTPUT_TOKENS.items():
            result = profiles.profile_settings(profiles.PROFILE_ID, profiles.MODEL_ID, role)
            self.assertEqual((result['context_tokens'], result['output_tokens']), (4096, output))
        for value in [('unknown', profiles.MODEL_ID, 'generation'),
                      (profiles.PROFILE_ID, 'qwen3-stock', 'generation'),
                      (profiles.PROFILE_ID, profiles.MODEL_ID, 'semantic-proximity'),
                      (profiles.PROFILE_ID, profiles.MODEL_ID, None)]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                profiles.profile_settings(*value)

    def test_hash_and_descriptor_are_stable_copies(self):
        descriptor = profiles.profile_descriptor()
        expected = profiles.canonical_sha256(descriptor)
        descriptor['role_output_tokens']['generation'] = 1
        self.assertEqual(profiles.profile_settings(profiles.PROFILE_ID, profiles.MODEL_ID,
                                                  'generation')['profile_sha256'], expected)
        self.assertEqual(digest({'unicode': '\u2603'}), profiles.canonical_sha256({'unicode': '\u2603'}))

    def test_request_exact_keys_and_hash(self):
        request = profiles.model_request(profiles.MODEL_ID, 'fixture', profiles.PROFILE_ID, 'generation')
        self.assertEqual(profiles.validate_model_request(request)['output_tokens'], 512)
        for mutation in ({**request, 'profile_sha256': '0' * 64}, {**request, 'output': 1},
                         {k: v for k, v in request.items() if k != 'profile_id'}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                profiles.validate_model_request(mutation)

    def test_legacy_exact_welcome_preserved(self):
        self.assertEqual(hashlib.sha256(WELCOME.encode()).hexdigest(), prompt_fit.SMOKE_PROMPT_SHA256)
        argv = factory.native_argv('bonsai-stock', WELCOME)
        self.assertEqual(argv[argv.index('--ctx-size') + 1], '512')
        self.assertEqual(argv[argv.index('--predict') + 1], '64')
        self.assertEqual(argv[argv.index('--fit') + 1], 'on')
        self.assertEqual(argv[argv.index('--load-mode') + 1], 'dio')

    def test_legacy_other_prompt_preserved(self):
        argv = factory.native_argv('bonsai-stock', WELCOME + ' Changed.')
        for flag, value in {'--ctx-size': '4096', '--predict': '64', '--fit': 'off',
                            '--gpu-layers': '99', '--override-tensor': '.*=Vulkan0'}.items():
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertNotIn('--load-mode', argv)

    def test_explicit_profile_ignores_welcome_shortcut_and_has_unique_flags(self):
        for role, output in profiles.ROLE_OUTPUT_TOKENS.items():
            argv = factory.native_argv('bonsai-stock', WELCOME, profile_id=profiles.PROFILE_ID, role=role)
            for flag, value in {'--ctx-size': '4096', '--predict': str(output), '--fit': 'on',
                                '--fit-target': '1536', '--load-mode': 'dio',
                                '--reasoning': 'off', '--threads': '4'}.items():
                self.assertEqual(argv.count(flag), 1)
                self.assertEqual(argv[argv.index(flag) + 1], value)
            self.assertNotIn('--gpu-layers', argv)
            self.assertNotIn('--override-tensor', argv)

    def test_role_requires_explicit_profile(self):
        with self.assertRaises(ValueError):
            factory.native_argv('bonsai-stock', 'fixture', role='generation')

    def test_guarded_envelope_preserves_limits_and_adds_capability(self):
        scope = {'pid': 12, 'start_ticks': 34, 'pid_namespace': 'fixture'}
        legacy = backend.guarded_argv(Path('/fixture'), scope, 100)
        demo = backend.guarded_argv(Path('/fixture'), scope, 100,
                                    demo_capability_path='/proc/123/fd/7')
        self.assertEqual(demo[:-2], legacy)
        self.assertEqual(demo[-2:], ['--demo-capability-path', '/proc/123/fd/7'])
        for item in ('7GiB', '12GiB', '16GiB', '64MiB', '512MiB', '128MiB', '256MiB',
                     '500000', '250000', '60', '50'):
            self.assertIn(item, demo)


class FitTests(unittest.TestCase):
    def test_fit_exact_boundary_includes_four_token_margin(self):
        for role in ('generation', 'reflection'):
            output = profiles.ROLE_OUTPUT_TOKENS[role]
            fit = fixture_fit(role=role, rendered_bytes=4096 - output - 4 - 2)
            profiles.require_profile_fit(fit, profiles.MODEL_ID, 'fixture', profiles.PROFILE_ID, role)
            fit['formatted_utf8_bytes'] += 1
            fit['input_token_upper_bound'] += 1
            with self.assertRaises(ValueError):
                profiles.require_profile_fit(fit, profiles.MODEL_ID, 'fixture', profiles.PROFILE_ID, role)

    def test_fit_cross_role_model_prompt_reserve_rejected(self):
        fit = fixture_fit()
        for key, value in [('role', 'calibration'), ('prompt_sha256', '0'*64),
                           ('output_tokens_reserved', 128), ('context', 512),
                           ('template_sha256', '0'*64), ('profile_sha256', '0'*64),
                           ('context_margin_tokens', 0), ('input_token_upper_bound', True),
                           ('layers', 64), ('metadata_bytes_read', 0)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                profiles.require_profile_fit({**fit, key: value}, profiles.MODEL_ID,
                                             'fixture', profiles.PROFILE_ID, 'generation')

    def test_family_requires_exact_reviewed_policy_and_per_prompt_fit(self):
        fit = fixture_fit()
        qualification = {'prompt_family': profiles.prompt_family_policy()}
        backend.require_prompt_fit(qualification, fit, model_id=profiles.MODEL_ID,
                                   prompt='fixture', profile_id=profiles.PROFILE_ID, role='generation')
        for invalid in ({}, {'prompt_family': {'schema': 'PMI_REVIEWED_PROMPT_FAMILY_V1'}},
                        {'prompt_family': {**profiles.prompt_family_policy(), 'loop_model_calls': 19}}):
            with self.assertRaises(ValueError):
                backend.require_prompt_fit(invalid, fit, model_id=profiles.MODEL_ID,
                                           prompt='fixture', profile_id=profiles.PROFILE_ID, role='generation')
        with self.assertRaises(ValueError):
            backend.require_prompt_fit(qualification, fit, model_id=profiles.MODEL_ID,
                                       prompt='other', profile_id=profiles.PROFILE_ID, role='generation')

    def test_legacy_fit_equality_unchanged(self):
        fit = {'prompt_sha256': 'fixture', 'output_tokens_reserved': 64}
        backend.require_prompt_fit({'prompt_fits': {'fixture': fit}}, fit)
        with self.assertRaises(ValueError):
            backend.require_prompt_fit({'prompt_fits': {'fixture': fit}}, {**fit, 'output_tokens_reserved': 128})

    def test_synthetic_renderer_metadata_fixture_only(self):
        raw = b'synthetic template only'
        metadata = {'general.architecture': 'qwen35', 'tokenizer.ggml.model': 'gpt2',
                    'tokenizer.ggml.pre': 'qwen35', 'tokenizer.chat_template': raw.decode(),
                    'tokenizer.ggml.tokens': list(prompt_fit.byte_alphabet()), 'qwen35.block_count': 64}
        template = Mock()
        environment = Mock()
        environment.from_string.return_value = template
        fake_jinja = types.SimpleNamespace(Environment=Mock(return_value=environment), StrictUndefined=object())
        with patch.object(prompt_fit.Path, 'read_bytes', return_value=raw), \
                patch.object(prompt_fit, 'gguf_metadata', return_value=(metadata, 100)) as reader, \
                patch.object(prompt_fit, 'BONSAI_TEMPLATE_SHA256', hashlib.sha256(raw).hexdigest()), \
                patch.dict(sys.modules, {'jinja2': fake_jinja}):
            template.render.return_value = 'x' * (4096 - 512 - 4 - 2)
            result = prompt_fit.fit_prompt('fixture', '/no-model', '/no-template', profile='bonsai-stock',
                                           profile_id=profiles.PROFILE_ID, role='generation')
            self.assertEqual((result['output_tokens_reserved'], result['context_margin_tokens']), (512, 4))
            self.assertEqual(reader.call_count, 1)
            template.render.return_value += 'x'
            with self.assertRaises(ValueError):
                prompt_fit.fit_prompt('fixture', '/no-model', '/no-template', profile='bonsai-stock',
                                      profile_id=profiles.PROFILE_ID, role='generation')


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        # Any accidentally reached real asset/process operation fails immediately.
        self.stack.enter_context(patch.object(backend, 'verify_runtime_and_model', side_effect=AssertionError('asset probe')))
        self.stack.enter_context(patch.object(prompt_fit, 'gguf_metadata', side_effect=AssertionError('GGUF probe')))
        self.stack.enter_context(patch.object(native_worker.os, 'execve', side_effect=AssertionError('native exec')))
        import subprocess
        self.stack.enter_context(patch.object(subprocess, 'Popen', side_effect=AssertionError('subprocess')))
        self.authority_module = types.ModuleType('research.demo_authority')
        self.authority_module.gate_model = Mock()
        self.authority_module.gate_native = Mock()
        self.stack.enter_context(patch.dict(sys.modules, {'research.demo_authority': self.authority_module}))

    def test_legacy_direct_backend_still_held_before_assets(self):
        from launch_policy import PlatformHeld
        with self.assertRaises(PlatformHeld):
            backend.LlamaBackend().run_stage('bonsai-stock', 'fixture', Path('/absent'), 1)
        backend.verify_runtime_and_model.assert_not_called()

    def test_demo_backend_refusal_before_assets(self):
        self.authority_module.gate_model.side_effect = PermissionError('fixture denied')
        with self.assertRaises(PermissionError):
            backend.LlamaBackend(authority=object()).run_stage('bonsai-stock', 'fixture', Path('/absent'), 1,
                                                               profile_id=profiles.PROFILE_ID, role='generation')
        backend.verify_runtime_and_model.assert_not_called()

    def test_demo_agent_refusal_before_stage_creation(self):
        self.authority_module.gate_model.side_effect = PermissionError('fixture denied')
        with tempfile.TemporaryDirectory() as root:
            stage = Path(root) / 'stage'
            with self.assertRaises(PermissionError):
                DemoRealAgent().call('generation', 'bonsai-stock', 'fixture', stage, 1)
            self.assertFalse(stage.exists())

    def test_legacy_agent_request_and_backend_signature(self):
        import launch_policy
        capture = fixture_capture()
        with tempfile.TemporaryDirectory() as root, patch.object(launch_policy, 'require_real_execution'), \
                patch.object(backend.LlamaBackend, 'run_stage', return_value=capture) as dispatch:
            stage = Path(root) / 'stage'
            reply = RealAgent().call('generation', 'bonsai-stock', 'fixture', stage, 1)
            self.assertEqual(set(reply), {'text', 'cleanup_verified', 'actual_inference'})
            self.assertEqual(json.loads((stage/'request.json').read_text()), {'model_id': 'bonsai-stock', 'prompt': 'fixture'})
            dispatch.assert_called_once_with('bonsai-stock', 'fixture', stage, 1)

    def test_demo_agent_propagation_and_receipt(self):
        authority, coordination = object(), object()
        with tempfile.TemporaryDirectory() as root, \
                patch.object(backend.LlamaBackend, 'run_stage', return_value=fixture_capture()) as dispatch:
            stage = Path(root) / 'stage'
            reply = DemoRealAgent(authority=authority, coordination=coordination).call(
                'generation', 'bonsai-stock', 'fixture', stage, 1)
            dispatch.assert_called_once_with('bonsai-stock', 'fixture', stage, 1,
                                             profile_id=profiles.PROFILE_ID, role='generation')
            self.assertEqual(reply['fit_sha256'], digest(reply['fit']))
            self.assertEqual(reply['output_tokens'], 512)
            profiles.validate_model_request(json.loads((stage/'request.json').read_text()))
            self.assertEqual(json.loads((stage/'result.json').read_text())['fit'], reply['fit'])
            calls = self.authority_module.gate_model.call_args_list
            self.assertEqual([call.args[1] for call in calls], ['agent', 'complete'])
            self.assertIs(calls[0].kwargs['coordination'], coordination)

    def test_demo_backend_fixture_pipeline_binds_capability_and_fit(self):
        owned = types.ModuleType('run_gpu_window')
        owned.STOP = False
        owned.interrupted = Mock()
        owned.event = Mock()
        owned.run_owned = Mock(return_value=0)
        common = types.ModuleType('window_common')
        common.alive = Mock(return_value=True)
        common.fingerprint = Mock(return_value={})
        common.require_scope = Mock(return_value={'pid': 123, 'start_ticks': 456, 'pid_namespace': 'fixture'})
        def passive(path, seconds, deadline, check):
            path.write_text(json.dumps({'samples': [{'available': 17*1024**3, 'vram': 0,
                                                     'time': time.monotonic()}]}))
        common.passive = passive
        admission = types.ModuleType('admission_policy')
        admission.admit = Mock(side_effect=passive)
        telemetry = types.ModuleType('load_telemetry')
        telemetry.LoadTelemetry = Mock()
        telemetry.LoadTelemetry.return_value.direct_io_evidence = [{'O_DIRECT': True}]
        lifecycle = []
        @contextmanager
        def capability(authority, request_path, request, deadline):
            lifecycle.append(('open', authority, request_path, request, deadline))
            yield '/proc/123/fd/7'
            lifecycle.append(('closed',))
        self.authority_module.native_capability = capability
        modules = {'run_gpu_window': owned, 'window_common': common,
                   'admission_policy': admission, 'load_telemetry': telemetry}
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, modules), \
                patch.object(backend, 'integrity_observation', return_value={'fixture': True}), \
                patch.object(backend, 'verify_runtime_and_model', return_value=(
                    {'layers': 65, 'actual_prompt_fit': fixture_fit()}, {})), \
                patch.object(backend.signal, 'signal'), patch.object(sys, 'path', sys.path[:]), \
                patch.object(backend, 'require_cleanup', return_value=(
                    {'status': 'completed', 'command_returncode': 0},
                    {'status': 'completed', 'command_returncode': 0})), \
                patch.object(backend, 'validate_placement', return_value={}) as placement, \
                patch.object(backend, 'validate_direct_loading', return_value={'mode': 'dio'}):
            stage = Path(root)
            request = profiles.model_request('bonsai-stock', 'fixture', profiles.PROFILE_ID, 'generation')
            (stage/'request.json').write_text(json.dumps(request))
            (stage/'native.stdout.log').write_bytes(b'fixture [end of text]')
            (stage/'native.stderr.log').write_text('using device Vulkan0 (AMD Radeon RX 6600 XT) (0000:04:00.0)')
            authority = object()
            deadline = time.monotonic() + 120
            result = backend.LlamaBackend(authority=authority, coordination=object()).run_stage(
                'bonsai-stock', 'fixture', stage, deadline, profile_id=profiles.PROFILE_ID, role='generation')
            self.assertEqual(result['fit_sha256'], digest(fixture_fit()))
            self.assertEqual((result['context_tokens'], result['output_tokens']), (4096, 512))
            self.assertEqual(lifecycle, [('open', authority, stage/'request.json', request, deadline), ('closed',)])
            self.assertEqual(owned.run_owned.call_args.args[0][-2:], ['--demo-capability-path', '/proc/123/fd/7'])
            self.assertEqual(placement.call_args.args[1:], (65, 4096, True))
            admission.admit.assert_called_once()
            integrity = json.loads((stage/'integrity-io.json').read_text())
            self.assertTrue(integrity['verified'])
            self.assertEqual(integrity['before'], {'fixture': True})
            self.assertEqual(integrity['after'], {'fixture': True})
            self.assertEqual(integrity['model_read_mode'], 'O_DIRECT/256KiB/aligned/no-buffered-fallback')

    def test_demo_capture_mismatch_or_truncation_is_released_failure(self):
        for changed in ({'output_tokens': 128}, {'raw_stdout': b'truncated'}, {'fit_sha256': '0'*64}):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as root, \
                    patch.object(backend.LlamaBackend, 'run_stage', return_value={**fixture_capture(), **changed}):
                stage = Path(root) / 'stage'
                with self.assertRaises(ReleasedStageError):
                    DemoRealAgent().call('generation', 'bonsai-stock', 'fixture', stage, 1)
                self.assertFalse((stage/'result.json').exists())
                self.assertEqual(json.loads((stage/'failure.json').read_text())['status'], 'ReleasedInvalidOutput')

    def test_native_refusal_before_descriptor_changes(self):
        request = profiles.model_request('bonsai-stock', 'fixture', profiles.PROFILE_ID, 'generation')
        self.authority_module.gate_native.side_effect = PermissionError('fixture denied')
        with patch.object(sys, 'argv', ['native_worker.py', '--request', '/fixture/request.json',
                                      '--demo-capability-path', '/proc/123/fd/7']), \
                patch.object(native_worker, 'bounded_read', return_value=json.dumps(request).encode()), \
                patch.object(native_worker.os, 'open', side_effect=AssertionError('descriptor mutation')):
            with self.assertRaises(PermissionError):
                native_worker.main()
        self.authority_module.gate_native.assert_called_once_with(
            Path('/fixture/request.json'), request, capability_path='/proc/123/fd/7')

    def test_native_legacy_gate_before_read(self):
        from launch_policy import PlatformHeld
        with patch.object(sys, 'argv', ['native_worker.py', '--request', '/fixture/request.json']), \
                patch.object(native_worker, 'bounded_read', side_effect=AssertionError('request read')):
            with self.assertRaises(PlatformHeld):
                native_worker.main()


if __name__ == '__main__':
    unittest.main()
