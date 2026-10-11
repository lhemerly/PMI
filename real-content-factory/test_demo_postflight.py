"""Captured passive samples and isolated pure policy fixtures; never collect live."""
import ast
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import admission_policy

CAPTURE = Path('/tmp/pmi-model-demo-v4/calibration/postflight.json')
HELPER = Path('/home/luizh/Documents/Codex/2026-10-06/task-3/upstream-v060/gpu-qualification/window_common.py')


def isolated_passive_reasons():
    # Execute only the reviewed pure function, with no module imports, device
    # reads, passive sampler, subprocess helpers, sleeps, or model operations.
    source = ast.parse(HELPER.read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'passive_reasons')
    module = ast.Module(body=[node], type_ignores=[])
    scope = {'math': math}
    exec(compile(module, str(HELPER), 'exec'), scope)
    return scope['passive_reasons']


class DemoPostflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.captured = json.loads(CAPTURE.read_text())
        cls.reasons = staticmethod(isolated_passive_reasons())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'postflight.json'
        self.samples = copy.deepcopy(self.captured['samples'])
        self.check = Mock()

    def run_policy(self, *, samples=None, reasons=None, wrapper=admission_policy.demo_postflight,
                   seconds=20, passive_success=False):
        samples = self.samples if samples is None else samples
        reasons = self.reasons(samples, seconds) if reasons is None else reasons
        def passive(output, requested_seconds, deadline, check):
            self.assertEqual((requested_seconds, deadline), (seconds, 999))
            check()
            output.write_text(json.dumps({'samples': samples, 'reasons': reasons}))
            if not passive_success:
                raise ValueError('; '.join(reasons))
            return samples[-1]['time']
        fake = types.ModuleType('window_common')
        fake.passive = Mock(side_effect=passive)
        fake.passive_reasons = self.reasons
        with patch.dict(sys.modules, {'window_common': fake}):
            return wrapper(self.output, seconds, 999, self.check)

    def test_captured_40_samples_pass_elapsed_coverage(self):
        self.assertEqual(len(self.samples), 40)
        self.assertEqual(self.captured['reasons'], ['passive coverage/count/duration'])
        result = self.run_policy()
        self.assertEqual(result, self.samples[-1]['time'])
        receipt = json.loads(Path(str(self.output)+'.coverage.json').read_text())
        self.assertEqual(receipt['policy'], 'model-demo-postflight-elapsed-coverage-v1')
        self.assertEqual(receipt['sample_count'], 40)
        self.assertTrue(20 <= receipt['duration_seconds'] <= 21)
        self.assertEqual(receipt['reasons'], [])
        self.check.assert_called_once()

    def test_all_other_passive_failures_rejected(self):
        changes = {
            'RAM': lambda s: s[0].update(available=13*1024**3),
            'VRAM': lambda s: s[0].update(vram=1024**3+1),
            'busy': lambda s: s[0].update(busy=11),
            'GPU thermal': lambda s: s[0].update(GPU_hotspot_mC=70001),
            'CPU thermal': lambda s: s[0].update(CPU_package_mC=80001),
            'OOM': lambda s: s[-1].update(oom_kill=s[0]['oom_kill']+1),
            'swap': lambda s: s[-1].update(swapout=s[0]['swapout']+4096),
            'PSI': lambda s: s[-1].update(PSI_full_us=s[0]['PSI_full_us']+250001),
            'identity': lambda s: s[-1].update(source_identity={'changed': True}),
            'regression': lambda s: s[-1].update(PSI_some_us=s[0]['PSI_some_us']-1),
            'gap': lambda s: s[1].update(time=s[0]['time']+1.01),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                samples = copy.deepcopy(self.samples)
                change(samples)
                with self.assertRaises(ValueError):
                    self.run_policy(samples=samples)
                self.assertFalse(Path(str(self.output)+'.coverage.json').exists())

    def test_saved_coverage_label_does_not_hide_other_failures(self):
        self.samples[-1]['oom_kill'] += 1
        with self.assertRaises(ValueError):
            self.run_policy(reasons=['passive coverage/count/duration'])

    def test_short_long_nonfinite_and_regressed_times_rejected(self):
        for end in (19.99, 21.01, float('nan'), float('inf'), -1):
            with self.subTest(end=end):
                samples = copy.deepcopy(self.samples)
                for i, sample in enumerate(samples):
                    sample['time'] = samples[0]['time'] + i * end / (len(samples)-1)
                with self.assertRaises(ValueError):
                    self.run_policy(samples=samples, reasons=['passive coverage/count/duration'])

    def test_nonfinite_interior_timestamp_rejected(self):
        self.samples[12]['time'] = float('nan')
        with self.assertRaises(ValueError):
            self.run_policy(reasons=['passive coverage/count/duration'])

    def test_unknown_and_combined_reasons_not_reclassified(self):
        for reasons in (['unrecognized passive failure'],
                        ['passive coverage/count/duration', 'passive memory pressure'], []):
            with self.subTest(reasons=reasons), self.assertRaises(ValueError):
                self.run_policy(reasons=reasons)

    def test_missing_artifact_failure_preserved(self):
        fake = types.SimpleNamespace(passive=Mock(side_effect=ValueError('collector failed')))
        with patch.dict(sys.modules, {'window_common': fake}), self.assertRaisesRegex(ValueError, 'collector failed'):
            admission_policy.demo_postflight(self.output, 20, 999, self.check)

    def test_unrelated_exception_preserved(self):
        fake = types.SimpleNamespace(passive=Mock(side_effect=RuntimeError('deadline failed')))
        with patch.dict(sys.modules, {'window_common': fake}), self.assertRaisesRegex(RuntimeError, 'deadline failed'):
            admission_policy.demo_postflight(self.output, 20, 999, self.check)

    def test_successful_original_passive_path_unchanged(self):
        self.assertEqual(self.run_policy(passive_success=True, reasons=[]), self.samples[-1]['time'])
        self.assertFalse(Path(str(self.output)+'.coverage.json').exists())

    def test_speech_receipt_and_semantics_preserved(self):
        self.run_policy(wrapper=admission_policy.speech_postflight)
        receipt = json.loads(Path(str(self.output)+'.coverage.json').read_text())
        self.assertEqual(receipt, {'policy': 'speech-postflight-elapsed-coverage-v1',
                                  'duration_seconds': self.samples[-1]['time']-self.samples[0]['time'],
                                  'sample_count': 40, 'reasons': []})

    def test_demo_duration_is_fixed(self):
        for seconds in (19, 21, True, 20.0):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                admission_policy.demo_postflight(self.output, seconds, 999, self.check)


if __name__ == '__main__':
    unittest.main()
