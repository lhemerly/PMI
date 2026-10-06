"""Offline tests for missing telemetry and post-exit sample exclusion."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('analysis', Path(__file__).resolve().parents[1] / 'tools/analyze_loader_pressure.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class AnalysisTests(unittest.TestCase):
    def test_live_window_and_missing_are_not_zero(self):
        rows = [dict(event='sample', elapsed_seconds=t, members=[dict(alive=live)], observations=None)
                for t,live in [(1,True),(2,False),(6,True)]]
        rows += [dict(event='final',status='limit_exceeded',reason='max_psi_full_us')]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'host.jsonl';p.write_text('\n'.join(map(json.dumps,rows)))
            result = analysis.summarize(p)
        self.assertEqual(result['sample_count'],1)
        self.assertIsNone(result['last_live_sample']['host_full_stall_us'])
        self.assertIsNone(result['last_live_sample']['context_anon_thp_growth_bytes'])
        self.assertEqual(result['final']['reason'],'max_psi_full_us')

    def test_zero_pressure_and_signed_gauges(self):
        self.assertEqual(analysis.metric({'x':{'delta_from_baseline':0}},'x'),0)
        self.assertEqual(analysis.metric({'x':{'delta_from_baseline':-4096}},'x'),-4096)
        self.assertIsNone(analysis.metric({},'x'))


if __name__ == '__main__':
    unittest.main()
