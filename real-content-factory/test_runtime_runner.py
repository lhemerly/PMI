"""CPU wrapper deadline checks with no runtime loading or candidate execution."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research.runtime_entry import RuntimeCorrectnessRunner


class RuntimeRunnerDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.runner = RuntimeCorrectnessRunner.__new__(RuntimeCorrectnessRunner)
        self.runner.authority = SimpleNamespace(operation_deadline=200., cpu_complete=Mock())
        self.runner.runner = Mock()
        self.clock = patch('research.runtime_entry.time.monotonic', return_value=100.)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.processes = patch('subprocess.Popen', side_effect=AssertionError('physical launch forbidden'))
        self.processes.start()
        self.addCleanup(self.processes.stop)

    def test_rejects_nonfinite_boolean_or_nonfuture_deadlines_before_runner(self):
        for deadline in (True, False, None, '150', float('nan'), float('inf'), 99., 100.):
            with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                self.runner.run('request', 'source', 'stage', deadline)
        self.runner.runner.run.assert_not_called()
        self.runner.authority.cpu_complete.assert_not_called()

    def test_rejects_stage_longer_than_sixty_seconds(self):
        with self.assertRaises(ValueError):
            self.runner.run('request', 'source', 'stage', 160.001)
        self.runner.runner.run.assert_not_called()

    def test_rejects_deadline_beyond_authorized_operation_cutoff(self):
        self.runner.authority.operation_deadline = 125.
        with self.assertRaises(ValueError):
            self.runner.run('request', 'source', 'stage', 125.001)
        self.runner.runner.run.assert_not_called()

    def test_passes_exact_valid_deadline_and_validates_returned_receipt(self):
        receipt = {'fixture': 'not execution evidence'}
        self.runner.runner.run.return_value = receipt
        self.runner.authority.operation_deadline = 160.
        result = self.runner.run('request', 'source', 'stage', 160.)
        self.assertIs(result, receipt)
        self.runner.runner.run.assert_called_once_with('request', 'source', 'stage', 160.)
        self.runner.authority.cpu_complete.assert_called_once_with(receipt)

    def test_underlying_failure_cannot_mark_cpu_operation_complete(self):
        self.runner.runner.run.side_effect = RuntimeError('owned cleanup uncertain')
        with self.assertRaisesRegex(RuntimeError, 'cleanup uncertain'):
            self.runner.run('request', 'source', 'stage', 150.)
        self.runner.authority.cpu_complete.assert_not_called()

    def test_authority_receipt_rejection_propagates(self):
        self.runner.authority.cpu_complete.side_effect = ValueError('receipt invalid')
        with self.assertRaisesRegex(ValueError, 'receipt invalid'):
            self.runner.run('request', 'source', 'stage', 150.)


if __name__ == '__main__':
    unittest.main()
