"""Inert syntax and mocked probe tests; never invoke the qualification worker."""
import ast
import errno
from pathlib import Path
import unittest
from unittest.mock import Mock,patch

from research import qualification_worker as worker


class QualificationWorkerStaticTests(unittest.TestCase):
    def test_fixed_check_names_and_no_candidate_execution_api(self):
        source=Path(worker.__file__).read_text();tree=ast.parse(source)
        self.assertEqual(len(worker.EXPECTED_CHECKS),16)
        self.assertEqual(len(set(worker.EXPECTED_CHECKS)),16)
        calls={node.func.id for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
        self.assertFalse(calls & {'exec','eval','compile','__import__'})
        self.assertNotIn('/candidate/source.py',source)

    def test_socket_probe_requires_eperm_and_closes_unexpected_success(self):
        with patch.object(worker.socket,'socket',side_effect=PermissionError(errno.EPERM,'fixture')):
            self.assertTrue(worker.denied_socket())
        with patch.object(worker.socket,'socket',side_effect=PermissionError(errno.EACCES,'fixture')):
            self.assertFalse(worker.denied_socket())
        socket=Mock()
        with patch.object(worker.socket,'socket',return_value=socket):self.assertFalse(worker.denied_socket())
        socket.close.assert_called_once()

    def test_fork_probe_requires_eperm_and_reaps_unexpected_success(self):
        with patch.object(worker.os,'fork',side_effect=PermissionError(errno.EPERM,'fixture')):
            self.assertTrue(worker.denied_fork())
        with patch.object(worker.os,'fork',side_effect=PermissionError(errno.EACCES,'fixture')):
            self.assertFalse(worker.denied_fork())
        with patch.object(worker.os,'fork',return_value=99),patch.object(worker.os,'waitpid',return_value=(99,0)) as wait:
            self.assertFalse(worker.denied_fork())
        wait.assert_called_once_with(99,0)


if __name__=='__main__':unittest.main()
