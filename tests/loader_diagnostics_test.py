import sys
from pathlib import Path
import tempfile
import unittest
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from loader_diagnostics import LoaderDiagnostics,zones

class Tests(unittest.TestCase):
 def test_zone_units_and_missing_counter(self):
  self.assertEqual(zones('Node 0, zone Normal\n pages free 20\n low 3\n high 8\n boost 2\n'),[dict(node=0,zone='Normal',free_pages=20,low_pages=3,high_pages=8,boost_pages=2)])
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'vmstat').write_text('compact_fail 3\nthp_fault_alloc 5\n')
   collector=LoaderDiagnostics(root,root)
   a=collector.sample()
   self.assertIsNone(a['vmstat']['allocstall_normal']['value'])
   self.assertEqual(a['vmstat']['thp_fault_alloc']['delta_from_baseline'],0)
   (root/'vmstat').write_text('compact_fail 4\nthp_fault_alloc 8\n')
   b=collector.sample()
   self.assertEqual(b['vmstat']['thp_fault_alloc']['delta_from_baseline'],3)
   (root/'vmstat').write_text('compact_fail 1\nthp_fault_alloc 1\n')
   c=collector.sample()
   self.assertTrue(c['vmstat']['compact_fail']['reset'])
   self.assertIsNone(c['vmstat']['compact_fail']['delta_from_baseline'])
   self.assertTrue(c['unavailable'])
if __name__=='__main__':unittest.main()
