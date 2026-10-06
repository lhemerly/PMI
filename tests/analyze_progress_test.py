import importlib.util,json,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('a',Path(__file__).resolve().parents[1]/'tools/analyze_progress.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
class Test(unittest.TestCase):
 def test_interrupted_and_partial(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'events';rs=[{'schema':'pmi.progress.v1','sequence':i,'monotonic_ns':i,'phase':phase}for i,phase in enumerate(['run_begin','evaluation_begin','upload_begin'])]
   rs[1].update(evaluation=0,prefill=True,input_tokens=1);rs[2]['request']=1
   p.write_text(''.join(json.dumps(x)+'\n' for x in rs)+'{"partial":')
   x=a.analyze(p);self.assertTrue(x['partial_tail']);self.assertFalse(x['complete']);self.assertEqual(x['last_cache_event']['phase'],'upload_begin');self.assertEqual(x['last_active_evaluation']['evaluation'],0)
   p.write_text('{malformed}\n');self.assertRaises(ValueError,a.analyze,p)
 def test_completed_metrics(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'events';rs=[{'schema':'pmi.progress.v1','sequence':i,'monotonic_ns':i,'phase':phase}for i,phase in enumerate(['run_begin','evaluation_complete','prefill_complete','run_complete'])]
   rs[1]['metrics']={'evaluation':0,'phase':'prefill'};rs[-1]['details_truncated']=True;p.write_text(''.join(json.dumps(x)+'\n'for x in rs))
   x=a.analyze(p);self.assertTrue(x['complete']);self.assertTrue(x['details_truncated']);self.assertEqual(x['completed_prefill_evaluations'],1)
if __name__=='__main__':unittest.main()
