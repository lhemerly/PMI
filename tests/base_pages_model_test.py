"""Generated-model policy/observer coverage; no trained model or GPU required."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('full',ROOT/'tests/full_model_test.py')
full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)


def main():
 p=argparse.ArgumentParser();p.add_argument('pmi',type=Path);p.add_argument('output',type=Path);p.add_argument('--observer-test',type=Path,default=Path(os.environ['PMI_OBSERVER_TEST']) if 'PMI_OBSERVER_TEST' in os.environ else None);p.add_argument('--vulkan',action='store_true')
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 shards=full.fixture.generate(a.output/'base.gguf',shards=2)
 common=['--prompt','hello world','--max-tokens','8','--context','64','--prefill-chunk','1','--cpu-repack','off','--trace-mode','compact']
 def run(name,flags):
  trace=a.output/(name+'.json')
  r=subprocess.run([str(a.pmi.resolve()),'run',*map(str,shards),*common,'--dense-load','direct-required','--cache-bytes','8192','--trace-json',str(trace),*flags],capture_output=True,text=True,timeout=60)
  (a.output/(name+'.stderr')).write_text(r.stderr)
  assert r.returncode==0,r.stderr[-4000:]
  return json.loads(trace.read_text())
 native=run('native',[])
 assert native['metrics']['dense_pages']==dict(base_requested=False,observed=False,owned_buffers=0,advised_buffers=0,advised_bytes=0)
 for base in (False,True):
  name='base' if base else 'native-observed';events=a.output/(name+'.jsonl')
  actual=run(name,['--dense-pages',name if base else 'native','--load-trace-jsonl',str(events)])
  full.compare(native,actual,compact=True)
  rows=[json.loads(l) for l in events.read_text().splitlines()]
  assert rows[0]['base_pages']==base
  rows=rows[1:]
  assert all(a['monotonic_ns']<=b['monotonic_ns'] for a,b in zip(rows,rows[1:]))
  buffers=[r for r in rows if r['phase']=='buffer_ready'];assert len(buffers)==1
  page=os.sysconf('SC_PAGESIZE')
  for b in buffers:
   assert b['owned_begin'] and b['owned_bytes']
   if base:
    assert b['advised_bytes'] and b['advised_begin']%page==0 and b['advised_bytes']%page==0
    assert b['owned_begin']<=b['advised_begin']<b['advised_begin']+b['advised_bytes']<=b['owned_begin']+b['owned_bytes']
    assert b['owned_bytes']-b['advised_bytes']<2*page
   else:assert b['advised_bytes']==0
  m=actual['metrics'];dense=m['dense_load']
  assert dense['fallback_count']==0 and dense['buffered_read_bytes']==0
  assert sum(r['completed_bytes'] for r in rows if r['phase']=='tensor_end')==dense['payload_completed_bytes']
  assert sum(r['completed_bytes'] for r in rows if r['phase']=='read_end')==dense['direct_read_bytes']
  assert m['dense_pages']['advised_bytes']==sum(b['advised_bytes'] for b in buffers)
  assert any(r['phase']=='read_begin' for r in rows) and any(r['phase']=='copy_end' for r in rows)
 if a.vulkan:
  actual=run('base-vulkan',['--dense-pages','base','--load-trace-jsonl',str(a.output/'base-vulkan.jsonl'),
    '--projection-backend','device-vulkan','--projection-device-index','0','--device-cache-bytes','16384','--device-graph-bytes','65536'])
  full.compare(native,actual,compact=True)
  assert actual['metrics']['dense_pages']['advised_buffers']==1
 # Exclusive trace creation must fail, leaving its prior bytes untouched.
 sentinel=a.output/'existing.jsonl';sentinel.write_text('preserve\n')
 r=subprocess.run([str(a.pmi.resolve()),'run',*map(str,shards),*common,'--dense-load','direct-required','--load-trace-jsonl',str(sentinel)],capture_output=True,text=True)
 assert r.returncode and sentinel.read_text()=='preserve\n' and not r.stdout
 if a.observer_test:
  r=subprocess.run([str(a.observer_test.resolve()),*map(str,shards)],capture_output=True,text=True,timeout=60)
  (a.output/'observer.stderr').write_text(r.stderr);(a.output/'observer.stdout').write_text(r.stdout)
  assert r.returncode==0,r.stderr[-4000:]
 print('Base/native/default model parity, owned advice ranges, exact chunk/tensor I/O accounting, exclusive trace, observer failure cleanup passed')
if __name__=='__main__':main()
