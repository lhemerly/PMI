"""Tiny CPU-backed device fixture only; no GPU or trained checkpoint."""
import argparse,importlib.util,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('full',ROOT/'tests/full_model_test.py')
full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)
p=argparse.ArgumentParser();p.add_argument('pmi',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);shards=full.fixture.generate(a.output/'tiny.gguf',shards=2)
cmd=[str(a.pmi.resolve()),'run',*map(str,shards),'--prompt','hello world','--max-tokens','3','--context','64',
     '--prefill-chunk','2','--cpu-repack','off','--cache-bytes','8192','--projection-backend','device-cpu',
     '--device-cache-bytes','16384','--device-graph-bytes','65536','--trace-mode','compact']
def run(name,extra):
 t=a.output/(name+'.json');r=subprocess.run([*cmd,'--trace-json',str(t),*extra],capture_output=True,text=True,timeout=30)
 (a.output/(name+'.stderr')).write_text(r.stderr);assert r.returncode==0,r.stderr[-3000:]
 return json.loads(t.read_text())
native=run('default',[]);progress=a.output/'progress.jsonl';actual=run('observed',['--progress-jsonl',str(progress)])
full.compare(native,actual,compact=True)
rs=[json.loads(x) for x in progress.read_text().splitlines()];ph=[x['phase'] for x in rs]
assert ph[0]=='run_begin' and ph[-1]=='run_complete'
assert [x['sequence'] for x in rs]==list(range(len(rs)))
assert all(x['monotonic_ns']<=y['monotonic_ns'] for x,y in zip(rs,rs[1:]))
for x,y in [('backend_complete','model_load_begin'),('model_load_complete','context_begin'),('context_complete','evaluation_begin'),('prefill_complete','first_output_complete')]:assert ph.index(x)<ph.index(y)
ends=[x['metrics'] for x in rs if x['phase']=='evaluation_complete'];assert len(ends)==len(actual['metrics']['evaluations'])
assert [x['evaluation'] for x in ends]==list(range(len(ends)))
assert any(x['phase']=='upload_begin' for x in rs) and any(x['phase']=='upload_end' for x in rs)
assert rs[-2]['phase']=='generation_complete'
assert not any(k in str(rs) for k in ['hello world','ffn_input','token_ids','predictions'])
split_progress=a.output/'split-progress.jsonl'
split=run('split',['--projection-dispatch','split','--progress-jsonl',str(split_progress)])
full.compare(native,split,compact=True)
assert json.loads(split_progress.read_text().splitlines()[-1])['phase']=='run_complete'
transient_progress=a.output/'transient-progress.jsonl'
transient=run('transient',['--device-upload-host-policy','transient','--progress-jsonl',str(transient_progress)])
full.compare(native,transient,compact=True)
assert transient['metrics']['device_upload_host_policy']=='transient'
assert native['metrics']['device_upload_host_policy']=='retained'
trs=[json.loads(x) for x in transient_progress.read_text().splitlines()]
assert all(x['host_current_bytes']<=8192 for x in trs if 'host_current_bytes' in x)
assert trs[-1]['phase']=='run_complete'
assert all(x['host_current_bytes']==0 for x in trs if x['phase']=='evaluation_complete')
assert actual['metrics']['cache_reserved_bytes']>transient['metrics']['cache_reserved_bytes']==0
transient_split=run('transient-split',['--device-upload-host-policy','transient','--projection-dispatch','split'])
full.compare(native,transient_split,compact=True)
r=subprocess.run([*cmd,'--progress-jsonl',str(progress)],capture_output=True,text=True,timeout=30)
assert r.returncode and progress.read_text().splitlines()[0]==json.dumps(rs[0],separators=(',',':'))
print('Progress model stages, per-evaluation completion, cache/upload markers, default parity and exclusive file passed')
