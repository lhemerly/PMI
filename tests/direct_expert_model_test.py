"""Small generated Qwen fixtures only; never initializes a GPU or trained model."""
import argparse,importlib.util,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('full',R/'tests/full_model_test.py')
full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)
p=argparse.ArgumentParser();p.add_argument('pmi',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
cases=0;native_cases=0;unsupported_required_cases=0
for name,options in [('f32',{'expert_type':'f32'}),('q8-ple',{'expert_type':'q8_0','ple_type':'iq4_nl'})]:
 shards=full.fixture.generate(a.output/(name+'.gguf'),shards=2,**options)
 cmd=[str(a.pmi.resolve()),'run',*map(str,shards),'--prompt','hello world','--max-tokens','3','--context','64',
      '--prefill-chunk','2','--cpu-repack','off','--cache-bytes','65536','--projection-backend','device-cpu',
      '--device-cache-bytes','16384','--device-graph-bytes','65536','--trace-mode','compact']
 def run(label,extra,success=True):
  path=a.output/(name+'-'+label+'.json');result=subprocess.run([*cmd,'--trace-json',str(path),*extra],capture_output=True,text=True,timeout=30)
  (a.output/(name+'-'+label+'.stderr')).write_text(result.stderr)
  if not success:
   assert result.returncode!=0,'required direct mode silently accepted unsupported filesystem'
   assert 'direct'in result.stderr.lower() or 'unsupported'in result.stderr.lower(),result.stderr[-3000:]
   return None
  assert result.returncode==0,result.stderr[-3000:]
  return json.loads(path.read_text())
 baseline=run('buffered',[])
 native=None
 for retain in ['retained','transient']:
  for mode in ['direct-preferred','direct-required']:
   label=retain+'-'+mode;progress=a.output/(name+'-'+label+'.progress.jsonl')
   extra=['--device-upload-host-policy',retain,'--device-upload-host-io',mode,'--progress-jsonl',str(progress)]
   if mode=='direct-required' and not native:
    run(label,extra,success=False);unsupported_required_cases+=1;continue
   actual=run(label,extra)
   # Same backend/weight bytes: require bit-exact floats, ordered routing and tokens.
   full.compare(baseline,actual,atol=0,rtol=0,compact=True)
   metrics=actual['metrics'];io=metrics['provider_io'];assert metrics['device_upload_host_io']==mode
   assert io['direct_requests']==io['direct_completed']+io['fallback_count'],(mode,io)
   if native is None:native=io['direct_completed']>0
   if native:
    assert io['direct_completed']>0 and io['fallback_count']==0,(mode,io)
    native_cases+=1
   else:
    assert mode=='direct-preferred' and io['fallback_count']>0 and io['direct_read_bytes']==0,(mode,io)
   assert io['direct_read_bytes']>=io['direct_logical_bytes'] and io['staging_peak_bytes']<=65536
   rows=[json.loads(x)for x in progress.read_text().splitlines()]
   assert rows[-1]['phase']=='run_complete'
   assert any('cumulative_provider_io'in x for x in rows if x['phase']=='evaluation_complete')
   assert any('host_direct_read_bytes'in x for x in rows if x['phase']=='provider_end')
   cases+=1
 # The same direct provider policy also survives split dispatch/frozen residency.
 split_mode='direct-required' if native else 'direct-preferred'
 actual=run('split-direct',['--device-upload-host-policy','transient','--device-upload-host-io',split_mode,'--projection-dispatch','split'])
 full.compare(baseline,actual,atol=0,rtol=0,compact=True);cases+=1
 if native:native_cases+=1
 print(name,'complete bit-exact direct policy matrix passed')
print('Direct expert complete-graph numerical/routing/token parity:',cases,'cases;',native_cases,
      'native cases;',unsupported_required_cases,'unsupported required rejections; device-cpu; no trained/GPU run')
