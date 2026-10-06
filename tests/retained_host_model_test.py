"""Scaled reusable-host-cache correctness; generated fixtures and device-CPU only."""
import argparse,importlib.util,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('full',R/'tests/full_model_test.py')
full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)
p=argparse.ArgumentParser();p.add_argument('pmi',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);cases=0;reuse_improvements=0
for name,options in [('f32',{'expert_type':'f32'}),('q8-ple',{'expert_type':'q8_0','ple_type':'iq4_nl'})]:
 shards=full.fixture.generate(a.output/(name+'.gguf'),shards=2,**options)
 cmd=[str(a.pmi.resolve()),'run',*map(str,shards),'--prompt','hello world','--max-tokens','3','--context','64',
      '--prefill-chunk','2','--cpu-repack','off','--projection-backend','device-cpu','--device-cache-bytes','16384',
      '--device-graph-bytes','65536','--trace-mode','compact','--device-upload-host-io','direct-preferred']
 actual={}
 for policy in ['retained','transient']:
  for quota in [65536,131072]:
   label=f'{name}-{policy}-{quota}';trace=a.output/(label+'.json')
   result=subprocess.run([*cmd,'--cache-bytes',str(quota),'--device-upload-host-policy',policy,'--trace-json',str(trace)],capture_output=True,text=True,timeout=30)
   (a.output/(label+'.stderr')).write_text(result.stderr);assert result.returncode==0,result.stderr[-3000:]
   t=json.loads(trace.read_text());actual[policy,quota]=t;m=t['metrics']
   assert m['cache_capacity_bytes']==quota and m['cache_peak_reserved_bytes']<=quota
   assert m['device_upload_host_policy']==policy and m['provider_io']['direct_requests']>0
   if cases or len(actual)>1:full.compare(actual['retained',65536],t,atol=0,rtol=0,compact=True)
   assert m['device_projection']['cache']['upload_bytes']==actual['retained',65536]['metrics']['device_projection']['cache']['upload_bytes'], 'host reuse unexpectedly changed device transfer sequence'
   cases+=1
 small=actual['retained',65536]['metrics'];large=actual['retained',131072]['metrics']
 assert large['provider_io']['direct_requests']<=small['provider_io']['direct_requests']
 if large['provider_io']['direct_requests']<small['provider_io']['direct_requests']:reuse_improvements+=1
 for quota in [65536,131072]:
  m=actual['transient',quota]['metrics'];assert m['cache_reserved_bytes']<8192,'transient expert retention was accidentally introduced'
 assert actual['transient',65536]['metrics']['provider_io']['direct_requests']==actual['transient',131072]['metrics']['provider_io']['direct_requests'], 'larger transient staging quota invented expert reuse'
 print(name,'retained direct requests',small['provider_io']['direct_requests'],'->',large['provider_io']['direct_requests'],
       'transient',actual['transient',65536]['metrics']['provider_io']['direct_requests'])
assert reuse_improvements,'fixtures did not exercise reusable-cache benefit'
print('Retained-host quota/byte-exact intermediate/routing/token/device-transfer parity:',cases,'cases; no GPU/trained model')
