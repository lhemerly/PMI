#!/usr/bin/env python3
"""Read bounded progress JSONL; tolerate only a partial final line after a kill."""
import argparse,json
from pathlib import Path
MAX_BYTES=32*1024*1024

def analyze(path):
 result={'complete':False,'partial_tail':False,'details_truncated':False,'records':0,
         'prefill_complete':False,'first_output_complete':False,'completed_prefill_evaluations':0,
         'completed_decode_evaluations':0,'last_completed_evaluation':None,'last_cache_event':None,
         'last_record':None,'last_active_evaluation':None,'interruption_cause':None}
 if path.stat().st_size>MAX_BYTES:raise ValueError('progress file exceeds bound')
 previous=-1; consumed=0; source=None
 with path.open('rb') as f:
  while True:
   line=f.readline(32769)
   if not line:break
   consumed+=len(line)
   if consumed>MAX_BYTES:raise ValueError('progress grew beyond bound')
   if len(line)>32768:raise ValueError('progress line exceeds bound')
   if not line.endswith(b'\n'):
    if result['complete']:raise ValueError('tail follows completion')
    if f.read(1):raise ValueError('oversized partial tail')
    result['partial_tail']=True;break
   row=json.loads(line)
   if not isinstance(row,dict) or row.get('schema')!='pmi.progress.v1' or type(row.get('sequence'))!=int or row.get('sequence')!=result['records']:raise ValueError('progress schema/sequence')
   stamp=row.get('monotonic_ns')
   if type(stamp)!=int or stamp<previous:raise ValueError('progress clock order')
   if result['complete']:raise ValueError('records follow completion')
   previous=stamp;result['records']+=1;result['last_record']=row
   phase=row['phase']
   if phase=='run_complete':result['complete']=True;result['details_truncated']=row['details_truncated']
   elif phase=='detail_limit':result['details_truncated']=True
   elif phase=='evaluation_begin':result['last_active_evaluation']={k:row[k] for k in ['evaluation','prefill','input_tokens']}
   elif phase=='evaluation_complete':
    e=row['metrics'];result['last_completed_evaluation']=e;result['last_active_evaluation']=None
    result['completed_'+e['phase']+'_evaluations']+=1
   elif phase=='prefill_complete':result['prefill_complete']=True
   elif phase=='first_output_complete':result['first_output_complete']=True
   elif 'request' in row:
    if phase=='cache_request_begin':source={k:row[k] for k in ['request','shard','offset','bytes']}
    result['last_cache_event']=dict(row)
    if source and source['request']==row['request']:result['last_cache_event']['source']=source
 result['last_cache_event_may_be_stale']=result['details_truncated']
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 data=json.dumps(analyze(a.input),indent=2)+'\n'
 if a.output:
  with a.output.open('x')as f:f.write(data)
 else:print(data,end='')
