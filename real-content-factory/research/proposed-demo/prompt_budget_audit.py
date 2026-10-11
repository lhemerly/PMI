"""Contract fixture only: no model, tokenizer, candidate, or executor invocation.

Read the proposal and extract only in-memory laboratory scheduling/prompt functions.
Fabricate clearly labeled experiment data to measure prompt byte growth. The shadow
pass deliberately records over-limit prompts and continues to expose later bounds;
the installed controller would stop at the first such prompt. Candidate code is
never compiled or executed. This script does not confer execution authority.
"""
import ast
from collections import Counter
import copy
import hashlib
import itertools
import json
from pathlib import Path
import re

REPOSITORY = Path('/home/luizh/Documents/Codex/2026-10-08/task-8/real-content-factory')
FUNCTIONS = frozenset(('enqueue','candidate','candidate_evidence','prompt','similarity',
                      'pairs','enqueue_match','pending_task','ratings_stable','schedule',
                      'source_digest','command_profile','apply'))
MODEL_ROLES = ('generation','reflection','debate','ranking','evolution','meta-review','overview')


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


class FixtureRequest:
    def __init__(self,task_id,candidate_id,source_sha256,evaluator_sha256,command_id,profile):
        self.values=dict(task_id=task_id,candidate_id=candidate_id,source_sha256=source_sha256,
                         evaluator_sha256=evaluator_sha256,command_id=command_id,profile=profile)
    def payload(self):return self.values


def fixture_evaluation(receipt,request,actual):
    if actual or receipt['actual_execution']:
        raise AssertionError('this audit accepts only fabricated fixture receipts')
    # Shapes and SHA lengths match the controller's receipt summary; these values
    # are synthetic, and no reference oracle or candidate evaluation is performed.
    return {'evaluator_sha256':request.values['evaluator_sha256'],'correct':True,
            'speedup':None,'objective_met':True,'receipt_sha256':digest(receipt)}


def load_functions(repository=REPOSITORY):
    raw=(repository/'research/laboratory.py').read_text()
    tree=ast.parse(raw)
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in FUNCTIONS]
    if {node.name for node in selected}!=FUNCTIONS:raise ValueError('audited functions missing')
    namespace={'itertools':itertools,'json':json,'re':re,'hashlib':hashlib,
               'text':lambda value:value,'encoded':encoded,'digest':digest,
               'ExperimentRequest':FixtureRequest,'validate_receipt':fixture_evaluation}
    # Only named in-memory functions are defined, not module imports, CLI, drive,
    # SQLite store, production agent, candidate program, or runtime executor.
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(repository/'research/laboratory.py'),'exec'),namespace)
    return namespace,hashlib.sha256(raw.encode()).hexdigest()


SOURCES=(
'''def merge_intervals(intervals):
    out=[]
    for a,b in sorted(intervals):
        if a==b: continue
        if out and a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)
        else: out.append([a,b])
    return out
''',
'''def merge_intervals(intervals):
    events=[]
    for a,b in intervals:
        if a<b:
            events.append((a,1))
            events.append((b,-1))
    events.sort()
    out=[]
    active=0
    i=0
    while i<len(events):
        point=events[i][0]
        before=active
        while i<len(events) and events[i][0]==point:
            active+=events[i][1]
            i+=1
        if before==0 and active>0: start=point
        if before>0 and active==0: out.append([start,point])
    return out
''',
'''def merge_intervals(intervals):
    ordered=sorted([list(pair) for pair in intervals if pair[0]<pair[1]])
    result=[]
    for pair in ordered:
        if result and pair[0]<=result[-1][1]:
            if pair[1]>result[-1][1]: result[-1][1]=pair[1]
        else: result.append(pair)
    return result
''')

REFLECTION='Fixture opinion: check empty, touching, nested and duplicate intervals; the receipt is synthetic and cannot establish actual correctness.'
DEBATE='Fixture debate: compare endpoint aggregation with sorted merging; preserve touching boundaries and discard zero-length intervals.'
META='Fixture meta-review: preserve the immutable input and output contract; address empty intervals and touching endpoints before claiming correctness.'
RATIONALE='Fixture preference only; no measured algorithm superiority.'


def fixture_text(base,size):
    return (base+' ')*((size//(len(base)+1))+1) if size is None else ((base+' ')*((size//(len(base)+1))+1))[:size]


def simulate(*,repository=REPOSITORY,duplicate=False,review_bytes=240,debate_bytes=160,meta_bytes=240,rationale_bytes=80,
             renderer=None,response_validator=None,candidate_sources=SOURCES):
    api,source_sha=load_functions(repository)
    proposal_path=repository/'research/end-to-end-proposal.json'
    spec=json.loads(proposal_path.read_text())
    if spec['options']['proximity']!='token-jaccard' or spec['initial_candidates']!=2 or spec['evolutions']!=1:
        raise ValueError('this audit is limited to the exact two-seed lexical proposal')
    state={'config':spec,'evaluator_sha256':digest(spec['evaluator']),'status':'ready','wave':'initial',
           'tasks':[],'candidates':[],'matches':[],'proximity':[],'meta_reviews':[],'overview':None,
           'incumbent_id':None,'mode':'fixture','inflight':None}
    for index in range(2):api['enqueue'](state,'seed-'+str(index+1),'generation')
    rows=[];counts=Counter();first_overflow=None
    for _ in range(100):
        task=api['pending_task'](state)
        if task is None:
            api['schedule'](state)
            if state['status']=='finished':break
            continue
        kind=task['kind'];counts[kind]+=1
        if kind in MODEL_ROLES:
            rendered=(renderer or api['prompt'])(state,task);length=len(rendered.encode())
            row={'task':task['id'],'role':kind,'utf8_bytes':length,'within_4096':length<=4096,
                 'prompt_sha256':hashlib.sha256(rendered.encode()).hexdigest()}
            rows.append(row)
            if length>4096 and first_overflow is None:first_overflow=copy.deepcopy(row)
        if kind in ('generation','evolution'):
            result=candidate_sources[0 if duplicate else len(state['candidates'])]
        elif kind=='experiment':
            result={'actual_execution':False,'fixture':True,'measurements':{'tests_passed':884,'tests_total':884,
                    'baseline_ns':None,'candidate_ns':None,'baseline_sha256':None}}
        elif kind=='reflection':result=fixture_text(REFLECTION,review_bytes)
        elif kind=='debate':result=fixture_text(DEBATE,debate_bytes)
        elif kind=='ranking':result=encoded({'winner':task['targets'][0],'rationale':fixture_text(RATIONALE,rationale_bytes)})
        elif kind=='meta-review':result=fixture_text(META,meta_bytes)
        elif kind=='overview':result='Fixture overview only; no execution, effectiveness or scientific novelty claim.'
        elif kind=='proximity':result=None
        else:raise AssertionError(kind)
        if response_validator is not None and kind in MODEL_ROLES:
            response_validator(kind,result,targets=task['targets'])
        api['apply'](state,task,result)
    else:raise AssertionError('shadow scheduling did not terminate')
    return {'claim_scope':('CONTRACT_FIXTURE_ONLY; compact planner only, no runtime/fit proof' if renderer else
                          'CONTRACT_FIXTURE_ONLY; unbounded shadow prompt measurement, not a runnable passing loop'),
            'actual_model_calls':0,'actual_candidate_executions':0,'token_fit_run':False,
            'lab_source_sha256':source_sha,'proposal_sha256':hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
            'candidate_source_bytes':[len(v.encode()) for v in candidate_sources],
            'fixture_reply_bytes':{'reflection':review_bytes,'debate':debate_bytes,'meta-review':meta_bytes,'rationale':rationale_bytes},
            'all_candidates_exact_duplicate':duplicate,'shadow_role_counts':dict(counts),
            'shadow_model_calls':sum(counts[role] for role in MODEL_ROLES),
            'shadow_experiment_calls':counts['experiment'],'max_prompt_bytes':max(row['utf8_bytes'] for row in rows),
            'first_real_controller_byte_block':first_overflow,'prompts':rows}


def main():
    scenarios={'compact':simulate(review_bytes=80,debate_bytes=64,meta_bytes=80,rationale_bytes=48),
               'realistic_short':simulate(),
               'duplicates':simulate(duplicate=True)}
    print(json.dumps(scenarios,indent=2))


if __name__=='__main__':main()
