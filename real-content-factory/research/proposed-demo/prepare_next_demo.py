"""Write local inactive fixture artifacts only; no token fit or execution."""
import hashlib
import json
import copy
from pathlib import Path
from next_demo_prompt import fixed_proposal,digest,make_prompt,validate_response,profile_manifest,raw_prompt_limit
from prompt_budget_audit import simulate,SOURCES


def initial_state():
    spec=fixed_proposal()
    return {'schema':1,'config':spec,'config_sha256':digest(spec),'evaluator_sha256':digest(spec['evaluator']),
            'status':'ready','reason':None,'wave':'initial',
            'tasks':[{'id':'seed-'+str(i),'kind':'generation','targets':[],'status':'pending',
                      'result':None,'result_sha256':None} for i in (1,2)],
            'candidates':[],'matches':[],'proximity':[],'meta_reviews':[],'overview':None,'incumbent_id':None,
            'model_calls':0,'experiment_calls':0,'elapsed_seconds':0.,'inflight':None,'cancel_requested':False,
            'mode':'fixture','actual_inference':False,'actual_experiments':False}


def main():
    root=Path(__file__).resolve().parent
    state=initial_state();prompt=make_prompt(state,state['tasks'][0])
    snapshots={}
    def capture(s,t):
        rendered=make_prompt(s,t);snapshots[digest(s)]=copy.deepcopy(s);return rendered
    report=simulate(renderer=capture,response_validator=validate_response,
                    review_bytes=192,debate_bytes=192,meta_bytes=192,rationale_bytes=96)
    for row in report['prompts']:
        row['raw_profile_limit']=raw_prompt_limit(row['role'])
        row['within_profile_envelope']=row['utf8_bytes']<=row['raw_profile_limit']
    report['profile']=profile_manifest()
    report['fixture_snapshots']=snapshots
    report['snapshot_storage_scope']='Serialized shadow fixture states; not real SQLite runs or execution evidence.'
    report['calibration_fixture']={'claim_scope':'EXACT_HANDWRITTEN_INITIAL_FIXTURE; NOT TOKEN_FIT OR MODEL_EXECUTION',
                                   'state_sha256':digest(state),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                                   'prompt_utf8_bytes':len(prompt.encode()),'raw_profile_limit':raw_prompt_limit('generation')}
    # Exercise all code and opinion caps, in addition to substantive fixtures.
    from next_demo_prompt import serialized_content_bytes
    padded=[]
    for source in SOURCES:
        value=source+'#';value+='x'*(640-serialized_content_bytes(value));padded.append(value)
    seen=[]
    def probe(s,t):
        seen.append(t['id']);return make_prompt(s,t)
    try:
        cap_report=simulate(renderer=probe,response_validator=validate_response,candidate_sources=tuple(padded),
                            review_bytes=192,debate_bytes=192,meta_bytes=192,rationale_bytes=96)
    except ValueError as error:
        report['all_response_caps_case']={'status':'blocked_without_clipping','task':seen[-1],'reason':str(error)}
    else:
        report['all_response_caps_case']={'status':'fixture_envelopes_passed_not_runtime_qualified',
                                         'code_serialized_content_bytes':640,'max_prompt_bytes':cap_report['max_prompt_bytes'],
                                         'prompts':cap_report['prompts']}
    files={'calibration_initial_fixture.json':json.dumps(state,indent=2)+'\n',
           'calibration_seed1_fixture.txt':prompt,
           'calibration_seed1_fixture.sha256':hashlib.sha256(prompt.encode()).hexdigest()+'  calibration_seed1_fixture.txt\n',
           'compact_prompt_audit.json':json.dumps(report,indent=2)+'\n'}
    for name,body in files.items():(root/name).write_text(body)
    print(json.dumps({'status':'INACTIVE_FIXTURES_PREPARED','max_prompt_bytes':report['max_prompt_bytes'],
                      'model_calls':0,'candidate_executions':0,'token_fit_run':False,
                      'calibration_fixture':report['calibration_fixture'],
                      'all_response_caps_case':report['all_response_caps_case'],'files':list(files)},indent=2))


if __name__=='__main__':main()
