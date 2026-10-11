"""Fixed first model-driven demonstration, submitted separately for launch review.

No arbitrary model, command, output directory, profile, or budget is accepted.
The original generic entry points remain held. An uncertain/consumed run is never
restarted automatically; receipts and the persisted queue are retained.
"""
import json
import sys
import time
from pathlib import Path

from .demo_authority import (ROOT,RUNTIME,RUNTIME_MANIFEST,CHECKOUT,
                             activate_for_fixed_entry,activate_child)


def _write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();__import__('os').fsync(stream.fileno())


class DemoCorrectnessRunner:
    actual_execution=True
    def __init__(self,authority,coordination):
        from .executor import BubblewrapCorrectnessRunner,verify_runtime
        from .owned_cpu import CPUAdmission,CPUOwnedTransport
        saved=json.loads(RUNTIME_MANIFEST.read_text())
        if set(saved)!={'root','files','runtime_sha256','primitive_sha256','packaging'} or Path(saved['root'])!=RUNTIME:
            raise ValueError('previous qualified minimal runtime manifest required')
        if verify_runtime(RUNTIME,saved['files'])!=saved['runtime_sha256']:
            raise ValueError('qualified minimal runtime changed')
        from .demo_authority import verify_integrity,hashed
        pin=verify_integrity()
        if pin.get('external_artifacts_sha256',{}).get(str(RUNTIME_MANIFEST))!=hashed(RUNTIME_MANIFEST):raise ValueError('qualified runtime manifest pin changed')
        self.authority=authority
        self.runner=BubblewrapCorrectnessRunner(RUNTIME,saved['files'],saved['primitive_sha256'],
            admission=CPUAdmission(authority=authority,coordination=coordination),
            transport=CPUOwnedTransport(authority=authority),authority=authority)

    def run(self,request,source,stage,deadline):
        receipt=self.runner.run(request,source,stage,deadline)
        self.authority.cpu_complete(receipt)
        return receipt


def child_main():
    from .deadline import read_supervisor_context
    from .coordination import CoordinationOwner,FACTORY_LEASE
    from .harness import DemoRealAgent
    from .laboratory import LaboratoryStore,demo_configuration,drive,summary
    report={'scope':'pmi-model-demo-v1','status':'blocked','actual_model_completions':0,
            'generated_candidate_receipts':0,'calibration':None,'workflow':None,
            'prior_native_attempts_charged_to_19_call_cap':1,
            'continuation_history':[
                {'root':'/tmp/pmi-model-demo-v1','model_reservations':0,'native_dispatches':0,'candidate_reservations':0,'reason':'import failure before reservation'},
                {'root':'/tmp/pmi-model-demo-v2','model_reservations':1,'native_dispatches':0,'candidate_reservations':0,'reason':'paging admission stop before native dispatch'},
                {'root':'/tmp/pmi-model-demo-v3','model_reservations':1,'native_dispatches':1,'candidate_reservations':0,'reason':'live-log reader race during pre-inference fitting; verified cleanup'}],
            'no_performance_measurement':True,'claim_scope':'engineering demonstration only; no paper reproduction or effectiveness claim'}
    authority=None
    with read_supervisor_context() as context:
        try:
            with CoordinationOwner() as coordination:
                context.retain_coordination(coordination)
                authority=activate_child(context)
                agent=DemoRealAgent(authority=authority,coordination=coordination)
                # Reuse the successful fixed minimal runtime. No new packaging or
                # bootstrap/candidate invocation is hidden in this scope.
                runner=DemoCorrectnessRunner(authority,coordination)
                prompt=(CHECKOUT/'research/proposed-demo/calibration_seed1_fixture.txt').read_text()
                result=agent.call('calibration','bonsai-stock',prompt,ROOT/'calibration',
                                  min(context.operation_deadline,time.monotonic()+120))
                authority.calibrated()
                report['calibration']={k:v for k,v in result.items() if k!='text'}
                store=LaboratoryStore(ROOT/'workflow');spec=demo_configuration()
                if spec!=json.loads((CHECKOUT/'research/model-demo-spec.json').read_text()):raise ValueError('fixed reviewed workflow specification changed')
                store.create(spec)
                state=drive(store,agent,runner,FACTORY_LEASE,coordination=coordination)
                report['workflow']=summary(state)
                demonstration=report['workflow'].get('demonstration',{})
                report['status']=('completed' if demonstration.get('completed_real_demonstration') is True
                                   and report['workflow']['objective_met'] is True else state['status'])
                if report['status']=='finished':report['status']='finished_without_full_demonstration'
        except BaseException as error:
            report['status']='blocked'
            report['error']=type(error).__name__+': '+str(error)
            if (ROOT/'workflow/laboratory.sqlite3').is_file():
                try:report['workflow']=summary(LaboratoryStore(ROOT/'workflow').load())
                except Exception as read_error:report['state_read_error']=type(read_error).__name__+': '+str(read_error)
        finally:
            # Count validated completion/experiment receipts even if the next
            # response-contract check blocks before the workflow can commit.
            completed=[];experiments=[]
            for path in ROOT.glob('**/result.json'):
                try:
                    row=json.loads(path.read_text())
                    if row.get('actual_inference') is True and row.get('status')=='RealTextAndReleasePass':completed.append(str(path.relative_to(ROOT)))
                except (OSError,ValueError):pass
            for path in ROOT.glob('**/experiment-receipt.json'):
                try:
                    row=json.loads(path.read_text())
                    if row.get('actual_execution') is True and row.get('cleanup_verified') is True:experiments.append(str(path.relative_to(ROOT)))
                except (OSError,ValueError):pass
            report['actual_model_completions']=len(completed)
            report['generated_candidate_receipts']=len(experiments)
            report['model_completion_receipts']=completed
            report['candidate_receipts']=experiments
            if authority is not None:
                report['model_stages_reserved']=authority.model_count
                report['candidate_stages_reserved']=authority.candidate_count
                report['uncertain_operation']=None if authority.current is None else {
                    k:v for k,v in authority.current.items() if k not in ('request','used')}
            _write(ROOT/'child-result.json',report)
            protocol_status=('finished' if report['status']=='completed' else
                             report['status'] if report['status'] in ('blocked','budget_exhausted','cancelled') else 'blocked')
            context.finish(protocol_status)
    return 0 if report['status']=='completed' else 2


def main(argv=None):
    args=sys.argv[1:] if argv is None else argv
    if args==['--supervised-child']:return child_main()
    if args:raise ValueError('fixed demo entry accepts no operator parameters')
    from .deadline import run_demo_supervisor
    authority=activate_for_fixed_entry()
    result=run_demo_supervisor(authority=authority)
    print(json.dumps(result,indent=2))
    return 0 if result.get('completed') is True else 2


if __name__=='__main__':raise SystemExit(main())
