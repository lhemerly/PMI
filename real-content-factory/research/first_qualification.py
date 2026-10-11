"""Fixed, one-shot CPU qualification. Submit this entry to platform review.

No arguments, installs, model calls, network, GPU or performance measurements.
Failure consumes this reservation; it cannot silently retry or broaden scope.
"""
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path
import time

from .qualification_authority import (CHECKOUT, ROOT, LEASE, PROBE_RUNTIME,
                                      INTERVAL_RUNTIME, PRIOR_REPORT, PRIOR_REPORT_SHA256, activate_for_fixed_entry)


def save_report(report):
    temporary = ROOT/'report.json.tmp'
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(ROOT/'report.json')


def main():
    if len(sys.argv) != 1:
        raise ValueError('fixed qualification entry accepts no arguments')
    from .runtime_bundle import build_runtime
    from .executor import build_plan, seccomp_x86_64, verify_capture, run_once
    from .owned_cpu import CPUAdmission, CPUOwnedTransport
    from .qualification_worker import EXPECTED_CHECKS
    authority = activate_for_fixed_entry()
    report = {'status': 'reserved', 'attempt': 2, 'automatic_retry': False,
              'prior_failed_report': {'path': str(PRIOR_REPORT), 'sha256': PRIOR_REPORT_SHA256}, 'scope': 'one authored interval candidate; CPU correctness only',
              'checks_passed': [], 'checks_failed': [], 'checks_not_run': [
                  'hostile resource-exhaustion probes', 'supervisor-parent-death probe',
                  'arbitrary command qualification', 'models', 'network access', 'GPU', 'performance benchmark'],
              'model_calls': 0, 'candidate_executed': False, 'actual_correctness': None,
              'general_gate_sha256': hashlib.sha256((CHECKOUT/'launch_policy.py').read_bytes()).hexdigest(),
              'interpretation': 'bootstrap evidence and bounded correctness only; no paper reproduction or research-effectiveness claim'}
    save_report(report)
    transport = None
    try:
        with LEASE.open('a+b') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifests = {}
            for role, destination in (('qualification', PROBE_RUNTIME), ('interval', INTERVAL_RUNTIME)):
                manifest = build_runtime(destination, authority.inventory[role], worker=role, authority=authority)
                manifests[role] = manifest
                (ROOT/('interval-manifest.json' if role == 'interval' else 'bootstrap-manifest.json')).write_text(json.dumps(manifest, indent=2)+'\n')
            report['checks_passed'].append('explicit hash-pinned minimal runtime packaging')
            report['runtime_sha256'] = {role: value['runtime_sha256'] for role,value in manifests.items()}
            authority.packaged(manifests)
            stage = ROOT/'bootstrap'; stage.mkdir(mode=0o700)
            for name in ('candidate','artifacts'): (stage/name).mkdir(mode=0o700)
            (stage/'seccomp.bpf').write_bytes(seccomp_x86_64())
            transport = CPUOwnedTransport(authority=authority)
            admission = CPUAdmission(authority=authority)
            with (stage/'seccomp.bpf').open('rb') as filter_file, (stage/'namespace-status.jsonl').open('x+b') as status:
                handshake = os.pipe()
                try:
                    plan = build_plan(PROBE_RUNTIME, stage/'candidate', stage/'artifacts', filter_file.fileno(), status.fileno(), [{'probe':'bootstrap'}], handshake)
                    deadline = time.monotonic()+30
                    try:
                        admitted = admission.qualify(plan, deadline)
                        if not admitted['approved'] or not time.monotonic() < admitted['expires_at'] <= deadline:
                            raise ValueError('bootstrap admission expired/refused')
                        report['bootstrap_admission'] = admitted
                        capture = verify_capture(transport.run(plan, deadline))
                    finally:
                        admission.release()
                finally:
                    for fd in handshake: os.close(fd)
            (stage/'stdout.txt').write_bytes(capture['stdout'])
            (stage/'stderr.txt').write_bytes(capture['stderr'])
            if capture['exitcode'] != 0:
                raise RuntimeError('bootstrap nonzero exit: '+capture['stderr'].decode('utf-8',errors='replace'))
            probe = json.loads(capture['stdout'])
            if (set(probe) != {'schema','probe','candidate_executed','checks','observed'} or probe['schema'] != 'PMI_CPU_BOOTSTRAP_V1'
                    or probe['probe'] != 'bootstrap' or probe['candidate_executed'] is not False
                    or set(probe['checks']) != set(EXPECTED_CHECKS)
                    or any(value is not True for value in probe['checks'].values())):
                raise ValueError('bootstrap confinement/limits check failed: '+json.dumps(probe))
            report['bootstrap'] = probe
            report['checks_passed'].extend(sorted(probe['checks']))
            report['checks_passed'].append('owned bootstrap namespace cleanup')
            report['bootstrap_cleanup'] = transport.last_cleanup
            report['status'] = 'bootstrap_passed'; save_report(report)
            authority.qualified(capture, probe, transport.last_cleanup)
            # The outer shared factory lease remains held. run_once uses a
            # dedicated inner lease, avoiding a second acquisition of the same lock.
            report['status'] = 'candidate_reserved'
            report['candidate_executed'] = None  # crash-safe: execution outcome unconfirmed
            save_report(report)
            result = run_once(CHECKOUT/'research/interval_candidate.py', ROOT/'interval-manifest.json',
                              ROOT/'correctness', ROOT/'candidate.lease', authority=authority)
            report['candidate_executed'] = result['receipt']['actual_execution']
            report['actual_correctness'] = result['receipt']['correctness_details']
            report['checks_passed'].append('owned candidate cleanup')
            if result['evaluation']['correct']:
                report['checks_passed'].append('authored interval program: 884 frozen cases')
                report['status'] = 'correctness_passed'
            else:
                report['checks_failed'].append('authored interval program correctness')
                report['status'] = 'correctness_failed'
    except BaseException as error:
        report['status'] = 'blocked'
        report['checks_failed'].append(type(error).__name__+': '+str(error))
        if transport is not None: report['bootstrap_cleanup'] = transport.last_cleanup
        if ('transport','candidate') in authority.used:
            report['candidate_executed'] = None
            report['candidate_launch_status'] = 'unconfirmed; transport reserved; inspect durable stage'
        else:
            report['candidate_executed'] = False
            report['checks_not_run'].append('candidate correctness: qualification did not admit it')
        save_report(report)
        print(json.dumps(report, indent=2))
        return 2
    save_report(report)
    print(json.dumps(report, indent=2))
    return 0 if report['status']=='correctness_passed' else 3


if __name__ == '__main__':
    raise SystemExit(main())
