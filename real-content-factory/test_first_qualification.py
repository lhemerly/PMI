import dataclasses
import unittest
from unittest.mock import patch

from launch_policy import PlatformHeld
from research import qualification_authority as qa
from research.executor import build_plan


class Tests(unittest.TestCase):
    def scope(self, phase='package'):
        scope=object.__new__(qa._FirstCPUQualification)
        scope._phase=phase;scope.used=set();scope.inventory={'interval':{'worker.py':{'source':'fixture','sha256':'a'*64}}}
        return scope

    def test_no_flags_or_forged_authority_release_default_hold(self):
        with self.assertRaises(PlatformHeld):qa.gate_cpu(None,'package')
        for value in (True,{},object()):
            with self.assertRaises(ValueError):qa.gate_cpu(value,'package')
        scope=self.scope()
        with self.assertRaises(ValueError):qa.gate_cpu(scope,'package')

    def test_scope_phase_cannot_be_assigned_through_public_property(self):
        scope=self.scope()
        with self.assertRaises(AttributeError):scope.phase='candidate'
        with self.assertRaises(ValueError):scope.packaged({})

    def test_exact_packaging_reservation_is_single_use_and_cannot_expand(self):
        scope=self.scope();inventory=scope.inventory['interval']
        with patch.object(qa,'verify_sources',return_value={}),patch.object(qa,'_active',scope):
            with self.assertRaises(ValueError):qa.gate_cpu(scope,'package',worker='interval',destination='/tmp/other',inventory=inventory)
            with self.assertRaises(ValueError):qa.gate_cpu(scope,'package',worker='interval',destination=qa.INTERVAL_RUNTIME,inventory={})
            qa.gate_cpu(scope,'package',worker='interval',destination=qa.INTERVAL_RUNTIME,inventory=inventory)
            with self.assertRaises(ValueError):qa.gate_cpu(scope,'package',worker='interval',destination=qa.INTERVAL_RUNTIME,inventory=inventory)

    def test_admission_plan_cannot_expand_limits_or_command(self):
        scope=self.scope('bootstrap');root=qa.ROOT/'bootstrap'
        plan=build_plan(qa.PROBE_RUNTIME,root/'candidate',root/'artifacts',5,6,[{'probe':'bootstrap'}],(7,8))
        with patch.object(qa,'verify_sources',return_value={}),patch.object(qa,'_active',scope):
            for changed in (dataclasses.replace(plan,as_bytes=1024**3),dataclasses.replace(plan,argv=plan.argv+('extra',)),dataclasses.replace(plan,stdin=b'{}')):
                with self.assertRaises(ValueError):qa.gate_cpu(scope,'admission',plan=changed)
            qa.gate_cpu(scope,'admission',plan=plan)
            with self.assertRaises(ValueError):qa.gate_cpu(scope,'admission',plan=plan)

    def test_candidate_requires_completed_bootstrap_and_cleanup(self):
        from research.qualification_worker import EXPECTED_CHECKS
        capture={'exitcode':0,'stdout':b'{}','stderr':b'','namespace_isolation':True,'seccomp_verified':True,
                 'resource_limits_verified':True,'remaining_pids':[],'cleanup_verified':True,'timed_out':False}
        probe={'schema':'PMI_CPU_BOOTSTRAP_V1','probe':'bootstrap','candidate_executed':False,
               'checks':dict.fromkeys(EXPECTED_CHECKS,True),'observed':{}}
        scope=self.scope('bootstrap')
        with self.assertRaises(ValueError):scope.qualified(capture,probe,{'cleanup_verified':True})
        scope.used={('admission','bootstrap'),('transport','bootstrap')}
        with self.assertRaises(ValueError):scope.qualified(capture,probe,{'cleanup_verified':False})
        probe['checks']['socket_denied']=False
        with self.assertRaises(ValueError):scope.qualified(capture,probe,{'cleanup_verified':True})
        probe['checks']['socket_denied']=True
        scope.qualified(capture,probe,{'cleanup_verified':True})
        self.assertEqual(scope.phase,'candidate')

    def test_extra_entry_arguments_rejected_before_reservation(self):
        from research.first_qualification import main
        with patch('sys.argv',['qualification','--anything']),patch('research.first_qualification.activate_for_fixed_entry') as activate:
            with self.assertRaises(ValueError):main()
        activate.assert_not_called()


if __name__=='__main__':unittest.main()
