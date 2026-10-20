"""Continuation must reject lineage drift and share one explicit budget schedule."""
import copy,importlib.util,math,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('continuation_policy',Path(__file__).resolve().parents[1]/'src/d12_100k/policy.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)

class ContinuationTests(unittest.TestCase):
 def setUp(self):
  self.current={'experiment_id':'D1','manifest_sha256':'same_manifest','sampler_sha256':'same_sampler','config_sha256':'new_config','trainer_sha256':'new_trainer','total_steps':100000,'run_path':'new/D1/formal','world_size':3,'seed':0}
  self.saved={**self.current,'config_sha256':'old_config','trainer_sha256':'old_trainer','total_steps':30000,'run_path':'old/D1/formal'}
  self.receipt={**self.saved,'step':30000,'sha256':'full_state_sha'}
 def check(self,saved=None,receipt=None):policy.validate_parent(saved or self.saved,self.current,receipt or self.receipt,'old/D1/formal','old_config','old_trainer')
 def test_only_explicit_budget_path_and_code_transition_allowed(self):self.check()
 def test_mismatched_parent_cannot_resume(self):
  for key,value in [('manifest_sha256','different_pool'),('sampler_sha256','different_sampler'),('world_size',1),('seed',1),('experiment_id','D2'),('total_steps',29999),('run_path','other/D1/formal')]:
   with self.subTest(key=key):
    saved=copy.deepcopy(self.saved);saved[key]=value
    with self.assertRaises(ValueError):self.check(saved=saved)
 def test_receipt_must_match_actual_loaded_identity(self):
  receipt={**self.receipt,'seed':1}
  with self.assertRaises(ValueError):self.check(receipt=receipt)
 def test_shared_extended_schedule_declares_resume_lr_increase(self):
  peak=2.598076211353316e-5;minimum=1e-6
  self.assertAlmostEqual(policy.learning_rate(30000,peak,minimum),minimum+(peak-minimum)*(1+math.cos(math.pi*25000/95000))/2)
  self.assertGreater(policy.learning_rate(30000,peak,minimum),minimum*20)
  values=[policy.learning_rate(s,peak,minimum) for s in [30000,50000,75000,99999]]
  self.assertEqual(values,sorted(values,reverse=True));self.assertTrue(all(minimum<=v<=peak for v in values))
  with self.assertRaises(ValueError):policy.learning_rate(100000,peak,minimum)
if __name__=='__main__':unittest.main()
