"""CPU-only guards for the new long-running fixed C-2 execution."""
import unittest,importlib.util,os,hashlib,json
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
class Job:
 def __init__(self,status):self.status=status
 def poll(self):return self.status
def controller():
 with patch.dict(os.environ,{'PILOT_C2_ROOT':str(ROOT/'experiments/Pilot-C2')}):
  spec=importlib.util.spec_from_file_location('c2_controller_guard_test',ROOT/'src/pilot_c2/controller.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m
class C2Guards(unittest.TestCase):
 def test_reservation_prevents_race_before_CUDA_memory_appears(self):
  m=controller();c=m.Controller.__new__(m.Controller);c.assignments={Job(None):[1]}
  with patch.object(m,'gpus',return_value={1:{'memory':0,'util':0},2:{'memory':0,'util':0}}):
   self.assertFalse(c.idle(1));self.assertTrue(c.idle(2))
 def test_finished_child_does_not_override_external_busy_device(self):
  m=controller();c=m.Controller.__new__(m.Controller);c.assignments={Job(0):[1,2]}
  with patch.object(m,'gpus',return_value={1:{'memory':0,'util':0},2:{'memory':12000,'util':99}}):
   self.assertTrue(c.idle(1));self.assertFalse(c.idle(2))
 def test_accepted_core_budget_and_readout_algorithms_are_fixed(self):
  p=json.loads((ROOT/'configs/pretrain/Pilot_C2_protocol.json').read_text())
  self.assertEqual((p['initial_step'],p['final_step'],p['additional_updates_per_model']),(30000,45000,15000))
  self.assertEqual((p['lambda'],p['margin'],p['reference_quantile'],p['Chebyshev_strictly_greater_than'],p['layer']),(1,.05,.5,2,23))
  self.assertFalse(p['diagnostic_VJP']);self.assertFalse(p['diagnostic_extra_student_forward'])
  self.assertEqual(hashlib.sha256((ROOT/'src/pilot_c2/barrier.py').read_bytes()).hexdigest(),p['barrier_core_sha256'])
  for name in ['downstream.py','linear.py','benchmark.py','gpu_scoring.py']:
   accepted=(ROOT/'src/pilot_b'/name).read_text(encoding='utf-8');current=(ROOT/'src/pilot_c2'/name).read_text(encoding='utf-8')
   self.assertEqual(current,accepted.replace('from common import ','from system_common import '),name)
if __name__=='__main__':unittest.main()
