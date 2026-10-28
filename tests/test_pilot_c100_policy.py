"""CPU regression checks for full-state45k continuation and original anchor."""
import unittest,importlib.util,json,copy,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pilot_c100_policy_test',ROOT/'src/pilot_c100/continuation_policy.py');policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
class ParentContract(unittest.TestCase):
 def setUp(self):self.p=json.loads((ROOT/'configs/pretrain/Pilot_C100_protocol.json').read_text())
 def parent(self,model):
  return {'step':45000,'identity':{'stage':'Pilot-C2','model':self.p['parents'][model]['model_45k'],'initial_step':30000,'final_step':45000,'barrier_enabled':model=='C1_Barrier_100k','protocol_sha256':self.p['parent_45k_protocol_sha256'],'source_commit':self.p['parent_45k_source_commit'],'parent_full_sha256':self.p['parent_full_sha256'],'global_batch':48,'world_size':3,'legacy_config_sha256':self.p['legacy_SSL_config_sha256']},'rng_by_rank':[{}, {}, {}],'sample_chain':self.p['initial_sample_chain'],'optimizer':{'state':{1:{'step':45000},2:{'step':45000}}},'student':{},'teacher':{},'dino_loss':{},'ibot_loss':{}}
 def test_each_arm_accepts_its_own45k_full_state(self):
  for model in policy.MODELS:policy.validate_parent(self.parent(model),model,self.p)
 def test_swapped45k_parent_is_rejected(self):
  with self.assertRaises(AssertionError):policy.validate_parent(self.parent(policy.MODELS[1]),policy.MODELS[0],self.p)
 def test_wrong_anchor_or_restarted_schedule_budget_is_rejected(self):
  for key,value in [('anchor_EMA_relative','codex/changed/step000045000/teacher_backbone.pth'),('parent_EMA_sha256','wrong'),('initial_step',30000),('final_step',155000)]:
   p=copy.deepcopy(self.p);p[key]=value
   with self.assertRaises(AssertionError):policy.validate_protocol(p)
 def test_stale_or_mixed_optimizer_steps_are_rejected(self):
  for steps in [(30000,30000),(45000,45001)]:
   saved=self.parent(policy.MODELS[0])
   for state,step in zip(saved['optimizer']['state'].values(),steps):state['step']=step
   with self.assertRaises(AssertionError):policy.validate_parent(saved,policy.MODELS[0],self.p)
 def test_boundary_rejects_wrong_chain_rng_or_step(self):
  old={'step':45000,'sample_chain':'chain','after_model_RNG':'rng'}
  policy.validate_boundary(old,'chain','rng',[45000])
  for chain,rng,steps in [('wrong','rng',[45000]),('chain','wrong',[45000]),('chain','rng',[30000])]:
   with self.assertRaises(AssertionError):policy.validate_boundary(old,chain,rng,steps)
 def test_legacy_policy_import_is_not_shadowed(self):
  import tempfile,subprocess,sys
  with tempfile.TemporaryDirectory() as folder:
   d=Path(folder);(d/'policy.py').write_text('learning_rate = 123\n',encoding='utf-8')
   script='import sys;sys.path[:0]='+repr([str(ROOT/'src/pilot_c100'),str(d)])+';import policy;from continuation_policy import validate_parent;assert policy.learning_rate==123;assert callable(validate_parent)'
   subprocess.run([sys.executable,'-c',script],check=True,capture_output=True)
 def test_accepted_loss_and_readouts_are_byte_identical(self):
  for name in ['barrier.py','production_adapter.py','downstream.py','linear.py','benchmark.py','gpu_scoring.py']:
   self.assertEqual((ROOT/'src/pilot_c100'/name).read_bytes(),(ROOT/'src/pilot_c2'/name).read_bytes(),name)
  self.assertEqual(hashlib.sha256((ROOT/'src/pilot_c100/barrier.py').read_bytes()).hexdigest(),self.p['barrier_core_sha256'])
if __name__=='__main__':unittest.main()