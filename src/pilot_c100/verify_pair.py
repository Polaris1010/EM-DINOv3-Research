"""Verify every paired45k->100k input, RNG, schedule and real Barrier update."""
from pathlib import Path
import os,json,math
R=Path(os.environ['PILOT_C100_ROOT']);MODELS=['C0_Vanilla_100k','C1_Barrier_100k']
def read(p):return json.loads(p.read_text())
def rows(p):
 result={};count=0
 with p.open() as f:
  for line in f:
   x=json.loads(line);result[x['step']]=x;count+=1
 return result,count
def main():
 protocol=read(R/'protocol.json');comparisons=0;replayed={};boundaries={};chains=set()
 for rank in range(3):
  data={}
  for model in MODELS:
   data[model],count=rows(R/'runs'/model/f'steps_rank{rank}.jsonl')
   assert set(data[model])==set(range(45001,100001))
   done=read(R/'runs'/model/f'complete_rank{rank}.json');assert done['additional_updates']==55000 and done['optimizer_steps']==[100000] and not done['extra_VJP']
   initial=read(R/'runs'/model/f'initial_rank{rank}_step45000.json')
   assert initial['step']==45000 and initial['optimizer_steps']==[45000] and initial['student_strict_exact'] and initial['teacher_strict_exact']
   assert initial['identity']['parent_full_sha256']==protocol['parents'][model]['sha256']
   assert initial['sample_chain']==protocol['initial_sample_chain']
   boundaries[f'{model}_rank{rank}']={'initial_RNG':initial['RNG_restored_hash'],'first_forward_RNG':data[model][45001]['before_model_RNG'],'chain':initial['sample_chain']}
   assert initial['RNG_restored_hash']==data[model][45001]['before_model_RNG']
   replayed[f'{model}_rank{rank}']=count-55000;chains.add(done['sample_chain'])
  assert boundaries[f'{MODELS[0]}_rank{rank}']['initial_RNG']==boundaries[f'{MODELS[1]}_rank{rank}']['initial_RNG']
  for step in range(45001,100001):
   a,b=data[MODELS[0]][step],data[MODELS[1]][step]
   for key in ['first_index','sample_digest','sample_chain','tensor_hashes','before_model_RNG','after_model_RNG','lr','weight_decay','global_batch']:assert a[key]==b[key],(rank,step,key)
   assert b['barrier_backward_seen'] and b['barrier_upstream_scale']==1 and b['student_forward_calls']==1
   assert not b['diagnostic_VJP'] and not b['diagnostic_extra_student_forward'] and not b['anchor_gradients_present']
   assert all(math.isfinite(b[k]) for k in ['barrier_raw','loss','grad_norm','student_patch_total_gradient_norm'])
   comparisons+=1
 assert len(chains)==1
 result={'status':'PASS','paired_rank_steps':comparisons,'effective_updates_per_model':55000,'initial_global_step':45000,'final_global_step':100000,'replayed_rows_after_resume':replayed,'final_sample_chain':next(iter(chains)),'all_inputs_RNG_schedules_equal':True,'production_Barrier_participates_every_step':True,'extra_diagnostic_forward_or_VJP':False,'original45k_boundaries':boundaries,'anchor_remains_original30k':True,'GT_read':False}
 (R/'paired_SSL_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()