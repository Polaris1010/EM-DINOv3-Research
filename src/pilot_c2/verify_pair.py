"""Full trajectory pairing and production participation; never score GT here."""
from pathlib import Path
import os,json,math,hashlib
R=Path(os.environ['PILOT_C2_ROOT']);MODELS=['C0_Vanilla_45k','C1_Barrier_45k']
def read(p):return json.loads(p.read_text())
def rows(p):
 result={};count=0
 with p.open() as f:
  for line in f:
   x=json.loads(line);result[x['step']]=x;count+=1
 return result,count
def main():
 comparisons=0;replayed={};parity={};chains=set()
 c1root=R.parent/'pilot_c1_barrier_preflight_20261027'
 for rank in range(3):
  d={};counts={}
  for model in MODELS:
   d[model],counts[model]=rows(R/'runs'/model/f'steps_rank{rank}.jsonl')
   assert set(d[model])==set(range(30001,45001))
   done=read(R/'runs'/model/f'complete_rank{rank}.json');assert done['additional_updates']==15000 and done['optimizer_steps']==[45000] and not done['extra_VJP']
   replayed[f'{model}_rank{rank}']=counts[model]-15000;chains.add(done['sample_chain'])
  for step in range(30001,45001):
   a,b=d[MODELS[0]][step],d[MODELS[1]][step]
   for key in ['first_index','sample_digest','sample_chain','tensor_hashes','before_model_RNG','after_model_RNG','lr','weight_decay','global_batch']:assert a[key]==b[key],(rank,step,key)
   assert b['barrier_backward_seen'] and b['barrier_upstream_scale']==1 and b['student_forward_calls']==1
   assert not b['diagnostic_VJP'] and not b['diagnostic_extra_student_forward'] and not b['anchor_gradients_present']
   assert all(math.isfinite(b[k]) for k in ['barrier_raw','loss','grad_norm','student_patch_total_gradient_norm'])
   comparisons+=1
  for branch,model in zip(['C0','C1'],MODELS):
   old=[json.loads(x) for x in (c1root/'smoke'/branch/f'steps_rank{rank}.jsonl').read_text().splitlines()]
   pairs=[(x,d[model][x['step']]) for x in old]
   assert all(x['sample_chain']==y['sample_chain'] and x['tensor_hashes']==y['tensor_hashes'] and x['before_model_RNG']==y['before_model_RNG'] and x['after_model_RNG']==y['after_model_RNG'] for x,y in pairs)
   parity[f'{branch}_rank{rank}']={'loss_max_abs_difference':max(abs(x['loss']-y['loss']) for x,y in pairs),'Barrier_max_abs_difference':max(abs(x.get('barrier_raw',0)-y.get('barrier_raw',0)) for x,y in pairs)}
 assert len(chains)==1
 result={'status':'PASS','paired_rank_steps':comparisons,'effective_updates_per_model':15000,'replayed_rows_after_resume':replayed,'final_sample_chain':next(iter(chains)),'all_inputs_RNG_schedules_equal':True,'production_Barrier_participates_every_step':True,'extra_diagnostic_forward_or_VJP':False,'accepted_C1_first20_parity':parity,'GT_read':False}
 (R/'paired_SSL_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
