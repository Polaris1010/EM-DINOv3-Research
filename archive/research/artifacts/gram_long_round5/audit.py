import os,json,math
from pathlib import Path
from omegaconf import OmegaConf
BASE=Path(__file__).resolve().parent

def main():
 import sys
 prefix=sys.argv[1];g=BASE/'jobs'/f'{prefix}_G_functional';c=BASE/'jobs'/f'{prefix}_C_functional'
 a=OmegaConf.to_container(OmegaConf.load(g/'resolved_config.yaml'));b=OmegaConf.to_container(OmegaConf.load(c/'resolved_config.yaml'));assert a['gram']['loss_weight_schedule']['end']==2 and b['gram']['loss_weight_schedule']['end']==0
 b['gram']['loss_weight_schedule']['end']=2;assert a==b
 for k,v in dict(use_loss=True,normalized=True,img_level=True,tokens_used='all',remove_neg=False,remove_only_teacher_neg=False,rep_update=True,update_frequency=10000,it_first_update=1010000,max_updates=3).items():assert a['gram'][k]==v,(k,a['gram'][k])
 assert a['crops']['gram_teacher_crops_size']==512 and a['crops']['global_crops_size']==256 and a['crops']['local_crops_size']==112 and a['crops']['local_crops_number']==8 and a['crops']['gram_teacher_no_distortions']
 for path,weight in [(g,2),(c,0)]:
  rows=[json.loads(x) for x in (path/'steps.jsonl').read_text().splitlines()]
  for row in rows:
   coeff=row['gram_backward_coefficient'];recorded=row['gram_loss_weight'];assert math.isclose(coeff,recorded,rel_tol=1e-6,abs_tol=1e-10)
   assert math.isfinite(row['total']) and math.isfinite(row['grad_norm'])
   if weight==0:assert coeff==0
  if weight:assert rows[-1]['gram_backward_coefficient']>0
 updates=[]
 for completed in range(1,30001):
  virtual=1000000+completed
  if virtual>=a['gram']['it_first_update'] and virtual%a['gram']['update_frequency']==0 and len(updates)<a['gram']['max_updates']:updates.append(completed)
 assert updates==[10000,20000,30000]
 result=dict(anchor_refresh_schedule=updates,status='complete',only_config_difference='gram.loss_weight_schedule.end',gram_semantics_official=True,actual_backward_weight_matches=True,gradient_finite=True)
 (BASE/'gram_config_audit.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
