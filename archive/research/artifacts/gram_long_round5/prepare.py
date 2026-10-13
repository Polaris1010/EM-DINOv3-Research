import os,sys,json,csv,hashlib,collections,subprocess,re,time
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=Path('/opt/data/3_117/tanghr');R4=ROOT/'codex/mitoverse_round4_20260909_103239';SRC=ROOT/'DINOv3/dinov3'
import numpy as np,torch
P1=ROOT/'DINOv3/med_checkpoints_continual_v2/eval/training_124999/teacher_checkpoint.pth'
EXPECTED='53f759c7019f8753d9d18b52aa79168b51b7480466934df47f46bc3d5f696524'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def save(n,x):(BASE/n).write_text(json.dumps(x,indent=2,default=str))
def main():
 assert sha(P1)==EXPECTED
 cachep=ROOT/'DINOv3/pretrain_data/patches_512_format/train/fast_cache.pth';cache=torch.load(cachep,map_location='cpu',weights_only=False)
 allowed={'jrc_cos7-1a','jrc_cos7-1b','jrc_jurkat-1','jrc_mus-kidney','jrc_mus-liver','jrc_macrophage-2','jrc_ctl-id8-1','jrc_sum159-1'}
 rows=[];excluded=collections.Counter()
 for p,c in cache['samples']:
  rel=p.split('/train/',1)[1];parts=rel.split('/');source=parts[1] if len(parts)>2 else 'unknown'
  if parts[0]!='JRC_Unlabeled_New' or source not in allowed:
   excluded[(parts[0],source)]+=1;continue
  m=re.match(r'z(\d+)_y(\d+)_x(\d+)\.png$',parts[-1]);assert m,parts[-1]
  rows.append(dict(index=len(rows),source=source,volume='/'.join(parts[1:-1]),z=int(m[1]),y=int(m[2]),x=int(m[3]),path=p))
 assert len(rows)>1000
 with (BASE/'ssl_dataset_manifest.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (BASE/'ssl_exclusion_report.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['pool','source','excluded_count','reason'])
  for (pool,source),n in sorted(excluded.items()):w.writerow([pool,source,n,'Only explicitly named JRC sources with no known identity match to round4 held-out sources retained; MitoEM/Lucchi excluded; anonymous/ambiguous crops excluded'])
 split=json.load(open(R4/'split_mitoverse_probe_v1.json'));heldout={split['volumes'][v]['source'] for k in ['validation','test1','test2','test3'] for v in split[k]}
 assert not heldout & allowed
 sources=sorted(set(r['source'] for r in rows));pools={s:[r['index'] for r in rows if r['source']==s] for s in sources}
 rng=np.random.default_rng(0);arr=np.lib.format.open_memmap(BASE/'ssl_schedule.npy',mode='w+',dtype=np.int64,shape=(480000,2))
 with (BASE/'ssl_schedule_seed0.jsonl').open('w') as f:
  for i in range(len(arr)):
   s=sources[int(rng.integers(len(sources)))];j=pools[s][int(rng.integers(len(pools[s])))];a=int(rng.integers(1,2**31));arr[i]=[j,a];r=rows[j]
   f.write(json.dumps(dict(sample=i,manifest_index=j,source=s,volume=r['volume'],z=r['z'],y=r['y'],x=r['x'],augmentation_seed=a),separators=(',',':'))+'\n')
 arr.flush();del arr
 save('locked_inputs.json',{'P1':str(P1),'P1_sha256':EXPECTED,'anchor':str(P1),'anchor_purpose':'Prevent degradation relative to P1; NOT restoration to a known superior early teacher. High-res teacher still supplies distinct patch relations.','cache':str(cachep),'cache_sha256':sha(cachep),'ssl_rows':len(rows),'source_counts':{s:len(pools[s]) for s in sources},'heldout_sources':sorted(heldout),'overlap_policy':'Exclude all MitoEM/Lucchi and all unapproved/anonymous image pools. Named retained JRC specimens are outside current MitoVerse train/validation/test selections; current OpenOrganelle was excluded in round4. Historical P1 exposure remains unchanged; this is a development holdout experiment, not pristine unseen evaluation.','manifest_sha256':sha(BASE/'ssl_dataset_manifest.csv'),'schedule_sha256':sha(BASE/'ssl_schedule_seed0.jsonl'),'original_P1_optimizer_available':False,'SSL_seed':0,'target_steps':30000,'head_seeds':[0,1,2],'X1_head_lr':1e-4,'X1_last4_lr':3e-6,'X2_X3':'not authorized in this round','anchor_refresh_effect':'after10k and20k affect subsequent training; after30k is terminal bookkeeping only','temperature':.07,'lr_policy':'Virtual iteration1M; official LR and temperature warmups already elapsed; batch16 LR1.5e-5, batch8 LR1.0606601717798212e-5; no new temperature reset','lambda_policy':'official cosine ramp 0->2 during local first1000 steps; C always0','stop_policy':'Stop immediately on fault, best effort checkpoint. Resume from common validated checkpoint, never continue waiting for a boundary under resource pressure.'})
 (BASE/'ssl_schedule_sha256.txt').write_text(sha(BASE/'ssl_schedule_seed0.jsonl')+'\n')
 state={}
 for p in [SRC,ROOT/'Med_DINOv3/MedDINOv3']:
  state[str(p)]={k:subprocess.check_output(['git','-C',str(p)]+cmd,text=True) for k,cmd in [('commit',['rev-parse','HEAD']),('dirty',['status','--porcelain'])]}
 save('repository_state.json',state)
 (BASE/'jobs').mkdir(exist_ok=True)
 save('prepare_complete.json',{'status':'complete','rows':len(rows),'sources':sources})
 print('PREPARED',len(rows),sources,flush=True)
if __name__=='__main__':main()
