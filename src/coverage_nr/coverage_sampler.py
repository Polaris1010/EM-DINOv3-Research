"""Fresh-P0 full-run ID plan; used by both CPU verification and the real loader."""
import hashlib,json
from pathlib import Path
import numpy as np
from historical_sampler import HistoricalSampler

BATCH=48
STEPS=100000

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def rng(tag):
 seed=int.from_bytes(hashlib.sha256(('Coverage-NR-P0-v1|'+tag).encode()).digest()[:16],'little')
 return np.random.Generator(np.random.PCG64(seed))

def quotas(cem,budget):
 sizes=np.array([g['images'] for g in cem['groups']],np.int64)
 if budget<int(sizes.sum()):raise ValueError('Not enough CEM slots for one full pass')
 p=np.sqrt(sizes);p/=p.sum();raw=(budget-int(sizes.sum()))*p
 extra=np.floor(raw).astype(np.int64)
 order=sorted(range(len(sizes)),key=lambda i:(-float(raw[i]-extra[i]),cem['groups'][i]['name']))
 for i in order[:budget-int(sizes.sum())-int(extra.sum())]:extra[i]+=1
 q=sizes+extra
 assert int(q.sum())==budget and np.all(q>=sizes)
 return q

def compile_plan(directory,pool_path,cem_path,historical_path=None):
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
 if (directory/'plan.json').exists():raise FileExistsError('Do not overwrite an existing plan')
 pool=json.loads(Path(pool_path).read_text());cem=json.loads(Path(cem_path).read_text())
 original=HistoricalSampler(pool,cem)
 if historical_path is None:
  historical=np.fromiter((original.choose(i).get('image_id',-1) for i in range(STEPS*BATCH)),dtype=np.int32,count=STEPS*BATCH)
 else:historical=np.load(historical_path,allow_pickle=False)
 assert historical.shape==(STEPS*BATCH,) and historical.dtype==np.int32
 slots=np.flatnonzero(historical>=0);budget=len(slots)
 assert budget==2398767 and cem['images']==1425433 and len(cem['groups'])==627
 q=quotas(cem,budget)
 labels=np.repeat(np.arange(len(q),dtype=np.int32),q)
 rng('collection-order').shuffle(labels)
 ids=np.full(STEPS*BATCH,-1,np.int32)
 owner=np.empty(cem['images'],np.int32)
 for i,g in enumerate(cem['groups']):
  assert g['last_id']-g['first_id']+1==g['images']
  owner[g['first_id']:g['last_id']+1]=i
  positions=slots[labels==i];filled=0;cycle=0
  while filled<len(positions):
   queue=rng('queue|'+g['name']+'|'+str(cycle)).permutation(np.arange(g['first_id'],g['last_id']+1,dtype=np.int32))
   take=min(len(queue),len(positions)-filled)
   ids[positions[filled:filled+take]]=queue[:take];filled+=take;cycle+=1
 assert np.array_equal(owner[ids[slots]],labels)
 counts=np.bincount(ids[slots],minlength=cem['images']);assert np.all(counts>=1)
 path=directory/'ids.npy';np.save(path,ids,allow_pickle=False)
 meta={'version':1,'strategy':'Coverage-NR','start_step':0,'end_step':STEPS,'global_batch':BATCH,'ssl_seed':0,
       'id_file':'ids.npy','id_sha256':sha(path),'length':len(ids),'cem_slots':budget,'mv_slots':len(ids)-budget,
       'pool_sha256':sha(pool_path),'cem_sha256':sha(cem_path),'historical_ids_sha256':sha(historical_path) if historical_path else None,
       'collections':[{'name':g['name'],'images':g['images'],'quota':int(q[i])} for i,g in enumerate(cem['groups'])],
       'planned_unique_cem_ids':int(np.count_nonzero(counts)),'planned_coverage':1.0,
       'max_image_visits':int(counts.max()),'code_sha256':sha(Path(__file__))}
 (directory/'plan.json').write_text(json.dumps(meta,indent=2)+'\n')
 return meta

class Plan:
 def __init__(self,directory,pool_path,cem_path):
  directory=Path(directory);self.meta=json.loads((directory/'plan.json').read_text())
  if sha(pool_path)!=self.meta['pool_sha256'] or sha(cem_path)!=self.meta['cem_sha256']:raise ValueError('Pool hash changed')
  if sha(Path(__file__))!=self.meta['code_sha256']:raise ValueError('Sampler implementation changed')
  if sha(directory/self.meta['id_file'])!=self.meta['id_sha256']:raise ValueError('ID plan hash changed')
  self.ids=np.load(directory/self.meta['id_file'],mmap_mode='r',allow_pickle=False)
  if self.ids.dtype!=np.int32 or self.ids.shape!=(STEPS*BATCH,):raise ValueError('Invalid full-run plan')
  pool=json.loads(Path(pool_path).read_text());self.cem=json.loads(Path(cem_path).read_text())
  self.original=HistoricalSampler(pool,self.cem)
  self.owner=np.empty(self.cem['images'],np.int32)
  for i,g in enumerate(self.cem['groups']):self.owner[g['first_id']:g['last_id']+1]=i
 def choose(self,index):
  if not 0<=index<len(self.ids):raise IndexError(index)
  original=self.original.choose(index);iid=int(self.ids[index])
  if original['domain']=='MitoVerse':
   if iid!=-1:raise ValueError('Domain plan mismatch')
   return original
  if not 0<=iid<len(self.owner):raise ValueError('Invalid CEM ID')
  g=self.cem['groups'][int(self.owner[iid])]
  return {**original,'source':g['name'],'volume_id':g['name'],'image_id':iid}
 def checkpoint_state(self,consumed_step):
  if not 0<=consumed_step<=STEPS:raise ValueError('Invalid consumed optimizer step')
  return {'version':1,'plan_sha256':self.meta['id_sha256'],'consumed_step':consumed_step,
          'next_global_index':consumed_step*BATCH,'world_size':3,'batch_per_rank':16}
 def validate_resume(self,state):
  if state!=self.checkpoint_state(state['consumed_step']):raise ValueError('Sampler checkpoint mismatch')
  return state['next_global_index']

def verify(directory,pool_path,cem_path):
 plan=Plan(directory,pool_path,cem_path)
 # Check every domain against the actual training accessor's historical logic.
 for i in range(STEPS*BATCH):
  expected=plan.original.choose(i)
  assert (int(plan.ids[i])>=0)==(expected['domain']=='CEM1.5M')
 # Reordered ranks/workers, prefetch and arbitrary resume positions cannot mutate state.
 checked=0
 for step in [0,1,9,10,9999,10000,29999,30000,99998]:
  state=plan.checkpoint_state(step);assert plan.validate_resume(state)==step*BATCH
  before=[plan.choose(i) for i in range(step*BATCH,(step+1)*BATCH)]
  reordered={}
  for worker in [1,0]:
   for rank in [2,0,1]:
    for j in range(16):
     i=step*BATCH+rank*16+j
     if j%2==worker:reordered[i]=plan.choose(i)
  assert before==[reordered[i] for i in sorted(reordered)]
  assert state==plan.checkpoint_state(step);checked+=len(before)
 bad={**plan.checkpoint_state(10),'plan_sha256':'0'*64}
 try:plan.validate_resume(bad)
 except ValueError:pass
 else:raise AssertionError('Corrupt plan accepted')
 ids=np.asarray(plan.ids);cem_ids=ids[ids>=0]
 counts=np.bincount(cem_ids,minlength=plan.cem['images'])
 assert np.all(counts>=1)
 import csv
 seen=np.zeros(plan.cem['images'],bool);curve=[]
 for end in range(1000,STEPS+1,1000):
  values=ids[(end-1000)*BATCH:end*BATCH];seen[values[values>=0]]=True
  curve.append({'step':end,'unique_cem_ids':int(seen.sum()),'coverage':float(seen.mean())})
 with (Path(directory)/'coverage_curve.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(curve[0]));w.writeheader();w.writerows(curve)
 result={'status':'PASS','all_domain_positions_checked':len(ids),'reordered_full_records':checked,
         'coverage':float(np.count_nonzero(counts)/len(counts)),'unique_cem_ids':int(np.count_nonzero(counts)),
         'id_plan_sha256':plan.meta['id_sha256'],'image_reads':0,'prefetch_order_independent':True}
 (Path(directory)/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
 return result

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['compile','verify']);p.add_argument('--directory',type=Path,required=True)
 p.add_argument('--pool',type=Path,required=True);p.add_argument('--cem',type=Path,required=True);p.add_argument('--historical',type=Path)
 a=p.parse_args()
 result=compile_plan(a.directory,a.pool,a.cem,a.historical) if a.mode=='compile' else verify(a.directory,a.pool,a.cem)
 print(json.dumps({k:v for k,v in result.items() if k!='collections'},indent=2))
