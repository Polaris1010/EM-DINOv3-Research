"""CPU plan compiler and immutable random-access sampler; no model/image imports.

The simulation uses CoveragePlan.choose, the same entry point a future loader
must call. Plans contain image IDs only; historical training code is untouched.
"""
import bisect,hashlib,json,random
from pathlib import Path
import numpy as np

START=30000
END=100000
BATCH=48

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

class HistoricalSampler:
    """Preserve original Random(index) draws, including the original view_seed."""
    def __init__(self,pool,cem,seed=0):
        self.pool,self.cem,self.seed=pool,cem,seed
        self.sources=sorted(pool['sampling_groups'])
        self.acqs={s:sorted(pool['sampling_groups'][s]) for s in self.sources}
        self.volumes={v['volume_id']:v for v in pool['volumes']}
        self.cumulative=[];total=0.
        for g in cem['groups']:total+=g['probability'];self.cumulative.append(total)
        self.total=total
    def choose(self,index):
        rng=random.Random(self.seed*1_000_000_007+int(index))
        if rng.random()<.5:
            g=self.cem['groups'][min(bisect.bisect_right(self.cumulative,rng.random()*self.total),len(self.cumulative)-1)]
            return {'domain':'CEM1.5M','source':g['name'],'volume_id':g['name'],
                    'image_id':rng.randint(g['first_id'],g['last_id']),'view_seed':rng.randrange(2**31-1),'index':int(index)}
        s=rng.choice(self.sources);a=rng.choice(self.acqs[s]);vid=rng.choice(self.pool['sampling_groups'][s][a]);v=self.volumes[vid]
        offset=rng.randrange(v['legal_slices'])
        for lo,hi in v['allowed_z']:
            if offset<hi-lo:z=lo+offset;break
            offset-=hi-lo
        return {'domain':'MitoVerse','source':s,'acquisition_sampling_key':a,'volume_id':vid,'z':z,
                'y':rng.randrange(max(1,v['shape'][1]-512+1)),'x':rng.randrange(max(1,v['shape'][2]-512+1)),
                'view_seed':rng.randrange(2**31-1),'index':int(index)}

def shuffle(values,tag):
    seed=int.from_bytes(hashlib.sha256(('D2-Coverage-v1|'+tag).encode()).digest()[:16],'little')
    return np.random.Generator(np.random.PCG64(seed)).permutation(values)

def coverage_repeat_quotas(missing,a_counts):
    """Integer minimum-TV restoration towards A's fixed future collection counts."""
    budget=int(a_counts.sum());unseen=int(missing.sum())
    if unseen>budget:raise ValueError('Insufficient CEM slots')
    repeat_budget=budget-unseen
    capacity=np.maximum(a_counts-missing,0).astype(np.int64)
    if repeat_budget==0:return np.zeros_like(capacity)
    # Use integer remainders to make tie-breaking and platform behavior explicit.
    numer=capacity*repeat_budget;denom=int(capacity.sum())
    if denom==0:raise ValueError('No legal repeat capacity')
    quotas=numer//denom;remaining=repeat_budget-int(quotas.sum())
    order=sorted(range(len(capacity)),key=lambda i:(-int(numer[i]%denom),i))
    for i in order[:remaining]:quotas[i]+=1
    assert int(quotas.sum())==repeat_budget and np.all(quotas<=capacity)
    return quotas

def compile_plan(strategy,historical_ids,cem):
    """Historical IDs: -1 for MV, CEM ID otherwise, global-index ordered."""
    prefix=historical_ids[:START*BATCH]
    seen=np.zeros(cem['images'],bool);seen[prefix[prefix>=0]]=True
    out=np.array(historical_ids[START*BATCH:],dtype=np.int32)
    slots=np.flatnonzero(out>=0)
    owner=np.empty(cem['images'],np.int32)
    for g,group in enumerate(cem['groups']):owner[group['first_id']:group['last_id']+1]=g
    if strategy=='A':return out
    if strategy=='B':
        owners=owner[out[slots]]
        for g,group in enumerate(cem['groups']):
            positions=slots[owners==g]
            ids=np.arange(group['first_id'],group['last_id']+1,dtype=np.int32)
            queue=shuffle(ids[~seen[ids]],f'B|{group["name"]}|unseen30k')
            take=min(len(queue),len(positions));out[positions[:take]]=queue[:take]
            filled=take;cycle=0
            while filled<len(positions):
                queue=shuffle(ids,f'B|{group["name"]}|cycle{cycle}')
                take=min(len(queue),len(positions)-filled)
                out[positions[filled:filled+take]]=queue[:take];filled+=take;cycle+=1
        return out
    if strategy=='C':
        missing=np.flatnonzero(~seen).astype(np.int32)
        if len(missing)>len(slots):raise ValueError('Insufficient fixed CEM slot budget for full coverage')
        out[slots[:len(missing)]]=shuffle(missing,'C|global_unseen30k')
        missing_counts=np.bincount(owner[missing],minlength=len(cem['groups']))
        a_counts=np.bincount(owner[historical_ids[START*BATCH:][slots]],minlength=len(cem['groups']))
        quotas=coverage_repeat_quotas(missing_counts,a_counts)
        repeat=[]
        for i,g in enumerate(cem['groups']):
            seed=int.from_bytes(hashlib.sha256(('D2-Coverage-v1|C|repeat|'+g['name']).encode()).digest()[:16],'little')
            rng=np.random.Generator(np.random.PCG64(seed))
            repeat.append(rng.integers(g['first_id'],g['last_id']+1,size=int(quotas[i]),dtype=np.int32))
        out[slots[len(missing):]]=shuffle(np.concatenate(repeat),'C|repeat_global_interleave')
        final=missing_counts+quotas
        assert int(np.abs(final-a_counts).sum())==2*int(np.maximum(missing_counts-a_counts,0).sum())
        return out
    raise ValueError(strategy)

def save_plan(directory,strategy,ids,pool_path,cem_path):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{strategy}_ids.npy';np.save(path,ids,allow_pickle=False)
    meta={'version':1,'strategy':strategy,'start_step':START,'end_step':END,'global_batch':BATCH,
          'ssl_seed':0,'id_file':path.name,'id_sha256':sha(path),'length':len(ids),
          'pool_sha256':sha(pool_path),'cem_sha256':sha(cem_path)}
    (directory/f'{strategy}_plan.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    return meta

class CoveragePlan:
    """Stateless per-index accessor: rank/worker order and prefetch cannot advance it."""
    def __init__(self,directory,strategy,pool_path,cem_path):
        directory=Path(directory)
        self.meta=json.loads((directory/f'{strategy}_plan.json').read_text())
        if sha(pool_path)!=self.meta['pool_sha256'] or sha(cem_path)!=self.meta['cem_sha256']:raise ValueError('Pool hash changed')
        if sha(directory/self.meta['id_file'])!=self.meta['id_sha256']:raise ValueError('Plan hash changed')
        self.ids=np.load(directory/self.meta['id_file'],mmap_mode='r',allow_pickle=False)
        if self.ids.shape!=(self.meta['length'],) or self.ids.dtype!=np.int32:raise ValueError('Plan layout changed')
        pool=json.loads(Path(pool_path).read_text());cem=json.loads(Path(cem_path).read_text())
        self.original=HistoricalSampler(pool,cem)
        self.owner=np.empty(cem['images'],np.int32);self.groups=cem['groups']
        for i,g in enumerate(self.groups):self.owner[g['first_id']:g['last_id']+1]=i
    def image_id(self,index):
        offset=int(index)-START*BATCH
        if not 0<=offset<len(self.ids):raise IndexError('Index outside planned continuation')
        return int(self.ids[offset])
    def choose(self,index):
        # Keep all original MV coordinates and view seeds; RNG is local per ID.
        original=self.original.choose(index);image_id=self.image_id(index)
        if original['domain']=='MitoVerse':
            if image_id!=-1:raise ValueError('Domain schedule mismatch')
            return original
        if not 0<=image_id<len(self.owner):raise ValueError('CEM ID invalid')
        g=self.groups[int(self.owner[image_id])]
        return {**original,'source':g['name'],'volume_id':g['name'],'image_id':image_id}
    def checkpoint_state(self,consumed_step):
        if not START<=consumed_step<=END:raise ValueError('Bad optimizer step')
        return {'version':1,'plan_sha256':self.meta['id_sha256'],'consumed_step':consumed_step,
                'next_global_index':consumed_step*BATCH,'world_size':3,'batch_per_rank':16}
    def validate_resume(self,state):
        if state!=self.checkpoint_state(state['consumed_step']):raise ValueError('Sampler checkpoint does not match plan')
        return state['next_global_index']
