"""Exact historical seed/index draw logic, extracted without changes."""
import bisect,random
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
