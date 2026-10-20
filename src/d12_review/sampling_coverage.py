"""Replay deterministic IDs only. No images, labels, GPU or model reads."""
import argparse,bisect,collections,importlib.util,json,random,time
from pathlib import Path
import numpy as np

def main(repo,out):
    start=time.monotonic()
    pool=json.loads((repo/'experiments/D12/mitoverse_pool.json').read_text())
    cem=json.loads((repo/'experiments/D12/cem_pool.json').read_text())
    spec=importlib.util.spec_from_file_location('original',repo/'src/d12/sampling.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    sources=sorted(pool['sampling_groups']);groups=pool['sampling_groups']
    acqs={s:sorted(groups[s]) for s in sources}
    volumes={v['volume_id']:v for v in pool['volumes']}
    offsets={};n=0
    for v in pool['volumes']:offsets[v['volume_id']]=n;n+=v['legal_slices']
    cumulative=[];total=0.
    for g in cem['groups']:total+=g['probability'];cumulative.append(total)
    def draw(index,fraction):
        rng=random.Random(index)  # Actual SSL seed=0; index is global draw index.
        if rng.random()<fraction:
            g=cem['groups'][min(bisect.bisect_right(cumulative,rng.random()*total),len(cumulative)-1)]
            return 'CEM1.5M',g['name'],rng.randint(g['first_id'],g['last_id']),None
        s=rng.choice(sources);a=rng.choice(acqs[s]);vid=rng.choice(groups[s][a])
        offset=rng.randrange(volumes[vid]['legal_slices'])
        return 'MitoVerse',s,offsets[vid]+offset,(vid,offset)
    # Independent comparison with unchanged original sampler, not self-comparison.
    checks=sorted(set(range(100))|{random.Random(i).randrange(4800000) for i in range(100)})
    for fraction in [0.,.5]:
        for i in checks:
            d,s,key,vo=draw(i,fraction);ref=original.choose(pool,cem,fraction,0,i)
            assert (d,s)==(ref['domain'],ref['source'])
            if d=='CEM1.5M':assert key==ref['image_id']
            else:
                vid,offset=vo;z=None
                for a,b in volumes[vid]['allowed_z']:
                    if offset<b-a:z=a+offset;break
                    offset-=b-a
                assert (vid,z)==(ref['volume_id'],ref['z'])
    result={'method':'Exact deterministic seed0 global indices, optimized ID-only replay; original sampler equivalence checks and all source counts asserted',
            'equivalence_checks':len(checks)*2,'scope':'unique MV legal Z slices and CEM image IDs, not crop pixel coverage/acquisition independence',
            'pool':{'mv_volumes':len(volumes),'mv_legal_z':n,'cem_images':cem['images'],'cem_groups':len(cem['groups'])},'runs':{}}
    for model,fraction in [('D1',0.),('D2',.5)]:
        mv_seen=np.zeros(n,bool);cem_seen=np.zeros(cem['images'],bool);counts=collections.Counter();seen_vol=set()
        for i in range(4800000):
            d,s,key,vo=draw(i,fraction);counts[d+'::'+s]+=1
            if d=='MitoVerse':mv_seen[key]=True;seen_vol.add(vo[0])
            else:cem_seen[key]=True
            if i+1 in [1440000,4800000]:
                stage='D12' if i+1==1440000 else 'D12-100k';prefix='D12' if stage=='D12' else 'D12_100k'
                recorded=json.loads((repo/f'experiments/{stage}/{prefix}_sampling_observed.json').read_text())[model]
                assert dict(counts)==recorded,'Replay source counts disagree with actual consumed counts'
                key_run=f'{model}_{(i+1)//48}'
                result['runs'][key_run]={'draws':i+1,'mv_unique_slices':int(mv_seen.sum()),'mv_unique_slice_fraction':float(mv_seen.mean()),
                    'mv_unique_volumes':len(seen_vol),'cem_unique_images':int(cem_seen.sum()),'cem_unique_image_fraction':float(cem_seen.mean()),
                    'mv_sources_seen':sum(k.startswith('MitoVerse::') for k in counts),'cem_groups_seen':sum(k.startswith('CEM1.5M::') for k in counts),
                    'consumed_source_counts_match':True}
                print(key_run,json.dumps(result['runs'][key_run]),flush=True)
        print('completed',model,'seconds',time.monotonic()-start,flush=True)
    result['seconds']=time.monotonic()-start
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.repo,a.out)
