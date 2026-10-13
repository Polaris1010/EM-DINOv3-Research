"""Record user exception separately; preserve strict empty allowlists and quarantines."""
import json,csv,hashlib,collections
from pathlib import Path
import numpy as np
from settings import *

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    assert sha(P1)==P1_SHA
    root=BASE/'ssl_exploratory';root.mkdir(exist_ok=False)
    split=json.loads((R4/'split_mitoverse_probe_v1.json').read_text())
    jrc=list(csv.DictReader((R5/'ssl_dataset_manifest.csv').open()))
    for r in jrc:r['kind']='png'
    tm=[]
    for vid in split['train']:
        r=split['volumes'][vid];assert r['split']=='train'
        shape=json.loads((Path(r['image_path'])/'.zarray').read_text())['shape']
        tm.append(dict(source=r['source'],volume=vid,path=r['image_path'],kind='zarr',shape=json.dumps(shape)))
    assert len(tm)==31
    exception=dict(authorization='User: 允许进行探索, after explicitly asked whether unresolved provenance may be used for exploratory SSL',strict_identity_gate_pass=False,leakage_risk='original specimen/ROI relations unresolved; no leak-free or pristine-unseen benchmark claims',split_changed=False,labels_used_for_SSL=False,known_heldout_sources_excluded=True,strict_allowlists_unchanged=True,TJ_TM_independence='no named scan match across proposed pools; unresolved identity does not prove independence',P1_historical_exposure='unchanged/uncertain')
    for arm,rows in [('TJ',jrc),('TM',tm)]:
        pool=root/arm;pool.mkdir();fields=sorted(set().union(*(r.keys() for r in rows)))
        with (pool/'ssl_dataset_manifest.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        groups={}
        for i,r in enumerate(rows):groups.setdefault(r['source'],{}).setdefault(r['volume'],{}).setdefault(int(r.get('z',0)),[]).append(i)
        sources=sorted(groups);vols={s:sorted(groups[s]) for s in sources};zs={(s,v):sorted(groups[s][v]) for s in sources for v in vols[s]}
        schedule=np.empty((160000,5),dtype=np.int64);counts=collections.Counter();volume_counts=collections.Counter()
        rng=np.random.default_rng(20260918)
        for i in range(len(schedule)):
            source=sources[int(rng.integers(len(sources)))];volume=vols[source][int(rng.integers(len(vols[source])))];counts[source]+=1;volume_counts[volume]+=1
            if arm=='TJ':
                z=zs[source,volume][int(rng.integers(len(zs[source,volume])))];inds=groups[source][volume][z];j=inds[int(rng.integers(len(inds)))];y=int(rows[j]['y']);x=int(rows[j]['x'])
            else:
                j=groups[source][volume][0][0];shape=json.loads(rows[j]['shape']);z=int(rng.integers(shape[0]));y=int(rng.integers(max(1,shape[1]-512+1)));x=int(rng.integers(max(1,shape[2]-512+1)))
            schedule[i]=[j,z,y,x,100000+i]
        np.save(pool/'ssl_schedule.npy',schedule)
        locked=dict(P1=str(P1),P1_sha256=P1_SHA,manifest_sha256=sha(pool/'ssl_dataset_manifest.csv'),schedule_sha256=sha(pool/'ssl_schedule.npy'),arm=arm,target_steps=10000,max_batch=16,SSL_seed=0,source_counts=dict(counts),unique_volumes=len(volume_counts),volume_counts=dict(volume_counts),sampling='uniform source -> uniform volume -> uniform z -> uniform available/legal crop',exception=exception)
        (pool/'locked_inputs.json').write_text(json.dumps(locked,indent=2))
    (BASE/'exploratory_ssl_authorization.json').write_text(json.dumps(exception,indent=2))
    print('EXPLORATORY_POOLS_READY: TJ and TM; strict quarantine preserved',flush=True)

if __name__=='__main__':main()
