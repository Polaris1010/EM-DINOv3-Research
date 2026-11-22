import csv,importlib.util,json,sys,time
from pathlib import Path
from collections import defaultdict,deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from core import gt_maps,metrics
O=Path(__file__).resolve().parent
R=O.parent/'full_phase_eval'
spec=importlib.util.spec_from_file_location('fullphase_reference',R/'run.py')
q=importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
from common import config
from infer import mitonet
NAMES=('C1 Baseline','C1 Stride-120','C1 Dual-Phase','MitoNet-v1')
GROUPS={'TEM-100','glycolytic_muscle','hela_cell','c_elegans','lucchi_pp','salivary_gland'}
MITO=q.BASE.parent/'mitonet_v1_orthoplane_20261120'/'pred'

def work(key,z,group,a,g,preds):
    gt=gt_maps(g)
    rows=[]; display=[]
    for name,p in zip(NAMES,preds):
        info,image=metrics(gt,p)
        display.append(image)
        for r in info:
            rows.append(dict(group=group,id=key,z=z,config=name,**r))
    return rows,a,display

def collect(future,output,tops):
    rows,a,display=future.result()
    output.write(json.dumps(rows)+'\n')
    output.flush()
    r=rows[0];g=r['group']
    if g not in GROUPS or not r['interior_px']:return
    key=(r['interior_fn_px']/r['interior_px'],r['interior_fn_px'],r['id'],r['z'])
    rank=sorted(tops[g]+[key],reverse=True)[:2]
    if key not in rank:return
    dest=O/'examples'/f"{g}__{r['id']}__{r['z']}.npz"
    np.savez_compressed(dest,raw=a,overlays=np.stack(display))
    tops[g]=rank

@torch.no_grad()
def main():
    start=time.monotonic()
    gpu=q.gpu_guard()
    torch.set_num_threads(2)
    meta=q.read(R/'protocol.json')
    net,probe,sha=q.primus('Barrier100')
    assert sha==meta['encoder_sha256']
    ck=Path(meta['decoder_path'])
    assert q.sha(ck)==meta['decoder_sha256']
    missing,extra=net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=False)
    assert not extra and all(k.startswith('dino_encoder.') for k in missing)
    net.eval()
    spec=next(r for r in config()['runs'] if r['model']=='Official-MitoNet-v1')
    assert q.sha(Path(spec['checkpoint_path']))==spec['checkpoint_sha256']
    run_tem=mitonet(spec)
    manifest=q.read(q.MB/'MB1_benchmark_manifest.json')['entries']
    refs={}
    with (R/'per_image.csv').open() as f:
        for r in csv.DictReader(f):
            if r['dataset']=='MB1' and r['config'] in q.CONFIGS[:3]:
                refs[r['id'],int(r['z']),r['config']]=[int(r[k]) for k in q.CHK]
    assert len(refs)==2595*3
    protocol=dict(gpu=gpu,encoder=meta['encoder_path'],decoder=str(ck),encoder_sha256=sha,decoder_sha256=meta['decoder_sha256'],
        mitonet_path=spec['checkpoint_path'],mitonet_sha256=spec['checkpoint_sha256'],
        mitonet_geometry='official 3D packed existing volume masks, official Engine2d for TEM-100',
        instance_policy='native GT instance IDs per slice',radii=[3,5],erosion='disk',hole_connectivity=4)
    (O/'protocol.json').write_text(json.dumps(protocol,indent=2))
    top=defaultdict(list)
    tfile=O/'examples'/'top.json'
    if tfile.exists():
        for k,v in json.loads(tfile.read_text()).items(): top[k]=[tuple(x) for x in v]
    manifest.sort(key=lambda r:(r['dataset']=='salivary_gland',r['dataset'],r['id']))
    n=0
    with ThreadPoolExecutor(max_workers=3) as workers:
        for r in manifest:
            group,key=r['dataset'],r['id']
            f=O/'units'/f'{key}.jsonl'
            part=f.with_suffix('.partial')
            if f.exists():
                n+=r['shape_zyx'][0]
                continue
            already=len(part.read_text().splitlines()) if part.exists() else 0
            n+=already
            b,g=[q.Planes(r[k],r['shape_zyx']) for k in ('image_path','label_path')]
            packed=None
            if group!='TEM-100':
                with np.load(MITO/f'mb1_{key}.npz') as z:
                    assert tuple(z['shape'])==tuple(r['shape_zyx'])
                    packed=z['foreground_bits']
            pending=deque()
            with part.open('a') as out:
                for z in range(already,r['shape_zyx'][0]):
                    a,gold=np.array(b[z]),np.array(g[z])
                    base,g0,_,_=q.measure_pass(net,probe,a,128,(0,0),True)
                    _,s120,_,_=q.measure_pass(net,probe,a,120,(0,0))
                    _,g8,_,_=q.measure_pass(net,probe,a,128,(8,8))
                    prep=[base>.5,s120>.5,(g0+g8)>1]
                    for c,mask in zip(q.CONFIGS,prep):
                        scores=q.sem(mask,gold>0)
                        assert [scores[k] for k in q.CHK]==refs[key,z,c],(key,z,c)
                    if packed is None:
                        m=np.asarray(run_tem(a))>0
                    else:m=np.unpackbits(packed[z],axis=-1,count=a.shape[1])>0
                    assert m.shape==a.shape
                    pending.append(workers.submit(work,key,z,group,a,gold,prep+[m]))
                    if len(pending)>=5: collect(pending.popleft(),out,top)
                    n+=1
                    if z%25==0:print(group,key,z+1,'/',r['shape_zyx'][0],f'{n}/2595',flush=True)
                while pending:collect(pending.popleft(),out,top)
            tmp=tfile.with_suffix('.tmp')
            tmp.write_text(json.dumps(top));tmp.replace(tfile)
            part.replace(f)
            b.close();g.close()
    assert n==2595
    (O/'receipt.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,gpu=gpu,planes=n),indent=2))
if __name__=='__main__':
    main()
