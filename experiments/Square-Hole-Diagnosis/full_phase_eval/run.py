import csv,json,sys,time
from pathlib import Path
import numpy as np
import torch,tifffile
from scipy import ndimage as nd
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'fusion_trace'))
import trace as tr
from probe1 import BASE,DATA,MB,Planes,gpu_guard,primus,read,sha
from probe2 import holes_of,mask_at,CONN
SIZE=256
x=np.arange(SIZE,dtype=np.float32)-(SIZE-1)/2
GAUSS=np.exp(-(x[:,None]**2+x[None,:]**2)/(2*64**2)).astype(np.float32)
CONFIGS=('Baseline','Gauss + Stride-120','Gauss + Dual-Phase','Gauss + Quad-Phase')
CHK=('tp','fp','fn','tn')

def starts(n,stride,phase):
    if stride==128 and phase==0:
        return sorted(set(range(0,max(1,n-SIZE+1),stride))|{max(0,n-SIZE)})
    first=phase-stride if phase else 0
    steps=max(0,(n-SIZE-first+stride-1)//stride)
    return [first+i*stride for i in range(steps+1)]

def settings(shape,stride,phase):
    return [(y,x) for y in starts(shape[0],stride,phase[0]) for x in starts(shape[1],stride,phase[1])]

def measure_pass(net,probe,a,stride,phase,baseline=False):
    tick=time.monotonic()
    h,w=a.shape
    coords=settings(a.shape,stride,phase)
    acc=np.zeros((4 if baseline else 2,h,w),np.float32)
    for start in range(0,len(coords),16):
        cs=coords[start:start+16]
        batch=[];items=[]
        for y,x in cs:
            y0,x0=max(0,y),max(0,x)
            y1,x1=min(h,y+SIZE),min(w,x+SIZE)
            dst=np.s_[y0:y1,x0:x1]
            src=np.s_[y0-y:y1-y,x0-x:x1-x]
            tile=a[dst].astype(np.float32)
            if tile.shape!=(SIZE,SIZE):
                tile=np.pad(tile,((y0-y,y+SIZE-y1),(x0-x,x+SIZE-x1)),mode='reflect')
            batch.append(tile)
            items.append((dst,src))
        p=probe.forward(net,tr.inp(np.stack(batch))).softmax(1)[:,1].cpu().numpy()
        for q,(dst,src) in zip(p,items):
            a0=q[src]; g=GAUSS[src]
            acc[0][dst]+=a0*g
            acc[1][dst]+=g
            if baseline:
                acc[2][dst]+=a0
                acc[3][dst]+=1
    assert acc[1].min()>0
    g=acc[0]/acc[1]
    b=acc[2]/acc[3] if baseline else None
    return b,g,len(coords),time.monotonic()-tick

def topo(mask):
    holes=holes_of(mask)
    labels,n=nd.label(holes,structure=CONN)
    areas=np.bincount(labels.ravel())[1:]
    count=0
    for a,b in zip(areas,nd.find_objects(labels)):
        h,w=(s.stop-s.start for s in b)
        lo,hi=sorted((h,w))
        count+=int(12<=lo<=68 and lo/hi>=.75 and a/(h*w)>=.75)
    return dict(fg_area_px=int(mask.sum()),hole_area_px=int(areas.sum()),hole_ratio=int(areas.sum())/max(1,int(mask.sum()+areas.sum())),squares=count,holes=n)

def sem(mask,g):
    if g is None:return dict.fromkeys(('dice','iou','precision','recall',*CHK))
    tp,fg,ps=int((mask&g).sum()),int(g.sum()),int(mask.sum())
    union=fg+ps-tp
    return dict(tp=tp,fp=ps-tp,fn=fg-tp,tn=g.size-union,dice=2*tp/(fg+ps) if fg+ps else 1.,
        iou=tp/union if union else 1.,precision=tp/ps if ps else None,recall=tp/fg if fg else None)

def run_plane(net,probe,a,gt,label,group,key,z,quad,historical):
    b,z0,n0,t0=measure_pass(net,probe,a,128,(0,0),True)
    baseline=b>.5
    score=sem(baseline,gt)
    if label=='HQ':
        diff=int(np.count_nonzero(baseline!=historical))
        assert diff==0
    else:
        diff=int(any(score[k]!=historical[k] for k in CHK))
        if min(a.shape)>=SIZE:
            assert diff==0,(group,key,z,score,historical)
    _,g120,n120,t120=measure_pass(net,probe,a,120,(0,0))
    _,z8,n8,t8=measure_pass(net,probe,a,128,(8,8))
    passes=[(CONFIGS[0],baseline,n0,t0),(CONFIGS[1],g120>.5,n120,t120),(CONFIGS[2],(z0+z8)>1,n0+n8,t0+t8)]
    if quad:
        _,zx,nx,tx=measure_pass(net,probe,a,128,(0,8))
        _,zy,ny,ty=measure_pass(net,probe,a,128,(8,0))
        passes.append((CONFIGS[3],(z0+z8+zx+zy)>2,n0+n8+nx+ny,t0+t8+tx+ty))
    by=[]
    for config,mask,n,t in passes:
        by.append(dict(dataset=label,group=group,id=key,z=z,config=config,
            windows=n,seconds=t,baseline_replay_diff=diff if config==CONFIGS[0] else None,reflect_small_image=min(a.shape)<SIZE,
            **sem(mask,gt),**topo(mask)))
    return by

@torch.no_grad()
def main():
    gpu=gpu_guard()
    torch.set_num_threads(2)
    begin=time.monotonic()
    net,p,enc=primus('Barrier100')
    base=BASE/'batch4/primus_cem/Barrier100'
    ck=base/'model_40000.pt'
    m=read(base/'trained.json')
    assert enc==m['init_encoder_sha'] and sha(ck)==m['checkpoint_sha'] and m['seed']==20261014 and m['steps']==40000
    missing,extra=net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=False)
    assert not extra and all(x.startswith('dino_encoder.') for x in missing)
    net.eval()
    ep=read(BASE.parent/'pilot_c100_decoder_seeds_20261102/protocol.json')['models']['C1_Barrier_100k']
    ep=BASE.parent.parent/ep['encoder_relative']
    assert sha(ep)==enc
    protocol=dict(gpu=gpu,encoder_path=str(ep),encoder_sha256=enc,decoder_path=str(ck),decoder_sha256=m['checkpoint_sha'],seed=m['seed'],batch=16,padding='reflect')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    manifest={r['id']:r for r in csv.DictReader((DATA/'manifest.csv').open(encoding='utf-8-sig'))}
    common=sorted({r['fov'] for r in csv.DictReader((OUT.parent/'probe2_hq5nm_holes_per_image.csv').open()) if r['pack']=='original' and r['model']=='Barrier100'})
    assert len(common)==20
    done=OUT/'units'
    done.mkdir(exist_ok=True)
    for key in common:
        f=done/('HQ_'+key+'.json')
        if f.exists():continue
        r=manifest[key];path=DATA/r['file']
        assert sha(path)==r['file_sha256']
        a=tifffile.imread(path)
        rows=run_plane(net,p,a,None,'HQ',r['organ'],key,0,True,mask_at('original','Barrier100/'+key+'.png'))
        tmp=f.with_suffix('.tmp')
        tmp.write_text(json.dumps(rows));tmp.replace(f)
        print('HQ',key,flush=True)
    hq=[r for f in done.glob('HQ_*.json') for r in read(f)]
    assert len(hq)==80
    entries=read(MB/'MB1_benchmark_manifest.json')['entries']
    avg=sum(r['seconds'] for r in hq if r['config']==CONFIGS[3])/sum(r['windows'] for r in hq if r['config']==CONFIGS[3])
    full_est=sum(r['shape_zyx'][0]*sum(len(settings(r['shape_zyx'][1:],128,phase)) for phase in ((0,0),(8,8),(0,8),(8,0))) for r in entries)
    quad_full=avg*full_est<=900
    protocol.update(quad_full=quad_full,quad_full_estimated_seconds=avg*full_est)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print('Quad full estimate',round(avg*full_est),'seconds. full=',quad_full,flush=True)
    refs={ (r['name'],r['z']):r for r in read(BASE/'zeroshot_repeat_20261120/Barrier100_20261014.json')['rows'] if r['split']=='mb1'}
    assert len(refs)==2595
    entries=sorted(entries,key=lambda r:(r['dataset']=='salivary_gland',r['dataset'],r['id']))
    for r in entries:
        group,key=r['dataset'],r['id']
        f=done/('MB1_'+key+'.jsonl')
        partial=f.with_suffix('.partial')
        if f.exists():continue
        already=len(partial.read_text().splitlines()) if partial.exists() else 0
        quad=quad_full or group in ('TEM-100','glycolytic_muscle')
        a,b=[Planes(r[k],r['shape_zyx']) for k in ('image_path','label_path')]
        with partial.open('a') as output:
            for z in range(already,r['shape_zyx'][0]):
                rows=run_plane(net,p,a[z],b[z]>0,'MB1',group,key,z,quad,refs[key,z])
                output.write(json.dumps(rows)+'\n');output.flush()
                if z%25==0:print('MB1',key,z+1,'/',r['shape_zyx'][0],flush=True)
        a.close();b.close()
        partial.replace(f)
    (OUT/'receipt.json').write_text(json.dumps(dict(seconds_wall=time.monotonic()-begin,gpu=gpu),indent=2))

if __name__=='__main__':
    main()
