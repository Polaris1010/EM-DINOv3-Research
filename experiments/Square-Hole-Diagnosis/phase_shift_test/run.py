import csv,json,sys,time
from pathlib import Path
import numpy as np
import torch,tifffile
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'fusion_trace'))
import trace as prev
from probe1 import DATA,BASE,MB,Planes,gpu_guard,primus,read,sha
from probe2 import mask_at,holes_of
SIZE,STRIDE=256,128
x=np.arange(SIZE,dtype=np.float32)-(SIZE-1)/2
gaussian=np.exp(-(x[:,None]**2+x[None,:]**2)/(2*(.25*SIZE)**2)).astype(np.float32)

def starts(n,phase,probe):
    if phase==0:
        return probe.starts(n)
    # Keep every shifted start on phase 8, including tiles covering image borders.
    return range(phase-STRIDE,n,STRIDE)

def pass_once(net,probe,a,phase):
    h,w=a.shape
    coords=[(y,x) for y in starts(h,phase,probe) for x in starts(w,phase,probe)]
    sums=np.zeros((4,h,w),np.float32)
    for start in range(0,len(coords),16):
        sub=coords[start:start+16]
        tiles=[]
        items=[]
        for y,x in sub:
            y0,x0=max(0,y),max(0,x)
            y1,x1=min(h,y+SIZE),min(w,x+SIZE)
            dest=np.s_[y0:y1,x0:x1]
            inner=np.s_[y0-y:y1-y,x0-x:x1-x]
            tile=np.full((SIZE,SIZE),127.5,np.float32)
            tile[inner]=a[dest]
            tiles.append(tile)
            items.append((dest,inner))
        t=prev.inp(np.stack(tiles))
        prob=probe.forward(net,t).softmax(1)[:,1].cpu().numpy()
        for p,(dst,src) in zip(prob,items):
            v=p[src];g=gaussian[src]
            sums[0][dst]+=v
            sums[1][dst]+=1
            sums[2][dst]+=v*g
            sums[3][dst]+=g
    assert (sums[1]>0).all() and (sums[3]>0).all()
    return sums[0]/sums[1],sums[2]/sums[3]

def evaluate(mask,gt):
    if gt is None:return (None,None)
    tp=np.count_nonzero(mask&gt)
    union=np.count_nonzero(mask|gt)
    return (2*tp/(union+tp) if union else 1.,tp/union if union else 1.)

def matches(box,part,other_labels,other_area,area):
    counts=np.bincount(other_labels[box][part],minlength=len(other_area))
    reach=counts[1:]/np.maximum(1,area+other_area[1:]-counts[1:])
    return float(max(reach,default=0.))

def stats(name,key,domain,config,mask,gt,base,moved,detail):
    squares,lab=prev.shapes(mask)
    indexes={i for i,b,area in squares}
    hole=holes_of(mask)
    orig=base[0]
    shift=moved[0]
    shifted_count=old_count=total=new_shift=new_old=new_edges=persistent=0
    for i,box,area in squares:
        part=lab[box]==i
        raw=[box[0].start,box[1].start,box[0].stop,box[1].stop]
        s=sum((a-8)%16<=1 for a in raw)
        o=sum(a%16<=1 for a in raw)
        shifted_count+=s;old_count+=o;total+=4
        b_iou=matches(box,part,orig,base[1],area)
        s_iou=matches(box,part,shift,moved[1],area)
        persistent+=int(b_iou>=.5 and s_iou>=.5)
        if b_iou<.5:
            new_shift+=s;new_old+=o;new_edges+=4
        detail.append(dict(model=name,sample=key,domain=domain,config=config,hole=i,
                           y0=raw[0],x0=raw[1],y1_excl=raw[2],x1_excl=raw[3],area=area,
                           matches_baseline_iou=b_iou,matches_shift_iou=s_iou,
                           new_phase_edges=s,old_phase_edges=o))
    h=int(hole.sum())
    fg=int(mask.sum())
    dice,iou=evaluate(mask,gt)
    return dict(model=name,sample=key,domain=domain,config=config,fg_area=fg,hole_area=h,hole_ratio=h/max(1,fg+h),
             squares=len(indexes),edges=total,shift_edges=shifted_count,old_edges=old_count,new_hole_edges=new_edges,
             new_hole_shift_edges=new_shift,new_hole_old_edges=new_old,persistent_squares=persistent,dice=dice,iou=iou)

def square_labels(mask):
    items,labels=prev.shapes(mask)
    valid=np.zeros(labels.max()+1,bool)
    valid[[i for i,b,a in items]]=True
    labels=np.where(valid[labels],labels,0)
    return labels,np.bincount(labels.ravel())

def samples():
    ff={r['id']:r for r in csv.DictReader((DATA/'manifest.csv').open(encoding='utf-8-sig'))}
    for q in read(OUT.parent/'probe2_top_fov.json')['selected']:
        key=q['fov']
        r=ff[key]
        im=DATA/r['file']
        assert sha(im)==r['file_sha256']
        yield key,'HQ',tifffile.imread(im),None
    for q in read(OUT.parent/'sample_manifest.json'):
        if q['domain']!='MB1':continue
        z=q['source']['z']
        key=q['id']
        id=key.removeprefix('MB1_').rsplit('_z',1)[0]
        r=next(x for x in read(MB/'MB1_benchmark_manifest.json')['entries'] if x['id']==id)
        a,b=[Planes(r[f],r['shape_zyx']) for f in ('image_path','label_path')]
        yield key,'MB1',np.array(a[z]),b[z]>0
        a.close()
        b.close()

def write(file,rows):
    with (OUT/file).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

@torch.no_grad()
def main():
    gpu=gpu_guard()
    t=time.monotonic()
    torch.set_num_threads(2)
    detail=[]; rows=[]
    for name in ('P0','Barrier100'):
        net,p,enc=primus(name)
        folder=BASE/'batch4/primus_cem'/name
        ck=folder/'model_40000.pt'
        meta=read(folder/'trained.json')
        assert sha(ck)==meta['checkpoint_sha'] and enc==meta['init_encoder_sha'] and meta['seed']==20261014
        miss,extra=net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=False)
        assert not extra and all(x.startswith('dino_encoder.') for x in miss)
        net.eval()
        for key,domain,a,gt in samples():
            base,g0=pass_once(net,p,a,0)
            shift,g8=pass_once(net,p,a,8)
            masks={k:q>.5 for k,q in [('Baseline',base),('Shift-8',shift),('Gauss-Center',g0),('Gauss+Dual-Phase',(g0+g8)/2)]}
            if domain=='HQ':
                assert np.array_equal(masks['Baseline'],mask_at('original',f'{name}/{key}.png'))
            b,s=map(square_labels,(masks['Baseline'],masks['Shift-8']))
            for config,mask in masks.items():
                rows.append(stats(name,key,domain,config,mask,gt,b,s,detail))
            print(name,key,'complete',flush=True)
        del net
        torch.cuda.empty_cache()
    assert len(rows)==80
    write('per_image.csv',rows)
    write('per_hole.csv',detail)
    (OUT/'receipt.json').write_text(json.dumps(dict(gpu=gpu,seconds=time.monotonic()-t,rows=len(rows)),indent=2))

if __name__=='__main__':
    main()
