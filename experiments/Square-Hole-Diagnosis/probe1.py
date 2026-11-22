import csv,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage as nd
import tifffile,torch
from probe2 import ROOT,mask_at,holes_of,CONN,write

BASE=ROOT.parent/'mix_mitolab_mv_20261118'
HRA=ROOT.parent/'c0_c1_gram_hra_20261120'
DATA=Path('/opt/data/3_117/zhangyc/Large-Field-Workspace/mitos/HQ-5nm_20261120')
sys.path.insert(0,str(HRA/'mb1_130k_seed20261014'))
import run as prior
from c0_matched import encoder as c0
from common import Planes,gpu_guard
from mixture import read,sha,MB
from fit import primus

def crop(key,domain,a,p,h,g,source):
    labels,_=nd.label(h,structure=CONN)
    count=np.bincount(labels.ravel())
    candidates=[]
    more=[]
    for i,b in enumerate(nd.find_objects(labels),1):
        j,k=[s.stop-s.start for s in b]
        lo,hi=sorted((j,k))
        pos=tuple((s.start+s.stop)//2 for s in b)
        (candidates if 12<=lo<=68 and lo/hi>=.75 and count[i]/(j*k)>=.75 else more).append(pos)
    def box(y,x):
        sy,sx=[min(max(0,v-128),max(0,n-256)) for v,n in zip((y,x),a.shape)]
        return np.s_[sy:sy+256,sx:sx+256]
    reg=max((box(y,x) for y,x in (candidates or more)),key=lambda b:h[b].sum())
    def side(x,v):
        if x is None:return None
        k=x[reg]
        return np.pad(k,((0,256-k.shape[0]),(0,256-k.shape[1])),constant_values=v)
    return dict(id=key,domain=domain,a=side(a,127.5),mask=side(p,False),holes=side(h,False),gt=side(g,False),valid=side(np.ones(a.shape,bool),False),origin=[s.start for s in reg],source=source)

def model(name):
    if name=='P0':return primus('P0')
    if name=='C0_100k':return c0('C0_Vanilla_100k')
    if name=='C1_100k':return primus('Barrier100')
    return prior.build('C0_Vanilla_100k' if name=='C0_130k_HRA' else 'C1_Barrier_100k')

@torch.no_grad()
def samples():
    rows=read(ROOT/'probe2_top_fov.json')['selected']
    data={r['id']:r for r in csv.DictReader((DATA/'manifest.csv').open(encoding='utf-8-sig'))}
    out=[]
    for r in rows:
        src=DATA/data[r['fov']]['file']
        assert sha(src)==data[r['fov']]['file_sha256']
        a=tifffile.imread(src)
        p=mask_at(r['pack'],r['entry'])
        out.append(crop(r['fov'],'HQ',a,p,holes_of(p),None,r))
    # Per-unit records contain metrics only. Recreate masks from the locked C1 HRA
    # 512 pipeline on the previous fixed 52-plane sample, without threshold changes.
    name='C1_Barrier_100k'
    net,probe,digest=prior.build(name)
    base=HRA/'mb1_130k_seed20261014'/name
    ck=base/'model_40000.pt'
    receipt=read(base/'trained.json')
    assert sha(ck)==receipt['checkpoint_sha'] and digest==receipt['init_encoder_sha']
    missing,extra=net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=False)
    assert not extra and all(k.startswith('dino_encoder.') for k in missing)
    net.eval()
    reference=read(ROOT.parent/'mb1_matched_window_20261120/P0.json')['rows']
    wanted={(r['id'],r['z']) for r in reference if r['window']==512}
    candidates=[]
    for r in read(MB/'MB1_benchmark_manifest.json')['entries']:
        zs=sorted(z for k,z in wanted if k==r['id'])
        if not zs:continue
        a,b=[Planes(r[key],r['shape_zyx']) for key in ('image_path','label_path')]
        for z in zs:
            raw=np.array(a[z]); gt=b[z]>0
            pred=prior.slide512(net,probe,raw)
            lab,_=nd.label(holes_of(pred),structure=CONN)
            count=np.bincount(lab.ravel())
            hits=np.bincount(lab[gt].ravel(),minlength=len(count))
            use=np.flatnonzero((hits>=24)&(hits>=.5*count))
            use=use[use>0]
            if len(use):
                key=f"MB1_{r['id']}_z{z}"
                score=int(hits[use].sum())
                sample=crop(key,'MB1',raw,pred,np.isin(lab,use),gt,dict(group=r['dataset'],z=z,reference='C1_130k_512',gt_inside_hole_px=score))
                candidates.append((score,r['dataset'],sample))
        a.close();b.close()
    del net
    torch.cuda.empty_cache()
    seen=set(); chosen=[]
    candidates.sort(key=lambda x:(-x[0],x[1],x[2]['id']))
    for _,g,s in candidates:
        if g not in seen:chosen.append(s);seen.add(g)
        if len(chosen)==4:break
    if len(chosen)<4:
        have={s['id'] for s in chosen}
        chosen += [s for _,_,s in candidates if s['id'] not in have][:4-len(chosen)]
    out+=chosen
    (ROOT/'sample_manifest.json').write_text(json.dumps([{k:s[k] for k in ('id','domain','origin','source')} for s in out],indent=2))
    print('samples',len(out),[s['id'] for s in chosen],flush=True)
    return out

def fractions(a):
    return a.reshape(16,16,16,16).mean(axis=(1,3)).ravel()

def scaled(v,lo=None,hi=None):
    if lo is None:lo,hi=np.percentile(v,[2,98])
    return np.clip((v-lo)/max(float(hi-lo),1e-8),0,1)

def heat(a,lo=None,hi=None):
    v=scaled(a,lo,hi)
    return np.stack([v,1-np.abs(2*v-1),1-v],axis=-1)

def tile(a):
    a=(a*255).astype(np.uint8)
    if a.ndim==2:a=np.repeat(a[...,None],3,axis=-1)
    return Image.fromarray(a).resize((216,216),Image.Resampling.NEAREST)

@torch.no_grad()
def main():
    gpu=gpu_guard()
    begin=time.monotonic()
    torch.set_num_threads(2)
    data=samples()
    stats=[]; panels={s['id']:[] for s in data}; hashes={}; missing={}
    for name in ('P0','C0_100k','C1_100k','C1_130k_HRA','C0_130k_HRA'):
        try:net,_,digest=model(name)
        except FileNotFoundError as e:
            missing[name]=str(e);continue
        enc=net.dino_encoder.eval(); hashes[name]=digest
        assert enc.patch_size==16 and enc.n_storage_tokens==4
        for s in data:
            a=s['a'].astype(np.float32)
            t=torch.from_numpy((a/255-.5)/.25)[None,None].repeat(1,3,1,1).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                raw=enc.get_intermediate_layers(t,n=[23],norm=False)[0]
                normed=enc.norm(raw)
            assert raw.shape==(1,(a.shape[0]//16)*(a.shape[1]//16),1024)
            f=normed[0].float().cpu().numpy()
            norms=np.linalg.norm(raw[0].float().cpu().numpy(),axis=1)
            h,p,v=[fractions(s[k]) for k in ('holes','mask','valid')]
            valid=v==1
            hole=(h>=.125)&valid
            mito=(p>=.5)&(h<=.05)&valid
            bg=(p<=.05)&(h<=.05)&valid
            proto=f[mito].mean(axis=0) if mito.any() else np.full(f.shape[1],np.nan)
            cos=(f@proto)/np.maximum(np.linalg.norm(f,axis=1)*np.linalg.norm(proto),1e-12)
            threshold=norms[valid].mean()+2*norms[valid].std()
            _,_,vec=np.linalg.svd(f[valid]-f[valid].mean(0),full_matrices=False)
            xyz=(f-f[valid].mean(0))@vec[:3].T
            scale=np.sqrt(np.mean((xyz[valid]-xyz[valid].mean(0))**2))
            dist=np.linalg.norm(xyz-xyz[mito].mean(0),axis=1)/max(scale,1e-12) if mito.any() else np.full(len(f),np.nan)
            for label,sel in [('hole',hole),('normal_mito',mito),('bg',bg)]:
                good=sel.any()
                stats.append(dict(model=name,sample=s['id'],domain=s['domain'],region=label,n=int(sel.sum()),
                    norm_mean=float(norms[sel].mean()) if good else None,norm_p95=float(np.percentile(norms[sel],95)) if good else None,
                    norm_high_count=int((norms[sel]>threshold).sum()),norm_high_fraction=float((norms[sel]>threshold).mean()) if good else None,
                    norm_threshold=float(threshold),cos_mean=float(cos[sel].mean()) if good and mito.any() else None,
                    pca_proto_distance=float(dist[sel].mean()) if good and mito.any() else None))
            if not mito.any():cos=np.zeros_like(cos)
            rgb=np.stack([scaled(xyz[:,i]) for i in range(3)],-1)
            panels[s['id']].append((name,heat(norms.reshape(16,16)),heat(cos.reshape(16,16),-1,1),rgb.reshape(16,16,3)))
        del net,enc
        torch.cuda.empty_cache()
        print(name,'done',flush=True)
    assert stats
    write('probe1_patch_stats.csv',stats)
    for s in data:
        art=[('Raw',tile(scaled(s['a'])))]
        mask=np.repeat(s['mask'][...,None],3,axis=2).astype(float)
        mask[s['holes']]=[1,0,0]
        art.append(('Mask / holes in red',tile(mask)))
        if s['gt'] is not None:art.append(('GT',tile(s['gt'].astype(float))))
        for name,n,c,p in panels[s['id']]:
            art.extend([(name+' L2',tile(n)),(name+' cosine',tile(c)),(name+' PCA',tile(p))])
        canvas=Image.new('RGB',(216*len(art),240),'#171717')
        draw=ImageDraw.Draw(canvas)
        for i,(label,panel) in enumerate(art):
            canvas.paste(panel,(216*i,24))
            draw.text((216*i+4,5),label,fill='white')
        canvas.save(ROOT/'panels'/(s['id']+'_comparison.png'))
    (ROOT/'probe1_receipt.json').write_text(json.dumps(dict(gpu=gpu,seconds=time.monotonic()-begin,checkpoint_hashes=hashes,missing_models=missing,samples=len(data)),indent=2))
    print('probe1 complete',len(stats),flush=True)

if __name__=='__main__':
    main()
