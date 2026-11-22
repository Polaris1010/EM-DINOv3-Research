import sys,csv,time,json
from pathlib import Path
import numpy as np
import torch,tifffile
from PIL import Image,ImageDraw
from scipy import ndimage as nd
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent))
from probe1 import DATA,BASE,primus,gpu_guard,read,sha,tile,heat,scaled
from probe2 import holes_of,mask_at,CONN

def shapes(mask):
    lab,_=nd.label(holes_of(mask),structure=CONN)
    a=np.bincount(lab.ravel())
    rows=[]
    for i,b in enumerate(nd.find_objects(lab),1):
        h,w=[v.stop-v.start for v in b]
        lo,hi=sorted((h,w))
        if 12<=lo<=68 and lo/hi>=.75 and a[i]/(h*w)>=.75:
            rows.append((i,b,int(a[i])))
    return rows,lab

def inp(a):
    return torch.from_numpy((np.asarray(a,dtype=np.float32)/255-.5)/.25)[:,None].cuda()

def panel(out,net,a,mask,component,logit,prob,origin):
    y,x=origin
    roi=np.s_[y:y+256,x:x+256]
    raw=a[roi]
    t=inp(raw[None]).repeat(1,3,1,1)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        f=net.dino_encoder.get_intermediate_layers(t,n=[23])[0][0].float()
    assert f.shape==(256,1024)
    selected=mask[roi].reshape(16,16,16,16).mean(axis=(1,3)).ravel()>=.5
    f=torch.nn.functional.normalize(f,dim=1)
    proto=torch.nn.functional.normalize(f[selected].mean(0),dim=0) if selected.any() else torch.zeros_like(f[0])
    cos=(f@proto).cpu().numpy().reshape(16,16)
    overlay=np.repeat(mask[roi][...,None],3,axis=2).astype(float)
    overlay[component[roi]]=[1,0,0]
    art=[('Original',tile(scaled(raw))),('Reference hole (red)',tile(overlay)),('Encoder cosine',tile(heat(cos,-1,1))),
         ('Single-window logit (zero=center)',tile(heat(logit,-5,5))),('Fused foreground probability',tile(heat(prob[roi],0,1)))]
    canvas=Image.new('RGB',(216*len(art),240),'#171717')
    draw=ImageDraw.Draw(canvas)
    for i,(title,img) in enumerate(art):
        canvas.paste(img,(216*i,24))
        draw.text((216*i+4,5),title,fill='white')
    canvas.save(out)
    h=component[roi].reshape(16,16,16,16).mean(axis=(1,3))>0
    return float(cos[h].mean()),float(cos.ravel()[selected].mean()) if selected.any() else None

def replay(net,probe,a):
    coord=[(y,x) for y in probe.starts(a.shape[0]) for x in probe.starts(a.shape[1])]
    acc=np.zeros(a.shape,np.float32)
    count=np.zeros(a.shape,np.uint8)
    sign=np.zeros(a.shape,np.uint8)
    logits={}
    for start in range(0,len(coord),16):
        cs=coord[start:start+16]
        t=inp(np.stack([a[y:y+256,x:x+256] for y,x in cs]))
        z=probe.forward(net,t)
        p=z.softmax(1)[:,1].cpu().numpy()
        v=(z[:,1]-z[:,0]).cpu().numpy()
        for (y,x),q,l in zip(cs,p,v):
            roi=np.s_[y:y+256,x:x+256]
            acc[roi]+=q
            count[roi]+=1
            sign[roi]+=l>0
            logits[y,x]=l
    return acc/count,sign,coord,logits

def describe(name,fov,a,net,mask,p,sign,coord,logits,components,lab,diff):
    painted=False
    for i,b,area in sorted(components,key=lambda c:(-c[2],c[0])):
        sy,sx=(v.start for v in b)
        ey,ex=(v.stop for v in b)
        h,w=ey-sy,ex-sx
        sub=lab[b]==i
        fits=[(y,x) for y,x in coord if y<=sy and x<=sx and y+256>=ey and x+256>=ex]
        cy,cx=(sy+ey)/2,(sx+ex)/2
        choice=min(fits,key=lambda t:sum((u+128-v)**2 for u,v in zip(t,(cy,cx)))) if fits else None
        info={}
        for y,x in fits:
            l=logits[y,x]; sl=np.s_[sy-y:ey-y,sx-x:ex-x]
            sublog=l[sl][sub]
            en=holes_of(l>0)[sl][sub]
            info[y,x]=(float((sublog<=0).mean()),float(en.mean()),float(sublog.mean()))
        cos_h=cos_m=None
        if not painted and choice:
            cos_h,cos_m=panel(OUT/'panels'/f'{name}_{fov}.png',net,a,mask,lab==i,logits[choice],p,choice)
            painted=True
        borders=(sy,sx,ey,ex)
        selected=info.get(choice,(None,None,None))
        yield dict(model=name,fov=fov,hole=i,y0=sy,x0=sx,y1_excl=ey,x1_excl=ex,height=h,width=w,area=area,
            y0_mod16=sy%16,x0_mod16=sx%16,y1_mod16=ey%16,x1_mod16=ex%16,y0_mod128=sy%128,x0_mod128=sx%128,
            y1_mod128=ey%128,x1_mod128=ex%128,borders_within1_patch=sum(min(x%16,(-x)%16)<=1 for x in borders),
            borders_within1_stride=sum(min(x%128,(-x)%128)<=1 for x in borders),
            fully_covering_tiles=len(fits),all_votes_negative_fraction=float((sign[b][sub]==0).mean()),
            any_positive_vote_fraction=float((sign[b][sub]>0).mean()),single_negative_fraction=selected[0],
            single_enclosed_fraction=selected[1],single_mean_logit=selected[2],
            any_tile_max_enclosed_fraction=max((v[1] for v in info.values()),default=None),
            fused_foreground_fraction=float((p[b][sub]>.5).mean()),fused_mean_probability=float(p[b][sub].mean()),
            panel_cos_hole=cos_h,panel_cos_normal=cos_m,mask_replay_differing_pixels=diff)

@torch.no_grad()
def main():
    card=gpu_guard()
    torch.set_num_threads(2)
    begin=time.monotonic()
    ff={r['id']:r for r in csv.DictReader((DATA/'manifest.csv').open(encoding='utf-8-sig'))}
    top=read(OUT.parent/'probe2_top_fov.json')['selected']
    rows=[]; images=[]
    for name in ('P0','Barrier100'):
        net,probe,enc=primus(name)
        folder=BASE/'batch4/primus_cem'/name
        ck=folder/'model_40000.pt'
        meta=read(folder/'trained.json')
        assert sha(ck)==meta['checkpoint_sha'] and enc==meta['init_encoder_sha'] and meta['seed']==20261014
        missing,extra=net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=False)
        assert not extra and all(x.startswith('dino_encoder.') for x in missing)
        net.eval()
        for sample in top:
            fov=sample['fov']
            r=ff[fov]
            path=DATA/r['file']
            assert sha(path)==r['file_sha256']
            a=tifffile.imread(path)
            mask=mask_at('original',name+'/'+fov+'.png')
            components,lab=shapes(mask)
            p,sign,coord,logits=replay(net,probe,a)
            diff=int(np.count_nonzero((p>.5)!=mask))
            rows.extend(describe(name,fov,a,net,mask,p,sign,coord,logits,components,lab,diff))
            images.append(dict(model=name,fov=fov,square_holes=len(components),mask_replay_differing_pixels=diff))
            print(name,fov,len(components),diff,flush=True)
        del net
        torch.cuda.empty_cache()
    with (OUT/'square_holes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT/'receipt.json').write_text(json.dumps(dict(gpu=card,seconds=time.monotonic()-begin,images=images),indent=2))

if __name__=='__main__':
    main()
