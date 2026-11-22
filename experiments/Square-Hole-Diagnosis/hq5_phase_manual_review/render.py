import csv,json,sys,time
from collections import Counter
from pathlib import Path
from zipfile import ZipFile,ZIP_STORED
import numpy as np
import torch,tifffile
from PIL import Image,ImageDraw,ImageFont
OUT=Path(__file__).resolve().parent
REFERENCE=OUT.parent/'full_phase_eval'
sys.path.insert(0,str(REFERENCE))
import importlib.util
spec=importlib.util.spec_from_file_location("full_eval_reference",REFERENCE/"run.py")
evalrun=importlib.util.module_from_spec(spec)
spec.loader.exec_module(evalrun)

labels=('Baseline','Gauss + Stride-120','Gauss + Dual-Phase','Gauss + Quad-Phase')
slugs=('baseline','stride120','dual_phase','quad_phase')
red=(255,48,48)
resample=Image.Resampling.NEAREST

def write(f,rows):
    with f.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def review_strip(id,img,plots):
    size=640
    pad=12
    canvas=Image.new('RGB',((size+pad)*5+pad,size+90),'#191919')
    draw=ImageDraw.Draw(canvas)
    title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',22)
    ui=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    draw.text((pad,8),f'{id}  |  Barrier100k + Primus  |  5 nm/px',fill='white',font=title)
    draw.rectangle((pad,43,pad+18,59),fill=red)
    draw.text((pad+26,40),'Red = detected square-like enclosed hole      White = predicted mitochondria      Black = other pixels',fill='white',font=ui)
    a=Image.fromarray(img).convert('RGB')
    for n,(name,im) in enumerate([('Original EM',a),*plots]):
        x=pad+n*(size+pad)
        draw.text((x,65),name,fill='white',font=ui)
        canvas.paste(im.resize((size,size),resample if n else Image.Resampling.LANCZOS),(x,88))
    canvas.save(OUT/'comparisons'/f'{id}.png')

@torch.no_grad()
def main():
    started=time.monotonic()
    gpu=evalrun.gpu_guard()
    torch.set_num_threads(2)
    m=evalrun.read(REFERENCE/'protocol.json')
    net,p,enc=evalrun.primus('Barrier100')
    ck=Path(m['decoder_path'])
    assert enc==m['encoder_sha256'] and evalrun.sha(ck)==m['decoder_sha256']
    missing,extra=net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=False)
    assert not extra and all(x.startswith('dino_encoder.') for x in missing)
    net.eval()
    with (evalrun.DATA/'manifest.csv').open(encoding='utf-8-sig') as f:
        manifest={r['id']:r for r in csv.DictReader(f)}
    with (REFERENCE/'per_image.csv').open() as f:
        ref={(r['id'],r['config']):r for r in csv.DictReader(f) if r['dataset']=='HQ'}
    ids=sorted({id for id,config in ref})
    assert len(ref)==80 and len(ids)==20
    (OUT/'comparisons').mkdir(exist_ok=True)
    for name in slugs:
        (OUT/'annotated_masks'/name).mkdir(parents=True,exist_ok=True)
    holes=[];counts=[];paths=[]
    for id in ids:
        r=manifest[id];source=evalrun.DATA/r['file']
        assert evalrun.sha(source)==r['file_sha256']
        a=tifffile.imread(source)
        b,g0,_,_=evalrun.measure_pass(net,p,a,128,(0,0),True)
        _,g120,_,_=evalrun.measure_pass(net,p,a,120,(0,0))
        _,g8,_,_=evalrun.measure_pass(net,p,a,128,(8,8))
        _,gx,_,_=evalrun.measure_pass(net,p,a,128,(0,8))
        _,gy,_,_=evalrun.measure_pass(net,p,a,128,(8,0))
        masks=(b>.5,g120>.5,(g0+g8)>1,(g0+g8+gx+gy)>2)
        assert np.array_equal(masks[0],evalrun.mask_at('original','Barrier100/'+id+'.png'))
        plots=[]
        for label,slug,mask in zip(labels,slugs,masks):
            previous=ref[id,label]
            actual=evalrun.topo(mask)
            assert all(actual[k]==int(previous[k]) for k in ('fg_area_px','hole_area_px','squares','holes'))
            components,lab=evalrun.tr.shapes(mask)
            assert len(components)==actual['squares']
            highlight=np.isin(lab,[n for n,_,_ in components])
            rgb=np.repeat((mask.astype(np.uint8)*255)[...,None],3,axis=2);rgb[highlight]=red
            picture=Image.fromarray(rgb)
            file=OUT/'annotated_masks'/slug/f'{id}.png'
            picture.save(file)
            paths.append(file)
            plots.append((f'{label}  ({len(components)} holes)',picture))
            counts.append(dict(fov=id,config=label,square_holes=len(components),png=file.relative_to(OUT).as_posix()))
            for n,box,area in components:
                y0,y1=box[0].start,box[0].stop
                x0,x1=box[1].start,box[1].stop
                h,w=y1-y0,x1-x0
                holes.append(dict(fov=id,config=label,component_id=n,x0=x0,y0=y0,x1_exclusive=x1,y1_exclusive=y1,height=h,width=w,area_px=area,fill=area/(h*w),aspect=min(h,w)/max(h,w)))
        review_strip(id,a,plots)
        paths.append(OUT/'comparisons'/f'{id}.png')
        print(id,'confirmed',flush=True)
    assert len(paths)==100 and len(counts)==80
    assert Counter(r['config'] for r in holes)==dict(zip(labels,(139,55,28,22)))
    write(OUT/'square_holes.csv',holes)
    write(OUT/'image_index.csv',counts)
    (OUT/'README.txt').write_text('C1 Barrier100k + Primus, 20 paired HQ-5nm FOVs.\nRed: only enclosed 4-connected holes with fill >= 0.75, aspect >= 0.75 and short side 12–68 native px.\nWhite: predicted mitochondria. Black: background and any other holes. The PNG masks are annotated for review.\nEach native mask is 2048 x 2048. Comparisons include the original EM view at reduced display size.\nNo GT is available. Counts and foreground and hole areas match the original full evaluation.\n',encoding='utf-8')
    target=OUT/'HQ5nm_C1_phase_square_holes_review.zip'
    with ZipFile(target.with_suffix('.tmp'),'w',ZIP_STORED) as z:
        for f in paths+[OUT/'square_holes.csv',OUT/'image_index.csv',OUT/'README.txt']:
            z.write(f,f.relative_to(OUT).as_posix())
    with ZipFile(target.with_suffix('.tmp')) as z:
        assert z.testzip() is None and len(z.namelist())==103
    target.with_suffix('.tmp').replace(target)
    (OUT/'receipt.json').write_text(json.dumps(dict(gpu=gpu,seconds=time.monotonic()-started,hole_counts=dict(Counter(r['config'] for r in holes)),native_png=80,comparison_png=20),indent=2))
    print(target,flush=True)

if __name__=='__main__':
    main()
