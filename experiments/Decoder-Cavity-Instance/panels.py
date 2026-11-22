import csv,sys
import torch,numpy as np
from PIL import Image,ImageDraw,ImageFont
from model import OUT,BASE,ARMS,primus,read,sha,gpu_guard,MB
from inference import load,predict
from common import Planes
D=OUT.parent/'square_hole_diagnosis_20261122/gt_interior_holes'
sys.path.insert(0,str(D))
from core import gt_maps,metrics

@torch.no_grad()
def old_model():
    net,p,enc=primus('Barrier100')
    folder=BASE/'batch4/primus_cem/Barrier100'
    rec=read(folder/'trained.json');ck=folder/'model_40000.pt'
    assert sha(ck)==rec['checkpoint_sha'] and enc==rec['init_encoder_sha']
    missing,extra=net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=False)
    assert not extra and all(x.startswith('dino_encoder.') for x in missing)
    net.eval()
    return net,p

def draw(path,a,group,id,z,overlays,titles):
    rgb=np.array([[0,0,0],[32,228,70],[255,225,0],[255,48,48],[30,130,255],[170,170,170]],np.uint8)
    sz=550; pad=10
    h=max(1,round(a.shape[0]*sz/a.shape[1]))
    h=min(h,660)
    canvas=Image.new('RGB',(4*(sz+pad)+pad,h+90),'#161616')
    d=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    d.text((pad,6),f'{group} / {id} z={z}',font=font,fill='white')
    d.text((pad,29),'GREEN TP  /  YELLOW boundary FN  /  RED interior FN  /  BLUE background FP  /  GRAY missed instance',font=font,fill='white')
    images=[Image.fromarray(a).convert('RGB')]+[Image.fromarray(rgb[x]) for x in overlays]
    for j,(im,t) in enumerate(zip(images,['Original EM',*titles])):
        x=pad+j*(sz+pad)
        d.text((x,60),t,font=font,fill='white')
        canvas.paste(im.resize((sz,h),Image.Resampling.NEAREST if j else Image.Resampling.LANCZOS),(x,85))
    canvas.save(path)

@torch.no_grad()
def main():
    if len(list((OUT/'visual_panels').glob('*.png')))==12 and len(list((OUT/'visual_panels_extra').glob('*.png')))==2:
        return
    gpu_guard()
    torch.set_num_threads(2)
    import importlib.util
    spec=importlib.util.spec_from_file_location('phase_reference',D.parent/'full_phase_eval/run.py')
    ref=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    net0,p0=old_model()
    nets=[load(a)[0] for a in ARMS]
    with (D/'visual_index.csv').open() as f:cases=list(csv.DictReader(f))
    assert len(cases)==12
    def fly_severity(row):return int(row['interior_fn_px'])/max(1,int(row['interior_px']))
    with (D/'per_plane.csv').open() as f:fly=sorted([r for r in csv.DictReader(f) if r['config']=='C1 Stride-120' and r['group']=='fly_brain' and r['radius']=='3'],key=fly_severity,reverse=True)[:2]
    refs={r['id']:r for r in read(MB/'MB1_benchmark_manifest.json')['entries']}
    for i,r in enumerate(cases+fly):
        d=OUT/('visual_panels' if i<12 else 'visual_panels_extra')
        d.mkdir(exist_ok=True)
        k=r['id'];z=int(r['z']); e=refs[k]
        a,b=[Planes(e[x],e['shape_zyx']) for x in ('image_path','label_path')]
        im,gt=a[z],gt_maps(b[z])
        _,old,_,_=ref.measure_pass(net0,p0,im,120,(0,0))
        masks=[old>.5]+[predict(n,im,'stride120')[0]>.5 for n in nets]
        outputs=[metrics(gt,m) for m in masks]
        titles=[]
        for t,(r0,_) in zip(('Old Primus','A MultiScale-Sem','B MultiScale-BCD'),outputs):
            x=r0[0];ratio=x['interior_fn_px']/max(1,x['interior_px'])
            titles.append(f'{t} · r3 FN {ratio:.1%}')
        draw(d/f'{k}_z{z}.png',im,e['dataset'],k,z,[x[1] for x in outputs],titles)
        a.close();b.close()
    print('12 original review panels plus 2 fly brain',flush=True)
if __name__=='__main__':
    main()
