from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv,io,json,time
from zipfile import ZipFile
import numpy as np
from PIL import Image
from scipy import ndimage as nd

ROOT=Path(__file__).resolve().parent
PACKS={
 'original':'HQ-5nm_80_predictions_PNG.zip',
 'half_input':'HQ-5nm_half_input_80_predictions_PNG.zip',
 'large_windows':'HQ-5nm_Primus_512_1024_predictions_PNG.zip',
 '130k':'HQ5nm_C0_C1_130k_256_512_PNG.zip'}
KEYS=('fg_area_px','hole_area_px','hole_area_ratio','num_internal_holes','num_square_like_holes','median_hole_area_px','p90_hole_area_px')
CONN=nd.generate_binary_structure(2,1)

def mask_at(pack,entry):
    with ZipFile(ROOT/'inputs'/PACKS[pack]) as f:
        with Image.open(io.BytesIO(f.read(entry))) as im:
            a=np.asarray(im)
            assert a.shape==(2048,2048)
            return a>0

def holes_of(mask):
    return nd.binary_fill_holes(mask,structure=CONN)&~mask

def measure(item):
    pack,entry=item
    parts=entry.split('/')
    model=parts[1] if pack=='large_windows' else parts[0]
    w=int(parts[0] if pack=='large_windows' else parts[1] if pack=='130k' else 0 if model=='Official-MitoNet-v1' else 256)
    mask=mask_at(pack,entry)
    holes=holes_of(mask)
    labels,n=nd.label(holes,structure=CONN)
    area=np.bincount(labels.ravel())[1:]
    sq=0
    for a,box in zip(area,nd.find_objects(labels)):
        h,x=[s.stop-s.start for s in box]
        sm,lg=sorted((h,x))
        sq+=int(12<=sm<=68 and sm/lg>=.75 and a/(h*x)>=.75)
    fg=int(mask.sum()); h=int(area.sum())
    return dict(pack=pack,model=model,window=w,fov=Path(entry).stem,entry=entry,fg_area_px=fg,hole_area_px=h,hole_area_ratio=h/max(1,fg+h),num_internal_holes=n,num_square_like_holes=sq,median_hole_area_px=float(np.median(area)) if n else 0.,p90_hole_area_px=float(np.percentile(area,90)) if n else 0.)

def write(name,rows):
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def main():
    begin=time.monotonic()
    items=[]
    for k,v in PACKS.items():
        with ZipFile(ROOT/'inputs'/v) as f:
            assert f.testzip() is None
            items += [(k,n) for n in sorted(f.namelist()) if n.endswith('.png')]
    assert len(items)==380
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(measure,items,chunksize=3))
    write('probe2_hq5nm_holes_per_image.csv',rows)
    by=defaultdict(list)
    for r in rows:
        by[r['pack'],r['model'],r['window']].append(r)
    common=set.intersection(*[{r['fov'] for r in rs} for rs in by.values()])
    assert len(common)==20
    summary=[]
    for (k,m,w),rs in sorted(by.items()):
        for scope,sub in [('common20',[r for r in rs if r['fov'] in common])]+([('full25',rs)] if k=='130k' else []):
            row=dict(pack=k,model=m,window=w,scope=scope,n=len(sub))
            for metric in KEYS:
                v=[r[metric] for r in sub]
                row.update({metric+'_mean':float(np.mean(v)),metric+'_sd':float(np.std(v,ddof=1)),metric+'_median':float(np.median(v))})
            summary.append(row)
    write('probe2_hq5nm_holes_summary.csv',summary)
    best={}
    for metric in ('hole_area_ratio','num_square_like_holes'):
        d={}
        for r in sorted((r for r in rows if r['fov'] in common),key=lambda r:(-r[metric],r['fov'],r['pack'],r['entry'])):
            d.setdefault(r['fov'],r)
        best[metric]=d
    ranks={metric:{fov:i for i,fov in enumerate(rs,1)} for metric,rs in best.items()}
    chosen=sorted(common,key=lambda f:(sum(rs[f] for rs in ranks.values()),max(rs[f] for rs in ranks.values()),f))[:6]
    picks=[]
    for f in chosen:
        # Use the largest count of square-like holes as reference; ties use highest hole ratio.
        r=max((r for r in rows if r['fov']==f),key=lambda r:(r['num_square_like_holes'],r['hole_area_ratio']))
        picks.append(dict(r,ratio_rank=ranks['hole_area_ratio'][f],square_rank=ranks['num_square_like_holes'][f]))
    (ROOT/'probe2_top_fov.json').write_text(json.dumps(dict(selected=picks,individual_top6={k:list(v.values())[:6] for k,v in best.items()},seconds=time.monotonic()-begin),indent=2))
    print('probe2 complete',len(rows),len(summary),round(time.monotonic()-begin,2),flush=True)

if __name__=='__main__':
    main()
