"""CPU-only review of saved binary predictions; never loads models or changes them."""
import ast,csv,hashlib,json,struct,time
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path('/opt/data/3_117/tanghr/codex')
OUT=R/'d12_muscle_review_20261020'
ROOTS={'30k':R/'ssl_diversity_d12_20261017','100k':R/'ssl_diversity_d12_100k_20261018'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
def saved_array(path):
    # Historical writer passed JSON shape lists to open_memmap. New numpy rejects
    # list-valued NPY headers. Parse literal header without rewriting any bytes.
    with path.open('rb') as f:
        assert f.read(6)==b'\x93NUMPY'
        version=tuple(f.read(2));assert version in [(1,0),(2,0),(3,0)]
        n=struct.unpack('<H' if version==(1,0) else '<I',f.read(2 if version==(1,0) else 4))[0]
        meta=ast.literal_eval(f.read(n).decode('utf-8' if version==(3,0) else 'latin1'));offset=f.tell()
    dtype=np.dtype(meta['descr']);shape=tuple(meta['shape'])
    assert dtype==np.uint8 and not meta['fortran_order']
    assert path.stat().st_size==offset+int(np.prod(shape))*dtype.itemsize
    return np.memmap(path,mode='r',dtype=dtype,shape=shape,offset=offset)
def csvout(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def overlay(raw,p,g):
    rgb=np.repeat((raw.astype(np.float32)/255)[...,None],3,axis=2)
    for mask,color in [(p&g,[0,1,.1]),(p&~g,[1,0,.65]),(~p&g,[0,.8,1])]:
        rgb[mask]=.4*rgb[mask]+.6*np.array(color)
    return rgb
def main():
    t=time.monotonic();OUT.mkdir(exist_ok=True)
    manifest=R/'mb1_official_benchmark_20261015/MB1_benchmark_manifest.json'
    entry=next(v for v in json.loads(manifest.read_text())['entries'] if v['id']=='glycolytic_muscle')
    ip=Path(entry['image_path']);gp=Path(entry['label_path'])
    assert sha(ip)==entry['image_file_sha256'] and sha(gp)==entry['gt_file_sha256']
    raw=tifffile.imread(ip);gt=tifffile.imread(gp)>0
    assert raw.dtype==np.uint8 and raw.shape==gt.shape==tuple(entry['shape_zyx'])
    paths={f'{m}_{b}':r/'benchmark'/m/'glycolytic_muscle_prediction.npy' for b,r in ROOTS.items() for m in ['D1','D2']}
    before={k:{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for k,p in paths.items()}
    pred={k:saved_array(p) for k,p in paths.items()}
    metrics={}
    for b,r in ROOTS.items():
        for m in ['D1','D2']:
            p=pred[f'{m}_{b}'];assert p.shape==raw.shape and set(np.unique(p)).issubset({0,1})
            saved=json.loads((r/'benchmark'/m/'metrics.json').read_text())
            metrics[f'{m}_{b}']={v['z']:v for v in saved['images'] if v['dataset']=='glycolytic_muscle'}
    rows=[];change_rows=[];by_grid=np.zeros((4,6),np.int64);area_grid=np.zeros((4,6),np.int64)
    heat=np.zeros(raw.shape[1:],np.int64);distance_counts=np.zeros(4,np.int64)
    yedges=np.linspace(0,raw.shape[1],5,dtype=int);xedges=np.linspace(0,raw.shape[2],7,dtype=int)
    for z in range(raw.shape[0]):
        g=gt[z];ps={k:np.asarray(p[z],bool) for k,p in pred.items()}
        for k,p in ps.items():
            tp=int((p&g).sum());fp=int((p&~g).sum());fn=int((~p&g).sum());tn=int((~p&~g).sum())
            old=metrics[k][z];assert [tp,fp,fn,tn]==[old[x] for x in ['tp','fp','fn','tn']]
            rows.append({'model_budget':k,'z':z,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'dice':2*tp/(2*tp+fp+fn)})
        oldfp=ps['D2_30k']&~g;newfp=ps['D2_100k']&~g
        added=newfp&~oldfp;removed=oldfp&~newfp;retained=oldfp&newfp
        heat+=added
        dist=ndi.distance_transform_edt(~g,sampling=entry['spacing_yx_nm'])
        for j,(lo,hi) in enumerate([(0,36),(36,90),(90,180),(180,np.inf)]):
            distance_counts[j]+=int((added&(dist>lo)&(dist<=hi)).sum())
        for y in range(4):
            for x in range(6):
                roi=added[yedges[y]:yedges[y+1],xedges[x]:xedges[x+1]]
                by_grid[y,x]+=int(roi.sum());area_grid[y,x]+=roi.size
        change_rows.append({'z':z,'D2_added_fp':int(added.sum()),'D2_removed_fp':int(removed.sum()),
            'D2_retained_fp':int(retained.sum()),'added_fp_also_D1_100k_fp':int((added&ps['D1_100k']).sum()),
            'D2_fp_net_change':int(newfp.sum())-int(oldfp.sum())})
    uniform=np.linspace(0,raw.shape[0]-1,6,dtype=int).tolist()
    extremes=[]
    for row in sorted(change_rows,key=lambda v:(-v['D2_added_fp'],v['z'])):
        if all(abs(row['z']-z)>=30 for z in extremes):extremes.append(row['z'])
        if len(extremes)==3:break
    selected=sorted(set(uniform+extremes))
    order=['D1_30k','D1_100k','D2_30k','D2_100k']
    for z in selected:
        fig,axs=plt.subplots(3,2,figsize=(12,10),layout='constrained')
        axs[0,0].imshow(raw[z],cmap='gray',vmin=0,vmax=255);axs[0,0].set_title('Raw EM (native gray 0..255)')
        axs[0,1].imshow(gt[z],cmap='gray',vmin=0,vmax=1);axs[0,1].set_title('GT: mitochondria > 0')
        for ax,k in zip(axs.flat[2:],order):
            p=np.asarray(pred[k][z],bool);ax.imshow(overlay(raw[z],p,gt[z]));ax.set_title(k.replace('_',' ')+' | Dice %.3f'%metrics[k][z]['dice'])
        for ax in axs.flat:ax.set_axis_off()
        fig.suptitle(f'Glycolytic muscle | Z={z} (zero based) | '+('uniform; ' if z in uniform else '')+('high added FP; ' if z in extremes else '')+
                    '\nGreen: TP | Magenta: FP | Cyan: FN | same native grid, no new inference',fontsize=12)
        fig.savefig(OUT/f'comparison_z{z:03}.png',dpi=120);plt.close(fig)
    z=extremes[0];g=gt[z];added=np.asarray(pred['D2_100k'][z],bool)&~np.asarray(pred['D2_30k'][z],bool)&~g
    # Fixed max-added-FP 128x128 window, purely for inspection, never inference.
    integral=np.pad(added.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1);w=128
    scores=integral[w:,w:]-integral[:-w,w:]-integral[w:,:-w]+integral[:-w,:-w]
    y,x=map(int,np.unravel_index(scores.argmax(),scores.shape));roi=(slice(y,y+w),slice(x,x+w))
    fig,axs=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    axs[0,0].imshow(raw[z][roi],cmap='gray',vmin=0,vmax=255);axs[0,0].set_title('Raw EM')
    axs[0,1].imshow(g[roi],cmap='gray',vmin=0,vmax=1);axs[0,1].set_title('GT')
    a=np.repeat((raw[z][roi].astype(float)/255)[...,None],3,axis=2);mask=added[roi];a[mask]=.35*a[mask]+.65*np.array([1,.5,0])
    axs[0,2].imshow(a);axs[0,2].set_title('Orange: D2 newly added FP')
    for ax,k in zip(axs[1],['D1_100k','D2_30k','D2_100k']):
        ax.imshow(overlay(raw[z][roi],np.asarray(pred[k][z],bool)[roi],g[roi]));ax.set_title(k.replace('_',' '))
    for ax in axs.flat:ax.set_axis_off()
    fig.suptitle(f'Error-selected detail | Z={z}, Y=[{y},{y+w}), X=[{x},{x+w})\nSame 128x128 native pixels; display enlargement only')
    fig.savefig(OUT/'added_fp_detail.png',dpi=120);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    axs[0].plot([r['z'] for r in change_rows],[r['D2_added_fp'] for r in change_rows],label='New FP')
    axs[0].plot([r['z'] for r in change_rows],[r['D2_removed_fp'] for r in change_rows],label='Removed FP')
    axs[0].set_xlabel('Native Z index');axs[0].set_ylabel('Pixels');axs[0].legend()
    h=axs[1].imshow(heat,cmap='magma');fig.colorbar(h,ax=axs[1],label='Number of Z planes with new FP')
    axs[1].set_title('D2 new FP projected over Z');axs[1].set_xlabel('X');axs[1].set_ylabel('Y')
    fig.savefig(OUT/'added_fp_distribution.png',dpi=120);plt.close(fig)
    csvout('per_plane_recheck.csv',rows);csvout('fp_change_by_z.csv',change_rows)
    totals={k:{name:sum(r[name] for r in rows if r['model_budget']==k) for name in ['tp','fp','fn','tn']} for k in pred}
    result={'status':'complete','gpu_used':False,'training_or_inference':False,'probabilities_available':False,
        'inputs':before,'image':{'path':str(ip),'sha256':sha(ip),'dtype':str(raw.dtype),'min':int(raw.min()),'max':int(raw.max())},
        'gt':{'path':str(gp),'sha256':sha(gp),'rule':'>0'},'shape_zyx':list(raw.shape),'spacing_yx_nm':entry['spacing_yx_nm'],
        'all_1208_plane_confusions_match_saved':True,'totals':totals,
        'changes':{k:sum(r[k] for r in change_rows) for k in change_rows[0] if k!='z'},
        'new_fp_distance_to_gt_2d_nm':dict(zip(['(0,36]','(36,90]','(90,180]','>180'],map(int,distance_counts))),
        'new_fp_grid_4y_6x':by_grid.tolist(),'grid_pixel_denominator':area_grid.tolist(),
        'uniform_z':uniform,'high_added_fp_z_min30_separation':extremes,'detail_roi':{'z':z,'y':[y,y+w],'x':[x,x+w]},
        'rendering':'Raw fixed uint8 0..255; TP green, FP magenta, FN cyan. Binary saved outputs only; no probability reconstruction.',
        'selection':'6 uniform planes incl endpoints; 3 greatest added-FP planes greedily separated >=30 Z; 128px max added-FP detail on greatest plane.',
        'interpretation_limit':'GT-relative FP, not independent tissue annotation; 2D distances only; no Z spacing assumption.'}
    result['reader_compatibility']='Historical NPY shape headers use lists; literal header parsed to a tuple in memory, exact dtype/layout/byte count checked; original files untouched.'
    probe=R/'mv_generalist_stage1_20261013/src/mv_probe.py'
    source=probe.read_text()
    assert '((a/255-.5)/.25)' in source and 'return acc/weight>.5' in source
    result['shared_inference_source']={'path':str(probe),'sha256':sha(probe),'normalization':'((uint8/255)-0.5)/0.25',
        'tile':256,'stride':128,'fusion':'arithmetic probability mean','threshold':0.5,
        'axis':'native ZYX; saved mask at native XY; same TIFF/GT for all four runs'}
    for k,p in paths.items():assert sha(p)==before[k]['sha256'] and p.stat().st_mtime_ns==before[k]['mtime_ns']
    result['prediction_files_unchanged']=True;result['seconds']=time.monotonic()-t
    dump('review_evidence.json',result)
    print(json.dumps({k:result[k] for k in ['status','changes','new_fp_distance_to_gt_2d_nm','uniform_z','high_added_fp_z_min30_separation','detail_roi','seconds']}),flush=True)
if __name__=='__main__':main()
