"""Stream original-grid binary labels and historical predictions; no volume RAM cache."""
import os,json,gzip,shutil,time
from pathlib import Path
import numpy as np
import nibabel as nib
BASE=Path(__file__).resolve().parent
ROOT=Path('/opt/data/3_117/tanghr/Med_DINOv3')
DATASETS=['Dataset111_CMCC','Dataset131_CMCT']
TRAINERS=['meddinov3_base_primus_multiscale_Trainer','dinov3_base_primus_multiscale_Trainer','dinoUNetTrainer','segformerTrainer']
def unpack(src,key):
    dst=BASE/'cache'/(key+'.nii');dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():
        tmp=dst.with_suffix('.tmp')
        with gzip.open(src,'rb') as f,open(tmp,'wb') as g:shutil.copyfileobj(f,g,1024*1024)
        tmp.replace(dst)
    return nib.load(dst,mmap=True)
def metrics(c):
    tp,fp,fn,tn=map(int,c)
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,recall=tp/(tp+fn) if tp+fn else 1.,precision=tp/(tp+fp) if tp+fp else 0.)
def confusion(p,t):
    p=np.asarray(p,dtype=bool);t=np.asarray(t,dtype=bool)
    return np.array([(p&t).sum(),(p&~t).sum(),(~p&t).sum(),(~p&~t).sum()],dtype=np.int64)
def main():
    report={'protocol':'original fold_0 validation, binary raw labels, original voxel grid; historical saved predictions rescored, not newly inferred','datasets':{},'time':time.time()}
    for ds in DATASETS:
        split=json.loads((ROOT/'nnUNet_preprocessed'/ds/'splits_final.json').read_text())[0]
        assert not set(split['train'])&set(split['val'])
        d={'split':split,'cases':{},'models':{}};report['datasets'][ds]=d
        for case in split['val']:
            gt=unpack(ROOT/'nnUNet_raw'/ds/'labelsTr'/(case+'.nii.gz'),case+'_gt')
            raw=unpack(ROOT/'nnUNet_raw'/ds/'imagesTr'/(case+'_0000.nii.gz'),case+'_raw')
            assert gt.shape==raw.shape and np.allclose(gt.affine,raw.affine)
            values=set()
            for z in range(gt.shape[2]):values.update(np.unique(np.asarray(gt.dataobj[:,:,z])).tolist())
            assert values<={0,1},values
            d['cases'][case]={'shape_xyz':gt.shape,'affine':gt.affine.tolist(),'label_values':sorted(values)}
        for trainer in TRAINERS:
            fold=ROOT/'nnUNet_results'/ds/(trainer+'__nnUNetPlans__2d')/'fold_0'
            cases={};total=np.zeros(4,dtype=np.int64)
            old=json.loads((fold/'validation/summary.json').read_text())
            for case in split['val']:
                gt=unpack(ROOT/'nnUNet_raw'/ds/'labelsTr'/(case+'.nii.gz'),case+'_gt')
                pred=unpack(fold/'validation'/(case+'.nii.gz'),case+'_'+trainer)
                assert gt.shape==pred.shape and np.allclose(gt.affine,pred.affine)
                c=np.zeros(4,dtype=np.int64)
                for z in range(gt.shape[2]):c+=confusion(np.asarray(pred.dataobj[:,:,z])>0,np.asarray(gt.dataobj[:,:,z])>0)
                total+=c;cases[case]=metrics(c)
            d['models'][trainer]={'source':'historical saved validation predictions','cases':cases,'macro_dice':float(np.mean([v['dice'] for v in cases.values()])),'micro':metrics(total),'historical_foreground_mean':old['foreground_mean']}
            print(ds,trainer,d['models'][trainer]['macro_dice'],flush=True)
        tmp=BASE/'cached_metrics.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(BASE/'cached_metrics.json')
    (BASE/'cache_ready.json').write_text(json.dumps({'complete':True,'time':time.time()}))
if __name__=='__main__':main()
