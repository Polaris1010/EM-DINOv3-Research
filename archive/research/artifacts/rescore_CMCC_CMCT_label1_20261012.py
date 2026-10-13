"""Recompute all three frozen encoders with explicit mitochondrial label == 1."""
from pathlib import Path
import os,json,hashlib,importlib.util,concurrent.futures,ast,gc,time

STAGE2=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002')
EXT=Path('/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012')
DATA=Path('/opt/data/3_117/tanghr/Med_DINOv3')
OUT=EXT/'evaluation/label1_only_recompute_20261012'
DATASETS=['Dataset111_CMCC','Dataset131_CMCT']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def save(p,obj):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False));temp.replace(p)

def source_run(ds,kind):
    if kind=='B_full150k':return EXT/'evaluation'/('B_full150k_'+ds)
    if kind=='B_full100k':return STAGE2/'branch_E/runs'/('B_full_'+ds)
    return STAGE2/'branch_E/control_20261011/runs'/('official_natural_'+ds)

def compute_group(task):
    ds,kind=task
    import numpy as np,torch,SimpleITK as sitk
    torch.set_num_threads(4);sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    spec=importlib.util.spec_from_file_location('repo',EXT/'src/repository_metrics.py')
    repo=importlib.util.module_from_spec(spec);spec.loader.exec_module(repo)
    split_path=DATA/'nnUNet_preprocessed'/ds/'splits_final.json'
    split=json.loads(split_path.read_text())[0];cases=split['val'];run=source_run(ds,kind)
    work=OUT/(kind+'_'+ds);preds=work/'predictions_label1';preds.mkdir(parents=True)
    labels=OUT/'labels_label1'/ds
    audits=[];expected={}
    for case in cases:
        p=run/ds/(case+'.npy');a=np.load(p,mmap_mode='r')
        assert set(np.unique(a))<=set([0,1]),'Existing frozen prediction is not binary'
        gt_img=sitk.ReadImage(str(labels/(case+'.nii.gz')));gt=sitk.GetArrayFromImage(gt_img)
        assert a.shape==gt.shape
        binary=(a==1).astype(np.uint8)
        image=sitk.GetImageFromArray(binary);image.CopyInformation(gt_img)
        destination=preds/(case+'.nii.gz');sitk.WriteImage(image,str(destination),True)
        tp=int(np.count_nonzero((binary==1)&(gt==1)))
        pred_n=int(np.count_nonzero(binary==1));gt_n=int(np.count_nonzero(gt==1))
        expected[case+'.nii.gz']=2*tp/(pred_n+gt_n)
        audits.append({'dataset':ds,'encoder_kind':kind,'case':case,'prediction_source':str(p),'prediction_sha256':sha(p),'prediction_rule':'prediction == 1','gt_rule':'raw label == 1','gt_mito_voxels':gt_n,'pred_mito_voxels':pred_n,'tp':tp,'fp':pred_n-tp,'fn':gt_n-tp})
        del a,binary,gt,image,gt_img;gc.collect()
    df,stats=repo.compute_segmentation_metrics(str(preds),str(labels),['mitochondria'],device=torch.device('cpu'))
    assert set(df.filename)==set(expected)
    rows=[]
    for r in df.to_dict('records'):
        assert abs(float(r['dice_class1'])-expected[r['filename']])<1e-12
        rows.append({'dataset':ds,'encoder_kind':kind,'case':r['filename'].removesuffix('.nii.gz'),**{m:float(r[m+'_class1']) for m in ['dice','hd95','nsd']}})
    s=stats['class_mitochondria']
    row={'dataset':ds,'encoder_kind':kind,'n':len(cases),**{m+'_'+v:float(s[m][v]) for m in ['dice','hd95','nsd'] for v in ['mean','std']}}
    df.to_csv(work/'segmentation_metrics.csv',index=False)
    save(work/'result.json',{'status':'complete','summary':row,'per_volume':rows,'audit':audits})
    return row,rows,audits

def main():
    import numpy as np,pandas as pd,SimpleITK as sitk
    assert not OUT.exists(),'Use a separate directory; do not replace previous results'
    OUT.mkdir(parents=True)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    labels_audit=[];preprocessed_audit=[];metadata={}
    for ds in DATASETS:
        raw=DATA/'nnUNet_raw'/ds;pp=DATA/'nnUNet_preprocessed'/ds
        metadata[ds]=json.loads((raw/'dataset.json').read_text())
        assert metadata[ds]['labels']['mitochondria']==1
        split=json.loads((pp/'splits_final.json').read_text())[0]
        out=OUT/'labels_label1'/ds;out.mkdir(parents=True)
        for case in split['val']:
            path=raw/'labelsTr'/(case+'.nii.gz');img=sitk.ReadImage(str(path));a=sitk.GetArrayFromImage(img);values,counts=np.unique(a,return_counts=True)
            b=(a==1).astype(np.uint8);export=sitk.GetImageFromArray(b);export.CopyInformation(img);dest=out/(case+'.nii.gz');sitk.WriteImage(export,str(dest),True)
            labels_audit.append({'dataset':ds,'case':case,'raw_label_path':str(path),'raw_label_sha256':sha(path),'label_values_counts':dict(zip(map(str,values),map(int,counts))),'mito_voxels':int(b.sum()),'other_positive_class_voxels':int(np.count_nonzero((a>0)&(a!=1))),'nonzero_equals_label1':bool(np.array_equal(a>0,a==1)),'shape_zyx':list(a.shape),'spacing_xyz':list(img.GetSpacing()),'label1_export_sha256':sha(dest)})
            print('LABEL_AUDIT',json.dumps(labels_audit[-1]),flush=True)
        for case in split['train']+split['val']:
            path=pp/'nnUNetPlans_2d'/(case+'_seg.npy');a=np.load(path,mmap_mode='r');v,c=np.unique(a,return_counts=True)
            assert set(v)<=set([-1,0,1]),'Frozen training contains other classes; requires corrected decoder training'
            preprocessed_audit.append({'dataset':ds,'case':case,'split':'train' if case in split['train'] else 'val','path':str(path),'counts':dict(zip(map(str,v),map(int,c)))})
    save(OUT/'label_audit.json',{'status':'pass','dataset_metadata':metadata,'validation_labels':labels_audit,'preprocessed_labels':preprocessed_audit,'preprocessed_count':len(preprocessed_audit)})
    src=EXT/'src/repository_metrics.py';tree=ast.parse(src.read_text());fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_segmentation_metrics')
    assert any(isinstance(n,ast.Compare) and isinstance(n.left,ast.Name) and n.left.id=='label_array' and isinstance(n.ops[0],ast.Eq) for n in ast.walk(fun))
    protocol={'target':'mitochondria only','gt_rule':'GT == 1; every other semantic class is background','prediction_rule':'prediction == 1','metric_evaluator_sha256':sha(src),'metric_entry':'unchanged MedDINOv3 compute_segmentation_metrics with single class mitochondria','nsd_threshold':1.0,'hd95':'MONAI percentile95, directed=False','spacing':'unchanged original-grid spacing=(1,1,1); voxel-equivalent units','aggregation':'volume-equally-weighted mean and sample SD ddof=1','seed':20260907,'decoder_steps':3000,'new_training':False,'annotation_audit_count':len(labels_audit),'preprocessed_audit_count':len(preprocessed_audit),'prior_gt_gt0_equals_gt1_on_all_validation_volumes':all(a['nonzero_equals_label1'] for a in labels_audit)}
    save(OUT/'protocol.json',protocol)
    tasks=[(ds,k) for ds in DATASETS for k in ['official_natural','B_full100k','B_full150k']]
    summaries=[];rows=[];audits=[]
    # Independent groups use only CPU and separate export directories.
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(compute_group,t):t for t in tasks}
        for f in concurrent.futures.as_completed(futures):
            summary,group,inputs=f.result();summaries.append(summary);rows.extend(group);audits.extend(inputs)
            save(OUT/'progress.json',{'completed_groups':len(summaries),'last':summary})
            print('COMPLETE_GROUP',json.dumps(summary),flush=True)
    order={t:i for i,t in enumerate(tasks)};summaries.sort(key=lambda r:order[(r['dataset'],r['encoder_kind'])]);rows.sort(key=lambda r:(order[(r['dataset'],r['encoder_kind'])],r['case']))
    old=json.loads((STAGE2/'branch_E/boundary_metrics_20261012/results.json').read_text())
    ext=json.loads((EXT/'evaluation/comparison_100k_150k.json').read_text())
    for row in summaries:
        if row['encoder_kind']=='B_full150k':ref=next(r for r in ext['summary'] if r['dataset']==row['dataset'])
        else:ref=next(r for r in old['summary'] if r['dataset']==row['dataset'] and r['encoder_kind']==('B_full' if row['encoder_kind']=='B_full100k' else 'official_natural'))
        for m in ['dice','hd95','nsd']:
            for v in ['mean','std']:assert abs(row[m+'_'+v]-ref[m+'_'+v])<1e-12
    result={'status':'complete','summary':summaries,'per_volume':rows,'protocol':protocol,'matches_previous_class1_results':True}
    pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'per_volume_metrics.csv',index=False)
    save(OUT/'prediction_audit.json',audits);save(OUT/'results.json',result)
    print('ALL_LABEL1_RECOMPUTATION_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main()
