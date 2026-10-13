"""Run the MedDINOv3 repository evaluator on existing frozen validation predictions."""
import ast, csv, gc, hashlib, importlib.util, inspect, json, os, shutil, time
from pathlib import Path
from urllib.request import urlopen, Request
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch, monai
from monai.metrics import compute_hausdorff_distance, compute_surface_dice

RUN=Path(__file__).resolve().parents[1]
OUT=RUN/'branch_E/boundary_metrics_20261012'
ROOT=Path('/opt/data/3_117/tanghr/Med_DINOv3')
DATASETS=['Dataset111_CMCC','Dataset131_CMCT']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def save(p,obj):
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    source=ROOT/'MedDINOv3/nnUNet/nnunetv2/compute_metrics.py'
    snap=OUT/'source_snapshot';snap.mkdir(exist_ok=True)
    shutil.copy2(source,snap/'compute_metrics_local.py')
    # Pin upstream source and verify the actual evaluation function, excluding configured main paths.
    metadata=json.loads((RUN/'src/meddinov3_metrics_source_20261012.json').read_text())
    commit=metadata['commit'];raw=metadata['url']
    official=(RUN/'src/meddinov3_compute_metrics_upstream_20261012.py').read_bytes()
    assert hashlib.sha256(official).hexdigest()==metadata['sha256']
    upstream=snap/'compute_metrics_upstream.py';upstream.write_bytes(official)
    name='compute_segmentation_metrics'
    def function_ast(p):
        return ast.dump(next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
    assert function_ast(source)==function_ast(upstream), 'Local evaluator differs from upstream'
    spec=importlib.util.spec_from_file_location('repository_metrics',snap/'compute_metrics_local.py')
    repo=importlib.util.module_from_spec(spec);spec.loader.exec_module(repo)
    a=np.zeros((15,17,19),bool);a[3:10,4:11,5:12]=True
    t=torch.from_numpy(a)[None,None]
    assert float(compute_hausdorff_distance(t,t,include_background=True,percentile=95,spacing=(1,1,1)))==0
    assert float(compute_surface_dice(t,t,class_thresholds=[1.0],include_background=True,spacing=(1,1,1)))==1
    shifted=np.roll(a,3,axis=2);p=torch.from_numpy(shifted)[None,None]
    hd=float(compute_hausdorff_distance(p,t,include_background=True,percentile=95,spacing=(1,1,1)))
    nsd=[float(compute_surface_dice(p,t,class_thresholds=[v],include_background=True,spacing=(1,1,1))) for v in [1.,2.,5.]]
    assert hd==3 and nsd==sorted(nsd) and nsd[-1]==1
    protocol={'status':'running','upstream_commit':commit,'upstream_url':raw,'upstream_source_sha256':sha(upstream),'local_source_sha256':sha(source),'evaluator_ast_matches_upstream':True,'wrapper_sha256':sha(__file__),'monai_version':monai.__version__,'torch_version':torch.__version__,'pandas_version':pd.__version__,'device':'cpu','dimensions':'3D full original-grid binary validation volumes','spacing':'read unchanged from each label SimpleITK image; all currently (1,1,1)','distance_unit':'stored header units; numerically voxel units at spacing 1; physical EM calibration unverified','nsd_threshold':1.0,'nsd_use_subvoxels':False,'hd95':'MONAI percentile=95, directed=False, euclidean; max of the two directional 95th percentiles','std_ddof':1,'aggregation':'equal-weight per validation volume; not per slice and not across seeds','seed':20260907,'new_training':False,'phantom_checks':{'identity_hd95':0,'identity_nsd':1,'shift3_hd95':hd,'shift3_nsd_1_2_5':nsd},'dependencies':{}}
    for fun in [compute_hausdorff_distance,compute_surface_dice]:
        path=Path(inspect.getsourcefile(fun));protocol['dependencies'][fun.__name__]={'path':str(path),'sha256':sha(path)};shutil.copy2(path,snap/path.name)
    save(OUT/'protocol.json',protocol)
    all_rows=[];summary_rows=[];audits=[]
    for ds in DATASETS:
        split_path=ROOT/'nnUNet_preprocessed'/ds/'splits_final.json'
        split=json.loads(split_path.read_text())[0];cases=split['val'];assert not set(cases)&set(split['train'])
        for kind in ['official_natural','B_full']:
            start=time.time();run=RUN/('branch_E/control_20261011/runs' if kind=='official_natural' else 'branch_E/runs')/(kind+'_'+ds)
            old=json.loads((run/'metrics.json').read_text())['datasets'][ds]
            work=OUT/(kind+'_'+ds);pred_dir=work/'predictions';pred_dir.mkdir(parents=True,exist_ok=True)
            labels=ROOT/'nnUNet_raw'/ds/'labelsTr';expected={}
            for case in cases:
                original=run/ds/(case+'.npy');before=sha(original);arr=np.load(original,mmap_mode='r')
                label_path=labels/(case+'.nii.gz');label=sitk.ReadImage(str(label_path));gt=sitk.GetArrayFromImage(label)
                assert arr.shape==gt.shape and set(np.unique(arr))<=set([0,1]) and set(np.unique(gt))<=set([0,1])
                assert gt.any() and arr.any()
                pred=sitk.GetImageFromArray(np.asarray(arr));pred.CopyInformation(label)
                destination=pred_dir/(case+'.nii.gz');sitk.WriteImage(pred,str(destination),True)
                reloaded=sitk.ReadImage(str(destination));assert np.array_equal(sitk.GetArrayFromImage(reloaded),arr)
                assert reloaded.GetSpacing()==label.GetSpacing() and reloaded.GetOrigin()==label.GetOrigin() and reloaded.GetDirection()==label.GetDirection()
                dice=float(repo.compute_dice(arr>0,gt>0));assert abs(dice-old['cases'][case]['dice'])<1e-12
                assert sha(original)==before
                expected[case+'.nii.gz']=dice
                audits.append({'dataset':ds,'encoder_kind':kind,'case':case,'shape_zyx':list(arr.shape),'spacing_xyz':list(label.GetSpacing()),'origin_xyz':list(label.GetOrigin()),'direction':list(label.GetDirection()),'prediction_source':str(original),'prediction_sha256':before,'label_path':str(label_path),'label_sha256':sha(label_path),'split_sha256':sha(split_path),'prediction_export_sha256':sha(destination),'dice_matches_previous':True,'GT_voxels':int((gt>0).sum()),'prediction_voxels':int((arr>0).sum())})
                del arr,gt,pred,label,reloaded;gc.collect()
                print('EXPORTED',kind,ds,case,flush=True)
            # Call the repository function unchanged; device override is the only execution change.
            df,stats=repo.compute_segmentation_metrics(str(pred_dir),str(labels),['mitochondria'],device=torch.device('cpu'))
            assert set(df.filename)==set(expected)
            for row in df.to_dict('records'):
                assert abs(float(row['dice_class1'])-expected[row['filename']])<1e-12
                assert all(np.isfinite(row[k]) for k in ['dice_class1','hd95_class1','nsd_class1'])
                all_rows.append({'dataset':ds,'encoder_kind':kind,'case':row['filename'].removesuffix('.nii.gz'),'dice':float(row['dice_class1']),'hd95':float(row['hd95_class1']),'nsd':float(row['nsd_class1'])})
            df.to_csv(work/'segmentation_metrics.csv',index=False)
            s=stats['class_mitochondria'];s={k:{kk:float(vv) for kk,vv in v.items()} for k,v in s.items()}
            for metric in ['dice','hd95','nsd']:
                assert abs(s[metric]['mean']-float(df[metric+'_class1'].mean()))<1e-12
                assert abs(s[metric]['std']-float(df[metric+'_class1'].std(ddof=1)))<1e-12
            assert abs(s['dice']['mean']-old['macro_dice'])<1e-12
            save(work/'summary.json',{'dataset':ds,'encoder_kind':kind,'n':len(cases),'stats':s,'seconds':time.time()-start})
            summary_rows.append({'dataset':ds,'encoder_kind':kind,'n':len(cases),**{m+'_'+v:s[m][v] for m in s for v in ['mean','std']}})
            save(OUT/'progress.json',{'completed_groups':len(summary_rows),'completed':summary_rows[-1]})
            print('COMPLETE_GROUP',json.dumps(summary_rows[-1]),flush=True);gc.collect()
    save(OUT/'input_audit.json',audits)
    pd.DataFrame(all_rows).to_csv(OUT/'per_volume_metrics.csv',index=False)
    pd.DataFrame(summary_rows).to_csv(OUT/'summary.csv',index=False)
    protocol['status']='complete';save(OUT/'protocol.json',protocol)
    save(OUT/'results.json',{'status':'complete','summary':summary_rows,'per_volume':all_rows,'protocol':protocol})
    lines=['# 冻结评测：Dice、HD95、NSD（2026-10-12）','','使用 MedDINOv3 官方 `compute_segmentation_metrics` 测试函数，已验证与上游函数 AST 相同。沿用已保存的 B_full / 官方自然图像初始化预测，不重新训练。','','| 数据集 | 冻结编码器 | 验证体积 n | Dice ↑ | HD95 ↓（voxel 等效单位） | NSD ↑（τ=1.0） |','|---|---|---:|---:|---:|---:|']
    for r in summary_rows:
        label='B_full（EM 100k）' if r['encoder_kind']=='B_full' else '官方 DINOv3 自然图像权重'
        f=lambda m: f"{r[m+'_mean']:.4f} ± {r[m+'_std']:.4f}" if m!='hd95' else f"{r[m+'_mean']:.2f} ± {r[m+'_std']:.2f}"
        lines.append(f"| {r['dataset'].split('_')[-1]} | {label} | {r['n']} | {f('dice')} | {f('hd95')} | {f('nsd')} |")
    lines.extend(['','统计说明：每个完整验证体积先计算指标，再等权平均；± 为体积间样本标准差（pandas std，ddof=1），不是多随机种子标准差，也不是标准误。CMCC n=3、CMCT n=2。','', '距离说明：官方入口读取标签文件 spacing，当前所有文件均为 1×1×1，NSD 的固定容差 1.0 数值上等于 1 voxel。NIfTI 头标有 mm，但 EM 真实物理尺度尚未校准，因此本表按 voxel 等效单位说明，不解释为真实毫米。没有重采样。','', '实现说明：入口实际调用 MONAI HD95 和 surface Dice；没有使用同文件中未被入口调用的 SimpleITK compute_hd95 / compute_nsd 辅助函数。MONAI NSD 默认 use_subvoxels=False。','', '评测范围：二值线粒体语义分割、原始网格完整 3D 体积。冻结 encoder、同一 decoder 协议、单 seed、原验证划分与原 decoder checkpoint 不变。官方自然图像权重不是 MedDINOv3 CT-3M 权重；不把这次两模型冻结评测与参考表中的完整训练模型混为一组。','', '该结果属于验证集测量，SSL/下游数据重叠审计仍待完成，不能声称独立测试或未见来源泛化。P 继续暂缓，不启动长训。','',f"源码版本：[{commit[:12]}](https://github.com/ricklisz/MedDINOv3/blob/{commit}/nnUNet/nnunetv2/compute_metrics.py)；MONAI {monai.__version__}。完整精度与逐体积结果见 results.json、summary.csv、per_volume_metrics.csv。"])
    (OUT/'REPORT_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('ALL_COMPLETE',flush=True)

if __name__=='__main__':
    try: main()
    except Exception:
        import traceback
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'error.txt').write_text(traceback.format_exc());raise
