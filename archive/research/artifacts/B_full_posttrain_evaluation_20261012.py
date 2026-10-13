"""After 150k SSL, run the matched frozen probes and repository surface metrics."""
import os,sys,json,subprocess,importlib.util,time,hashlib
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
DATA=Path('/opt/data/3_117/tanghr/Med_DINOv3')
STAGE2A2=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002')
PY='/opt/data/3_117/tanghr/envs/meddinov3/bin/python'
def main():
    checkpoint=RUN/'runs/B_full/formal_attempt1/checkpoints/step000150000/teacher_backbone.pth'
    assert checkpoint.is_file()
    env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':'GPU-5a68cc29-a13a-bf90-858e-079e14de807a','B_FULL_CHECKPOINT':str(checkpoint),'B_FULL_STEP':'150000','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONPATH':'/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps:/opt/data/3_117/tanghr/DINOv3/dinov3:/opt/data/3_117/tanghr/Med_DINOv3/MedDINOv3/nnUNet'})
    import importlib.util
    guard_spec=importlib.util.spec_from_file_location('continuation_guard',RUN/'scripts/gpu_guard.py');guard=importlib.util.module_from_spec(guard_spec);guard_spec.loader.exec_module(guard)
    def guarded_probe(command,log):
        initial=guard.gpu_inventory();apps=guard.gpu_apps(initial)
        baseline={(a['gpu'],a['pid']):a['birth'] for a in apps if a['gpu'] in (1,2,3)}
        assert initial[3]['used_mib']<100,'GPU3 occupied before frozen evaluation'
        with log.open('w') as f:
            p=subprocess.Popen(command,env=env,cwd=RUN,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            while p.poll() is None:
                current=guard.gpu_apps(guard.gpu_inventory())
                for a in current:
                    own=guard.is_own_process(a['pid'],p.pid)
                    if own and a['gpu']==0:raise RuntimeError('Own probe on forbidden GPU0')
                    if not own and a['gpu'] in (1,2,3) and baseline.get((a['gpu'],a['pid']))!=a['birth']:
                        os.killpg(p.pid,__import__('signal').SIGTERM);p.wait();raise RuntimeError('GPU conflict during frozen evaluation')
                if not guard.website_ok():
                    os.killpg(p.pid,__import__('signal').SIGTERM);p.wait();raise RuntimeError('Website unhealthy during frozen evaluation')
                time.sleep(15)
            assert p.returncode==0,f'Frozen evaluation failed: {log}'
    for ds in ['Dataset111_CMCC','Dataset131_CMCT']:
        out=RUN/'evaluation'/('B_full150k_'+ds);assert not out.exists();env['RUN_OUTPUT']=str(out)
        guarded_probe([PY,str(RUN/'src/frozen_probe_150k.py'),'--dataset',ds,'--encoder-kind','B_full','--steps','3000','--seed','20260907'],RUN/'evaluation'/(ds+'.log'))
    import torch,numpy as np,pandas as pd,SimpleITK as sitk
    torch.set_num_threads(4);sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    spec=importlib.util.spec_from_file_location('repository_metrics',RUN/'src/repository_metrics.py');repo=importlib.util.module_from_spec(spec);spec.loader.exec_module(repo)
    old=json.loads((STAGE2A2/'branch_E/boundary_metrics_20261012/results.json').read_text())
    summaries=[];case_rows=[]
    for ds in ['Dataset111_CMCC','Dataset131_CMCT']:
        out=RUN/'evaluation'/('B_full150k_'+ds);metrics=json.loads((out/'metrics.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
        prev=STAGE2A2/'branch_E/runs'/('B_full_'+ds);previous=json.loads((prev/'metrics.json').read_text())
        assert protocol['split_sha256']==previous['split_sha256'] and protocol['decoder_initial_sha256']==previous['decoder_initial_sha256'] and protocol['seed']==previous['seed']==20260907
        split=json.loads((DATA/'nnUNet_preprocessed'/ds/'splits_final.json').read_text())[0];labels=DATA/'nnUNet_raw'/ds/'labelsTr';exports=out/'predictions_nifti';exports.mkdir()
        for case in split['val']:
            arr=np.load(out/ds/(case+'.npy'),mmap_mode='r');gt=sitk.ReadImage(str(labels/(case+'.nii.gz')));assert tuple(arr.shape)==tuple(reversed(gt.GetSize()))
            img=sitk.GetImageFromArray(np.asarray(arr));img.CopyInformation(gt);sitk.WriteImage(img,str(exports/(case+'.nii.gz')),True)
        df,summary=repo.compute_segmentation_metrics(str(exports),str(labels),['mitochondria'],device=torch.device('cpu'));s=summary['class_mitochondria']
        assert abs(float(s['dice']['mean'])-metrics['datasets'][ds]['macro_dice'])<1e-12
        row={'dataset':ds,'encoder_kind':'B_full150k','n':len(split['val']),**{m+'_'+v:float(s[m][v]) for m in s for v in ['mean','std']}}
        baseline=next(r for r in old['summary'] if r['dataset']==ds and r['encoder_kind']=='B_full')
        row.update({m+'_150k_minus_100k':row[m+'_mean']-baseline[m+'_mean'] for m in ['dice','hd95','nsd']});summaries.append(row)
        df.to_csv(out/'segmentation_metrics.csv',index=False)
        for r in df.to_dict('records'):case_rows.append({'dataset':ds,**{k:float(v) if k!='filename' else v for k,v in r.items()}})
    result={'status':'complete','summary':summaries,'cases':case_rows,'interpretation':'same frozen decoder budget and split; single seed; volume-level sample SD; validation measurement; SSL-data overlap caveat remains','continuation_schedule':'constant original min_lr=1e-6; full state resumed from 100k','comparison_reference':str(STAGE2A2/'branch_E/boundary_metrics_20261012/results.json')}
    (RUN/'evaluation/comparison_100k_150k.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame(summaries).to_csv(RUN/'evaluation/comparison_100k_150k.csv',index=False)
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
