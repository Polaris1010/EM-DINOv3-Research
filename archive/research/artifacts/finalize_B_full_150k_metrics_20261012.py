"""Finalize completed probes, auditing initialization from original 100k code."""
import os,sys,json,time,hashlib,importlib.util,gc
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
STAGE2A2=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002')
DATA=Path('/opt/data/3_117/tanghr/Med_DINOv3')
DATASETS=['Dataset111_CMCC','Dataset131_CMCT']

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def atomic_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def main():
    comparison=RUN/'evaluation/comparison_100k_150k.json'
    for _ in range(40):
        state=json.loads((RUN/'STATUS.json').read_text())
        if comparison.exists():
            print('Existing comparison complete',flush=True);return
        if state.get('evaluation')=='failed':break
        print('Waiting for original evaluation:',state.get('evaluation'),flush=True)
        time.sleep(30)
    else:raise RuntimeError('Original evaluation did not finish within 20 minutes')
    for ds in DATASETS:
        m=json.loads((RUN/'evaluation'/('B_full150k_'+ds)/'metrics.json').read_text())
        assert m['status']=='complete' and m['encoder_step']==150000
    os.environ.update({'CUDA_VISIBLE_DEVICES':'GPU-5a68cc29-a13a-bf90-858e-079e14de807a','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
    sys.path[:0]=['/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps','/opt/data/3_117/tanghr/DINOv3/dinov3','/opt/data/3_117/tanghr/Med_DINOv3/MedDINOv3/nnUNet']
    guard=load_module('audit_guard',RUN/'scripts/gpu_guard.py')
    assert guard.website_ok() and guard.gpu_inventory()[3]['used_mib']<100
    original=STAGE2A2/'src/frozen_probe.py'
    source_sha=hashlib.sha256(original.read_bytes()).hexdigest()
    assert source_sha=='6c2bd66ff8deb2f3d2dde4260c232e45d7fbea7fe9c7bc3613687e2b6f94945c'
    probe=load_module('original_100k_probe',original)
    import torch,numpy as np,pandas as pd,SimpleITK as sitk
    net=probe.model(probe.ENCODER,20260907)
    h=hashlib.sha256()
    for name,tensor in sorted(net.state_dict().items()):
        if name.startswith('dino_encoder.'):continue
        h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(str(tuple(tensor.shape)).encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    initial_sha=h.hexdigest();del net,tensor;gc.collect();torch.cuda.empty_cache()
    assert initial_sha=='4b19c61be6451662481678ca87ffac1162812df24c31422c0d8d3f63acc358cf'
    assert guard.website_ok()
    atomic_json(RUN/'audit/100k_decoder_initialization_reconstruction.json',{'status':'pass','original_probe_sha256':source_sha,'seed':20260907,'decoder_initial_sha256':initial_sha,'method':'reconstruct original 100k model initialization using unchanged original source and original encoder; no decoder optimization'})
    print('Original 100k decoder initialization hash matches 150k:',initial_sha,flush=True)
    torch.set_num_threads(4);sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
    repo=load_module('repository_metrics',RUN/'src/repository_metrics.py')
    old=json.loads((STAGE2A2/'branch_E/boundary_metrics_20261012/results.json').read_text())
    summaries=[];case_rows=[]
    for ds in DATASETS:
        out=RUN/'evaluation'/('B_full150k_'+ds)
        metrics=json.loads((out/'metrics.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
        previous=json.loads((STAGE2A2/'branch_E/runs'/('B_full_'+ds)/'metrics.json').read_text())
        assert previous['source_sha256']==source_sha
        assert protocol['split_sha256']==previous['split_sha256'] and protocol['decoder_initial_sha256']==initial_sha and protocol['seed']==previous['seed']==20260907
        assert protocol['steps']==previous['steps']==3000
        assert metrics['frozen_encoder']
        split=json.loads((DATA/'nnUNet_preprocessed'/ds/'splits_final.json').read_text())[0]
        labels=DATA/'nnUNet_raw'/ds/'labelsTr';exports=out/'predictions_nifti';exports.mkdir(exist_ok=True)
        for case in split['val']:
            arr=np.load(out/ds/(case+'.npy'),mmap_mode='r');gt=sitk.ReadImage(str(labels/(case+'.nii.gz')))
            assert tuple(arr.shape)==tuple(reversed(gt.GetSize()))
            img=sitk.GetImageFromArray(np.asarray(arr));img.CopyInformation(gt);sitk.WriteImage(img,str(exports/(case+'.nii.gz')),True)
        assert {p.name for p in exports.glob('*.nii.gz')}=={c+'.nii.gz' for c in split['val']}
        df,summary=repo.compute_segmentation_metrics(str(exports),str(labels),['mitochondria'],device=torch.device('cpu'))
        s=summary['class_mitochondria']
        assert abs(float(s['dice']['mean'])-metrics['datasets'][ds]['macro_dice'])<1e-12
        row={'dataset':ds,'encoder_kind':'B_full150k','n':len(split['val']),**{m+'_'+v:float(s[m][v]) for m in s for v in ['mean','std']}}
        baseline=next(r for r in old['summary'] if r['dataset']==ds and r['encoder_kind']=='B_full')
        row.update({m+'_150k_minus_100k':row[m+'_mean']-baseline[m+'_mean'] for m in ['dice','hd95','nsd']})
        summaries.append(row);df.to_csv(out/'segmentation_metrics.csv',index=False)
        for r in df.to_dict('records'):case_rows.append({'dataset':ds,**{k:float(v) if k!='filename' else v for k,v in r.items()}})
        print('DATASET_METRICS_COMPLETE',json.dumps(row),flush=True)
    result={'status':'complete','summary':summaries,'cases':case_rows,'interpretation':'matched original 100k initialization independently reconstructed; same frozen decoder budget and split; single seed; volume-level sample SD; validation measurement; SSL-data overlap caveat remains','continuation_schedule':'constant original min_lr=1e-6; full state resumed from 100k','comparison_reference':str(STAGE2A2/'branch_E/boundary_metrics_20261012/results.json'),'initialization_audit':str(RUN/'audit/100k_decoder_initialization_reconstruction.json')}
    pd.DataFrame(summaries).to_csv(RUN/'evaluation/comparison_100k_150k.csv',index=False)
    atomic_json(comparison,result)
    state=json.loads((RUN/'STATUS.json').read_text())
    atomic_json(RUN/'STATUS.json',{**state,'evaluation':'complete','evaluation_original_exit_code':state.get('evaluation_exit_code'),'evaluation_exit_code':0,'metrics_finalization_repaired':True})
    print('COMPARISON_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main()
