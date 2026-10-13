"""Three readout workers on physical GPU1/2/3; only own workers can be stopped."""
from pathlib import Path
import os,sys,json,time,hashlib,subprocess,urllib.request,signal,csv,math,statistics,tarfile
import psutil
R=Path(__file__).resolve().parents[1]
PYTHON='/opt/data/3_117/tanghr/envs/meddinov3/bin/python'
EXT=Path('/opt/data/3_117/tanghr/codex/mitoverse_B_full_extend150k_20261012')
S2=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002')
DATA=Path('/opt/data/3_117/tanghr/Med_DINOv3')
DATASETS=['Dataset111_CMCC','Dataset131_CMCT']
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def safe_json(path):
 try:return json.loads(path.read_text())
 except (OSError,json.JSONDecodeError):return None
def gpu_apps():
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
 return [{'uuid':v[0].strip(),'pid':int(v[1]),'mib':int(v[2])} for v in csv.reader(raw.splitlines()) if len(v)==3]
def inventory():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
 return {int(v[0]):{'uuid':v[1].strip(),'free_mib':int(v[2])} for v in csv.reader(raw.splitlines())}
def website_ok():
 try:
  with urllib.request.urlopen('http://127.0.0.1:7860/',timeout=5) as response:return response.status==200
 except Exception:return False
def own(pid,leader):
 try:return pid==leader or any(p.pid==leader for p in psutil.Process(pid).parents())
 except psutil.Error:return False

def worker(model_id):
 registry=json.loads((R/'model_registry.json').read_text())['models'];m=next(x for x in registry if x['id']==model_id)
 step={'P0':0,'B100k':100000,'B150k':150000}[model_id]
 for ds in DATASETS:
  output=R/'runs'/(model_id+'_'+ds);assert not output.exists()
  env={**os.environ,'RUN_OUTPUT':str(output),'B_FULL_CHECKPOINT':m['checkpoint_path'],'B_FULL_STEP':str(step)}
  with (R/'runs'/(model_id+'_'+ds+'.log')).open('w') as log:
   result=subprocess.run([PYTHON,'-u',str(R/'src/frozen_probe_official.py'),'--dataset',ds,'--encoder-kind','official_natural' if model_id=='P0' else 'B_full','--steps','3000','--seed','20260907'],env=env,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(f'{model_id}/{ds} failed with {result.returncode}')
  assert json.loads((output/'metrics.json').read_text())['status']=='complete'
  print('HEAD COMPLETE',model_id,ds,flush=True)

def score_and_report():
 import importlib.util
 import numpy as np,pandas as pd,SimpleITK as sitk,torch
 torch.set_num_threads(4);sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4)
 spec=importlib.util.spec_from_file_location('official_repository_metrics',R/'src/repository_metrics.py');repo=importlib.util.module_from_spec(spec);spec.loader.exec_module(repo)
 assert sha(R/'src/repository_metrics.py')=='8abc23bfbbee06b1a6289b1f70ec7a5fda2a02cb15d8c870b28890de6152477d'
 registry=json.loads((R/'model_registry.json').read_text())['models']
 summaries=[];cases=[];initial_hashes=set();splits={}
 for ds in DATASETS:
  splitp=DATA/'nnUNet_preprocessed'/ds/'splits_final.json';split=json.loads(splitp.read_text())[0]
  for m in registry:
   output=R/'runs'/(m['id']+'_'+ds);metrics=json.loads((output/'metrics.json').read_text())
   assert metrics['frequency_gates'] is False and metrics['frozen_encoder'] and metrics['steps']==3000 and metrics['seed']==20260907
   assert metrics['split_sha256']==sha(splitp) and metrics['encoder_sha256']==m['sha256']
   initial_hashes.add(metrics['decoder_initial_sha256']);splits[ds]=metrics['split_sha256']
   exports=output/'predictions_nifti';exports.mkdir();labels=DATA/'nnUNet_raw'/ds/'labelsTr'
   for case in split['val']:
    pred=np.load(output/ds/(case+'.npy'),mmap_mode='r');gt=sitk.ReadImage(str(labels/(case+'.nii.gz')))
    assert pred.shape==tuple(reversed(gt.GetSize())) and tuple(gt.GetSpacing())==(1.,1.,1.)
    assert set(np.unique(pred)).issubset({0,1})
    img=sitk.GetImageFromArray(np.asarray(pred==1,dtype=np.uint8));img.CopyInformation(gt);sitk.WriteImage(img,str(exports/(case+'.nii.gz')),True)
   df,summary=repo.compute_segmentation_metrics(str(exports),str(labels),['mitochondria'],device=torch.device('cpu'))
   s=summary['class_mitochondria'];assert len(df)==len(split['val'])
   assert abs(float(s['dice']['mean'])-metrics['datasets'][ds]['macro_dice'])<1e-12
   row={'dataset':ds,'model':m['id'],'decoder':'official_primus_multiscale','n':len(df),'selected_step':metrics['datasets'][ds]['selected_step'],**{metric+'_'+v:float(s[metric][v]) for metric in ['dice','hd95','nsd'] for v in ['mean','std']}}
   summaries.append(row);df.to_csv(output/'segmentation_metrics.csv',index=False)
   for item in df.to_dict('records'):cases.append({'dataset':ds,'model':m['id'],'case':item['filename'].removesuffix('.nii.gz'),**{metric:float(item[metric+'_class1']) for metric in ['dice','hd95','nsd']}})
   atomic(R/'SCORING_PROGRESS.json',{'completed_groups':len(summaries),'latest':row});print('SCORED',json.dumps(row),flush=True)
 assert len(initial_hashes)==1
 result={'status':'complete','decoder':'official MedDINOv3 Primus_Multiscale without gates','upstream_commit':'a2460c2547899283446e520b84523454f7e25608','decoder_sha256':sha(R/'src/official_primus.py'),
 'summary':summaries,'per_volume':cases,'seed':20260907,'steps':3000,'decoder_initial_sha256':next(iter(initial_hashes)),'target':'GT==1 / pred==1 mitochondria only',
 'hd95_unit':'voxel-equivalent; original grid spacing=(1,1,1)','nsd_threshold':1.,'std_definition':'sample SD across validation volumes, ddof=1; not seed SD',
 'interpretation':'development validation; transductive B_full; not independent MitoNet comparison'}
 atomic(R/'results.json',result)
 pd.DataFrame(summaries).to_csv(R/'summary.csv',index=False);pd.DataFrame(cases).to_csv(R/'per_volume_metrics.csv',index=False)
 old=json.loads((EXT/'evaluation/label1_only_recompute_20261012/results.json').read_text());mapping={'P0':'official_natural','B100k':'B_full100k','B150k':'B_full150k'}
 differences=[]
 for row in summaries:
  baseline=next(x for x in old['summary'] if x['dataset']==row['dataset'] and x['encoder_kind']==mapping[row['model']])
  differences.append({'dataset':row['dataset'],'model':row['model'],**{metric+'_official_minus_gated':row[metric+'_mean']-baseline[metric+'_mean'] for metric in ['dice','hd95','nsd']}})
 pd.DataFrame(differences).to_csv(R/'official_vs_gated_deltas.csv',index=False)
 lines=['# 原版 MedDINOv3 Primus 冻结评测','','六组重新初始化 head 并训练完成。固定官方源码，无频域门控。','','| Dataset | Encoder | Dice ↑ | HD95 ↓ | NSD@1 ↑ |','|---|---|---:|---:|---:|']
 for row in summaries:lines.append(f"| {row['dataset'].split('_')[-1]} | {row['model']} | {row['dice_mean']:.6f} ± {row['dice_std']:.6f} | {row['hd95_mean']:.2f} ± {row['hd95_std']:.2f} | {row['nsd_mean']:.6f} ± {row['nsd_std']:.6f} |")
 lines+=['','## 原版 − 历史门控版均值差','','| Dataset | Encoder | ΔDice | ΔHD95 | ΔNSD |','|---|---|---:|---:|---:|']
 for row in differences:lines.append(f"| {row['dataset'].split('_')[-1]} | {row['model']} | {row['dice_official_minus_gated']:+.6f} | {row['hd95_official_minus_gated']:+.2f} | {row['nsd_official_minus_gated']:+.6f} |")
 lines+=['','## 协议','', '- Encoder 冻结，官方 ViT-L/16，层索引 [5,11,17,23]；官方 Primus 参数 44,576,514，无 LAST/频域 gates。','- 全部 head 初始化 SHA 一致，seed 20260907，batch1，3000 steps，AdamW/CE+Dice/采样/验证选择/四翻转 TTA 与历史协议匹配。','- 标签仅 GT==1 / prediction==1，HD95 与 NSD 使用固定 MedDINOv3 官方测试入口。NSD 容差1，距离为 voxel-equivalent，未校准 nm。','- SD 为 CMCC 3 / CMCT 2 个验证体积间样本标准差，不是 seeds 的波动；单 seed 不作显著性结论。','- B_full 为 transductive 开发资产；标注完整性、物理 spacing 与 acquisition 隔离尚待补证，尚不能证明超过 MitoNet。','- 既有 checkpoint、数据 manifest、历史门控版结果和网站代码保持原样；本次只训练新官方解码器。']
 (R/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')

def main():
 assert website_ok();assert json.loads((R/'audit/smoke.json').read_text())['status']=='PASS'
 registry=json.loads((R/'model_registry.json').read_text())['models'];gpu=inventory();baseline=gpu_apps()
 assert all(gpu[i]['free_mib']>16000 for i in [1,2,3])
 allowed={}
 for app in baseline:
  if app['uuid'] in [gpu[i]['uuid'] for i in [1,2,3]]:
   proc=psutil.Process(app['pid']);cmd=proc.cmdline()
   assert '/opt/data/3_117/tanghr/mira_superhuman_demo_20260825/venv/bin/python' in cmd
   allowed[app['pid']]=proc.create_time()
 protected=[Path(m['checkpoint_path']) for m in registry]+[Path(registry[0]['resolved_config_path']),S2/'src/frozen_probe.py',EXT/'src/frozen_probe_label1.py',Path(registry[0]['primus_source_path'])]
 protected.append(Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120/data/pretrain_all_mitoverse.json'))
 for ds in DATASETS:
  protected.append(DATA/'nnUNet_preprocessed'/ds/'splits_final.json')
  for prefix,parent in [('official_natural_',S2/'branch_E/control_20261011/runs'),('B_full_',S2/'branch_E/runs'),('B_full150k_',EXT/'evaluation')]:protected.append(parent/(prefix+ds)/'metrics.json')
 before={str(p):sha(p) for p in protected};atomic(R/'audit/protected_before.json',before)
 for m in registry:assert before[m['checkpoint_path']]==m['sha256']
 children=[];handles=[]
 try:
  for i,m in zip([1,2,3],registry):
   env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu[i]['uuid'],'B_FULL_CHECKPOINT':m['checkpoint_path'],'B_FULL_STEP':str({'P0':0,'B100k':100000,'B150k':150000}[m['id']]),
    'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONPATH':'/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps'}
   log=(R/'runs'/(m['id']+'_worker.log')).open('w');handles.append(log)
   children.append(subprocess.Popen([PYTHON,'-u',__file__,'--worker',m['id']],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
  while True:
   status=[]
   for proc,m in zip(children,registry):
    row={'model':m['id'],'pid':proc.pid,'exit_code':proc.poll(),'runs':[]}
    for ds in DATASETS:
     path=R/'runs'/(m['id']+'_'+ds)
     progress=safe_json(path/'progress.json')
     metric=safe_json(path/'metrics.json')
     row['runs'].append({'dataset':ds,'progress':progress,'complete':bool(metric and metric['status']=='complete')})
    status.append(row)
   atomic(R/'STATUS.json',{'stage':'official_primus_reprobe','status':'running','workers':status,'updated_unix':time.time()})
   if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Official probe worker failed')
   if not website_ok():raise RuntimeError('Website unhealthy; stopping own workers')
   for app in gpu_apps():
    if app['uuid'] not in [gpu[i]['uuid'] for i in [1,2,3]]:continue
    known=app['pid'] in allowed and psutil.Process(app['pid']).create_time()==allowed[app['pid']]
    if not known and not any(own(app['pid'],p.pid) for p in children):raise RuntimeError('New external process on assigned GPU; stopping own workers')
   if all(p.poll()==0 for p in children):break
   time.sleep(10)
  atomic(R/'STATUS.json',{'status':'scoring','head_training_complete':True,'workers':status})
  os.environ['CUDA_VISIBLE_DEVICES']=''
  score_and_report()
  after={str(p):sha(p) for p in protected};assert before==after
  atomic(R/'audit/protected_integrity.json',{'status':'PASS','before':before,'after':after})
  assert website_ok()
  atomic(R/'STATUS.json',{'status':'complete','head_training_complete':True,'metric_computation_complete':True,'website_http_after':200,'physical_gpus':[1,2,3]})
  with tarfile.open(R/'delivery.tar.gz','w:gz') as tar:
   for name in ['results.json','summary.csv','per_volume_metrics.csv','official_vs_gated_deltas.csv','REPORT_zh.md','STATUS.json','build_audit.json','audit','references','src']:
    tar.add(R/name,arcname=name)
   for folder in (R/'runs').iterdir():
    if folder.is_dir():
     for name in ['metrics.json','protocol.json','progress.json','segmentation_metrics.csv']:
      if (folder/name).exists():tar.add(folder/name,arcname='runs/'+folder.name+'/'+name)
     for name in ['curve.json']:
      if (folder/folder.name.split('_',1)[1]/name).exists():tar.add(folder/folder.name.split('_',1)[1]/name,arcname='runs/'+folder.name+'/'+name)
  print('OFFICIAL COMPARISON COMPLETE',flush=True)
 except BaseException as e:
  for p in children:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
  atomic(R/'STATUS.json',{'status':'failed','error':str(e),'only_own_workers_stopped':True});raise
 finally:
  for h in handles:h.close()

if __name__=='__main__':
 if len(sys.argv)==3 and sys.argv[1]=='--worker':worker(sys.argv[2])
 else:main()
