"""Native-grid paired development metrics, explicitly separating volume and seed SD."""
from pathlib import Path
import hashlib,ast
import sys,json,csv,collections,math,statistics
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),'/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps']
from mv_data import manifest,open_cache,sha,atomic
import numpy as np,torch
from monai.metrics import compute_hausdorff_distance,compute_surface_dice
from metric_cache import install
verification=json.loads((R/'evidence/metric_memoization_verification.json').read_text())
assert verification['status']=='PASS' and verification['cache_source_sha256']==sha(R/'src/metric_cache.py')
compute_hausdorff_distance,compute_surface_dice,_metric_cache=install(compute_hausdorff_distance,compute_surface_dice)
def csvout(path,rows):
 assert rows
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(values):
 values=[float(x) for x in values if x is not None and not math.isnan(float(x))]
 if not values:return None,None,0
 if any(math.isinf(x) for x in values):return math.inf,None,len(values)
 return statistics.mean(values),statistics.stdev(values) if len(values)>1 else None,len(values)
def score(run):
 # Use the very same official MONAI functions as MedDINOv3's evaluation entrypoint.
 torch.set_num_threads(2);meta=json.loads((run/'metrics.json').read_text());assert meta['status']=='complete' and not meta['pilot']
 assert meta['manifest_sha256']==sha(R/'MV-Dev-v1.json') and meta['protocol_sha256']==sha(R/'MV_frozen_probe_protocol.yaml')
 rows=[]
 for r in meta['validation']['rows']:
  row=next(x for x in manifest()['entries'] if x['id']==r['volume_id']);im,gt=open_cache(row)
  pred=np.load(run/(row['id']+'_prediction.npy'),mmap_mode='r');assert pred.shape==gt.shape
  assert all(int(n) in {0,1} for n in np.unique(pred))
  p=torch.from_numpy(np.asarray(pred,dtype=bool).copy())[None,None];g=torch.from_numpy(np.asarray(gt,dtype=bool).copy())[None,None]
  if not g.any():raise ValueError('Audited nonempty volume unexpectedly has no GT')
  spacing=row['spacing_zyx_nm']
  if p.any():
   hd=float(compute_hausdorff_distance(p,g,include_background=True,percentile=95,spacing=spacing)[0,0])
   hdv=float(compute_hausdorff_distance(p,g,include_background=True,percentile=95,spacing=[1,1,1])[0,0])
   nsd={nm:float(compute_surface_dice(p,g,class_thresholds=[nm],include_background=True,spacing=spacing)[0,0]) for nm in [8,16,32]}
   nsdv=float(compute_surface_dice(p,g,class_thresholds=[1],include_background=True,spacing=[1,1,1])[0,0])
  else:hd=hdv=math.inf;nsd={nm:0. for nm in [8,16,32]};nsdv=0.
  out={'model':meta['model'],'seed':meta['seed'],'volume_id':row['id'],'source_id':row['source_id'],
   'dice':r['dice'],'hd95_nm':hd,'nsd_16nm':nsd[16],'precision':r['precision'],'recall':r['recall'],
   'nsd_8nm':nsd[8],'nsd_32nm':nsd[32],'hd95_voxel':hdv,'nsd_1voxel':nsdv,
   'tp':r['tp'],'fp':r['fp'],'fn':r['fn'],'tn':r['tn'],'empty_prediction':r['foreground_empty_prediction'],
   'spacing_z_nm':spacing[0],'spacing_y_nm':spacing[1],'spacing_x_nm':spacing[2],'modality':row['modality'],
   'ssl_exposed':True,'supervised_role':'dev','acquisition_independence':'UNVERIFIED','locked_steps':meta['steps']}
  # Independently cross-check confusion values from stored predictions/labels.
  assert int(np.logical_and(pred>0,gt>0).sum())==out['tp'] and int(np.logical_and(pred>0,gt==0).sum())==out['fp']
  rows.append(out);print('SCORED',meta['model'],meta['seed'],row['id'],out['dice'],hd,nsd[16],flush=True)
 csvout(run/'per_volume_metrics.csv',rows)
 atomic(run/'scoring_receipt.json',{'status':'PASS','csv_sha256':sha(run/'per_volume_metrics.csv'),
  'run_metrics_sha256':sha(run/'metrics.json'),'manifest_sha256':sha(R/'MV-Dev-v1.json'),
  'protocol_sha256':sha(R/'MV_frozen_probe_protocol.yaml'),'score_logic_sha256':score_logic_sha(),
  'prediction_sha256':{r['volume_id']:sha(run/(r['volume_id']+'_prediction.npy')) for r in rows}})
 return rows
def score_logic_sha():
 tree=ast.parse(Path(__file__).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='score')
 return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
def score_or_reuse(run):
 receipt=run/'scoring_receipt.json'
 if not receipt.exists():return score(run)
 j=json.loads(receipt.read_text());assert j['status']=='PASS' and j['score_logic_sha256']==score_logic_sha()
 assert j['csv_sha256']==sha(run/'per_volume_metrics.csv') and j['run_metrics_sha256']==sha(run/'metrics.json')
 assert j['manifest_sha256']==sha(R/'MV-Dev-v1.json') and j['protocol_sha256']==sha(R/'MV_frozen_probe_protocol.yaml')
 for name,h in j['prediction_sha256'].items():assert h==sha(run/(name+'_prediction.npy'))
 rows=list(csv.DictReader((run/'per_volume_metrics.csv').open()))
 integer={'seed','tp','fp','fn','tn','locked_steps'};boolean={'empty_prediction','ssl_exposed'}
 text_fields={'model','volume_id','source_id','modality','supervised_role','acquisition_independence'}
 for row in rows:
  for k,v in row.items():
   if k in integer:row[k]=int(v)
   elif k in boolean:row[k]=v=='True'
   elif k not in text_fields:row[k]=float(v) if v else None
 print('REUSED_AUDITED_SCORE',str(run),flush=True);return rows
def strata_and_panels():
 import zarr
 from PIL import Image,ImageDraw
 candidates=list((R/'runs').glob('*/seed20261013'))
 assert len(candidates)==3
 strata=[];panels=R/'failure_panels';panels.mkdir(exist_ok=True)
 for row in manifest()['entries']:
  if row['supervised_role']!='dev':continue
  g=zarr.open_group(row['image_path'],mode='r');im,lab=open_cache(row);N=int(g['mito'][:1].max())
  # Whole volume is admitted; original instance ids do not define train/dev membership.
  count={};gray={};ring={};touch=set();tp={name:{} for name in ['P0','B100k','B150k']}
  preds={r.parent.name:np.load(r/(row['id']+'_prediction.npy'),mmap_mode='r') for r in candidates}
  per_z_error={name:[] for name in preds}
  for z in range(im.shape[0]):
   ids=np.asarray(g['mito'][z]);a=np.asarray(im[z]);K=int(ids.max())+1
   c=np.bincount(ids.ravel(),minlength=K);sums=np.bincount(ids.ravel(),weights=a.ravel(),minlength=K)
   for k in np.flatnonzero(c[1:])+1:count[int(k)]=count.get(int(k),0)+int(c[k]);gray[int(k)]=gray.get(int(k),0)+float(sums[k])
   if z in {0,im.shape[0]-1}:touch.update(map(int,np.unique(ids)))
   touch.update(map(int,np.unique(np.concatenate([ids[0],ids[-1],ids[:,0],ids[:,-1]]))))
   for inside,outside,ig,og in [(ids[:,1:],ids[:,:-1],a[:,1:],a[:,:-1]),(ids[:,:-1],ids[:,1:],a[:,:-1],a[:,1:]),(ids[1:],ids[:-1],a[1:],a[:-1]),(ids[:-1],ids[1:],a[:-1],a[1:])]:
    valid=(inside>0)&(outside==0);ix=inside[valid];weights=og[valid]
    rc=np.bincount(ix,minlength=K);rs=np.bincount(ix,weights=weights,minlength=K)
    for k in np.flatnonzero(rc):
     old=ring.get(int(k),(0.,0));ring[int(k)]=(old[0]+float(rs[k]),old[1]+int(rc[k]))
   for name,p in preds.items():
    hit=np.bincount(ids[p[z]>0].ravel(),minlength=K)
    for k in np.flatnonzero(hit[1:])+1:tp[name][int(k)]=tp[name].get(int(k),0)+int(hit[k])
    per_z_error[name].append(int(np.count_nonzero((p[z]>0)!=(ids>0))))
  size_values=[v*np.prod(row['spacing_zyx_nm']) for k,v in count.items() if k not in touch]
  cuts=np.quantile(size_values,[1/3,2/3]) if size_values else [np.nan,np.nan]
  contrasts={k:abs(gray[k]/count[k]-ring[k][0]/ring[k][1]) for k in count if k in ring and ring[k][1]}
  contrast_cut=float(np.median(list(contrasts.values()))) if contrasts else np.nan
  for k,n in count.items():
   size=float(n*np.prod(row['spacing_zyx_nm']));size_bin='ROI_truncated' if k in touch else ('small' if size<=cuts[0] else ('medium' if size<=cuts[1] else 'large'))
   for name in preds:
    strata.append({'model':name,'seed':20261013,'volume_id':row['id'],'source_id':row['source_id'],'instance_id':k,
    'voxel_count':n,'size_nm3':size,'size_bin_within_volume':size_bin,'ROI_border_touch':k in touch,
    'contrast_uint8':contrasts.get(k),'contrast_bin_within_volume':'UNVERIFIED' if k not in contrasts else ('low' if contrasts[k]<=contrast_cut else 'high'),
    'voxel_recall':tp[name].get(k,0)/n,'modality':row['modality'],'other_membrane_proximity':'UNVERIFIED'})
  # Three predetermined views plus worst B100 slice; panels are development-only failure diagnostics.
  zs=sorted(set([0,im.shape[0]//2,im.shape[0]-1,int(np.argmax(per_z_error['B100k']))]))
  for z in zs:
   # Display downscales for inspection only; metrics always on native grid.
   W,H=im.shape[2],im.shape[1];scale=min(1.,512/max(H,W));wh=(max(1,int(W*scale)),max(1,int(H*scale)))
   ims=[]
   for title,seg in [('EM',None),('GT',lab[z])]+[(name,p[z]) for name,p in preds.items()]:
    a=np.asarray(im[z]);rgb=np.repeat(a[:,:,None],3,2).copy()
    if seg is not None:
     mask=seg>0;rgb[mask]=(.5*rgb[mask]+.5*np.array([255,190,20])).astype(np.uint8)
    panel=Image.fromarray(rgb).resize(wh);ims.append((title,panel))
   canvas=Image.new('RGB',(len(ims)*wh[0],wh[1]+28),'white');draw=ImageDraw.Draw(canvas)
   for j,(title,panel) in enumerate(ims):canvas.paste(panel,(j*wh[0],28));draw.text((j*wh[0]+6,7),title,fill='black')
   canvas.save(panels/(row['id']+f'_z{z:04d}.png'))
 csvout(R/'target_strata_seed1.csv',strata)
def strata_or_reuse():
 receipt=R/'evidence/seed1_diagnostics_receipt.json'
 if not receipt.exists():return strata_and_panels()
 j=json.loads(receipt.read_text());assert j['status']=='PASS' and j['manifest_sha256']==sha(R/'MV-Dev-v1.json')
 node=next(n for n in ast.parse(Path(__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='strata_and_panels')
 assert j['logic_sha256']==hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
 for name,h in {**j['predictions_sha256'],**j['outputs_sha256']}.items():assert sha(R/name)==h,name
 print('REUSED_SEED1_DIAGNOSTICS',flush=True)

def summarize(rows):
 metrics=['dice','hd95_nm','nsd_16nm','precision','recall'];groups=collections.defaultdict(list)
 for r in rows:groups[(r['model'],r['seed'],r['source_id'])].append(r)
 summaries=[]
 for (model,seed,source),rs in sorted(groups.items()):
  a={'model':model,'seed':seed,'source_id':source,'n_dev_volumes':len(rs),'eligible_for_primary_macro':len(rs)>=2}
  for met in metrics:
   mean,sd,n=stats([r[met] for r in rs]);a.update({met+'_mean':mean,met+'_sd_volumes':sd,met+'_n_defined':n})
  summaries.append(a)
 csvout(R/'source_metrics_by_seed.csv',summaries)
 multiseed=[]
 for model in ['P0','B100k','B150k']:
  for source in sorted({r['source_id'] for r in rows}):
   rs=[r for r in summaries if r['model']==model and r['source_id']==source]
   a={'model':model,'source_id':source,'n_seeds':len(rs),'n_dev_volumes_per_seed':rs[0]['n_dev_volumes']}
   for met in metrics:
    mean,sd,n=stats([r[met+'_mean'] for r in rs]);a.update({met+'_mean':mean,met+'_sd_seeds':sd})
   multiseed.append(a)
 csvout(R/'source_metrics_three_seeds.csv',multiseed)
 index={(r['model'],r['seed'],r['volume_id']):r for r in rows};paired=[]
 for r in rows:
  if r['model'] not in {'B100k','B150k'}:continue
  for baseline in ['P0']+(['B100k'] if r['model']=='B150k' else []):
   ref=index[baseline,r['seed'],r['volume_id']];a={'model':r['model'],'baseline':baseline,'seed':r['seed'],'source_id':r['source_id'],'volume_id':r['volume_id']}
   for met in metrics:a['delta_'+met]=r[met]-ref[met] if r[met] is not None and ref[met] is not None else None
   paired.append(a)
 csvout(R/'paired_volume_deltas.csv',paired)
 aggregate=[]
 for (model,seed) in sorted({(r['model'],r['seed']) for r in rows}):
  rs=[r for r in summaries if r['model']==model and r['seed']==seed];primary=[r for r in rs if r['eligible_for_primary_macro']]
  a={'model':model,'seed':seed,'n_descriptive_sources':len(rs),'n_primary_sources':len(primary),
   'primary_macro_interpretation':'only BetaSeg qualifies; not a robust cross-source generalist summary',
   'primary_macro_dice':float(np.mean([r['dice_mean'] for r in primary])) if primary else None,
   'primary_worst_source_dice':min(r['dice_mean'] for r in primary) if primary else None,
   'descriptive_all_source_macro_dice':float(np.mean([r['dice_mean'] for r in rs]))}
  aggregate.append(a)
 csvout(R/'aggregate_by_seed.csv',aggregate)
 return summaries,multiseed,paired,aggregate
def report(rows,multiseed,paired,aggregate):
 p=json.loads((R/'MV_frozen_probe_protocol.yaml').read_text())
 def fmt(x):return 'NA' if x is None else f'{x:.6f}'
 lines=['# MV-1 首批多来源冻结表征评测','','## 完成范围','',
  f"原版 Primus，P0/B100k/B150k，3 个 decoder seeds，每组 {p['locked_steps']} steps。3 sources，4 train volumes，4 dev volumes，共 36 个体积×模型×seed 结果。",
  '本批为 MV-Dev-v1 的首批可准入来源，不代表 13 sources 全部评测完成。',
  '物理 acquisition independence 未证实；B_full 的所有图像均在 SSL 全量池。结果仅为 transductive source-volume 开发验证。',
  '标签由发布者声明为经 proofreading 的线粒体实例，且完整数值 QC 通过；这不是独立重标注完整性认证。',
  '', '## 每来源结果：均值 ± decoder seed SD','',
  '下表 SD 为 3 seeds 的来源均值波动。体积间 SD 单独存放在 source_metrics_by_seed.csv；Guay/Jiang n=1 时为 NA。',
  '', '| Source | Encoder | n volumes | Dice ↑ | HD95 nm ↓ | NSD@16nm ↑ | Precision ↑ | Recall ↑ |',
  '|---|---|---:|---:|---:|---:|---:|---:|']
 for r in multiseed:
  cells=[fmt(r[k+'_mean'])+' ± '+fmt(r[k+'_sd_seeds']) for k in ['dice','hd95_nm','nsd_16nm','precision','recall']]
  lines.append('| '+r['source_id']+' | '+r['model']+' | '+str(r['n_dev_volumes_per_seed'])+' | '+' | '.join(cells)+' |')
 lines+=['','## P0 → B100 / B100 → B150：逐体积配对','',
 '| Source | Volume | ΔDice B100−P0（3 seeds mean） | ΔDice B150−B100 | ΔNSD16 B100−P0 | ΔHD95nm B100−P0 |',
 '|---|---|---:|---:|---:|---:|']
 for vid in sorted({r['volume_id'] for r in rows}):
  a=[r for r in paired if r['volume_id']==vid and r['model']=='B100k' and r['baseline']=='P0'];b=[r for r in paired if r['volume_id']==vid and r['model']=='B150k' and r['baseline']=='B100k']
  lines.append('| '+a[0]['source_id']+' | '+vid+' | '+fmt(stats([r['delta_dice'] for r in a])[0])+' | '+fmt(stats([r['delta_dice'] for r in b])[0])+' | '+fmt(stats([r['delta_nsd_16nm'] for r in a])[0])+' | '+fmt(stats([r['delta_hd95_nm'] for r in a])[0])+' |')
 lines+=['','## 来源与目标失效分析','',
  '- failure_panels/：开发体积首/中/末切片与 B100 最大像素错误切片，显示原图、GT、三种预测；仅用于误差审查。',
  '- target_strata_seed1.csv：首 seed 每个 GT 实例的体素 recall、nm³体积、对比度、边界截断标记与 modality。大小分位数按各开发体积定义，未用于选训练样本。',
  '- 对比度为实例内部平均灰度与跨 XY 边界邻接背景灰度的差；背景按 crossing edge 加权，不能直接当作膜强度。',
  '- 仅有线粒体标签，无法可靠判定附近膜属于 ER 等类别；other_membrane_proximity 保持 UNVERIFIED。',
  '- 主 source-macro/worst-source 要求 >=2 个开发 source volumes：仅 BetaSeg 达标。不能将当前主 macro 解读成充分验证的多来源通用成绩。',
  '- descriptive_all_source_macro 在 aggregate_by_seed.csv 中作为探索性摘要，明确包括 n1 来源；无显著性声明。',
  '- HD95/NSD 采用与 MedDINOv3 原入口相同的 MONAI 函数。物理 spacing 来自官方 ingest 的 XYZ → ZYX；NSD 容差16nm是预先固定的工程容差，同时给出8/32nm与1voxel敏感性。不能与历史NSD@1voxel混比。',
  ]
 # Source-level directions across matched decoder seeds; these are readout variability checks, not independent cohorts.
 lines+=['','### 预训练收益方向在decoder seeds间是否一致','',
 '| Source | B100−P0 ΔDice mean | ΔDice seed SD | Positive seeds | B150−B100 ΔDice mean |',
 '|---|---:|---:|---:|---:|']
 lookup={(r['model'],r['source_id']):r for r in multiseed}
 for source in sorted({r['source_id'] for r in rows}):
  ds=[];later=[]
  for seed in sorted({r['seed'] for r in rows}):
   vals={name:statistics.mean(r['dice'] for r in rows if r['seed']==seed and r['source_id']==source and r['model']==name) for name in ['P0','B100k','B150k']}
   ds.append(vals['B100k']-vals['P0']);later.append(vals['B150k']-vals['B100k'])
  lines.append('| '+source+' | '+fmt(statistics.mean(ds))+' | '+fmt(statistics.stdev(ds))+' | '+str(sum(x>0 for x in ds))+'/3 | '+fmt(statistics.mean(later))+' |')
 lines+=['','decoder seeds仅表示相同开发标签上的训练波动，不增加独立采集体数量；表中方向一致不能替代跨域验证。',
 '', '### 当前数据支持的结论', '',
 '仅对这批开发体积、固定decoder和共同训练预算解释结果。来源间相反的收益不能由一个总体均值掩盖；CMCC上的改善也不能直接外推到所有MitoVerse来源。',
 '20k是共同预算预实验到达的预设上限，未达到预定义收敛规则；这些结果不等于任一encoder的监督适配能力上限。',
 '100k→150k的分割变化本身不足以证明dense-feature degradation；Gram选择还需要特征诊断证据。',
 '', '### 首seed目标分层（按共同GT定义）','',
 '| Source | GT stratum | Nontruncated instances | P0 voxel Recall | B100 voxel Recall | B150 voxel Recall |',
 '|---|---|---:|---:|---:|---:|']
 objects=list(csv.DictReader((R/'target_strata_seed1.csv').open()))
 for source in sorted({r['source_id'] for r in rows}):
  for field,bin_name in [('size_bin_within_volume','small'),('size_bin_within_volume','large'),('contrast_bin_within_volume','low'),('contrast_bin_within_volume','high')]:
   subset=[o for o in objects if o['source_id']==source and o[field]==bin_name and o['ROI_border_touch']=='False']
   values=[];n=0
   for name in ['P0','B100k','B150k']:
    osub=[o for o in subset if o['model']==name];n=len(osub)
    denominator=sum(int(o['voxel_count']) for o in osub)
    values.append(sum(float(o['voxel_recall'])*int(o['voxel_count']) for o in osub)/denominator if denominator else None)
   lines.append('| '+source+' | '+('size-' if field.startswith('size') else 'contrast-')+bin_name+' | '+str(n)+' | '+' | '.join(fmt(v) for v in values)+' |')
 lines+=['','此处是该GT实例集合内的体素Recall，不能当作实例检测Recall；边界截断实例排除。对比度和大小分位在各开发体积内定义，不能解释成跨来源统一物理尺寸阈值。',
 '不同来源同时改变了modality、采样间距、目标分布和监督样本；当前来源差异不能单独归因到物理尺度或某一种EM modality。',
 '', '## 下一步','',
  '1. 扩展有效来源及每来源独立开发体积数；补齐 parent acquisition/ROI、标注完整性与 ignore 极性证据。',
  '2. 对照本批 paired volume/source/目标分层结果确定主要误差；再决定 dense-feature、物理尺度或分辨率诊断。',
  '3. MitoNet 对照使用独立冻结协议；本批没有训练/评测 MitoNet，不能宣布超过 MitoNet。',
  '4. 不自动训练 Gram/HRA/LAST/TopoMask 或延长 SSL；P 持续暂停。']
 (R/'source_failure_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 audit=['# MV-Dev-v1 split/exposure audit','',
  '- 数据发布固定 revision d0af9a6fd11faaf16ce0822233b7bbf72553ce7d；代码 revision df884f6bf353b3cd6b4b6a5ee3620daabd8a79be。',
  '- Guay vol0=train / vol1=dev / vol2=reserved test；Jiang vol1=train / vol2=dev / vol0=reserved test，沿用官方 overlay。',
  '- BetaSeg high_c1/low_c1=train；high_c3/low_c2=dev，全部来自官方 TRAIN 集；high_c2/high_c4/low_c3仍为 reserved test。',
  '- 8 stores 与 source image-file identity 在监督 train/dev 间不重复，坐标是完整已存 ZYX volume；未推断 parent crop offset。',
  '- 未取得物理 acquisition 独立性的证据，acquisition_id保留null；禁止 unseen acquisition 结论。已知重叠 crop / split 不完整来源未纳入。',
  '- B100/B150 全部8volumes在正式SSL全量manifest中：ssl_exposed=true仅表示采样池暴露，不声称每像素都被抽到。P0在本项目未进行 EM SSL；不能证实其自然预训练集合不存在 EM。',
  '- 历史424样本 GT crop diagnostic覆盖全部212volume，开发曝光记录不能清零；reserved test不意味着原来untouched。',
  '- 本次训练/预算/选模/误差图未读reserved test labels，全部36评测行的supervised_role=dev。',
  '- 官方 split overlays、SSL encoder checkpoints、历史 FAIL/结果及网站 Primus 源码哈希保护；本批仅新建cache、head与输出。',
  '- 严格 B_clean 的 Stage0 NO_GO 保留；本开发基准不解锁严格训练/最终测试。',
  '- 官方Primus代码孤立导入，未修改函数或共用网站模块；无FFT、频域gate、LAST。',
  '- 协议单seed通过训练及完整评测后才扩到3seeds；每个seed同head初始化与同采样序列，所有模型固定同steps。',
  '- 体积SD（ddof1）与seedSD分开报告；不足样本不填0，空预测保留Dice0、NSD0、HD95inf并显式标记。']
 (R/'split_exposure_audit.md').write_text('\n'.join(audit)+'\n',encoding='utf8')
def main(single_seed=False):
 import concurrent.futures,multiprocessing
 import psutil
 runs=[r for r in sorted((R/'runs').glob('*/seed*')) if (r/'metrics.json').exists()]
 rows=[]
 # Native 3D EDT has large temporary arrays. Reserve 96 GiB per worker, with a four-worker cap.
 workers=min(4,max(1,int(psutil.virtual_memory().available//(96*2**30))))
 print('SCORING_WORKERS',workers,flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as ex:
  for batch in ex.map(score_or_reuse,runs):rows.extend(batch)
 if single_seed:
  assert len(rows)==12 and len({(r['model'],r['seed'],r['volume_id']) for r in rows})==12
  assert all(0<=r['dice']<=1 and 0<=r['nsd_16nm']<=1 and r['hd95_nm']>=0 for r in rows)
  assert len({(r['source_id'],r['volume_id']) for r in rows})==4
  csvout(R/'single_seed_results.csv',rows)
  atomic(R/'evidence/single_seed_result_audit.json',{'status':'PASS','result_rows':12,'models':3,'sources':3,'metric_ranges_valid':True,'full_grid_confusion_recomputed':True,'test_labels_accessed':False})
  return
 assert len(rows)==36 and len({(r['model'],r['seed'],r['volume_id']) for r in rows})==36
 csvout(R/'MV_frozen_probe_results.csv',rows);summaries,multiseed,paired,aggregate=summarize(rows)
 strata_or_reuse()
 report(rows,multiseed,paired,aggregate)
 atomic(R/'analysis_completion.json',{'status':'complete','rows':len(rows),'groups':9,'test_labels_accessed':False})
 print('MV_ANALYSIS_COMPLETE',flush=True)
if __name__=='__main__':main('--single-seed' in sys.argv)
