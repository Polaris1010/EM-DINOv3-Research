"""Matched-region system comparison with all three existing decoder seeds."""
import collections,csv,math,statistics
from pathlib import Path
from common import R,M3,MODEL,rows,read,atomic,sha,config,evaluation_arrays,image_array
METRICS=['dice','precision','recall','hd95_nm','nsd_16nm','predicted_foreground_fraction']
def avg(xs):
 xs=[x for x in xs if x is not None];return statistics.mean(xs) if xs else None
def fmt(x):return 'NA' if x is None else 'inf' if math.isinf(x) else f'{x:.4f}'
def csvout(path,items):
 keys=list(dict.fromkeys(k for item in items for k in item))
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(items)
def baseline():
 rr=list(csv.DictReader((M3/'MV3_per_region_metrics.csv').open()));assert len(rr)==228
 for r in rr:
  for k in METRICS:r[k]=float(r[k]) if r[k] else None
  for k in ['seed','valid_voxels','gt_foreground_voxels','tp','fp','fn','tn']:r[k]=int(r[k])
 return rr
def cases(mito,base):
 import numpy as np,matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=R/'failure_cases';out.mkdir(exist_ok=True);results=[]
 byid={r['id']:r for r in rows()}
 for source in sorted({r['source_id'] for r in mito}):
  selected=min((r for r in mito if r['source_id']==source),key=lambda r:r['dice']);region=selected['region_id'];row=byid[region]
  gt,v,j=evaluation_arrays(row);im,_=image_array(row)
  path=R/'runs'/MODEL/(region+'_prediction.npy');p=np.load(path,mmap_mode='r')
  errors=[]
  for z in range(gt.shape[0]):
   a=p[z]>0;b=gt[z]>0;valid=v[z]>0 if v is not None else np.ones(a.shape,bool);errors.append(int(((a!=b)&valid).sum()))
  z=int(np.argmax(errors));valid=v[z]>0 if v is not None else np.ones(im[z].shape,bool)
  pp=[np.load(M3/'runs'/model/'seed20261014'/(region+'_prediction.npy'),mmap_mode='r')[z] for model in ['G-P0','G-B100']]
  inst=np.load(R/'runs'/MODEL/(region+'_instances.npy'),mmap_mode='r')[z]
  panels=[im[z],np.ma.masked_where(~valid,gt[z]),*pp,p[z],np.ma.masked_where(inst==0,inst%29)]
  titles=['Original native XY','GT (invalid pixels masked)','G-P0 seed20261014','G-B100 seed20261014','Official MitoNet foreground','Official MitoNet instances (colors repeat)']
  fig,axes=plt.subplots(2,3,figsize=(15,10))
  for ax,a,title in zip(axes.flat,panels,titles):
   ax.imshow(a,cmap='tab20' if 'instances' in title else 'gray',interpolation='nearest');ax.set_title(title);ax.axis('off')
  fig.suptitle(f'{source}: {region}\nlocal Z={z}; region Dice={selected["dice"]:.4f}; example selected AFTER inference by largest pixel-error count')
  fig.tight_layout();target=out/(source+'.png');fig.savefig(target,dpi=110);plt.close(fig)
  results.append({'source_id':source,'region_id':region,'z_local':z,'z_native':z+row['region_zyx'][0][0],'mitonet_dice':selected['dice'],'slice_fp_plus_fn':errors[z],'figure':str(target),'selection':'lowest MitoNet region Dice per source, then maximum valid-pixel FP+FN plane; visualization only'})
 csvout(R/'MV4_failure_cases.csv',results);return results
def report():
 c=config();rr=rows();expected={r['id'] for r in rr};sources=sorted({r['source_id'] for r in rr})
 mito=[read(p) for p in (R/'runs'/MODEL).glob('*_metrics.json')]
 assert len(mito)==38 and {r['region_id'] for r in mito}==expected
 base=baseline();assert {r['region_id'] for r in base}==expected
 idx={r['region_id']:r for r in mito}
 for r in base:
  m=idx[r['region_id']]
  assert r['valid_voxels']==m['valid_voxels'] and r['gt_foreground_voxels']==m['gt_foreground_voxels']
  assert r['manifest_sha256']==m['manifest_sha256']==c['manifest_sha256'] and r['valid_mask_policy']==m['valid_mask_policy']
 for r in mito:assert r['distance_evaluator_source_sha256']==sha(M3.parent/'mv_dev_v2_notta_20261013/src/validation.py')
 allrows=base+mito;csvout(R/'MV4_per_region_metrics.csv',allrows);summary=[];pairs=[]
 for source in sources:
  for model in ['G-P0','G-B100',MODEL]:
   selected=[r for r in allrows if r['source_id']==source and r['model']==model]
   item={'source_id':source,'model':model,'regions':len({r['region_id'] for r in selected}),'decoder_seeds':3 if model!=MODEL else 0}
   for k in METRICS:item[k]=avg([r[k] for r in selected])
   item['spacing_status']=';'.join(sorted({r['spacing_status'] for r in selected}));summary.append(item)
 for r in base:
  m=idx[r['region_id']];item={'region_id':r['region_id'],'source_id':r['source_id'],'dinov3_model':r['model'],'decoder_seed':r['seed']}
  for k in METRICS:item[k+'_dinov3_minus_mitonet']=r[k]-m[k] if r[k] is not None and m[k] is not None else None
  pairs.append(item)
 csvout(R/'MV4_source_metrics.csv',summary);csvout(R/'MV4_paired_deltas.csv',pairs)
 sidx={(r['source_id'],r['model']):r for r in summary};deltas=[]
 for source in sources:
  item={'source_id':source,'P0_dice':sidx[source,'G-P0']['dice'],'B100_dice':sidx[source,'G-B100']['dice'],'MitoNet_dice':sidx[source,MODEL]['dice']}
  item['P0_minus_MitoNet']=item['P0_dice']-item['MitoNet_dice'];item['B100_minus_MitoNet']=item['B100_dice']-item['MitoNet_dice'];deltas.append(item)
 csvout(R/'MV4_source_deltas.csv',deltas)
 macro={model:{k:avg([r[k] for r in summary if r['model']==model]) for k in METRICS} for model in ['G-P0','G-B100',MODEL]}
 wins=sum(r['B100_minus_MitoNet']>0 for r in deltas);lose=sum(r['B100_minus_MitoNet']<0 for r in deltas)
 figures=cases(mito,base);result={'status':'complete','source_macro':macro,'B100_minus_MitoNet_macro_dice':macro['G-B100']['dice']-macro[MODEL]['dice'],
  'B100_source_wins':wins,'B100_source_losses':lose,'source_count':12,'regions':38,'reused_DINOv3_rows':228,'official_MitoNet_rows':38,
  'exclude_OpenOrganelle_B100_minus_P0':avg([sidx[s,'G-B100']['dice']-sidx[s,'G-P0']['dice'] for s in sources if s!='openorganelle']),
  'manifest_sha256':c['manifest_sha256'],'protocol_sha256':sha(R/'MV4_protocol.json'),'official_assets':read(R/'assets/official_assets.json'),
  'limitations':c['limitations']}
 atomic(R/'MV4_summary.json',result)
 decision='多数来源G-B100超过官方MitoNet；下一项优先决策是研究Guay/Müller跨来源稳健性。这里只提出决策，不启动SSL。' if wins>6 else '先针对实际MitoNet优势来源与失败图判断监督适配/实例分离/输入分辨率的主要差距；此对照没有操纵这些因素，不能凭Dice将原因归于某种预训练机制。'
 lines=['# MV-4：Official MitoNet vs DINOv3 Generalist','',
  '三个完整系统在MV-Generalist-v2的同38开发区域、12来源、同有效GT和物理坐标上比较。G-P0/G-B100直接复用40k、三个decoder seeds的既有预测/指标；本轮无训练、无重新推理DINOv3、无TTA修改。','',
  f'来源等权Dice：P0 **{macro["G-P0"]["dice"]:.6f}**，B100 **{macro["G-B100"]["dice"]:.6f}**，官方MitoNet **{macro[MODEL]["dice"]:.6f}**。B100−MitoNet **{result["B100_minus_MitoNet_macro_dice"]:+.6f}**；B100在{wins}/12来源更高、{lose}/12更低。','',
  '## 实际推理与数据协议','',
  '官方MitoNet_v1来自Zenodo6861565，发布者MD5及文件字节数校验后另计算SHA256；完整权重和输出留在服务器。官方empanada-dl0.1.7与hash锁定Engine2d/Preprocessor原定义，通过AST原样编译提供无Qt入口；没有重写模型/后处理。官方二维批量工作流：native XY/Z、scale1、confidence0.5、center threshold0.1、NMS kernel3、label divisor10000、padding16、coarse boundaries、fill holes OFF、TTA OFF；统一tile1280与官方实例合并。','',
  '推理不读取GT/annotation mask值，不使用GT box。所有原生Z推理；有效区仅参与评测。Han按用户确认忽略mask、整幅GT。保留原始逐层panoptic ID，以及跨层不重复的逐层instance ID；这些ID不代表三维追踪。','',
  f'Manifest SHA256 `{c["manifest_sha256"]}`；MV4 protocol SHA256 `{result["protocol_sha256"]}`。实际资产哈希/来源详见MV4_summary.json的official_assets及MV4_protocol.json。','',
  'Dice/Precision/Recall来自有效像素混淆矩阵；HD95 nm和NSD@16nm直接调用MV3沿用的MV2物理距离实现。来源内等区域，来源间等权；DINOv3每个来源先平均三个seed，MitoNet是一个固定官方系统。undefined precision/recall保留NA，inf HD95不隐藏。','',
  '## 每来源Dice','', '| 来源 | G-P0 | G-B100 | Official MitoNet | B100−MitoNet |','|---|---:|---:|---:|---:|']
 lines += [f'| {r["source_id"]} | {r["P0_dice"]:.4f} | {r["B100_dice"]:.4f} | {r["MitoNet_dice"]:.4f} | {r["B100_minus_MitoNet"]:+.4f} |' for r in deltas]
 lines += ['', '## 完整指标','', '| 来源 | 模型 | Dice | Precision | Recall | HD95 nm | NSD@16nm | 前景比例 |','|---|---|---:|---:|---:|---:|---:|---:|']
 lines += ['| '+r['source_id']+' | '+r['model']+' | '+' | '.join(fmt(r[k]) for k in METRICS)+' |' for r in summary]
 lines += ['', '## 典型错误与实例输出','',
  'MV4_failure_cases.csv及failure_cases/*.png保存每来源MitoNet最低Dice区域、该区域最大FP+FN平面，展示原图、有效GT、两种DINOv3首seed、MitoNet前景及实例颜色。此选择发生于完成推理后，只用于检查错误；不会反馈到阈值/推理。其他两个seed仍完整纳入所有数值。','',
  'MitoNet的raw panoptic、逐层instance ID及semantic foreground全部保留在runs/Official-MitoNet-v1/*_{panoptic,instances,prediction}.npy。当前没有将二值GT的连通分量冒充真实实例GT；未声称实例AP/PQ或三维追踪优劣。弱边界/小目标的机制性判断需结合实图，单个语义Dice无法证明。','',
  '## 研究判断与边界','',decision,'',
  f'排除OpenOrganelle后，MV3的B100−P0来源等权增益为{result["exclude_OpenOrganelle_B100_minus_P0"]:+.6f}，与用户指出的收益集中一致。MV4不重复MV3训练，也不将MV2/MV3不同监督数据协议混成排行榜。','',
  '官方MitoNet训练于CEM-MitoLab，其训练图像/标注与我们的MitoVerse156训练区域不同；具体重叠映射未核验，因此本轮是完整系统竞争力开发对照，不是等监督预算、等训练暴露或严格隔离测试。B100有无标签transductive exposure；三个decoder seeds不增加独立acquisition数量。Han整幅GT沿用用户确认；Turner物理spacing仍UNVERIFIED，nm表面指标须附此限制。','',
  '本轮完成后停止。未启动Gram/HRA/LAST/TopoCL、encoder预训练、head延长训练或新对照模型。']
 (R/'MV4_final_report.md').write_text('\n'.join(lines)+'\n')
 print('MV4_REPORT_DONE',macro[MODEL]['dice'],wins,flush=True)
if __name__=='__main__':report()
