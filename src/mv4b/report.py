"""Report measured path differences without changing historical MV4 claims."""
import argparse,csv,math,statistics
import numpy as np
from shared import R,M4,M3,MODEL,read,sha,atomic,rows,evaluation_arrays,image_array,config,output
KEYS=['dice','precision','recall','hd95_nm','nsd_16nm','predicted_foreground_fraction']
def avg(xs):
 xs=[x for x in xs if x is not None];return statistics.mean(xs) if xs else None
def fmt(x):return 'NA' if x is None else 'inf' if math.isinf(x) else f'{x:.6g}' if x!=0 and abs(x)<1e-4 else f'{x:.6f}'
def csvout(path,rr):
 keys=list(dict.fromkeys(k for r in rr for k in r))
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rr)
def figures(rr):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 byid={r['id']:r for r in rows()};folder=R/'comparison_figures';folder.mkdir(exist_ok=True);selected=[]
 for source in sorted({r['source_id'] for r in rr}):
  q=max((r for r in rr if r['source_id']==source),key=lambda r:r['semantic_foreground_lost_in_panoptic']);region=q['region_id'];row=byid[region]
  z=max(q['plane_stats'],key=lambda s:s['semantic_lost_in_panoptic'])['z_local']
  selection='largest semantic foreground loss'
  if q['semantic_foreground_lost_in_panoptic']==0:
   oldcases=list(csv.DictReader((M4/'MV4_failure_cases.csv').open()))
   previous=next((x for x in oldcases if x['region_id']==region),None)
   if previous is not None:z=int(previous['z_local']);selection='reuse prior MV4 visualization plane; all new semantic-loss planes tied at zero'
  gt,v,_=evaluation_arrays(row);im,_=image_array(row)
  valid=v[z]>0 if v is not None else np.ones(im[z].shape,bool)
  prob=np.load(output()/(region+'_probability.npy'),mmap_mode='r')[z]
  raw=np.load(output()/(region+'_raw_threshold.npy'),mmap_mode='r')[z];sem=np.load(output()/(region+'_prediction.npy'),mmap_mode='r')[z]
  pan=np.load(M4/'runs/Official-MitoNet-v1'/(region+'_prediction.npy'),mmap_mode='r')[z]
  din=[np.load(M3/'runs'/m/'seed20261014'/(region+'_prediction.npy'),mmap_mode='r')[z] for m in ['G-P0','G-B100']]
  panels=[im[z],np.ma.masked_where(~valid,gt[z]),prob,raw,sem,pan,*din]
  titles=['Same uint8 native input','GT (invalid masked)','Raw semantic probability','Raw probability >=0.5','Official Semantic Only','Existing official Panoptic union','G-P0 seed20261014','G-B100 seed20261014']
  fig,axes=plt.subplots(2,4,figsize=(18,10.5))
  for i,(ax,a,title) in enumerate(zip(axes.flat,panels,titles)):
   ax.imshow(a,cmap='magma' if i==2 else 'gray',interpolation='nearest',**({'vmin':0,'vmax':1} if i==2 else {}));ax.set_title(title);ax.axis('off')
  fig.suptitle(f'{source}: {region}; local Z={z}\nSemantic Dice={fmt(q["semantic"]["dice"])}, Panoptic Dice={fmt(q["panoptic"]["dice"])}; thresholds fixed')
  fig.subplots_adjust(left=.02,right=.98,bottom=.04,top=.86,wspace=.10,hspace=.28)
  path=folder/(source+'.png');fig.savefig(path,dpi=110);plt.close(fig)
  selected.append({'source_id':source,'region_id':region,'z_local':z,'z_native':z+row['region_zyx'][0][0],'file':str(path),'sha256':sha(path),'selection':selection+'; visualization only'})
 csvout(R/'MV4B_figures.csv',selected)
def report(pilot=False):
 c=config();ids=c['pilot_regions'] if pilot else [r['id'] for r in rows()]
 suffix='_path_comparison.json' if pilot else '_metrics.json'
 rr=[read(output()/(region+suffix)) for region in ids];assert len(rr)==(4 if pilot else 38)
 paths=[]
 for r in rr:
  item={k:r[k] for k in ['region_id','source_id','semantic_minus_panoptic_dice','semantic_foreground_lost_in_panoptic','lost_semantic_foreground_fraction','panoptic_foreground_outside_semantic','semantic_true_positive_lost_in_panoptic','zero_center_planes','lost_planes']}
  for mode in ['raw_threshold','semantic','panoptic']:
   for k in ['dice','precision','recall','predicted_foreground_fraction']:item[mode+'_'+k]=r[mode][k]
  paths.append(item)
 csvout(R/'MV4B_path_comparison.csv',paths)
 base=list(csv.DictReader((M4/'MV4_per_region_metrics.csv').open()));base=[r for r in base if r['region_id'] in ids]
 for r in base:
  for k in KEYS:r[k]=float(r[k]) if r[k] else None
  if r['model']=='Official-MitoNet-v1':r.update(model='MitoNet-Panoptic-v1',historical_model='Official-MitoNet-v1')
 for r in rr:
  new={k:r[k] for k in ['region_id','source_id','protocol_sha256','manifest_sha256','checkpoint_sha256','spacing_status','valid_mask_policy','tta']}
  new.update(model='MitoNet-Semantic-v1',**r['semantic']);base.append(new)
 csvout(R/'MV4B_per_region_metrics.csv',base)
 sources=sorted({r['source_id'] for r in rr});models=['G-P0','G-B100','MitoNet-Panoptic-v1','MitoNet-Semantic-v1'];srows=[]
 for source in sources:
  for model in models:
   pp=[r for r in base if r['source_id']==source and r['model']==model]
   srows.append({'source_id':source,'model':model,**{k:avg([r.get(k) for r in pp]) for k in KEYS}})
 csvout(R/'MV4B_source_metrics.csv',srows)
 macro={model:{k:avg([r[k] for r in srows if r['model']==model]) for k in KEYS} for model in models}
 sidx={(r['source_id'],r['model']):r for r in srows};wins=sum(sidx[s,'G-B100']['dice']>sidx[s,'MitoNet-Semantic-v1']['dice'] for s in sources)
 historical=list(csv.DictReader((M4/'MV4_source_deltas.csv').open()));sens=[]
 for scope,excluded in [('all12',set()),('without_Guay_Muller',{'guay21','muller21'}),('without_Guay_Muller_Han',{'guay21','muller21','han24'})]:
  hh=[r for r in historical if r['source_id'] not in excluded];b=avg([float(r['B100_dice']) for r in hh]);p=avg([float(r['MitoNet_dice']) for r in hh]);sens.append({'scope':scope,'sources':len(hh),'B100_dice':b,'historical_panoptic_dice':p,'B100_minus_panoptic':b-p})
 share=sum(float(r['B100_minus_MitoNet']) for r in historical if r['source_id'] in ['guay21','muller21'])/sum(float(r['B100_minus_MitoNet']) for r in historical)
 csvout(R/'MV4B_historical_sensitivity.csv',sens)
 figures(rr)
 summary={'status':'complete','scope':'four-region pilot' if pilot else 'full38 development','regions':len(ids),'sources':len(sources),'source_macro':macro,
 'B100_source_wins_vs_semantic':wins,'historical_MV4_sensitivity':sens,'Guay_Muller_fraction_of_historical_B100_panoptic_gain':share,
 'protocol_sha256':sha(R/'MV4B_protocol.json'),'official_weight_sha256':c['official_weight_sha256'],'historical_outputs_preserved':True}
 atomic(R/'MV4B_summary.json',summary)
 lines=['# MV-4B：Official Semantic vs Panoptic输出复核','',
  '同官方MitoNet_v1权重、同uint8图像/GT/有效区、norms0.57571/0.12765、scale1、confidence0.5、tile0、TTA OFF。只改变官方semantic_only选项；原始概率在同一次官方forward捕获，无GT阈值/尺度调参。MV3/MV4结果与预测全部只读保留。','',
  f'实际范围：{len(ids)}开发区域/{len(sources)}来源。'+('按用户条件先完成四区域；未将这四来源均值冒充12来源结果。' if pilot else '四区域差异明确后，根据单次保存的extension_decision扩展到原38开发区域。'),'',
  '## 原始概率、官方语义与原Panoptic差异','',
  '| 区域 | raw>=0.5 Dice | Semantic Only Dice | Panoptic Dice | Semantic−Panoptic | Semantic前景丢失比例 | 无中心平面 |','|---|---:|---:|---:|---:|---:|---:|']
 lines += [f'| {r["region_id"]} | {fmt(r["raw_threshold"]["dice"])} | {fmt(r["semantic"]["dice"])} | {fmt(r["panoptic"]["dice"])} | {fmt(r["semantic_minus_panoptic_dice"])} | {fmt(r["lost_semantic_foreground_fraction"])} | {r["zero_center_planes"]} |' for r in rr]
 lines += ['','完整Precision/Recall/前景比例及有效前景丢失TP、丢失平面在MV4B_path_comparison.csv；每平面中心数量/保留比例在每区域JSON。官方Semantic Only把类1视为stuff，单平面总前景小于64时过滤；raw阈值掩码另存，全部平面已断言两者仅由该官方规则不同。','',
  '## 系统同图像指标','', '| 系统 | 来源等权Dice | Precision | Recall | HD95 nm | NSD@16nm | 前景比例 |','|---|---:|---:|---:|---:|---:|---:|']
 lines += ['| '+model+' | '+' | '.join(fmt(macro[model][k]) for k in KEYS)+' |' for model in models]
 lines += ['',f'G-B100在{wins}/{len(sources)}来源高于官方Semantic Only。DINOv3来源均值使用原三个decoder seeds，Semantic/Panoptic分别为同一个官方固定模型的不同输出路径。','',
  '## 每来源Dice','', '| 来源 | P0 | B100 | Panoptic | Semantic Only | B100−Semantic |','|---|---:|---:|---:|---:|---:|']
 lines += [f'| {s} | {sidx[s,"G-P0"]["dice"]:.6f} | {sidx[s,"G-B100"]["dice"]:.6f} | {sidx[s,"MitoNet-Panoptic-v1"]["dice"]:.6f} | {sidx[s,"MitoNet-Semantic-v1"]["dice"]:.6f} | {sidx[s,"G-B100"]["dice"]-sidx[s,"MitoNet-Semantic-v1"]["dice"]:+.6f} |' for s in sources]
 lines += ['','## 历史MV4敏感性分析（不删除主结果）','', '| 范围 | 来源数 | B100 | 既有Panoptic | 差值 |','|---|---:|---:|---:|---:|']
 lines += [f'| {r["scope"]} | {r["sources"]} | {r["B100_dice"]:.6f} | {r["historical_panoptic_dice"]:.6f} | {r["B100_minus_panoptic"]:+.6f} |' for r in sens]
 lines += ['',f'用完整精度原指标复算，Guay/Müller合计贡献历史来源等权差值的{share:.2%}。这是描述性敏感性分析，不改变12来源主指标或历史报告。','',
  '## 表述与结论边界','',
  'MV4的0.610209必须命名为“released MitoNet_v1逐层2D Panoptic前景union的语义Dice”。此轮Semantic Only才是直接语义输出主对照；不能把两个成绩混成一个MitoNet名次。MitoVerse监督适配后的DINOv3使用156区域标签，MitoNet未经本项目标签适配；官方训练重叠未核验，不能绝对称为无训练暴露的零样本。B100预训练含全部MitoVerse无标签图像；不是公平独立跨域测试或等监督预训练方法比较。匹配P0/B100的EM持续预训练收益仍约+0.012707。','',
  '原图与输入灰度、物理scale1、2D而非orthoplane3D的限制继续披露；本轮未做原始强度/极性转换或尺度优化，输入分布机制未被操纵。Han整幅GT按用户指令继承，GT外输出不直接证明生物学假阳性；Turner spacing UNVERIFIED继续限制nm指标。没有真实实例GT，未计算AP/PQ，不将panoptic语义覆盖差距称作实例分割优劣。','',
  f'权重SHA256 `{c["official_weight_sha256"]}`；MV4B protocol SHA256 `{sha(R/"MV4B_protocol.json")}`。资产/源码/环境复用MV4已验证版本；实际部署commit与每文件SHA见evidence/deployment*.json。概率、raw threshold、official semantic留runs/{MODEL}/；旧实例/前景仍在MV4目录。','',
  'comparison_figures/*.png提供原图、GT、概率、raw>=0.5、Semantic Only、旧Panoptic、两种DINOv3首seed的八面板图；选择最大语义前景丢失平面仅用于解释，没有反馈推理。','',
  '本轮完成后停止，不自动运行灰度诊断、MitoNet fine-tuning、尺度搜索、3D或新的SSL。']
 (R/'MV4B_final_report.md').write_text('\n'.join(lines)+'\n')
 print('MV4B_REPORT_DONE',len(ids),macro['MitoNet-Semantic-v1']['dice'],wins,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();report(a.pilot)
