"""Report completed MV3 only, with paired new metrics and stored-prediction references."""
import argparse,collections,csv,json,math,statistics,sys
from pathlib import Path
import numpy as np
from data import R,OLD,MV2,manifest,read,atomic,sha,open_arrays

METRICS=['dice','precision','recall','hd95_nm','nsd_16nm','predicted_foreground_fraction']
def csvout(path,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
def avg(values):
 values=[v for v in values if v is not None];return statistics.mean(values) if values else None
def sd(values):
 values=[v for v in values if v is not None]
 return statistics.stdev(values) if len(values)>1 and all(math.isfinite(v) for v in values) else None
def fmt(v):return 'NA' if v is None else 'inf' if math.isinf(v) else f'{v:.4f}'
def overlap_counts(pred,gt,valid):
 a=pred[valid];b=gt[valid];tp=int((a&b).sum());fp=int((a&~b).sum());fn=int((~a&b).sum());tn=int((~a&~b).sum())
 return {'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'tp':tp,'fp':fp,'fn':fn,'tn':tn}

def historical_reference():
 """No old head is loaded/inferred; recompute overlap of stored predictions on NEW heldout regions."""
 rows=[];unavailable=[];old_manifest=read(MV2/'MV_dev_validation_manifest.json');oldrows={r['volume_id']:r for r in old_manifest['entries']}
 for row in manifest()['entries']:
  if row['supervised_role']!='dev':continue
  old=oldrows.get(row['volume_id'])
  if old is None:
   unavailable.append({'region_id':row['id'],'reason':'No heldout MV2 prediction for this volume; do not rerun old heads'});continue
  bounds=row['region_zyx'];previous=old['region_zyx']
  if not all(a<=x<y<=b for (a,b),(x,y) in zip(previous,bounds)):
   unavailable.append({'region_id':row['id'],'reason':'New region outside stored MV2 prediction; no rerun'});continue
  _,gt,valid,j=open_arrays(row);key=tuple(slice(x-a,y-a) for (a,b),(x,y) in zip(previous,bounds))
  for model in ['P0','B100k']:
   for seed in [20261013,20261014,20261015]:
    p=MV2/'runs'/model/f'seed{seed}'/(row['volume_id']+'_prediction.npy')
    if not p.exists():p=OLD/'runs'/model/f'seed{seed}'/(row['volume_id']+'_prediction.npy')
    if not p.exists():unavailable.append({'region_id':row['id'],'model':model,'seed':seed,'reason':'Stored prediction unavailable; no rerun'});continue
    prediction=np.load(p,mmap_mode='r')[key];assert prediction.shape==gt.shape
    counts=np.zeros(4,np.int64)
    for z in j['valid_z']:
     v=valid[z].astype(bool) if valid is not None else np.ones(gt[z].shape,bool)
     c=overlap_counts(prediction[z]>0,gt[z]>0,v);counts+=np.array([c[k] for k in ['tp','fp','fn','tn']])
    tp,fp,fn,tn=map(int,counts)
    rows.append({'region_id':row['id'],'volume_id':row['volume_id'],'source_id':row['source_id'],'historical_model':model,'historical_seed':seed,'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'valid_mask_policy':row['valid_mask_policy'],'reference_type':'MV2_stored_prediction_NEW_MV3_holdout_and_mask; not_historical_MV2_metric','historical_MV1_supervised_exposed':row['historical_MV1_supervised_exposed']})
 csvout(R/'MV3_historical_prediction_reference.csv',rows);atomic(R/'MV3_historical_reference_unavailable.json',unavailable);return rows

def report(first=False):
 protocol=read(R/'MV3_protocol.json');seeds=protocol['seed_sequence'][:1] if first else protocol['seed_sequence']
 expected={r['id'] for r in manifest()['entries'] if r['supervised_role']=='dev'};rows=[];checkpoints=[];curves=[]
 for model in ['G-P0','G-B100']:
  for seed in seeds:
   out=R/'runs'/model/f'seed{seed}';assert read(out/'validation_complete.json')['status']=='complete'
   rr=[read(p) for p in out.glob('*_metrics.json')];assert {r['region_id'] for r in rr}==expected and len(rr)==len(expected)
   rows.extend(rr);completion=read(out/'training_complete.json');assert completion['completed_steps']==40000 and completion['encoder_unchanged']
   for step in [10000,20000,40000]:
    sidecar=read(out/f'head_step{step:06d}.json');assert sidecar['step']==step
    checkpoints.append({'model':model,'seed':seed,'step':step,'path':sidecar['path'],'sha256':sidecar['sha256'],'encoder_sha256':completion['encoder_sha256'],'decoder_initial_sha256':completion['decoder_initial_sha256'],'manifest_sha256':completion['manifest_sha256'],'protocol_sha256':completion['protocol_sha256']})
   curves.extend({'model':model,'seed':seed,'step':c['step'],'loss_median200':c['loss_median200'],'source_macro_dice':c['source_macro_dice']} for c in completion['curve'])
 assert len(rows)==len(seeds)*2*len(expected)
 flatkeys=['model','seed','region_id','volume_id','source_id','dev_kind','acquisition_id',*METRICS,'tp','fp','fn','tn','valid_voxels','gt_foreground_voxels','prediction_foreground_voxels','spacing_status','tta','head_sha256','manifest_sha256','valid_mask_policy']
 csvout(R/('MV3_first_seed_metrics.csv' if first else 'MV3_per_region_metrics.csv'),[{k:r.get(k) for k in flatkeys} for r in rows])
 summary=[];seed_macro=collections.defaultdict(list)
 for source in sorted({r['source_id'] for r in rows}):
  for model in ['G-P0','G-B100']:
   source_rows=[r for r in rows if r['source_id']==source and r['model']==model];entry={'source_id':source,'model':model,'n_regions':len({r['region_id'] for r in source_rows}),'n_decoder_seeds':len(seeds)}
   for metric in METRICS:
    vals=[avg([r.get(metric) for r in source_rows if r['seed']==s]) for s in seeds];entry[metric+'_mean']=avg(vals);entry[metric+'_seed_sd']=sd(vals)
   entry['hd95_inf_rows']=sum(math.isinf(r['hd95_nm']) for r in source_rows);summary.append(entry)
 for model in ['G-P0','G-B100']:
  for seed in seeds:seed_macro[model].append(avg([avg([r['dice'] for r in rows if r['model']==model and r['seed']==seed and r['source_id']==source]) for source in sorted({r['source_id'] for r in rows})]))
 index={(r['model'],r['seed'],r['region_id']):r for r in rows};pairs=[]
 for seed in seeds:
  for region in sorted(expected):
   a=index['G-P0',seed,region];b=index['G-B100',seed,region];pair={'seed':seed,'region_id':region,'source_id':a['source_id']}
   for metric in METRICS:
    av=a.get(metric);bv=b.get(metric);pair[metric+'_P0']=av;pair[metric+'_B100']=bv;pair[metric+'_delta']=bv-av if av is not None and bv is not None and math.isfinite(av) and math.isfinite(bv) else None
   pairs.append(pair)
 overall={model:{'source_equal_dice_mean':avg(seed_macro[model]),'decoder_seed_sd':sd(seed_macro[model])} for model in ['G-P0','G-B100']}
 overall['B100_minus_P0']=avg(seed_macro['G-B100'])-avg(seed_macro['G-P0'])
 atomic(R/('MV3_first_seed_summary.json' if first else 'MV3_summary.json'),{'stage':'MV-3','overall':overall,'sources':summary,'seeds':seeds,'region_count':len(expected),'tta':False,'selection':'fixed_step40000'})
 csvout(R/'MV3_checkpoint_hashes.csv',checkpoints);csvout(R/'MV3_training_curves.csv',curves)
 if not first:
  csvout(R/'MV3_source_metrics.csv',summary);csvout(R/'MV3_P0_B100_paired_deltas.csv',pairs)
  failures=sorted(rows,key=lambda r:r['dice'])[:36];csvout(R/'MV3_failure_cases.csv',[{k:r.get(k) for k in flatkeys} for r in failures])
 gains={s:avg([r['dice_delta'] for r in pairs if r['source_id']==s]) for s in sorted({r['source_id'] for r in pairs})}
 lines=['# MV-3 Generalist Frozen Segmentation '+('首seed报告' if first else '最终报告'),'','状态：'+('首seed两组训练及全来源开发指标完成；另外两个seeds仍须执行。' if first else '2 encoders × 3 decoder seeds全部完成；每run40000 optimizer steps，38 dev区域、12来源。'),'',
 '## 协议与实际执行','',f"MV-Generalist-v1:156 train区域/12来源，38 dev区域/12来源。manifest SHA `{sha(R/'MV-Generalist-v1.json')}`；protocol SHA `{sha(R/'MV3_protocol.json')}`。",'冻结官方DINOv3 ViT-L/16，原版Primus_Multiscale，[5,11,17,23]，无gates，TTA=OFF。共同source-aware采样、初始化、预算和固定40k选模；10k/20k仅监测。共同AdamW及native-grid预处理见锁定protocol；逐seed匹配回执见evidence/matched_seed*.json。','每来源实际训练/dev体积、有效体素与区域见MV3_actual_supervision.json；train/dev空间互斥，正式test不读取。官方validation保留；自建dev已重新分配，重新进入训练的区域不作为MV3验证。',
 'Han沿用已知mask!=0有效末端，mask外正标签也ignore；MV2原mask约定及历史指标不改写。Casser/Xiao为同acquisition不重叠Z，不能声称独立采集泛化。OpenOrganelle按既有sample分组留出。Wei-H/Lucchi的既有test坐标缺失，整资产不用于MV3；Wei-R提供Wei来源。Turner物理间距仍UNVERIFIED。','B100k此前见过全量无标签图像，因此结论限定transductive development。','',
 '## 来源等权总览','', '|模型|Dice均值|decoder seed SD|','|---|---:|---:|']
 for model in ['G-P0','G-B100']:lines.append(f"|{model}|{fmt(overall[model]['source_equal_dice_mean'])}|{fmt(overall[model]['decoder_seed_sd'])}|")
 lines+=['',f"B100−P0来源等权ΔDice：{fmt(overall['B100_minus_P0'])}。",'', '## 每来源结果','', '|来源|模型|区域数|Dice|Precision|Recall|HD95 nm|NSD@16nm|预测前景比例|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for row in summary:lines.append('|'+row['source_id']+'|'+row['model']+'|'+str(row['n_regions'])+'|'+'|'.join(fmt(row[m+'_mean']) for m in METRICS)+'|')
 lines+=['','HD95 inf保留；undefined precision和SD标NA，不静默删除失败。逐区域同seed配对及前景比例见CSV。seeds是decoder波动，不增加独立生物acquisition数量。','', '## 共同预算收敛状态','', '|模型|seed|step|训练loss中位数|dev监测Dice|','|---|---:|---:|---:|---:|']
 for c in curves:lines.append(f"|{c['model']}|{c['seed']}|{c['step']}|{fmt(c['loss_median200'])}|{fmt(c['source_macro_dice'])}|")
 lines+=['','40k是预设公平比较预算，不证明收敛。10k/20k/40k曲线如仍有改善，报告尚未证明收敛；不单独延长某个encoder。']
 if not first:
  references=historical_reference();ref_index={(r['source_id'],r['region_id'],r['historical_model']) for r in references};comparisons=[]
  for source in sorted(gains):
   for new,old in [('G-P0','P0'),('G-B100','B100k')]:
    common={r['region_id'] for r in references if r['source_id']==source and r['historical_model']==old}
    before=avg([r['dice'] for r in references if r['source_id']==source and r['historical_model']==old and r['region_id'] in common]);after=avg([r['dice'] for r in rows if r['source_id']==source and r['model']==new and r['region_id'] in common])
    comparisons.append({'source_id':source,'model':new,'common_heldout_regions':len(common),'historical_stored_prediction_dice':before,'MV3_dice_on_same_regions':after,'delta':after-before if before is not None and after is not None else None})
  csvout(R/'MV3_same_region_supervision_gain.csv',comparisons)
  lines+=['','## 五项研究回答','','1. **扩大监督后提高多少：** 下表只在相同MV3 heldout区域、相同新有效mask上复用MV2已存预测计算前后Dice；不重新加载/运行旧9个head，不改历史MV2指标。不能把MV2的0.3492/0.4067与新划分总均值直接当作严格配对。','', '|来源|模型|共同区域|旧预测在新有效区Dice|MV3同区域Dice|Δ|','|---|---|---:|---:|---:|---:|']
  for c in comparisons:lines.append(f"|{c['source_id']}|{c['model']}|{c['common_heldout_regions']}|{fmt(c['historical_stored_prediction_dice'])}|{fmt(c['MV3_dice_on_same_regions'])}|{fmt(c['delta'])}|")
  oldmacro={m:avg([c['historical_stored_prediction_dice'] for c in comparisons if c['model']==m and c['common_heldout_regions']]) for m in ['G-P0','G-B100']}
  newmacro={m:avg([c['MV3_dice_on_same_regions'] for c in comparisons if c['model']==m and c['common_heldout_regions']]) for m in ['G-P0','G-B100']}
  positive=[s for s,d in gains.items() if d>0];negative=[s for s,d in gains.items() if d<0]
  lines+=['',f"同区域来源等权监督适配收益：G-P0 {fmt(newmacro['G-P0']-oldmacro['G-P0'])}；G-B100 {fmt(newmacro['G-B100']-oldmacro['G-B100'])}。Han有效mask改变已应用在前后双方；Müller官方high_c1无旧heldout预测，仅共同low_c2参与该项比较。MV2为20k、MV3为40k且decoder seeds不同，收益包含来源覆盖和训练预算的共同改变，不宣称单独增加来源的因果效应。",'',f"2. **B100是否仍有多来源优势：** 同seed同区域G-B100−G-P0总ΔDice {fmt(overall['B100_minus_P0'])}；正向来源{len(positive)}/12：{', '.join(positive)}；负向：{', '.join(negative) or '无'}。逐seed方向见配对CSV。",'', '3. **Guay/Müller是否改善：** 以上同区域监督适配收益表列出两组前后值；B100−P0在Guay为 '+fmt(gains['guay21'])+'，在Müller为 '+fmt(gains['muller21'])+'。Müller新official high_c1过去被MV1监督暴露，不能构造独立旧基线。','']
  b100=sorted([r for r in summary if r['model']=='G-B100'],key=lambda r:r['dice_mean']);worst=b100[:4]
  adaptation_gain=avg([newmacro[m]-oldmacro[m] for m in ['G-P0','G-B100']])
  priority='监督适配' if adaptation_gain>=abs(overall['B100_minus_P0']) else '预训练表示的来源稳健性'
  lines+=['4. **主要短板：** '+', '.join(r['source_id']+' Dice='+fmt(r['dice_mean']) for r in worst)+'。失败案例CSV包含最低Dice区域、Precision/Recall及预测前景比例；区分空预测、假阳性和来源偏差，不能仅归因encoder。','',
   f'5. **当前优先方向：{priority}。** 同区域描述性适配收益平均为{fmt(adaptation_gain)}，新共同协议下encoder差异绝对值为{fmt(abs(overall["B100_minus_P0"]))}；这一幅度比较用于排序已测得的工程改进方向，不是因果分解或新的PASS门槛。输入尺度未被本轮操纵，来源相关尺度退化仍为待检验假设。下一项唯一动作先运行官方MitoNet同图像/有效GT对照，以判断剩余短板是否为本系统特有；本轮不自动改配置或启动新SSL。','',
   '## 下一项研究决策','', 'MV3训练结束并停止。下一项任务是按同一图像与有效真值运行官方MitoNet对照，需后续明确任务；不自动执行MitoNet、Gram/HRA或新预训练。','', '## Checkpoint哈希','', '|模型|seed|step|SHA256|','|---|---:|---:|---|']
  for c in checkpoints:lines.append(f"|{c['model']}|{c['seed']}|{c['step']}|{c['sha256']}|")
 (R/('MV3_first_seed_report.md' if first else 'MV3_final_report.md')).write_text('\n'.join(lines)+'\n')
 print('REPORT_WRITTEN','first_seed' if first else 'final',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--first-seed',action='store_true');a=ap.parse_args();report(a.first_seed)
