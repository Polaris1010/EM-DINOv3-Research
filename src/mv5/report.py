"""Four-system matched-region comparison; historic metrics are read-only."""
import csv,math,statistics
from pathlib import Path
from common import R,M3,MODEL,read,atomic,sha,config,rows
OLD=M3.parent/'mv4_official_mitonet_20261014'
METRICS=['dice','precision','recall','hd95_nm','nsd_16nm','predicted_foreground_fraction']
MODELS=['G-P0','G-B100','Official-MitoNet-v1',MODEL]
def avg(v):
 v=[x for x in v if x is not None];return statistics.mean(v) if v else None
def fmt(v):return 'NA' if v is None else 'inf' if math.isinf(v) else f'{v:.5f}'
def csvout(path,items):
 fields=list(dict.fromkeys(k for r in items for k in r))
 with Path(path).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(items)
def main():
 c=config();expected={r['id'] for r in rows()};new=[read(p) for p in (R/'runs'/MODEL).glob('*_metrics.json')]
 assert len(new)==38 and {r['region_id'] for r in new}==expected
 old=list(csv.DictReader((OLD/'MV4_per_region_metrics.csv').open()));assert len(old)==266
 for r in old:
  for k in METRICS:r[k]=float(r[k]) if r[k] else None
  for k in ['valid_voxels','gt_foreground_voxels']:r[k]=int(r[k])
 for r in new:r['seed']=c['seed']
 for r in old:
  other=next(n for n in new if n['region_id']==r['region_id'])
  for k in ['valid_voxels','gt_foreground_voxels','valid_mask_policy','manifest_sha256']:assert r[k]==other[k],(r['region_id'],k)
 allrows=old+new;csvout(R/'MV5_per_region_metrics.csv',allrows);sources=sorted({r['source_id'] for r in new});summary=[]
 for s in sources:
  for m in MODELS:
   selected=[r for r in allrows if r['model']==m and r['source_id']==s]
   summary.append({'source_id':s,'model':m,'regions':len({r['region_id'] for r in selected}),'seeds':3 if m in MODELS[:2] else 1,**{k:avg([r[k] for r in selected]) for k in METRICS}})
 csvout(R/'MV5_source_metrics.csv',summary);idx={(r['source_id'],r['model']):r for r in summary};deltas=[]
 for s in sources:
  item={'source_id':s,**{m+'_dice':idx[s,m]['dice'] for m in MODELS}}
  item.update(adaptation_dice_gain=idx[s,MODEL]['dice']-idx[s,'Official-MitoNet-v1']['dice'],B100_minus_adapted=idx[s,'G-B100']['dice']-idx[s,MODEL]['dice']);deltas.append(item)
 csvout(R/'MV5_source_deltas.csv',deltas);paired=[]
 for r in new:
  original=next(x for x in old if x['region_id']==r['region_id'] and x['model']=='Official-MitoNet-v1')
  paired.append({'region_id':r['region_id'],'source_id':r['source_id'],**{k+'_adapted_minus_official':r[k]-original[k] if r[k] is not None and original[k] is not None else None for k in METRICS}})
 csvout(R/'MV5_adaptation_paired_deltas.csv',paired)
 macro={m:{k:avg([idx[s,m][k] for s in sources]) for k in METRICS} for m in MODELS}
 checkpoints=[read(R/'training'/f'model_step{n:06d}.json') for n in c['milestones']];csvout(R/'MV5_checkpoint_hashes.csv',checkpoints)
 wins=sum(d['B100_minus_adapted']>0 for d in deltas);gap=macro['G-B100']['dice']-macro[MODEL]['dice'];gain=macro[MODEL]['dice']-macro['Official-MitoNet-v1']['dice']
 result={'status':'complete','source_macro':macro,'adaptation_gain':gain,'B100_minus_adapted':gap,'B100_source_wins':wins,'regions':38,'sources':12,'reused_metric_rows':266,'new_metric_rows':38,'total_rows':304,'protocol_sha256':sha(R/'MV5_protocol.json'),'manifest_sha256':c['manifest_sha256'],'checkpoint_hashes':checkpoints,'training':read(R/'training/training_complete.json'),'limitations':c['limitations']};atomic(R/'MV5_summary.json',result)
 # Reuse the established display method without changing any inference or decisions.
 import types
 mod=types.ModuleType('mv5_display')
 source=(OLD/'src/report.py').read_text().replace('Official MitoNet foreground','MitoNet-MV foreground').replace('Official MitoNet instances','MitoNet-MV instances')
 exec(compile(source,str(OLD/'src/report.py'),'exec'),mod.__dict__)
 mod.R=R;mod.MODEL=MODEL
 mod.cases(new,old)
 (R/'MV4_failure_cases.csv').rename(R/'MV5_failure_cases.csv')
 lines=['# MV-5：同 MitoVerse 监督区域的 MitoNet 适配对照','',f'MitoNet-MV 来源等权 Dice **{macro[MODEL]["dice"]:.6f}**；官方未适配 MitoNet **{macro["Official-MitoNet-v1"]["dice"]:.6f}**，适配增益 **{gain:+.6f}**。G-B100 **{macro["G-B100"]["dice"]:.6f}**，与适配后差距 **{gap:+.6f}**；B100 在 {wins}/12 来源更高。','',
 '同 MV-Generalist-v2 的156训练区域/38开发区域、12来源；正式test及官方validation不进入训练。Han忽略mask、整幅GT沿用用户确认，保留部分标注的解释限制。G-P0/G-B100三seed和官方MitoNet已有266条指标原样复用。本轮仅一个MitoNet适配seed，结果不能当作多seed稳定性或独立acquisition显著性证明。','',
 '## 固定训练与推理','',
 '从原官方MitoNet_v1 TorchScript权重直接继续，官方训练forward、中心热图sigma6、偏移和PointRend目标、官方PanopticLoss权重1/200/0.01/1，top20% CE。全模型55158486参数训练；源内合法Z采样、源间等权、50%前景中心，native256 crop、水平/垂直翻转，batch4，AdamW weight_decay0.1，OneCycleLR max_lr0.003/pct_start0.3，AMP/clip12。固定40000实际optimizer updates，保存10k/20k/40k，固定40k选模，不用开发GT调阈值。','',
 '官方FINETUNE提供完整损失/目标；headless入口复用官方configure_optimizer。为沿用本研究的源均衡原生采样，未使用napari默认batch16、随机缩放/旋转/亮度增强或weight_gamma采样。因此仅称使用官方panoptic机制的目标域适配，不能称所有训练超参数均为官方默认。小于256的区域只对人工padding忽略loss；未改真实GT有效区。部分输入没有可确认的独立实例标号时，从二值crop做4连通伪实例；多ID直接保留语义并稠密重编号。此约定不支持真实实例AP/PQ的结论。实际target_mode计数见training_complete.json。','',
 'AdamW/OneCycleLR来自官方微调配置；MitoNet全模型与冻结DINOv3原版Primus在结构、上游数据、batch4对batch1、优化器和监督曝光次数上仍不同。这是同监督区域完整系统比较，不是严格等计算/等优化预算或自监督预训练消融。','',
 '推理完全沿用MV4固定官方Engine2d流程：native scale1、阈值0.5、center0.1、NMS3、整图、padding16、coarse boundaries、TTA OFF，不读GT box。保留panoptic、每层全局不重复instance IDs及前景，未做三维追踪。指标调用既有MV2物理evaluator：Dice、Precision、Recall、HD95 nm、NSD@16nm，Turner spacing限制继续披露。','',
 '| 来源 | G-P0 | G-B100 | 官方 MitoNet | MitoNet-MV | 适配增益 | B100−适配后 |','|---|---:|---:|---:|---:|---:|---:|']
 lines+=['| '+d['source_id']+' | '+' | '.join(fmt(d[m+'_dice']) for m in MODELS)+' | '+fmt(d['adaptation_dice_gain'])+' | '+fmt(d['B100_minus_adapted'])+' |' for d in deltas]
 lines+=['','## 完整指标','','| 来源 | 模型 | Dice | Precision | Recall | HD95 nm | NSD@16nm | 前景比例 |','|---|---|---:|---:|---:|---:|---:|---:|']
 lines+=['| '+r['source_id']+' | '+r['model']+' | '+' | '.join(fmt(r[k]) for k in METRICS)+' |' for r in summary]
 lines+=['','## 结论与下一项决策','',
 f'适配改变了系统差距：官方基线原差距0.216240，当前差距{gap:+.6f}。监督适配收益不能归于DINOv3自监督创新，B100对P0约0.012707的原结论保持。Guay/Müller逐源变化见上表；不能把四区域MV4B的机制排除扩张为其余34区已核验。',
 ('适配后MitoNet已接近或超过G-B100；原21.62个百分点不得作为核心预训练创新优势。优先研究有明确来源依据的DINOv3表示收益，下一实验需另行授权。' if gap<.03 else '适配后G-B100仍有开发集系统优势；应结合本轮逐来源失败案例判断剩余任务/表示差距，不能直接宣称新SSL机制有效。下一实验需另行授权。'),'',
 '本轮完成后停止，不继续MV4B实例分组排查、不开始SSL、尺度/阈值搜索或新模型。现有开发结果受transductive暴露及同acquisition空间划分限制；训练完40k不等于证明收敛。训练曲线保留用于描述，禁止为某个模型临时加预算。','',
 '## 复现与哈希','',f'Protocol SHA256 `{result["protocol_sha256"]}`；manifest `{c["manifest_sha256"]}`。','']
 lines+=[f'- step {j["step"]}: `{j["sha256"]}`，`{j["path"]}`' for j in checkpoints]
 (R/'MV5_final_report.md').write_text('\n'.join(lines)+'\n')
 atomic(R/'MV5_results_verification.json',{'status':'PASS','rows':304,'new_regions':38,'sources':12,'matching_valid_GT_counts':True,'historic_rows_reused':266,'protocol_sha256':result['protocol_sha256'],'checkpoint_sha256':checkpoints[-1]['sha256']})
if __name__=='__main__':main()
