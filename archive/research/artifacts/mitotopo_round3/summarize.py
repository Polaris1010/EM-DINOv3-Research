"""Validate full-volume counts and summarize only completed comparable probes."""
import json,math,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
RUNS={
 'em':('probe_em_final','EM，无额外 SSL'),
 'gram200k':('probe_gram200k_v3','旧版 Gram 200k，旧学习率'),
 'b2_w0':('probe_w0','batch 2 / λ=0 / 修正 LR'),
 'b2_w1':('probe_w1','batch 2 / λ=1 / 修正 LR'),
 'b2_w2':('probe_w2','batch 2 / λ=2 / 修正 LR'),
 'b4_w0':('probe_ssl_b4_w0_s2000','batch 4 / λ=0 / 修正 LR'),
 'b4_w2':('probe_ssl_b4_w2_s2000','batch 4 / λ=2 / 修正 LR'),
 'b2_w0_lr1':('probe_ssl_b2_w0_lr1_s2000','batch 2 / λ=0 / 旧 LR'),
 'b2_w2_lr1':('probe_ssl_b2_w2_lr1_s2000','batch 2 / λ=2 / 旧 LR'),
}
DATASETS=['Dataset111_CMCC','Dataset131_CMCT']
def main():
 reference=json.loads((BASE/'cached_metrics.json').read_text());models={};pending=[]
 for key,(run,label) in RUNS.items():
  p=BASE/'jobs'/run
  if not (p/'guard_result.json').exists() or not (p/'metrics.json').exists():pending.append(key);continue
  g=json.loads((p/'guard_result.json').read_text());m=json.loads((p/'metrics.json').read_text())
  if g['exit_code']!=0 or g['stop_reason'] is not None or m['status']!='complete':pending.append(key);continue
  for ds in DATASETS:
   ref=reference['datasets'][ds];assert set(m['datasets'][ds]['cases'])==set(ref['split']['val'])
   for case,c in m['datasets'][ds]['cases'].items():
    truth=ref['models']['meddinov3_base_primus_multiscale_Trainer']['cases'][case]
    assert sum(c[k] for k in ['tp','fp','fn','tn'])==math.prod(ref['cases'][case]['shape_xyz'])
    assert c['tp']+c['fn']==truth['tp']+truth['fn']
  m.update(label=label,run=run,equal_dataset_mean=sum(m['datasets'][ds]['macro_dice'] for ds in DATASETS)/2);models[key]=m
 contrasts={}
 for name,treat,control in [('Gram λ1 vs λ0, b2','b2_w1','b2_w0'),('Gram λ2 vs λ0, b2','b2_w2','b2_w0'),('Gram λ2 vs λ0, b4','b4_w2','b4_w0'),('Gram λ2 vs λ0, b2 old LR','b2_w2_lr1','b2_w0_lr1'),('LR correction, b2 λ2','b2_w2','b2_w2_lr1'),('old 200k vs original EM (confounded by extra SSL)','gram200k','em')]:
  if treat not in models or control not in models:continue
  contrasts[name]={'treatment':treat,'control':control,'dataset_delta_pp':{ds:100*(models[treat]['datasets'][ds]['macro_dice']-models[control]['datasets'][ds]['macro_dice']) for ds in DATASETS},'case_delta_pp':{case:100*(v['dice']-models[control]['datasets'][ds]['cases'][case]['dice']) for ds in DATASETS for case,v in models[treat]['datasets'][ds]['cases'].items()}}
 report={'status':'complete' if not pending else 'partial','timestamp':time.time(),'protocol':'single-seed fixed-budget frozen-backbone probes; 3000 decoder updates per dataset; selected on stride-20 validation, final all 1481 slices on original grid; not independent test or converged full fine-tuning','models':models,'pending':pending,'contrasts':contrasts}
 (BASE/'comparison.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
 lines=['# 匹配分割头比较','',f"状态：{report['status']}。这是同种子、同监督训练预算的冻结编码器验证实验，不是独立测试或充分收敛的全量微调。",'', '| 编码器 / SSL 参数 | CMCC Dice | CMCT Dice | 两数据集等权均值 |','|---|---:|---:|---:|']
 for key,(run,label) in RUNS.items():
  if key in models:
   m=models[key];lines.append(f"| {label} | {m['datasets'][DATASETS[0]]['macro_dice']:.6f} | {m['datasets'][DATASETS[1]]['macro_dice']:.6f} | {m['equal_dataset_mean']:.6f} |")
  else:lines.append(f'| {label} | 尚未完成 | 尚未完成 | — |')
 lines+=['','λ=0 与 λ>0 的同 batch、同 LR 对照用于判断 Gram 在本轮预算内的影响。batch=4 看过的图像数为 batch=2 的两倍，因此跨 batch 排名不等于纯 batch 因果效应。','', '## 配对差值（百分点）','', '| 对照 | CMCC ΔDice | CMCT ΔDice |','|---|---:|---:|']
 for name,c in contrasts.items():lines.append(f"| {name} | {c['dataset_delta_pp'][DATASETS[0]]:+.3f} | {c['dataset_delta_pp'][DATASETS[1]]:+.3f} |")
 lines+=['','逐体积计数、差值及选中训练步数见 comparison.json。未完成或异常退出的运行不参与排名。']
 (BASE/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps({'status':report['status'],'pending':pending,'scores':{k:[m['datasets'][d]['macro_dice'] for d in DATASETS] for k,m in models.items()},'contrasts':contrasts},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
