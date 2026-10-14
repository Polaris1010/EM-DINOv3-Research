"""Verify collected MV3 arithmetic/coverage and expose matched decoder-seed results.

Uses completed outputs only; does not read images, load models, or launch experiments.
"""
import argparse,csv,datetime,hashlib,json,math,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'experiments/MV-3/v2'
MODELS=['G-P0','G-B100']
METRICS=['dice','precision','recall','hd95_nm','nsd_16nm','predicted_foreground_fraction']

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def number(value):return float(value) if value not in ['',None] else None
def close(actual,expected):
 assert actual==expected or (actual is not None and expected is not None and math.isclose(actual,expected,abs_tol=1e-12,rel_tol=1e-12)),(actual,expected)
def mean(values):
 values=[v for v in values if v is not None]
 return statistics.mean(values) if values else None
def csvwrite(path,rows):
 with path.open('w',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)

def verify(first=False,augment=False):
 index=read(FOLDER/'run_index.json');manifest=read(ROOT/index['manifest']);protocol=read(ROOT/index['protocol'])
 seeds=protocol['seed_sequence'][:1] if first else protocol['seed_sequence']
 expected={row['id']:row for row in manifest['entries'] if row['supervised_role']=='dev'}
 suffix='first_seed' if first else 'final'
 metrics_file=FOLDER/('MV3_first_seed_metrics.csv' if first else 'MV3_per_region_metrics.csv')
 with metrics_file.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
 keys={(r['model'],int(r['seed']),r['region_id']) for r in rows}
 assert len(keys)==len(rows)==2*len(seeds)*len(expected)
 assert keys=={(m,s,r) for m in MODELS for s in seeds for r in expected}
 manifest_hash=sha(ROOT/index['manifest']);protocol_hash=sha(ROOT/index['protocol'])
 for row in rows:
  region=expected[row['region_id']]
  assert row['source_id']==region['source_id'] and row['valid_mask_policy']==region['valid_mask_policy']
  assert row['tta']=='False' and row['manifest_sha256']==manifest_hash
  tp,fp,fn,tn=[int(row[k]) for k in ['tp','fp','fn','tn']];assert min(tp,fp,fn,tn)>=0
  n=tp+fp+fn+tn;assert int(row['valid_voxels'])==n
  assert int(row['gt_foreground_voxels'])==tp+fn and int(row['prediction_foreground_voxels'])==tp+fp
  close(number(row['dice']),2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.)
  close(number(row['precision']),tp/(tp+fp) if tp+fp else None)
  close(number(row['recall']),tp/(tp+fn) if tp+fn else None)
  close(number(row['predicted_foreground_fraction']),(tp+fp)/n)
 summary=read(FOLDER/('MV3_first_seed_summary.json' if first else 'MV3_summary.json'))
 assert summary['seeds']==seeds
 source_ids=sorted({r['source_id'] for r in rows});assert len(source_ids)==12
 for source in summary['sources']:
  rr=[r for r in rows if r['model']==source['model'] and r['source_id']==source['source_id']]
  for metric in METRICS:
   values=[mean([number(r[metric]) for r in rr if int(r['seed'])==s]) for s in seeds]
   close(source[metric+'_mean'],mean(values))
 matched=[];seed_rows=[];checkpoint_count=0
 for seed in seeds:
  completions=[];macro={}
  for model in MODELS:
   out=FOLDER/'runs'/model/f'seed{seed}'
   completion=read(out/'training_complete.json');completions.append(completion)
   assert completion['completed_steps']==40000 and completion['encoder_unchanged'] and completion['encoder_frozen']
   assert completion['tta'] is False and completion['physical_gpu'] in [1,2,3]
   assert completion['manifest_sha256']==manifest_hash and completion['protocol_sha256']==protocol_hash
   assert read(out/'validation_complete.json')['status']=='complete'
   for step in protocol['milestones']:
    side=read(out/f'head_step{step:06d}.json');meta=side['metadata']
    assert side['step']==step and meta['model']==model and meta['seed']==seed
    assert meta['manifest_sha256']==manifest_hash and meta['protocol_sha256']==protocol_hash
    assert len(side['sha256'])==64;checkpoint_count+=1
   head=read(out/'head_step040000.json')['sha256']
   rr=[r for r in rows if r['model']==model and int(r['seed'])==seed];assert all(r['head_sha256']==head for r in rr)
   macro[model]=mean([mean([number(r['dice']) for r in rr if r['source_id']==source]) for source in source_ids])
  a,b=completions
  assert a['decoder_initial_sha256']==b['decoder_initial_sha256'] and a['sampling_hash']==b['sampling_hash'] and a['source_draws']==b['source_draws']
  matched.append({'seed':seed,'matched_initialization':True,'matched_sampling':True,'steps_each':40000})
  seed_rows.append({'seed':seed,'P0_source_equal_dice':macro['G-P0'],'B100_source_equal_dice':macro['G-B100'],'B100_minus_P0':macro['G-B100']-macro['G-P0']})
 for model,column in [('G-P0','P0_source_equal_dice'),('G-B100','B100_source_equal_dice')]:
  values=[r[column] for r in seed_rows]
  close(summary['overall'][model]['source_equal_dice_mean'],statistics.mean(values))
  close(summary['overall'][model]['decoder_seed_sd'],statistics.stdev(values) if len(values)>1 else None)
 deltas=[r['B100_minus_P0'] for r in seed_rows];close(summary['overall']['B100_minus_P0'],statistics.mean(deltas))
 paired_sources=[]
 for source in source_ids:
  values=[]
  for seed in seeds:
   get=lambda model:mean([number(r['dice']) for r in rows if r['source_id']==source and int(r['seed'])==seed and r['model']==model])
   values.append(get('G-B100')-get('G-P0'))
  paired_sources.append({'source_id':source,'paired_dice_delta_mean':statistics.mean(values),'paired_decoder_seed_sd':statistics.stdev(values) if len(values)>1 else None,'positive_decoder_seeds':sum(v>0 for v in values),'decoder_seeds':len(values)})
 csvwrite(FOLDER/f'MV3_{suffix}_decoder_seed_summary.csv',seed_rows)
 csvwrite(FOLDER/f'MV3_{suffix}_source_paired_seed_summary.csv',paired_sources)
 result={'status':'PASS','checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'metric_rows':len(rows),'regions_per_run':len(expected),'sources':12,'decoder_seeds':seeds,'checkpoint_sidecars':checkpoint_count,'matched':matched,'source_equal_paired_delta_mean':statistics.mean(deltas),'source_equal_paired_decoder_seed_sd':statistics.stdev(deltas) if len(deltas)>1 else None,'checkpoint_hash_evidence':'server save/inference digests and collected sidecars; no checkpoint downloads','new_experiments_started':False}
 (FOLDER/f'MV3_{suffix}_results_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
 if augment:
  assert not first
  report=ROOT/'docs/MV3_final_report.md';text=report.read_text(encoding='utf-8');marker='## 三个匹配 decoder seed 的直接结果'
  assert marker not in text,'Do not append this section twice'
  lines=['',marker,'','|seed|P0 来源等权 Dice|B100 来源等权 Dice|配对差值|','|---|---:|---:|---:|']
  for row in seed_rows:lines.append(f"|{row['seed']}|{row['P0_source_equal_dice']:.6f}|{row['B100_source_equal_dice']:.6f}|{row['B100_minus_P0']:+.6f}|")
  lines+=['',f"配对来源等权 ΔDice = {result['source_equal_paired_delta_mean']:+.6f} ± {result['source_equal_paired_decoder_seed_sd']:.6f}（三个 decoder seeds 的样本 SD；不作为独立 acquisition 置信区间）。",'', '|来源|配对 ΔDice 均值|decoder seed SD|正向 seed 数|','|---|---:|---:|---:|']
  for row in paired_sources:lines.append(f"|{row['source_id']}|{row['paired_dice_delta_mean']:+.6f}|{row['paired_decoder_seed_sd']:.6f}|{row['positive_decoder_seeds']}/3|")
  lines+=['','上述228条指标覆盖、混淆矩阵算术、来源汇总、六组40k完成回执、18份checkpoint sidecar和三个seed的初始化/采样匹配已交叉核验。详见MV3_final_results_verification.json。输入尺度未被操纵，不能用本轮结果判定尺度的因果影响。Turner间距仍UNVERIFIED，对其物理距离值保持该限制。','']
  report.write_text(text+'\n'.join(lines),encoding='utf-8',newline='\n')
 print(json.dumps(result))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--first-seed',action='store_true');parser.add_argument('--augment-report',action='store_true');args=parser.parse_args();verify(args.first_seed,args.augment_report)
