"""Independently verify completed comparison and derive paired changes."""
from pathlib import Path
import hashlib,json,csv,math,statistics
R=Path(__file__).resolve().parent
def load(name):return json.loads((R/name).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 result=load('results.json');status=load('STATUS.json');smoke=load('audit/smoke.json');integrity=load('audit/protected_integrity.json')
 assert result['status']==status['status']=='complete'
 assert smoke['status']==integrity['status']=='PASS'
 assert integrity['before']==integrity['after'] and status['website_http_after']==200
 assert result['decoder_sha256']==sha(R/'src/official_primus.py')==load('build_audit.json')['official_primus_sha256']
 assert result['steps']==3000 and result['seed']==20260907 and result['nsd_threshold']==1
 assert len(result['summary'])==6 and len(result['per_volume'])==15
 old=Path(__file__).resolve().parents[1]/'representation_diagnosis_live_20261013/evidence/final_results.json'
 prior=json.loads(old.read_text(encoding='utf-8'));mapping={'P0':'official_natural','B100k':'B_full100k','B150k':'B_full150k'}
 decoder_deltas=[];encoder_deltas=[];paired=[];initial=set()
 for summary in result['summary']:
  ds,model=summary['dataset'],summary['model']
  metrics=load(f'runs/{model}_{ds}/metrics.json')
  assert metrics['decoder_variant']=='official_meddinov3_primus_multiscale' and metrics['frequency_gates'] is False and metrics['frozen_encoder']
  assert metrics['source_sha256']==sha(R/'src/frozen_probe_official.py') and metrics['decoder_source_sha256']==result['decoder_sha256']
  assert metrics['steps']==3000 and metrics['seed']==20260907
  initial.add(metrics['decoder_initial_sha256'])
  rows=[x for x in result['per_volume'] if x['dataset']==ds and x['model']==model]
  assert len(rows)==summary['n']==(3 if ds.endswith('CMCC') else 2)
  assert {x['case'] for x in rows}==set(metrics['datasets'][ds]['cases'])
  for metric in ['dice','hd95','nsd']:
   values=[x[metric] for x in rows]
   if all(math.isfinite(x) for x in values):
    assert math.isclose(statistics.mean(values),summary[metric+'_mean'],rel_tol=3e-6,abs_tol=1e-7)
    assert math.isclose(statistics.stdev(values),summary[metric+'_std'],rel_tol=3e-6,abs_tol=1e-7)
  previous=next(x for x in prior['summary'] if x['dataset']==ds and x['encoder_kind']==mapping[model])
  decoder_deltas.append({'dataset':ds,'model':model,**{m+'_official_minus_gated':summary[m+'_mean']-previous[m+'_mean'] for m in ['dice','hd95','nsd']}})
 assert initial=={result['decoder_initial_sha256']}=={x['head_initial_sha256'] for x in smoke['models']}
 for ds in ['Dataset111_CMCC','Dataset131_CMCT']:
  bymodel={x['model']:x for x in result['summary'] if x['dataset']==ds}
  for a,b in [('P0','B100k'),('B100k','B150k')]:
   encoder_deltas.append({'dataset':ds,'comparison':b+' minus '+a,**{m+'_delta':bymodel[b][m+'_mean']-bymodel[a][m+'_mean'] for m in ['dice','hd95','nsd']}})
   av={x['case']:x for x in result['per_volume'] if x['dataset']==ds and x['model']==a}
   bv={x['case']:x for x in result['per_volume'] if x['dataset']==ds and x['model']==b}
   assert av.keys()==bv.keys()
   for case in av:paired.append({'dataset':ds,'case':case,'comparison':b+' minus '+a,**{m+'_delta':bv[case][m]-av[case][m] for m in ['dice','hd95','nsd']}})
 for name,rows in [('encoder_mean_deltas.csv',encoder_deltas),('paired_encoder_deltas.csv',paired)]:
  with (R/name).open('w',newline='',encoding='utf-8-sig') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 text=(R/'REPORT_zh.md').read_text(encoding='utf-8')
 text+='\n## 同一原版 decoder 内的 encoder 均值差\n\n| Dataset | 对比 | ΔDice | ΔHD95 | ΔNSD |\n|---|---|---:|---:|---:|\n'
 for x in encoder_deltas:text+=f"| {x['dataset'].split('_')[-1]} | {x['comparison']} | {x['dice_delta']:+.6f} | {x['hd95_delta']:+.2f} | {x['nsd_delta']:+.6f} |\n"
 text+='\n上述差值已按逐体积结果独立复算；这些是单种子开发验证结果，不能作显著性或最终独立泛化结论。\n'
 (R/'REPORT_verified_zh.md').write_text(text,encoding='utf-8')
 (R/'verification_results.json').write_text(json.dumps({'status':'PASS','groups':6,'validation_volume_results':15,'shared_official_head_initial_sha256':result['decoder_initial_sha256'],
 'protected_inputs_unchanged':True,'website_http_after':200,'encoder_mean_deltas':encoder_deltas,'decoder_mean_deltas':decoder_deltas},indent=2),encoding='utf-8')
 print(json.dumps({'status':'PASS','summary':result['summary'],'encoder_deltas':encoder_deltas,'decoder_deltas':decoder_deltas},indent=2))
if __name__=='__main__':main()
