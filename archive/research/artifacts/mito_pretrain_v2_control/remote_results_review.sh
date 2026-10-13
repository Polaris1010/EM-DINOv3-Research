set -euo pipefail
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json,pathlib,statistics,datetime,collections
r=pathlib.Path('/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227')
print('UTC',datetime.datetime.now(datetime.timezone.utc).isoformat())
for p in sorted((r/'src/jobs').glob('formal_seed0_*_part*')):
 print('JOB',p.name)
 if (p/'progress.json').exists():print('PROGRESS', (p/'progress.json').read_text())
 windows={k:[] for k in (1000,10000,30000,60000)}; tail=collections.deque(maxlen=1000);bad=0
 with (p/'steps.jsonl').open() as f:
  for line in f:
   try:x=json.loads(line)
   except ValueError:bad+=1;continue
   tail.append(x)
   for k in windows:
    if k-999<=x['step']<=k:windows[k].append(x)
 windows['latest']=list(tail)
 for k,v in windows.items():
  if not v:continue
  z={'window':k,'n':len(v),'first':v[0]['step'],'last':v[-1]['step']}
  for field in ('loss','grad_norm'):z[field+'_mean']=statistics.mean(x[field] for x in v)
  z['components']={a:statistics.mean(x['loss_components'][a] for x in v) for a in v[-1]['loss_components']}
  z['seconds_per_step']=(v[-1]['seconds']-v[0]['seconds'])/max(1,v[-1]['step']-v[0]['step'])
  z['feature_first']=v[0].get('feature_stat');z['feature_last']=v[-1].get('feature_stat')
  print('WINDOW',json.dumps(z))
 print('EXPORTS',[x.name for x in p.glob('ema_step*.json')], 'bad_lines',bad)
for name in ('probe_selection.json','finetune_selection.json','formal_seed0_admission.json'):
 p=r/'evidence'/name
 if p.exists():print('EVIDENCE',name,p.read_text())
print('DOWNSTREAM_RESULTS', [str(p.relative_to(r)) for p in (r/'src/jobs').glob('*/result.json') if p.parent.name.startswith(('probe_','finetune_','final_'))])
PY
