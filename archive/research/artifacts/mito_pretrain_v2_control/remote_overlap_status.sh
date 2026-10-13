set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
systemctl show mito-pretrain-v2-seed0 mito-pretrain-v2-watcher -p ActiveState -p MainPID -p Result --no-pager
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - "$run" <<'PY'
import pathlib,json,sys
s=pathlib.Path(sys.argv[1])/'src'
for name in ('overlap_status.json','STOP.json','expanded_status.json'):
 if (s/name).exists():print(name,(s/name).read_text())
for p in sorted((s/'jobs').glob('*')):
 if p.name.startswith(('probe_','formal_seed0_LT_part1','expanded_')):
  print('JOB',p.name)
  for name in ('progress.json','result.json'):
   if (p/name).exists():
    d=json.loads((p/name).read_text());print(name,{k:d.get(k) for k in ('status','step','completed_steps','loss','best_validation_dice','peak_gpu_bytes','completed_volumes','volume','z')})
  if (p/'train.log').exists():print('LOGTAIL','\n'.join((p/'train.log').read_text()[-2400:].splitlines()[-4:]))
PY
tail -n 10 "$run/src/overlap_coordinator.log"
