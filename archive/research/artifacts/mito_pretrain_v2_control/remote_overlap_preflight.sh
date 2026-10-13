set -euo pipefail
nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
systemctl show mito-pretrain-v2-seed0 -p MainPID -p KillMode -p ExecStart --no-pager
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - "$run" <<'PY'
import json,sys,pathlib
r=pathlib.Path(sys.argv[1]);s=r/'src'
for name in ('formal_seed0_status.json','process_registry.json','probes_status.json','STOP.json'):
 p=s/name
 if p.exists():print(name,p.read_text())
for p in sorted((s/'jobs').glob('formal_seed0_LT_part*')):
 for name in ('progress.json','latest_checkpoint.json'):
  if (p/name).exists():print(p.name,name,(p/name).read_text())
print('probe_jobs',[p.name for p in (s/'jobs').glob('probe*')])
PY
