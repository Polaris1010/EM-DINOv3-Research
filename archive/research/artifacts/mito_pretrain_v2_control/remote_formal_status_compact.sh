set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
systemctl show mito-pretrain-v2-seed0 -p ActiveState -p MainPID -p Result --no-pager
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - "$run" <<'PY'
import json,pathlib,sys,time
r=pathlib.Path(sys.argv[1]); src=r/'src'
print('stop_marker', (src/'STOP.json').exists())
for arm in 'B','L','T','LT':
    paths=sorted((src/'jobs').glob(f'formal_seed0_{arm}_part*/progress.json'))
    if paths:
        x=json.loads(paths[-1].read_text());print(arm,'step',x['step'],'target',x['target_steps'],'loss',x['loss'],'grad',x['grad_norm'],'seconds',round(x['seconds'],1),'source_exposure',sum(x['source_exposure'].values()))
    else:print(arm,'queued_or_initializing')
heartbeat=src/'coordinator.heartbeat'
if heartbeat.exists():print('heartbeat_age_seconds',round(time.monotonic()-json.loads(heartbeat.read_text())['monotonic'],1))
PY
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
nvidia-smi --query-gpu=index,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader,nounits
