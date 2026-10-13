set -euo pipefail
OLD=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/meddinov3/bin/python - "$OLD" <<'PY'
import sys,json,hashlib,time
from pathlib import Path
import torch
p=Path(sys.argv[1]);c=p/'runs/B_full/formal_attempt1/checkpoints/step000100000/full.pth'
print('CHECKPOINT_BYTES',c.stat().st_size,flush=True)
k=torch.load(c,map_location='cpu',mmap=True,weights_only=False)
print('IDENTITY',json.dumps(k['identity']),flush=True)
print('STATE',json.dumps({'step':k['step'],'keys':list(k),'student_keys':len(k['student']),'teacher_keys':len(k['teacher']),'optimizer_states':len(k['optimizer']['state']),'rng_ranks':len(k['rng_by_rank']),'optimizer_lr_groups':sorted(set(g['lr'] for g in k['optimizer']['param_groups'])),'sample_chain':k['sample_chain']}),flush=True)
print('TRAINER_SOURCE_SHA',hashlib.sha256((p/'scripts/train_B_full.py').read_bytes()).hexdigest())
print('SAMPLER_SOURCE_SHA',hashlib.sha256((p/'scripts/ssl_full_data.py').read_bytes()).hexdigest())
print('GUARD_SOURCE_BEGIN',flush=True)
print((p/'scripts/gpu_guard.py').read_text(),flush=True)
PY
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
free -h
df -h "$OLD"
curl -s -o /dev/null -w 'website_http=%{http_code}\n' --max-time 8 http://127.0.0.1:7860/
systemctl list-units --type=service --state=running --no-pager | grep -E 'mitoverse|mito|stage|dino' || true
