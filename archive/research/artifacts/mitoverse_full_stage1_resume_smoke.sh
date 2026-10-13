set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
job="$run/runs/B_full/resume_smoke"
test ! -e "$job"
test "$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:7860/)" = 200
gpu3_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -d ' ')
test "$gpu3_used" -lt 100
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-5a68cc29-a13a-bf90-858e-079e14de807a
export PYTHONPATH=/opt/data/3_117/tanghr/codex/mitotopo_20260908_round2/deps:/opt/data/3_117/tanghr/DINOv3/dinov3
python=/opt/data/3_117/tanghr/dinov3_shared/bin/python
trainer="$run/scripts/train_B_full.py"
manifest="$run/data/pretrain_all_mitoverse.json"
config="$run/configs/B_full.yaml"
weight=/opt/data/3_117/tanghr/DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
"$python" "$trainer" --manifest "$manifest" --config "$config" --weight "$weight" --output "$job" --steps 10 --seed 0
checkpoint="$job/checkpoints/step000000010/full.pth"
test -s "$checkpoint"
"$python" "$trainer" --manifest "$manifest" --config "$config" --weight "$weight" --output "$job" --steps 20 --seed 0 --resume "$checkpoint"
"$python" - "$job" "$run/runs/B_full/smoke20" <<'PY'
import json,sys
from pathlib import Path
resume, continuous = map(Path,sys.argv[1:])
a=json.loads((resume/'progress.json').read_text())
b=json.loads((continuous/'progress.json').read_text())
result={'status':'pass' if a['sample_chain']==b['sample_chain'] else 'fail',
        'resume_step':a['step'],'continuous_step':b['step'],
        'sample_chain_equal':a['sample_chain']==b['sample_chain'],
        'resume_chain':a['sample_chain'],'continuous_chain':b['sample_chain']}
(resume/'resume_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
if result['status']!='pass':raise SystemExit(2)
PY
