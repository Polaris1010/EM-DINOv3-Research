set -euo pipefail
base=/opt/data/3_117/tanghr/codex
stage1="$base/mitoverse_full_stage1_20260928_163120"
stamp=$(date -u +%Y%m%d_%H%M%S)
run="$base/mitoverse_full_stage2_$stamp"
test ! -e "$run"
mkdir -p "$run"/{stage1_reference,audit,topology_calibration/{sam_vit_b,sam_vit_h,equivariance,offline_prior,crop_score,cache_benchmark,throughput,figures},topology_cache/{metadata,mam},configs,src/{topology_prior,topology_crop,tests},runs/T1,reports}
printf '%s\n' '3fd614082bf584c21e212884856875e328104ea09a293a95510c0f5f3117edb9' > "$run/stage1_reference/manifest.sha256"
printf '%s\n' '12bbf09ffcd7f93cbad88c4e55df55d36b3a299990ad976bf29af51a118a265b' > "$run/stage1_reference/B_full_checkpoint.sha256"
cp "$stage1/configs/B_full_resolved.yaml" "$run/stage1_reference/B_full_resolved_config.yaml"
cp "$stage1/topology_feasibility/sample_manifest.json" "$run/topology_calibration/sample_manifest.json"
cat > "$run/stage1_reference/stage1_paths.json" <<EOF
{
  "stage1_root": "$stage1",
  "manifest": "$stage1/data/pretrain_all_mitoverse.json",
  "B_full_checkpoint": "$stage1/runs/B_full/formal_attempt1/checkpoints/step000100000/full.pth",
  "official_weight": "/opt/data/3_117/tanghr/DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
  "dinov3_commit": "91ffb6803c74aeb6927209b82560142743763dd7"
}
EOF
cat > "$run/README.md" <<EOF
# Full MitoVerse Stage 2A

New isolated run created $stamp UTC. Stage 1 is read-only. Stage 2A calibrates
SAM priors, geometric equivariance, local crop topology scores, offline cache,
fairness and throughput. Stage 2B T1 may start only after every hard gate passes.
T1 must initialize from official natural DINOv3 weights, never B_full.
EOF
cat > "$run/STATUS.json" <<EOF
{"stage":"stage2A_preparation","stage2A_gate":"pending","auto_launch_T1":false,"T1":"not_started","stage1_immutable":true,"created_utc":"$stamp"}
EOF
date -Is > "$run/audit/preflight.txt"
hostname >> "$run/audit/preflight.txt"
uptime >> "$run/audit/preflight.txt"
free -h >> "$run/audit/preflight.txt"
df -h "$run" > "$run/audit/disk_state.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader > "$run/audit/gpu_state.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader >> "$run/audit/gpu_state.csv"
git -C /opt/data/3_117/tanghr/DINOv3 rev-parse HEAD > "$run/audit/source_state.txt"
git -C /opt/data/3_117/tanghr/DINOv3 status --short >> "$run/audit/source_state.txt"
curl -fsS -o /dev/null http://127.0.0.1:7860/
git -C "$run" init -q -b codex/stage2
cat > "$run/.gitignore" <<'EOF'
*.pth
*.pt
*.npz
*.npy
*.log
*.png
*.tmp
topology_cache/mam/
EOF
RUN="$run" /opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json,os
from pathlib import Path
run=Path(os.environ['RUN'])
s=json.loads((run/'topology_calibration/sample_manifest.json').read_text())
m=json.loads((Path(json.loads((run/'stage1_reference/stage1_paths.json').read_text())['manifest'])).read_text())
print('stage2_root',run)
print('sample_count',len(s['samples']),'full_volumes',len(m['volumes']))
print('full_first_keys',list(m['volumes'][0]))
print('sample_first',s['samples'][0])
PY
printf 'STAGE2_ROOT=%s\n' "$run"
