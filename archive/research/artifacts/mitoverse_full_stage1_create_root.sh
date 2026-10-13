set -euo pipefail
base=/opt/data/3_117/tanghr/codex
run="$base/mitoverse_full_stage1_$(date -u +%Y%m%d_%H%M%S)"
test ! -e "$run"
mkdir -p "$run"/{audit,data,src,env,configs,scripts,runs/B_full,topology_feasibility/{maps,persistence,figures},reports}
printf '%s\n' "$run"
date -u > "$run/audit/preflight.txt"
hostname >> "$run/audit/preflight.txt"
uptime >> "$run/audit/preflight.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv > "$run/audit/gpu_state_before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$run/audit/process_state_before.txt"
systemctl show mitoverse-full-seed0.service mito-pretrain-v2-seed0.service mito-pretrain-v2-watcher.service -p Id -p ActiveState -p MainPID -p Result --no-pager > "$run/audit/service_state_before.txt"
df -h "$run" > "$run/audit/disk_state.txt"
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7860/ > "$run/audit/website_health.txt"
sha256sum /opt/data/3_117/tanghr/DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth > "$run/audit/official_weight_sha256.txt"
git -C /opt/data/3_117/tanghr/DINOv3/dinov3 rev-parse HEAD > "$run/src/dinov3_commit.txt"
git -C /opt/data/3_117/tanghr/DINOv3/dinov3 status --short > "$run/src/modified_files.txt"
/opt/data/3_117/tanghr/dinov3_shared/bin/python --version > "$run/env/python_version.txt" 2>&1
/opt/data/3_117/tanghr/dinov3_shared/bin/python -m pip freeze > "$run/env/pip_freeze.txt"
/opt/data/3_117/tanghr/dinov3_shared/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())' > "$run/env/torch_cuda.txt"
nvidia-smi > "$run/env/nvidia_smi.txt"
cat > "$run/README.md" <<'EOF'
# Full MitoVerse Stage 1

Fresh Stage 1 work area. The 2026-09-28 full-restart and older experiments are
preserved separately and are not initialization sources. Formal training must
not start until the Stage 1 admission gates all pass.
EOF
cat > "$run/STATUS.json" <<'EOF'
{"stage":"preparation","B_full":"not_started","topology_feasibility":"not_started","formal_admission":"pending"}
EOF
printf '%s\n' "$run"
