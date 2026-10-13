set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
git -C "$run" add scripts/train_B_full.py scripts/gpu_guard.py scripts/admit_B_full.py
git -C "$run" -c user.name=Codex -c user.email=codex@local commit -m 'Use NFS-safe guard heartbeat for B_full'
git -C "$run" rev-parse HEAD
git -C "$run" status --short
