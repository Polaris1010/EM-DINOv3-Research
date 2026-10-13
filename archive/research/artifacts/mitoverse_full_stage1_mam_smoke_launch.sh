set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
mam="$run/topology_feasibility/mam_sam_vit_b"
test -f "$mam/weights/sam_vit_b_01ec64.pth"
test ! -f "$mam/smoke_result.json"
test "$(systemctl show mitoverse-stage1-B-full-v2.service -p ActiveState --value)" = active
systemd-run --unit=mitoverse-stage1-mam-smoke.service \
  --description='CPU-only frozen SAM ViT-B MAM single-slice feasibility' \
  --property=Restart=no --property=Nice=15 --property=CPUQuota=400% --property=MemoryMax=24G \
  --setenv=CUDA_VISIBLE_DEVICES= --setenv=OMP_NUM_THREADS=4 --setenv=OPENBLAS_NUM_THREADS=1 \
  /bin/bash -lc "/opt/data/3_117/tanghr/envs/microsam/bin/python -u '$run/scripts/mam_sam_vit_b_smoke.py' > '$mam/smoke.log' 2>&1"
sleep 3
systemctl show mitoverse-stage1-mam-smoke.service -p ActiveState -p MainPID -p Result --no-pager
stat -c 'weight_bytes=%s' "$mam/weights/sam_vit_b_01ec64.pth"
