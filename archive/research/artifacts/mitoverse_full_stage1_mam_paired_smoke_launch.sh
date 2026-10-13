set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
paired="$run/topology_feasibility/mam_sam_vit_b/paired"
mkdir -p "$paired"
test ! -f "$paired/maps/sample_000000.npz"
systemd-run --unit=mitoverse-stage1-mam-paired-smoke.service \
  --description='CPU-only paired MAM augmentation smoke on one slice' \
  --property=Restart=no --property=Nice=15 --property=CPUQuota=400% --property=MemoryMax=24G \
  --setenv=CUDA_VISIBLE_DEVICES= --setenv=OMP_NUM_THREADS=4 --setenv=OPENBLAS_NUM_THREADS=1 \
  /bin/bash -lc "/opt/data/3_117/tanghr/envs/microsam/bin/python -u '$run/scripts/mam_generate_paired.py' --limit 1 > '$paired/paired_smoke.log' 2>&1"
sleep 3
systemctl show mitoverse-stage1-mam-paired-smoke.service -p ActiveState -p MainPID -p Result --no-pager
