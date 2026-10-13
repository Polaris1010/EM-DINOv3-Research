set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
top="$run/topology_feasibility"
paired="$top/mam_sam_vit_b/paired"
test "$(systemctl show mitoverse-stage1-B-full-v2.service -p ActiveState --value)" = active
test "$(systemctl show mitoverse-stage1-mam-paired-smoke.service -p Result --value)" = success
test -f "$paired/maps/sample_000000.npz"
test ! -f "$paired/generation_complete.json"
test "$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:7860/)" = 200
systemd-run --unit=mitoverse-stage1-mam-paired-full.service \
  --description='CPU-only paired raw EM and OrgMIM SAM MAM PH feasibility on 424 slices' \
  --property=Restart=no --property=Nice=15 --property=CPUQuota=400% --property=MemoryMax=24G \
  --setenv=CUDA_VISIBLE_DEVICES= --setenv=OMP_NUM_THREADS=4 --setenv=OPENBLAS_NUM_THREADS=1 \
  /bin/bash -lc "set -euo pipefail; /opt/data/3_117/tanghr/envs/microsam/bin/python -u '$run/scripts/mam_generate_paired.py' > '$paired/generation.log' 2>&1; PYTHONPATH='$top/vendor:$run/scripts' /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u '$run/scripts/mam_paired_ph.py' > '$paired/ph.log' 2>&1; /opt/data/3_117/tanghr/envs/meddinov3/bin/python -u '$run/scripts/mam_paired_summary.py' > '$paired/summary.log' 2>&1"
sleep 5
systemctl show mitoverse-stage1-mam-paired-full.service -p ActiveState -p MainPID -p Result --no-pager
if test -f "$paired/generation_progress.json"; then cat "$paired/generation_progress.json"; fi
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
