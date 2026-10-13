set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
cd "$run/src"
py=/opt/data/3_117/tanghr/envs/meddinov3/bin/python
"$py" coordinator_smoke.py --arm B --gpu 3 --batch 16 --steps 5
"$py" coordinator_smoke.py --arm B --gpu 3 --batch 16 --steps 10 \
  --resume "$run/src/jobs/B_seed0_batch16_smoke5/checkpoint_step5.pth"
"$py" verify_ssl_resume.py \
  --continuous "$run/src/jobs/B_seed0_batch16_smoke10/checkpoint_step10.pth" \
  --resumed "$run/src/jobs/B_seed0_batch16_resume10/checkpoint_step10.pth" \
  | tee "$run/evidence/B_resume_exact.json"
