set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
PYTHONPATH=/opt/data/3_117/tanghr/DINOv3/dinov3 /opt/data/3_117/tanghr/envs/meddinov3/bin/python "$run/scripts/validate_ssl_data.py" \
  --dinov3-root /opt/data/3_117/tanghr/DINOv3/dinov3 \
  --manifest "$run/data/pretrain_all_mitoverse.json" \
  --output "$run/audit/no_label_access_check.json" \
  --samples 1000
