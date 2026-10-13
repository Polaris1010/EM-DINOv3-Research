set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
/opt/data/3_117/tanghr/envs/meddinov3/bin/python "$run/scripts/validate_reader.py" --manifest "$run/data/pretrain_all_mitoverse.json" --output "$run/audit/reader_validation.json"
