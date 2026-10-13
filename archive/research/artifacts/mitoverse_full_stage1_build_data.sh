set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
root=/opt/data/3_117/tanghr/MitoVerse_data
legacy=/opt/data/3_117/tanghr/codex/mitoverse_round6_20260917_044606/data_manifest_v3.csv
/opt/data/3_117/tanghr/dinov3_shared/bin/python "$run/scripts/build_full_manifest.py" --root "$root" --legacy-csv "$legacy" --output "$run/data/pretrain_all_mitoverse.json"
/opt/data/3_117/tanghr/dinov3_shared/bin/python "$run/scripts/validate_reader.py" --manifest "$run/data/pretrain_all_mitoverse.json" --output "$run/audit/reader_validation.json"
