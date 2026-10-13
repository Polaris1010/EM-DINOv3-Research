set -euo pipefail
base=/opt/data/3_117/tanghr
run="$base/codex/mito_pretrain_v2_20260926_060227"
"$base/envs/meddinov3/bin/python" "$run/src/prepare_data.py" \
  --source-manifest "$base/codex/mitoverse_round6_20260917_044606/data_manifest_v3.csv" \
  --output-dir "$run/data"
sha256sum "$run/src/prepare_data.py"
find "$run/data" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
