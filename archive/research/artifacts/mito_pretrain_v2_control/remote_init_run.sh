set -euo pipefail
base=/opt/data/3_117/tanghr
codex="$base/codex"
site_status=$(curl --silent --output /dev/null --write-out '%{http_code}' --max-time 5 http://127.0.0.1:7860/)
test "$site_status" = 200
test -f "$codex/mitoverse_round6_20260917_044606/data_manifest_v3.csv"
stamp=$(date -u +%Y%m%d_%H%M%S)
run="$codex/mito_pretrain_v2_${stamp}"
mkdir -m 750 "$run"
printf '%s\n' "$run"
for d in "$base/DINOv3/dinov3" "$base/Med_DINOv3/MedDINOv3"; do
  if [ -d "$d/.git" ]; then
    printf '%s commit=' "$d"
    git -C "$d" rev-parse HEAD
    printf '%s dirty_count=' "$d"
    git -C "$d" status --porcelain | wc -l
  else
    printf '%s no_git\n' "$d"
  fi
done
