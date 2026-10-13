set -e
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
date -u +%FT%TZ
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
printf 'website_http=%s\n' "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:7860/)"
printf 'directional_maps=%s\n' "$(find branch_S/directional_affinity/maps -maxdepth 1 -name 'sample_*.npz' | wc -l)"
tail -2 branch_S/directional_affinity/generate.log 2>/dev/null || true
cat branch_E/runs/B_full_CMCC_smoke/result.json 2>/dev/null || true
