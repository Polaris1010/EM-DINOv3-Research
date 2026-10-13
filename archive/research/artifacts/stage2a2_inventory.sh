set -e
BASE=/opt/data/3_117/tanghr
date -u +%FT%TZ
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
free -h | head -2
df -h "$BASE" | tail -1
printf 'website_http=%s\n' "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:7860/)"
printf 'stage2_status='; head -1 "$BASE/codex/mitoverse_full_stage2_20260930_082816/STATUS"
printf 'stage1_checkpoint='; find "$BASE/codex/mitoverse_full_stage1_20260928_163120" -maxdepth 4 -type f \( -name '*.pth' -o -name '*.pt' \) -printf '%p %s bytes\n' | head -20
printf 'top_level='; find "$BASE" -maxdepth 2 -type d -printf '%p\n' | head -80
printf 'labeled_candidates='; find "$BASE" -maxdepth 5 -type d \( -iname '*CMCC*' -o -iname '*CMCT*' -o -iname '*MitoEM*' -o -iname '*Lucchi*' -o -iname '*eval*' \) -printf '%p\n' | head -100
