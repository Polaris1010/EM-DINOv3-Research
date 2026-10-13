set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
top="$run/topology_feasibility"
test "$(systemctl show mitoverse-stage1-raw-ph.service -p ActiveState --value)" = inactive
test "$(systemctl show mitoverse-stage1-raw-ph.service -p Result --value)" = success
test "$(/opt/data/3_117/tanghr/envs/meddinov3/bin/python -c 'import csv,sys;print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))' "$top/metrics.csv")" = 4240
/opt/data/3_117/tanghr/envs/meddinov3/bin/python "$run/scripts/topology_summarize.py"
cat "$top/report.md"
