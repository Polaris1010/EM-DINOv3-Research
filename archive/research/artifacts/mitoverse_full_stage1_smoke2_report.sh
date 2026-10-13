set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
job="$run/runs/B_full/smoke2"
cat "$job/result.json"
cat "$job/progress.json"
cat "$job/latest_checkpoint.json"
tail -n 2 "$job/steps_rank0.jsonl"
du -h "$job/checkpoints/step000000002/full.pth"
test "$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:7860/)" = 200
