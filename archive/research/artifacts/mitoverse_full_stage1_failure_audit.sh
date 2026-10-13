set -u
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
date -u
systemctl show mitoverse-stage1-B-full.service -p ActiveState -p MainPID -p Result -p ExecMainStatus --no-pager
cat "$run/STATUS.json"
if test -f "$run/runs/B_full/formal/STOP_REASON.json"; then cat "$run/runs/B_full/formal/STOP_REASON.json"; fi
if test -f "$run/runs/B_full/formal/STOP"; then cat "$run/runs/B_full/formal/STOP"; fi
tail -n 5 "$run/audit/gpu_guard.jsonl" 2>/dev/null | cut -c 1-1800
tail -n 100 "$run/runs/B_full/formal_train.log" 2>/dev/null
journalctl -u mitoverse-stage1-B-full.service --no-pager -n 70
ps -eo pid,ppid,etimes,args | grep -E 'gpu_guard.py|train_B_full.py|torch.distributed.run' | grep -v grep
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
