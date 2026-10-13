set -u
base=/opt/data/3_117/tanghr
round6="$base/codex/mitoverse_round6_20260917_044606"
python3 - "$round6" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
for name in ('stage_status.json','coordinator_pid.json','allowed_gpu_processes.json'):
 f=p/name
 print(name, f.exists())
 if f.exists():
  try: print(json.dumps(json.loads(f.read_text()),ensure_ascii=False)[:5000])
  except Exception as e: print(type(e).__name__,str(e))
PY
ps -eo pid,ppid,user,etimes,rss,args | grep -E 'mitoverse_round6|DINOv3|Med_DINOv3|mira_superhuman_demo' | grep -v grep | cut -c 1-220 | head -n 35
ps -p 1971241 -o pid,ppid,user,etime,rss,args --no-headers
curl --silent --show-error --output /dev/null --write-out 'site_http=%{http_code} seconds=%{time_total}\n' --max-time 5 http://127.0.0.1:7860/
for d in "$base/DINOv3" "$base/Med_DINOv3"; do
  if [ -d "$d/.git" ]; then
    printf '%s commit=' "$d"
    git -C "$d" rev-parse HEAD
    printf '%s dirty_count=' "$d"
    git -C "$d" status --porcelain | wc -l
  else
    printf '%s no_git\n' "$d"
  fi
done
find "$base/codex" -maxdepth 1 -type d -name 'mito_pretrain_v2_*' -printf '%f\n' 2>/dev/null | sort
find "$base/DINOv3" "$base/Med_DINOv3" -maxdepth 4 -type f \( -iname '*vitl*pth' -o -iname '*vitl*pt' -o -iname '*vit_l*pth' -o -iname '*vit_l*pt' \) -printf '%p %s\n' 2>/dev/null | head -n 30
test -d "$base/MitoVerse_data/data" && printf 'mitoverse_data_dir=present\n' || printf 'mitoverse_data_dir=missing\n'
test -x "$base/envs/meddinov3/bin/python" && printf 'med_env=present\n' || printf 'med_env=missing\n'
