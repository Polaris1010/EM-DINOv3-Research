set -euo pipefail
base=/opt/data/3_117/tanghr
run="$base/codex/mito_pretrain_v2_20260926_060227"
test -x "$base/dinov3_shared/bin/python"
"$base/dinov3_shared/bin/python" -m py_compile "$run/src/train_ssl.py" "$run/src/ssl_data.py" "$run/src/last_selection.py"
"$base/envs/meddinov3/bin/python" -m py_compile "$run/src/coordinator_smoke.py" "$run/src/guard.py" "$run/src/worker.py"
test "$(curl --silent --output /dev/null --write-out '%{http_code}' --max-time 5 http://127.0.0.1:7860/)" = 200
cd "$run/src"
"$base/envs/meddinov3/bin/python" -u coordinator_smoke.py --arm L --gpu 3 --batch 16
