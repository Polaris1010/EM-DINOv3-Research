set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227
cd "$run/src"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python coordinator_resume_batch.py
