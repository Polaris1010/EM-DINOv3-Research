set -euo pipefail
RUN=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
cd "$RUN"
grep -qxF '*.zip' .gitignore || printf '%s\n' '*.zip' >> .gitignore
grep -qxF 'branch_S/blind_review/answer_key_CONCEALED.csv' .gitignore || printf '%s\n' 'branch_S/blind_review/answer_key_CONCEALED.csv' >> .gitignore
test "$(find branch_S/directional_affinity/maps -maxdepth 1 -name 'sample_*.npz' | wc -l)" = 424
test "$(find branch_S/blind_review/panels -maxdepth 1 -name 'P*.png' | wc -l)" = 100
test -f branch_E/runs/B_full_Dataset111_CMCC/result.json
test -f branch_E/runs/B_full_Dataset131_CMCT/result.json
git add .
git -c user.name='Codex' -c user.email='codex@local' commit -m 'Complete Stage 2A2 S pilot and frozen B_full evaluation' >/dev/null
printf 'commit=%s\n' "$(git rev-parse --short HEAD)"
printf 'git_status='; test -z "$(git status --porcelain)" && echo clean || git status --short
printf 'website_http=%s\n' "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:7860/)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
