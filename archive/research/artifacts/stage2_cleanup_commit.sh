set -euo pipefail
ROOT=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2_20260930_082816
cd "$ROOT"
test "$(cat STATUS | head -1)" = STAGE2A_GATE=FAIL
test -f topology_calibration/sam_vit_h/sam_vit_h_4b8939.pth
test "$(sha256sum topology_calibration/sam_vit_h/sam_vit_h_4b8939.pth | cut -d ' ' -f 1)" = a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e
for FILE in topology_calibration/sam_vit_h/sam_vit_h_4b8939.pth.uploading audit/transfer_probe_64m.bin; do
  TARGET=$(realpath -m "$ROOT/$FILE")
  case "$TARGET" in "$ROOT"/*) ;; *) exit 5 ;; esac
  if test -e "$TARGET"; then rm -f -- "$TARGET"; fi
done
cat >> .gitignore <<'EOF'
*.uploading
*.transfer
*.pid
__pycache__/
EOF
git config user.name Codex
git config user.email codex@local
git add -A
git diff --cached --stat | tail -15
git commit -m 'Record Stage 2A calibration failure before T1'
printf 'commit=%s\n' "$(git rev-parse HEAD)"
printf 'working_tree_status=%s\n' "$(git status --porcelain | wc -l)"
printf 'stage2_status=%s\n' "$(head -1 STATUS)"
printf 'T1_run_file_count=%s\n' "$(find runs/T1 -type f | wc -l)"
printf 'website_http=%s\n' "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:7860/)"
