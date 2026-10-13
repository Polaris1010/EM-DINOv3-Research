set -euo pipefail
ROOT=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2_20260930_082816
cd "$ROOT"
EXPECTED=a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e
TRANSFER=topology_calibration/sam_vit_h/sam_vit_h_4b8939.pth.transfer
FINAL=topology_calibration/sam_vit_h/sam_vit_h_4b8939.pth
test "$(stat -c %s "$TRANSFER")" = 2564550879
ACTUAL=$(sha256sum "$TRANSFER" | cut -d ' ' -f 1)
test "$ACTUAL" = "$EXPECTED"
mv "$TRANSFER" "$FINAL"
printf 'weight_sha256=%s\n' "$ACTUAL"
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/microsam/bin/python src/sam_h_paired.py --limit 1
