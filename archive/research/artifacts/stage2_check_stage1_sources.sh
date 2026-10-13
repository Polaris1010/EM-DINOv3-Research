set -e
ROOT=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
find "$ROOT" -maxdepth 4 -type f \( -name 'mam_sam_vit_b_smoke.py' -o -name 'mam_utils.py' \) -print
ls "$ROOT/topology_feasibility/mam_sam_vit_b" | head
