set -u
base=/opt/data/3_117/tanghr
run="$base/codex/mito_pretrain_v2_20260926_060227"
git ls-remote https://github.com/ChengShiest/LAST-ViT.git HEAD 2>&1 | head -n 2
git ls-remote https://github.com/gm3g11/TopoCL.git HEAD 2>&1 | head -n 2
grep -Ei '^(gudhi|ripser|giotto|scikit-tda|persim|opencv|scipy|zarr|torch|torchvision)==' "$run/evidence/pip_freeze.txt" | head -n 20
find "$base/codex" -maxdepth 3 -type d \( -name 'gudhi' -o -name 'persim' \) -printf '%p\n' 2>/dev/null | head -n 15
