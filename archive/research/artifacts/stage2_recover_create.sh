set -euo pipefail
base=/opt/data/3_117/tanghr/codex
find "$base" -maxdepth 1 -type d -name 'mitoverse_full_stage2_*' -printf '%p\n' | sort
ls -ld /opt/data/3_117/tanghr/DINOv3 /opt/data/3_117/tanghr/DINOv3/dinov3 2>/dev/null || true
for dir in /opt/data/3_117/tanghr/DINOv3 /opt/data/3_117/tanghr/DINOv3/dinov3 /opt/data/3_117/tanghr/DINOv3/dinov3/dinov3; do
 if test -d "$dir/.git"; then echo "GIT_ROOT=$dir"; git -C "$dir" rev-parse HEAD; fi
done
