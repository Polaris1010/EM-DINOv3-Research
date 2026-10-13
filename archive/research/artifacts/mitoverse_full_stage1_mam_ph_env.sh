set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
/opt/data/3_117/tanghr/envs/microsam/bin/python - <<'PY'
import sys,cv2,scipy,skimage
print(sys.version)
print('cv2',cv2.__version__)
PY
