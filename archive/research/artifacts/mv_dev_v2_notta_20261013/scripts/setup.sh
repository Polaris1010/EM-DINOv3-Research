set -euo pipefail
TASK=/opt/data/3_117/tanghr/codex/mv_dev_v2_notta_20261013
mkdir -p "$TASK"/scripts "$TASK"/src "$TASK"/evidence "$TASK"/sources "$TASK"/logs "$TASK"/runs
systemctl is-active mv-dev-v2-validation-20261013.service || true
