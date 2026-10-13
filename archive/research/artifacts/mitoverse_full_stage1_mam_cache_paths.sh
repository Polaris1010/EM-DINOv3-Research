set -euo pipefail
getent passwd 1015 || true
env | grep -E '^(MICROSAM_CACHEDIR|XDG_CACHE_HOME|HOME)=' || true
/opt/data/3_117/tanghr/envs/microsam/bin/python - <<'PY'
import micro_sam.util,pooch
print('micro_sam_cache',micro_sam.util.get_cache_directory())
print('pooch_cache',pooch.os_cache('micro_sam'))
PY
for dir in /home/tanghr/.cache/micro_sam /home/1015/.cache/micro_sam /opt/data/3_117/tanghr/.cache/micro_sam /root/.cache/micro_sam /opt/data/3_117/tanghr/microsam/cache; do if test -d "$dir"; then echo "DIR $dir"; ls -lah "$dir" | head -30; fi; done
sed -n '1,115p' /opt/data/3_117/tanghr/microsam/micro-sam/micro_sam/util.py
