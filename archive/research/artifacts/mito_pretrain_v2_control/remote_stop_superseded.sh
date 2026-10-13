set -euo pipefail
systemctl stop mito-pretrain-v2-watcher.service
for unit in mito-pretrain-v2-seed0.service mito-pretrain-v2-probes.service mito-pretrain-v2-finetune.service mito-pretrain-v2-independent.service mito-pretrain-v2-final.service; do
 if systemctl is-active --quiet "$unit"; then systemctl stop "$unit"; fi
done
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import pathlib,json,time
s=pathlib.Path('/opt/data/3_117/tanghr/codex/mito_pretrain_v2_20260926_060227/src')
p=s/'SUPERSEDED_BY_FULL_MITOVERSE.json'
p.write_text(json.dumps({'time':time.time(),'status':'cancelled_by_user','reason':'Restart all pretraining from official natural weights using entire MitoVerse; cancel LT, probes, expanded evaluation and dependent batches.'},indent=2))
print(p)
PY
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
systemctl show mito-pretrain-v2-seed0.service mito-pretrain-v2-watcher.service -p ActiveState -p MainPID --no-pager
