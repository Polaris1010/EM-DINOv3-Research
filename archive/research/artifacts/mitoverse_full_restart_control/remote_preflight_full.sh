set -euo pipefail
base=/opt/data/3_117/tanghr
printf 'time_utc='; date -u +%FT%TZ
printf 'hostname='; hostname
printf 'services\n'
systemctl show mito-pretrain-v2-watcher.service mito-pretrain-v2-seed0.service mito-pretrain-v2-probes.service mito-pretrain-v2-finetune.service mito-pretrain-v2-independent.service mito-pretrain-v2-final.service -p Id -p ActiveState -p MainPID -p Result --no-pager
printf 'gpu\n'
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
printf 'gpu_processes\n'
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
printf 'training_processes\n'
ps -eo pid,lstart,args | grep -E 'train_ssl.py|coordinator_|watch_all_batches.py|expanded_eval.py|seg_eval.py' | grep -v grep || true
printf 'site_http='; curl -sS -o /dev/null -w '%{http_code}\n' --max-time 5 http://127.0.0.1:7860/
printf 'disk\n'; df -h "$base" "$base/MitoVerse_data"
printf 'python_packages\n'
"$base/dinov3_shared/bin/python" - <<'PY'
import importlib
for name in ['torch','zarr','numpy','omegaconf','gudhi','scipy','skimage','psutil']:
    try:
        m=importlib.import_module(name)
        print(name,getattr(m,'__version__','present'))
    except Exception as exc:
        print(name,type(exc).__name__,str(exc)[:160])
PY
printf 'data_inventory\n'
"$base/dinov3_shared/bin/python" - <<'PY'
import collections,json,pathlib
r=pathlib.Path('/opt/data/3_117/tanghr/MitoVerse_data')
c=json.loads((r/'catalog.json').read_text())
p=list((r/'data').rglob('*.zarr'))
print(json.dumps({'catalog_count':len(c),'disk_count':len(p),'source_counts':dict(collections.Counter(x['dataset_id'] for x in c.values())),'catalog_example':next(iter(c.items()))},default=str))
PY
printf 'old_run_marker\n'
test -f "$base/codex/mito_pretrain_v2_20260926_060227/src/SUPERSEDED_BY_FULL_MITOVERSE.json"
cat "$base/codex/mito_pretrain_v2_20260926_060227/src/SUPERSEDED_BY_FULL_MITOVERSE.json"
