# MitoVerse full restart, seed 0

Server run: `/opt/data/3_117/tanghr/codex/mitoverse_full_restart_20260928_133515`

Service: `mitoverse-full-seed0.service`. Physical GPU1=B, GPU2=L, GPU3=T;
LT is queued on the first freed GPU. GPU0 remains unassigned. The old v2
watcher and all old training and evaluation services were inactive at launch.

The full image manifest is `data/pretrain_all_mitoverse.json`: 212 volumes from
13 sources, including all 157 OpenOrganelle volumes. SHA256:
`e25c88f86d79116d9eef21f733328ed0d5f4518c25ac1061f910684bb0c36fa0`.
`evidence/full_manifest_audit.json` lists every ID and confirms equality of the
catalog, Zarr directory set, and manifest paths. Historical split is retained
only in `legacy_split`. No label or mask is consulted for SSL image selection.

All four formal arms start from the official natural-image DINOv3 ViT-L/16
checkpoint at optimizer step 0. SHA256:
`8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`.
The independent protocol is `protocol_seed0.json`; the formal gate is
`evidence/formal_seed0_admission.json`. New topology scores and the locked
selection are under `topology_calibration/`. Old model checkpoints, heads,
optimizer state, scheduler state, and watcher queues are not inputs.

Read-only monitoring on the server:

```bash
run=/opt/data/3_117/tanghr/codex/mitoverse_full_restart_20260928_133515
systemctl show mitoverse-full-seed0.service -p ActiveState -p MainPID -p Result --no-pager
cat "$run/src/formal_seed0_status.json"
for arm in B L T LT; do
  for job in "$run/src/jobs/formal_seed0_${arm}_part"*; do
    test -d "$job" || continue
    printf '%s: ' "$job"
    if test -f "$job/progress.json"; then
      /opt/data/3_117/tanghr/envs/meddinov3/bin/python -c 'import json,sys; x=json.load(open(sys.argv[1])); print(x["step"],x["loss"],x["unique_volumes"])' "$job/progress.json"
    else
      echo initializing
    fi
  done
done
test ! -e "$run/src/STOP.json"
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7860/
nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv,noheader,nounits
```

`src/jobs/*/initialization.json` records official weight and step-0 evidence;
`progress.json` records actual source and volume exposures. Full checkpoints
are saved at least every 1,000 updates; EMA exports are due at 10k, 30k,
60k, and 100k. A failed Guard creates `src/STOP.json` and halts only its
registered workers. Investigate the resource or data error and the checkpoint
receipt before resuming. No old evaluations are automatically dispatched.

Supervised training and segmentation evaluation have not been started. They
must use all MitoVerse volumes if started; label and valid-region semantics
need a separate source-level audit first. Results on volumes included in
supervision are training-set results, not independent test performance.
