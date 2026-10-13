set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json,subprocess
from pathlib import Path
run=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120')
p=json.loads((run/'runs/B_full/formal_attempt1/progress.json').read_text())
init=json.loads((run/'runs/B_full/formal_attempt1/initialization.json').read_text())
top=json.loads((run/'topology_feasibility/summary.json').read_text())
service=subprocess.check_output(['systemctl','show','mitoverse-stage1-B-full-v2.service','-p','ActiveState','--value'],text=True).strip()
assert service=='active',service
checkpoints=sorted((run/'runs/B_full/formal_attempt1/checkpoints').glob('step*'))
checkpoint_text=checkpoints[-1].name if checkpoints else 'none yet'
b=f'''# B_full training report (interim)

Status: **running**, {p['step']:,} / 100,000 optimizer steps at this snapshot. This is not a completed 100k result. The active service is `mitoverse-stage1-B-full-v2.service`; output is `runs/B_full/formal_attempt1/` and latest checkpoint is `{checkpoint_text}`. The first formal attempt failed at step 0 after a false stale-heartbeat alarm caused by NFS file timestamp skew. Its logs remain under `runs/B_full/formal/`; attempt1 restarted from the official weights after the heartbeat check was corrected and retested.

The formal admission passed 869 gates. Strict official natural ViT-L/16 initialization matched all 368 backbone keys, and official inference versus the training student's CLS and patch tokens had maximum absolute difference 0. The 200-step three-GPU smoke passed, including identical tracked parameters across ranks. A separate guard-connected two-step smoke passed after the heartbeat fix.

The run uses physical GPUs 1, 2, 3; GPU 0 is forbidden. Global batch is 48; the objective is vanilla DINO 1 + iBOT 1 + KoLeo 0.1, Gram disabled, with no LAST/topology/segmentation branch. The run uses manifest SHA256 `3fd614082bf584c21e212884856875e328104ea09a293a95510c0f5f3117edb9` and official weight SHA256 `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`. Configuration and source hashes are in `initialization.json` and `audit/formal_B_full_admission.json`.

Latest loss {p['loss']:.4f}, gradient norm {p['grad_norm']:.4f}, iteration {p['iteration_seconds']:.3f} s. These optimization signals do not establish representation quality; downstream evaluation is outside Stage 1.
'''
(run/'reports/B_training_report.md').write_text(b)
s=f'''# Full-MitoVerse Stage 1 interim summary

Stage 1 is **in progress**. Fresh `B_full` has reached optimizer step {p['step']:,} / 100,000 on GPUs 1–3; the service is active and the website health guard is running. The existing B/L/T restart was stopped with its old checkpoints preserved; no old checkpoint initialized this run.

The full SSL manifest has 212 volumes from 13 sources and 48,722 legal Z slices. Catalog and on-disk Zarr path sets matched. All 636 first/middle/last image slice reads passed. The label-free uniform slice sampler passed 1,000 augmented samples. Official DINOv3 initialization, single-GPU smoke, distributed smoke, and same-run resume passed. Formal admission passed 869 gates.

Topology: 424 image-only samples from all 212 volumes completed raw EM PH analysis across 10 augmentation types, yielding {top['rows']:,} rows and source-stratified figures. GUDHI was installed in the isolated topology vendor directory. The OrgMIM MAM prior was unavailable, so raw-vs-MAM comparison and a defensible Stage 2 TopoAug threshold remain pending. Raw PH is exploratory and cannot stand in for a membrane prior. Details: `topology_feasibility/report.md`.

The 100k checkpoint, final audit, representation comparison, and complete Stage 1 acceptance are pending. Current checkpoint: `{checkpoint_text}`.
'''
(run/'reports/stage1_summary.md').write_text(s)
(run/'audit/final_stage1_audit.md').write_text(f'# Final Stage 1 audit pending\n\nB_full is at step {p["step"]:,}/100,000 at this snapshot. Do not treat this interim report as final acceptance. MAM comparison is pending.\n')
print('step',p['step'],'checkpoint',checkpoint_text,'service',service)
PY
git -C "$run" add scripts/topology_raw_ph.py scripts/topology_summarize.py reports/B_training_report.md reports/stage1_summary.md audit/final_stage1_audit.md topology_feasibility/report.md topology_feasibility/summary.json topology_feasibility/metrics.csv
git -C "$run" -c user.name=Codex -c user.email=codex@local commit -m 'Record raw PH feasibility and running B_full status'
git -C "$run" rev-parse HEAD
