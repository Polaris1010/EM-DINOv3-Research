set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
mam="$run/topology_feasibility/mam_sam_vit_b"
cat > "$mam/README.md" <<'EOF'
# CPU-only OrgMIM-style MAM feasibility

The frozen visual model is Meta's original SAM ViT-B checkpoint
`sam_vit_b_01ec64.pth` from
`https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`.
SHA256: `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`.
This is a ViT-B feasibility test, not the OrgMIM README's ViT-H example.

`mam_utils.py` is from the official OrgMIM repository:
`https://github.com/yanchaoz/OrgMIM/blob/main/preparation/mam_utils.py`.
The image-only manifest contains 424 slices, two from each of 212 volumes.
No GT masks, label-selected crops, or segmentation fine-tuning are used.
Images are center-cropped to at most 512 square pixels, normalized at the
1st/99th intensity percentiles, and replicated to RGB. Frozen SAM returns a
256x64x64 embedding. OrgMIM's `embeddings_to_affinities` uses delta_v=.5 and
delta_d=1.5. The minimum of vertical/horizontal affinities becomes the MAM.

The single-slice CPU test passed: MAM standard deviation .289, 5.46 seconds
for predictor and affinity work. The paired 10-augmentation test passed in
47.77 seconds. The full background service is
`mitoverse-stage1-mam-paired-full.service` with CPUQuota=400%, MemoryMax=24G,
Nice=15, and empty CUDA_VISIBLE_DEVICES. It generates paired raw/MAM maps,
then H0/H1 distributions and figures using a separate GUDHI environment.
Inspect `paired/generation_progress.json` while running. Completion markers:
`paired/generation_complete.json`, `paired/ph_complete.json`, and
`paired/report.md`.

The single-slice PH check found a nonzero MAM distance under flip, reflecting
that SAM features are not strictly flip equivariant. The full distribution
must be checked before choosing any TopoAug threshold.
EOF
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json
from pathlib import Path
run=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120')
p=run/'STATUS.json'
d=json.loads(p.read_text())
d['topology_feasibility']='paired_sam_vit_b_mam_running_cpu'
d['topology_mam_service']='mitoverse-stage1-mam-paired-full.service'
d['topology_mam_weight_sha256']='ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912'
t=p.with_suffix('.json.tmp')
t.write_text(json.dumps(d,indent=2))
t.replace(p)
s=run/'reports/stage1_summary.md'
text=s.read_text()
text += '\n## Topology MAM update\n\nAn isolated CPU-only Meta SAM ViT-B + official OrgMIM affinity feasibility job is now running on all 424 label-free slices. It does not use any training GPU. The paired raw-vs-MAM H0/H1 analysis and final threshold decision are pending its completion. See `topology_feasibility/mam_sam_vit_b/README.md`.\n'
s.write_text(text)
PY
git -C "$run" add scripts/mam_sam_vit_b_smoke.py scripts/mam_generate_paired.py scripts/mam_paired_ph.py scripts/mam_paired_summary.py topology_feasibility/mam_sam_vit_b/mam_utils.py topology_feasibility/mam_sam_vit_b/README.md reports/stage1_summary.md
git -C "$run" -c user.name=Codex -c user.email=codex@local commit -m 'Start isolated CPU SAM MAM feasibility on full sample set'
git -C "$run" rev-parse HEAD
