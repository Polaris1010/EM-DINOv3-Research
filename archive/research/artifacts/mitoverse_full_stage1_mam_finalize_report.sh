set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
paired="$run/topology_feasibility/mam_sam_vit_b/paired"
test "$(systemctl show mitoverse-stage1-mam-paired-full.service -p Result --value)" = success
test -f "$paired/ph_complete.json"
test -f "$paired/summary.json"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json
from pathlib import Path
run=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120')
paired=run/'topology_feasibility/mam_sam_vit_b/paired'
summary=json.loads((paired/'summary.json').read_text())
assert summary['rows']==12720 and summary['samples']==424 and summary['sources']==13
raw=summary['modes']['raw_em_paired']; mam=summary['modes']['mam_affinity']
def p50(mode,aug): return mode[aug]['d_equal']['p50']
finding=f'''\n## Verified finding and Stage 2 decision\n\nThe paired 424-slice comparison is complete. In the exploratory equal-weight H0+H1 distance, raw EM median horizontal-flip distance is {p50(raw,'flip_horizontal'):.3f}, while SAM MAM is {p50(mam,'flip_horizontal'):.3f}. MAM's aggressive-crop median is {p50(mam,'aggressive_crop'):.3f}, and strong-blur median is {p50(mam,'strong_blur'):.3f}. Aggressive crop exceeds mild crop in {100*mam['pairwise_fraction']['aggressive_crop_gt_mild_crop']:.1f}% of paired slices; strong blur exceeds weak blur in {100*mam['pairwise_fraction']['strong_blur_gt_weak_blur']:.1f}%. This shows a useful strength ordering, but a large MAM flip response. Source median MAM flip distance ranges from {min(x['flip_horizontal'] for x in summary['source_medians']['mam_affinity'].values()):.3f} to {max(x['flip_horizontal'] for x in summary['source_medians']['mam_affinity'].values()):.3f}.\n\nDo not set a production TopoAug threshold from this ViT-B MAM result. A strict distance gate would reject many harmless flips, and source-specific shifts are material. Next test flip canonicalization or compare MAM after applying the known geometric transform to the original map; then repeat with the intended prior (the OrgMIM README illustrates SAM ViT-H) and visual review. The present distances do not prove mitochondrial-specific topology, and no GT mask was used.\n'''
report=paired/'report.md'
body=report.read_text()
if '## Verified finding and Stage 2 decision' not in body:
 report.write_text(body+finding)
stage=run/'reports/stage1_summary.md'
body=stage.read_text()
update='\n## Completed paired MAM study\n\nThe CPU-only SAM ViT-B/OrgMIM affinity study completed on all 424 image-only slices: 12,720 raw/MAM/inverse-MAM PH distance rows and 13-source figures. MAM responded to stronger crop and blur, but also assigned large distance to flips. No production TopoAug threshold is accepted; see `topology_feasibility/mam_sam_vit_b/paired/report.md`. B_full training continues independently.\n'
if '## Completed paired MAM study' not in body:
 stage.write_text(body+update)
status=run/'STATUS.json'
d=json.loads(status.read_text())
d['topology_feasibility']='paired_sam_vit_b_mam_complete_threshold_not_accepted'
d['topology_mam_samples']=424
d['topology_mam_ph_rows']=12720
t=status.with_suffix('.json.tmp')
t.write_text(json.dumps(d,indent=2))
t.replace(status)
print('mam_complete',summary['rows'],'raw_flip',p50(raw,'flip_horizontal'),'mam_flip',p50(mam,'flip_horizontal'))
PY
git -C "$run" add topology_feasibility/mam_sam_vit_b/paired/generation_complete.json topology_feasibility/mam_sam_vit_b/paired/ph_complete.json topology_feasibility/mam_sam_vit_b/paired/metrics_paired.csv topology_feasibility/mam_sam_vit_b/paired/summary.json topology_feasibility/mam_sam_vit_b/paired/report.md reports/stage1_summary.md
git -C "$run" -c user.name=Codex -c user.email=codex@local commit -m 'Record paired raw EM and MAM PH feasibility result'
git -C "$run" rev-parse HEAD
systemctl show mitoverse-stage1-B-full-v2.service -p ActiveState -p MainPID -p Result --no-pager
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
