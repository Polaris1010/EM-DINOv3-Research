set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
import json
from pathlib import Path
run=Path('/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120')
manifest=json.loads((run/'data/manifest_summary.json').read_text())
reader=json.loads((run/'audit/reader_validation.json').read_text())
admission=json.loads((run/'audit/formal_B_full_admission.json').read_text())
init=json.loads((run/'audit/initialization_check.json').read_text())
out=run/'runs/B_full/formal_attempt1'
train=json.loads((out/'result.json').read_text())
progress=json.loads((out/'progress.json').read_text())
checkpoint=json.loads((out/'latest_checkpoint.json').read_text())
logs=json.loads((run/'audit/final_training_log_audit.json').read_text())
mam=json.loads((run/'topology_feasibility/mam_sam_vit_b/paired/summary.json').read_text())
assert manifest['volume_count']==212 and manifest['source_count']==13 and manifest['catalog_disk_manifest_paths_equal']
assert reader['status']=='pass' and reader['read_count']==636
assert admission['status']=='admitted' and len(admission['gates'])==869
assert init['status']=='pass' and init['coverage_ratio']==1.0
assert train['status']=='complete' and train['completed_steps']==100000 and train['max_rank_parameter_difference']==0
assert checkpoint['step']==100000 and Path(checkpoint['path']).is_file()
assert mam['rows']==12720 and mam['samples']==424
for rank in range(3):
 r=logs[f'rank{rank}']
 assert r['lines']==100000 and r['step_gaps']==0 and r['nonfinite_scalars']==0 and r['first_sample_unique_volumes']==212
hours=progress['elapsed_seconds']/3600
avg=progress['elapsed_seconds']/100000
peak=progress['peak_gpu_bytes']/1024**3
raw=mam['modes']['raw_em_paired']
aff=mam['modes']['mam_affinity']
raw_flip=raw['flip_horizontal']['d_equal']['p50']
mam_flip=aff['flip_horizontal']['d_equal']['p50']
mam_crop=aff['aggressive_crop']['d_equal']['p50']
mam_blur=aff['strong_blur']['d_equal']['p50']
report=f'''# Full-MitoVerse Stage 1 final execution report

## Current facts

The catalog and on-disk Zarr path sets matched exactly. The fresh manifest has **{manifest['volume_count']} volumes** from **{manifest['source_count']} sources**, including **{manifest['source_counts']['openorganelle']} OpenOrganelle volumes**, and **{manifest['slice_count']:,} legal Z slices**. All {reader['read_count']} first/middle/last representative image reads passed; no unreadable volume was found in this audit and no repair was required. This is a representative read audit, not an all-chunk checksum. Manifest SHA256: `{manifest['manifest_sha256']}`. Sampling is uniform over legal slices, with no old split or label filter.

## B_full

Initialization was the official natural-image DINOv3 ViT-L/16 weight, SHA256 `{train['official_weight_sha256']}`; DINOv3 source commit `{train['dinov3_commit']}`. All {init['matched_keys']} backbone keys matched, with zero feature difference for checked CLS and patch outputs. **No old P1/P2/B/L/T/LT checkpoint or optimizer was loaded.** The successful formal attempt started at step 0 with no resume; an earlier attempt failed at step 0 due to an NFS heartbeat timestamp issue and was preserved separately.

`B_full` completed **100,000/100,000 optimizer steps** on physical GPUs 1–3. The global batch was {train['effective_global_batch']}; total wall time was {hours:.2f} hours and mean end-to-end time was {avg:.3f} seconds per step, including checkpoint overhead. Peak allocated GPU memory reported by PyTorch was {peak:.2f} GiB per rank (live nvidia-smi usage during training was about 11 GiB per card). The objective was DINO=1, iBOT=1, KoLeo=.1, with Gram, LAST, topology, and segmentation branches disabled.

Three rank logs each contain 100,000 continuous steps with zero nonfinite loss/gradient/schedule scalar and zero step gaps. Each rank's first-sample records covered all 212 volumes. DINO+iBOT+KoLeo loss decreased from 16.625 at step 1 to the last rank-0 loss {progress['loss']:.4f}; student and teacher tracked parameters moved, and final tracked inter-rank parameter difference was {train['max_rank_parameter_difference']:.1f}. Loss stability does not establish representation improvement. The final full optimizer checkpoint is `runs/B_full/formal_attempt1/checkpoints/step000100000/full.pth`, SHA256 `{checkpoint['sha256']}`; student and teacher backbone exports exist beside it. Training service exited successfully; the website stayed available. Formal admission passed 869 gates.

## Topology feasibility and Stage 2 gate

Raw EM PH and a frozen Meta SAM ViT-B/OrgMIM-affinity MAM were tested on two label-free slices from every volume: 424 images, 10 augmentations, 13 sources, and 12,720 paired H0/H1 distance rows across raw, affinity, and inverse-boundary filtrations. No GT mask was used. The MAM test is ViT-B feasibility; the OrgMIM example uses SAM ViT-H. Code reference: https://github.com/yanchaoz/OrgMIM and TopoCL topology reference: https://github.com/gm3g11/TopoCL .

For the exploratory equal-weight H0+H1 bottleneck distance, median horizontal-flip distance was **{raw_flip:.3f} for raw EM** and **{mam_flip:.3f} for MAM**. MAM median aggressive-crop and strong-blur distances were **{mam_crop:.3f}** and **{mam_blur:.3f}**. Strong crop exceeded mild crop in {100*aff['pairwise_fraction']['aggressive_crop_gt_mild_crop']:.1f}% of paired images; strong blur exceeded weak blur in {100*aff['pairwise_fraction']['strong_blur_gt_weak_blur']:.1f}%. The MAM flip median varied by source from {min(x['flip_horizontal'] for x in mam['source_medians']['mam_affinity'].values()):.3f} to {max(x['flip_horizontal'] for x in mam['source_medians']['mam_affinity'].values()):.3f}. Raw grayscale PH is not a trustworthy proxy for mitochondrial-specific topology; MAM has useful strength ordering but is **not more stable under flips**. Neither result proves biological topology preservation.

**No absolute TopoAug threshold is accepted for Stage 2.** A current distance gate would reject many harmless flips. First test a transform-aligned MAM comparison and review representative membranes across sources, ideally with the intended SAM prior. After those checks, a defensible starting search is a source-normalized benign-distance quantile in the **85th–95th percentile** range, then assess aggressive-crop false acceptance. This is a calibration proposal, not a selected threshold. Do not launch formal T1 until the flip and source-dependence gate passes. See `topology_feasibility/mam_sam_vit_b/paired/report.md` for full distributions.

## Reproducibility and scope

The resolved config, environment snapshot, source commit and patch, manifest, admission, step logs, checkpoints, and PH metrics are stored in this separate run. Old experiments remain preserved. No LAST/LT/T long run, segmentation fine-tuning, MitoNet evaluation, or frozen probe was started. Baseline training is complete; topology threshold readiness remains blocked by the observed MAM flip response.
'''
(run/'reports/stage1_summary.md').write_text(report)
(run/'reports/B_training_report.md').write_text(report.split('## B_full',1)[1].split('## Topology feasibility',1)[0].join(['# B_full final training report\n\n','']))
(run/'audit/final_stage1_audit.md').write_text('# Stage 1 final audit\n\nB_full training: PASS, 100000 steps, checkpoint and exports present, rank logs continuous and finite. Full data coverage audit: PASS for catalog/disk equality and 636 representative image reads. Topology feasibility: COMPLETE with negative flip-stability finding. Stage 2 T1 admission: NOT PASSED; no validated TopoAug threshold. See `reports/stage1_summary.md` and `topology_feasibility/mam_sam_vit_b/paired/report.md`.\n')
status_path=run/'STATUS.json'
status=json.loads(status_path.read_text())
status.update({'stage':'stage1_baseline_complete_topology_gate_not_passed','B_full':'complete','B_full_steps':100000,'B_full_checkpoint_sha256':checkpoint['sha256'],'B_full_rank_audit':'pass','topology_feasibility':'complete_negative_flip_stability','stage2_T1_ready':False,'website_health_at_final_audit':'http_200'})
tmp=status_path.with_suffix('.json.tmp')
tmp.write_text(json.dumps(status,indent=2))
tmp.replace(status_path)
print('final_report_written',hours,avg,peak)
PY
git -C "$run" add reports/stage1_summary.md reports/B_training_report.md audit/final_stage1_audit.md audit/final_training_log_audit.json
git -C "$run" -c user.name=Codex -c user.email=codex@local commit -m 'Finalize 100k B_full audit and topology gate finding'
git -C "$run" rev-parse HEAD
