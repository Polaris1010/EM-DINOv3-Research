set -euo pipefail
cd /opt/data/3_117/tanghr/codex/mitoverse_full_stage2a2_20261002
env OPENBLAS_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/meddinov3/bin/python - <<'PY'
from pathlib import Path
import json,csv
p=Path('branch_S/gt_mito_diagnostic_20261012')
r=json.loads((p/'summary.json').read_text());assert r['samples']==424 and r['candidates']==1696
with (p/'GT_crop_metrics_424x4.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
assert len(rows)==1696
assert sum(x['has_labelled_mito']=='True' for x in rows)==r['candidates_with_GT_mito']
for x in rows:
    assert 0<=int(x['mito_pixels']) and 0<=float(x['mito_fraction'])<=1
    assert 0<=int(x['mito_cross_boundary_same_id_edges'])<=int(x['boundary_edges'])
    if x['ranking_eligible']=='True':assert x['annotation_status']=='no_region_mask_available' and x['has_labelled_mito']=='True' and int(x['observable_boundary_sides'])==4
assert len(list((p/'gt_panels_first20').glob('*.png')))==20
s=Path('STATUS.json');d=json.loads(s.read_text())
d['research_objective']='GENERALIST_ELECTRON_MICROSCOPY_MITOCHONDRIA_SEGMENTATION_BEAT_OFFICIAL_MITONET'
d['GT_usage']='ALLOWED_FOR_OFFLINE_MITO_SPECIFIC_DIAGNOSTICS_AND_SUPERVISED_TRAINING; FINAL_TEST_LABELS_NOT_FOR_METHOD_TUNING'
d['S_GT_diagnostic']='COMPLETE_424x4; CUT_MEAN_MEDIAN_SPEARMAN_VS_GT_CROSS_MITO_EDGES_0_ON_232_IMAGES'
d['S_manual_review_priority']='SECONDARY_GT_QUALITY_AND_AMBIGUITY_AUDIT; NO_ADDITIONAL_20_PAIR_REVIEW_REQUIRED_NOW'
d['next_benchmark_priority']='OFFICIAL_MITONET_DIRECT_COMPARISON_AND_TRAIN_PRETRAIN_TEST_EXPOSURE_AUDIT'
assert d['long_training']=='PROHIBITED' and d['P'].startswith('PAUSED')
s.write_text(json.dumps(d,indent=2))
print('GT_DIAGNOSTIC_VERIFIED_1696_ROWS_20_PANELS; OBJECTIVE_RECORDED; P_PAUSED_LONG_TRAINING_PROHIBITED')
PY
