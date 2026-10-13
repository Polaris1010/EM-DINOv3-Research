set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - "$run" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1]); manifest=json.loads((run/'data/pretrain_all_mitoverse.json').read_text())
rows=[]
for volume in manifest['volumes']:
    depth,height,width=volume['shape']
    for fraction in (0.25,0.75):
        z=min(depth-1,int(depth*fraction))
        rows.append({'volume_id':volume['volume_id'],'source':volume['source_dataset'],
                     'slice_index':z,'image_shape':[height,width],
                     'voxel_size':volume['voxel_size'],'voxel_size_unit':'nm',
                     'image_array_path':volume['image_array_path'],
                     'local_path':volume['local_path']})
out={'sample_version':'stage1_topology_feasibility_v1',
     'selection':'every volume at 25% and 75% Z depth; no label or ROI selection',
     'manifest_sha256':(run/'data/pretrain_all_mitoverse.sha256').read_text().split()[0],
     'volume_count':len(manifest['volumes']),'sample_count':len(rows),'samples':rows}
path=run/'topology_feasibility/sample_manifest.json'
path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'volume_count':out['volume_count'],'sample_count':out['sample_count']}))
PY
cat > "$run/topology_feasibility/README.md" <<'EOF'
# Topology feasibility

The 25% and 75% Z slices of every Full MitoVerse volume were selected from
image metadata only. No mitochondria label, valid mask, or ROI was read.
Raw EM and OrgMIM-style membrane attention priors must both be measured before
claiming a topology preservation threshold. This directory is diagnostic only;
no T model is trained in Stage 1.
EOF
