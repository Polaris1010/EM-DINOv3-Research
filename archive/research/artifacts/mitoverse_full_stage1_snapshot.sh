set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
dino=/opt/data/3_117/tanghr/DINOv3/dinov3
git -C "$dino" diff > "$run/src/local_diff.patch"
git -C "$dino" status --short > "$run/src/modified_files.txt"
/opt/data/3_117/tanghr/envs/meddinov3/bin/python - "$run" "$dino" <<'PY'
import json,subprocess,sys
from pathlib import Path
run,dino=map(Path,sys.argv[1:])
commit=subprocess.check_output(['git','-C',str(dino),'rev-parse','HEAD'],text=True).strip()
status=subprocess.check_output(['git','-C',str(dino),'status','--short'],text=True)
row={'dinov3_commit':commit,'dinov3_worktree_modified':bool(status.strip()),
     'dinov3_diff_path':'src/local_diff.patch','stage1_source':'scripts/',
     'official_weight_initialization_only':True,'old_checkpoint_initialization':False}
(run/'src/repo_state.json').write_text(json.dumps(row,indent=2)+'\n')
PY
cat > "$run/src/SUPERSEDES_OLD_EXPERIMENTS.md" <<'EOF'
# Supersedes prior experiments

The 2026-09-28 full-restart B/L/T service was stopped and its checkpoint files
were preserved. This Stage 1 run uses a new manifest, uniform slice sampler,
official natural ViT-L/16 initialization, and a single B_full protocol.
No old P1/P2, B/L/T/LT, optimizer, teacher, or sampler state is an input.
EOF
cat > "$run/reports/B_smoke_report.md" <<'EOF'
# B_full smoke report

- 1000 image-only augmented samples passed; non-image Zarr getter calls: 0.
- Official-versus-student backbone check: 368 matched state keys, 100% tensor
  coverage, zero CLS and patch feature difference on fixed inputs.
- Single-GPU 20-step smoke passed with finite losses/gradients and movement of
  student and EMA teacher parameters. Peak VRAM was about 10.2 GB.
- Same-run 10-to-20-step resume matched the continuous sample chain.
- Three-GPU 100-to-200-step smoke passed with global batch 48.
- The final trainer version also passed a fresh 200-step distributed smoke;
  maximum cross-rank parameter difference was 0. Peak VRAM was about 10.2 GB
  per GPU and the website stayed HTTP 200.
- Smoke checkpoints are technical validation only and are not formal initializers.
EOF
cat > "$run/.gitignore" <<'EOF'
runs/
topology_feasibility/maps/
topology_feasibility/persistence/
topology_feasibility/figures/
*.tmp
__pycache__/
EOF
git -C "$run" init --initial-branch=codex/stage1 >/dev/null
git -C "$run" add .gitignore README.md scripts configs data src audit env topology_feasibility/README.md topology_feasibility/sample_manifest.json reports
git -C "$run" -c user.name=Codex -c user.email=codex@local commit -m 'Prepare audited Full MitoVerse Stage 1 B_full baseline' >/dev/null
git -C "$run" rev-parse HEAD
git -C "$run" status --short
