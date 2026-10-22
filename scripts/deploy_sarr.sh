#!/bin/bash
# SARR Phase 1: Server deployment and execution script
# Deploy from local to server, then run training
#
# Prerequisites:
#   - Coverage-NR training should be completed or on different GPUs
#   - GPU 0 is FORBIDDEN
#   - Website at 127.0.0.1:7860 must remain accessible
#
# Usage:
#   1. Copy this script and src/sarr/ to server
#   2. Run: bash deploy_sarr.sh
#
# Estimated GPU time: ~8 hours total (3 seeds SARR + 2 ablation baselines)

set -euo pipefail

# ============================================================
# Configuration
# ============================================================
SERVER_ROOT="/opt/data/3_117/tanghr/codex"
SARR_ROOT="${SERVER_ROOT}/sarr_phase1_$(date +%Y%m%d)"
PYTHON="/opt/data/3_117/tanghr/envs/meddinov3/bin/python"
DINOV3_ROOT="/opt/data/3_117/tanghr/DINOv3/dinov3"

# GPU allocation: ONLY use GPU 1,2,3. GPU 0 is FORBIDDEN.
export CUDA_VISIBLE_DEVICES=1,2,3

# ============================================================
# Pre-flight checks
# ============================================================
echo "=== SARR Phase 1 Pre-flight Checks ==="

# Check website is accessible
if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:7860/ | grep -q "200"; then
    echo "[OK] Website accessible (HTTP 200)"
else
    echo "[FAIL] Website not accessible. Aborting."
    exit 1
fi

# Check GPU 0 is not in our visible set
echo "[OK] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (GPU 0 excluded)"

# Check Python environment
if ${PYTHON} -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.device_count()} GPUs visible')" 2>/dev/null; then
    echo "[OK] Python environment"
else
    echo "[FAIL] Python environment check failed"
    exit 1
fi

# Check encoder checkpoints exist
P0_PATH="/opt/data/3_117/tanghr/DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
B100K_PATH="/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120/runs/B_full/formal_attempt1/checkpoints/step000100000/teacher_backbone.pth"

[ -f "${P0_PATH}" ] && echo "[OK] P0 checkpoint" || { echo "[FAIL] P0 checkpoint not found"; exit 1; }
[ -f "${B100K_PATH}" ] && echo "[OK] B100k checkpoint" || { echo "[FAIL] B100k checkpoint not found"; exit 1; }

# ============================================================
# Setup experiment directory
# ============================================================
mkdir -p "${SARR_ROOT}"
echo "Experiment root: ${SARR_ROOT}"

# Copy source code
cp -r src/sarr "${SARR_ROOT}/src_sarr/"
cp src/segmentation/official_primus.py "${SARR_ROOT}/src_sarr/"
cp src/segmentation/mv_data.py "${SARR_ROOT}/src_sarr/"
cp src/segmentation/mv_probe.py "${SARR_ROOT}/src_sarr/"

echo "=== Source code deployed ==="

# ============================================================
# Smoke test
# ============================================================
echo "=== Running smoke test ==="
${PYTHON} "${SARR_ROOT}/src_sarr/train_sarr.py" \
    --mode sarr \
    --seed 20261014 \
    --out "${SARR_ROOT}/smoke" \
    --smoke

if [ $? -eq 0 ]; then
    echo "[OK] Smoke test passed"
else
    echo "[FAIL] Smoke test failed. Check ${SARR_ROOT}/smoke/"
    exit 1
fi

# ============================================================
# Training runs
# ============================================================
STEPS=40000

echo "=== Starting SARR training runs ==="

# Run 1-3: Full SARR with 3 seeds
for SEED in 20261014 20261015 20261016; do
    echo "--- SARR seed ${SEED} ---"
    ${PYTHON} "${SARR_ROOT}/src_sarr/train_sarr.py" \
        --mode sarr \
        --seed ${SEED} \
        --steps ${STEPS} \
        --out "${SARR_ROOT}/runs/sarr/seed${SEED}"
done

# Run 4: Uniform average baseline (1 seed)
echo "--- Uniform average baseline ---"
${PYTHON} "${SARR_ROOT}/src_sarr/train_sarr.py" \
    --mode uniform_avg \
    --seed 20261014 \
    --steps ${STEPS} \
    --out "${SARR_ROOT}/runs/uniform_avg/seed20261014"

# Run 5: Global gate baseline (1 seed)
echo "--- Global gate baseline ---"
${PYTHON} "${SARR_ROOT}/src_sarr/train_sarr.py" \
    --mode global_gate \
    --seed 20261014 \
    --steps ${STEPS} \
    --out "${SARR_ROOT}/runs/global_gate/seed20261014"

echo "=== All training runs completed ==="

# ============================================================
# Post-flight check
# ============================================================
if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:7860/ | grep -q "200"; then
    echo "[OK] Website still accessible after training"
else
    echo "[WARNING] Website may be down. Check manually."
fi

echo "=== SARR Phase 1 Complete ==="
echo "Results in: ${SARR_ROOT}/runs/"
