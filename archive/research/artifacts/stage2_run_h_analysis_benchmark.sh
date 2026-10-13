set -euo pipefail
ROOT=/opt/data/3_117/tanghr/codex/mitoverse_full_stage2_20260930_082816
STAGE1=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
cd "$ROOT"
test "$(find topology_calibration/sam_vit_h/paired_maps -maxdepth 1 -name 'sample_*.npz' | wc -l)" = 424
test -f topology_calibration/sam_vit_h/generation_complete.json
test "$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:7860/)" = 200
nohup env CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/microsam/bin/python src/h_tile_benchmark.py > topology_calibration/cache_benchmark/vit_h_raw_tile_10.log 2>&1 < /dev/null &
echo $! > topology_calibration/cache_benchmark/vit_h_raw_tile_10.pid
PYTHONPATH="$STAGE1/topology_feasibility/vendor" OPENBLAS_NUM_THREADS=1 /opt/data/3_117/tanghr/envs/meddinov3/bin/python src/sam_h_analyze.py
