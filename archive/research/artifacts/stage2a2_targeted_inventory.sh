set -e
BASE=/opt/data/3_117/tanghr
S1="$BASE/codex/mitoverse_full_stage1_20260928_163120"
printf 'checkpoint_dir\n'; ls -lh "$S1/runs/B_full/formal_attempt1/checkpoints/step000100000" | head -30
printf 'eval_datasets\n'; find "$BASE/Med_DINOv3/nnUNet_preprocessed" -maxdepth 2 -name splits_final.json -print 2>/dev/null | head -30
printf 'round3_eval\n'; ls "$BASE/codex/mitotopo_20260909_round3" | head -30
printf 'lucchi_pair\n'; find "$BASE/DINOv3/pretrain_data/Lucchi_official_pair" -maxdepth 2 -type f -printf '%p %s\n' | head -30
printf 'cmcc_ct\n'; find "$BASE/Med_DINOv3/nnUNet_preprocessed" -maxdepth 2 -type d \( -name 'Dataset111_CMCC' -o -name 'Dataset131_CMCT' \) -print 2>/dev/null
