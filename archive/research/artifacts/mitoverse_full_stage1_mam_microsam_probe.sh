set -euo pipefail
ls -lah /root/.cache/torch /root/.cache/torch/hub 2>/dev/null || true
grep -nE 'get_cache_directory|cache_dir|checkpoint_path|SAM_MODEL|download_model|model_url' /opt/data/3_117/tanghr/microsam/micro-sam/micro_sam/util.py | head -75 || true
grep -nE 'vit_b_lm|vit_l|vit_h|checkpoint_path' /opt/data/3_117/tanghr/microsam/micro-sam/microsam_baseline_nnunet_CMCC.py /opt/data/3_117/tanghr/microsam/micro-sam/1.py /opt/data/3_117/tanghr/microsam/micro-sam/2.py | head -65 || true
ls -lah /opt/data/3_117/tanghr/microsam/micro-sam/baseline_results_microsam_vitl_CMCC | head -30
