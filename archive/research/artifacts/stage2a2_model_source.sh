set -e
BASE=/opt/data/3_117/tanghr/DINOv3/dinov3
grep -Rni 'def build_model_from_cfg\|def get_intermediate_layers' "$BASE/dinov3/models" | head -30
grep -n -A75 'def build_model_from_cfg' "$BASE/dinov3/models/__init__.py" | head -95
grep -n -A65 'def get_intermediate_layers' "$BASE/dinov3/models/vision_transformer.py" | head -80
