set -u
grep -R -n 'class DataAugmentationDINO' /opt/data/3_117/tanghr/DINOv3/dinov3/dinov3/data --include='*.py' | head -n 5
