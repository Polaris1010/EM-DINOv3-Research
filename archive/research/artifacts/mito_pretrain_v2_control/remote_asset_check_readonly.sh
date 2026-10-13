set -u
base=/opt/data/3_117/tanghr
weight="$base/DINOv3/pretrained_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
ps -p 3726505 -o pid,ppid,user,etime,args --no-headers || true
for i in 1 2 3; do
  curl --silent --show-error --output /dev/null --write-out "site_check_${i}=%{http_code} seconds=%{time_total}\n" --max-time 5 http://127.0.0.1:7860/
done
sha256sum "$weight"
ls -lah "$base/DINOv3" | head -n 30
ls -lah "$base/Med_DINOv3" | head -n 30
find "$base/DINOv3" "$base/Med_DINOv3" -maxdepth 4 -type f \( -iname '*ssl*train*.py' -o -name 'train.py' -o -name 'ssl_default_config.yaml' \) -printf '%p\n' 2>/dev/null | head -n 45
find "$base/codex" -maxdepth 1 -type d -name 'mito_pretrain_v2_*' -printf '%f\n' 2>/dev/null | sort
du -sh "$base/MitoVerse_data" "$base/DINOv3" "$base/Med_DINOv3" 2>/dev/null
