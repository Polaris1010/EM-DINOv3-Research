set +e
curl --noproxy '*' -I --max-time 5 http://10.207.21.130:8765/stage2_sam_vit_h_4b8939.pth 2>&1 | tail -5
