set -euo pipefail
run=/opt/data/3_117/tanghr/codex/mitoverse_full_stage1_20260928_163120
paired="$run/topology_feasibility/mam_sam_vit_b/paired"
systemctl show mitoverse-stage1-mam-paired-smoke.service -p ActiveState -p MainPID -p Result --no-pager
if test -f "$paired/generation_progress.json"; then cat "$paired/generation_progress.json"; fi
if test -f "$paired/maps/sample_000000.npz"; then stat -c 'paired_sample_bytes=%s' "$paired/maps/sample_000000.npz"; fi
tail -n 10 "$paired/paired_smoke.log"
systemctl show mitoverse-stage1-B-full-v2.service -p ActiveState -p Result --no-pager
curl -sS -o /dev/null -w 'website_http=%{http_code}\n' http://127.0.0.1:7860/
