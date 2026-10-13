"""Within-sample score/proxy ordering and crop severity diagnostic."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / "src"))
from crop_score_pilot import score  # noqa: E402

OUT = RUN / "topology_calibration/crop_score"
MAPS = RUN / "topology_calibration/sam_vit_h/paired_maps"


def main() -> None:
    pilot = json.loads((OUT / "crop_score_pilot_424x4.json").read_text())
    with (OUT / "crop_score_pilot_424x4.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1696
    within_rho = []
    low_minus_high_proxy = []
    for i in range(424):
        group = rows[i * 4:(i + 1) * 4]
        s = [float(r["topo_score"]) for r in group]
        p = [float(r["border_proxy"]) for r in group]
        rho = spearmanr(s, p).statistic
        if np.isfinite(rho):
            within_rho.append(float(rho))
        order = np.argsort(s)
        low_minus_high_proxy.append(p[order[0]] - p[order[-1]])
    samples = json.loads((RUN / "topology_calibration/sample_manifest.json").read_text())["samples"]
    fractions = (.95, .80, .65, .45)
    severity = []
    edges = pilot["area_bin_edges"]
    for i, sample in enumerate(samples):
        with np.load(MAPS / f"sample_{i:06d}.npz") as pack:
            mam = pack["mam"][0]
        group = []
        for fraction in fractions:
            side = round(512 * fraction)
            y = x = (512 - side) // 2
            d0, d1, proxy = score(mam, (y, x, side, side))
            area = (side / 512) ** 2
            bin_index = int(area > edges[0]) + int(area > edges[1])
            key = f"{sample['source']}:{bin_index}"
            norm = pilot["normalizers"].get(key, pilot["normalizers"][f"GLOBAL:{bin_index}"])
            s = .5 * (d0 - norm["d0"][0]) / norm["d0"][1] + .5 * (d1 - norm["d1"][0]) / norm["d1"][1]
            group.append({"fraction": fraction, "score": float(s), "d0": d0, "d1": d1, "proxy": proxy})
        severity.append(group)
        if i % 50 == 0:
            print(f"severity {i + 1}/424", flush=True)
    arrays = {str(f): np.array([group[j]["score"] for group in severity]) for j, f in enumerate(fractions)}
    raw_d0 = {str(f): np.array([group[j]["d0"] for group in severity]) for j, f in enumerate(fractions)}
    raw_d1 = {str(f): np.array([group[j]["d1"] for group in severity]) for j, f in enumerate(fractions)}
    result = {"status": "complete", "within_sample_spearman_median": float(np.median(within_rho)),
              "within_sample_spearman_mean": float(np.mean(within_rho)),
              "within_sample_spearman_positive_fraction": float(np.mean(np.array(within_rho) > 0)),
              "lower_score_minus_higher_score_border_proxy_mean": float(np.mean(low_minus_high_proxy)),
              "lower_score_has_less_border_proxy_fraction": float(np.mean(np.array(low_minus_high_proxy) < 0)),
              "severity_score_medians": {f: float(np.median(a)) for f, a in arrays.items()},
              "severity_d0_medians": {f: float(np.median(a)) for f, a in raw_d0.items()},
              "severity_d1_medians": {f: float(np.median(a)) for f, a in raw_d1.items()},
              "within_training_scale_more_destructive_greater_fraction": float(np.mean(arrays['0.65'] > arrays['0.95'])),
              "within_training_scale_more_destructive_delta_median": float(np.median(arrays['0.65'] - arrays['0.95'])),
              "within_training_scale_more_destructive_wilcoxon_p": float(wilcoxon(arrays['0.65'] - arrays['0.95']).pvalue),
              "out_of_range_045_greater_than_095_fraction": float(np.mean(arrays['0.45'] > arrays['0.95']))}
    (OUT / "crop_score_diagnostic_424.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
