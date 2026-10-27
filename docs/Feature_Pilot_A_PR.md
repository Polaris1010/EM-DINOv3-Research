Pilot A found that D2/30k→100k muscle segmentation degradation also occurs with a frozen train-only linear readout, while a stable, readout-independent loss of foreground/background separability remains unestablished.

- Same locked 633 native-grid crops; five historical encoder SHA256s; layers [5,11,17,23]. Only 20 lightweight MV3-train probes were fitted. No SSL or Primus training.
- Muscle random-FOV Linear23 Dice: 0.5090→0.2906; remote FP rate: 0.31%→9.72%. Deep effective rank: 33.3074→21.8166. Linear AUROC remains 0.9727, while prototype AUROC improves, separating spectral compression from semantic confusion.
- Periodicity evidence is weak and readout-dependent: Primus rho 0.0914, Z-block interval crosses zero; adjusted linear association is near zero. Final reviewed verdict is INCONCLUSIVE for a stable positive periodicity mechanism.
- Archive complete CSVs, matched-token spectra, SVG figures, locked coordinates, train-only fitting/label-access evidence, final report and work log. Original automatic verdicts are preserved; final interpretation is in reviewed_verdicts.json.

Validation: GPU numerical smoke; independent reconciliation of 8,025 crop rows, 125 pooled metrics, 40 paired AUROC rows and 300 spectra; unchanged 59 historical registry entries; repository checks and unit tests. Final command outcomes are archived in evidence/Feature-Pilot-A/closeout_checks.json.

Server service stopped successfully; GPU0 unused, website HTTP200, historical checkpoint hashes unchanged. Pilot B/C and further experiments were not started. Draft only; no automatic merge.

Report: [Feature_Mechanism_Pilot_Report.md](https://github.com/Polaris1010/EM-DINOv3-Research/blob/codex/feature-mechanism-pilot-a/docs/Feature_Mechanism_Pilot_Report.md).
