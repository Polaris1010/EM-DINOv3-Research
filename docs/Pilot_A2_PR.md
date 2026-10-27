On 64 fixed Glycolytic muscle FOVs, the existing layer-23 linear probes lose native-pixel ranking from D2/30k to D2/100k: AUROC 0.9429→0.9207 and AP 0.5043→0.3228. Two global offsets fitted only on the fixed 312 MV3 train FOVs reduce 100k remote false positives by 80.11%, but Dice improves only 0.2906→0.3666 while recall falls 0.8019→0.3986. All four control-source Dice point estimates decline.

About 74% of the apparent reversal of the calibrated 30k/100k gap comes from degradation of the 30k control. The report separates that effect from the absolute 100k recovery. Independently trained probes prevent attributing cross-model logits to encoder spatial common-mode drift. Retire global-bias repair as the main direction; no new method or training was started.

Execution: locked layer 23 and identical native pixels, existing probabilities, T=1, source-equal BCE with natural class proportions; baseline scoring, isolated train-only scalar fitting, then fixed five-source scoring and 200 paired Z-block bootstrap draws. GPU 3 only; service completed in 50.2 seconds and stopped, website HTTP200, 11 historical input SHA256s unchanged. No encoder forward or network update; no View-Mean, PCA or source-specific calibration.

Validation: GPU weighted-tie AUROC/AP and scalar closed-form fixtures; independent reconciliation of 1,280 crop rows, 20 pooled metrics, 20 ranking rows and 85 paired estimates/intervals; 60 historical registry entries unchanged; repository checks and 51 unit tests (3 existing platform skips). Code, metrics, SVGs, report and work log included. Raw images, GT arrays, features and weights remain outside Git.

Report: docs/Pilot_A2_Readout_Calibration_Report.md. This independent draft PR is based on codex/feature-mechanism-pilot-a; no automatic merge.
