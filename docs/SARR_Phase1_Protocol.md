# SARR Phase 1: Source-Adaptive Representation Routing

**Status:** PLANNED — Phase 0 PASS, awaiting server deployment  
**Branch:** `codex/sarr-phase1`  
**Phase 0 commit:** dc06d88

## Research Question

Can a lightweight per-patch spatial router that dynamically selects between
P0 (natural-image) and B100k (MitoVerse SSL) frozen representations improve
cross-source EM mitochondria segmentation, especially on sources where B100k
shows negative transfer?

## Phase 0 Result: Feasibility Confirmed

Conservative oracle Dice (per-region confusion matrix bounds):
- P0 source-eq Dice: 0.8130
- B100k source-eq Dice: 0.8273
- **Conservative Oracle: 0.9193** (+0.0888 vs better single)
- Upper-bound Oracle: 0.9891 (+0.1586 vs better single)

Key complementarity sources: Haberl (+0.300), Turner (+0.156), Guay (+0.155),
OpenOrganelle (+0.123). Even B100k-degraded sources (Guay, Muller) show
significant oracle improvement.

## Protocol

### Encoders (FROZEN, not modified)
- P0: Official DINOv3 ViT-L/16 natural-image weights
- B100k: MitoVerse 100k continual SSL teacher backbone

### Router Architecture (~500K trainable params)
- Input: P0 + B100k concatenated multi-layer features (8192 ch at 16x16)
- 1x1 Conv(8192, 256) + LayerNorm + GELU
- 1x1 Conv(256, 1) + Sigmoid -> alpha(x) in [0,1]
- Routed features: alpha * f_P0 + (1-alpha) * f_B100k

### Decoder
- Original Primus_Multiscale PatchDecode (same as MV-3)
- Layers [5,11,17,23], no frequency gates
- ~44.6M trainable params

### Training
- Same MV-Generalist-v2: 156 train / 38 dev / 12 sources
- Same optimizer: AdamW lr=1e-4, wd=0.05, betas=(0.9, 0.98)
- Same LR: cosine with 100-step warmup
- Same loss: CE + Dice
- Same sampling: source-balanced
- Budget: 40k steps per run
- 3 decoder seeds: 20261014, 20261015, 20261016
- TTA=OFF

### Evaluation
- MV-3 dev: 38 regions, 12 sources, full-volume
- MB-1: 7 benchmarks (already-exposed diagnostic, not blind test)
- Router weight visualization: mean alpha per source

### Ablation Baselines (single seed 20261014)
| Mode | Description |
|---|---|
| sarr | Full spatial router (main method) |
| uniform_avg | Fixed alpha=0.5 everywhere |
| global_gate | Single learned scalar alpha |
| p0_only | Only P0 encoder (should match G-P0) |
| b100_only | Only B100k encoder (should match G-B100) |

### Success Criteria
1. SARR source-eq dev Dice > G-B100 (0.8264) AND Guay/Muller not degraded
2. SARR MB-1 7-benchmark Dice > G-B100 (0.5027) AND Glycolytic muscle > 0.50
3. SARR > uniform_avg > max(p0_only, b100_only) (spatial routing > global > single)
4. At least 2/3 seeds positive

### Failure / Stop Rules
- After first seed: if dev Dice < G-B100, inspect router weights
- If router alpha collapses to 0 or 1: add entropy regularization
- After two seeds with no improvement: STOP, pivot to Scale-Aware Adapter

## Estimated Compute
| Step | GPU hours (3x 4090) |
|---|---|
| Smoke test | ~0.1 |
| SARR x 3 seeds (train + eval) | ~6 |
| uniform_avg x 1 | ~2 |
| global_gate x 1 | ~2 |
| **Total** | **~10** |

p0_only / b100_only are not re-run; we reuse existing G-P0/G-B100 results.

## Limitations
- Dual encoder = ~2x inference FLOPs (will note in paper)
- Router trained on same 156 MV regions (no new supervision)
- MB-1 already exposed in prior research decisions
- Oracle analysis used confusion matrix bounds, not exact pixel overlap
