# B_full configuration derivation

Pinned DINOv3 source commit: `91ffb6803c74aeb6927209b82560142743763dd7`.
`B_full.yaml` is merged over that checkout's `get_default_config()`. The full
merge is saved as `B_full_resolved.yaml` in the remote Stage 1 directory. No
legacy YAML, checkpoint, optimizer, scheduler, sampler cache, or resume path is
loaded into the new formal run.

| Setting | Official DINOv3 default | Old 31-volume B | New B_full |
|---|---|---|---|
| Image pool | Generic configured dataset | 31 old train volumes | Every catalog-matched MitoVerse image volume |
| Sampling | Official dataset sampler | Uniform source, then volume | Uniform over all 48,722 Z slices |
| Backbone | ViT-L/16 config | ViT-L/16 with 4 storage tokens, mask-k bias | Same architecture, strict official weight load |
| Initialization | Configuration dependent | Official natural ViT-L | Official natural ViT-L; no old checkpoint |
| SSL losses | DINO=1, iBOT=1, KoLeo=0.1 | Same | Same |
| iBOT mask ratio | 0.1–0.5 | 0.1–0.5 | 0.1–0.5 |
| Gram | Off | Off | Off |
| LAST / topology | Absent | B arm off | Off |
| Global / local crop | 224 / 96 | 256 / 112, 8 local | 256 / 112, 8 local |
| Batch per GPU | 16 | 16 on one GPU | 16 on each of three GPUs; global 48 |
| Peak LR | Official 0.001 before batch scaling | 1.5e-5 at global 16 | 2.5980762e-5 at global 48, square-root scaled from old stable value |
| LR schedule | Cosine with warmup | 5k linear warmup from 10% peak, then flat | 5k warmup from zero, cosine to 1e-6 over 100k |
| Weight decay | 0.04 to 0.4 | 0.04 | Constant 0.04 |
| Teacher momentum | 0.992 to 1 | 0.999 | Constant 0.999 |
| Optimizer updates | Epoch-based config | 100,000 | Exactly 100 × 1,000 = 100,000 |

The new LR schedule follows the official warmup/cosine shape while keeping a
conservative peak anchored to the old completed B run. This is an explicit
hyperparameter change from the old 31-volume run, not a hidden inheritance.
The old run is not a valid performance comparator because its data scope and
sampler differ.
