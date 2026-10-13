# B_full final training report



Initialization was the official natural-image DINOv3 ViT-L/16 weight, SHA256 `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`; DINOv3 source commit `91ffb6803c74aeb6927209b82560142743763dd7`. All 368 backbone keys matched, with zero feature difference for checked CLS and patch outputs. **No old P1/P2/B/L/T/LT checkpoint or optimizer was loaded.** The successful formal attempt started at step 0 with no resume; an earlier attempt failed at step 0 due to an NFS heartbeat timestamp issue and was preserved separately.

`B_full` completed **100,000/100,000 optimizer steps** on physical GPUs 1–3. The global batch was 48; total wall time was 22.25 hours and mean end-to-end time was 0.801 seconds per step, including checkpoint overhead. Peak allocated GPU memory reported by PyTorch was 9.54 GiB per rank (live nvidia-smi usage during training was about 11 GiB per card). The objective was DINO=1, iBOT=1, KoLeo=.1, with Gram, LAST, topology, and segmentation branches disabled.

Three rank logs each contain 100,000 continuous steps with zero nonfinite loss/gradient/schedule scalar and zero step gaps. Each rank's first-sample records covered all 212 volumes. DINO+iBOT+KoLeo loss decreased from 16.625 at step 1 to the last rank-0 loss 12.5625; student and teacher tracked parameters moved, and final tracked inter-rank parameter difference was 0.0. Loss stability does not establish representation improvement. The final full optimizer checkpoint is `runs/B_full/formal_attempt1/checkpoints/step000100000/full.pth`, SHA256 `12bbf09ffcd7f93cbad88c4e55df55d36b3a299990ad976bf29af51a118a265b`; student and teacher backbone exports exist beside it. Training service exited successfully; the website stayed available. Formal admission passed 869 gates.

