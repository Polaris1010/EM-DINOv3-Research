# Matched batch-four CEM and mixed supervised experiments

| Family | Starting weight | SHA256 |
|---|---|---|
| P0 | Official natural-image DINOv3 ViT-L/16 | `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035` |
| B100 | MitoVerse-only continued SSL, B_full EMA at 100k | `9603877244ba40f3cbeccdea18d69b0d3b818dc0fc6b09f839fd6d4189ce2de5` |
| Barrier100 | Pilot C100 C1 Barrier EMA at 100k, MitoVerse and CEM1.5M SSL | `9188d26f90216ee966fb8f80fdfde9d33d685202b35e62fa78b34b5b41c073a4` |
| MitoNet | Official released MitoNet-v1 | `487d63dd1deb971a3e374785b350673ee5427e0ffcd44880bf9a54f6c2c4bf04` |

| Split | P0-cem | B100-cem | Barrier100-cem | MitoNet-cem | P0-mix | B100-mix | Barrier100-mix | MitoNet-mix | Official-MitoNet-v1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mv3 | 0.6122 | 0.6083 | 0.6305 | 0.5164 | 0.8230 | 0.8296 | 0.8410 | 0.6661 | 0.6102 |
| cem_validation | 0.6162 | 0.6026 | 0.6829 | 0.6173 | 0.5712 | 0.5622 | 0.6766 | 0.5760 | 0.8875 |
| mb1 | 0.6923 | 0.6976 | 0.7453 | 0.4989 | 0.6628 | 0.6579 | 0.7258 | 0.5034 | 0.7122 |

All eight adapted arms use real batches of four, 40,000 optimizer updates, seed 20261014 and exactly 160,000 sampled crops. Models in the same supervision arm use the same ordered crop stream. Mix uses two CEM and two MitoVerse crops per update. CEM-only retains the same two CEM slots per update and fills the other slots with CEM crops.

Each DINOv3 model uses a frozen ViT-L/16 and a fresh Primus head with the same initialization. Both adapted MitoNet arms start at the official MitoNet-v1 release and update the network with the matched MV-5 optimizer and schedule. The released MitoNet retains its upstream weights. Primus and MitoNet retain their own architectures, objectives, trainable parameter scope and optimizer protocols.

The B100-to-Barrier100 comparison includes upstream differences in SSL data, sampling and the relation loss. Upstream CEM supervised exposure remains possible for official MitoNet. MB-1 is an exposed diagnostic. All checkpoints are evaluated on the same 4,057 labeled units using their fixed native-grid inference policies.
