---
license: other
task_categories:
  - image-segmentation
tags:
  - electron-microscopy
  - connectomics
  - mitochondria
  - 3d-segmentation
  - instance-segmentation
pretty_name: MitoVerse
size_categories:
  - 10B<n<100B
---

# MitoVerse

A generalist **3D-EM mitochondria segmentation** benchmark: many domains (modality, species, tissue,
resolution) in one consistent format, with **instance** labels.

**Code, loaders, ingestion, and PyTorchConnectomics usage:**
👉 https://github.com/PytorchConnectomics/mitoverse

Each volume is one zarr store `data/<dataset>/<volume>.zarr` with arrays `img` (uint8, ZYX) and
`mito` (instance, 0=background); per-volume metadata (voxel size, modality, species, tissue) lives in
its `.zattrs` and in `catalog.json`. Semantic masks are `mito > 0` — derive as needed.

## 3D EM Datasets with Mitochondria Annotation

`*` = instance labels we converted from binary segmentation. `†` = redistribution caveat.
Shape is the average `(x,y,z)` per volume, `×N` = number of volumes.

| Microscope | Dataset | Species | Region | Avg Shape (xyz) | Resolution (nm) | # Mitos |
|---|---|---|---|---|---|---|
| FIB-SEM | [OpenOrganelle](https://openorganelle.janelia.org/) (Xu'21), [CellMap](https://cellmapchallenge.janelia.org/) annot. | mixed | various cells/tissues | (213,204,203)×147 | 8×8×8 | 1580 |
| FIB-SEM | └ + MitoEM2.0 chunks (Liu et al. '26) | mixed | jurkat/macro/kidney/liver/cardiac | (1024,1024,256)×10 | 8×8×8 | 2910 |
| FIB-SEM | [Müller'21 (BetaSeg)](https://betaseg.github.io/)* | Mouse | Pancreas (β-cells) | (874,669,979)×7 | 16×16×16 | 1852 |
| FIB-SEM | [Mekuč'20 (UroCell)](https://github.com/MancaZerovnikMekuc/UroCell) | Mouse | Urinary bladder | (256,256,256)×5 | 16×16×15 | 287 |
| FIB-SEM | Kunduri'22 | Drosophila | Gonad (spermatocyte) | (1024,1024,341)×3 | 16×16×16 | 621 |
| FIB-SEM | [Conrad'23 (MitoNet)](https://volume-em.github.io/empanada.html) | mixed | fly brain / C. elegans / HeLa / muscle / salivary | (546,446,466)×5 | 12–24 (iso) | 635 |
| ssSEM | [Wei'20 (MitoEM)](https://mitoem.grand-challenge.org/) | Human | Temporal lobe (L2) | (4096,4096,500) | 8×8×30 | 10552 |
| ssSEM | [Wei'20 (MitoEM)](https://mitoem.grand-challenge.org/) | Rat | V1 (L2/3) | (4096,4096,500) | 8×8×30 | 5446 |
| ssSEM | [Casser'20 (Kasthuri++)](https://sites.google.com/view/connectomics/)* | Mouse | Somatosensory (L4/5) | (700,791,80)×2 | 12×12×30 | 265 |
| FIB-SEM | └ Lucchi++ ([Lucchi'13](https://www.epfl.ch/labs/cvlab/data/data-em/))* | Mouse | Hippocampus | (512,768,82) | 10×10×10 | 70 |
| ssSEM | [Xiao'18](https://www.frontiersin.org/articles/10.3389/fnana.2018.00092/full) | Mouse | Auditory cortex | (2156,2104,31) | 8×8×50 | 468 |
| ssSEM | Han'24 | Mouse | Cerebellum (mossy fiber) | (1357,1411,331)×5 | 8×8×30 | 1144 |
| ssTEM | [Turner'22 (MICrONS, Pinky)](https://www.sciencedirect.com/science/article/pii/S0092867422001349)† | Mouse | V1 (Pinky + MICrONS1) | (1203,1282,83)×16 | 8×8×40 | 1420 |
| SBF-SEM | [Jiang'25](https://www.sciencedirect.com/science/article/pii/S2667237525000256) | Mouse | Brainstem | (1000,1000,100)×3 | 15×15×40 | 2057 |
| SBF-SEM | [Guay'21](https://leapmanlab.github.io/dense-cell/)* | Human | Platelet | (667,736,65)×3 | 10×10×50 | 279 |
| SBF-SEM | [Haberl'18 (CDeep3M)](https://doi.org/10.1038/s41592-018-0106-z)* | Mouse | Cerebellum (brain) | (256,256,15)×2 | 10×10×24 | 247 |

**Total: 212 volumes, 29,833 mitochondria.** Volumes are organized by imaging source; benchmarks are
split overlays in [`splits/`](splits/): `cellmap.json` (CellMap challenge over OpenOrganelle crops) and
`mitoem2.0.json` (MitoEM2.0 generalization suite). The OpenOrganelle `*_test0/_test1` volumes are
lab-annotated and belong to `mitoem2.0`.

## MitoEM2.0 benchmark

MitoEM2.0 (Liu et al., 2026) is a multi-domain generalization benchmark of 8 sub-datasets. Official
nnU-Net release: **[Zenodo 10.5281/zenodo.20417683](https://zenodo.org/records/20417683)**. Within
MitoVerse it is **not duplicated** — it is the split overlay [`splits/mitoem2.0.json`](splits/mitoem2.0.json)
over the source volumes below (muller21, openorganelle, han24, wei20, jiang25, kunduri22). EFI/DCI are
per-dataset difficulty metrics from the MitoEM2.0 paper; `†` redistribution caveat, `*` converted to instances.

| Name | Organism | Tissue | Cell type | Modality | Avg Shape (xyz) | Voxels | Split (tr/val/te) | # Mito | EFI/DCI | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| ME2-Beta   | Mouse      | Pancreas   | β-cell       | FIB-SEM | (874,669,979)   | 3961M | 3/1/3  | 1935 | 0.02/2.68 | Müller'21 † |
| ME2-Jurkat | Human      | Cell line  | Jurkat       | FIB-SEM | (1024,1024,256) | 805M  | 1/1/1  | 372  | 0.00/1.62 | Xu'21 * |
| ME2-Macro  | Human      | Cell line  | Macrophages  | FIB-SEM | (1024,1024,256) | 805M  | 1/1/1  | 497  | 0.01/1.38 | Xu'21 * |
| ME2-Mossy  | Mouse      | Cerebellum | Mossy Fiber  | ssSEM   | (1024,1024,500) | 3333M | 3/1/3  | 1099 | 0.10/4.13 | Han'24 |
| ME2-Pyra   | Human      | Cerebrum   | Pyramidal    | ssSEM   | (4096,4096,500) | 8388M | 3/1/11 | 45   | 0.96/0.53 | Wei'20 |
| ME2-Podo   | Mouse      | Kidney     | Podocyte     | FIB-SEM | (1024,1024,256) | 805M  | 1/1/1  | 1333 | 0.00/2.43 | CellMap'24 † |
| ME2-Sperm  | Drosophila | Gonad      | Spermatocyte | FIB-SEM | (1024,1024,341) | 1073M | 3/1/4  | 775  | 0.01/0.96 | Kunduri'22 † |
| ME2-Stem   | Mouse      | Brainstem  | Mix          | SBF-SEM | (1024,1024,100) | 300M  | 1/1/1  | 2058 | 0.09/0.80 | Jiang'25 |

## License

Per-volume; see each volume's `source` in `catalog.json`. Some sources (MICrONS †, CellMap) carry
attribution or redistribution caveats — check before reuse.

## Acknowledgement

- [Aswath et al. 2023. Segmentation in large-scale cellular electron microscopy with deep learning: A literature survey](https://arxiv.org/abs/2206.07171)
