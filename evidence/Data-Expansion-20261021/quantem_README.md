---
license: cc-by-4.0
pretty_name: QuantEM — base model data sources
tags:
  - electron-microscopy
  - provenance
  - attribution
  - pretraining-data
configs:
  - config_name: default
    data_files: sources.csv
---

# QuantEM — base model data sources

Every dataset in the corpus the **QuantEM ViT-B encoder** was pretrained on: public repository holdings, data contributed by external laboratories through the QuantEM outreach campaign, and in-house acquisitions.

Emitted verbatim from **Supplementary Table 2** of the QuantEM manuscript — 657 rows. Please cite the
original sources listed here alongside QuantEM; rows carry a DOI or repository URL where one
exists.

Related:

* [`ArrojoeDrigoLab/quantem`](https://huggingface.co/ArrojoeDrigoLab/quantem) — the released model weights
* [`ArrojoeDrigoLab/quantem-organelle-model-sources`](https://huggingface.co/datasets/ArrojoeDrigoLab/quantem-organelle-model-sources) — the other sources table

## Breakdown by grouping

Row counts over the table below.

| Grouping | Datasets |
|---|--:|
| Public | 502 |
| Outreach | 117 |
| Internal | 38 |


## Sources

First 60 of 657 rows — browse or download the full table in the Data Studio viewer above, or from `sources.csv`.

| Dataset | Experiment | Dataset DOI / URL | Grouping | 2D images | 3D acquisitions | Training tiles |
|---|---|---|---|---|---|---|
| Desmosomes in A431 Cells FIB-SEM Volume | A431 Cell Desmosome FIB-SEM | https://openorganelle.janelia.org/datasets/aic_desmosome-1 | Public | 0 | 1 | 108 |
| Desmosomes in A431 Cells - aic_desmosome-2 (FIB-SEM) | A431 Cell Desmosome-ER FIB-SEM | https://doi.org/10.25378/janelia.22670176 | Public | 0 | 1 | 582 |
| Desmosomes in A431 Cells - aic_desmosome-3 (FIB-SEM) | A431 Cell Desmosome-ER FIB-SEM | https://doi.org/10.25378/janelia.22675078 | Public | 0 | 1 | 716 |
| Data set 01 - Giardia lamblia adherent SBF-SEM series | Adherent Cell SBF-SEM | https://doi.org/10.5281/zenodo.11196419 | Public | 0 | 1 | 308 |
| Data set 02 - Giardia lamblia adherent SBF-SEM series | Adherent Cell SBF-SEM | https://doi.org/10.5281/zenodo.11197168 | Public | 0 | 1 | 320 |
| Data set 04 - HeLa adherent SBF-SEM series | Adherent Cell SBF-SEM | https://doi.org/10.5281/zenodo.11198586 | Public | 0 | 1 | 400 |
| Data set 05 - HeLa adherent SBF-SEM series | Adherent Cell SBF-SEM | https://doi.org/10.5281/zenodo.11198691 | Public | 0 | 1 | 400 |
| Data set 06 - HeLa adherent SBF-SEM series | Adherent Cell SBF-SEM | https://doi.org/10.5281/zenodo.11199320 | Public | 0 | 1 | 400 |
| Data set 07 - HeLa adherent SBF-SEM series | Adherent Cell SBF-SEM | https://doi.org/10.5281/zenodo.11199433 | Public | 0 | 1 | 352 |
| Data set 08 - HeLa adherent SBF-SEM series | Adherent Cell SBF-SEM | https://doi.org/10.5281/zenodo.11202406 | Public | 0 | 1 | 360 |
| Data set 09 - HeLa adherent SBF-SEM series | Adherent Cell SBF-SEM | https://doi.org/10.5281/zenodo.11203027 | Public | 0 | 2 | 640 |
| Data set 10 - HeLa adherent SBF-SEM series | Adherent Cell SBF-SEM | https://doi.org/10.5281/zenodo.11203782 | Public | 0 | 1 | 391 |
| An Aedes aegypti mosquito antennal lobe EM dataset | Aedes aegypti Antennal Lobe ssTEM | https://doi.org/10.60533/boss-2025-gcyr | Public | 0 | 1 | 400 |
| Cali 2018 aging mouse cortex FIB-SEM volumes | Aging Mouse Cortex FIB-SEM | https://doi.org/10.5061/dryad.bh78sn5 | Public | 0 | 6 | 51 |
| Horizontal cell connectivity in the anchovy retina - a 3D electron microscopic study - Scan 1 | Anchovy Retina Horizontal Cell Connectivity | https://doi.org/10.6019/EMPIAR-12585 | Public | 0 | 1 | 210 |
| Horizontal cell connectivity in the anchovy retina - a 3D electron microscopic study - Scan 2 | Anchovy Retina Horizontal Cell Connectivity | https://doi.org/10.6019/EMPIAR-12586 | Public | 0 | 1 | 400 |
| Horizontal cell connectivity in the anchovy retina - a 3D electron microscopic study - Scan 3 | Anchovy Retina Horizontal Cell Connectivity | https://doi.org/10.6019/EMPIAR-12587 | Public | 0 | 1 | 400 |
| Horizontal cell connectivity in the anchovy retina - a 3D electron microscopic study - Scan 4b | Anchovy Retina Horizontal Cell Connectivity | https://doi.org/10.6019/EMPIAR-12591 | Public | 0 | 1 | 288 |
| Horizontal cell connectivity in the anchovy retina - a 3D electron microscopic study - Scan 5 part 1 | Anchovy Retina Horizontal Cell Connectivity | https://doi.org/10.6019/EMPIAR-12592 | Public | 0 | 1 | 100 |
| Horizontal cell connectivity in the anchovy retina - a 3D electron microscopic study - Scan 5 part 2 | Anchovy Retina Horizontal Cell Connectivity | https://doi.org/10.6019/EMPIAR-12610 | Public | 0 | 1 | 78 |
| Horizontal cell connectivity in the anchovy retina - a 3D electron microscopic study - Scan 6 | Anchovy Retina Horizontal Cell Connectivity | https://doi.org/10.6019/EMPIAR-12609 | Public | 0 | 1 | 40 |
| Supporting data for "Socially regulated developmental plasticity in the color pattern of an anemonefish" | Anemonefish Skin TEM | https://doi.org/10.5281/zenodo.16400562 | Public | 31 | 0 | 324 |
| SBF-SEM micrographs of A. algerae microsporidia spores, 45 min germination | Anncaliia algerae Spore Germination SBF-SEM | https://doi.org/10.6019/EMPIAR-11368 | Public | 0 | 1 | 400 |
| SBF-SEM micrographs of A. algerae microsporidia spores, 5 min germination | Anncaliia algerae Spore Germination SBF-SEM | https://doi.org/10.6019/EMPIAR-11367 | Public | 0 | 3 | 1200 |
| SBF-SEM micrographs of A. algerae spores, Ungerminated | Anncaliia algerae Ungerminated Spores SBF-SEM | https://doi.org/10.6019/EMPIAR-11369 | Public | 0 | 1 | 400 |
| Dataset of microscopy (Confocal and electron) - Outer epidermal edges mediate cell-cell adhesion for tissue integrity in plants | Arabidopsis Hypocotyl Epidermis Cell Adhesion TEM | https://doi.org/10.5281/zenodo.15125559 | Public | 476 | 0 | 1895 |
| Arabidopsis Root Phloem SBF-SEM Volumes | Arabidopsis Root Phloem SBF-SEM | https://doi.org/10.6019/EMPIAR-10442 | Public | 0 | 4 | 464 |
| Ultrastructure description of the spermatozoa of the Amazon bonytongue Arapaima gigas (Schinz, 1822) | Arapaima gigas Spermatozoa | https://doi.org/10.5281/zenodo.8359643 | Public | 20 | 0 | 19 |
| BANC Adult Drosophila CNS EM Volume | BANC Adult Drosophila CNS Connectome ssTEM | https://doi.org/10.60533/boss-2025-941r | Public | 0 | 1 | 220 |
| Volume electron microscopy reveals bacterial endosymbiosis within host mitochondria | Bacterial Endosymbiosis in Host Mitochondria vEM | https://www.ebi.ac.uk/biostudies/studies/S-BIAD2413 | Public | 0 | 13 | 690 |
| Borrelia burgdorferi between neighbouring endothelial cells in a lymphatic capillary | Borrelia burgdorferi in Mouse Skin Vasculature SBF-SEM | https://doi.org/10.6084/m9.figshare.30043093.v1 | Public | 0 | 1 | 51 |
| Borrelia burgdorferi engulfed by a neutrophil | Borrelia burgdorferi in Mouse Skin Vasculature SBF-SEM | https://doi.org/10.6084/m9.figshare.30042721.v1 | Public | 0 | 1 | 100 |
| Borrelia burgdorferi in contact with the pericyte projection and the fibroblast | Borrelia burgdorferi in Mouse Skin Vasculature SBF-SEM | https://doi.org/10.6084/m9.figshare.30042805.v1 | Public | 0 | 1 | 137 |
| Borrelia burgdorferi partially enclosed within a lymphatic capillary of an endothelial cell | Borrelia burgdorferi in Mouse Skin Vasculature SBF-SEM | https://doi.org/10.6084/m9.figshare.30042739.v1 | Public | 0 | 1 | 19 |
| Borrelia burgdorferi under a pericyte in dorsal skin | Borrelia burgdorferi in Mouse Skin Vasculature SBF-SEM | https://doi.org/10.6084/m9.figshare.30041983.v2 | Public | 0 | 1 | 54 |
| Borrelia burgdorferi within a collagen bundle in mouse ear | Borrelia burgdorferi in Mouse Skin Vasculature SBF-SEM | https://doi.org/10.6084/m9.figshare.30042088.v1 | Public | 0 | 1 | 18 |
| Immune cells migrating from the vein in dorsal skin | Borrelia burgdorferi in Mouse Skin Vasculature SBF-SEM | https://doi.org/10.6084/m9.figshare.30042151.v1 | Public | 0 | 1 | 400 |
| Multisite adhesion of Borrelia burgdorferi to the blood capillary surface in dorsal skin | Borrelia burgdorferi in Mouse Skin Vasculature SBF-SEM | https://doi.org/10.6084/m9.figshare.30042646.v1 | Public | 0 | 1 | 80 |
| Pericyte targeting by Borrelia burgdorferi | Borrelia burgdorferi in Mouse Skin Vasculature SBF-SEM | https://doi.org/10.6084/m9.figshare.30042550.v1 | Public | 0 | 1 | 64 |
| Transcellular migration of Borrelia burgdorferi through a lymphatic capillary endothelial cell | Borrelia burgdorferi in Mouse Skin Vasculature SBF-SEM | https://doi.org/10.6084/m9.figshare.30043204.v1 | Public | 0 | 1 | 19 |
| C. elegans Anterior Pharynx ssTEM | C. elegans Anterior Pharynx ssTEM | https://doi.org/10.60533/BOSS-2015-FBTX | Public | 0 | 1 | 400 |
| Witvliet 2020 Dataset 1 L1 C. elegans Brain | C. elegans Brain Postnatal Development | https://doi.org/10.60533/BOSS-2020-FQI7 | Public | 0 | 1 | 400 |
| Witvliet 2020 Dataset 2 C. elegans Brain | C. elegans Brain Postnatal Development | https://doi.org/10.60533/BOSS-2020-FQI7 | Public | 0 | 1 | 400 |
| Witvliet 2020 Dataset 3 L1 C. elegans Brain | C. elegans Brain Postnatal Development | https://doi.org/10.60533/BOSS-2020-FQI7 | Public | 0 | 1 | 400 |
| Witvliet 2020 Dataset 4 C. elegans Brain | C. elegans Brain Postnatal Development | https://doi.org/10.60533/BOSS-2020-FQI7 | Public | 0 | 1 | 400 |
| Witvliet 2020 Dataset 5 L2 C. elegans Brain | C. elegans Brain Postnatal Development | https://doi.org/10.60533/BOSS-2020-FQI7 | Public | 0 | 1 | 400 |
| Witvliet 2020 Dataset 6 L3 C. elegans Brain | C. elegans Brain Postnatal Development | https://doi.org/10.60533/BOSS-2020-FQI7 | Public | 0 | 1 | 400 |
| Witvliet 2020 Dataset 7 Young Adult C. elegans Brain | C. elegans Brain Postnatal Development | https://doi.org/10.60533/BOSS-2020-FQI7 | Public | 0 | 1 | 400 |
| Witvliet 2020 Dataset 8 Young Adult C. elegans Brain | C. elegans Brain Postnatal Development | https://doi.org/10.60533/BOSS-2020-FQI7 | Public | 0 | 1 | 400 |
| Entire C. elegans dauer larva (e1370) volume | C. elegans Dauer Larva FIB-SEM | https://doi.org/10.25378/janelia.20376591 | Public | 0 | 1 | 219 |
| Yim Choe Bae 2024 C. elegans Dauer Nerve Ring TEM | C. elegans Dauer Nerve Ring TEM | https://doi.org/10.60533/BOSS-2023-RTPH | Public | 0 | 1 | 400 |
| White et al. 1986 C. elegans N2U Nerve Ring | C. elegans Nervous System Connectome TEM | https://doi.org/10.60533/boss-1986-745j | Public | 0 | 1 | 239 |
| Overexpression of an ALS-associated FUS mutation in C. elegans disrupts NMJ morphology and leads to defective neuromuscular transmission | C. elegans Neuromuscular Junction ALS-FUS ET | https://doi.org/10.5061/dryad.9ghx3ffg0 | Public | 0 | 6 | 37 |
| COS-7 cell overexpressing ER and mitochondria markers for CLEM | COS-7 Cell ER & Mitochondria FIB-SEM | https://doi.org/10.25378/janelia.13123385 | Public | 0 | 1 | 71 |
| FIB-SEM Dataset (Obara + Nixon-Abell et al., 2023) | COS-7 Cell ER-Mitochondria Contact Site FIB-SEM | https://doi.org/10.25378/janelia.24502954.v1 | Public | 0 | 1 | 332 |
| COS-7 Cell FIB-SEM Volume | COS-7 Cell FIB-SEM | https://doi.org/10.25378/janelia.24943716 | Public | 0 | 1 | 358 |
| Interphase COS-7 Cell FIB-SEM Volume | COS-7 Cell FIB-SEM (Xu 2021) | https://doi.org/10.25378/janelia.24944559 | Public | 0 | 1 | 208 |
| C. elegans motor circuit serial-section EM volumes | Caenorhabditis elegans Motor Circuit SEM | https://doi.org/10.60533/BOSS-2022-1JI1 | Public | 0 | 5 | 1592 |
| Quantitative subcellular reconstruction reveals a lipid mediated inter-organelle biogenesis network | Cal51 Cell Organelle Biogenesis Knockout FIB-SEM | https://doi.org/10.6019/EMPIAR-11695 | Public | 0 | 6 | 435 |
| SBF-SEM datasets related to the manuscript "Trans-cellular tunnels induced by the fungal pathogen Candida albicans facilitate invasion through successive epithelial cells without host damage" by Lachat et al, 2022. | Candida albicans Epithelial Invasion SBF-SEM | https://doi.org/10.5281/zenodo.5776104 | Public | 0 | 11 | 450 |


657 datasets. Grouping: Public = external public repository; Outreach = contributed by an external laboratory through the outreach campaign; Internal = own group or close collaborators. Internal datasets are named as in their BioImage Archive depositions: treatment and condition labels are deliberately non-identifying, and the same label within one experiment denotes the same arm.

## Citation

Acree *et al.*, *QuantEM: An optimized platform of vision transformer-based models for
segmentation and analysis of electron microscopy data.* Citation details on publication.
