# MitoVerse — Open Questions / TODO

Unsolved items as of the current build (212 volumes, 13 datasets). Grouped by theme.

## Data integrity

- **kunduri22 gt not connected.** `train01`/`train02` are separate stores; an instance crossing that
  z-cut gets a different id in each. Ingested count is 621 (218+172+231) vs the paper's **775** — likely
  this split + double/under-count. Connect the gt across train01/train02 (and decide whether to then
  merge into one 512z volume).
- **ME2-Pyra # Mito = 45?** The MitoEM2.0 table lists ME2-Pyra at 45 mitochondria, but `wei20/mitoEM-H`
  has 10,552. Confirm the ME2-Pyra figure (is it per held-out crop, or an error?).
- **wei20 test withheld.** Only train+val (500z) ingested for MitoEM-H/R, to keep the grand-challenge
  test hidden. Confirm this stays the policy for any public release.

## Missing metadata

- **turner22 voxel** = 8×8×40 (MICrONS default) — verify the Pinky-specific resolution; the 5
  `microns1_vol*` crops may differ.
- ~~conrad23 voxel/species~~ — DONE (filled per-volume from the MitoNet paper; FIB-SEM isotropic 12–24 nm).
- **Per-volume license / redistribution rights** — especially unpublished OpenOrganelle crops, MICrONS
  (†), and internal kunduri22. Clear before any HuggingFace push.

## Not-yet-ingested data

- **MICrONS1 full volume** (`mito/lichtman/microns1_*`): 16 tiles of (750,1024,1024). Decide: stitch the
  4×4 grid into one volume and add to `turner22`, or skip. (Only the 5 small `microns1_vol*` crops are in.)
- **Peng's other 3D sets**: `cerebellum` (4 quadrant vols, instance), `wilson19`, `zhu22` — voxel/metadata
  TBD. `Boston3Mouse` is image-only (no labels) → skip unless labels appear.
- **OME-Zarr refined labels**: only ME2-Sperm was taken from `MitoEM2.0_OMEZarr`. The other 7 ME2 OME-Zarr
  sub-datasets may carry refined labels vs the MitoLE-sourced volumes — decide whether to re-ingest from
  the OME-Zarr (needs the zarr-v3 path).

## Candidate external datasets to acquire (mito · 3D)

Sourced from the torch-em EM dataset tracker (full sheet cached at
`<pytc>/lib/mitoverse/tmp/em_datasets.csv`). Filtered to mitochondria + volumetric + usable labels.

### Ready — instance mito, 3D, not yet in MitoVerse
- **VNC** (Gerhard'13) — ssTEM, Drosophila larval ventral nerve cord; instance mito. Small, easy add.
  Loader: https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/electron_microscopy/vnc.py
  (NB: different from our COSEM `jrc_fly-vnc-1`.)
- **MBLiver** (Nat Commun 2024) — FIB-SEM, mouse liver; instance mito. ~2.1 TB native → take a 16 nm
  ROI / downsample. https://doi.org/10.6019/EMPIAR-12017 · paper https://www.nature.com/articles/s41467-024-48272-7
- **MICrONS minnie (cortical mm³)** — ssTEM, mouse cortex; instance mito (+ neurites). Complements our
  Pinky (`turner22`). https://www.microns-explorer.org/cortical-mm3#voxel-segmentation · https://zenodo.org/records/5760218
- **MitoNet_OpenOrganelle_Mouse_Kidney** — auto MitoNet instance mito, **not proofread** → low priority.
  https://figshare.com/articles/dataset/MitoNet_automatic_instance_segmentation_of_mitochondria_in_the_OpenOrganelle_Mouse_Kidney_dataset/20749729/2

### Needs binary→instance conversion (semantic mito source)
- **ASEM** — FIB-SEM whole-cell; semantic mito + golgi/ER/nucleus → cc3d to instances.
  https://doi.org/10.1083/jcb.202208005 · loader https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/electron_microscopy/asem.py
- **DeepContact** — semantic mito/ER/lipid (figshare server was flaky).
  https://figshare.com/articles/dataset/DeepContact_Training_Data/19898404/1
- **Human Organoids** — FIB-SEM, semantic organelles incl. mito; EMPIAR-11380.
  https://www.ebi.ac.uk/empiar/EMPIAR-11380/
- **ProbTEM** — semantic mito but **2D TEM** → skip for the 3D benchmark.
  https://yoonlab.unist.ac.kr/index.php/research/mitochondria-tem-dataset/

### Extend what we already have
- **CellMap** — more annotated `jrc_*` crops (being extended upstream).
  https://doi.org/10.25378/janelia.c.7456966 · loader https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/electron_microscopy/cellmap.py
- **OpenOrganelle** — additional `jrc_*` volumes with mito (e.g. mb-liver, more `mus-liver-zon`/`kidney`).
  https://openorganelle.janelia.org/datasets

### Already in MitoVerse (no download)
Kasthuri & Lucchi (`casser20`) · MitoEM (`wei20`) · UroCell (`mekuc20`) · BetaSeg (`muller21`) ·
DenseCell/Guay (`guay21`) · Haberl (`haberl18`) · Xiao'18 · Jiang'25 · CEM/MitoNet-3D-benchmark
(`conrad23`) · OpenOrganelle/CellMap subset (`openorganelle`) · Pinky/MICrONS (`turner22`).

## Benchmark splits

- **cellmap.json**: all 147 crops currently marked `train`. Apply the official CellMap challenge
  train/val/test split.
- **mitoem.json**: not written yet — classic MitoEM (wei20) split (region-based).
- **Within-volume regions (PyTC feature).** ME2-Pyra (wei20 crops) and classic MitoEM splits need PyTC to
  support coordinate regions as train/val/test within one volume. Until then they reference whole volumes.
- **Self-annotated OpenOrganelle volumes not in any split** (e.g. some cardiac/mus-liver `test` crops) have
  ambiguous annotation provenance — decide where they belong.

## Tooling / publishing

- **zarr v3 env.** OME-Zarr ingestion currently uses a throwaway pip venv. Stand up a persistent conda env
  (`conda create -n zarrv3 -c conda-forge "zarr>=3" ...`) for repeatable v3 reads.
- **HuggingFace push (pending, human auth):** (a) MitoVerse data + splits + card; (b) deprecation banners
  on `pytc/MitoEM` and `pytc/MitoEM2.0` (committed locally at `weidf/lib/hf_legacy_cards/`). Also fix the
  malformed YAML frontmatter on the `pytc/MitoEM` card (line 1 sits above the opening `---`).

## Definitions

- **EFI / DCI** difficulty metrics in the MitoEM2.0 table are quoted but not defined here — add their
  definitions (or a paper reference) so the numbers are interpretable.
