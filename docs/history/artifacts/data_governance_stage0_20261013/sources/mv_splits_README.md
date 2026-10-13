# Benchmark splits

A *benchmark* is a JSON file here that names which MitoVerse volumes it uses and their role. The
volumes themselves (`../data/<dataset>/<vol>.zarr`) are **original chunks** — never physically cut
into train/val/test. A split file is just an overlay, so the same data backs many benchmarks.

## Schema

```jsonc
{
  "name": "guay21",
  "arrays": {"image": "img", "label": "mito"},   // zarr sub-arrays PyTC reads
  "volumes": [
    {"id": "guay21_vol0", "zarr": "data/guay21/vol0.zarr", "split": "train"},
    {"id": "guay21_vol2", "zarr": "data/guay21/vol2.zarr", "split": "test"}
  ]
}
```

`split` ∈ {train, val, test}. Whole-volume assignment works today.

### Regions (coming, via PyTC)

When a benchmark splits *within* a volume (e.g. MitoEM's train = z 0–400, test = z 500–1000), the
volume stays one store and the ranges go in a `regions` field — honored once PyTorchConnectomics
supports region specs:

```jsonc
{"id": "wei20_mitoEM-H", "zarr": "data/wei20/mitoEM-H.zarr",
 "regions": {"train": [[0,400],[0,4096],[0,4096]], "test": [[500,1000],[0,4096],[0,4096]]}}
```

## Use with PyTorchConnectomics

```bash
python <pytc>/lib/mitoverse/scripts/to_pytc.py splits/guay21.json
# emits the cfg.data.{train,val,test}.{image,label} block of *.zarr/img + *.zarr/mito paths
```

## Files

- `guay21.json` — runnable example (3 ingested volumes).
- `mitoem2.0.json` — the MitoEM2.0 8-sub-dataset benchmark as an overlay over existing source volumes
  (muller21, openorganelle, han24, wei20, jiang25, kunduri22). ME2-Pyra = masked crop/regions of wei20.
- `cellmap.json` — CellMap challenge over the OpenOrganelle `_crop` volumes.
- `mitoem.json` — TODO: classic MitoEM (wei20) train/val/test split (region-based, pending PyTC regions).
