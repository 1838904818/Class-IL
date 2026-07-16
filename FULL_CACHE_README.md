# Uncapped streaming feature cache

`fullcache` is an independent preprocessing path for the seven OFRA validation
datasets. It does not import or modify the frozen loaders, `ofra.py`, or
`multi_seed_ofra.py`, and it does not train a model.

## Data contract

- CSV/text inputs are read in chunks; no dataset-wide DataFrame is built.
- No class-level or benign-row cap is applied.
- Every output array is little-endian `float32` and is stored at
  `<output>/<dataset>/<split>/<class>/part-*.npy`.
- Every raw file and output shard has a SHA256 entry in the dataset manifest.
- The manifest records byte length and SHA256 for every `fullcache/*.py` source
  file, a canonical builder-source hash, and a canonical manifest hash.
- `feature_schema.json` fixes column order, class order, cleaning rules, and
  train-only categorical vocabularies.
- `row_counts.json` records all split/class counts and every dropped-row reason.
- `streaming_manifest.json` is the fail-closed manifest consumed directly by
  `streaming_full`; it contains explicit class ids, task schedule, semantic
  profile, and non-empty train/test shard lists per class. NIDS manifests use
  explicit `normal_class_id=0`; generic application-classification manifests
  require `normal_class_id=null`.
- CIC-IDS-2017, CIC-IDS-2018, and CIC-IoT-2023 use an 80/20 deterministic
  model-feature row hash. Identical cleaned feature rows cannot cross splits.
- NF-ToN-IoT-v2, NSL-KDD, and UNSW-NB15 preserve their official train/test
  files. NSL/UNSW one-hot vocabularies are learned from train only; a category
  seen only in test produces an all-zero block and is counted in the manifest.
- Those three official-split datasets also receive an exact cleaned-feature
  overlap audit. An external SQLite table is keyed by the complete float32 row
  bytes, so SHA/64-bit collisions are not used to decide equality. The report
  records unique overlapping rows and train/test rows involved; it never drops
  or reallocates an official row.
- CIC-IDS-2018 must contain the official ten files. The 84-column
  `Thuesday-20-02-2018` file loses only its four extra identifiers (`Flow ID`,
  `Src IP`, `Src Port`, `Dst IP`); shared `Dst Port` remains a model feature.
- MalayaNetwork_GT is pinned to revision
  `384a59278f98490ee6e93aae017e748078d29b6a`. Labels come from its nested
  application directories. The build verifies the frozen 31-file hash
  contract, drops IP addresses, ports, and timestamp, retains 77 numeric
  features, and holds out one complete capture per class. Its exact cross-split
  row overlap is audited but never silently deleted or reallocated.

The default CLI rejects incomplete mirrors and unknown labels. Relaxation flags
exist for diagnostics and are always recorded in the manifest.

## Obtain the pinned Malaya CSV snapshot

Install Git LFS, then run the following from the repository root. Smudge is
disabled during checkout so the PCAP objects are not downloaded; the explicit
pull fetches only the public flow CSVs.

```powershell
git lfs install
New-Item -ItemType Directory -Force .data | Out-Null
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone --no-checkout `
  https://huggingface.co/datasets/Afifhaziq/MalayaNetwork_GT `
  .data/malaya-network-gt
git -C .data/malaya-network-gt checkout --detach `
  384a59278f98490ee6e93aae017e748078d29b6a
git -C .data/malaya-network-gt lfs pull `
  --include="csv_output/**" --exclude="PCAP/**"
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
```

The expected layout is `.data/malaya-network-gt/csv_output/`. The fixed split
and all 31 CSV hashes are versioned in
`fullcache/contracts/malaya_network_gt_capture_split.json`; no raw data is
stored in this repository.

## Build command

Run from the repository root:

```powershell
python -m fullcache `
  --data-root .data `
  --output-root .artifacts/fullcache `
  --dataset malaya-network-gt `
  --chunk-rows 100000 `
  --overlap-work-directory .artifacts/overlap-work
```

Build one or several datasets by repeating `--dataset`. Existing completed
dataset directories are protected unless `--overwrite` is supplied. A staging
directory is promoted only after all sidecars and hashes are complete.

The convenience runner verifies the cache, creates the duplicate-excluded
sensitivity view, and evaluates both classifier sizes. It defaults to seeds
`1, 2, 3, 4, 42` and writes only below `.artifacts/` unless paths are
overridden:

```powershell
.\run_malaya_validation.ps1 -DataRoot .data
```

## Synthetic smoke tests

The tests create only temporary miniature files and never open the real raw
datasets:

```powershell
python -m unittest discover -s tests -p "test_fullcache_smoke.py" -v
```

Verify hashes, shapes, finite values, row accounting, CIC split assignments,
and `streaming_full` compatibility without loading a complete dataset at once:

```powershell
python -m fullcache.verify `
  --cache-root .artifacts/fullcache `
  --dataset malaya-network-gt `
  --recompute-overlap `
  --overlap-work-directory .artifacts/overlap-work
```

The full verifier re-reads raw files for SHA256. For a quick shard-only queue
gate, add `--skip-raw-hashes`; retain the full verification for reportable runs.
Add `--recompute-overlap` to independently rebuild and compare the exact
official-split overlap result. The standalone bounded-memory audit command is:

```powershell
python -m fullcache.overlap `
  --cache-root .artifacts/fullcache `
  --dataset malaya-network-gt `
  --work-directory .artifacts/overlap-work
```

Each validated dataset can then be passed to the independent runner:

```powershell
python -m streaming_full `
  --manifest .artifacts/fullcache/malaya-network-gt/streaming_manifest.json `
  --evaluation-view duplicate_excluded=.artifacts/fullcache/malaya-network-gt/evaluation_views/duplicate_excluded/evaluation_view_manifest.json `
  --config-json configs/malaya_mlp_128x2.json `
  --seeds 1 2 3 4 42 `
  --output-dir .artifacts/results/malaya_mlp_128x2
```
