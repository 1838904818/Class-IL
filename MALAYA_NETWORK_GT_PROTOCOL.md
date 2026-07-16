# MalayaNetwork_GT external validation protocol

## Scope

MalayaNetwork_GT is used as an external application-traffic classification
benchmark. Its ten labels identify applications or services; they do not
identify benign and malicious traffic. Results from this dataset therefore
test the generic class-incremental mechanism and non-IID capture transfer, not
intrusion-detection performance.

The source is pinned to Hugging Face revision
`384a59278f98490ee6e93aae017e748078d29b6a` under CC BY 4.0. The experiment
uses the 31 public flow CSV files. PCAP files are not needed for this protocol
and are not copied into the experiment cache.

## Labels and tasks

Directory names provide the labels. Two upstream spellings are normalized:
`Bittorent` to `BitTorrent`, and `Teamviewer` to `TeamViewer`.

The fixed class order is:

1. BitTorrent
2. ChromeRDP
3. Discord
4. EA Origin
5. Microsoft Teams
6. Slack
7. Steam
8. TeamViewer
9. Webex
10. Zoom

The five tasks introduce consecutive class pairs: `(0, 1)`, `(2, 3)`,
`(4, 5)`, `(6, 7)`, and `(8, 9)`.

## Feature contract

Each source CSV has 82 columns. The model excludes `src_ip`, `dst_ip`,
`src_port`, `dst_port`, and `timestamp` to reduce direct endpoint, port, and
capture-time shortcuts. The remaining 77 numeric flow features, including
`protocol`, are stored as little-endian float32 shards. Feature order and all
source and shard hashes are fixed by the cache manifests.

## Split contract

The Hugging Face configurations are class partitions, not train/test splits.
Random row splitting is also unsuitable because flows from the same capture
would appear on both sides. The primary holdout keeps one complete capture per
class for testing, choosing the capture whose row count is closest to 20% of
that class (lexicographic path order breaks ties). This produces 36,372 train
rows and 10,370 test rows. The split is fixed independently of model seeds.

The main limitation is the small TeamViewer holdout (54 flows). Final reporting
should therefore supplement the fixed holdout with grouped cross-validation or
repeated capture-level folds; five training seeds alone do not remove split
uncertainty.

An independent float32 row audit found 74 unique 77-feature vectors on both
sides of this capture split. They account for 10,782 training rows and 3,780
test rows (36.45% of the fixed holdout), because a small number of repeated
flow-statistic patterns occur many times. The cache records the exact overlap
without changing the fixed split. A `duplicate_excluded` evaluation view removes
the 3,780 affected test rows for sensitivity analysis while retaining all
training rows; the official and duplicate-excluded results must be shown
together.

The exact assignment and all 31 source CSV hashes are stored in
`fullcache/contracts/malaya_network_gt_capture_split.json`. Cache construction
verifies this bundled contract before preprocessing any row and copies the
verified contract into the completed cache for independent review.

## Semantics and metrics

The streaming manifest declares:

```json
{
  "problem_type": "application_classification",
  "task_semantics": "class_incremental",
  "metric_profile": "generic_multiclass",
  "normal_class_id": null
}
```

Report accuracy, macro-F1, balanced accuracy, average task accuracy, and
average forgetting. Benign false-positive rate and attack-detection recall are
undefined for these labels and must not be generated or compared with the NIDS
datasets.

## Model-size comparison

Dataset addition and classifier scaling are separate experimental factors.
The first validation compares the existing MLP encoder (`d_model=128`, two
layers) with a larger MLP encoder (`d_model=256`, four layers) under the same
data manifest, task order, optimization settings, router cap, and seed. For 77
input features, the encoder parameter count increases from 26,496 to 217,344.

Raw rows, IP addresses, source CSVs, and PCAPs remain outside the repository.
Only aggregate metrics, plots, protocol hashes, source revision, and
configuration metadata are suitable for external reporting.

## Source

- Dataset: <https://huggingface.co/datasets/Afifhaziq/MalayaNetwork_GT>
- Pinned revision: <https://huggingface.co/datasets/Afifhaziq/MalayaNetwork_GT/tree/384a59278f98490ee6e93aae017e748078d29b6a>
