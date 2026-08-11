# Dataset access and preprocessing scope

The current study evaluates each dataset in a separate class-incremental run.
Rows from different datasets are never pooled into one training stream. The
shared model architecture is reused, while the loader, label map, feature
schema, split contract, and task schedule are dataset-specific.

## Current five-dataset suite

| Dataset | Data used by this repository | Source |
|---|---|---|
| NSL-KDD | `KDDTrain+.txt`, `KDDTest+.txt` | https://www.unb.ca/cic/datasets/nsl.html |
| UNSW-NB15 | official training and testing CSVs | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| CIC-IDS-2017 | eight labelled flow CSVs | https://www.unb.ca/cic/datasets/ids-2017.html |
| CSE-CIC-IDS2018 | ten labelled traffic CSVs from the AWS public bucket | https://registry.opendata.aws/cse-cic-ids2018/ |
| MalayaNetwork_GT | 31 public derived-flow CSVs at revision `384a59278f98490ee6e93aae017e748078d29b6a` | https://huggingface.co/datasets/Afifhaziq/MalayaNetwork_GT/tree/384a59278f98490ee6e93aae017e748078d29b6a |

The public Kaggle distribution is available at
https://www.kaggle.com/datasets/wuliqiang/leon-nids-classil. The distribution
records the exact upstream source, byte size, and SHA-256 digest of the large
CSE-CIC-IDS2018 Tuesday file.

## Preprocessing boundary

`fullcache/specs.py` is the executable preprocessing contract. It fixes each
dataset's expected filenames, label normalization, class order, excluded
identifier columns, split strategy, feature count, and task schedule.

- NSL-KDD and UNSW-NB15 retain their official train/test files. Categorical
  vocabularies are fitted on training data only.
- CIC-IDS-2017 and CSE-CIC-IDS2018 use a deterministic 80/20 split grouped by
  cleaned model-feature bytes, preventing an identical cleaned row from being
  assigned to both sides.
- MalayaNetwork_GT uses a frozen capture-level holdout. IP addresses, ports,
  and timestamps are excluded, leaving 77 numerical flow features. Its labels
  are applications, not benign/attack classes, so NIDS-specific detection
  metrics are not reported for this dataset.

Every generated cache records raw-file and shard SHA-256 hashes, feature and
class schemas, row accounting, and the task stream. The strict verifier rejects
missing files, unknown labels, non-finite values, and hash mismatches.

## Licence and attribution

The datasets retain their original ownership and terms. MalayaNetwork_GT is
published under CC BY 4.0 and must be attributed to Azizi Ariffin and Afif
Haris. The Kaggle distribution includes source links and attribution; it does
not alter the upstream licences.
