# Historical Windows checkpoints evaluated on Linux (493931)

Scope: CSE iCaRL seed10 and BiC-common seeds10–12, four checkpoints each.
No training, threshold search, rounding adjustment, or original-result mutation.
The same saved models and original prior-shift thresholds were evaluated on Linux.
This is a derived runtime comparison, not a replacement registered primary result.

| Method | Seed | Windows final1% BAcc (%) | Linux final1% BAcc (%) | Delta (pp) |
|---|---:|---:|---:|---:|
| icarl | 10 | 22.68540742915 | 22.68540211358 | -0.000005315565 |
| cer-bic-common | 10 | 31.93154244038 | 31.93154244038 | 0.000000000000 |
| cer-bic-common | 11 | 28.64984722153 | 28.64984722153 | 0.000000000000 |
| cer-bic-common | 12 | 47.86844969961 | 47.86844969961 | 0.000000000000 |

The iCaRL final1% change is one additional Normal false positive; exact BAcc delta
is -4/752507 percentage points. BiC final1% confusion matrices agree exactly for
all three seeds, but this does not imply equality for all stages or thresholds.
13 of64 TEST confusion matrices differ;18 of48 holdout FP counts differ. The
largest holdout difference is961 (BiC10/task0/5%). All exact cell-level differences
are retained in aggregate.json and the four result files. No tolerance was used
to relabel a discrepancy as exact agreement. These counts do not prove a cause.

Independent validation checks integer CM deltas, unchanged supports, original
reference/identity/checkpoint/tau bindings, recalls and rational final BAcc. It
does not independently rerun model predictions or establish raw-logit bitwise
identity. The independent forward functions were reused from their fixed source.
iCaRL uses default batch4096, calibrated batch2048 and original NME prototype
construction; BiC uses batch512. Platform versions remain in each result file.

Job493931 completed0:0:694s wall,1304.862s CPU,step RSS11238544KiB. This is job cost,
not an end-to-end workflow speedup. CPU-progress stopping was not triggered.
Both completed diagnostic stderr and Slurm stderr are empty. Frozen logs are
path-redacted gzip; aggregate.json retains original-byte SHA receipts. Export and
publication logs produced after this freeze remain working records, not archive
contents. Original data, checkpoints and registered results remain unchanged.

RQ1/RQ3, substantive innovation, the41-item requirements, manuscript current-source
visual QA and backup completion remain open. This group does not close them.
