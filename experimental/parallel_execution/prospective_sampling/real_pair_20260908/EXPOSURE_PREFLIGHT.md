# Planned exposure is not matched across P and O

Read-only calculation on 9 September 2026 from the verified embedding manifests,
training-label arrays, and current `batches_for` implementation. These are planned
counts, not completed training measurements or elapsed-time estimates.

| Per loss arm, 10 epochs | P | O |
|---|---:|---:|
| Optimizer steps | 19,720 | 35,400 |
| Row presentations, including negatives | 1,836,910 | 3,195,890 |

Both CE and conditional-focal heads would each incur these counts. The two
loss arms share identical batches within P and within O, but P versus O does
not have equal optimizer steps or exposure. Thus a fixed-epoch P/O result alone
cannot isolate sampling composition from training budget.

Calculation: for each class, positives are all rows for that arriving class.
The negative pool consists of other current-task rows plus at most 50 stored
rows per prior class. Selected negatives per epoch equal the smaller of this
pool and four times the positive count. The positive chunk width is
floor(384 / 5) = 76. Steps are ceil(positive count / 76) per epoch; row
presentations are positive count plus selected negatives per epoch.

The changed Task-0 normal count also changes the negatives available to the
other Task-0 head. Subsequent-task planned totals are equal across P/O, but
that does not prove equal sampled identities or learned states.

Required interpretation and next protocol:

1. Keep this existing fixed-epoch configuration only as a disclosed total
   pipeline comparison, not a matched-budget causal architecture claim.
2. A separate pre-specified matched-budget comparison must hold per-head
   optimizer steps, batch sizes, positive/negative presentations, initialization,
   and loss fixed while changing only the intended sampling policy. Record
   repeated versus unique rows; equal presentation counts do not imply equal
   data coverage. Do not tune budgets using held-out test performance.
3. Keep the within-policy CE versus focal comparison separate from P/O.
4. This L1 harness has no DP-Means router and cannot close the full OFRA
   mechanism, capacity or deployment-memory comparison.

Bindings: P manifest f2a1e3ff141bf44642207c92305c080ea453971510060ff4287f84fb476e641c;
O manifest 9db3e5c00cf87a3548e9ee54375cd78a21efd8627ec6ed36484592ab19540775;
runner 24ecba51264cb21bcd7c2601f1376634c528e3fdb1856722926700fc53952229.
