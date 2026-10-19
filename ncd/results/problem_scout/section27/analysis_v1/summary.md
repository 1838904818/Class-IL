# Section27 calibrated operating-point analysis

A logically determined pass/fail is kept separate from evidence completeness; missing endpoints remain explicitly missing.

α is selected on the common permanent Normal holdout; actual test FPR is reported, not forced to α. Values are five-seed mean±sample SD (%). Incomplete groups remain pending.

| Dataset | Method | n/5 | Calibration FPR | Test FPR | ADR | Macro-F1 | Attack guard |
|---|---|---:|---:|---:|---:|---:|---:|
| NSL-KDD | Replay | 5/5 | 1.990±0.000 | 4.080±2.162 | 65.386±5.301 | 46.829±4.930 | 34.830±5.777 |
| NSL-KDD | DER++ | 5/5 | 1.990±0.000 | 2.506±0.452 | 65.275±2.730 | 50.460±3.152 | 38.268±3.773 |
| NSL-KDD | WACB | 5/5 | 1.990±0.000 | 3.623±0.871 | 66.829±3.773 | 57.029±3.910 | 46.770±6.493 |
| NSL-KDD | Freq | 5/5 | 1.990±0.000 | 4.928±2.423 | 69.667±2.534 | 56.843±4.322 | 53.503±3.495 |
| NSL-KDD | Replay+LA | 5/5 | 1.990±0.000 | 4.080±2.162 | 65.386±5.301 | 46.829±4.930 | 34.830±5.777 |
| NSL-KDD | AA-CIL | 5/5 | 1.990±0.000 | 6.146±1.794 | 62.838±3.460 | 55.664±3.364 | 49.364±1.981 |
| NSL-KDD | TS-mem | 5/5 | 1.990±0.000 | 4.409±2.117 | 63.192±4.660 | 51.441±2.160 | 45.622±1.921 |
| NSL-KDD | AA-noAttMem | 5/5 | 1.990±0.000 | 4.317±1.824 | 60.295±3.157 | 56.672±3.337 | 47.294±3.049 |
| NSL-KDD | FARQ (supplementary) | 5/5 | 1.990±0.000 | 3.680±0.989 | 67.448±2.773 | 58.860±5.788 | 48.699±4.938 |
| NSL-KDD | Joint (offline) | 5/5 | 1.990±0.000 | 6.786±1.002 | 66.633±1.795 | 59.371±2.898 | 50.009±2.750 |
| UNSW-NB15 | Replay | 5/5 | 1.993±0.000 | 1.784±0.201 | 38.521±3.773 | 26.266±0.855 | 17.037±0.984 |
| UNSW-NB15 | DER++ | 5/5 | 1.993±0.000 | 1.985±0.268 | 18.477±4.124 | 14.470±1.436 | 5.146±1.159 |
| UNSW-NB15 | WACB | 5/5 | 1.993±0.000 | 1.931±0.239 | 48.970±4.140 | 32.807±1.176 | 22.389±1.464 |
| UNSW-NB15 | Freq | 5/5 | 1.993±0.000 | 2.009±0.264 | 50.419±2.932 | 31.224±1.729 | 21.251±1.542 |
| UNSW-NB15 | Replay+LA | 5/5 | 1.993±0.000 | 1.784±0.201 | 38.521±3.773 | 26.266±0.855 | 17.037±0.984 |
| UNSW-NB15 | AA-CIL | 5/5 | 1.993±0.000 | 1.922±0.153 | 69.090±0.928 | 41.178±0.277 | 50.269±1.181 |
| UNSW-NB15 | TS-mem | 0/5 | pending | pending | pending | pending | pending |
| UNSW-NB15 | AA-noAttMem | 0/5 | pending | pending | pending | pending | pending |
| UNSW-NB15 | FARQ (supplementary) | 5/5 | 1.993±0.000 | 1.838±0.184 | 45.099±4.200 | 30.247±1.676 | 20.095±1.619 |
| UNSW-NB15 | Joint (offline) | 5/5 | 1.993±0.000 | 1.928±0.152 | 67.109±1.453 | 37.219±1.000 | 28.417±0.759 |
| CIC-IDS-2017 | Replay | 5/5 | 2.000±0.000 | 1.950±0.059 | 78.889±5.634 | 51.515±4.018 | 47.868±6.547 |
| CIC-IDS-2017 | DER++ | 5/5 | 2.000±0.000 | 1.917±0.067 | 89.290±3.012 | 48.852±3.191 | 47.683±3.626 |
| CIC-IDS-2017 | WACB | 5/5 | 2.000±0.000 | 1.917±0.055 | 88.915±2.301 | 57.844±1.126 | 59.903±0.675 |
| CIC-IDS-2017 | Freq | 5/5 | 2.000±0.000 | 1.930±0.024 | 88.548±2.324 | 57.972±4.333 | 61.812±5.970 |
| CIC-IDS-2017 | Replay+LA | 5/5 | 2.000±0.000 | 1.950±0.059 | 78.889±5.634 | 51.515±4.018 | 47.868±6.547 |
| CIC-IDS-2017 | AA-CIL | 0/5 | pending | pending | pending | pending | pending |
| CIC-IDS-2017 | TS-mem | 0/5 | pending | pending | pending | pending | pending |
| CIC-IDS-2017 | AA-noAttMem | 0/5 | pending | pending | pending | pending | pending |
| CIC-IDS-2017 | FARQ (supplementary) | 5/5 | 2.000±0.000 | 1.936±0.040 | 85.395±5.483 | 55.127±2.263 | 55.925±5.533 |
| CIC-IDS-2017 | Joint (offline) | 5/5 | 2.000±0.000 | 1.932±0.036 | 98.190±1.938 | 68.427±1.297 | 78.558±3.581 |

| Section27.8 item | Status |
|---|---|
| 1. AA-CIL calibrated-alpha2% ADR higher than metric-wise strongest baseline on at least two datasets | pending |
| 2. No significant MF1 decline versus metric-wise strongest baseline on any dataset | pending |
| 3. No significant attack-family guard decline versus Replay | pending |
| 4. CIC AA-CIL MF1 significantly higher than TS-mem | pending |
| 5. Calibrated Joint gaps and separate G/F/summed costs reported descriptively | pending |

NSL-KDD AA-CIL (5/5): recorded G/F/sum training 175.948/159.800/335.747 s; G/F steps 44280/40000. Per-task actual memory, benign stream and separate calibration rows/bytes are in result.json; capacity is an upper limit, not actual occupancy.

NSL-KDD TS-mem (5/5): recorded G/F/sum training 158.666/156.346/315.012 s; G/F steps 40000/40000. Per-task actual memory, benign stream and separate calibration rows/bytes are in result.json; capacity is an upper limit, not actual occupancy.

NSL-KDD AA-noAttMem (5/5): recorded G/F/sum training 173.364/153.185/326.549 s; G/F steps 44280/40000. Per-task actual memory, benign stream and separate calibration rows/bytes are in result.json; capacity is an upper limit, not actual occupancy.

UNSW-NB15 AA-CIL (5/5): recorded G/F/sum training 205.047/188.788/393.835 s; G/F steps 52880/50000. Per-task actual memory, benign stream and separate calibration rows/bytes are in result.json; capacity is an upper limit, not actual occupancy.

Strongest baselines use one fixed five-seed mean winner per dataset/metric among the seven Section27.4 controls; FARQ is supplementary only. Replay+LA shares Replay weights and calibrated total τ, so its physical cost is not charged twice.
No detected significant harm is not proof of noninferiority/equivalence. These paired tests are unadjusted, use five seeds, and retain selection/multiplicity limitations. Joint is an offline access exception, not a guaranteed attainable upper bound. This RQ2 sub-study does not close RQ1, RQ3, or substantive innovation.
Payload mode: verify; actual payload hashes checked: 475. No new inference or threshold fitting was performed by this analyzer.
