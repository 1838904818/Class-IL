# Five-seed per-class analysis

Source: DICC Job 425539. All downloaded sources passed their protected SHA-256 bindings.

Class-wise accuracy means recall. Values below are five-seed mean +/- sample standard deviation on one fixed official test split.

| Class | Support | Last-epoch recall | Calibration recall | Last-epoch F1 | Calibration F1 |
|---|---:|---:|---:|---:|---:|
| Benign | 174,421 | 86.9% +/- 8.3% | 89.6% +/- 5.0% | 90.7% +/- 4.4% | 92.1% +/- 2.8% |
| DoS GoldenEye | 2,059 | 97.8% +/- 0.9% | 94.7% +/- 4.8% | 90.2% +/- 7.3% | 92.5% +/- 2.0% |
| DoS Hulk | 46,215 | 81.0% +/- 13.7% | 81.1% +/- 18.1% | 80.6% +/- 8.2% | 82.7% +/- 7.5% |
| DoS Slowhttptest | 1,100 | 75.0% +/- 14.3% | 79.5% +/- 5.8% | 54.6% +/- 7.7% | 53.8% +/- 8.0% |
| DoS slowloris | 1,159 | 69.0% +/- 21.2% | 74.2% +/- 11.0% | 48.3% +/- 16.5% | 49.6% +/- 10.5% |
| FTP-Patator | 1,588 | 69.0% +/- 44.3% | 70.1% +/- 24.9% | 30.9% +/- 28.5% | 51.5% +/- 19.9% |
| Heartbleed | 2 | 60.0% +/- 54.8% | 60.0% +/- 54.8% | 9.4% +/- 13.3% | 7.6% +/- 10.4% |
| SSH-Patator | 1,179 | 82.3% +/- 23.3% | 69.0% +/- 38.7% | 30.4% +/- 14.7% | 21.0% +/- 16.6% |

## Decision

Training-only checkpoint calibration is not promoted to the primary configuration. It slightly increases final accuracy and Macro-F1, but increases forgetting and reduces average task accuracy and attack recall. Every paired 95% confidence interval crosses zero.
