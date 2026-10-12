# CRR Section 13 correction and pre-run gate (2026-10-12)

**Decision: CONTINUE to the dependency-gated five-seed rerun.** This is an
engineering and benchmark-learnability decision only; it is not evidence that
CRR is superior or that the paper is publication-ready.

## Corrected controls

- **Global-IR:** preserves the intervention effect on the old sample's own
  log-probability. The student probability is normalized over every seen class,
  including the new class; the teacher probability is normalized over old
  classes only.
- **RCL-tab:** samples eight feature interventions per minibatch, shared by
  teacher and student; forms a feature-axis reliance profile; and minimizes
  gated JS divergence. The tracked auxiliary `rcl_tab_nogate` removes only the
  gate. Following the RCL defaults, `tau_r=0.5`, `beta=1`, and clean prediction
  KD weight `0.5`; the reliance lambda is selected on seeds 5 and 6 from
  `{0.25, 0.5, 1, 2}`. The gate uses normalized teacher entropy
  (`tau_H=0.5`, `gamma_H=0.1`) and batch teacher-sensitivity median/MAD as the
  deterministic threshold/scale. RCL source: <https://arxiv.org/html/2607.02020>.

## Section 13.3 learnability gate

Commit `89d6b5216308f0b45757015eb505e57a95b90189`, Replay, seed 1, eight base
and transition epochs. The operational gate requires every `k>=2` row to have
`d recall >= 0.5` and `c<->d confusion < 0.5`, plus positive `c`-to-`f2`
dependency change at the largest registered shift (`k=3`) in each dataset.

| Dataset | k | d recall | c recall | c<->d confusion | c-f2 dependency change |
|---|---:|---:|---:|---:|---:|
| NSL-KDD | 2 | 0.8934 | 0.0000 | 0.4440 | +0.5207 |
| NSL-KDD | 3 | 0.8937 | 0.2299 | 0.3269 | +1.1317 |
| UNSW-NB15 | 2 | 0.9979 | 0.0000 | 0.4985 | -0.0627 |
| UNSW-NB15 | 3 | 0.9995 | 0.0953 | 0.4470 | +0.1294 |

The registered gate passes, but low absolute `c` recall and the near-0.5 UNSW
confusion at `k=2` remain explicit limitations. The five-seed results must
still satisfy Section 12.7; these rows do not establish scientific success.

## Auditable row data

- Corrected-method smoke rows (16 rows):
  `ncd/results/crr/section13/local_smoke_seed1_89d6b52/crr_rows.csv`, SHA-256
  `6ea0defd5b61b1a131b5e0a4ceb4db8efca7b6dc0572db48e778bcca5d5b8f82`.
- Full-epoch Replay gate rows (6 rows):
  `ncd/results/crr/section13/local_learnability_seed1_89d6b52/crr_rows.csv`,
  SHA-256
  `c74372ceb309c1bbccf534d5cb235835cafecdd296d3b1b96e8286f4f33ca4c3`.
- Gate receipt:
  `ncd/results/crr/section13/local_learnability_seed1_89d6b52/LEARNABILITY_VERDICT.json`,
  SHA-256
  `6cf2ec2604cc898b5bdb2689228f9bb1aac0990792b37a5d4857a253864133df`.

Formal calibration and five-seed row CSVs will be added after their protected
HPC outputs reach a terminal successful state.
