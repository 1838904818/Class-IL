# Supervisor review response

This note maps the nine review findings to the v2.2 manuscript and its bound
evidence. `Done` means the stated document, code, or bounded diagnostic change
is present and auditable. `Partial` means the audit is complete but a
compute-dependent corrective rerun remains open.

| Item | Status | Response and evidence |
|---:|:---:|---|
| 1 | Done | Related Work now discusses E2D2, Maseno et al., Goncalves and Alomari, and Adabara et al. Each is contrasted with legitimate class-incremental routed-score drift and non-suppressing governance. The complete citation records are retained in `references/REFERENCES.md` and `references/ofra_etg_references.bib`. |
| 2 | Done | The manuscript now describes OFRA prediction and an integrated offline/post-hoc ETG evidence ledger. It does not claim that ETG changes future routing, replay, or training. |
| 3 | Done | The legacy `6/298` headline is removed. The current source-bound result is `12/17` eligible Malaya seed-1 class-by-adjacent-checkpoint transitions for the actual `joint_cap3000` class margin. It is labelled as a single-seed transition result wherever reported. |
| 4 | Done | Prediction and retention, additive classifier comparison, and ETG governance are reported in separate result subsections with explicit RQ labels. |
| 5 | Done | The LoRA citation uses the official ICLR OpenReview record; the mismatched DOI is not used. |
| 6 | Done, bounded pilot | Expected Gradients, single-feature ablation, and Gradient x Input were evaluated on the same 30 probe rows. All-method agreement is 40.0% for admission, 43.3% for ETG state, and 20.0% for the silent-drift conclusion. Integrated Gradients failed the recorded completeness check and is excluded from the primary comparison. See `audits/ATTRIBUTION_ROBUSTNESS_2026-08-19.md`. |
| 7 | Partial | The code/data audit confirms Task-0-only numerical scaling. It also finds 6 future-only NSL-KDD and 8 future-only UNSW-NB15 one-hot columns, affecting 115/12,703 (0.905%) and 712/79,341 (0.897%) later-task rows respectively. This is bounded transductive schema exposure, not test leakage. A strict Task-0-only vocabulary rebuild and affected reruns remain open. See `audits/NO_LOOKAHEAD_AUDIT_2026-08-19.md`. |
| 8 | Done | The Discussion now states when the accuracy-forgetting trade-off may be operationally justified: retaining rare consequential attack classes may outweigh a small aggregate accuracy loss, while the same argument does not automatically apply to benign application-traffic subclasses. |
| 9 | Done | Funding, conflict-of-interest, and ethics/data-governance declarations are included. |

## Additional closure evidence

The repository also records the separate five-seed CSE-CIC-IDS2018
FT-Transformer 256x4 closure under a one-Task-0-epoch plus one-later-task-epoch
protocol. Joint cap 3,000 raises mean Macro-F1 by 12.24 percentage points and
balanced accuracy by 28.26 points relative to head-only, while lowering mean
accuracy by 16.72 points and increasing mean forgetting by 11.41 points. This
is reported as a trade-off and is not pooled with the FT-Transformer 512x12
eight/ten-epoch results.

## Remaining compute-dependent work

- rebuild NSL-KDD and UNSW-NB15 categorical vocabularies using Task-0 data only;
- rerun the affected matched comparisons after the strict rebuild;
- extend SHAP/ETG robustness beyond the current Malaya seed-1 bounded pilot;
- complete matched five-dataset statistics under one frozen protocol.
