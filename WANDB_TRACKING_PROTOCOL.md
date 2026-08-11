# W&B tracking protocol for OFRA full-data runs

## Scope

W&B is an outbound experiment-recording client. It does not replace the
auditable JSON files written by `streaming_full`, and the JSON files remain the
source of record for paper tables, hashes, and reproducibility checks.

Each dataset seed is recorded as a separate W&B run. Runs that share an
identical protocol are grouped by the protocol SHA-256 unless an explicit group
is supplied. The dedicated HPC project name is `ofra-etg-leon-hpc`, which
avoids modifying existing research-group projects.

## Live checkpoint metrics

The x-axis is the class-incremental checkpoint index. For every evaluation
view and decision arm, the tracker records:

- overall accuracy;
- macro-F1;
- balanced accuracy;
- average task accuracy over tasks seen by that checkpoint;
- average forgetting over previously observed tasks;
- benign false-positive rate and attack-detection recall when the dataset uses
  the NIDS metric profile.

The decision arms remain explicit: head-only, router-only, and joint score,
each with the registered capped/uncapped or exposure-prior diagnostic variant.
They are not merged into an undocumented aggregate.

## End-of-seed tables

At successful completion, W&B receives long-form tables for:

- the final summary metrics by evaluation view and decision arm;
- the checkpoint-by-task accuracy matrix;
- the final confusion matrix, including raw count and row-normalized value;
- checkpoint-monitor manifest paths and SHA-256 values.

The confusion matrix is deliberately a table rather than an automatically
rendered dense chart, because class labels overlap on larger NIDS datasets.

## SHAP and ETG status

The current checkpoint monitor freezes model state and fixed-probe score
traces, but it does not implement an explanation method. Therefore W&B records
`shap_status=not_computed`, `etg_status=not_computed`, and
`explanation_method=null`. SHAP drift or ETG metrics must not be reported until
a separately versioned explanation method consumes the checkpoint artifacts
and passes its own reproducibility tests.

## Credential handling

The account owner enters `WANDB_API_KEY` once through a hidden terminal prompt
for the scheduled connectivity job. That job calls the official
`wandb.login(key=..., verify=True)` API, which verifies the credential and
stores it in the account owner's `~/.netrc`. The job then enforces file mode
`0600` and verifies the stored credential without printing it. Later jobs rely
on W&B's standard `.netrc` lookup and do not ask for the key again.

The key is never accepted as a command-line option and is never written to the
repository, an sbatch file, project configuration, W&B run configuration, or a
log. The persistent credential is private to the DICC account but remains a
long-lived secret: revoke it in W&B User Settings and repeat the one-time setup
if the key or HPC account may have been exposed.
