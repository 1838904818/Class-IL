"""Regenerate the completed Section 178 tables from published confusion counts.

Default: read this checkout's result files. --git-ref reads immutable Git blobs
instead (useful when the manuscript checkout intentionally has an old HEAD).
No training, inference, result edits, imputation, or significance tests.
"""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DATASETS = [('unsw-nb15', 'UNSW'), ('cic-ids-2017', 'CIC-2017'),
            ('cse-cic-ids2018', 'CSE-2018')]
METHODS = [('derpp', 'DER++'), ('icarl', 'iCaRL (adapt.)'),
           ('ecbrs-paper', 'ECBRS (paper)'), ('cer-bic-common', 'BiC (common flow)')]
VIEWS = ('default', '0.01', '0.02', '0.05')


def read_result(path, ref):
    if ref:
        exists = subprocess.run(['git', '-C', str(ROOT), 'cat-file', '-e', ref+':'+path],
                                capture_output=True)
        if exists.returncode:
            return None
        raw = subprocess.check_output(['git', '-C', str(ROOT), 'show', ref+':'+path])
    else:
        source = ROOT/path
        if not source.exists():
            return None
        raw = source.read_bytes()
    return json.loads(raw)


def measure(result, view):
    assert result['completed']
    identity = result['identity']
    tasks = result.get('tasks') or identity['tasks']
    assert len(tasks) == len(result['checkpoints'])
    names = result.get('class_order') or identity['class_order']
    recalls, points, supports = [], [], {}
    for t, cp in enumerate(result['checkpoints']):
        if view == 'default':
            point = cp.get('default_decision') or result['default_stages'][t]
            point = point.get('metrics', point)
        else:
            point = cp if 'operating_points' in cp else result['calibrated_stages'][t]
            point = point['operating_points'][view]
        cm = point['confusion_matrix']
        assert len(cm) == len(names) and all(len(row) == len(names) for row in cm)
        assert all(isinstance(v, int) and v >= 0 for row in cm for v in row)
        seen = [c for task in tasks[:t+1] for c in task]
        assert seen == cp['seen_classes']
        rows = [sum(row) for row in cm]
        cols = [sum(row[c] for row in cm) for c in range(len(cm))]
        for c in seen:
            assert rows[c] > 0 and rows[c] == supports.setdefault(c, rows[c])
        assert all(rows[c] == 0 for c in range(len(cm)) if c not in seen)
        recalls.append({c: cm[c][c]/rows[c] for c in seen})
        points.append(dict(bacc=mean(recalls[-1].values()),
                           accuracy=sum(cm[c][c] for c in seen)/sum(rows),
                           macro_f1=mean(2*cm[c][c]/(rows[c]+cols[c]) for c in seen),
                           fpr=sum(cm[0][1:])/rows[0]))
    old = [c for task in tasks[:-1] for c in task]
    losses = [max(h[c] for h in recalls[:-1] if c in h)-recalls[-1][c] for c in old]
    return dict(points[-1], forgetting=mean(losses))


def group(dataset, method, ref, reference=False):
    prefix = 'section165_confirmation' if reference else 'section178_literature'
    # The published CSE confirmation and added-baseline collections use
    # different directory names. Keep the reference on its original path.
    stored_dataset = ('cic-ids-2018' if dataset == 'cse-cic-ids2018' and not reference
                      else dataset)
    paths = [f'paper_aacil/results/{prefix}/{stored_dataset}/{method}_seed{s}/result.json'
             for s in range(10, 15)]
    runs = [read_result(path, ref) for path in paths]
    count = sum(r is not None and r.get('completed', False) for r in runs)
    if count != 5:
        return count, None
    for seed, r in zip(range(10, 15), runs):
        assert r['identity']['seed'] == seed
        assert (r.get('tasks') or r['identity']['tasks']) == (runs[0].get('tasks') or runs[0]['identity']['tasks'])
        config = r['identity'].get('effective_config', {})
        assert not r['identity'].get('smoke', config.get('smoke', False))
        if method == 'cer-bic-common':
            assert config['schedule'] == 'cer_common_adam_8epochs_min2000'
            assert r['identity']['method'] == 'cer-bic-flow-cnn-common'
    return count, {v: {k: (mean(p[k] for p in points), stdev(p[k] for p in points))
                          for k in points[0]}
                   for v in VIEWS for points in [[measure(r, v) for r in runs]]}


def fmt(value, sd=False):
    m, s = value
    return f'{100*m:.2f}' + (rf' $\pm$ {100*s:.2f}' if sd else '')


def completed_diagnostics(ref, output):
    """Format already published analyses; no new tests or model evaluation."""
    base = 'paper_aacil/results/'
    score = read_result(base+'section178_literature/same-score-robustness-493707/aggregate.json', ref)
    nsl = read_result(base+'section194_diagnostics/nsl-kdd/score_shift_remaining_eight/aggregate.json', ref)
    wacb = read_result(base+'section194_diagnostics/cic-ids-2018/wacb_score_distributions_493836/aggregate.json', ref)
    assert score and nsl and wacb
    out = ['% Formatting immutable published analyses, not new experiments or tests.',
           '% Evidence revision: '+str(ref),
           r'\begin{table}[htbp]\centering\small',
           r'\caption{Completed CSE-2018 registered S1 companion to Table~\ref{tab:section178-s1}. Original prior-shift rule, five paired seeds, one-sided $t$ and Holm over four baselines. Differences are pp; exact sign-flip sensitivity remains unadjusted.}',
           r'\label{tab:cse-completed-s1}',
           r'\scriptsize\setlength{\tabcolsep}{3pt}',
           r'\begin{tabular}{lrrrr}\toprule',
           r'Baseline & Difference & $t$ & Holm $p$ & Sign-flip $p$\\\midrule']
    cse = score['section344']['cic-ids-2018']['0.01']['rules']['original_prior_shift']
    for method, label in METHODS:
        q = cse[method]['final_bacc_comparison']
        assert q['n'] == 5 and q['paired_t_reject_at_0_05']
        out.append(f"{label} & {100*q['mean_paired_difference']:.2f} & {q['paired_t_statistic']:.2f} & {q['paired_t_holm_p']:.3g} & {q['exact_signflip_one_sided_p']:.5f}"+r'\\')
    out += [r'\bottomrule\end{tabular}\end{table}',
            r'\begin{table*}[t]\centering\small',
            r'\caption{Post-hoc common-score robustness at the 1\% holdout target. Each entry is original prior-shift $\rightarrow$ common-score final BAcc (five-seed mean, percent). This does not replace the registered primary or match measured TEST FPR. New baseline forward evaluations use Linux; historical reference/prior results remain as published.}',
            r'\label{tab:same-score-completed}',
            r'\begin{tabular}{lrrrr}\toprule',
            r'Dataset & DER++ & iCaRL & ECBRS & BiC common\\\midrule']
    for ds,label in [('unsw-nb15','UNSW'),('cic-ids-2017','CIC-2017'),('cic-ids-2018','CSE-2018')]:
        rules = score['section344'][ds]['0.01']['rules']
        values = []
        for method,_ in METHODS:
            old,new = [rules[rule][method]['aggregate']['balanced_accuracy']['mean'] for rule in ('original_prior_shift','same_score_holdout')]
            values.append(f'{100*old:.2f}'+r'$\rightarrow$'+f'{100*new:.2f}')
        out.append(' & '.join([label]+values)+r'\\')
    out += [r'\bottomrule\end{tabular}\end{table*}',
            r'\begin{table}[htbp]\centering\small',
            r'\caption{Unregistered NSL-KDD TEST-label-informed post-hoc operating points, five-seed means (percent). At 2\% all methods have 194/9711 false positives per seed. At 5\%, AR-Stateless has a lower actual FPR, so that column is not strictly matched. No new significance decision is attached.}',
            r'\label{tab:nsl-completed-posthoc}',
            r'\begin{tabular}{lrrr}\toprule',
            r'Method & 2\% BAcc & 5\% BAcc & 5\% actual FPR\\\midrule']
    for method in ('R3-A','R3-L','ar','replay_bs','freq','wacb','derpp','icarl','ecbrs','bic'):
        q = nsl['methods'][method]['posthoc']
        assert all(row['posthoc']['0.02']['false_positives'] == 194 for row in nsl['methods'][method]['seeds'])
        label = {'R3-A':'AR-Stateless','ar':'Neural AR','replay_bs':'Replay-BS'}.get(method,method.replace('_',r'\_'))
        out.append(' & '.join([label]+[f'{100*x:.2f}' for x in (q['0.02']['balanced_accuracy']['mean'],q['0.05']['balanced_accuracy']['mean'],q['0.05']['measured_test_fpr']['mean'])])+r'\\')
    out += [r'\bottomrule\end{tabular}\end{table}',
            r'\begin{table}[htbp]\centering\small',
            r'\caption{CSE-2018 WACB saved-model score diagnosis. Original 1\% prior-shift rule; BAcc/FPR are percent. KS compares benign holdout and TEST scores within each model, not across models.}',
            r'\label{tab:wacb-completed-score}',
            r'\begin{tabular}{rrrr}\toprule',
            r'Seed & BAcc & TEST FPR & Benign KS\\\midrule']
    for row in wacb['independent']['seeds']:
        q = row['original_views']['0.01']
        out.append(f"{row['seed']} & {100*q['balanced_accuracy']:.2f} & {100*q['measured_test_fpr']:.4f} & {row['ks_statistic']:.6f}"+r'\\')
    out += [r'\bottomrule\end{tabular}\end{table}', '']
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(out),encoding='utf-8',newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--git-ref')
    parser.add_argument('--output', type=Path,
                        default=ROOT/'paper_aacil/generated/published_baselines.tex')
    parser.add_argument('--completed-diagnostics-output', type=Path)
    args = parser.parse_args()
    ref = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', args.git_ref],
                                  text=True).strip() if args.git_ref else None
    rows, status = [], []
    completed = 0
    for dataset, label in DATASETS:
        n, values = group(dataset, 'R3-A', ref, reference=True)
        assert n == 5, 'Missing AR-Stateless reference'
        rows.append((label, 'AR-Stateless', values))
        counts = []
        for method, name in METHODS:
            n, values = group(dataset, method, ref)
            counts.append(n)
            completed += n
            if values is not None:
                rows.append((label, name, values))
        status.append((label, counts))
        if dataset == 'unsw-nb15':
            n, values = group(dataset, 'ecbrs-author', ref)
            if values is not None:
                rows.append((label, r'ECBRS (author)$^\dagger$', values))
            n, values = group(dataset, 'cer-bic', ref)
            if values is not None:
                rows.append((label, r'BiC (native sched.)$^\ddagger$', values))
    out = ['% Generated from published confusion counts; do not hand edit.',
           '% Evidence revision: '+(ref or 'result files in this checkout'),
           r'\begin{table*}[t]\centering\small',
           r'\caption{Completed added-baseline blocks and AR-Stateless reference, default decisions. Five seeds (10--14); BAcc is mean $\pm$ sample SD, other entries are means. Accuracy, BAcc, macro-F1 and FPR are percentages; signed forgetting is pp. Adapted implementations and access limits are described in the text.}',
           r'\label{tab:published-baselines-default}',
           r'\begin{tabular}{llrrrrr}\toprule',
           r'Dataset & Method & Accuracy & Final BAcc & Macro-F1 & FPR & Forgetting\\\midrule']
    for ds, method, values in rows:
        p = values['default']
        out.append(' & '.join([ds, method, fmt(p['accuracy']), fmt(p['bacc'], True),
                               fmt(p['macro_f1']), fmt(p['fpr']), fmt(p['forgetting'])])+r'\\')
    out += [r'\bottomrule\end{tabular}',
            r'\par\smallskip\footnotesize $^\dagger$UNSW-only author-memory sensitivity; not an additional primary cell or a native binary reproduction.',
            r'\par\footnotesize $^\ddagger$Historical native-stage-1 schedule sensitivity on flow inputs; excluded from the common-budget primary family and Holm comparisons. Not native packet reproduction.',
            r'\end{table*}', r'\begin{table*}[t]\centering\scriptsize',
            r'\caption{The same completed blocks at empirical calibration budgets. Final BAcc at 1\% is mean $\pm$ sample SD; other entries are means. BAcc/FPR are percentages and signed forgetting is pp. Targets are not guarantees of test FPR.}',
            r'\label{tab:published-baselines-calibrated}',
            r'\begin{tabular}{llrrrrrrr}\toprule',
            r'& & \multicolumn{3}{c}{1\% target} & \multicolumn{2}{c}{2\% target} & \multicolumn{2}{c}{5\% target}\\',
            r'Dataset & Method & BAcc & FPR & Forgetting & BAcc & FPR & BAcc & FPR\\\midrule']
    for ds, method, values in rows:
        p, q, s = (values[v] for v in ('0.01', '0.02', '0.05'))
        out.append(' & '.join([ds, method, fmt(p['bacc'], True), fmt(p['fpr']),
                               fmt(p['forgetting']), fmt(q['bacc']), fmt(q['fpr']),
                               fmt(s['bacc']), fmt(s['fpr'])])+r'\\')
    out += [r'\bottomrule\end{tabular}\end{table*}',
            r'\begin{table}[htbp]\centering\small',
            rf'\caption{{Added-baseline execution/publication coverage: {completed}/60 common-family primary cells. Each entry is completed seeds out of five; zero means no completed published seed, not a negative result. The five author-memory and five historical BiC native-schedule sensitivity cells are additional and excluded from this family.}}',
            r'\label{tab:published-baselines-coverage}',
            r'\begin{tabular}{lrrrr}\toprule',
            r'Dataset & DER++ & iCaRL & ECBRS & CER\\\midrule']
    for ds, counts in status:
        out.append(' & '.join([ds]+[f'{n}/5' for n in counts])+r'\\')
    out += [r'\bottomrule\end{tabular}\end{table}', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(out), encoding='utf-8', newline='\n')
    if args.completed_diagnostics_output:
        completed_diagnostics(ref, args.completed_diagnostics_output)
    print(json.dumps(dict(completed_primary=completed, remaining_primary=60-completed,
                          evidence_revision=ref, rows=len(rows)), sort_keys=True))


if __name__ == '__main__':
    main()
