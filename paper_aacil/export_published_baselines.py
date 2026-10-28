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
           ('ecbrs-paper', 'ECBRS (paper)'), ('cer-bic', 'BiC (flow adapt.)')]
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
    paths = [f'paper_aacil/results/{prefix}/{dataset}/{method}_seed{s}/result.json'
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
    return count, {v: {k: (mean(p[k] for p in points), stdev(p[k] for p in points))
                          for k in points[0]}
                   for v in VIEWS for points in [[measure(r, v) for r in runs]]}


def fmt(value, sd=False):
    m, s = value
    return f'{100*m:.2f}' + (rf' $\pm$ {100*s:.2f}' if sd else '')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--git-ref')
    parser.add_argument('--output', type=Path,
                        default=ROOT/'paper_aacil/generated/published_baselines.tex')
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
            rf'\caption{{Added-baseline execution/publication coverage: {completed}/60 primary cells. Each entry is completed seeds out of five; zero means no completed published seed, not a negative result. The five author-memory sensitivity cells are additional.}}',
            r'\label{tab:published-baselines-coverage}',
            r'\begin{tabular}{lrrrr}\toprule',
            r'Dataset & DER++ & iCaRL & ECBRS & CER\\\midrule']
    for ds, counts in status:
        out.append(' & '.join([ds]+[f'{n}/5' for n in counts])+r'\\')
    out += [r'\bottomrule\end{tabular}\end{table}', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(out), encoding='utf-8', newline='\n')
    print(json.dumps(dict(completed_primary=completed, remaining_primary=60-completed,
                          evidence_revision=ref, rows=len(rows)), sort_keys=True))


if __name__ == '__main__':
    main()
