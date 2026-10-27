"""Read-only independent confusion-count check for the post-hoc discussion.

Run from any directory: python -I -B paper_aacil/check_forgetting.py
No training, saved inference, or result-file writes. Equal seed/class weights;
peak excludes the final checkpoint. Historical joint uses its original seeds.
"""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'paper_aacil/results'


def load(dataset, method, seed):
    if method.startswith('R3-'):
        p = RESULTS / 'section165_confirmation' / dataset / f'{method}_seed{seed}/result.json'
    else:
        p = RESULTS / 'section123_confirmation' / dataset / f'{method}_seed{seed}.json'
    return json.loads(p.read_text(encoding='utf-8'))


def matrices(result, default=False):
    out = []
    for i, cp in enumerate(result['checkpoints']):
        if default:
            src = cp['default_decision'] if 'default_decision' in cp else result['default_stages'][i]
            src = src.get('metrics', src)
        else:
            src = cp if 'operating_points' in cp else result['calibrated_stages'][i]
            src = src['operating_points']['0.01']
        out.append(src['confusion_matrix'])
    return out


def recall(cm, c):
    assert sum(cm[c]) > 0
    return 100 * cm[c][c] / sum(cm[c])


def check(dataset, method, default=False):
    runs = [load(dataset, method, s) for s in range(10, 15)]
    tasks = runs[0].get('tasks') or runs[0]['identity']['tasks']
    names = runs[0]['class_order']
    intro = {c: t for t, classes in enumerate(tasks) for c in classes}
    old = [c for task in tasks[:-1] for c in task]
    cms = [matrices(r, default) for r in runs]
    for r, seq in zip(runs, cms):
        assert r['class_order'] == names and len(seq) == len(tasks)
        for t, cm in enumerate(seq):
            assert all(v >= 0 and int(v) == v for row in cm for v in row)
            for c in [c for task in tasks[:t + 1] for c in task]:
                assert sum(cm[c]) == sum(seq[-1][c])
    print(dataset, method, 'default' if default else '1pct')
    losses = {}
    for c in old:
        first = [recall(seq[intro[c]], c) for seq in cms]
        peak = [max(recall(cm, c) for cm in seq[intro[c]:-1]) for seq in cms]
        final = [recall(seq[-1], c) for seq in cms]
        losses[c] = mean(p - f for p, f in zip(peak, final))
        print(names[c], 'first/peak/final/loss', *(round(v, 6) for v in
              (mean(first), mean(peak), mean(final), losses[c])))
    print('mean forgetting', mean(losses.values()))
    fprs = [100 - recall(cm, 0) for seq in cms for cm in seq]
    print('test FPR min/max', min(fprs), max(fprs), 'final mean', mean(100 - recall(seq[-1], 0) for seq in cms))
    if dataset == 'unsw-nb15' and method == 'R3-A' and not default:
        for pair in [(2, 4), (3, 6), (1,)]:
            contribution = sum(losses[c] for c in pair) / len(old)
            print('contribution', pair, contribution, 'percent', 100 * contribution / mean(losses.values()))
        for t in (2, 3, 4):
            print('stage', t + 1, 'Exploits recall / to DoS / DoS recall',
                  mean(recall(seq[t], 2) for seq in cms),
                  mean(100 * seq[t][2][4] / sum(seq[t][2]) for seq in cms),
                  mean(recall(seq[t], 4) for seq in cms))
        for c in (3, 6):
            print(names[c], 'final to Normal', mean(100 * seq[-1][c][0] / sum(seq[-1][c]) for seq in cms))


if __name__ == '__main__':
    for ds, methods in [('unsw-nb15', ['R3-A', 'R3-L', 'ar']),
                        ('cic-ids-2017', ['R3-A']), ('cse-cic-ids2018', ['R3-A'])]:
        for method in methods:
            check(ds, method)
            if ds == 'unsw-nb15':
                check(ds, method, True)
    for c in (1, 2, 3, 4):
        vals = []
        for seed in (1, 2, 3, 4, 42):
            p = ROOT / f'ncd/results/problem_scout/section46/external_low_fpr_v1/unsw_nb15/joint_seed{seed}/task_004.json'
            cm = json.loads(p.read_text())['operating_points']['0.01']['metrics']['confusion_matrix']
            vals.append(recall(cm, c))
        print('historical joint final class', c, mean(vals))
    nsl = []
    base = ROOT / 'ncd/results/problem_scout/section46/external_low_fpr_v1/nsl_kdd'
    for method in ('replay', 'freq', 'wacb', 'derpp', 'joint'):
        for seed in (1, 2, 3, 4, 42):
            for stage in range(4):
                p = base / f'{method}_seed{seed}/task_{stage:03d}.json'
                cm = json.loads(p.read_text())['operating_points']['0.01']['metrics']['confusion_matrix']
                nsl.append(100 - recall(cm, 0))
    assert len(nsl) == 100
    print('historical NSL external 1pct test FPR range', min(nsl), max(nsl))
