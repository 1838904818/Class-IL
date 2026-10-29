"""Default-decision tables from explicit default confusion matrices only."""
import json
import itertools
from pathlib import Path
from statistics import mean, stdev
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'paper_aacil/results'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def metrics(stages, tasks):
    values, recalls = [], []
    for stage in stages:
        cm, seen = stage['confusion_matrix'], stage['seen_classes']
        support = [sum(row) for row in cm]
        counts = [sum(row[c] for row in cm) for c in range(len(cm))]
        assert all(support[c] > 0 for c in seen)
        total = sum(support)
        f1 = {c: 2*cm[c][c]/(support[c]+counts[c]) for c in seen}
        values.append((sum(cm[c][c] for c in seen)/total, mean(f1.values()), sum(support[c]*f1[c] for c in seen)/total))
        recalls.append({c: cm[c][c]/support[c] for c in seen})
    assert len(tasks) == len(stages)
    old = [(i,c) for i, task in enumerate(tasks[:-1]) for c in task]
    bwt = mean(recalls[-1][c]-recalls[i][c] for i,c in old)
    forgetting = mean(max(s[c] for s in recalls[i:-1])-recalls[-1][c] for i,c in old)
    return (*values[-1], mean(v[0] for v in values), bwt, forgetting)


def build():
    out = [r'% Generated from explicit default decisions; no calibrated matrix substitution.']
    for dataset, slug in [('unsw-nb15','unsw'), ('cic-ids-2017','cic')]:
        groups = {}
        out += [r'\begin{table*}[t]\centering\scriptsize',
                r'\caption{'+dataset.upper()+r': historical confirmation-source snapshot (default decision, no FPR calibration). Seeds 10--14, mean $\pm$ sample SD; percentages or percentage points. ACC is mean stage micro accuracy; BWT is mean old-class recall change since introduction; forgetting is signed best pre-final minus final old-class recall, including Normal. NA denotes absence from these historical confirmation/default-export collections, not a zero or an omitted failed seed. Later completed added-baseline blocks are reported separately in Table~\ref{tab:published-baselines-default}.}',
                r'\label{tab:default-'+slug+'}',r'\begin{tabular}{lrrrrrrr}\toprule',
                r'Method & $n/5$ & Accuracy & Macro-F1 & Weighted-F1 & ACC & BWT & Forgetting \\\midrule']
        for method, label in [('R3-A','AR-Stateless'),('R3-L','R3-L'),('replay','Replay'),('derpp','DER++'),('wacb','WACB'),('freq','Freq'),('replay_bs','Replay-BS'),('aa_cil','AA-CIL'),('kdwa','KD+WA'),('akd','AKD'),('joint','Joint (offline)'),('ar','AR'),('ar_joint','AR-Joint'),('er_bal_tabt','ER-Bal-TabT')]:
            rows = []
            for seed in range(10,15):
                path = RESULTS/'section123_confirmation'/dataset/f'{method}_seed{seed}.json'
                if method.startswith('R3-'):
                    path = RESULTS/'section165_confirmation'/dataset/f'{method}_seed{seed}/result.json'
                if method == 'er_bal_tabt':
                    path = RESULTS/'section137_defaults'/f'{slug}_seed{seed}.json'
                if not path.exists():
                    continue
                r = read(path)
                if method == 'er_bal_tabt':
                    source = read(ROOT/f'ncd/results/problem_scout/section123/publication_v1/{dataset.replace("-","_")}/fixed200/er_bal_tabt_seed{seed}/result.json')
                    tasks = source['identity']['tasks']
                    stages = [cp['default_decision'] for cp in r['checkpoints']]
                else:
                    tasks = r.get('tasks', r.get('identity', {}).get('tasks'))
                    stages = [cp['metrics'] for cp in r['default_stages']] if 'default_stages' in r else [cp['default_decision'] for cp in r['checkpoints']]
                computed = metrics(stages,tasks)
                if method == 'er_bal_tabt':
                    for index, key in [(1,'final_macro_f1'),(3,'acc'),(4,'bwt_per_class_recall'),(5,'forgetting_all_old_classes')]:
                        assert abs(computed[index]-r['default_summary'][key]) < 1e-12
                rows.append(computed)
            if len(rows) == 5:
                groups[label] = rows
            columns = []
            for i in range(6):
                values = [r[i] for r in rows]
                columns.append('NA' if not values else f'{100*mean(values):.1f}' + (f' $\\pm$ {100*stdev(values):.1f}' if len(values)>1 else r' $\pm$ NA'))
            out.append(label+' & '+str(len(rows))+' & '+' & '.join(columns)+r' \\')
        out += [r'\bottomrule\end{tabular}\end{table*}']
        out += [r'\begin{table*}[t]\centering\scriptsize',
                r'\caption{'+dataset.upper()+r': exploratory paired default-decision comparisons within the historical source collections, historical neural AR minus comparator, seeds 10--14. Two-sided paired $t$ and exact sign-flip $p$ values (all 32 assignments), unadjusted for multiple comparisons. Only complete paired blocks are tested; these tests do not replace historical decisions. Positive accuracy/F1/ACC/BWT differences favor AR; negative forgetting differences favor AR.}',
                r'\begin{tabular}{llrrr}\toprule Comparator & Endpoint & Mean difference (pp) & $p_t$ & $p_{flip}$ \\\midrule']
        for label, rows in groups.items():
            if label == 'AR':
                continue
            for i, endpoint in enumerate(('Accuracy','Macro-F1','Weighted-F1','ACC','BWT','Forgetting')):
                a, b = [r[i] for r in groups['AR']], [r[i] for r in rows]
                diff = [x-y for x,y in zip(a,b)]
                observed = abs(mean(diff))
                flip = sum(abs(mean(d*s for d,s in zip(diff,signs))) >= observed-1e-12 for signs in itertools.product((-1,1),repeat=5))/32
                p = float(ttest_rel(a,b).pvalue)
                out.append(f'{label} & {endpoint} & {100*mean(diff):.2f} & {p:.3g} & {flip:.4f}'+r' \\')
        out += [r'\bottomrule\end{tabular}\end{table*}']
    return '\n'.join(out)+'\n'


if __name__ == '__main__':
    (ROOT/'paper_aacil/generated/default_decision_view.tex').write_text(build(),encoding='utf-8',newline='\n')
    print('Default-decision table generated; absent seed cells remain explicit.')
