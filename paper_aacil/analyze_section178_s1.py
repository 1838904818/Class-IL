"""Registered S1: immutable published CMs, five paired seeds, four tests/dataset.

No training, inference, threshold fitting, candidate selection or result edits.
Run with --output-dir pointing to a NEW directory; otherwise validate/print only.
Requires NumPy and SciPy. All rates are fractions; differences are percentage points.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
REF = '4cdbcdbaf46f0aea6f8842a658c8e1855b031075'
SEEDS = list(range(10, 15))
DATASETS = {'unsw-nb15': 'UNSW-NB15', 'cic-ids-2017': 'CIC-IDS-2017'}
METHODS = {'derpp': 'derpp', 'icarl': 'icarl',
           'ecbrs-paper': 'ecbrs-paper-literal', 'cer-bic-common': 'cer-bic-flow-cnn-common'}
LABELS = {'R3-A': 'AR-Stateless', 'derpp': 'DER++', 'icarl': 'iCaRL (adapt.)',
          'ecbrs-paper': 'ECBRS (paper)', 'cer-bic-common': 'BiC (common flow)'}


def blob(repo, ref, path):
    return subprocess.check_output(['git', '-C', str(repo), 'show', ref+':'+path])


def result_path(ds, method, seed):
    prefix = 'section165_confirmation' if method == 'R3-A' else 'section178_literature'
    return f'paper_aacil/results/{prefix}/{ds}/{method}_seed{seed}/result.json'


def measure(run, view):
    tasks = run.get('tasks') or run['identity']['tasks']
    names = run.get('class_order') or run['identity']['class_order']
    assert names[0] == 'Normal' and len(run['checkpoints']) == len(tasks)
    assert sorted(c for task in tasks for c in task) == list(range(len(names)))
    stages, recalls, supports = [], [], []
    for t, cp in enumerate(run['checkpoints']):
        assert cp['task_index'] == t
        seen = [c for task in tasks[:t+1] for c in task]
        assert seen == cp['seen_classes']
        if view == 'default':
            src = cp.get('default_decision') or run['default_stages'][t]
            src = src.get('metrics', src)
        else:
            src = cp if 'operating_points' in cp else run['calibrated_stages'][t]
            src = src['operating_points'][view]
        raw = src['confusion_matrix']
        assert all(type(x) is int and x >= 0 for row in raw for x in row)
        cm = np.asarray(raw, dtype=np.int64)
        assert cm.shape == (len(names), len(names))
        rows, cols = cm.sum(axis=1), cm.sum(axis=0)
        unseen = sorted(set(range(len(names))) - set(seen))
        assert np.all(rows[seen] > 0) and np.all(rows[unseen] == 0) and np.all(cols[unseen] == 0)
        recall = {c: float(cm[c, c]/rows[c]) for c in seen}
        recalls.append(recall)
        supports.append(rows.tolist())
        stages.append(dict(final_bacc=float(np.mean(list(recall.values()))),
                           final_accuracy=float(np.trace(cm)/cm.sum()),
                           final_macro_f1=float(np.mean([2*cm[c,c]/(rows[c]+cols[c]) for c in seen])),
                           final_fpr=float(cm[0,1:].sum()/rows[0]),
                           final_attack_recall=float(cm[1:,1:].sum()/rows[1:].sum())))
    for t, rows in enumerate(supports):
        assert all(rows[c] == supports[-1][c] for c in run['checkpoints'][t]['seen_classes'])
    old = [c for task in tasks[:-1] for c in task]
    forgetting = float(np.mean([max(row[c] for row in recalls[:-1] if c in row)-recalls[-1][c] for c in old]))
    metrics = dict(stages[-1], mean_incremental_bacc=float(np.mean([s['final_bacc'] for s in stages])),
                   stage_mean_accuracy=float(np.mean([s['final_accuracy'] for s in stages])),
                   signed_forgetting=forgetting)
    return dict(metrics=metrics, stages=stages, supports=supports)


def holm(p):
    order = np.argsort(p, kind='stable')
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1., np.maximum.accumulate(np.asarray(p)[order] * np.arange(len(p), 0, -1)))
    return adjusted.tolist()


def build(repo, ref):
    output = dict(schema_version=1, analysis='Section 178 S1', source_revision=ref,
        registered_endpoint='AR-Stateless minus each added baseline: final BAcc at alpha=0.01',
        paired_seeds=SEEDS, family=list(METHODS), excluded_sensitivity=['ecbrs-author','cer-bic'],
        test='paired one-sided Student t, H1 difference > 0, df=4; Holm over four baselines separately within each dataset; p<0.05',
        sign_flip='exact 32 sign assignments on paired mean differences; unadjusted sensitivity reported, not the registered decision rule',
        units='rates as fractions; paired mean differences as percentage points; sample SD uses ddof=1',
        source_data_not_recomputed=True, new_training=False,
        limitations=['Five seed replicates on fixed shared data, not independent datasets or traffic samples.',
          'The t test assumes approximately normal independent seed-level differences; n=5 limits assessment.',
          'Exact sign-flip assumes sign exchangeability and has minimum one-sided p=1/32.',
          'A successful S1 endpoint does not imply lower forgetting, superior default decisions, native reproduction or substantive innovation.',
          'AR-Stateless retains its refreshed-benign access; added baselines keep benign inside memory without refresh.',
          'ECBRS paper/supplement ambiguity, iCaRL softmax-KL/NME adaptation and BiC flow/common-schedule override remain.',
          'CSE-2018 family remains uncomputed; RQ1, RQ3, innovation and visual QA remain open.'],
        software=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), datasets={})
    for ds, dataset in DATASETS.items():
        groups, baseline_binding, baseline_tasks, baseline_names = {}, None, None, None
        for method in ['R3-A', *METHODS]:
            per_seed=[]
            for seed in SEEDS:
                path=result_path(ds,method,seed)
                raw=blob(repo,ref,path); run=json.loads(raw); identity=run['identity']; cfg=identity['effective_config']
                assert run['completed'] is True and run.get('scientific_result', True) is True
                assert identity['dataset']==dataset and identity['seed']==seed
                assert not identity.get('smoke',cfg.get('smoke',False)) and not cfg.get('smoke',False)
                assert identity['method']==('R3-A' if method=='R3-A' else METHODS[method])
                if method=='R3-A': assert cfg['section165_memory']=='exact_c1'
                if method=='cer-bic-common': assert cfg['schedule']=='cer_common_adam_8epochs_min2000'
                tasks=run.get('tasks') or identity['tasks']; names=run.get('class_order') or identity['class_order']
                if baseline_binding is None: baseline_binding,baseline_tasks,baseline_names=identity['data_binding'],tasks,names
                assert identity['data_binding']==baseline_binding and tasks==baseline_tasks and names==baseline_names
                views={view:measure(run,view) for view in ('default','0.01')}
                assert views['default']['supports']==views['0.01']['supports']
                if method=='R3-A':
                    holds=[cp['calibration']['0.01'] for cp in run['checkpoints']]
                    assert len({h['row_ids_sha256'] for h in holds})==1
                    hold=holds[0]['row_ids_sha256']; hold_n=holds[0]['calibration_rows']
                else:
                    h=(identity.get('holdout') or identity['section26_holdout'])['0']
                    hold,hold_n=h['validation_indices_sha256'],h['validation_count']
                    ref_seed=groups['R3-A']['per_seed'][SEEDS.index(seed)]
                    assert hold==ref_seed['normal_holdout_sha256'] and hold_n==ref_seed['normal_holdout_rows']
                    assert views['default']['supports']==ref_seed['views']['default']['supports']
                per_seed.append(dict(seed=seed, source_path=path, source_sha256=hashlib.sha256(raw).hexdigest(),
                    normal_holdout_sha256=hold,normal_holdout_rows=hold_n,views=views))
            aggregate={view:{key:dict(mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1)))
                 for key in per_seed[0]['views'][view]['metrics']
                 for v in [[r['views'][view]['metrics'][key] for r in per_seed]]} for view in ('default','0.01')}
            groups[method]=dict(label=LABELS[method],per_seed=per_seed,aggregate=aggregate)
        comparisons=[]
        a=np.array([r['views']['0.01']['metrics']['final_bacc'] for r in groups['R3-A']['per_seed']])
        for method in METHODS:
            b=np.array([r['views']['0.01']['metrics']['final_bacc'] for r in groups[method]['per_seed']]); diff=a-b
            tt=stats.ttest_rel(a,b,alternative='greater')
            signs=np.array(list(itertools.product((-1,1),repeat=5)))
            sf=float(np.mean((signs*diff).mean(axis=1)>=diff.mean()-1e-12))
            comparisons.append(dict(method=method, paired_differences_pp=(100*diff).tolist(),
                mean_difference_pp=float(100*diff.mean()),sample_sd_difference_pp=float(100*diff.std(ddof=1)),
                t=float(tt.statistic),df=4,p_raw=float(tt.pvalue),exact_sign_flip_p=sf))
        for row,adjusted in zip(comparisons,holm([r['p_raw'] for r in comparisons])):
            row.update(p_holm=adjusted,registered_S1_reject_null=bool(adjusted<0.05))
        output['datasets'][ds]=dict(tasks=baseline_tasks,class_order=baseline_names,data_binding=baseline_binding,
            groups=groups,comparisons=comparisons,all_four_registered_S1_pass=all(r['registered_S1_reject_null'] for r in comparisons))
    return output


def tables(result):
    out=['% Generated by analyze_section178_s1.py from immutable published confusion matrices.',
         '% Evidence revision: '+result['source_revision'],
         r'\begin{table*}[t]\centering\small',
         r'\caption{Registered S1 at the 1\% calibration target. Five paired seeds (10--14); difference is AR-Stateless minus baseline in percentage points. Paired one-sided $t$ tests use df=4; Holm correction covers four baselines separately within each dataset. Exact sign-flip values are unadjusted sensitivity results, not a second decision rule.}',
         r'\label{tab:section178-s1}',r'\begin{tabular}{llrrrrr}\toprule',
         r'Dataset & Baseline & Difference & $t$ & Raw $p$ & Holm $p$ & Sign-flip $p$\\\midrule']
    for ds,data in result['datasets'].items():
        label='UNSW' if ds=='unsw-nb15' else 'CIC-2017'
        for r in data['comparisons']:
            out.append(f"{label} & {LABELS[r['method']]} & {r['mean_difference_pp']:+.2f} & {r['t']:.2f} & {r['p_raw']:.3g} & {r['p_holm']:.3g} & {r['exact_sign_flip_p']:.5f}"+r'\\')
    out += [r'\bottomrule\end{tabular}\end{table*}',
        r'\begin{table*}[t]\centering\scriptsize',
        r'\caption{Descriptive companion metrics for the same seeds, including lower-forgetting baselines. All entries are seed means (percent or pp). Default final Accuracy and stage-mean ACC are distinct. At 1\%, Inc. is stage-mean BAcc and Forget. is signed best-prior minus final old-class recall, including Normal. No tests are attached to these descriptive columns.}',
        r'\label{tab:section178-s1-descriptive}',r'\begin{tabular}{llrrrrrrr}\toprule',
        r'& & \multicolumn{4}{c}{Default} & \multicolumn{3}{c}{1\% target}\\',
        r'Dataset & Method & Final Acc. & ACC & Macro-F1 & Forget. & Final BAcc & Inc. & Forget.\\\midrule']
    for ds,data in result['datasets'].items():
        label='UNSW' if ds=='unsw-nb15' else 'CIC-2017'
        for m,g in data['groups'].items():
            d,a=g['aggregate']['default'],g['aggregate']['0.01']
            values=[d[k]['mean'] for k in ('final_accuracy','stage_mean_accuracy','final_macro_f1','signed_forgetting')]
            values += [a[k]['mean'] for k in ('final_bacc','mean_incremental_bacc','signed_forgetting')]
            out.append(' & '.join([label,LABELS[m]]+[f'{100*v:.2f}' for v in values])+r'\\')
    out += [r'\bottomrule\end{tabular}\end{table*}','']
    return '\n'.join(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-repo',type=Path,default=ROOT)
    p.add_argument('--git-ref',default=REF)
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args()
    if args.output_dir: assert not args.output_dir.exists(), 'Never overwrite analysis output'
    ref=subprocess.check_output(['git','-C',str(args.source_repo),'rev-parse',args.git_ref],text=True).strip()
    result=build(args.source_repo,ref)
    if args.output_dir:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        (args.output_dir/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
        (args.output_dir/'section178_s1.tex').write_text(tables(result),encoding='utf-8',newline='\n')
    for ds,d in result['datasets'].items():
        print(ds,json.dumps(d['comparisons'],sort_keys=True),flush=True)
    print('50_PUBLISHED_RESULTS_8_REGISTERED_TESTS_NO_TRAINING',flush=True)


if __name__=='__main__':
    main()
