"""Apply the registered TB schedule rule to published results; no training."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1, 2, 3, 4, 42)


def main():
    sources, rows = [], []

    def read(path):
        raw = path.read_bytes()
        value = json.loads(raw)
        sources.append(dict(path=path.relative_to(ROOT).as_posix(),
                            sha256=hashlib.sha256(raw).hexdigest()))
        assert value['completed']
        return value

    def final(value):
        point = value['checkpoints'][-1]['operating_points']['0.01']
        cm = point['confusion_matrix']
        assert all(sum(row) > 0 for row in cm)
        derived = mean(row[i] / sum(row) for i, row in enumerate(cm))
        saved = point['balanced_accuracy']
        assert math.isclose(saved, derived, rel_tol=0, abs_tol=1e-12)
        return saved

    for slug in ('unsw_nb15', 'cic_ids_2017'):
        for seed in SEEDS:
            tb = ROOT / 'paper_aacil/results/section123_tb'
            es = read(tb / f'{slug}_ar_es_seed{seed}.json')
            bs = read(tb / f'{slug}_replay_bs_es_seed{seed}.json')
            fixed = read(ROOT / f'ncd/results/problem_scout/section28/publication_v1/{slug}/aacil_nogkd_wa_seed{seed}/result.json')
            er = read(ROOT / f'ncd/results/problem_scout/section81/publication_v1/{slug}/fixed200/er_bal_tabt_seed{seed}/result.json')
            assert all(v['identity']['seed'] == seed for v in (es, bs, fixed, er))
            def field(v, key):
                return v[key] if key in v else v['identity'][key]
            assert field(es, 'tasks') == field(fixed, 'tasks') == field(er, 'tasks')
            assert field(es, 'class_order') == field(fixed, 'class_order') == field(er, 'class_order')
            rows.append(dict(dataset=slug, seed=seed, ar_es=final(es),
                             replay_bs_es=final(bs), ar_fixed=final(fixed), er_tabt=final(er)))
    means = {d: {k: mean(r[k] for r in rows if r['dataset'] == d)
                 for k in ('ar_es', 'ar_fixed', 'replay_bs_es', 'er_tabt')}
             for d in ('unsw_nb15', 'cic_ids_2017')}
    es_mean = mean(v['ar_es'] for v in means.values())
    fixed_mean = mean(v['ar_fixed'] for v in means.values())
    output = dict(scope='Section 123.1 registered descriptive schedule selection; bounded RQ2',
                  units='BAcc fractions; difference in percentage points',
                  alpha=0.01, seeds=list(SEEDS), completed_tb_cells=20,
                  per_seed=rows, dataset_means=means, ar_es_mean=es_mean,
                  ar_fixed_mean=fixed_mean, difference_pp=100*(es_mean-fixed_mean),
                  rule='Use ES iff AR-ES mean final BAcc averaged equally over both datasets >= AR fixed mean',
                  selected_schedule='ES' if es_mean >= fixed_mean else 'original_fixed',
                  registered_descriptive_reading='training_budget' if all(v['ar_es'] >= v['er_tabt']-.01 for v in means.values()) else 'structure_backbone',
                  sources=sources,
                  limitations=['Descriptive registered reading, not causal identification or a new significance test.',
                               'No training, inference, recalibration or cell re-export.',
                               'Fresh-seed confirmation remains pending; RQ1, RQ3 and substantive innovation remain unresolved.'])
    target = ROOT / 'paper_aacil/results/section123_schedule_selection.json'
    with target.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: output[k] for k in ('ar_es_mean', 'ar_fixed_mean', 'difference_pp', 'selected_schedule', 'registered_descriptive_reading')}))


if __name__ == '__main__':
    main()
