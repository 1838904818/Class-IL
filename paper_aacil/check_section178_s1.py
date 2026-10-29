"""Independent standard-library verification of the registered S1 analysis.

Recomputes every stage and aggregate from immutable Git blobs without importing
the analysis module. Uses the original lead's t survival/sign-flip functions,
and closed-testing intersections rather than the analysis's step-down Holm code.
"""
import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import subprocess


def git_blob(repo, ref, path):
    return subprocess.check_output(['git', '-C', str(repo), 'show', ref+':'+path])


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b)
    assert abs(a-b) < 1e-12, (a, b)


def recalculate(run, view):
    stages, recalls, supports = [], [], []
    tasks = run.get('tasks') or run['identity']['tasks']
    for i, checkpoint in enumerate(run['checkpoints']):
        if view == 'default':
            item = checkpoint.get('default_decision') or run['default_stages'][i]
            item = item.get('metrics', item)
        else:
            item = checkpoint if 'operating_points' in checkpoint else run['calibrated_stages'][i]
            item = item['operating_points'][view]
        matrix = item['confusion_matrix']
        seen = checkpoint['seen_classes']
        row = [sum(r) for r in matrix]
        col = [sum(matrix[r][c] for r in range(len(matrix))) for c in range(len(matrix))]
        diagonal = sum(matrix[c][c] for c in seen)
        recall = {c: matrix[c][c]/row[c] for c in seen}
        recalls.append(recall); supports.append(row)
        stages.append(dict(final_accuracy=diagonal/sum(row), final_bacc=statistics.mean(recall.values()),
            final_macro_f1=statistics.mean(2*matrix[c][c]/(row[c]+col[c]) for c in seen),
            final_fpr=(row[0]-matrix[0][0])/row[0],
            final_attack_recall=sum(sum(r[1:]) for r in matrix[1:])/sum(row[1:])))
    old = [c for task in tasks[:-1] for c in task]
    metrics = dict(stages[-1],
        mean_incremental_bacc=statistics.mean(s['final_bacc'] for s in stages),
        stage_mean_accuracy=statistics.mean(s['final_accuracy'] for s in stages),
        signed_forgetting=statistics.mean(max(r[c] for r in recalls[:-1] if c in r)-recalls[-1][c] for c in old))
    return metrics, stages, supports


def closed_testing(pvalues):
    adjusted = [0.0]*len(pvalues)
    for count in range(1, len(pvalues)+1):
        for subset in itertools.combinations(range(len(pvalues)), count):
            intersection = min(1.0, count*min(pvalues[i] for i in subset))
            for i in subset:
                adjusted[i] = max(adjusted[i], intersection)
    return adjusted


def check(repo, result_path, analysis_path):
    result = json.loads(result_path.read_text(encoding='utf-8'))
    assert result['analysis_source_sha256'] == hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    assert result['paired_seeds'] == [10, 11, 12, 13, 14]
    family = ['derpp', 'icarl', 'ecbrs-paper', 'cer-bic-common']
    assert result['family'] == family
    assert set(result['datasets']) == {'unsw-nb15', 'cic-ids-2017'}
    ref = result['source_revision']
    lead_raw = git_blob(repo, ref, 'defence/forgetting_analysis/s178_check.py')
    tree = ast.parse(lead_raw)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('t_greater', '_t_sf', 'signflip')]
    assert len(selected) == 3
    namespace = dict(mean=statistics.mean, stdev=statistics.stdev, sqrt=math.sqrt, itertools=itertools)
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'published_lead_functions', 'exec'), namespace)
    assert namespace['signflip']([1, 2, 3, 4, 5]) == 1/32
    assert namespace['signflip']([0]*5) == 1
    for actual, expected in zip(closed_testing([.001, .01, .03, .2]), [.004, .03, .06, .2]): close(actual, expected)
    assert closed_testing([.02]*4) == [.08]*4
    total = 0
    for dataset, data in result['datasets'].items():
        independently = {}
        assert set(data['groups']) == {'R3-A', *family}
        for method, group in data['groups'].items():
            per_seed = group['per_seed']
            assert [r['seed'] for r in per_seed] == [10, 11, 12, 13, 14]
            independently[method] = {'default': [], '0.01': []}
            for record in per_seed:
                prefix = 'section165_confirmation' if method == 'R3-A' else 'section178_literature'
                assert record['source_path'] == f"paper_aacil/results/{prefix}/{dataset}/{method}_seed{record['seed']}/result.json"
                raw = git_blob(repo, ref, record['source_path'])
                assert hashlib.sha256(raw).hexdigest() == record['source_sha256']
                run = json.loads(raw)
                assert run['identity']['seed'] == record['seed'] and run['completed']
                for view in ('default', '0.01'):
                    metrics, stages, supports = recalculate(run, view)
                    reported = record['views'][view]
                    assert reported['supports'] == supports
                    for key, value in metrics.items(): close(value, reported['metrics'][key])
                    assert len(stages) == len(reported['stages'])
                    for stage, expected in zip(stages, reported['stages']):
                        for key, value in stage.items(): close(value, expected[key])
                    independently[method][view].append(metrics)
                total += 1
            for view, rows in independently[method].items():
                for key in rows[0]:
                    values = [r[key] for r in rows]
                    close(statistics.mean(values), group['aggregate'][view][key]['mean'])
                    close(statistics.stdev(values), group['aggregate'][view][key]['sample_sd'])
        a = [r['final_bacc'] for r in independently['R3-A']['0.01']]
        pvalues = []
        assert [r['method'] for r in data['comparisons']] == family
        for comparison in data['comparisons']:
            b = [r['final_bacc'] for r in independently[comparison['method']]['0.01']]
            diffs = [x-y for x,y in zip(a,b)]
            t, p = namespace['t_greater'](a,b)
            # Large t values amplify sub-ulp CM aggregation differences; retain
            # absolute 1e-12 for rate metrics and use relative 1e-12 for t only.
            assert math.isclose(t, comparison['t'], rel_tol=1e-12, abs_tol=1e-12)
            close(p, comparison['p_raw'])
            assert abs(p-comparison['p_raw'])/p < 1e-10
            close(namespace['signflip'](diffs), comparison['exact_sign_flip_p'])
            close(100*statistics.mean(diffs), comparison['mean_difference_pp'])
            close(100*statistics.stdev(diffs), comparison['sample_sd_difference_pp'])
            assert comparison['df'] == 4
            for d, expected in zip(diffs, comparison['paired_differences_pp']): close(100*d, expected)
            pvalues.append(p)
        adjusted = closed_testing(pvalues)
        for p, comparison in zip(adjusted, data['comparisons']):
            close(p, comparison['p_holm'])
            assert comparison['registered_S1_reject_null'] == (p < .05)
        assert data['all_four_registered_S1_pass'] == all(p < .05 for p in adjusted)
        print(dataset, 'INDEPENDENT_CM_METRICS_T_SIGNFLIP_CLOSED_TESTING_PASS', flush=True)
    assert total == 50
    print('LEAD_FUNCTIONS_SHA256', hashlib.sha256(lead_raw).hexdigest(), flush=True)
    print('VERIFIED_50_RESULTS_ALL_STAGES_AGGREGATES_8_TESTS; no training; CSE and broader gates open', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    check(args.source_repo, args.result, Path(__file__).with_name('analyze_section178_s1.py'))
