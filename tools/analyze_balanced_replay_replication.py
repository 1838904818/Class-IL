"""Reproduce paired summaries from byte-bound, non-sensitive research exports.

Python 3.10+, standard library only. No network or training.
"""
import argparse
import glob
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as st

SEEDS = (1, 2, 3, 4, 42)
METRICS = ('average_task_accuracy', 'final_overall_accuracy', 'final_macro_f1',
           'final_balanced_accuracy', 'average_forgetting',
           'final_attack_detection_recall', 'final_benign_false_positive_rate')
LOWER = {'average_forgetting', 'final_benign_false_positive_rate'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def exact_p(values, ranked=False):
    values = [v for v in values if abs(v) > 1e-15]
    if not values:
        return 1.0
    magnitudes = [abs(v) for v in values]
    if ranked:
        magnitudes = [st.mean([i + 1 for i, x in enumerate(sorted(magnitudes))
                              if math.isclose(x, v, rel_tol=0, abs_tol=1e-15)])
                      for v in magnitudes]
    observed = abs(sum(math.copysign(m, v) for m, v in zip(magnitudes, values)))
    return sum(abs(sum(s*m for s, m in zip(signs, magnitudes))) >= observed-1e-12
               for signs in itertools.product((-1, 1), repeat=len(values))) / 2**len(values)

def holm(values):
    running = 0.0
    result = {}
    for i, (key, p) in enumerate(sorted(values.items(), key=lambda row: row[1])):
        running = max(running, min(1.0, (len(values)-i)*p))
        result[key] = running
    return result

def summary(values):
    # Student-t 0.975 critical values for df=3 and df=4.
    critical = {4: 3.182446305284263, 5: 2.7764451051977987}[len(values)]
    mean, sd = st.mean(values), st.stdev(values)
    half = critical * sd / math.sqrt(len(values))
    return dict(values=values, mean=mean, sample_sd=sd,
                descriptive_t95_interval=[mean-half, mean+half])

def analyze(paths):
    records, inputs = {}, {}
    for root in {p.resolve().parent for p in paths}:
        manifest=root/'EXPORT_SHA256.json'
        if manifest.exists():
            for name, expected in json.loads(manifest.read_text(encoding='utf-8')).items():
                if Path(name).name != name or sha(root/name) != expected:
                    raise ValueError('Export manifest mismatch')
    for path in paths:
        pair = json.loads(path.read_text(encoding='utf-8'))
        if pair.get('status') != 'verified_single_seed_pair':
            raise ValueError('Unverified pair')
        seed = pair['seed']
        if seed in records or path.name in inputs:
            raise ValueError('Duplicate seed or filename')
        audit_path = path.with_name(path.name.replace('_comparison.json', '_audit.json'))
        if sha(audit_path) != pair['baseline_audit_sha256']:
            raise ValueError('Audit hash mismatch')
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        if (audit['status'], audit['seed'], audit['dataset'], audit['result_file_sha256']) != (
                'verified', seed, pair['dataset'], pair['baseline_result_sha256']):
            raise ValueError('Audit identity mismatch')
        expected = set(METRICS[:5] if pair['dataset']=='malaya-network-gt' else METRICS)
        if set(pair['metrics']) != expected:
            raise ValueError('Incomplete metric set')
        for metric, row in pair['metrics'].items():
            b, o = row['balanced_replay50'], row['ofra_joint_cap3000']
            if not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)
                       for v in (b, o)):
                raise ValueError('Non-finite metric')
            if b != audit['methods']['balanced_replay50'][metric]:
                raise ValueError('Baseline metric differs from audit')
            if not math.isclose(row['ofra_minus_replay_pp'], 100*(o-b), abs_tol=1e-10):
                raise ValueError('Paired difference mismatch')
        records[seed] = pair
        inputs[path.name] = sha(path)
        inputs[audit_path.name] = sha(audit_path)
    if tuple(sorted(records)) != SEEDS:
        raise ValueError('Expected exactly seeds 1, 2, 3, 4, 42')
    first = records[1]
    identity = lambda r: (r['dataset'], r['checkpoints'], r['final_test_rows'],
                         [(p['class_id'], p['class_name'], p['support']) for p in r['per_class']])
    if any(identity(r) != identity(first) for r in records.values()):
        raise ValueError('Dataset, checkpoints or class support mismatch')
    cohorts = {}
    for name, seeds in [('all_five_descriptive', SEEDS), ('new_seeds_1_to_4_sensitivity', SEEDS[:4])]:
        metrics = {}
        for metric in first['metrics']:
            b = [records[s]['metrics'][metric]['balanced_replay50']*100 for s in seeds]
            o = [records[s]['metrics'][metric]['ofra_joint_cap3000']*100 for s in seeds]
            delta = [ov-bv for ov, bv in zip(o, b)]
            benefit = [-d if metric in LOWER else d for d in delta]
            sd = st.stdev(benefit)
            metrics[metric] = dict(balanced_replay50=summary(b), ofra_joint_cap3000=summary(o),
                ofra_minus_replay_pp=summary(delta), lower_is_better=metric in LOWER,
                ofra_wins=sum(d>1e-12 for d in benefit), ties=sum(abs(d)<=1e-12 for d in benefit),
                ofra_losses=sum(d < -1e-12 for d in benefit),
                paired_hedges_gz_approx=(st.mean(benefit)/sd*(1-3/(4*len(seeds)-5))) if sd else None,
                exact_sign_flip_two_sided_p=exact_p(delta),
                exact_wilcoxon_two_sided_p=exact_p(delta, ranked=True))
        for test in ('sign_flip', 'wilcoxon'):
            adjusted = holm({k: v[f'exact_{test}_two_sided_p'] for k, v in metrics.items()})
            for k, p in adjusted.items():
                metrics[k][f'holm_{test}_p'] = p
        cohorts[name] = dict(seeds=list(seeds), metrics=metrics)
    return dict(status='verified_paired_aggregate_exports', dataset=first['dataset'],
        analysis_source_lf_sha256=hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode('utf-8')).hexdigest(),
        input_sha256=dict(sorted(inputs.items())), units='percent for metrics; percentage points for paired differences',
        cohorts=cohorts, statistical_contract={
            'pairing': 'Same training seed on one fixed dataset/task-order contract, not independently matched random draws.',
            'intervals': 'Descriptive Student-t intervals across training seeds; normality of seed effects is unverified at n=4/5.',
            'tests': 'Exact two-sided sign enumeration under sign-symmetry/exchangeability; average tied ranks, drop zero differences.',
            'multiplicity': 'Holm within dataset and cohort across all reported metrics, separately per test; not a study-wide confirmatory family.',
            'minimum_two_sided_p': {'n5': 0.0625, 'n4': 0.125},
            'selection': 'Seed 42 selected the comparator; all-five summaries are descriptive. Seeds 1-4 are separate sensitivity evidence, not a new held-out dataset.'},
        limitations=['Fixed split/task order: training-seed variation does not establish domain or task-order generalization.',
            'Balanced replay doubles later-task row exposure; equal epochs are not equal compute.',
            'OFRA retains router centroids in addition to exemplars; total memory is not matched.',
            'Method-level comparison, not isolation of LoRA or router causality, nor proof of state-of-the-art performance.',
            'Negative signed forgetting denotes backward improvement and is not clipped.',
            'W&B URLs are recorded but cloud contents are not independently verified.',
            'No new SHAP/ETG experiment or publication-acceptance claim follows from this replication.'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison', nargs='+', help='Exact filenames or wildcard patterns')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    paths=[]
    for pattern in args.comparison:
        matches=sorted(glob.glob(pattern))
        if not matches:
            parser.error('No comparison file matched: '+pattern)
        paths.extend(Path(p) for p in matches)
    args.output.write_text(json.dumps(analyze(paths), indent=2, allow_nan=False)+'\n',
                           encoding='utf-8', newline='\n')
