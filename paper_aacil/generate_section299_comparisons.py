"""Section29.9 KD reassessment from saved stagewise metrics; no model calls.

UNSW is exploratory. NSL outcomes were available before direct authorization,
so the requested NSL/CIC comparison is not described as blind confirmation.
Original Section29.8 CBF and Section27.8 decisions are never overwritten.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_aacil.generate_forgetting import DATA, METHODS, escape, table, value
from paper_aacil.generate_section29_comparisons import (
    SEEDS, EXTERNAL, AIA, FORGET, ADR, relative, paired, complete,
    strongest, test_status, all_required, number,
)

DATASETS = ['UNSW-NB15', 'NSL-KDD', 'CIC-IDS-2017']
REQUIRED = ['NSL-KDD', 'CIC-IDS-2017']
KD = 'aacil_kd'
SOURCE = '7418b7d418f9974e88cba5dcc66c6126df117264'
DOCUMENT_TIME = '2026-10-19T20:47:57+08:00'
AUTHORIZATION_TIME_APPROX = '2026-10-19T21:29:00+08:00'
HARD_CUTOFF = '2026-10-19T21:50:26+08:00'
FINAL_BACC = 'final_balanced_accuracy'
METRICS = [AIA, FINAL_BACC, FORGET, ADR, 'final_macro_f1', 'final_normal_fpr']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_snapshot(folder):
    summary = read(folder/'result.json')
    groups, rows, sources = {}, {}, []
    for group in summary['groups']:
        dataset, method = group['dataset'], group['method']
        if dataset not in DATASETS or method not in EXTERNAL+[KD, 'aacil', 'replay_bs']:
            continue
        groups[dataset, method] = group
        for seed in group['seeds']:
            if seed not in SEEDS:
                raise ValueError('Only fixed formal seeds may enter Section29.9')
            path = folder/dataset.lower().replace('-', '_')/f'{method}_seed{seed}'/'result.json'
            row = read(path)
            if (not row.get('completed') or row['protocol'] != summary['protocol'] or
                    (row['dataset'], row['method'], row['seed']) != (dataset, method, seed)):
                raise ValueError('Stagewise cell is not part of the selected completed snapshot')
            if method == KD:
                cfg = row.get('training_configuration', {})
                if (not cfg.get('section298') or cfg.get('selection_only') or
                        row['training_git_commit'] != SOURCE or
                        any(cfg.get(k) != v for k, v in {'lambda_g': .5, 'derpp_alpha': 1., 'derpp_beta': 1.}.items())):
                    raise ValueError('KD source or frozen hyperparameters differ from Section29.9')
                times = {b['source_completed_at'] for b in row.get('stage_bindings', []) if b.get('source_completed_at')}
                if len(times) != 1:
                    raise ValueError('KD result availability time must be explicit and consistent')
                completed_at = times.pop()
                stamp = datetime.fromisoformat(completed_at)
                if stamp.tzinfo is None:
                    raise ValueError('Result availability time must include its timezone')
                sources.append({'dataset': dataset, 'method': method, 'seed': seed,
                    'source_result': row['source_result'], 'source_completed_at': completed_at,
                    'stagewise_result': relative(path), 'training_git_commit': row['training_git_commit'],
                    'before_document_commit': stamp < datetime.fromisoformat(DOCUMENT_TIME),
                    'before_approximate_direct_authorization': stamp < datetime.fromisoformat(AUTHORIZATION_TIME_APPROX)})
            rows[dataset, method, seed] = row | {'comparison_input': relative(path)}
    return summary, groups, rows, sources


def assess(summary, groups, rows, sources):
    results = []
    for dataset in DATASETS:
        lowest = strongest(groups, dataset, FORGET)
        forgetting = (paired(rows, groups, dataset, KD, lowest['selected'], FORGET)
            if lowest['selected'] else {'dataset': dataset, 'candidate': KD, 'reference': None,
                'metric': FORGET, 'status': 'pending', 'n': 0, 'paired_values': [], 'reason': lowest['reason']})
        forgetting.update(criterion=1, comparator_selection=lowest, p_direction='greater',
            rule='Fail if KD all-old forgetting is significantly higher: p_greater < .05',
            decision=test_status(forgetting, 'greater', True))
        accuracy = paired(rows, groups, dataset, KD, 'replay_bs', FINAL_BACC)
        accuracy.update(criterion=2, p_direction='greater',
            rule='Pass only if KD final BAcc exceeds Replay-BS: p_greater < .05',
            decision=test_status(accuracy, 'greater'))
        detection = paired(rows, groups, dataset, KD, 'aacil', ADR)
        detection.update(criterion=3, p_direction='less',
            rule='Fail if KD alpha=.02 ADR is significantly lower than original AA: p_less < .05',
            decision=test_status(detection, 'less', True))
        tests = [forgetting, accuracy, detection]
        g = groups.get((dataset, KD), {})
        results.append({'dataset': dataset, 'required': dataset in REQUIRED,
            'interpretation': 'exploratory_after_UNSW_outcomes' if dataset == 'UNSW-NB15' else
                'not_blind_confirmatory_results_available_before_authorization' if dataset == 'NSL-KDD' else
                'prospectively_requested_followup_incomplete_until_all_comparators_present',
            'n': g.get('n', 0), 'seeds': g.get('seeds', []), 'kd_complete_five_seed': complete(groups, dataset, KD),
            'completed_seeds': [s for s in SEEDS if s in g.get('seeds', [])],
            'missing_seeds': [s for s in SEEDS if s not in g.get('seeds', [])],
            'descriptive_only': not complete(groups, dataset, KD),
            'metrics': {k: g.get('metrics', {}).get(k) for k in METRICS},
            'available_group_counts': {m: groups.get((dataset, m), {}).get('n', 0) for m in EXTERNAL+[KD, 'aacil', 'replay_bs']},
            'tests': tests, 'status': all_required([t['decision'] for t in tests]),
            'complete': all(complete(groups, dataset, m) for m in EXTERNAL+[KD, 'aacil', 'replay_bs'])})
    required = [d for d in results if d['required']]
    verdict = all_required([d['status'] for d in required])
    nsl = [s for s in sources if s['dataset'] == 'NSL-KDD']
    return {'completed_at': datetime.now().astimezone().isoformat(), 'section': '29.9',
        'candidate': KD, 'frozen_configuration': {'lambda_g': .5, 'derpp_alpha': 1., 'derpp_beta': 1.},
        'training_source_commit': SOURCE, 'formal_seeds': SEEDS, 'required_datasets': REQUIRED,
        'external_five': EXTERNAL, 'source_summary_completed_at': summary['completed_at'],
        'stagewise_protocol': summary['protocol'], 'dataset_results': results,
        'required_numeric_verdict': verdict, 'required_comparisons_complete': all(d['complete'] for d in required),
        'blind_confirmation_established': False, 'original_section298_cbf_verdict': 'fail',
        'original_section278_verdict': 'fail', 'source_bindings': sources,
        'timing_disclosure': {'document_commit_at': DOCUMENT_TIME,
            'direct_authorization_approximately_at': AUTHORIZATION_TIME_APPROX,
            'authorization_time_precision': 'approximately one minute; not a claimed exact timestamp',
            'nsl_seeds_available_before_document_commit': [s['seed'] for s in nsl if s['before_document_commit']],
            'nsl_seeds_available_before_approximate_direct_authorization': [s['seed'] for s in nsl if s['before_approximate_direct_authorization']],
            'interpretation': 'Result availability is not evidence that the user inspected those results. NSL is not represented as blind confirmation.'},
        'positioning': 'Required KD comparison fails; retain a trade-off study' if verdict == 'fail' else
            'Specified numerical conditions met, but not independent blind confirmation' if verdict == 'pass' else
            'Required KD evidence remains incomplete; no success claim',
        'limitations': ['UNSW is exploratory because KD was reconsidered after its formal outcomes were available.',
            'NSL seeds 1/2 were available before the Section29.9 document commit, and all five before direct authorization; no user inspection is inferred.',
            'No inference, training, checkpoint audit or CM metric recomputation is performed by this reporting script.',
            'Only complete five-seed groups permit significance; missing CIC KD, original-AA or Replay-BS seeds remain pending.',
            'Partial group means use only completed trajectories at their actual n; a single seed has null sample SD. Partial checkpoints never supply completed-cell metrics.',
            'Known necessary-condition failure takes precedence over other missing evidence; no seed is dropped.',
            'Paired one-sided tests are unadjusted; no detected harm is not equivalence or noninferiority.',
            'Section29.8 CBF failure and the original Section27.8 failure remain unchanged.']}


def render(result):
    rows = []
    for d in result['dataset_results']:
        rows.append([escape(DATA[d['dataset']]), str(d['n'])] +
            [value(d['metrics'][k]) if d['n'] >= 1 and d['metrics'][k] else '--' for k in METRICS])
    text = table('Fixed-KD reassessment at stagewise Normal-holdout calibration target $\\alpha=2\\%$.',
        'section299-kd', 'lrrrrrrr', ['Data', '$n$', 'Mean inc. BAcc', 'Final BAcc', 'Forget (pp)', 'ADR', 'MF1', 'FPR'], rows,
        r'Mean$\pm$sample SD over completed trajectories at the stated $n$; $n=1$ has undefined SD (--). '
        r'Incomplete groups are descriptive only; significance requires all five formal seeds. '
        r'UNSW is exploratory. NSL results were available before direct authorization, so NSL is not blind confirmation. '
        r'KD uses fixed $\lambda_G=.5$ and DER $\alpha=\beta=1$. The validation-selected CBF and original superiority decisions remain failed.')
    rows = []
    for d in result['dataset_results']:
        for t in d['tests']:
            delta = t.get('delta_pp', {})
            effect = '--' if delta.get('mean') is None else f"{delta['mean']:.2f}"+rf"$\pm${delta['sample_sd']:.2f}"
            rows.append([escape(DATA[d['dataset']]), str(t['criterion']),
                escape(METHODS.get(t.get('reference'), 'pending')), str(t['n']), effect,
                number(t.get('p_'+t['p_direction'])), t['decision']])
    if any(t['n'] == 5 and t['status'] == 'complete' for d in result['dataset_results'] for t in d['tests']):
        text += table('Fixed-KD reassessment: candidate minus reference.', 'section299-tests', 'lllrrrl',
            ['Data', 'Criterion', 'Reference', '$n$', r'$\Delta$ (pp)', 'One-sided $p$', 'Decision'], rows,
            r'1: forgetting harm, $p_{>}<.05$ means failure against the lowest-forgetting external-five member. '
            r'2: final BAcc superiority over Replay-BS requires $p_{>}<.05$. 3: ADR harm relative to original AA means $p_{<}<.05$. '
            r'NSL and CIC17 are both required; UNSW does not replace either. Unadjusted tests; no detected harm is not equivalence. '
            'Required numerical verdict: '+escape(result['required_numeric_verdict'])+'.')
    missing = ['{} completed seeds {}; missing {}'.format(DATA[d['dataset']],
        ','.join(map(str, d['completed_seeds'])) or 'none', ','.join(map(str, d['missing_seeds'])) or 'none')
        for d in result['dataset_results'] if d['missing_seeds']]
    if missing:
        text += r'\par\noindent\footnotesize '+escape('; '.join(missing))+r'.\normalsize\par'+'\n'
    execution = result.get('execution', {})
    if execution.get('hard_cutoff_ended'):
        text += (r'\par\noindent\footnotesize The initial manuscript window ended on 19 October 2026 at 21:50:26 MYT. '
            'That attempt recorded status '+escape(execution.get('saved_execution_status') or execution.get('saved_status') or 'unavailable')+
            r'. Follow-up completion of the remaining CIC17 KD seeds and five original-AA seeds is authorized without an owner absolute cutoff; '
            r'these tables contain only completed results in the saved snapshot.\normalsize\par'+'\n')
        if execution.get('termination_cause') == 'owner_hard_deadline':
            text += r'\par\noindent\footnotesize The initial runner recorded the configured deadline TimeoutError, rather than an out-of-memory failure.\normalsize\par'+'\n'
        if execution.get('known_partial_checkpoint_count') is not None:
            cost = execution.get('saved_partial_stage_cost_seconds', {}).get('training_seconds')
            text += (r'\par\noindent\footnotesize The incomplete seed has '+str(execution['known_partial_checkpoint_count'])+
                ' saved stages'+(f' with {cost:.2f} s of recorded stage training' if cost is not None else '')+
                r'. This is a lower bound, excluding the unfinished unsaved stage and unrecorded overhead; it supplies no final-seed metric.\normalsize\par'+'\n')
    return '% Fixed-KD reassessment; original decisions unchanged.\n'+text


def execution_metadata(path, hard_cutoff_ended):
    """Optional post-exit small JSON only; never inspect a live process or weights."""
    out = {'hard_cutoff_at': HARD_CUTOFF, 'hard_cutoff_ended': hard_cutoff_ended,
           'saved_execution_status': None, 'saved_status': None,
           'known_partial_checkpoint_count': None, 'recorded_cost_seconds': None,
           'note': 'Execution metadata is separate from completed scientific trajectories; unavailable costs/counts are not zero.'}
    if path is None:
        return out
    raw = read(path)
    out.update(source=relative(path), input_rule='Caller supplies this saved file only after the training process has exited',
               saved_execution_status=raw.get('execution_status'), saved_status=raw.get('status'),
               saved_completed=raw.get('completed'), saved_completed_at=raw.get('completed_at'))
    if raw.get('error_type') == 'TimeoutError' and raw.get('error') == 'Owner experiment hard deadline reached; preserve completed checkpoints':
        out.update(termination_cause='owner_hard_deadline', saved_failure_at=raw.get('failed_at'),
                   termination_note='The frozen runner recorded the configured deadline, not a model efficacy or out-of-memory failure.')
    # A supplied incomplete physical result can expose saved checkpoints/costs.
    if raw.get('identity') and raw.get('completed') is not True and isinstance(raw.get('checkpoints'), list):
        out['known_partial_checkpoint_count'] = len(raw['checkpoints'])
        out['known_partial_task_indices'] = [c.get('task_index') for c in raw['checkpoints']]
    costs = {k: v for k, v in raw.items() if k.endswith('_seconds_total') and isinstance(v, (int, float)) and not isinstance(v, bool)}
    out['recorded_cost_seconds'] = costs or None
    current = raw.get('current_cell')
    if not isinstance(current, dict):
        return out
    dataset, method, seed = (current.get(k) for k in ('dataset', 'method', 'seed'))
    if dataset not in DATASETS or method != KD or seed not in SEEDS:
        raise ValueError('Execution metadata current_cell is not a fixed Section29.9 KD cell')
    cell = path.parent/dataset.lower().replace('-', '_')/f'{method}_seed{seed}'
    out['current_cell'] = {'dataset': dataset, 'method': method, 'seed': seed}
    final = cell/'result.json'
    if final.exists() and read(final).get('completed') is True:
        out['partial_cell_note'] = 'The saved current_cell already has a completed result; it is not counted as partial work.'
        return out
    folder = cell/'checkpoints'
    if not folder.is_dir():
        out['partial_cell_note'] = 'No checkpoint sidecar directory is available; saved partial cost is unknown.'
        return out
    stages = []
    for sidecar in sorted(folder.glob('checkpoint_*.json')):
        record = read(sidecar)
        task = record.get('task_index')
        if (not isinstance(task, int) or isinstance(task, bool) or task < 0 or
                sidecar.name != f'checkpoint_{task:03d}.json'):
            raise ValueError('Partial checkpoint sidecar has an ambiguous stage index')
        stage_cost = {k: v for k, v in record.items()
            if k.endswith('_seconds') and isinstance(v, (int, float)) and not isinstance(v, bool)}
        stages.append({'task_index': task, 'sidecar': relative(sidecar), 'recorded_cost_seconds': stage_cost})
    fields = sorted({k for stage in stages for k in stage['recorded_cost_seconds']})
    out.update(known_partial_checkpoint_count=len(stages),
        known_partial_task_indices=[s['task_index'] for s in stages],
        saved_partial_stages=stages,
        saved_partial_stage_cost_seconds={k: sum(s['recorded_cost_seconds'].get(k, 0.) for s in stages) for k in fields},
        saved_partial_cost_field_stage_counts={k: sum(k in s['recorded_cost_seconds'] for s in stages) for k in fields},
        partial_cost_scope='Lower bound from saved stage JSON only. The unsaved current stage, loading, serialization and unrecorded work are unknown. Training includes gate+family; family includes base-family+CBF, so these nested fields must not be added again. Cumulative training fields are excluded.',
        partial_cell_note='Saved partial stages do not supply completed-seed metrics or significance; no checkpoint payload was read.')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'ncd/results/problem_scout/section28/forgetting_v1')
    parser.add_argument('--output', type=Path, default=ROOT/'ncd/results/problem_scout/section29/section299_comparisons_v1/result.json')
    parser.add_argument('--tex-output', type=Path, default=ROOT/'paper_aacil/generated/section299_comparisons.tex')
    parser.add_argument('--hard-cutoff-ended', action='store_true',
                        help='Declare the fixed hard cutoff has elapsed; never infer this from an in_progress file')
    parser.add_argument('--execution-status', type=Path,
                        help='Optional saved matrix or incomplete result JSON, read only after process exit; requires --hard-cutoff-ended')
    parser.add_argument('--render-existing', action='store_true',
                        help='Render the saved comparison JSON only; do not recompute tests or alter historical execution metadata')
    args = parser.parse_args()
    if args.render_existing:
        if args.hard_cutoff_ended or args.execution_status:
            parser.error('--render-existing cannot change execution metadata')
        result=read(args.output)
        if result.get('section')!='29.9' or result.get('candidate')!=KD:
            parser.error('--render-existing requires the saved fixed-KD comparison')
        args.tex_output.parent.mkdir(parents=True, exist_ok=True)
        args.tex_output.write_text(render(result), encoding='utf-8')
        print(json.dumps({'rendered_existing':str(args.output), 'recomputed_statistics':False}))
        return
    if args.execution_status and not args.hard_cutoff_ended:
        parser.error('--execution-status is restricted to post-exit hard-cutoff reporting')
    if args.hard_cutoff_ended and datetime.now().astimezone() < datetime.fromisoformat(HARD_CUTOFF):
        parser.error('The fixed hard cutoff has not elapsed')
    summary, groups, rows, sources = load_snapshot(args.input.resolve())
    result = assess(summary, groups, rows, sources)
    result['source_summary'] = relative(args.input/'result.json')
    result['execution'] = execution_metadata(args.execution_status.resolve() if args.execution_status else None, args.hard_cutoff_ended)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    args.tex_output.write_text(render(result), encoding='utf-8')
    print(json.dumps({'section': '29.9', 'required_numeric_verdict': result['required_numeric_verdict'],
        'required_comparisons_complete': result['required_comparisons_complete'], 'blind_confirmation_established': False}))


if __name__ == '__main__':
    main()
