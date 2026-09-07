"""Post-hoc summaries of immutable records; never trains or evaluates a model."""
from pathlib import Path
import argparse, ast, hashlib, itertools, json, math, statistics
import numpy as np

SEEDS = [1, 2, 3, 4, 42]
METHODS = ['expected_gradients', 'feature_ablation', 'gradient_x_input']
ANALYZER_SHA = 'f32069bf41069cbd69114ad8e440091f3c91ddb6110236a9b2c2770763d1b2f6'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def canonical(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def dump(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0

def source_ledger(path, threshold):
    if sha(path) != ANALYZER_SHA:
        raise ValueError('historical analyzer hash mismatch')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_etg_ledger')
    scope = {'jaccard': jaccard, 'PRIMARY_JACCARD_THRESHOLD': threshold}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<verified historical ledger function>', 'exec'), scope)
    return scope['build_etg_ledger']

def signflip_p(values):
    v = np.asarray(values, dtype=np.float64)
    if not len(v) or not np.all(np.isfinite(v)):
        raise ValueError('finite nonempty paired differences required')
    observed = abs(float(v.mean()))
    means = [abs(float(np.mean(v * signs))) for signs in itertools.product([-1, 1], repeat=len(v))]
    return sum(x >= observed - 1e-12 for x in means) / len(means)

def holm(pvalues):
    if any(not 0 <= x <= 1 for x in pvalues):
        raise ValueError('invalid p value')
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    result = [None] * len(order)
    last = 0.0
    for k, i in enumerate(order):
        last = max(last, min(1., (len(order) - k) * pvalues[i]))
        result[i] = last
    return result

def validate_file(folder, name, registry_key):
    registry = {line.split(None, 1)[1].strip(): line.split(None, 1)[0] for line in (folder/'SHA256SUMS').read_text().splitlines()}
    p = folder/name
    if sha(p) != registry[registry_key]:
        raise ValueError(f'input checksum mismatch: {name}')
    return p

def primary_pairs(path):
    x=json.loads(path.read_text(encoding='utf-8'))
    if x['checkpoint_policies']['primary'] != 'Job 425539 last epoch, not superseded':
        raise ValueError('ReplayIDS primary checkpoint policy mismatch')
    if x['inference_primary'] != 'official/joint_cap3000':
        raise ValueError('ReplayIDS primary scoring arm mismatch')
    cohort=x['cohorts']['all_five_descriptive']
    if cohort['seeds'] != SEEDS: raise ValueError('ReplayIDS primary seed order mismatch')
    metrics=cohort['versus_replay']['primary']
    if len(metrics)!=7: raise ValueError('Seven primary ReplayIDS metrics required')
    return [{'seed':seed,'metrics':{key:{'ofra_minus_replay_pp':v['ofra_minus_replay_pp']['values'][i]}
        for key,v in metrics.items()}} for i,seed in enumerate(SEEDS)]

def run(evidence_root, comparison_root, source, out, primary_diagnostic):
    ledger_primary = source_ledger(source, .7)
    all_rows, common_rows, grid, alternative, inputs = [], [], [], [], []
    for seed in SEEDS:
        folder = evidence_root/f'seed-{seed}'
        files = [('analysis.json','./expected-gradients-etg/analysis.json'),
                 ('attribution_robustness.json','./robustness/attribution_robustness.json'),
                 (f'result_seed_{seed}.json',f'./result_seed_{seed}.json')]
        paths = [validate_file(folder, n, k) for n, k in files]
        a, r, training = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
        assert a['seed'] == r['seed'] == seed
        probe_sha = sha(folder/'probe_manifest.json')
        metadata=json.loads((folder/'protocol_metadata.json').read_text())
        registry={line.split(None,1)[1].strip():line.split(None,1)[0] for line in (folder/'SHA256SUMS').read_text().splitlines()}
        assert metadata['original_protocol_file_sha256']==registry['./expected-gradients-etg/analysis_protocol.json']
        assert metadata['analysis_protocol_sha256']==a['analysis_protocol_sha256']
        assert metadata['probe_manifest_file_sha256']==probe_sha
        assert metadata['original_verified_before_redacted_export'] is True
        inputs.append({'seed': seed, 'files': {p.name:sha(p) for p in paths},
                       'checksum_registry_sha256':sha(folder/'SHA256SUMS'),
                       'probe_manifest_local_sha256':probe_sha,
                       'probe_manifest_verified_via_protected_protocol':True,
                       'metadata_capsule_sha256':sha(folder/'protocol_metadata.json'),
                       'metadata_scope':'Redacted metadata attestation derived from a protocol whose original file hash was verified before export; original personal path is not redistributed.'})
        maps = {}
        for method in METHODS:
            rows, transitions = r['checkpoint_rows'][method], r['transition_rows'][method]
            historical = ledger_primary(rows, transitions)
            hist_map = {(x['checkpoint'],x['class_id']):x for x in historical}
            for row in rows:
                h = hist_map[(row['checkpoint'],row['class_id'])]
                assert h['state_after'] == row['etg_state'] and h['action'] == row['etg_action']
                enriched = dict(row, method=method, method_definition=r['methods'][method],
                    ledger_implementation_sha256=ANALYZER_SHA,
                    attribution_implementation_sha256=r['source_bindings']['script_sha256'],
                    method_package_version=metadata['dependencies']['shap'] if method=='expected_gradients' else 'custom source-hash version',
                    historical_environment_dependencies=metadata['dependencies'],
                    package_version_status='paired protected analysis protocol; custom methods identified by implementation hash',
                    analysis_protocol_sha256=a['analysis_protocol_sha256'],
                    policy_sha256=canonical(r['thresholds']),
                    source_robustness_sha256=sha(paths[1]),
                    training_result_sha256=r['source_bindings']['training_result_file_sha256'],
                    probe_background_manifest_local_sha256=probe_sha,
                    attribution_scope=r['attribution_scope'],
                    provenance_status='retrospective enrichment; original rows unchanged',
                    display_state={'CERTIFIED_STABLE':'EXPLANATION_RULE_PASSED','UNEXPLAINABLE':'EXPLANATION_RULE_FAILED','DRIFTED':'EXPLANATION_CHANGE_FLAGGED'}[row['etg_state']],
                    evidence_use='audit only; no safety certification or repair action validated')
                all_rows.append(enriched)
            maps[method] = {(x['checkpoint'],x['class_id']):x for x in rows}
            for j, guard in itertools.product([.5,.6,.7,.8],[0.,.02,.05,.1]):
                updated = [dict(t, primary_eligible=t['delta_recall'] > -guard,
                     primary_event=t['delta_recall'] > -guard and t['jaccard_top15'] < j) for t in transitions]
                ledger = source_ledger(source,j)(rows,updated)
                grid.append({'seed':seed,'method':method,'k':15,'jaccard':j,'allowed_recall_drop':guard,
                    'events':sum(t['primary_event'] for t in updated),
                    'eligible':sum(t['primary_eligible'] for t in updated),
                    'state_changed_from_primary':sum(x['state_after'] != hist_map[(x['checkpoint'],x['class_id'])]['state_after'] for x in ledger),
                    'states':{s:sum(x['state_after']==s for x in ledger) for s in ['CERTIFIED_STABLE','UNEXPLAINABLE','DRIFTED']},
                    'actions':{act:sum(x['action']==act for x in ledger) for act in sorted({x['action'] for x in ledger})}})
        keys = sorted(maps[METHODS[0]])
        assert all(sorted(maps[m])==keys for m in METHODS)
        for k in keys:
            rows = [maps[m][k] for m in METHODS]
            assert len({x['recall'] for x in rows}) == 1
            recall=rows[0]['recall']
            bin_id = 0 if recall<.2 else 1 if recall<.5 else 2 if recall<.8 else 3
            common_rows.append({'seed':seed,'checkpoint':k[0],'class_id':k[1], 'class_name':rows[0]['class_name'],
                'recorded_class_recall':recall,'bin':bin_id,
                'admission_agreement':len({x['admitted'] for x in rows})==1,
                'state_agreement':len({x['etg_state'] for x in rows})==1,
                'any_historical_certified_state':any(x['etg_state']=='CERTIFIED_STABLE' for x in rows)})
        ts = a['transition_rows']
        alternative.append({'seed':seed,'transitions':len(ts),
            'cosine_mean':statistics.mean(t['cosine_similarity'] for t in ts),
            'kendall_tau_b_mean':statistics.mean(t['kendall_tau_b'] for t in ts),
            'cosine_range':[min(t['cosine_similarity'] for t in ts),max(t['cosine_similarity'] for t in ts)],
            'kendall_range':[min(t['kendall_tau_b'] for t in ts),max(t['kendall_tau_b'] for t in ts)]})
    bins=[]
    for b,label in enumerate(['[0,0.2)','[0.2,0.5)','[0.5,0.8)','[0.8,1]']):
        rows=[r for r in common_rows if r['bin']==b]
        bins.append({'recall_bin':label,'rows':len(rows),
            **{key+'_count':sum(r[key] for r in rows) for key in ['admission_agreement','state_agreement','any_historical_certified_state']},
            **{key+'_rate':statistics.mean(r[key] for r in rows) if rows else None for key in ['admission_agreement','state_agreement']}})
    grid_summary=[]
    for method,j,guard in itertools.product(METHODS,[.5,.6,.7,.8],[0.,.02,.05,.1]):
        selected=[g for g in grid if (g['method'],g['jaccard'],g['allowed_recall_drop'])==(method,j,guard)]
        grid_summary.append({'method':method,'jaccard':j,'allowed_recall_drop':guard,
            'events':sum(g['events'] for g in selected),'eligible':sum(g['eligible'] for g in selected),
            'changed_states':sum(g['state_changed_from_primary'] for g in selected),
            'checkpoint_class_rows':150,
            'mean_seed_drift_rate':statistics.mean(g['events']/g['eligible'] for g in selected if g['eligible'])})
    stats=[]; compare_inputs=[]
    base=comparison_root
    for dataset in ['malaya','replayids_d2']:
        pairs=[]
        if dataset=='replayids_d2':
            pairs=primary_pairs(primary_diagnostic)
            diagnostic=json.loads(primary_diagnostic.read_text(encoding='utf-8'))
            compare_inputs.append({'file':primary_diagnostic.name,'sha256':sha(primary_diagnostic),
                'checkpoint_policy':'Job 425539 last epoch primary; never guarded Job 426307',
                'input_sha256':diagnostic['input_sha256']})
        else:
            export_registry=json.loads((base/'EXPORT_SHA256.json').read_text(encoding='utf-8'))
            for seed in SEEDS:
                p=base/f'{dataset}_seed{seed}_comparison.json'
                if sha(p)!=export_registry[p.name]:raise ValueError('Malaya comparison export hash mismatch')
                x=json.loads(p.read_text()); assert x['seed']==seed
                pairs.append(x); compare_inputs.append({'file':p.name,'sha256':sha(p),'ofra_result_sha256':x['ofra_result_sha256'],'baseline_result_sha256':x['baseline_result_sha256']})
        for n,seeds in [(5,SEEDS),(4,SEEDS[:4])]:
            for metric in pairs[0]['metrics']:
                if not all(metric in p['metrics'] for p in pairs): continue
                deltas=[p['metrics'][metric]['ofra_minus_replay_pp'] for p in pairs[:n]]
                stats.append({'dataset':dataset,'seeds':seeds,'n':n,'metric':metric,
                    'checkpoint_policy':'registered last epoch primary' if dataset=='replayids_d2' else 'registered Malaya primary',
                    'differences_pp':deltas,'mean_difference_pp':statistics.mean(deltas),
                    'sample_sd_pp':statistics.stdev(deltas),'positive_pairs':sum(d>0 for d in deltas),
                    'negative_pairs':sum(d<0 for d in deltas),'two_sided_signflip_p':signflip_p(deltas)})
    for n in [5,4]:
        rows=[r for r in stats if r['n']==n]
        adjusted=holm([r['two_sided_signflip_p'] for r in rows])
        for row,p in zip(rows,adjusted):
            row['holm_adjusted_p']=p; row['family_size']=len(rows)
    result={'schema_version':'review_saved_evidence_v1','status':'retrospective_reanalysis_no_model_execution',
        'source_job':434747,'seeds':SEEDS,'ledger_rows':len(all_rows),'common_rows':len(common_rows),
        'ledger_primary_reproduction':'450 of 450 states and actions match historical records',
        'consensus_by_recall':bins,'alternative_metrics':alternative,
        'etg_top15_policy_grid':grid_summary,'etg_top15_seed_grid':grid,
        'statistical_comparisons':stats,'input_bindings':inputs,'comparison_bindings':compare_inputs,
        'statistical_scope':'Post-hoc descriptive paired sign-flip tests assume symmetric exchangeable differences. Holm family covers all listed metrics across both datasets separately for n=5 and n=4. No confirmatory significance or architecture-only effect is established; n=4 excludes screening seed 42.',
        'consensus_scope':'Descriptive recall bins fixed in this analyzer before execution. Checkpoint-class rows share seeds/classes and are not independent replicates. Agreement is not correctness. No thresholds fitted to these results.',
        'grid_scope':'Replays historical state machine at fixed top-15 and fixed measured deletion admissions; varying top-k admission requires new masked evaluations. No model, SHAP, repair or operational experiment ran.',
        'historical_semantics':'CERTIFIED_STABLE can coexist with low class recall; certification only denotes the historical explanation rule, not predictive safety. New display labels do not alter stored states.',
        'metadata_provenance':'Historical package versions and probe hash recovered read-only from protected analysis protocols; originals verified before redacted metadata export. Public capsule is a provenance attestation, not the byte-identical source file.'}
    dump(out/'AUDIT_SUMMARY.json', result)
    dump(out/'VERSIONED_ETG_ROWS.json', all_rows)
    dump(out/'CONSENSUS_ROWS.json', common_rows)
    dump(out/'ANALYSIS_BINDINGS.json', {'analyzer_sha256':sha(Path(__file__)),'historical_analyzer_sha256':sha(source),
         'files':{p.name:sha(p) for p in [out/'AUDIT_SUMMARY.json',out/'VERSIONED_ETG_ROWS.json',out/'CONSENSUS_ROWS.json']}})
    print(json.dumps({'rows':len(all_rows),'bins':bins,'alternatives':alternative,
        'statistical_tests':len(stats),'min_adjusted_p':min(s['holm_adjusted_p'] for s in stats)}))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--evidence-root',type=Path,required=True)
    p.add_argument('--comparison-root',type=Path,required=True)
    p.add_argument('--historical-analyzer',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--primary-diagnostic',type=Path,required=True,
                   help='Output of replayids_primary_diagnostic.py after its source checks; guarded comparison exports are not accepted.')
    args=p.parse_args(); run(args.evidence_root,args.comparison_root,args.historical_analyzer,args.output,args.primary_diagnostic)
