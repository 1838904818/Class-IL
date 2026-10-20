"""Generate manuscript tables directly from completed, saved result JSON.

No training, inference, test-set parameter selection, or weight auditing occurs
here. Existing Day1 analysis is reused; additional completed cells are included
once, using the earliest timestamp, never their score. Partial groups remain
descriptive and never satisfy the original five-seed decision rule.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import statistics as st
import sys
import numpy as np
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PAPER = Path(__file__).resolve().parent
DATASETS = ['NSL-KDD', 'UNSW-NB15', 'CIC-IDS-2017']
METHODS = ['replay', 'derpp', 'wacb', 'freq', 'replay_la', 'ts_mem', 'aa_noattmem', 'aacil', 'replay_bs', 'joint']
NAMES = {'replay':'Replay', 'derpp':'DER++', 'wacb':'WACB', 'freq':'Freq', 'replay_la':'Replay+LA',
         'aacil':'AA-CIL', 'ts_mem':'TS-mem', 'aa_noattmem':'AA-noAttMem', 'replay_bs':'Replay-BS (post hoc)', 'joint':'Joint (offline)'}
SHORT = {'NSL-KDD':'NSL', 'UNSW-NB15':'UNSW', 'CIC-IDS-2017':'CIC17'}
SEEDS = [1, 2, 3, 4, 42]
ALPHAS = ['0.01', '0.02', '0.05']
BASELINES = ['replay','derpp','wacb','freq','replay_la','ts_mem','aa_noattmem']
EXTERNAL = BASELINES[:5]
RETENTION = ['aacil_kd','aacil_iso','aacil_wa','aacil_cbf','aacil_ncm']
RETENTION_NAMES = {m:'AA-CIL-'+m.removeprefix('aacil_').upper() for m in RETENTION}
RETENTION_METRICS = ['average_incremental_accuracy','final_balanced_accuracy',
                     'forgetting_all_old_classes','final_attack_detection','final_macro_f1','final_normal_fpr']
COMPACT_NAMES = {'aacil_kd_weak':'KD-weak', 'aacil_kd_wa':'KD+WA',
                 'aacil_kd_cbf':'KD+CBF', 'aacil_ema_kd':'EMA-KD', 'aacil_dual':'DUAL'}
COMPACT_DATASETS = ['UNSW-NB15', 'NSL-KDD']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def summary(vals):
    vals = list(vals)
    if not vals or any(x is None for x in vals):
        return {'n':len(vals), 'mean':None, 'sample_sd':None}
    return {'n':len(vals), 'mean':st.mean(vals), 'sample_sd':st.stdev(vals) if len(vals)>1 else None}


def defined_precision_summary(vals):
    """Section29.7: preserve nulls, with an explicit precision denominator."""
    vals = list(vals)
    defined = [value for value in vals if value is not None]
    return {'n':len(vals), 'defined_n':len(defined), 'values':vals,
            'mean':st.mean(defined) if defined else None,
            'sample_sd':st.stdev(defined) if len(defined)>1 else None}


def fmt(s, scale=100, digits=2):
    if s['mean'] is None:
        return '--'
    mean = f"{s['mean']*scale:.{digits}f}"
    return mean if s['sample_sd'] is None else mean + r'$\pm$' + f"{s['sample_sd']*scale:.{digits}f}"


def metric(raw):
    return {'normal_fpr':raw['normal_fpr'], 'attack_detection':raw['attack_recall'],
            'macro_f1':raw['macro_f1'], 'balanced_accuracy':raw['balanced_accuracy'],
            'normal_recall':1-raw['normal_fpr'],
            'attack_guard_recall':raw['attack_family_macro_recall_support_ge20'],
            'attack_family_macro_recall':raw['attack_family_macro_recall'],
            'per_class':raw['per_class'], 'confusion_matrix':raw['confusion_matrix']}


def from_raw(path):
    x = read(path)
    if not x.get('completed') or not x.get('checkpoints'):
        return None
    ident, cp = x['identity'], x['checkpoints'][-1]
    cfg = ident['effective_config']
    if cfg.get('smoke'):
        return None
    costs = {k.removesuffix('_total'):v for k,v in x.items() if k.endswith('_seconds_total')}
    traj=[]
    for q in x['checkpoints']:
        a=q['allocation']
        traj.append({'task':q['task_index'], 'actual_total':a['actual_total'], 'capacity':a['capacity_total'],
                     'replay_storage':a.get('replay_memory_storage',{}),
                     'normal_training_storage':a.get('normal_gate_training_storage',a.get('normal_training_storage',{})),
                     'calibration_storage':q.get('calibration_pool_storage',{}),
                     'gate_steps':q.get('gate_training_schedule',{}).get('optimizer_steps'),
                     'family_steps':q.get('family_training_schedule',{}).get('optimizer_steps')})
    return {'dataset':ident['dataset'], 'method':ident['method'], 'seed':ident['seed'], 'physical_method':ident['method'],
            'completed_at':x['completed_at'], 'source_result':path.relative_to(ROOT).as_posix(),
            'original_source_result':x.get('publication_derivation',{}).get('original_result',path.relative_to(ROOT).as_posix()),
            'original_source_result_sha256':x.get('publication_derivation',{}).get('original_result_sha256'),
            'training_source_commit':ident.get('git_commit'), 'costs':costs, 'trajectory':traj,
            'points':{a:{'status':'available','metrics':metric(cp['operating_points'][a])} for a in ALPHAS},
            'effective_config':cfg, 'class_order':x['class_order'], 'tasks':x['tasks']}


def paired(a,b,key,alternative='greater',alpha='0.02'):
    aa={x['seed']:x['points'][alpha]['metrics'][key] for x in a}
    bb={x['seed']:x['points'][alpha]['metrics'][key] for x in b}
    seeds=[s for s in SEEDS if s in aa and s in bb]
    vals=[100*(aa[s]-bb[s]) for s in seeds]
    out={'seeds':seeds, 'complete_five_seed':seeds==SEEDS, 'delta_pp':vals, **summary(vals)}
    out['paired_d_z']=out['mean']/out['sample_sd'] if out['sample_sd'] else None
    out['p_one_sided']=float(ttest_rel([aa[s] for s in seeds],[bb[s] for s in seeds],alternative=alternative).pvalue) if len(seeds)>1 else None
    out['alternative']=alternative
    return out


def table(name,caption,header,rows,cols=None,wide=True,note=''):
    env='table*' if wide else 'table'
    cols=cols or 'l'*len(header)
    text=[r'\begin{'+env+r'}[t]',r'\centering\scriptsize',r'\setlength{\tabcolsep}{3.8pt}',
          r'\caption{'+caption+r'}\label{tab:'+name.replace('_','-')+'}',r'\begin{tabular}{'+cols+'}',r'\toprule',
          ' & '.join(header)+r' \\',r'\midrule']
    text += [' & '.join(map(str,r))+r' \\' for r in rows]
    text += [r'\bottomrule',r'\end{tabular}']
    if note: text += [r'\par\vspace{2pt}\begin{minipage}{.98\linewidth}\footnotesize '+note+r'\end{minipage}']
    text += [r'\end{'+env+'}']
    (PAPER/'generated'/f'{name}.tex').write_text('\n'.join(text)+'\n',encoding='utf-8')


def architecture():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from matplotlib.path import Path as MplPath
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig, ax=plt.subplots(figsize=(7.0,3.05));ax.set(xlim=(0,10.35),ylim=(0,4));ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05',facecolor=color,edgecolor='#39424c',linewidth=.8))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center')
    def arrow(a,b,label=None):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,color='#444444',linewidth=.9))
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',fontsize=8)
    box(.15,2.7,2.7,.9,'Refreshed benign TRAIN\nOutside replay budget M','#d9edf7')
    box(.15,1.3,2.7,.9,'Current attacks + retained\nold-attack memory (one copy)','#fff0ce')
    box(.15,.05,2.7,.8,'Permanent benign holdout\nCalibration only; outside M','#e8e3f0')
    box(4,2.35,2.2,1.0,'Binary gate G\nBenign / attack','#bde3ee')
    box(4,.9,2.2,1.0,'Family classifier F\nAttack classes only','#f8d88c')
    box(7.3,2.35,2.4,1.0,'Margin > threshold?\nNo: Normal','#e6e6e6')
    box(7.3,.9,2.4,1.0,'Yes: attack-family\nprediction from F','#e2efdb')
    arrow((2.9,3.15),(3.95,2.95));arrow((2.9,1.75),(3.95,2.6));arrow((2.9,1.65),(3.95,1.4))
    calibration_path=MplPath([(2.9,.45),(10.1,.45),(10.1,2.85),(9.75,2.85)],
                             [MplPath.MOVETO,MplPath.LINETO,MplPath.LINETO,MplPath.LINETO])
    ax.add_patch(FancyArrowPatch(path=calibration_path,arrowstyle='-|>',mutation_scale=10,
                                color='#6b5486',linewidth=1.0,linestyle='--'))
    ax.text(6.4,.15,'FPR calibration only (no optimization)',ha='center',fontsize=8,color='#6b5486')
    arrow((6.25,2.85),(7.25,2.85));arrow((6.25,1.4),(7.25,1.4));arrow((8.5,2.3),(8.5,1.95))
    fig.tight_layout(pad=.1);fig.savefig(PAPER/'figures'/'architecture.pdf',bbox_inches='tight');plt.close(fig)


def section39_execution_matches(execution, method, seed):
    invocation=execution.get('invocation',{})
    cell=execution.get('cell_attempt',{})
    return (invocation.get('actual_entrypoint')=='ncd/run_section39_local.py'
        and invocation.get('dataset')=='CIC-IDS-2017'
        and invocation.get('method')==method and invocation.get('seed')==seed
        and invocation.get('descriptive_development_dataset') is True
        and invocation.get('selection_changed') is False
        and invocation.get('CSE18_model_metrics_accessed') is False
        and invocation.get('smoke') is False
        and invocation.get('launcher_commit')=='a4d55cd8e8d4087ea67f33a26535c721547483c3'
        and invocation.get('launcher_sha256')=='b0a76a0dbb8ef7d2acd488e584eedd4219990b917b10bf576e4bb82bbdedded5'
        and cell.get('method')==method and cell.get('seed')==seed
        and cell.get('exit_code')==0)


def require_current_comparison_inputs(comparison):
    """Allow a verified append without rerunning unchanged historical tests.

    A changed clock alone is not a changed dependency. The exception requires
    unchanged protocol, saved group values, paired scalar inputs, comparator
    means, seed availability and recorded source/configuration bindings.
    """
    summary_path=ROOT/comparison['source_summary']
    current=read(summary_path)
    if current['completed_at']==comparison['source_summary_completed_at']:
        return
    def fail():
        raise ValueError('Historical comparison inputs changed; regenerate the affected comparison')
    if current['protocol']!=comparison['stagewise_protocol']:
        fail()
    prior_time=dt.datetime.fromisoformat(comparison['source_summary_completed_at'])
    for cell in current['cells']:
        if dt.datetime.fromisoformat(cell['completed_at'])<=prior_time:
            continue
        if cell['method'] in COMPACT_NAMES and cell['dataset'] in COMPACT_DATASETS:
            continue
        # Section39 adds only CIC17 KDWA and the previously missing BS seed42.
        # This permits their presence, not changed dependencies: all group,
        # pair, comparator, count and source checks below must still pass.
        if cell['dataset']=='CIC-IDS-2017' and (
                cell['method']=='aacil_kd_wa' or
                (cell['method']=='replay_bs' and cell['seed']==42)):
            path=summary_path.parent/'cic_ids_2017'/f"{cell['method']}_seed{cell['seed']}"/'result.json'
            execution=read(path).get('section39_execution',{})
            if section39_execution_matches(execution,cell['method'],cell['seed']):
                continue
        fail()
    groups={(g['dataset'],g['method']):g for g in current['groups']}
    cells={(c['dataset'],c['method'],c['seed']):c for c in current['cells']}
    rows={}
    def row(dataset, method, seed):
        key=(dataset,method,seed)
        if key not in rows:
            if key not in cells:
                fail()
            path=summary_path.parent/dataset.lower().replace('-','_')/f'{method}_seed{seed}'/'result.json'
            r=read(path); entry=cells[key]
            if (not r.get('completed') or (r['dataset'],r['method'],r['seed'])!=key
                    or r['protocol']!=current['protocol'] or r['source_result']!=entry['source_result']
                    or r['completed_at']!=entry['completed_at']):
                fail()
            rows[key]=r
        return rows[key]
    descriptive=comparison.get('all_versions_descriptive',[])
    if comparison.get('section')=='29.9':
        descriptive=[dict(d,method=comparison['candidate']) for d in comparison['dataset_results']]
    for old in descriptive:
        group=groups.get((old['dataset'],old['method']),{})
        if old['n']!=group.get('n',0) or old['seeds']!=group.get('seeds',[]):
            fail()
        for metric, value in old.get('metrics',{}).items():
            if value!=group.get('metrics',{}).get(metric):
                fail()
    tests=[t for d in comparison['dataset_results'] for t in d.get('tests',[])]+comparison.get('ISO_minus_KD_descriptive',[])
    for test in tests:
        ds, candidate, reference, metric=(test[k] for k in ('dataset','candidate','reference','metric'))
        cg, rg=groups.get((ds,candidate),{}), groups.get((ds,reference),{})
        paired_seeds=[s for s in SEEDS if s in cg.get('seeds',[]) and s in rg.get('seeds',[])]
        if paired_seeds!=[p['seed'] for p in test.get('paired_values',[])]:
            fail()
        for pair in test.get('paired_values',[]):
            if (row(ds,candidate,pair['seed'])['metrics'].get(metric)!=pair['candidate']
                    or row(ds,reference,pair['seed'])['metrics'].get(metric)!=pair['reference']):
                fail()
        selection=test.get('comparator_selection')
        if selection:
            now={m:groups[ds,m]['metrics'][metric]['mean'] for m in selection['external_set']
                 if groups.get((ds,m),{}).get('complete_five_seed')
                 and sorted(groups[ds,m].get('seeds',[]))==SEEDS}
            if now!=selection['five_seed_group_means']:
                fail()
    for d in comparison['dataset_results']:
        for method, count in d.get('available_group_counts',{}).items():
            if groups.get((d['dataset'],method),{}).get('n',0)!=count:
                fail()
    for binding in comparison.get('formal_selection_bindings',[])+comparison.get('source_bindings',[]):
        r=row(binding['dataset'],binding['method'],binding['seed'])
        if r['training_git_commit']!=binding['training_git_commit']:
            fail()
        if any(r.get('training_configuration',{}).get(k)!=v for k,v in binding.get('configuration',{}).items()):
            fail()
        if 'source_result' in binding and r['source_result']!=binding['source_result']:
            fail()
        if 'source_completed_at' in binding and {
                b.get('source_completed_at') for b in r['stage_bindings']}!={binding['source_completed_at']}:
            fail()


def retention_evidence(selection_path, comparison_path):
    """Read the frozen choice and the separately generated formal assessment.

    This presentation layer neither selects a version nor recalculates a test.
    Development metric values are deliberately not copied into formal tables.
    """
    out={'selection_complete':False, 'assessment_available':False, 'groups':[],
         'per_version':[], 'selected_main':None, 'verdict':'pending', 'dataset_results':[],
         'iso_minus_kd':[]}
    if not selection_path.exists():
        return out
    s=read(selection_path)
    if (not s.get('completed') or not s.get('selection_complete') or s.get('section')!='29.8'
            or s.get('selection_seeds')!=[5,6] or s.get('test_metrics_used_for_selection') is not False):
        raise ValueError('Section29.8 requires the complete validation-only selection.json')
    keep=lambda v:{k:v[k] for k in ['method','selected_candidate_id','eligible','selection_reason','configuration']}
    versions={v['method']:keep(v) for v in s['per_version']}
    if set(versions)!=set(RETENTION):
        raise ValueError('Section29.8 selection must retain all five versions')
    selected=keep(s['selected_main']) if s['selected_main'] is not None else None
    out.update(selection_complete=True, selection_completed_at=s['completed_at'],
        source_commit=s['source_commit'], selection_result=selection_path.relative_to(ROOT).as_posix(),
        per_version=[versions[m] for m in RETENTION], selected_main=selected,
        main_selection_status=s['main_selection_status'])
    if not comparison_path.exists():
        return out
    a=read(comparison_path)
    if (a.get('section')!='29.8' or a.get('selection_completed_at')!=s['completed_at']
            or a.get('selection_source_commit')!=s['source_commit']
            or a.get('selection_test_metrics_used') is not False
            or a.get('selected_main')!=s['selected_main']):
        raise ValueError('Regenerate Section29.8 comparisons for the current frozen selection')
    require_current_comparison_inputs(a)
    for g in a['all_versions_descriptive']:
        v=versions[g['method']]
        if g['selected_candidate_id']!=v['selected_candidate_id'] or g['development_eligible']!=v['eligible']:
            raise ValueError('Formal presentation does not match the selected version')
        full=(g.get('complete_five_seed') and g.get('n')==5 and sorted(g.get('seeds',[]))==sorted(SEEDS))
        out['groups'].append({**g, 'complete_five_seed':bool(full),
            'metrics':g['metrics'] if full else {k:None for k in RETENTION_METRICS}})
    out.update(assessment_available=True, comparison_result=comparison_path.relative_to(ROOT).as_posix(),
        assessment_completed_at=a['completed_at'], source_summary_completed_at=a['source_summary_completed_at'],
        verdict=a['overall_required_dataset_verdict'], dataset_results=a['dataset_results'],
        iso_minus_kd=a['ISO_minus_KD_descriptive'],
        required_selected_comparisons_complete=a['required_selected_comparisons_complete'],
        all_five_versions_report_complete=a['all_five_versions_required_dataset_report_complete'])
    return out


def retention_outputs(evidence):
    """Write only presentation tables; preserve original Section27.8 decisions."""
    appendix=retention_comparison_outputs(evidence)
    if not evidence['selection_complete']:
        return {'RetentionSelectionText':'The validation-only retention selection remains incomplete.',
                'RetentionFormalResultsText':'No complete formal retention group is presented.',
                'RetentionDecisionText':'The validation-selected retention protocol remains pending.',
                'RetentionComparisonAppendix':appendix}
    main=evidence['selected_main']
    if main:
        cfg=main['configuration']
        choice=(RETENTION_NAMES[main['method']]+rf" ($\lambda_G={cfg['lambda_g']:g}$")
        if main['method']=='aacil_cbf':
            choice+=f", {cfg['cbf_epochs']} CBF epoch(s), learning-rate multiplier {cfg['cbf_lr_multiplier']:g}"
        choice+=')'
        selection_text=('UNSW training-validation seeds 5/6 locked '+choice+' as the eligible main version. ')
    else:
        selection_text='No version met the development eligibility constraint; no main version was selected. '
    ineligible=[RETENTION_NAMES[v['method']] for v in evidence['per_version'] if not v['eligible']]
    selection_text+=(', '.join(ineligible)+' are reported as development-ineligible fallback configurations.'
        if ineligible else 'All five versions remain in the formal report.')
    rows=[]
    for v in evidence['per_version']:
        c=v['configuration']
        rows.append([RETENTION_NAMES[v['method']],f"{c['lambda_g']:g}",
            str(c['cbf_epochs']) if v['method']=='aacil_cbf' else '--',
            f"{c['cbf_lr_multiplier']:g}" if v['method']=='aacil_cbf' else '--',
            'eligible main' if main and v['method']==main['method'] else 'eligible alternative' if v['eligible'] else 'ineligible fallback'])
    table('retention_selection','Validation-selected retention protocol: frozen development configurations.',
          ['Version',r'$\lambda_G$','CBF epochs','CBF LR multiplier','Development role'],rows,'lrrrl',
          note='Selection uses UNSW training-validation seeds 5/6 only. DER weights are fixed at one. Ineligible versions are still reported on the five formal seeds, without converting fallback selection into eligibility.')
    group_map={(g['dataset'],g['method']):g for g in evidence['groups']}
    rows=[]; completed=[]
    datasets=['UNSW-NB15','NSL-KDD']+(['CIC-IDS-2017'] if any(g['dataset']=='CIC-IDS-2017' for g in evidence['groups']) else [])
    for ds in datasets:
        count=0
        for m in RETENTION:
            g=group_map.get((ds,m),{}); full=g.get('complete_five_seed',False)
            count+=int(full)
            rows.append([SHORT[ds],RETENTION_NAMES[m],g.get('n',0)]+[
                fmt(g['metrics'][k]) if full and g['metrics'].get(k) else '--' for k in RETENTION_METRICS])
        completed.append(f'{SHORT[ds]} {count}/5 versions')
    table('retention_results','Retention variants at stagewise Normal-holdout calibration target $\\alpha=2\\%$.',
          ['Data','Version','n','Mean inc. BAcc','Final BAcc','Forget (pp)','ADR','MF1','FPR'],rows,'llrrrrrrr',
          note='Only complete seeds 1/2/3/4/42 receive means and sample SD; incomplete groups show n and dashes. All outcomes come from formal test trajectories, never development selection metrics. Signed forgetting includes old Normal; rates/accuracies are percentages. Eligibility is a development property, not formal success.')
    formal_text=('Complete formal five-seed groups: '+', '.join(completed)+'.')
    iso=group_map.get(('UNSW-NB15','aacil_iso'),{})
    kd=group_map.get(('UNSW-NB15','aacil_kd'),{})
    contrasts={c['metric']:c for c in evidence['iso_minus_kd'] if c['dataset']=='UNSW-NB15'}
    endpoints=['forgetting_all_old_classes','average_incremental_accuracy']
    if (iso.get('complete_five_seed') and kd.get('complete_five_seed') and
            all(contrasts.get(k,{}).get('status')=='complete' and contrasts[k].get('n')==5 for k in endpoints)):
        def probability_tex(value):
            parts=f'{value:.3g}'.split('e')
            return parts[0] if len(parts)==1 else parts[0]+r'\times10^{'+str(int(parts[1]))+'}'
        forget,incremental=(contrasts[k] for k in endpoints)
        formal_text+=(' On UNSW-NB15, AA-CIL-ISO recorded signed forgetting of '
            +fmt(iso['metrics']['forgetting_all_old_classes'],100,3)+' pp alongside final balanced accuracy of '
            +fmt(iso['metrics']['final_balanced_accuracy'])+r'\%. '
            +'ISO minus KD differences were '+fmt(forget['delta_pp'],1)+' pp for forgetting '
            +r'($p_{<}='+probability_tex(forget['p_less'])+r'$) and '
            +fmt(incremental['delta_pp'],1)+' pp for mean incremental balanced accuracy '
            +r'($p_{<}='+probability_tex(incremental['p_less'])+r'$), using five paired seeds and unadjusted one-sided tests. ')
    verdict=evidence['verdict']
    decision=('The validation-selected retention protocol fails a necessary condition; the false-alarm--forgetting--accuracy trade-off remains unresolved.' if verdict=='fail'
              else 'The selected main satisfies the three retention conditions on both required datasets.' if verdict=='pass'
              else 'The validation-selected retention protocol remains pending on UNSW and NSL.')
    return {'RetentionSelectionText':selection_text,'RetentionFormalResultsText':formal_text,
            'RetentionDecisionText':decision,'RetentionComparisonAppendix':appendix}


def retention_comparison_outputs(evidence):
    """Render useful complete comparisons already computed in Section29.8."""
    useful=lambda c: c.get('n')==5 and c.get('status') in ['complete','unavailable'] and c.get('delta_pp',{}).get('mean') is not None
    def pvalue(p):
        if p is None:
            return '--'
        if 0 < p < .0001 or .99999 < p < 1:
            value = p if p < .5 else 1-p
            mantissa, exponent = f'{value:.3e}'.split('e')
            return '$'+('1-' if p > .5 else '')+mantissa+r'\times10^{'+str(int(exponent))+'}$'
        return f'{p:.5f}'
    rows=[]
    for d in evidence['dataset_results']:
        for t in d['tests']:
            if not useful(t):continue
            rows.append([SHORT[d['dataset']],str(t['criterion']),
                NAMES.get(t.get('reference'),RETENTION_NAMES.get(t.get('reference'),'--')),
                t['n'],fmt(t['delta_pp'],1),pvalue(t.get('p_'+t['p_direction'])),t['decision']])
    iso_rows=[]
    for c in evidence['iso_minus_kd']:
        if not useful(c):continue
        iso_rows.append([SHORT[c['dataset']], 'Forget (all)' if c['metric']=='forgetting_all_old_classes' else 'Mean inc. BAcc',
            c['n'],fmt(c['delta_pp'],1),pvalue(c.get('p_less')),pvalue(c.get('p_greater')),pvalue(c.get('p_two_sided'))])
    definitions=[('retention_necessary_tests',rows,
        'Necessary comparisons under the validation-selected retention protocol.',
        ['Data','Criterion','Reference','n',r'$\Delta$ (pp)','One-sided $p$','Decision'],'llrrrll',
        r'Candidate minus reference; exact five paired formal seeds. 1: all-old forgetting, $p_{>}<.05$ means harm/failure against the lowest-forgetting external baseline. '
        r'2: final BAcc, $p_{>}<.05$ is required against Replay-BS. 3: ADR, $p_{<}<.05$ means harm/failure against original AA-CIL. '
        r'Only useful complete comparisons are shown; absent comparisons remain pending and are not passed. Both UNSW and NSL are required. No detected harm is not equivalence. Unadjusted tests.'),
        ('retention_iso_kd',iso_rows,'Completed descriptive AA-CIL-ISO minus AA-CIL-KD comparisons.',
        ['Data','Metric','n',r'$\Delta$ (pp)',r'$p_{<}$',r'$p_{>}$',r'$p_{2s}$'],'llrrrrr',
        r'Exact five paired formal seeds; mean$\pm$sample SD of differences in percentage points. These unadjusted comparisons neither select the main version nor add a success requirement. '
        r'Undefined tests remain --; no seed is discarded. The original superiority and earlier retention decisions are unchanged.')]
    included=[]
    for name,values,caption,header,cols,note in definitions:
        if values:
            table(name,caption,header,values,cols,note=note)
            included.append(r'\input{generated/'+name+'.tex}')
        else:
            # Replace an earlier generated table with a non-rendering marker,
            # rather than retaining stale comparisons or printing an empty table.
            (PAPER/'generated'/f'{name}.tex').write_text('% No complete useful comparison in this snapshot.\n',encoding='utf-8')
    return (r'\section{Retention-extension paired comparisons}'+''.join(included)) if included else ''


def compact_evidence(selection_path, comparison_path):
    """Present the frozen compact selection and saved development-dataset groups.

    A CSE result is displayed only after revalidating its saved all35/all5
    release bindings. No tests, inference, or selection are performed here.
    """
    out={'selection_complete':False, 'groups':[], 'per_version':[],
         'selected_main':None, 'cse_status':'sealed_not_consumed'}
    if not selection_path.is_file():
        return out
    s=read(selection_path)
    if (s.get('completed') is not True or s.get('selection_complete') is not True
            or s.get('selection_datasets')!=COMPACT_DATASETS
            or s.get('selection_seeds')!=[5,6] or s.get('formal_seeds')!=SEEDS
            or s.get('test_metrics_used_for_selection') is not False):
        raise ValueError('Compact presentation requires the complete validation-only selection')
    keep=('method','selected_candidate_id','configuration','eligible','selection_reason')
    versions={v['method']:{k:v[k] for k in keep+('average_validation_final_bacc',
        'average_validation_forgetting','forgetting_constraint_by_dataset')} for v in s['per_version']}
    if set(versions)!=set(COMPACT_NAMES):
        raise ValueError('The immutable compact selection must retain all five candidates')
    main={k:s['selected_main'][k] for k in keep}
    if any(main[k]!=versions[main['method']][k] for k in keep):
        raise ValueError('Compact main differs from its immutable per-version choice')
    capacity=[]
    for v in s['per_version']:
        for ds in COMPACT_DATASETS:
            c=v['datasets'][ds]['source_results'][0]['architecture_capacity']
            capacity.append({'dataset':ds,'method':v['method'], **{k:c[k] for k in
                ('training_parameters','inference_parameters','optimizer_parameters','extra_components')}})
    out.update(selection_complete=True, selection_result=selection_path.relative_to(ROOT).as_posix(),
        selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        selection_completed_at=s['completed_at'], source_commit=s['source_commit'],
        main_selection_status=s['main_selection_status'], selected_main=main,
        per_version=[versions[m] for m in COMPACT_NAMES], capacity=capacity,
        forgetting_ceiling_by_dataset=s['forgetting_ceiling_by_dataset'])
    if not comparison_path.is_file():
        return out
    a=read(comparison_path)
    if (a.get('section')!='31' or a.get('selection_sha256')!=out['selection_sha256']
            or a.get('selection_test_metrics_used') is not False or a.get('selected_main')!=main):
        raise ValueError('Regenerate compact comparisons for the immutable selection')
    if a.get('cse_release_guard') is not None:
        from paper_aacil.generate_section31_comparisons import cse_release_guard, verify_saved_bindings
        guard=a['cse_release_guard']
        reference_path=ROOT/guard['reference_summary']
        _,_,current=cse_release_guard(reference_path.parent)
        selected_path=ROOT/guard['selected_summary']
        selected_path.resolve().relative_to(ROOT)
        selected=read(selected_path)
        if (any(guard[k]!=v for k,v in current.items())
                or hashlib.sha256(selected_path.read_bytes()).hexdigest()!=guard['selected_summary_sha256']
                or selected.get('completed') is not True or selected.get('smoke') is not False
                or selected.get('physical_cells')!=5 or selected.get('seeds')!=SEEDS
                or selected.get('test_selected') is not False
                or a.get('CSE18_all_required_evidence_complete') is not True
                or len(a.get('CSE18_necessary_tests',[]))!=4
                or a.get('cse_release_append',{}).get('previous_groups_recomputed') is not False):
            raise ValueError('CSE presentation requires both intact released groups and saved registered tests')
        for b in selected['source_results']:
            p=selected_path.parent/b['path']
            p.resolve().relative_to(selected_path.parent.resolve())
            if hashlib.sha256(p.read_bytes()).hexdigest()!=b['sha256']:
                raise ValueError('Selected released cell changed')
        verify_saved_bindings(a['source_bindings'])
        cse_groups=[g for g in a['descriptive_groups'] if g['dataset']=='CSE-CIC-IDS2018']
        if len(cse_groups)!=9 or any(g['n']!=5 or g['seeds']!=SEEDS for g in cse_groups):
            raise ValueError('All nine logical CSE groups required')
        out.update(cse_status='released_assessed',cse_result={
            'descriptive_groups':cse_groups,'CSE18_necessary_tests':a['CSE18_necessary_tests'],
            'CSE18_necessary_verdict':a['CSE18_necessary_verdict'],
            'release_guard':guard})
        first=read(selected_path.parent/selected['source_results'][0]['path'])
        raw=read(selected_path.parent/selected['source_results'][0]['path'].rsplit('/',1)[0]/first['raw_result'])
        out['cse_protocol']={'identity':first['identity'],
                             'calibration_storage':raw['checkpoints'][0]['calibration_pool_storage']}
    if any(g['dataset'] not in COMPACT_DATASETS+['CIC-IDS-2017','CSE-CIC-IDS2018']
           for g in a.get('descriptive_groups',[])):
        raise ValueError('Unexpected compact dataset')
    for g in a.get('descriptive_groups',[]):
        if g['dataset']=='CSE-CIC-IDS2018':
            if out['cse_status']!='released_assessed':
                raise ValueError('CSE group without verified release')
            continue
        if g['dataset']=='CIC-IDS-2017':
            if (g['method']!='aacil_kd_wa' or g.get('role')!='development_dataset_descriptive'
                    or g.get('used_for_frozen_selection') is not False
                    or g.get('blind_confirmation') is not False
                    or a.get('section39_append',{}).get('selection_changed') is not False):
                raise ValueError('CIC17 must retain its Section39 descriptive-only role')
            capacities=[r['architecture_capacity'] for r in g['resources_by_seed']]
            if len(capacities)!=5 or any(c!=capacities[0] for c in capacities):
                raise ValueError('CIC17 formal capacity differs across seeds')
            out['capacity'].append({'dataset':g['dataset'],'method':g['method'],**capacities[0]})
        v=versions[g['method']]
        if (g.get('n')!=5 or sorted(g.get('seeds',[]))!=SEEDS or not g.get('complete_five_seed')
                or g.get('selected_candidate_id')!=v['selected_candidate_id']
                or g.get('development_eligible')!=v['eligible']):
            raise ValueError('Compact formal table accepts only complete fixed-configuration groups')
        out['groups'].append(g)
    out.update(comparison_result=comparison_path.relative_to(ROOT).as_posix(),
        comparison_completed_at=a['completed_at'])
    return out


def compact_outputs(evidence):
    if not evidence['selection_complete']:
        return {'CompactSelectionText':'The compact-family validation selection is incomplete.',
                'CompactResultsText':'No complete compact-family group is presented.',
                'CompactValidationTable':'', 'CompactFormalTables':'', 'CompactCapacityAppendix':''}
    main=evidence['selected_main']; c=main['configuration']; name=COMPACT_NAMES[main['method']]
    eligible=sum(v['eligible'] for v in evidence['per_version'])
    text=(f'The fixed-capacity retention protocol completed validation-only selection at '
          +evidence['selection_completed_at'].replace('T',' ')+'. ')
    text+=(f'{eligible} of five candidates satisfy the joint forgetting constraint. ' if eligible else
           'None of the five candidates satisfies the forgetting constraint on both development datasets. ')
    text+=(name+(' is the eligible validation-selected version' if main['eligible'] else
        ' is retained only as an ineligible descriptive fallback with the lowest average signed validation forgetting')
        +r', with $\lambda_G='+f"{c['lambda_g']:g}"+r'$ and family DER coefficients $\alpha='
        +f"{c['der_alpha']:g}"+r'$, $\beta='+f"{c['der_beta']:g}"+'$. ')
    if not main['eligible']:
        text+='This selection did not meet the retention requirement and does not resolve the trade-off.'
    rows=[]
    for v in evidence['per_version']:
        c=v['configuration']; flags=v['forgetting_constraint_by_dataset']
        rows.append([COMPACT_NAMES[v['method']], f"{c['lambda_g']:g}",
            f"{c['der_alpha']:g}/{c['der_beta']:g}", f"{100*v['average_validation_final_bacc']:.2f}",
            f"{100*v['average_validation_forgetting']:.2f}",
            'yes' if flags['UNSW-NB15'] else 'no', 'yes' if flags['NSL-KDD'] else 'no'])
    ceilings=evidence['forgetting_ceiling_by_dataset']
    table('compact_selection','Fixed-capacity protocol: validation-only choices, not formal test results.',
        ['Candidate',r'$\lambda_G$',r'DER $\alpha/\beta$','Mean val. BAcc','Mean val. forget','UNSW','NSL'],rows,
        cols='lrrrrcc',note=(r'Seeds 5/6 only. BAcc is in percent and signed forgetting in percentage points; means average the two development datasets. '
        'The last two columns indicate each dataset\'s forgetting constraint. Ceilings are '
        +f"{100*ceilings['UNSW-NB15']:.2f} pp (UNSW) and {100*ceilings['NSL-KDD']:.2f} pp (NSL), "
        'each equal to its lowest external-five validation forgetting plus 1 pp. Negative values are not clipped. '
        'Rows with no eligible configuration are fixed descriptive fallbacks.'))
    groups=evidence['groups']
    formal=''
    if groups:
        rows=[[SHORT[g['dataset']], COMPACT_NAMES[g['method']], '5']+
            [fmt(g['metrics'][k]) for k in RETENTION_METRICS] for g in groups]
        table('compact_results',r'Fixed-capacity candidates on development datasets at stagewise $\alpha=2\%$.',
            ['Data','Candidate','$n$','Mean inc. BAcc','Final BAcc','Forget (pp)','ADR','MF1','FPR'],
            rows,cols='llrrrrrrr',note=(r'Mean$\pm$sample SD over complete seeds 1/2/3/4/42. '
            'UNSW, NSL and CIC17 were used for development; these five-seed results are not blind confirmation. '
            'CIC17 was not used in the frozen UNSW/NSL validation-only selection; only its selected candidate was added. '
            'Only complete groups are displayed. No test outcome changes the fixed selection or fallback eligibility.'))
        formal=r'\input{generated/compact_results.tex}'
    capacity=[[SHORT[r['dataset']],COMPACT_NAMES[r['method']],str(r['training_parameters']),
        str(r['optimizer_parameters']),str(r['inference_parameters']),str(r['extra_components'])]
        for r in evidence['capacity']]
    table('compact_capacity','Fixed family-model capacity of the selected per-candidate configurations.',
        ['Data','Candidate','Training-resident','Optimizer','Inference','Extra components'],capacity,cols='llrrrr',
        note=('F only; the common G and transient previous-task gate teacher are separate. Counts come from saved development architectures '
        'and remain fixed across tasks. EMA training-resident count includes its slow copy; only EMA is used at inference. '
        'DUAL uses two fixed encoders and one joint head within a single F.'))
    development_count=sum(g['dataset'] in COMPACT_DATASETS for g in groups)
    progress=(f'{development_count} of ten UNSW/NSL five-seed candidate/dataset groups are complete. ' if groups else
              'No compact candidate has a complete formal five-seed group at this snapshot. ')
    if any(g['dataset']=='CIC-IDS-2017' for g in groups):
        progress+='The additional CIC17 KD+WA five-seed group is descriptive, not part of validation selection. '
    progress+='UNSW, NSL and CIC17 remain development datasets. '
    progress+=('The separately released CSE-CIC-IDS2018 assessment is reported below; it does not change the frozen selection.'
               if evidence['cse_status']=='released_assessed' else
               'CSE-CIC-IDS2018 metrics are sealed and were not read by this presentation; no external confirmation is available.')
    return {'CompactSelectionText':text,'CompactResultsText':progress,
        'CompactValidationTable':r'\input{generated/compact_selection.tex}', 'CompactFormalTables':formal,
        'CompactCapacityAppendix':r'\section{Fixed-capacity family counts}\input{generated/compact_capacity.tex}'}


def compact_summary_zh(evidence, aggregate_text):
    if not evidence['selection_complete']:
        return []
    main=evidence['selected_main']; c=main['configuration']
    role='合格的验证选定版本' if main['eligible'] else '未满足双数据集遗忘约束的描述性回退，不是合格主方法'
    lines=['', '**固定容量保持协议：** '+COMPACT_NAMES[main['method']]+'为'+role+
        f"（λG={c['lambda_g']:g}，DER α/β={c['der_alpha']:g}/{c['der_beta']:g}）。仅用UNSW/NSL seeds5/6及验证集选择；配置已冻结。"]
    if not any(v['eligible'] for v in evidence['per_version']):
        lines.append('五个候选均未同时满足两数据集的验证遗忘约束，不能宣称权衡已解决。')
    if evidence['groups']:
        lines+=['', '| 开发数据集 | 紧凑候选 | n | 增量 BAcc (%) | 最终 BAcc (%) | 遗忘 (pp) | ADR (%) |',
            '|---|---|---:|---:|---:|---:|---:|']
        for g in evidence['groups']:
            lines.append('| '+' | '.join([SHORT[g['dataset']],COMPACT_NAMES[g['method']],'5']+
                [aggregate_text(g['metrics'][k]) for k in RETENTION_METRICS[:4]])+' |')
    else:
        lines.append('正式五seed完整组尚未纳入；不以开发验证数字填充正式结果。')
    lines+=['UNSW/NSL/CIC17正式结果仍属开发数据集结果，不是盲确认；CIC17未参与冻结的UNSW/NSL验证选型。'
            +('CSE18已按完整35参考单元及5候选单元解封，判定如下。' if evidence['cse_status']=='released_assessed'
              else 'CSE18汇总封存、未读取。')+'旧CBF失败与固定KD复评失败不变。', '']
    return lines


def cse_presentation(evidence):
    """Format saved CSE evidence only; never call the statistical test code."""
    if evidence.get('cse_status')!='released_assessed':
        return {'CSEAbstractText':r'\RetentionAbstractText','CSEConclusionText':'',
                'CSEConfirmationText':'CSE18 remains sealed.', 'CSEConfirmationTables':'', 'CSEProtocolText':''}
    from paper_aacil.generate_section31_comparisons import render
    result=evidence['cse_result']
    groups={g['method']:g for g in result['descriptive_groups']}
    main=groups[evidence['selected_main']['method']]
    tests=result['CSE18_necessary_tests']
    if result['CSE18_necessary_verdict']!='fail' or [t['decision'] for t in tests]!=['pass','fail','fail','pass']:
        raise ValueError('Revise the evidence-bounded narrative for a different saved decision; do not reuse this failure text')
    bacc=100*main['metrics']['final_balanced_accuracy']['mean']
    forget=100*main['metrics']['forgetting_all_old_classes']['mean']
    adr=100*main['metrics']['final_attack_detection']['mean']
    fpr=100*main['metrics']['final_normal_fpr']['mean']
    bs=-tests[1]['delta_pp']['mean']; aa=-tests[2]['delta_pp']['mean']
    identity=evidence['cse_protocol']['identity']; binding=identity['data_binding']
    calibration=evidence['cse_protocol']['calibration_storage']
    tasks='; '.join(', '.join(identity['class_order'][c] for c in task) for task in identity['tasks'])
    (PAPER/'generated/cse_confirmation.tex').write_text(render(result),encoding='utf-8',newline='\n')
    short=(f'On the previously withheld CSE-CIC-IDS2018 assessment, the fixed KD+WA candidate reaches '
           f'{bacc:.2f}'+r'\% final balanced accuracy, '+f'{forget:.2f} pp signed forgetting and '
           f'{adr:.2f}'+r'\% attack detection at '+f'{fpr:.2f}'+r'\% test FPR. '
           f'Its final accuracy is {bs:.2f} pp below the exposure-matched single classifier and {aa:.2f} pp below original AA-CIL. ')
    return {
        'CSEProtocolText':('The additional withheld CSE-CIC-IDS2018 assessment uses the bound full cache with '
            f"{binding['x_train']['shape'][0]:,} training rows, {binding['x_test']['shape'][0]:,} test rows and "
            f"{binding['x_train']['shape'][1]} features. Its four tasks are: "+tasks+'. '
            'The same full-cache rows, task order, update rule and calibration policy are bound across methods. '
            f"The permanent Normal calibration holdout contains {calibration['rows']:,} rows "
            f"({calibration['feature_and_index_array_bytes']:,} feature/index bytes), outside replay memory and never optimized. "
            'These fields are taken from the released execution identities, not inferred from the dataset name.'),
        'CSEAbstractText':('Validation selected no candidate satisfying the joint forgetting constraint; KD+WA was frozen as an ineligible fallback. '
            +short+'It fails two of four predeclared conditions. Earlier CBF and fixed-KD protocol failures are retained. '
            'These results support an unresolved detection--retention--accuracy trade-off, not a universally superior architecture.'),
        'CSEConclusionText':short+'The accuracy-superiority and two-point accuracy-cost conditions fail, '
            'while the forgetting and attack-detection harm tests do not detect significant harm. '
            'This is not equivalence or proof that retention is solved. No test result changes the selected configuration.',
        'CSEConfirmationText':short+'All 35 physical reference cells and five selected-candidate cells completed before the guarded releases. '
            'The four saved decisions are pass, fail, fail, pass. Criterion 1 uses Freq, the lowest-mean forgetting member of the fixed external five. '
            'Paired one-sided t-tests determine the registered significance conditions; exact sign-flip values enumerate all 32 assignments as sensitivity checks. '
            'Criterion 3 uses only the mean difference, not a significance test. '
            'Reference seeds 2/42 and selected seed 42 ran locally; the other cells ran on HPC A100. Costs retain this hardware distinction. '
            'The proposed new-attack mask is already present in the executed KD+WA code: current new attacks receive binary CE only, '
            'and gate KD applies to benign and replayed old attacks. The proposed duplicate variant was therefore not trained; '
            'any additional intervention requires a genuinely distinct specification. This source check does not establish a causal explanation of the accuracy loss.',
        'CSEConfirmationTables':r'\input{generated/cse_confirmation.tex}'}


def cse_summary_zh(evidence, aggregate_text):
    if evidence.get('cse_status')!='released_assessed':
        return []
    r=evidence['cse_result']
    from paper_aacil.generate_section31_comparisons import NAMES
    lines=['### CSE-CIC-IDS2018 完整冻结候选判定','',
        '35个参考物理单元及5个固定候选单元全部完成后，以原冻结汇总流程解封。无训练、推理或旧统计重跑。候选不重选，验证不合格状态不变。','',
        '| 方法 | 增量BAcc | 最终BAcc | 遗忘(pp) | ADR | Macro-F1 | FPR |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for g in r['descriptive_groups']:
        lines.append('| '+' | '.join([NAMES[g['method']]]+[aggregate_text(g['metrics'][k]) for k in RETENTION_METRICS])+' |')
    lines+=['','| 标准 | 参照 | 差值均值±SD(pp) | 单侧t p | 精确符号翻转p | 判定 |','|---|---|---:|---:|---:|---|']
    for t in r['CSE18_necessary_tests']:
        direction=t['p_direction']
        tp=f"{t['p_'+direction]:.6f}" if direction else '均值规则'
        sp=f"{t['exact_sign_flip']['p_'+direction]:.5f}" if direction else '不用于判定'
        d=t['delta_pp']
        lines.append(f"| {t['criterion']} | {NAMES[t['reference']]} | {d['mean']:.4f}±{d['sample_sd']:.4f} | {tp} | {sp} | {t['decision']} |")
    lines+=['','四项为pass/fail/fail/pass，整体FAIL。最终BAcc比Replay-BS低7.06pp、比原AA-CIL低5.85pp；不能宣称权衡已解决。'
        '第3项只按差值均值≥−2pp判定；其对零差值的描述性t p_less=0.002195、符号翻转p_less=0.03125亦保存在JSON，不参与门槛。',
        '未检出遗忘/ADR伤害不等于等效。精确符号翻转枚举32种，仅作敏感性核对，不替换预登记t规则。'
        '本地与HPC成本不混称同硬件比较；旧FAIL、RQ1/RQ3及创新门槛不因此闭合。',
        '§42要求的gate KD mask已经存在于已执行代码，不能改名重复训练；新的额外候选需要真正不同的预注册干预。机制诊断尚未得到因果证明。','']
    return lines


def kd_reassessment_evidence(path):
    if not path.exists():
        return {}
    row=read(path)
    if (row.get('section')!='29.9' or row.get('candidate')!='aacil_kd' or
            row.get('blind_confirmation_established') is not False or
            row.get('original_section298_cbf_verdict')!='fail'):
        raise ValueError('Expected the separately scoped Section29.9 KD reassessment')
    require_current_comparison_inputs(row)
    return row | {'comparison_result':path.relative_to(ROOT).as_posix()}


def kd_reassessment_macros(evidence):
    if not evidence:
        return {'KDReassessmentText':'The separately authorized KD reassessment remains pending; it cannot amend the original CBF failure.',
                'KDReassessmentTables':'', 'CICKDCompletedSeeds':'0',
                'CICKDGroupStatusText':'The CIC-IDS-2017 KD group and its paired comparisons remain pending.'}
    status=evidence['required_numeric_verdict']
    conclusion=('A necessary numerical condition fails; the trade-off remains unresolved.' if status=='fail' else
        'The specified numerical conditions pass on both required datasets, but this does not establish independent blind confirmation.' if status=='pass' else
        'The required NSL/CIC17 numerical comparison remains incomplete; no success claim is made.')
    text=(r'The fixed-KD reassessment uses $\lambda_G=.5$ and DER $\alpha=\beta=1$. ')
    nsl=next((d for d in evidence['dataset_results'] if d['dataset']=='NSL-KDD'),{})
    accuracy=next((t for t in nsl.get('tests',[]) if t['criterion']==2),{})
    if accuracy.get('n')==5 and accuracy.get('status')=='complete':
        text+=('On NSL-KDD, the final BAcc difference from Replay-BS is '
            +fmt(accuracy['delta_pp'],1)+r' pp ($p_{>}='+f"{accuracy['p_greater']:.4f}"+r'$); '
            +('the required superiority condition is not met. ' if accuracy['decision']=='fail' else 'the required superiority condition is met. '))
    text+=conclusion
    for d in evidence['dataset_results']:
        if 0 < d['n'] < 5:
            text+=(' '+SHORT[d['dataset']]+f" has {d['n']}/5 completed KD seeds: mean incremental BAcc "
                +fmt(d['metrics']['average_incremental_accuracy'])+r'\%, final BAcc '
                +fmt(d['metrics']['final_balanced_accuracy'])+r'\%, and signed forgetting '
                +fmt(d['metrics']['forgetting_all_old_classes'])+' pp, descriptive only; missing seeds are '
                +', '.join(map(str,d.get('missing_seeds',[s for s in SEEDS if s not in d['seeds']])))+'.')
    cic=next((d for d in evidence['dataset_results'] if d['dataset']=='CIC-IDS-2017'),{})
    cic_n=cic.get('n',0)
    if cic.get('kd_complete_five_seed'):
        measures=cic['metrics']
        text+=(' Across all five CIC17 KD seeds, mean incremental BAcc is '
            +fmt(measures['average_incremental_accuracy'])+r'\%, final BAcc is '
            +fmt(measures['final_balanced_accuracy'])+r'\%, and signed forgetting is '
            +fmt(measures['forgetting_all_old_classes'])+r' pp. At the nominal 2\% calibration target, test FPR, ADR, and Macro-F1 are '
            +fmt(measures['final_normal_fpr'])+r'\%, '
            +fmt(measures['final_attack_detection'])+r'\%, and '
            +fmt(measures['final_macro_f1'])+r'\%, respectively.')
        cic_status='The CIC-IDS-2017 KD group is complete (5/5 seeds).'
    else:
        cic_status=f'The CIC-IDS-2017 KD group contains {cic_n}/5 completed seeds; available group estimates are descriptive only.'
    counts=cic.get('available_group_counts',{})
    missing_references=[name+f" ({counts.get(method,0)}/5 seeds)" for method,name in
        [('aacil','original AA-CIL'),('replay_bs','Replay-BS')] if counts.get(method,0)!=5]
    if missing_references:
        cic_status+=' Paired comparisons requiring '+ ' and '.join(missing_references)+' remain pending.'
    elif not cic.get('complete'):
        cic_status+=' Any paired comparison lacking a complete candidate or reference group remains pending.'
    return {'KDReassessmentText':text,
            'KDReassessmentTables':r'\section{Fixed-KD reassessment}\input{generated/section299_comparisons.tex}',
            'CICKDCompletedSeeds':str(cic_n), 'CICKDGroupStatusText':cic_status}


def retention_narrative_evidence(retention, kd_reassessment):
    """Read saved headline measures and tests; do not recalculate statistics."""
    path=ROOT/'ncd/results/problem_scout/section28/forgetting_v1/result.json'
    if not path.exists() or not retention.get('assessment_available') or not kd_reassessment:
        return None
    groups={(g['dataset'],g['method']):g for g in read(path)['groups']}
    datasets=['UNSW-NB15','NSL-KDD']
    required=[(d,m) for d in datasets for m in ['aacil','aacil_kd']]+[('UNSW-NB15','aacil_cbf')]
    if any(groups.get(k,{}).get('n')!=5 for k in required):
        return None
    checks={d['dataset']:d for d in kd_reassessment['dataset_results']}
    nsl_accuracy=next((t for t in checks.get('NSL-KDD',{}).get('tests',[]) if t['criterion']==2),{})
    if nsl_accuracy.get('n')!=5 or nsl_accuracy.get('status')!='complete':
        return None
    measure=lambda d,m,k:groups[d,m]['metrics'][k]['mean']
    forget_checks=[next((t for t in checks[d]['tests'] if t['criterion']==1),{}) for d in datasets]
    return {
        'original_forgetting':[measure(d,'aacil','forgetting_all_old_classes') for d in datasets],
        'cbf_forgetting':measure('UNSW-NB15','aacil_cbf','forgetting_all_old_classes'),
        'kd_forgetting':[measure(d,'aacil_kd','forgetting_all_old_classes') for d in datasets],
        'kd_adr':[measure(d,'aacil_kd','final_attack_detection') for d in datasets],
        'kd_nsl_fpr':measure('NSL-KDD','aacil_kd','final_normal_fpr'),
        'kd_nsl_accuracy_p':nsl_accuracy['p_greater'],
        'kd_nsl_accuracy_decision':nsl_accuracy['decision'],
        'kd_forgetting_harm_not_detected':all(t.get('n')==5 and t.get('status')=='complete' and t.get('decision')=='pass' for t in forget_checks),
        'retention_verdict':retention['verdict'],
        'selected_method':(retention.get('selected_main') or {}).get('method'),
    }


def retention_narrative_macros(retention, kd_reassessment):
    facts=retention_narrative_evidence(retention,kd_reassessment)
    if facts is None:
        pending='The saved retention and fixed-KD comparisons remain incomplete.'
        return {'RetentionAbstractText':pending, 'RetentionConclusionText':pending}
    pair=lambda values:' and '.join(f'{100*v:.2f}' for v in values)
    original=pair(facts['original_forgetting'])
    kd_forget=pair(facts['kd_forgetting'])
    kd_adr=pair(facts['kd_adr'])
    cbf=f"{100*facts['cbf_forgetting']:.2f}"
    p=f"{facts['kd_nsl_accuracy_p']:.4f}"
    fpr=f"{100*facts['kd_nsl_fpr']:.2f}"
    cbf_outcome=('failed the retention criterion' if facts['retention_verdict']=='fail' else
        'remains subject to the retention criterion')
    harm=('KD forgetting was not significantly higher than the lowest-forgetting external baseline on either dataset. '
        if facts['kd_forgetting_harm_not_detected'] else '')
    accuracy=('its final balanced-accuracy advantage over the matched-exposure Replay-BS control on NSL-KDD was not significant '
        if facts['kd_nsl_accuracy_decision']=='fail' else 'its final balanced-accuracy advantage over Replay-BS on NSL-KDD was significant ')
    scope='KD was not the validation-selected main version; UNSW is exploratory and NSL is not blind confirmation.'
    abstract=('Original AA-CIL combines strong attack detection with substantial forgetting: '+original+
        ' pp on UNSW-NB15 and NSL-KDD. The validation-selected CBF variant '+cbf_outcome+
        ' with '+cbf+' pp forgetting on UNSW. AA-CIL-KD reduced forgetting to '+kd_forget+
        ' pp while achieving attack-detection rates of '+kd_adr+r'\%, respectively, at nominal calibration target $\alpha=2\%$. '+
        harm+'However, '+accuracy+r'($p='+p+r'$), and its NSL test FPR was '+fpr+r'\%. '+
        'The false-alarm--forgetting--accuracy trade-off therefore remains unresolved. '+scope)
    conclusion=('Benign-stream training with an independently calibrated gate supports strong attack detection in the studied setting, '
        'but the original decoupled model exhibits '+original+' pp forgetting on UNSW-NB15 and NSL-KDD. '
        'The validation-selected CBF variant '+cbf_outcome+'; KD reduced forgetting to '+kd_forget+
        ' pp with attack-detection rates of '+kd_adr+r'\% at nominal $\alpha=2\%$. '+
        'Nevertheless, '+accuracy+r'($p='+p+r'$), while the NSL test FPR of '+fpr+r'\% exposes calibration shift. '+
        'These results support a trade-off study, not a fully resolved false-alarm--forgetting--accuracy solution. '+scope)
    return {'RetentionAbstractText':abstract, 'RetentionConclusionText':conclusion}


def kd_reassessment_summary_zh(evidence):
    if not evidence:
        return []
    counts='、'.join(f"{SHORT[d['dataset']]} KD {d['n']}/5（{d['status']}）" for d in evidence['dataset_results'])
    partial=[]
    for d in evidence['dataset_results']:
        if 0 < d['n'] < 5:
            def mean_only(key):
                m=d['metrics'].get(key)
                return '--' if not m or m.get('mean') is None else f"{100*m['mean']:.2f}"
            partial.append(SHORT[d['dataset']]+' 已完成种子的增量/最终BAcc/遗忘均值为 '
                +'/'.join(mean_only(k) for k in ['average_incremental_accuracy','final_balanced_accuracy','forgetting_all_old_classes'])
                +'（前两项%，遗忘pp），仅描述；缺seed '+','.join(map(str,d.get('missing_seeds',[s for s in SEEDS if s not in d['seeds']])))+'。')
    cutoff=('历史初稿窗口于2026-10-19 21:50:26 MYT结束；该次保存状态为 '+str(evidence.get('execution',{}).get('saved_execution_status') or '未知')+'。'
            if evidence.get('execution',{}).get('hard_cutoff_ended') else '')
    if evidence.get('execution',{}).get('termination_cause') == 'owner_hard_deadline':
        cutoff+='该次中断为预设截止 TimeoutError，不是 OOM；部分阶段不计入完整种子指标。'
    cic=next((d for d in evidence['dataset_results'] if d['dataset']=='CIC-IDS-2017'),{})
    counts_cic=cic.get('available_group_counts',{})
    if counts_cic.get('aacil_kd')==5 and counts_cic.get('aacil')==5:
        decisions={'pass':'满足条款','fail':'未通过','pending':'待定'}
        cutoff+='CIC17 KD与原AA均已完成5/5；三项检验依次为'+ '、'.join(
            decisions.get(t['decision'],t['decision']) for t in cic.get('tests',[]))+'。'
        if counts_cic.get('replay_bs',0)<5:
            cutoff+='Replay-BS仅'+str(counts_cic.get('replay_bs',0))+'/5，缺seed42，最终BAcc配对检验仍待定。'
    else:
        cutoff+='后续已授权继续CIC17 KD seeds3/4/42及原AA五个seed，不设负责人绝对截止；纳入数量以表中已完成轨迹为准。'
    return ['**固定KD复评：** '+counts+'。UNSW 为探索性；NSL seeds1/2 在20:47:57文档提交前已可得，'
            '五个均在约21:29直接授权前已可得，因此不是盲法确认，也不推测用户是否看过。'
            'NSL/CIC 两者均为要求的数据集；当前判定为 '+evidence['required_numeric_verdict']+'，原CBF失败保留。'
            +''.join(partial)+cutoff, '']


def retention_summary_zh(evidence, aggregate_text, followup=None):
    if not evidence['selection_complete']:
        return ['**验证选择尚未完成，保持协议的正式结论待齐。**']
    main=evidence['selected_main']
    lines=[]
    if main:
        c=main['configuration']
        config=f"λG={c['lambda_g']:g}"
        if main['method']=='aacil_cbf':
            config+=f"，CBF epochs={c['cbf_epochs']}，LR 倍率={c['cbf_lr_multiplier']:g}"
        lines.append(f"**验证集锁定主版本：{RETENTION_NAMES[main['method']]}（{config}）**。"
            '仅用 UNSW 训练验证 seeds5/6，先满足最终 BAcc≥原 AA−3 pp，再选最低遗忘。')
    else:
        lines.append('没有版本通过开发资格约束，未选择主版本；回退版本的正式结果不能将开发失败改为成功。')
    ineligible=[RETENTION_NAMES[v['method']] for v in evidence['per_version'] if not v['eligible']]
    if ineligible:
        lines.append('、'.join(m.removeprefix('AA-CIL-') for m in ineligible)+' 未通过开发资格约束，仍报告固定回退配置的正式结果，不称为 eligible。')
    lines+=['', '| 数据集 | 版本 | n | 增量 BAcc (%) | 最终 BAcc (%) | 遗忘 (pp) | ADR (%) |',
            '|---|---|---:|---:|---:|---:|---:|']
    gm={(g['dataset'],g['method']):g for g in evidence['groups']}
    for ds in ['UNSW-NB15','NSL-KDD']:
        for m in RETENTION:
            g=gm.get((ds,m),{}); full=g.get('complete_five_seed',False)
            lines.append('| '+' | '.join([SHORT[ds],m.removeprefix('aacil_').upper(),str(g.get('n',0))]+[
                aggregate_text(g['metrics'].get(k)) if full else '--' for k in RETENTION_METRICS[:4]])+' |')
    for g in followup or []:
        if g.get('n')==5:
            lines.append('| '+' | '.join(['CIC17', '原AA' if g['method']=='aacil' else 'KD', '5']+[
                aggregate_text(g['metrics'].get(k)) for k in RETENTION_METRICS[:4]])+' |')
    lines+=['', '正式 seeds1/2/3/4/42 齐全才显示均值±样本 SD；-- 为未齐，不以开发数字填补。']
    if evidence['assessment_available']:
        labels={'pass':'满足当前条款','fail':'FAIL（至少一项必要条件失败）','pending':'未完成'}
        lines.append('验证选择保持协议：**'+labels.get(evidence['verdict'],evidence['verdict'])+'**。'
            'UNSW 与 NSL 均须满足：遗忘不显著劣于最低遗忘外部基线、最终 BAcc 显著高于 Replay-BS、ADR 不显著低于原 AA；'
            f"[完整配对结果](../{evidence['comparison_result']})。")
    else:
        lines.append('保持协议的正式比较尚未生成；n=0表示尚未纳入结果。')
    lines+=['不显著有害不等于等效或非劣；已知必要条件失败优先于未完成。', '']
    return lines


SECTION39_SENTENCE = ('On CIC-IDS-2017 the retention components remove forgetting but cost 8.3 pp '
    'final BAcc and give no gain over the same-data single classifier.')


def section39_development_rows():
    """Read the already aggregated five-seed development evidence; no new tests."""
    saved=read(ROOT/'ncd/results/problem_scout/section28/forgetting_v1/result.json')
    groups={(g['dataset'],g['method']):g for g in saved['groups']}
    methods=[('aacil_kd_wa','AA-CIL-KD+WA'),('aacil_kd','AA-CIL-KD'),
             ('aacil','AA-CIL'),('replay_bs','Replay-BS'),('wacb','WACB')]
    fields=RETENTION_METRICS[:5]
    rows=[]
    for method,label in methods:
        group=groups['CIC-IDS-2017',method]
        if group['n']!=5 or group['seeds']!=SEEDS:
            raise ValueError('Section39 presentation requires five complete CIC17 seeds')
        if any(group['metrics'][key]['n']!=5 or group['metrics'][key]['mean'] is None
               for key in fields):
            raise ValueError('Section39 presentation requires all five defined metrics')
        rows.append([label]+[f"{100*group['metrics'][key]['mean']:.1f}" for key in fields])
    return rows


def section39_development_outputs(compact):
    rows=section39_development_rows()
    table('cic_retention_development',
          r'CIC-IDS-2017 development results at saved stagewise $\alpha=2\%$: five-seed means.',
          ['Method','AIA','Final BAcc','Signed forget.','ADR','Macro-F1'], rows,
          note=r'Rates are percentages; signed forgetting is in percentage points and is not clipped. '
          r'Seeds are 1, 2, 3, 4, 42. AA-CIL-KD+WA is the fixed validation-selected candidate; '
          r'CIC17 was not used for frozen validation selection. These are descriptive development results, '
          r'not CSE18 confirmation. Sample SDs and per-seed values remain in the accompanying result artifacts.')
    return {'CICRetentionDevelopmentText':SECTION39_SENTENCE+' '
        "Here ``remove forgetting'' refers to the negative group-average signed forgetting endpoint, "
        'not zero loss for every class or proof of a causal retention mechanism. '
        'The 8.3 pp cost is relative to original AA-CIL; the same-data single classifier is Replay-BS, '
        'which matches exposure but not total computation. This descriptive comparison neither '
        'reselects the fixed candidate nor changes the failed historical protocols.'}


def section39_summary_zh(compact):
    return ['### CIC-IDS-2017 开发结果（五种子，非盲确认）','',
        '| 方法 | AIA | 最终BAcc | 有符号遗忘(pp) | ADR | Macro-F1 |',
        '|---|---:|---:|---:|---:|---:|',
        *['| '+' | '.join(row)+' |' for row in section39_development_rows()], '',
        SECTION39_SENTENCE,
        '“消除遗忘”仅指组均值有符号遗忘为负，不表示每个类别零遗忘或已证明机制。'
        '最终BAcc相对原AA-CIL降低8.3个百分点，对同数据单分类器Replay-BS没有观察到平均准确率收益；'
        '后者匹配曝光但不匹配总计算。KD+WA仍是固定选定候选，验证遗忘约束未满足，'
        '不因本结果重选。'+('CSE18完整判定已报告。' if compact.get('cse_status')=='released_assessed' else 'CSE18仍封存。'),'']


def write_summary_zh(groups, comparisons, statuses, macros, sensitivity, retention, kd_reassessment, compact):
    """One-page author summary: one formal table, no duplicate old matrices."""
    def aggregate_text(value):
        if not value or value.get('mean') is None:
            return '--'
        sd=value.get('sample_sd')
        return f"{100*value['mean']:.2f} ± "+('--' if sd is None else f'{100*sd:.2f}')
    forgetting_path=ROOT/'ncd/results/problem_scout/section28/forgetting_v1/result.json'
    forgetting=read(forgetting_path) if forgetting_path.exists() else {}
    cil={(g['dataset'],g['method']):g for g in forgetting.get('groups',[])}
    def original(ds,metric):
        return aggregate_text(cil.get((ds,'aacil'),{}).get('metrics',{}).get(metric))
    facts=retention_narrative_evidence(retention,kd_reassessment)
    key_findings=[]
    if facts:
        pair=lambda values:' / '.join(f'{100*v:.2f}' for v in values)
        key_findings=[
            '**核心结果（UNSW / NSL）：** 原AA-CIL遗忘 '+pair(facts['original_forgetting'])+
            ' pp；验证选定的CBF在UNSW仍有 '+f"{100*facts['cbf_forgetting']:.2f}"+
            ' pp遗忘，未通过保持标准。KD将遗忘降到 '+pair(facts['kd_forgetting'])+
            ' pp，ADR为 '+pair(facts['kd_adr'])+'%。',
            'KD在NSL相对Replay-BS的最终BAcc优势不显著（p='+f"{facts['kd_nsl_accuracy_p']:.4f}"+
            '），测试FPR为 '+f"{100*facts['kd_nsl_fpr']:.2f}"+
            '%，并非校准目标2%。**误报–遗忘–准确率权衡尚未完全解决。** KD不是验证选定的主版本；UNSW为探索性，NSL不是盲确认。',
            '',
        ]
    lines = [
        '# AA-CIL 中文摘要',
        '',
        f"结果快照与修订时间：{macros['EvidenceCutoff']}。原始优越性协议仍为FAIL。",
        '',
        *key_findings,
        *(retention_summary_zh(retention, aggregate_text, [cil[k] for k in
            [('CIC-IDS-2017','aacil'),('CIC-IDS-2017','aacil_kd')] if k in cil]) if not compact['groups'] else
          ['原验证选择保持协议和固定KD复评均未通过；历史五版本及CIC完整结果保留于论文和源数据。', '']),
        *kd_reassessment_summary_zh(kd_reassessment),
        *compact_summary_zh(compact, aggregate_text),
        '### UNSW 最终任务混淆图（开发描述性）', '',
        '论文比较 Freq 与固定候选 AA-CIL-KD+WA：仅绘制已保存且源绑定核对一致的 α=2% 矩阵，'
        '五种子1/2/3/4/42先求和计数再逐行归一化。每种子是同一批30,248测试行，'
        '不能当作新增独立样本；2%是校准目标，不是实际测试FPR保证。图中≥1%的单元格标注四舍五入整数，'
        '空白标签不代表零。最终混淆不能单独证明遗忘或因果机制，也不改变CSE失败判定。', '',
        *section39_summary_zh(compact),
        *cse_summary_zh(compact, aggregate_text),
        '口径：每阶段用永久训练域 Normal 留出池定 α=2%；增量 BAcc 为各阶段已见类均值，'
        '遗忘为旧类此前最高 recall 减最终 recall（含 Normal、保留负值）。低遗忘须联看 BAcc，ADR 不等于家族识别。',
        '',
        '边界：不显著有害不等于等效；Replay-BS匹配曝光但不匹配G+F总计算。保持协议要求UNSW/NSL，'
        '固定KD复评要求NSL/CIC，不能互换。CIC Replay-BS已补齐五种子，仅更新受影响的KD准确率配对比较；'
        '其余历史检验未重算，原FAIL不变。解释漂移、治理及总体创新仍未闭合。',
        '',
        '[完整逐阶段指标与逐类结果](../ncd/results/problem_scout/section28/forgetting_v1/result.json)；其余 FPR/MF1、配对检验与成本见论文及源结果。',
    ]
    (PAPER / 'summary_zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def cic_result_text(groups, comparisons, stats):
    """Complete sentences; inference is limited to the exact five paired seeds."""
    ds = 'CIC-IDS-2017'
    family = ['aacil', 'ts_mem', 'aa_noattmem']
    count = lambda m: len(groups[ds,m])
    complete = lambda m: sorted(x['seed'] for x in groups[ds,m]) == sorted(SEEDS)

    def pvalue(value):
        if value is None or not np.isfinite(value):
            return 'undefined'
        if 0 < value < .00001:
            coefficient, exponent = f'{value:.2e}'.split('e')
            return coefficient+r'\times10^{'+str(int(exponent))+'}'
        return f'{value:.5f}'

    def contrast(value):
        return fmt(value, 1, 2) + ' pp'

    if complete('aacil'):
        primary = [r'At $\alpha=2\%$, CIC-IDS-2017 AA-CIL has all five seeds: actual test benign FPR '
                   + fmt(stats(ds,'aacil','normal_fpr')) + r'\%, attack detection '
                   + fmt(stats(ds,'aacil','attack_detection')) + r'\%, and Macro-F1 '
                   + fmt(stats(ds,'aacil','macro_f1')) + r'\% (mean $\pm$ sample SD).']
    else:
        primary = [f'CIC-IDS-2017 AA-CIL is incomplete ($n={count("aacil")}/5$); '
                   'available cells are descriptive and do not support a complete five-seed inference.']
    for scope, candidates, label in [('registered', BASELINES, 'registered seven-comparator set'),
                                      ('external', EXTERNAL, 'external-only comparator set')]:
        c = comparisons[ds].get(scope+'_attack_detection')
        if (all(complete(m) for m in ['aacil']+candidates) and c and c.get('complete_five_seed')):
            outcome = ('meets the dataset-level positive-difference and one-sided $p<0.05$ condition'
                       if c['mean'] > 0 and c['p_one_sided'] is not None and c['p_one_sided'] < .05
                       else 'does not meet the dataset-level positive-difference and one-sided $p<0.05$ condition')
            primary.append('Against '+NAMES[c['baseline']]+' from the '+label+', AA-CIL minus comparator ADR is '
                           +contrast(c)+r' (one-sided $p='+pvalue(c['p_one_sided'])+'$), which '+outcome+'.')
        else:
            primary.append('The '+label+' is not yet complete for all five paired seeds; its CIC inference remains pending.')

    interpretation = []
    for method in ['ts_mem', 'aa_noattmem', 'replay_bs']:
        prefix = 'The post-hoc Replay-BS comparison' if method == 'replay_bs' else 'The '+NAMES[method]+' comparison'
        if not (complete('aacil') and complete(method)):
            interpretation.append(prefix+f' is incomplete (AA-CIL $n={count("aacil")}/5$, '
                                  +NAMES[method]+f' $n={count(method)}/5$); no complete-group inference is made.')
            continue
        adr = paired(groups[ds,'aacil'], groups[ds,method], 'attack_detection')
        mf1 = (comparisons[ds]['ts_ablation'] if method == 'ts_mem' else
               paired(groups[ds,'aacil'], groups[ds,method], 'macro_f1'))
        sentence = (prefix+' gives paired AA-CIL minus comparator differences of '+contrast(adr)
                    +' in ADR and '+contrast(mf1)+' in Macro-F1')
        if method == 'ts_mem':
            sentence += r'; the registered one-sided Macro-F1 test has $p='+pvalue(mf1['p_one_sided'])+'$.'
        else:
            sentence += r' (one-sided ADR $p='+pvalue(adr['p_one_sided'])+'$).'
        interpretation.append(sentence)
    coverage = 'All three CIC AA-family groups contain five seeds' if all(complete(m) for m in family) else 'CIC AA-family coverage remains incomplete'
    conclusion = coverage+'; the original superiority protocol remains failed.'
    return {'CICPrimaryResultsText':' '.join(primary),
            'CICInterpretationText':' '.join(interpretation), 'CICConclusionText':conclusion,
            'CICResultStatus':'CIC-IDS-2017 AA-family coverage: '+str(sum(count(m) for m in family))
                +'/15 cells across AA-CIL ($n='+str(count('aacil'))+'/5$), TS-mem ($n='
                +str(count('ts_mem'))+'/5$), and AA-noAttMem ($n='+str(count('aa_noattmem'))
                +'/5$); Replay-BS is reported separately.'}


def refresh_presentation(args, retention, kd_reassessment, compact):
    """Revise saved presentation only, without rerunning any scientific calculation."""
    macro_path=PAPER/'generated'/'macros.tex'
    macros={}
    for line in macro_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        match=re.fullmatch(r'\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}',line)
        if not match:
            raise ValueError('Presentation-only refresh requires the existing single-line generated macros')
        macros[match[1]]=match[2]
    renames={
        'Section 27.8':'original superiority protocol',
        'Section27.8':'original superiority protocol',
        'Section 29.8':'validation-selected retention protocol',
        'Section29.8':'validation-selected retention protocol',
        'Section 29.9':'fixed-KD reassessment',
        'Section29.9':'fixed-KD reassessment',
        'Section 29.4':'earlier retention protocol',
        'Section29.4':'earlier retention protocol',
    }
    def rename(text):
        for old,new in renames.items():text=text.replace(old,new)
        return text
    macros={k:rename(v) for k,v in macros.items()}
    if args.cutoff:macros['EvidenceCutoff']=args.cutoff
    macros.update(retention_outputs(retention))
    macros.update(kd_reassessment_macros(kd_reassessment))
    macros.update(retention_narrative_macros(retention,kd_reassessment))
    macros.update(compact_outputs(compact))
    macros.update(section39_development_outputs(compact))
    macros.update(cse_presentation(compact))
    historical_prefix='In the retained original operating-point snapshot: '
    if not macros['CICInterpretationText'].startswith(historical_prefix):
        macros['CICInterpretationText']=historical_prefix+macros['CICInterpretationText']
    macro_path.write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n',encoding='utf-8')
    for name in ['registered_criteria','calibration_shift','forgetting_results','section298_comparisons']:
        path=PAPER/'generated'/f'{name}.tex'
        text=path.read_text(encoding='utf-8')
        revised=rename(text)
        if revised!=text:path.write_text(revised,encoding='utf-8')
    saved_path=PAPER/'generated'/'paper_results.json'
    saved=read(saved_path)
    saved.update(cutoff=macros['EvidenceCutoff'],section298=retention,section299=kd_reassessment,section31=compact)
    saved_path.write_text(json.dumps(saved,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write_summary_zh(None,None,None,macros,[],retention,kd_reassessment,compact)
    print(json.dumps({'presentation_only':True,'recomputed_statistics':False,
        'evidence_label':macros['EvidenceCutoff'],'cic_kd_completed_seeds':macros['CICKDCompletedSeeds']}))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',action='append',default=[]);ap.add_argument('--cutoff',default=None)
    ap.add_argument('--retention-selection',type=Path,default=ROOT/'ncd/results/problem_scout/section29/retention_queue_v1/selection.json')
    ap.add_argument('--retention-comparisons',type=Path,default=ROOT/'ncd/results/problem_scout/section29/retention_comparisons_v1/result.json')
    ap.add_argument('--kd-reassessment',type=Path,default=ROOT/'ncd/results/problem_scout/section29/section299_comparisons_v1/result.json')
    ap.add_argument('--compact-selection',type=Path,default=ROOT/'ncd/results/problem_scout/section31/development_v1/selection.json')
    ap.add_argument('--compact-comparisons',type=Path,default=ROOT/'ncd/results/problem_scout/section31/comparisons_v1/result.json')
    ap.add_argument('--presentation-only',action='store_true',
                    help='Reuse saved paper statistics and comparison JSON; revise text/tables only')
    args=ap.parse_args()
    retention=retention_evidence(args.retention_selection.resolve(),args.retention_comparisons.resolve())
    kd_reassessment=kd_reassessment_evidence(args.kd_reassessment.resolve())
    compact=compact_evidence(args.compact_selection.resolve(),args.compact_comparisons.resolve())
    if args.presentation_only:
        refresh_presentation(args,retention,kd_reassessment,compact)
        return
    (PAPER/'generated').mkdir(exist_ok=True);(PAPER/'figures').mkdir(exist_ok=True)
    analysis=read(ROOT/'ncd/results/problem_scout/section27/analysis_v1/result.json')
    cells={(x['dataset'],x['method'],x['seed']):x for x in analysis['cells'] if x['points']['0.02']['status']=='available'}
    inputs=args.input or ['ncd/results/problem_scout/section28/replay_bs_local_v1',
                         'ncd/results/problem_scout/section28/cic_local_queue_v1',
                         'ncd/results/problem_scout/section28/cic_aacil_fallback_v1',
                         'ncd/results/problem_scout/section28/hpc_cic_section28']
    for input_path in inputs:
        base=ROOT/input_path
        if not base.exists():continue
        for p in sorted(base.rglob('result.json')):
            # Only physical cell records, never the mutable matrix summary.
            raw=read(p)
            if 'identity' not in raw or 'checkpoints' not in raw:continue
            # Compact candidates have their own frozen selection and saved
            # comparison layer; never mix them into the older method matrix.
            if raw['identity'].get('method') in COMPACT_NAMES:continue
            x=from_raw(p)
            if x is None:continue
            k=(x['dataset'],x['method'],x['seed'])
            if k not in cells or dt.datetime.fromisoformat(x['completed_at'])<dt.datetime.fromisoformat(cells[k]['completed_at']):cells[k]=x
    groups=defaultdict(list)
    for x in cells.values():groups[x['dataset'],x['method']].append(x)
    # Read only the recorded curve-cell elapsed total. Nested forward timings
    # and the shared resume/render ledger must not be added to it again.
    curve_root=ROOT/'ncd/results/problem_scout/section28/paper_curves_v1'
    for path in curve_root.glob('*/*/result.json'):
        curve=read(path)
        if not curve.get('completed') or curve.get('method') not in ['aacil','ts_mem','aa_noattmem','replay_bs']:
            continue
        key=(curve['dataset'],curve['method'],curve['seed'])
        selected=cells.get(key)
        if selected is None:continue
        binding=curve.get('binding',{})
        source=selected.get('original_source_result',selected['source_result'])
        original_sha=selected.get('original_source_result_sha256')
        if binding.get('source_result')!=source or (original_sha and binding.get('source_result_sha256')!=original_sha):
            continue
        elapsed=curve.get('cost_seconds',{}).get('cell_total')
        if isinstance(elapsed,(int,float)) and not isinstance(elapsed,bool) and np.isfinite(elapsed) and elapsed>=0:
            selected.setdefault('costs',{})['curve_analysis_elapsed_seconds']=elapsed
    def stats(ds,m,key,a='0.02'):
        return summary(x['points'][a]['metrics'].get(key) for x in groups[ds,m])
    def val(ds,m,key):return stats(ds,m,key)['mean']
    def n(ds,m):return len(groups[ds,m])
    comparisons={}
    for ds in DATASETS:
        comparisons[ds]={}
        for scope,candidates in [('registered',BASELINES),('external',EXTERNAL)]:
            for key,alt in [('attack_detection','greater'),('macro_f1','less')]:
                if all(n(ds,m)==5 for m in ['aacil']+candidates):
                    best=max(candidates,key=lambda m:val(ds,m,key))
                    comparisons[ds][scope+'_'+key]={'baseline':best,**paired(groups[ds,'aacil'],groups[ds,best],key,alt)}
        comparisons[ds]['guard']=paired(groups[ds,'aacil'],groups[ds,'replay'],'attack_guard_recall','less')
        comparisons[ds]['ts_ablation']=paired(groups[ds,'aacil'],groups[ds,'ts_mem'],'macro_f1')
    rows=[]
    for ds in DATASETS:
        for m in METHODS:
            rows.append([SHORT[ds],NAMES[m],n(ds,m)]+[fmt(stats(ds,m,k)) for k in ['normal_fpr','attack_detection','macro_f1','balanced_accuracy','attack_guard_recall']])
    table('main_results','Final-task test performance at a benign-holdout calibration target of $\\alpha=2\\%$. Rates are percentages; mean $\\pm$ sample SD. Actual test FPR is not fixed at 2\\%.',
          ['Data','Method','n','Test FPR','Attack detection','Macro-F1','BAcc','Attack guard'],rows,'llrrrrrr',
          note='Replay+LA and Replay share the same calibrated total logit adjustment and weights. Joint has offline old-training access. Replay-BS is post hoc. A dash means unavailable; $n<5$ is descriptive only. The guard alone excludes attack classes with test support below 20.')
    rows=[]
    for ds in DATASETS:
        for m in ['aacil','ts_mem','aa_noattmem','replay_bs']:
            p=paired(groups[ds,'aacil'],groups[ds,m],'attack_detection') if m!='aacil' else None
            rows.append([SHORT[ds],NAMES[m],n(ds,m),fmt(stats(ds,m,'attack_detection')),fmt(stats(ds,m,'macro_f1')),fmt(p,1) if p else '--', f"{p['p_one_sided']:.4f}" if p and p['p_one_sided'] is not None else '--'])
    table('ablation_results','Ablations at $\\alpha=2\\%$. Paired changes are AA-CIL minus the named ablation. Tests involving Replay-BS are post hoc; small groups do not amend registered decisions.',
          ['Data','Method','n','Detection (\\%)','MF1 (\\%)','$\\Delta$ detection (pp)','$p_{>}$'],rows,'llrrrrr')
    rows=[]
    overheadrows=[]
    for ds in DATASETS:
        for m in ['replay','derpp','wacb','freq','aacil','ts_mem','aa_noattmem','replay_bs','joint']:
            cc=groups[ds,m]
            if not cc:continue
            cost=lambda k:summary(x.get('costs',{}).get(k) for x in cc)
            occupancy=summary(x['trajectory'][-1]['actual_total'] for x in cc)
            normal_bytes=[x['trajectory'][-1].get('normal_training_storage',{}).get('feature_and_index_array_bytes') for x in cc]
            ns=summary(value/1048576 if value is not None else None for value in normal_bytes)
            rows.append([SHORT[ds],NAMES[m],len(cc),fmt(cost('gate_training_seconds'),1,1),fmt(cost('family_training_seconds'),1,1),fmt(cost('training_seconds'),1,1),fmt(occupancy,1,0),fmt(ns,1,2)])
            overheadrows.append([SHORT[ds],NAMES[m],len(cc)]+[
                fmt(cost(key),1,3) for key in ['buffer_selection_seconds','calibration_seconds','evaluation_seconds',
                                              'curve_analysis_elapsed_seconds','extended_calibration_analysis_elapsed_seconds']])
    table('costs','Per-seed total training seconds across tasks and final memory occupancy. G and F are separately trained models; single-classifier methods have no G/F split. Timing excludes data loading/saving and is not an exclusive-GPU benchmark.',
          ['Data','Method','n','G (s)','F (s)','Total (s)','Used rows','G/BS benign MiB'],rows,'llrrrrrr',
          note='G/BS benign MiB reports final-task benign input feature-plus-index array bytes, not additional persistent storage. AA-CIL, AA-noAttMem and the single-model Replay-BS use up to 50,000 refreshed benign inputs outside M. TS-mem uses its retained training portion (normally 180 of 200 Normal slots) inside M; these bytes must not be added again to replay storage. Missing baseline byte records are shown as --, not zero. Dense-label materialization, permanent calibration features/indices, model weights and other transient copies are separate and excluded from this column; see the setup table. Joint is not a matched compute/storage baseline.')
    table('overhead_costs','Per-seed recorded overhead across tasks (seconds, mean $\\pm$ sample standard deviation). Separate post-training CPU analyses are shown without adding their nested subcomponents.',
          ['Data','Method','n','Selection','In-run calib.','In-run eval.','CPU curves','CPU ext. calib.'],overheadrows,'llrrrrrr',
          note='Selection, in-run calibration and in-run evaluation use the recorded buffer-selection, calibration and evaluation totals respectively. A missing key is --, never an assumed zero. AA-family and Replay-BS CPU curves use the source-matched recorded cell-total seconds from the independent post-hoc curve ledger when available; missing or mismatched cell records remain --. For older baselines the separate CPU curve-analysis and extended-calibration elapsed totals are not embedded in the in-run columns; nested holdout/search/inference sub-timings must not be added again. Shared resume/render-attempt overhead remains in the independent curve ledger and is excluded here. These columns are not a single end-to-end walltime total and do not duplicate Replay+LA, which shares Replay weights and calibration.')
    rows=[]
    for ds in DATASETS:
        for scope in ['registered','external']:
            c=comparisons[ds].get(scope+'_attack_detection')
            rows.append([SHORT[ds],scope, NAMES[c['baseline']] if c else 'Pending',fmt(c,1,3) if c else '--',f"{c['p_one_sided']:.5f}" if c else '--',f"{c['paired_d_z']:.3f}" if c and c['paired_d_z'] is not None else '--',('Pass' if c['p_one_sided']<.05 and c['mean']>0 else 'Fail') if c else 'Pending'])
    complete=all(n(ds,m)==5 for ds in DATASETS for m in ['aacil']+BASELINES)
    statuses=['FAIL']
    for key in ['registered_macro_f1','guard']:
        tests=[comparisons[ds].get(key) for ds in DATASETS]
        fail=any(c and c.get('complete_five_seed') and c['mean']<0 and c['p_one_sided']<.05 for c in tests)
        statuses.append('FAIL' if fail else ('PASS' if complete else 'PENDING'))
    c=comparisons['CIC-IDS-2017']['ts_ablation']
    statuses.append(('PASS' if c['mean']>0 and c['p_one_sided']<.05 else 'FAIL') if c['complete_five_seed'] else 'PENDING')
    statuses.append('PASS' if complete else 'PENDING')
    table('registered_criteria','Original registered superiority test and separately identified external-only comparison. The latter does not replace the former.',
          ['Data','Comparison set','Fixed mean winner','$\\Delta$ ADR (pp)','$p_{>}$','$d_z$','Dataset test'],rows,'lllrrrl',
          note='Original superiority protocol, criteria 1--5: '+', '.join(f'{i+1}: {v}' for i,v in enumerate(statuses))+'. Overall decision: FAIL, because NSL and UNSW cannot contribute the required two significant wins. The comparator set includes TS-mem and AA-noAttMem.')
    safetyrows=[]
    for ds in DATASETS:
        for key,label in [('registered_macro_f1','MF1 vs strongest'),('guard','Guard vs Replay')]:
            c=comparisons[ds].get(key)
            done=c and c.get('complete_five_seed')
            harm=done and c['mean']<0 and c['p_one_sided']<.05
            safetyrows.append([SHORT[ds],label,NAMES[c['baseline']] if c and 'baseline' in c else ('Replay' if key=='guard' else 'Pending'),
                               fmt(c,1,3) if done else '--',f"{c['p_one_sided']:.5f}" if done else '--',
                               ('Harm detected' if harm else 'No detected harm') if done else 'Pending'])
    c=comparisons['CIC-IDS-2017']['ts_ablation']
    safetyrows.append(['CIC17','MF1 gate-refresh ablation','TS-mem',fmt(c,1,3) if c['complete_five_seed'] else '--',
                       f"{c['p_one_sided']:.5f}" if c['complete_five_seed'] else '--',statuses[3]])
    table('registered_safety','Remaining registered contrasts. Differences are AA-CIL minus the named comparator. The harm tests use the less-than alternative; the CIC gate-refresh ablation uses the greater-than alternative.',
          ['Data','Endpoint','Comparator','$\\Delta$ (pp)','$p$','Outcome'],safetyrows,'lllrrl',
          note='Only complete five-seed groups determine these outcomes. No detected harm does not establish equivalence or noninferiority. Criterion 5 is descriptive completion of Joint-gap and cost reporting, not superiority over Joint.')
    jointrows=[]
    for ds in DATASETS:
        deltas={key:paired(groups[ds,'aacil'],groups[ds,'joint'],key) for key in ['normal_fpr','attack_detection','macro_f1']}
        jointrows.append([SHORT[ds],len(deltas['normal_fpr']['seeds'])]+[fmt(deltas[key],1,3) for key in deltas])
    table('joint_gaps','Descriptive final-task differences from the offline Joint reference at $\\alpha=2\\%$. Values are AA-CIL minus Joint in percentage points; mean $\\pm$ sample SD across paired completed seeds.',
          ['Dataset','Paired n','$\\Delta$ FPR','$\\Delta$ detection','$\\Delta$ MF1'],jointrows,'lrrrr',
          note='Joint uses offline old-training access and is neither a matched deployment baseline nor a guaranteed empirical upper bound. These gaps have no success threshold.')
    rows=[]
    for ds in DATASETS:
        for m in METHODS:
            if not n(ds,m):continue
            for a in ['0.01','0.05']:
                rows.append([SHORT[ds],NAMES[m],int(float(a)*100),n(ds,m)]+[fmt(stats(ds,m,k,a)) for k in ['normal_fpr','attack_detection','macro_f1']])
    # Split a large appendix into per-dataset floats, preventing an over-height float.
    alpha_file=PAPER/'generated'/'alpha_sensitivity.tex'
    parts=[]
    for ds in DATASETS:
        name='alpha_'+SHORT[ds].lower()
        table(name,'Additional calibration targets: '+ds+'. Rates in percent; mean $\\pm$ sample SD.',
              ['Method','$\\alpha$','n','Test FPR','Detection','MF1'],[r[1:] for r in rows if r[0]==SHORT[ds]],'lrrrrr')
        parts.append(r'\input{generated/'+name+'.tex}')
    alpha_file.write_text('\n'.join(parts)+'\n',encoding='utf-8')
    rows=[]
    raw_example=read(ROOT/'ncd/results/problem_scout/section27/local_v1/nsl_kdd/aacil_seed1/result.json')
    cfg=raw_example['identity']['effective_config']
    for ds in DATASETS:
        source=next((x for x in groups[ds,'aacil']),None)
        if source and source.get('class_order'):cl=source['class_order']; tasks=source['tasks']
        else:
            candidates=groups[ds,'replay']
            raw=read(ROOT/candidates[0]['source_result']) if candidates else None
            # Baseline analysis references a trained source.result; class order lives in identity.
            cl=raw.get('class_order',raw.get('identity',{}).get('class_order',[])) if raw else []
            tasks=raw.get('tasks',raw.get('identity',{}).get('tasks',[])) if raw else []
        # Use an existing AA or baseline calibration row; no new data loads.
        tr=(groups[ds,'aacil'] or groups[ds,'replay'])[0]['trajectory'][-1]
        cs=tr.get('calibration_storage',{})
        if not cs:
            slug=ds.lower().replace('-','_')
            cal=read(ROOT/f'ncd/results/problem_scout/section27/baseline_calibration_v2/{slug}/replay_seed1/result.json')['normal_holdout_calibration']['pool']
            cs={'rows':cal['count'],'feature_and_index_array_bytes':cal['feature_bytes']+cal['index_bytes']}
        rows.append([ds,str(len(cl)),str(len(tasks)),str(cfg['buffer_per_class']*len(cl)),str(cs.get('rows','--')),str(cs.get('feature_and_index_array_bytes','--'))])
    table('protocol','Dataset/task and final-budget summary. Calibration bytes include float32 features and int64 row indices, exclude model state and dense labels. The pool is shared by all methods, excluded from optimization, and outside M.',
          ['Dataset','Classes','Tasks','M ceiling','Calib rows','Calib bytes'],rows,'lrrrrr')
    rows=[]
    labelnotes=[]
    for ds in DATASETS:
        raw=read(ROOT/groups[ds,'replay'][0]['source_result'])
        ident=raw['identity']; binding=ident['data_binding']
        tasktext='; '.join(','.join(map(str,t)) for t in raw['tasks'])
        rows.append([ds,binding['x_train']['shape'][0],binding['x_test']['shape'][0],binding['x_train']['shape'][1],tasktext])
        labelnotes.append(ds+': '+', '.join(str(i)+'='+name for i,name in enumerate(raw['class_order']))+'.')
    table('dataset_details','Prepared split sizes, feature counts and fixed task groupings (class IDs). These are the exact prepared benchmark arrays, not an assertion of temporal deployment validity.',
          ['Dataset','TRAIN rows','TEST rows','Features','Task IDs (semicolon separates tasks)'],rows,'lrrrl',note=' '.join(labelnotes))
    rows=[]
    perclass={}
    for ds,classes in [('NSL-KDD',['R2L','U2R']),('UNSW-NB15',['Fuzzers','Worms']),('CIC-IDS-2017',['Botnet','Infiltration'])]:
        for m in ['aacil','ts_mem','aa_noattmem','replay_bs']:
            for name in classes:
                selected=[]
                precision_seed_values=[]
                for c in groups[ds,m]:
                    pcs=c['points']['0.02']['metrics'].get('per_class',[])
                    matches=[v for v in pcs if v['class_name']==name]
                    selected += matches
                    precision_seed_values += [{'seed':c['seed'],'value':v['precision']} for v in matches]
                if not selected:continue
                ss={key:summary(v[key] for v in selected) for key in ['recall','f1']}
                ss['precision']=defined_precision_summary(v['precision'] for v in selected)
                ss['precision_seed_values']=precision_seed_values
                perclass[ds,m,name]=ss
                rows.append([SHORT[ds],NAMES[m],name,len(selected),selected[0]['support'],ss['precision']['defined_n']]+[fmt(ss[k]) for k in ['precision','recall','f1']])
    table('per_class','Selected low-support or weak-family results at $\\alpha=2\\%$, to expose failure modes hidden by binary detection. All classes remain in Macro-F1 and BAcc; only the attack guard excludes support below 20.',
          ['Data','Method','Class','n','Support','$n_P$','Precision','Recall','F1'],rows,'lllrrrrrr',note='Precision uses defined seeds only; $n_P$ is defined\\_n and n is the full completed seed count. Original null precision values remain null in the generated JSON. With no defined value the mean is --; with fewer than two the sample SD is undefined. No seed is removed from other defined endpoints. Complete per-class confusion matrices accompany the saved result JSON.')
    bsrows=[[SHORT[ds],n(ds,'replay_bs')]+[fmt(stats(ds,'replay_bs',k)) for k in ['normal_fpr','attack_detection','macro_f1']] for ds in DATASETS]
    table('replay_bs','Post-hoc Replay-BS: single multiclass model with the AA-CIL benign exposure and attack replay allocation. This control was added after the original primary result.',
          ['Data','n','Test FPR','Detection','MF1'],bsrows,'lrrrr',wide=False)
    sens=ROOT/'ncd/results/problem_scout/section28/paper_curves_v1/summary.json'
    sensitivity=read(sens).get('nsl_calibration_sensitivity',[]) if sens.exists() else []
    shiftrows=[]
    pools=[('permanent_TRAIN_Normal','TRAIN Normal holdout',False),
           ('TEST_Normal_post_hoc_non_deployable','TEST Normal (post hoc)',True)]
    for method in ['aacil','ts_mem','aa_noattmem']:
        points={source:{} for source,_,_ in pools}
        for cell in sensitivity:
            if cell['method']!=method or cell['seed'] not in SEEDS:
                continue
            for point in cell['points']:
                if abs(point['alpha']-.02)>1e-12 or point['calibration_source'] not in points:
                    continue
                source=point['calibration_source']
                expected_test_labels=next(flag for pool,_,flag in pools if pool==source)
                if point['test_labels_used_for_threshold']!=expected_test_labels:
                    raise ValueError('Inconsistent calibration sensitivity provenance')
                if cell['seed'] in points[source]:
                    raise ValueError('Duplicate calibration sensitivity seed')
                points[source][cell['seed']]=point
        paired_seeds=sorted(set.intersection(*(set(v) for v in points.values())))
        for source,label,_ in pools:
            selected=[points[source][seed] for seed in paired_seeds]
            shiftrows.append([NAMES[method],label,len(selected)]+[
                fmt(summary(point[key] for point in selected))
                for key in ['normal_fpr','attack_detection','macro_f1']])
    table('calibration_shift',
          r'Post-hoc NSL-KDD calibration-distribution sensitivity at $\alpha=2\%$. Test metrics are mean $\pm$ sample standard deviation in percent.',
          ['Method','Threshold calibration pool','n','Test FPR','ADR','MF1'],shiftrows,'llrrrr',
          note=r'Each method uses the same paired seeds in both rows. TEST-Normal thresholds use test Normal labels and are evaluated on the same test population: this is a test-label-conditioned, non-deployable sensitivity analysis. The original superiority decision is unchanged; incomplete seed groups remain descriptive.')
    macros={'EvidenceCutoff':args.cutoff or dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M MYT'),
            'NumCompletedNew':str(sum(n(ds,m) for ds in DATASETS for m in ['aacil','ts_mem','aa_noattmem'])),
            'PrimaryAlpha':str(int(100*analysis['primary_alpha'])), 'GateBenignRows':str(cfg['normal_training_rows']),
            'BatchSize':str(cfg['batch_size']),'EpochBudget':str(cfg['epochs_per_task']),'MinSteps':str(cfg['min_steps_per_task']),
            'HiddenWidth':str(cfg['hidden']),'LearningRate':str(cfg['learning_rate']),'MemoryPerClass':str(cfg['buffer_per_class']),'SeedCount':str(len(SEEDS))}
    for ds,prefix in [('NSL-KDD','NSL'),('UNSW-NB15','UNSW')]:
        for key,suffix in [('normal_fpr','FPR'),('attack_detection','ADR'),('macro_f1','MFOne')]:
            macros[prefix+'AACIL'+suffix]=f'{100*val(ds,"aacil",key):.2f}'
    macros.update({'UNSWExternalADRDelta':f"{comparisons['UNSW-NB15']['external_attack_detection']['mean']:.2f}",
                   'UNSWRegisteredADRDelta':f"{comparisons['UNSW-NB15']['registered_attack_detection']['mean']:.2f}",
                   'UNSWRegisteredP':f"{comparisons['UNSW-NB15']['registered_attack_detection']['p_one_sided']:.4f}",
                   'UNSWTSADR':f"{100*val('UNSW-NB15','ts_mem','attack_detection'):.2f}",
                   'UNSWNoAttADR':f"{100*val('UNSW-NB15','aa_noattmem','attack_detection'):.2f}",
                   'NSLFreqADR':f"{100*val('NSL-KDD','freq','attack_detection'):.2f}",
                   'NSLWACBMFOne':f"{100*val('NSL-KDD','wacb','macro_f1'):.2f}",
                   'NSLAAFreqADRDelta':f"{comparisons['NSL-KDD']['registered_attack_detection']['mean']:.2f}",
                   'NSLAAFreqP':f"{comparisons['NSL-KDD']['registered_attack_detection']['p_one_sided']:.4f}",
                   'UNSWAAExternalP':f"{comparisons['UNSW-NB15']['external_attack_detection']['p_one_sided']:.5f}"})
    for ds,prefix in [('NSL-KDD','NSL'),('UNSW-NB15','UNSW')]:
        contrast=paired(groups[ds,'aacil'],groups[ds,'replay_bs'],'attack_detection')
        if contrast['mean'] is not None:
            macros[prefix+'AABSDiff']=f"{contrast['mean']:.2f}"
            macros[prefix+'AABSP']=f"{contrast['p_one_sided']:.4f}" if contrast['p_one_sided'] is not None else 'undefined'
    worms_precision=perclass['UNSW-NB15','aacil','Worms']['precision']
    macros['UNSWWormsPrecision']=f"{100*worms_precision['mean']:.2f}" if worms_precision['mean'] is not None else 'undefined'
    macros['UNSWWormsPrecisionDefinedN']=str(worms_precision['defined_n'])
    macros['UNSWWormsRecall']=f"{100*perclass['UNSW-NB15','aacil','Worms']['recall']['mean']:.2f}"
    macros['UNSWFuzzersRecall']=f"{100*perclass['UNSW-NB15','aacil','Fuzzers']['recall']['mean']:.2f}"
    prior=read(ROOT/'ncd/results/problem_scout/section23/analysis_v1/analysis.json')
    for ds,suffix in [('UNSW-NB15','UNSW'),('CIC-IDS-2017','CIC')]:
        ratios=[c['closure']['ratio_of_five_seed_means'] for c in prior['configurations'] if c['dataset']==ds and c['closure']['ratio_meaningful']]
        macros['BiasClosure'+suffix]=f'{100*max(ratios):.2f}'
    macros.update(cic_result_text(groups, comparisons, stats))
    macros.update(retention_outputs(retention))
    macros.update(kd_reassessment_macros(kd_reassessment))
    macros.update(retention_narrative_macros(retention,kd_reassessment))
    macros.update(compact_outputs(compact))
    macros.update(section39_development_outputs(compact))
    macros.update(cse_presentation(compact))
    (PAPER/'generated'/'macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n',encoding='utf-8')
    derived={'cutoff':macros['EvidenceCutoff'],'registered_original_overall':'fail','registered_criteria_status':statuses,
             'comparisons':comparisons,'groups':[{'dataset':ds,'method':m,'n':n(ds,m),'points':{a:{k:stats(ds,m,k,a) for k in ['normal_fpr','attack_detection','macro_f1','balanced_accuracy','attack_guard_recall']} for a in ALPHAS}} for ds in DATASETS for m in METHODS],
             'selected_per_class_summaries':[{'dataset':ds,'method':m,'class_name':name,**values} for (ds,m,name),values in perclass.items()],
             'undefined_precision_aggregation':'Per-seed null preserved; mean/sample SD use only defined precision values with defined_n; other endpoints keep their original seed sets.',
             'selected_sources':[{'dataset':x['dataset'],'method':x['method'],'seed':x['seed'],'source_result':x['source_result'],'completed_at':x['completed_at']} for x in cells.values()],
             'section298':retention, 'section299':kd_reassessment, 'section31':compact}
    (PAPER/'generated'/'paper_results.json').write_text(json.dumps(derived,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    architecture()
    print(json.dumps({'selected_logical_cells':len(cells),'new_aa_cells':macros['NumCompletedNew'],'ReplayBS':{ds:n(ds,'replay_bs') for ds in DATASETS},'criteria':statuses}))
    write_summary_zh(groups, comparisons, statuses, macros, sensitivity, retention, kd_reassessment,compact)


if __name__=='__main__':main()
