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
import json
from pathlib import Path
import statistics as st
import numpy as np
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
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
    source_summary=ROOT/a['source_summary']
    if read(source_summary)['completed_at']!=a['source_summary_completed_at']:
        raise ValueError('Stagewise summary is newer; regenerate Section29.8 comparisons before paper results')
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
                'RetentionDecisionText':'Section 29.8 remains pending; no success is inferred.',
                'RetentionComparisonAppendix':appendix}
    main=evidence['selected_main']
    if main:
        cfg=main['configuration']
        choice=(RETENTION_NAMES[main['method']]+rf" ($\lambda_G={cfg['lambda_g']:g}$")
        if main['method']=='aacil_cbf':
            choice+=f", {cfg['cbf_epochs']} CBF epoch(s), learning-rate multiplier {cfg['cbf_lr_multiplier']:g}"
        choice+=')'
        selection_text=('UNSW training-validation seeds 5/6 locked '+choice+' as the eligible main version. '
            'This is a development selection, not a formal test result; test outcomes cannot reselect the version. ')
    else:
        selection_text='No version met the development eligibility constraint; no main version was selected. '
    ineligible=[RETENTION_NAMES[v['method']] for v in evidence['per_version'] if not v['eligible']]
    selection_text+=(', '.join(ineligible)+' remain mandatory descriptive formal fallbacks, not eligible successes.'
        if ineligible else 'All five versions remain in the formal report.')
    rows=[]
    for v in evidence['per_version']:
        c=v['configuration']
        rows.append([RETENTION_NAMES[v['method']],f"{c['lambda_g']:g}",
            str(c['cbf_epochs']) if v['method']=='aacil_cbf' else '--',
            f"{c['cbf_lr_multiplier']:g}" if v['method']=='aacil_cbf' else '--',
            'eligible main' if main and v['method']==main['method'] else 'eligible alternative' if v['eligible'] else 'ineligible fallback'])
    table('retention_selection','Locked Section 29.8 development selection; no test performance is shown here.',
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
    table('retention_results','Formal Section 29.8 results at stagewise Normal-holdout calibration target $\\alpha=2\\%$.',
          ['Data','Version','n','Mean inc. BAcc','Final BAcc','Forget (pp)','ADR','MF1','FPR'],rows,'llrrrrrrr',
          note='Only complete seeds 1/2/3/4/42 receive means and sample SD; incomplete groups show n and dashes. All outcomes come from formal test trajectories, never development selection metrics. Signed forgetting includes old Normal; rates/accuracies are percentages. Eligibility is a development property, not formal success.')
    formal_text=('Complete formal five-seed groups in the saved assessment: '+', '.join(completed)+'. '
                 'Incomplete groups have no aggregate performance or inferential conclusion in this presentation.')
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
            +fmt(iso['metrics']['final_balanced_accuracy'])+r'\%; low forgetting alone does not establish effective class-incremental learning. '
            +'ISO minus KD differences were '+fmt(forget['delta_pp'],1)+' pp for forgetting '
            +r'($p_{<}='+probability_tex(forget['p_less'])+r'$) and '
            +fmt(incremental['delta_pp'],1)+' pp for mean incremental balanced accuracy '
            +r'($p_{<}='+probability_tex(incremental['p_less'])+r'$), using five paired seeds and unadjusted one-sided tests. ')
        fallback=all(not v['eligible'] for v in evidence['per_version'] if v['method'] in ['aacil_iso','aacil_kd'])
        formal_text+=('These development-ineligible fallback contrasts' if fallback else 'These descriptive contrasts')
        formal_text+=' do not alter the validation-locked main selection.'
    verdict=evidence['verdict']
    decision=('Section 29.8 fails a necessary condition; the false-alarm--forgetting trade-off remains unresolved.' if verdict=='fail'
              else 'The selected main satisfies the three Section 29.8 conditions on both required datasets.' if verdict=='pass'
              else 'Section 29.8 remains pending until the selected main and its required comparisons are complete on both UNSW and NSL.')
    decision+=' Nonsignificant harm is not equivalence or noninferiority. The original Section 27.8 failure and its comparison set are unchanged.'
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
        'Completed Section 29.8 necessary comparisons for the validation-selected main version.',
        ['Data','Criterion','Reference','n',r'$\Delta$ (pp)','One-sided $p$','Decision'],'llrrrll',
        r'Candidate minus reference; exact five paired formal seeds. 1: all-old forgetting, $p_{>}<.05$ means harm/failure against the lowest-forgetting external baseline. '
        r'2: final BAcc, $p_{>}<.05$ is required against Replay-BS. 3: ADR, $p_{<}<.05$ means harm/failure against original AA-CIL. '
        r'Only useful complete comparisons are shown; absent comparisons remain pending and are not passed. Both UNSW and NSL are required. No detected harm is not equivalence. Unadjusted tests.'),
        ('retention_iso_kd',iso_rows,'Completed descriptive AA-CIL-ISO minus AA-CIL-KD comparisons.',
        ['Data','Metric','n',r'$\Delta$ (pp)',r'$p_{<}$',r'$p_{>}$',r'$p_{2s}$'],'llrrrrr',
        r'Exact five paired formal seeds; mean$\pm$sample SD of differences in percentage points. These unadjusted comparisons neither select the main version nor add a success requirement. '
        r'Undefined tests remain --; no seed is discarded. Prior Sections 27.8 and 29.4 are unchanged.')]
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


def kd_reassessment_evidence(path):
    if not path.exists():
        return {}
    row=read(path)
    if (row.get('section')!='29.9' or row.get('candidate')!='aacil_kd' or
            row.get('blind_confirmation_established') is not False or
            row.get('original_section298_cbf_verdict')!='fail'):
        raise ValueError('Expected the separately scoped Section29.9 KD reassessment')
    if read(ROOT/row['source_summary'])['completed_at']!=row['source_summary_completed_at']:
        raise ValueError('Regenerate Section29.9 comparisons for the current stagewise snapshot')
    return row | {'comparison_result':path.relative_to(ROOT).as_posix()}


def kd_reassessment_macros(evidence):
    if not evidence:
        return {'KDReassessmentText':'The separately authorized KD reassessment remains pending; it cannot amend the original CBF failure.',
                'KDReassessmentTables':''}
    status=evidence['required_numeric_verdict']
    conclusion=('A necessary numerical condition fails; the trade-off remains unresolved.' if status=='fail' else
        'The specified numerical conditions pass on both required datasets, but this does not establish independent blind confirmation.' if status=='pass' else
        'The required NSL/CIC17 numerical comparison remains incomplete; no success claim is made.')
    text=(r'Section 29.9 reconsiders KD with fixed $\lambda_G=.5$ and DER $\alpha=\beta=1$; its UNSW outcome is exploratory. '
        'NSL seeds 1/2 were available before the document commit at 20:47:57 MYT, and all five were available before direct authorization at approximately 21:29 MYT; '
        'NSL is therefore not represented as blind confirmation, and availability does not imply that the user inspected the outcomes. '
        +conclusion+' The original Section 29.8 CBF failure and Section 27.8 failure remain unchanged.')
    return {'KDReassessmentText':text,
            'KDReassessmentTables':r'\section{Separately scoped KD reassessment}\input{generated/section299_comparisons.tex}'}


def kd_reassessment_summary_zh(evidence):
    if not evidence:
        return []
    counts='、'.join(f"{SHORT[d['dataset']]} KD {d['n']}/5（{d['status']}）" for d in evidence['dataset_results'])
    return ['**29.9 单独重评 KD：** '+counts+'。UNSW 为探索性；NSL seeds1/2 在20:47:57文档提交前已可得，'
            '五个均在约21:29直接授权前已可得，因此不是盲法确认，也不推测用户是否看过。'
            'NSL/CIC 两者的三个固定条件仍须完整；当前数值判定为 '+evidence['required_numeric_verdict']+'，不改原CBF失败。', '']


def retention_summary_zh(evidence, aggregate_text):
    if not evidence['selection_complete']:
        return ['**29.8 开发选择尚未完成，正式结论待齐。** 不使用开发数字充当正式测试结果。']
    main=evidence['selected_main']
    lines=[]
    if main:
        c=main['configuration']
        config=f"λG={c['lambda_g']:g}"
        if main['method']=='aacil_cbf':
            config+=f"，CBF epochs={c['cbf_epochs']}，LR 倍率={c['cbf_lr_multiplier']:g}"
        lines.append(f"**原29.8锁定主版本：{RETENTION_NAMES[main['method']]}（{config}）**。"
            '仅用 UNSW 训练验证 seeds5/6，先满足最终 BAcc≥原 AA−3 pp，再选最低遗忘；开发选择不是正式测试成功。')
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
    lines+=['', '正式 seeds1/2/3/4/42 齐全才显示均值±样本 SD；-- 为未齐，不以开发数字填补。']
    if evidence['assessment_available']:
        labels={'pass':'满足当前条款','fail':'FAIL（至少一项必要条件失败）','pending':'未完成'}
        lines.append('29.8：**'+labels.get(evidence['verdict'],evidence['verdict'])+'**。'
            'UNSW 与 NSL 均须满足：遗忘不显著劣于最低遗忘外部基线、最终 BAcc 显著高于 Replay-BS、ADR 不显著低于原 AA；'
            f"[完整配对结果](../{evidence['comparison_result']})。")
    else:
        lines.append('正式 29.8 比较文件尚未生成；表中 n=0 指当前已纳入摘要的组数，不代表训练进程状态。')
    lines+=['不显著有害不等于等效或非劣；已知必要条件失败优先于未完成。', '']
    return lines


def write_summary_zh(groups, comparisons, statuses, macros, sensitivity, retention, kd_reassessment):
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
    lines = [
        '# AA-CIL 中文摘要',
        '',
        f"截止：{macros['EvidenceCutoff']}。**检出收益不能替代类增量保持，原 27.8 注册结论仍为 FAIL。** "
        f"原 AA 的 UNSW 增量/最终 BAcc 为 {original('UNSW-NB15','average_incremental_accuracy')}/"
        f"{original('UNSW-NB15','final_balanced_accuracy')}%，遗忘 {original('UNSW-NB15','forgetting_all_old_classes')} pp；"
        f"NSL 遗忘 {original('NSL-KDD','forgetting_all_old_classes')} pp。",
        '',
        *retention_summary_zh(retention, aggregate_text),
        *kd_reassessment_summary_zh(kd_reassessment),
        '口径：每阶段用永久训练域 Normal 留出池定 α=2%；增量 BAcc 为各阶段已见类均值，'
        '遗忘为旧类此前最高 recall 减最终 recall（含 Normal、保留负值）。低遗忘须联看 BAcc，ADR 不等于家族识别。',
        '',
        '边界：校准目标不保证测试 FPR；Replay-BS 匹配曝光但不匹配总计算。正式顺序 UNSW→NSL；'
        '21:50:26 MYT 硬截止，未完成项目保留缺失；29.8要求UNSW/NSL，29.9另要求NSL/CIC，不能互换。解释漂移、治理及总体创新仍未闭合。',
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
        return f'{value:.5f}' if value is not None and np.isfinite(value) else 'undefined'

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
    interpretation.append('These contrasts do not establish causal component effects or equivalence; '
                          'Replay-BS matches the specified benign exposure and replay allocation, not the combined G+F compute budget.')
    coverage = 'All three CIC AA-family groups contain five seeds' if all(complete(m) for m in family) else 'CIC AA-family coverage remains incomplete'
    conclusion = (coverage+', but the original Section 27.8 overall decision remains FAIL. '
                  'The separately reported external-only and post-hoc comparisons do not replace that decision, '
                  'prove novelty or causal benefit, or constitute compute-matched evidence.')
    return {'CICPrimaryResultsText':' '.join(primary),
            'CICInterpretationText':' '.join(interpretation), 'CICConclusionText':conclusion,
            'CICResultStatus':'CIC-IDS-2017 AA-family coverage: '+str(sum(count(m) for m in family))
                +'/15 cells across AA-CIL ($n='+str(count('aacil'))+'/5$), TS-mem ($n='
                +str(count('ts_mem'))+'/5$), and AA-noAttMem ($n='+str(count('aa_noattmem'))
                +'/5$); Replay-BS is reported separately.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',action='append',default=[]);ap.add_argument('--cutoff',default=None)
    ap.add_argument('--retention-selection',type=Path,default=ROOT/'ncd/results/problem_scout/section29/retention_queue_v1/selection.json')
    ap.add_argument('--retention-comparisons',type=Path,default=ROOT/'ncd/results/problem_scout/section29/retention_comparisons_v1/result.json')
    ap.add_argument('--kd-reassessment',type=Path,default=ROOT/'ncd/results/problem_scout/section29/section299_comparisons_v1/result.json')
    args=ap.parse_args()
    retention=retention_evidence(args.retention_selection.resolve(),args.retention_comparisons.resolve())
    kd_reassessment=kd_reassessment_evidence(args.kd_reassessment.resolve())
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
          note='Original Section 27.8 criteria 1--5: '+', '.join(f'{i+1}: {v}' for i,v in enumerate(statuses))+'. Overall registered decision: FAIL, because NSL and UNSW cannot contribute the required two significant wins. The registered set includes TS-mem and AA-noAttMem. No detected harm is not equivalence/noninferiority.')
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
          note=r'Each method uses the same paired seeds in both rows. TEST-Normal thresholds use test Normal labels and are evaluated on the same test population: this is a test-label-conditioned, non-deployable sensitivity analysis, not an independent evaluation. It does not change the registered Section 27.8 decisions. Incomplete seed groups remain descriptive.')
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
    (PAPER/'generated'/'macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n',encoding='utf-8')
    derived={'cutoff':macros['EvidenceCutoff'],'registered_original_overall':'fail','registered_criteria_status':statuses,
             'comparisons':comparisons,'groups':[{'dataset':ds,'method':m,'n':n(ds,m),'points':{a:{k:stats(ds,m,k,a) for k in ['normal_fpr','attack_detection','macro_f1','balanced_accuracy','attack_guard_recall']} for a in ALPHAS}} for ds in DATASETS for m in METHODS],
             'selected_per_class_summaries':[{'dataset':ds,'method':m,'class_name':name,**values} for (ds,m,name),values in perclass.items()],
             'undefined_precision_aggregation':'Per-seed null preserved; mean/sample SD use only defined precision values with defined_n; other endpoints keep their original seed sets.',
             'selected_sources':[{'dataset':x['dataset'],'method':x['method'],'seed':x['seed'],'source_result':x['source_result'],'completed_at':x['completed_at']} for x in cells.values()],
             'section298':retention, 'section299':kd_reassessment}
    (PAPER/'generated'/'paper_results.json').write_text(json.dumps(derived,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    architecture()
    print(json.dumps({'selected_logical_cells':len(cells),'new_aa_cells':macros['NumCompletedNew'],'ReplayBS':{ds:n(ds,'replay_bs') for ds in DATASETS},'criteria':statuses}))
    write_summary_zh(groups, comparisons, statuses, macros, sensitivity, retention, kd_reassessment)


if __name__=='__main__':main()
