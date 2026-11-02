"""CM-derived section344 summaries and paired statistical sensitivity checks.

Run only after the complete saved-checkpoint evaluation. No model evaluation.
"""
import argparse,hashlib,itertools,json,math,statistics
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def holm(values):
    order=sorted(range(len(values)),key=values.__getitem__);out=[0.]*len(values);last=0.
    for rank,index in enumerate(order):last=max(last,min(1.,(len(values)-rank)*values[index]));out[index]=last
    return out
def paired(reference,baseline):
    d=np.array(reference)-np.array(baseline)
    if np.std(d,ddof=1)==0:
        stat,p=(0.,1.) if np.mean(d)==0 else (None,0. if np.mean(d)>0 else 1.)
    else:
        result=ttest_rel(reference,baseline,alternative='greater');stat,p=float(result.statistic),float(result.pvalue)
    exact=sum(float(np.mean(d*np.array(signs)))>=float(d.mean())-1e-12 for signs in itertools.product((1,-1),repeat=len(d)))/2**len(d)
    return dict(mean_paired_difference=float(d.mean()),paired_differences=d.tolist(),paired_t_statistic=stat,paired_t_one_sided_p=p,exact_signflip_one_sided_p=exact,n=len(d),alternative='reference final BAcc > baseline')
def trajectory(matrices,seen):
    histories=[];points=[]
    for cm,classes in zip(matrices,seen):
        cm=np.array(cm,dtype=np.int64);support=cm.sum(1)
        assert np.all(cm>=0) and np.all(support[classes]>0)
        rec={c:float(cm[c,c]/support[c]) for c in classes};histories.append(rec)
        points.append(dict(balanced_accuracy=statistics.mean(rec.values()),attack_recall=float(1-cm[1:,0].sum()/support[1:].sum()),measured_test_fpr=float(1-cm[0,0]/support[0])))
    per_class=[]
    for c in seen[-1]:
        values=[h[c] for h in histories if c in h];prior=values[:-1]
        per_class.append(dict(class_id=c,first=values[0],peak=max(values),best_prior=max(prior) if prior else None,final=values[-1],signed_forgetting=max(prior)-values[-1] if prior else None))
    return dict(**points[-1],incremental_balanced_accuracy=statistics.mean(t['balanced_accuracy'] for t in points),
        signed_best_prior_forgetting=statistics.mean(v['signed_forgetting'] for v in per_class if v['signed_forgetting'] is not None),per_class=per_class)
def summarize(rows):
    keys=('balanced_accuracy','incremental_balanced_accuracy','attack_recall','measured_test_fpr','signed_best_prior_forgetting')
    return {key:dict(mean=statistics.mean(r[key] for r in rows),sample_sd=statistics.stdev(r[key] for r in rows)) for key in keys}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path);p.add_argument('--references',type=Path);p.add_argument('--output',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:
        assert np.allclose(holm([.04,.01,.03,.002]),[.06,.03,.06,.008])
        assert paired([1,2,3,4,5],[0,1,2,3,4])['exact_signflip_one_sided_p']==1/32
        assert paired([1]*5,[1]*5)['paired_t_one_sided_p']==1
        print('SAME_SCORE_AGGREGATION_SOFTWARE_FIXTURES_OK');return
    assert not a.output.exists();report={};sources={};methods=('derpp','icarl','ecbrs-paper','cer-bic-common')
    for ds in ('unsw-nb15','cic-ids-2017','cic-ids-2018'):
        reference=[];dsref='cse-cic-ids2018' if ds=='cic-ids-2018' else ds
        for seed in range(10,15):
            p=a.references/f'paper_aacil/results/section165_confirmation/{dsref}/R3-A_seed{seed}/result.json'
            r=json.loads(p.read_text(encoding='utf-8'));sources[p.relative_to(a.references).as_posix()]=sha(p);reference.append(r)
        report[ds]={}
        for alpha in ('0.01','0.02','0.05'):
            ref=[]
            for r in reference:
                matrices=[];seen=[]
                for j,cp in enumerate(r['checkpoints']):
                    cal=cp if 'operating_points' in cp else r['calibrated_stages'][j]
                    matrices.append(cal['operating_points'][alpha]['confusion_matrix']);seen.append(cp['seen_classes'])
                ref.append(trajectory(matrices,seen))
            rules={}
            for rule in ('original_prior_shift','same_score_holdout'):
                rows={};tests=[]
                for method in methods:
                    values=[]
                    for seed in range(10,15):
                        p=a.results/ds/method/f'seed{seed}/same_score_calibration.json'
                        r=json.loads(p.read_text(encoding='utf-8'));assert not r['software_only'] and not r['registered_primary_changed']
                        sources[p.relative_to(a.results).as_posix()]=sha(p)
                        matrices=([t['views'][alpha]['metrics']['confusion_matrix'] for t in r['stages']] if rule=='same_score_holdout' else [t['operating_points'][alpha]['confusion_matrix'] for t in r['original_prior_shift_stages']])
                        seen=[t['seen_classes'] for t in r['stages']];value=trajectory(matrices,seen)
                        if rule=='same_score_holdout':
                            for key in ('balanced_accuracy','attack_recall','measured_test_fpr'):assert abs(value[key]-r['final'][alpha][key])<1e-12
                            assert abs(value['signed_best_prior_forgetting']-r['final'][alpha]['mean_signed_best_prior_forgetting'])<1e-12
                        values.append(value)
                    rows[method]=dict(per_seed=values,aggregate=summarize(values))
                    tests.append(paired([v['balanced_accuracy'] for v in ref],[v['balanced_accuracy'] for v in values]))
                adjusted=holm([t['paired_t_one_sided_p'] for t in tests]);exact_adjusted=holm([t['exact_signflip_one_sided_p'] for t in tests])
                for method,test,pt,pe in zip(methods,tests,adjusted,exact_adjusted):
                    test.update(paired_t_holm_p=pt,exact_signflip_holm_p=pe,holm_family='four baseline comparisons within dataset and calibration rule',paired_t_reject_at_0_05=pt<=.05)
                    rows[method]['final_bacc_comparison']=test
                rules[rule]=rows
            report[ds][alpha]=dict(reference=summarize(ref),rules=rules)
    output=dict(section344=report,sources_sha256=sources,statistical_primary='one-sided paired t, Holm four comparisons within dataset; 1% registered rule',
        exact_signflip='sensitivity only; five pairs have minimum nonzero one-sided p=1/32',
        registered_primary_changed=False,overall_s1_closed=False,
        limitation='Original prior-shift and reference R3-A values are as published; same-score baselines are new uniform-runtime read-only evaluation. This summary alone does not claim all historical runtime harmonization or remaining S1/research requirements.',
        diagnostic_source_sha256=sha(Path(__file__)))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print('SAME_SCORE_CM_AGGREGATION_COMPLETE_60_CELLS')
if __name__=='__main__':main()
