"""Extract CSE WACB seed spread from unchanged saved matrices and calibration search receipts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean,stdev

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
B=a.base.resolve(strict=True)
assert not a.output.exists()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
rows=[]
for seed in range(10,15):
    root=B/f'validation/hpc-20261102T1345/b/protected-b-v2/cic-ids-2018/wacb/seed{seed}'
    r=read(root/'cell/result.json')
    d=read(root/'default_stages.json')[-1]['metrics']
    c=read(root/'calibrated_stages.json')[-1]
    cal=c['calibration']; point=cal['operating_points']['0.01']
    n=len(r['identity']['class_order'])
    assert r['completed'] and read(root/'summary.json')['completed']
    stage={}
    for view,m in [('default',d)]+list(c['operating_points'].items()):
        cm=m['confusion_matrix'];support=[sum(v) for v in cm]
        assert len(cm)==n and all(len(v)==n for v in cm) and all(s>0 for s in support)
        recalls=[cm[j][j]/support[j] for j in range(n)]
        bacc=mean(recalls);fpr=1-cm[0][0]/support[0]
        assert abs(fpr-m['normal_fpr'])<1e-12
        if 'balanced_accuracy' in m:assert abs(bacc-m['balanced_accuracy'])<1e-12
        else:
            for j,name in enumerate(r['identity']['class_order']):
                assert abs(recalls[j]-m['per_class_recall'][str(j)])<1e-12
        stage[view]=dict(final_bacc=bacc,test_normal_fpr=fpr,
            classes=[dict(class_id=j,name=name,recall=recalls[j],to_normal=cm[j][0]/support[j],support=support[j],prediction_counts=cm[j])
                     for j,name in enumerate(r['identity']['class_order'])])
    assert point['status']=='available' and point['calibration_false_positives']<=point['maximum_false_positives']
    assert point['calibration_lower_false_positives']>point['maximum_false_positives']
    evaluated=sorted(cal['search_evaluations'],key=lambda x:x['tau'])
    assert all(x['false_positives']>=y['false_positives'] for x,y in zip(evaluated,evaluated[1:])), 'Non-monotone stored search'
    tau=point['total_tau'];counts=cal['original_train_counts']
    row=dict(seed=seed,views=stage,onepct_calibration=point,
        holdout_row_id_sha256=cal['calibration_indices_sha256'],stored_holdout_search_evaluations=evaluated,
        relative_normal_logit_advantage=[dict(class_id=j,name=r['identity']['class_order'][j],shift=tau*math.log(counts[0]/counts[j])) for j in range(1,n)],
        calibrated_minus_default_bacc_pp=100*(stage['0.01']['final_bacc']-stage['default']['final_bacc']),
        original_result_sha256=hashlib.sha256((root/'cell/result.json').read_bytes()).hexdigest(),
        calibrated_stages_sha256=hashlib.sha256((root/'calibrated_stages.json').read_bytes()).hexdigest())
    rows.append(row)
    print(json.dumps(dict(seed=seed,tau=tau,holdout_fpr=point['calibration_fpr'],test_fpr=stage['0.01']['test_normal_fpr'],
        bacc=stage['0.01']['final_bacc'],default_bacc=stage['default']['final_bacc'],
        onepct_class_recalls=[x['recall'] for x in stage['0.01']['classes']],onepct_to_normal=[x['to_normal'] for x in stage['0.01']['classes']])),flush=True)
out=dict(method='wacb',dataset='CSE-CIC-IDS2018',job='493534',seeds=rows,
    final_onepct_bacc=dict(mean=mean(x['views']['0.01']['final_bacc'] for x in rows),sample_sd=stdev(x['views']['0.01']['final_bacc'] for x in rows)),
    finding='All five stored searches are monotone and bracket the requested holdout false-positive budget. Different selected taus are not evidence that the bisection solver malfunctioned.',
    scope='Saved CM and search-count decomposition only; raw completion does not become checkpoint/prediction acceptance here.',
    limitation='Stored search counts characterize holdout tail crossings only. Full holdout and TEST score arrays are not in this small JSON snapshot; KS statistics, score quantiles and a causal account of between-seed score geometry have NOT been computed. No claim of an unstable solver, seed-specific tuning or causal budget effect.',
    formulas=dict(bacc='mean(class true positives / class supports)',test_fpr='1-Normal true positives/Normal support',
        relative_normal_logit_advantage='tau * log(original TRAIN Normal count / original TRAIN class count)'))
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print('WACB_STORED_OPERATING_POINT_DIAGNOSIS_COMPLETE',flush=True)
