"""Independently verify saved score arrays, KS, quantiles and posthoc CMs."""
import argparse,json,math,statistics
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--train-labels',type=Path,required=True);a=p.parse_args()
counts=np.bincount(np.load(a.train_labels));prior=np.log(counts/counts.sum()).astype(np.float32);rows={};verified=0
for path in sorted(a.results.glob('*.json')):
    r=json.loads(path.read_text());assert not r['software_only'] and not r['original_cm_discrepancies'];d=np.load(path.with_suffix('.npz'));h=d['holdout_score'];s=d['test_score'];y=d['test_labels'];z=d['test_logits']
    assert len(s)==22544 and sum(y==0)==9711 and len(h)==6735
    assert abs(ks_2samp(h,s[y==0]).statistic-r['ks_statistic'])<1e-12
    for values,key in ((h,'holdout_quantiles'),(s[y==0],'test_benign_quantiles')):
        ordered=sorted(float(v) for v in values)
        for q,expected in zip(r['quantile_probabilities'],r[key]):
            index=q*(len(ordered)-1);low=math.floor(index);high=math.ceil(index);v=ordered[low]+(ordered[high]-ordered[low])*(index-low);assert abs(v-expected)<1e-10
    for alpha,point in r['test_informed_operating_points'].items():
        tau=point['threshold'];method=r['method']
        if method in ('R3-L','ar'):
            family=z[:,2:].argmax(1)+1 if method=='ar' else z[:,1:].argmax(1)+1
            pred=np.where(s>tau,family,0)
        else:pred=(z+np.float32(tau)*prior).argmax(1)
        matrix=[[0]*5 for _ in range(5)]
        for true,guess in zip(y.tolist(),pred.tolist()):matrix[true][guess]+=1
        assert matrix==point['confusion_matrix'],(path.name,alpha)
        fp=sum(matrix[0][1:]);support=[sum(row) for row in matrix];fpr=fp/support[0];bacc=sum(matrix[c][c]/support[c] for c in range(5))/5;recall=1-sum(matrix[c][0] for c in range(1,5))/sum(support[1:])
        assert fp<=point['maximum_false_positives']==math.floor(float(alpha)*support[0]);assert abs(bacc-point['balanced_accuracy'])<1e-12 and abs(fpr-point['measured_test_fpr'])<1e-12 and abs(recall-point['attack_recall'])<1e-12
        rows.setdefault((method,alpha),[]).append((bacc,fpr))
    verified+=1
assert verified==40
for (method,alpha),values in rows.items():
    assert len(values)==5;print(json.dumps(dict(method=method,alpha=alpha,bacc_percent_mean=100*statistics.mean(v[0] for v in values),test_fpr_percent_mean=100*statistics.mean(v[1] for v in values))))
print('NSL_REMAINING_INDEPENDENT_40_KS_960_QUANTILES_80_POSTHOC_CMS_OK')
