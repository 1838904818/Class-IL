"""Independent saved-array WACB score/CDF/CM mathematics; never model evaluation."""
import argparse,bisect,collections,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def quantiles(values,probabilities):
    s=np.sort(values.astype(np.float64));out=[]
    for q in probabilities:
        pos=q*(len(s)-1);lo=math.floor(pos);hi=math.ceil(pos);out.append(float(s[lo]+(s[hi]-s[lo])*(pos-lo)))
    return out
def matrix(y,p,n):
    counts=collections.Counter(zip(y.tolist(),p.tolist()));return [[counts[(i,j)] for j in range(n)] for i in range(n)]
def point(m):
    n=len(m);support=[sum(row) for row in m];rec=[m[c][c]/support[c] for c in range(n)]
    return dict(balanced_accuracy=sum(rec)/n,measured_test_fpr=sum(m[0][1:])/support[0],attack_recall=1-sum(m[c][0] for c in range(1,n))/sum(support[1:]),per_class_recall=rec,per_class_to_normal=[m[c][0]/support[c] for c in range(n)])
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    common=np.load(a.results/'shared-test-rows-labels.npz');y=common['test_labels'];rowids=common['test_row_ids'];assert len(y)==3255102 and np.array_equal(rowids,np.arange(len(y)))
    checked=[]
    for seed in range(10,15):
        path=a.results/f'wacb-seed{seed}.json';r=json.loads(path.read_text());oldpath=a.base/f'validation/hpc-20261102T1345/b/protected-b-v2/cic-ids-2018/wacb/seed{seed}/cell/result.json';old=json.loads(oldpath.read_text())
        assert not r['software_only'] and not r['original_cm_discrepancies'];assert r['source_result_sha256']==sha(oldpath) and r['checkpoint_sha256']==old['checkpoints'][-1]['checkpoint_sha256'];assert r['data_binding']==old['identity']['data_binding']
        assert r['diagnostic_source_sha256']=='736c77dc6a926a4c1591b1caf25b646c235e384237952dcbed960ae3e522296c'
        with np.load(path.with_suffix('.npz')) as arrays:
            assert set(arrays.files)=={'holdout_row_ids','holdout_score','test_score','holdout_logits','test_logits'}
            ids=arrays['holdout_row_ids'];h=arrays['holdout_logits'];t=arrays['test_logits'];hs=arrays['holdout_score'];ts=arrays['test_score'];n=t.shape[1]
            assert len(ids)==1070273 and len(h)==len(ids) and len(t)==len(y);assert hashlib.sha256(ids.tobytes()).hexdigest()==r['permanent_normal_holdout_indices_sha256']==old['identity']['section26_holdout']['0']['validation_indices_sha256']
            prior=np.log(np.array(r['original_train_counts'])/sum(r['original_train_counts'])).astype(np.float32);assert np.array_equal(prior,np.array(r['prior_float32'],dtype=np.float32))
            for values,scores in ((h,hs),(t,ts)):
                for off in range(0,len(values),65536):
                    block=values[off:off+65536];calculated=np.max((block[:,1:].astype(float)-block[:,0,None].astype(float))/(prior[0]-prior[1:]).astype(float),axis=1);assert np.array_equal(calculated,scores[off:off+len(block)])
            views={};holdout={}
            for name,expected in r['original_views'].items():
                if name=='default':prediction=t.argmax(1)
                else:
                    tau=np.float32(r['original_thresholds'][name]);prediction=(t+tau*prior).argmax(1);fp=int(np.count_nonzero((h+tau*prior).argmax(1)))
                    original=r['original_holdout_calibration']['operating_points'][name]['calibration_false_positives'];holdout[name]=dict(original_false_positives=original,recomputed_false_positives=fp,delta=fp-original,count=len(h))
                m=matrix(y,prediction,n);assert m==expected['confusion_matrix'],(seed,name);values=point(m)
                for key,v in values.items():assert np.allclose(v,expected[key],rtol=0,atol=1e-12),(seed,name,key)
                views[name]=dict(confusion_matrix=m,**values)
            benign=ts[y==0];ks=float(ks_2samp(hs,benign).statistic);assert abs(ks-r['ks_statistic'])<1e-12
            for values,key in ((hs,'holdout_quantiles'),(benign,'test_benign_quantiles')):assert np.allclose(quantiles(values,r['quantile_probabilities']),r[key],rtol=0,atol=1e-10)
            for c in range(n):
                selected=ts[y==c];assert len(selected)==r['per_class'][str(c)]['support'];assert np.allclose(quantiles(selected,r['quantile_probabilities']),r['per_class'][str(c)]['score_quantiles'],rtol=0,atol=1e-10)
            for values,key in ((hs,'holdout_cdf'),(benign,'test_benign_cdf')):
                ordered=np.sort(values);cdf=[bisect.bisect_right(ordered,v)/len(ordered) for v in r['cdf_knots']];assert np.allclose(cdf,r[key],rtol=0,atol=1e-12) and all(cdf[j]<=cdf[j+1] for j in range(len(cdf)-1))
            checked.append(dict(seed=seed,original_views=views,ks_statistic=ks,holdout_recomputed_fp=holdout,holdout_q99=r['holdout_quantiles'][9],test_benign_q99=r['test_benign_quantiles'][9],original_thresholds=r['original_thresholds'],per_class=r['per_class'],raw_json_sha256=sha(path),raw_npz_sha256=sha(path.with_suffix('.npz'))))
            print(json.dumps(dict(event='WACB_RAW_ARRAY_MATH_VERIFIED',seed=seed,ks=ks,bacc_onepct=views['0.01']['balanced_accuracy'],test_fpr_onepct=views['0.01']['measured_test_fpr'],holdout_delta={k:v['delta'] for k,v in holdout.items()})),flush=True)
    aggregate={view:{key:dict(mean=statistics.mean(v['original_views'][view][key] for v in checked),sample_sd=statistics.stdev(v['original_views'][view][key] for v in checked)) for key in ('balanced_accuracy','measured_test_fpr','attack_recall')} for view in ('default','0.01','0.02','0.05')}
    report=dict(seeds=checked,aggregate=aggregate,checks=dict(cells=5,test_confusion_matrices=20,score_values='all saved holdout/test rows',ks=5,quantiles=5*9*12,cdf='all saved knots',holdout_fp=15),class_order=old['identity']['class_order'],diagnostic_forward_batch=4096,original_config_eval_batch=2048,verifier_sha256=sha(Path(__file__)),scope='Independent mathematics from saved arrays only; no models/training. Original CM exactness is not original logit bitexactness. Holdout FP differences, if present, are retained without threshold changes.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n');print('WACB_FIVE_SEED_RAW_ARRAY_CDF_KS_CM_AND_HOLDOUT_MATHEMATICS_COMPLETE',flush=True)
if __name__=='__main__':main()
