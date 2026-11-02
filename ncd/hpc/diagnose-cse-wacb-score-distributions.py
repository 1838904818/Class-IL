"""Read-only WACB final-checkpoint score transfer diagnostics on CSE2018.

Raw prior-crossing scores describe distributions; all decisions retain the
original float32 prior-shift operation, including boundary rounding and ties.
"""
import argparse,hashlib,json,math,platform,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import ks_2samp

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(y,p,n):
    m=np.bincount(y*n+p,minlength=n*n).reshape(n,n);support=m.sum(1)
    return dict(confusion_matrix=m.tolist(),balanced_accuracy=float(np.mean(np.diag(m)/support)),measured_test_fpr=float(1-m[0,0]/support[0]),attack_recall=float(1-m[1:,0].sum()/support[1:].sum()),per_class_recall=(np.diag(m)/support).tolist(),per_class_to_normal=(m[:,0]/support).tolist())
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cells',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=list(range(10,15)));p.add_argument('--sample-per-class',type=int,default=0);a=p.parse_args()
    assert not a.output.exists() and a.sample_per_class>=0;torch.set_num_threads(2);torch.set_num_interop_threads(1)
    data=tuple(np.load(a.data/(name+'.npy'),mmap_mode='r') for name in ('train_x','train_y','test_x','test_y'));x,y,xt,yt=data
    binding={key:dict(dtype=v.dtype.str,shape=list(v.shape),sha256=hashlib.sha256(memoryview(v).cast('B')).hexdigest()) for key,v in zip(('x_train','y_train','x_test','y_test'),data)}
    counts=np.bincount(y);n=len(counts);prior=np.log(counts/counts.sum()).astype(np.float32);assert np.all(prior[0]>prior[1:])
    test_ids=np.arange(len(yt))
    if a.sample_per_class:test_ids=np.sort(np.concatenate([np.flatnonzero(yt==c)[:a.sample_per_class] for c in range(n)]))
    truth=yt[test_ids];a.output.mkdir(parents=True);np.savez_compressed(a.output/'shared-test-rows-labels.npz',test_row_ids=test_ids,test_labels=truth)
    failures=[];completed=0;quantiles=[0,.01,.05,.25,.5,.75,.9,.95,.98,.99,.995,1.]
    for seed in a.seeds:
        try:
            start=time.perf_counter();root=a.cells/f'seed{seed}';cell=root/'cell' if (root/'cell/result.json').exists() else root;r=read(cell/'result.json');i=r['identity'];cp=r['checkpoints'][-1]
            assert r['completed'] and i['seed']==seed and i['data_binding']==binding and cp['seen_classes']==list(range(n)) and r['class_order'][0]=='Normal'
            path=cell/cp['checkpoint_file'];assert sha(path)==cp['checkpoint_sha256'];s=torch.load(path,map_location='cpu',weights_only=True);assert s['identity']==i;w=s['model_state']
            ids=np.random.default_rng(seed*100003+17).permutation(np.flatnonzero(y==0))[:max(5,math.ceil(int(counts[0])*.1))];holdout_hash=hashlib.sha256(ids.tobytes()).hexdigest();assert holdout_hash==i['section26_holdout']['0']['validation_indices_sha256']
            if a.sample_per_class:ids=ids[:a.sample_per_class]
            def logits(array,indices):
                result=np.empty((len(indices),n),dtype=np.float32)
                with torch.inference_mode():
                    for off in range(0,len(indices),4096):
                        v=torch.from_numpy(np.array(array[indices[off:off+4096]],dtype=np.float32,copy=True));v=F.relu(F.linear(v,w['features.0.weight'],w['features.0.bias']));v=F.relu(F.linear(v,w['features.2.weight'],w['features.2.bias']));v=F.linear(v,w['head.weight'],w['head.bias']);assert torch.isfinite(v).all();result[off:off+len(v)]=v.numpy()
                return result
            h=logits(x,ids);t=logits(xt,test_ids)
            def crossing(z):return np.max((z[:,1:].astype(np.float64)-z[:,0,None].astype(np.float64))/(prior[0]-prior[1:]).astype(np.float64),axis=1)
            hs=crossing(h);ts=crossing(t);cal=read(root/'calibrated_stages.json')[-1];defaults=read(root/'default_stages.json')[-1]['metrics'];views={'default':metrics(truth,t.argmax(1),n)};discrepancies=[]
            if not a.sample_per_class and views['default']['confusion_matrix']!=defaults['confusion_matrix']:discrepancies.append('default')
            thresholds={}
            for alpha,point in cal['calibration']['operating_points'].items():
                tau=point['total_tau'];thresholds[alpha]=tau;pred=(torch.from_numpy(t)+float(tau)*torch.from_numpy(prior)).argmax(1).numpy();views[alpha]=metrics(truth,pred,n)
                if not a.sample_per_class and views[alpha]['confusion_matrix']!=cal['operating_points'][alpha]['confusion_matrix']:discrepancies.append(alpha)
            assert not discrepancies,discrepancies
            benign=ts[truth==0];hold=np.sort(hs);test=np.sort(benign);grid=np.unique(np.concatenate((hold,test)));ks=float(np.max(np.abs(np.searchsorted(hold,grid,side='right')/len(hold)-np.searchsorted(test,grid,side='right')/len(test))))
            assert abs(ks-float(ks_2samp(hs,benign).statistic))<1e-12
            # CDFs evaluated on shared, within-model quantile knots. Different models' absolute score units are not interchangeable.
            knots=np.unique(np.quantile(np.concatenate((hs,benign)),np.linspace(0,1,101)))
            per_class={str(c):dict(support=int(sum(truth==c)),score_quantiles=np.quantile(ts[truth==c],quantiles).tolist()) for c in range(n)}
            record=dict(seed=seed,method='wacb',dataset=i['dataset'],software_only=bool(a.sample_per_class),source_git_commit=i['git_commit'],effective_config=i['effective_config'],command=sys.argv,log_file='@@OUTPUT_ROOT@@/diagnosis.stdout.log',checkpoint_sha256=cp['checkpoint_sha256'],source_result_sha256=sha(cell/'result.json'),diagnostic_source_sha256=sha(Path(__file__)),data_binding=binding,permanent_normal_holdout_indices_sha256=holdout_hash,original_train_counts=counts.tolist(),prior_float32=prior.tolist(),runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,platform=platform.system()),score_definition='max_attack((attack_logit-Normal_logit)/(float32_prior_Normal-float32_prior_attack)); descriptive real-arithmetic crossing tau. Original predictions retain torch float32 prior shift.',original_cm_discrepancies=discrepancies,original_thresholds=thresholds,original_holdout_calibration=cal['calibration'],original_views=views,quantile_probabilities=quantiles,holdout_quantiles=np.quantile(hs,quantiles).tolist(),test_benign_quantiles=np.quantile(benign,quantiles).tolist(),ks_statistic=ks,cdf_knots=knots.tolist(),holdout_cdf=(np.searchsorted(hold,knots,side='right')/len(hold)).tolist(),test_benign_cdf=(np.searchsorted(test,knots,side='right')/len(test)).tolist(),per_class=per_class,elapsed_seconds=time.perf_counter()-start,scope='Read-only score-transfer association and per-class geometry description; no training/calibration changes or feature-causal explanation. Large seed BAcc spread with near1% TEST FPR does not establish numerical bisection failure.')
            stem=a.output/f'wacb-seed{seed}';stem.with_suffix('.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8');np.savez_compressed(stem.with_suffix('.npz'),holdout_row_ids=ids,holdout_score=hs,test_score=ts,holdout_logits=h,test_logits=t)
            completed+=1;print(json.dumps(dict(event='CSE_WACB_SCORE_CELL_COMPLETE',seed=seed,software_only=bool(a.sample_per_class))),flush=True)
        except Exception as e:traceback.print_exc();failures.append(dict(seed=seed,error=repr(e)))
    print(json.dumps(dict(event='CSE_WACB_SCORE_COMPLETE',completed=completed,failures=failures,software_only=bool(a.sample_per_class))),flush=True);raise SystemExit(bool(failures))
if __name__=='__main__':main()
