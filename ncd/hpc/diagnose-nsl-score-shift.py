"""Read-only final-checkpoint score shift and explicitly TEST-informed operating points.

Only original, hash-bound, locally produced checkpoints may be deserialized.
No fitting, optimizer, original threshold or result update is performed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import numpy as np
import torch
import torch.nn.functional as F

def ks_distance(a,b):
    a,b=np.sort(a),np.sort(b)
    grid=np.unique(np.concatenate((a,b)))
    return float(np.max(np.abs(np.searchsorted(a,grid,side='right')/len(a)-np.searchsorted(b,grid,side='right')/len(b))))

def matrix(y,p,n):
    return np.bincount(y*n+p,minlength=n*n).reshape(n,n)

def metrics(cm):
    return dict(confusion_matrix=cm.tolist(),balanced_accuracy=float(np.mean(np.diag(cm)/cm.sum(1))),
                measured_test_fpr=float(1-cm[0,0]/cm[0].sum()))

def gate_predict(logits,threshold):
    v=torch.from_numpy(logits)
    margin=(torch.logsumexp(v[:,1:],dim=1)-v[:,0]).numpy()
    return np.where(margin>threshold,logits[:,1:].argmax(1)+1,0)

def shifted_predict(logits,prior,tau):
    return (torch.from_numpy(logits)+float(tau)*torch.from_numpy(prior)).argmax(1).numpy()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path)
    p.add_argument('--data',type=Path)
    p.add_argument('--stateless-root',type=Path)
    p.add_argument('--derpp-root',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(10,15)))
    p.add_argument('--methods',nargs='+',choices=('R3-A','derpp'),default=['R3-A','derpp'])
    p.add_argument('--sample-per-class',type=int,default=0)
    p.add_argument('--require-exact',action='store_true')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if a.self_test:
        from scipy.stats import ks_2samp
        for x,y in (([0,0,1,3],[1,1,2,2,4]),([1,2,3],[1,2,3]),([0,0],[1,1])):
            assert abs(ks_distance(np.array(x),np.array(y))-ks_2samp(x,y).statistic)<1e-12
        assert metrics(matrix(np.array([0,0,1,1]),np.array([0,1,1,0]),2))['balanced_accuracy']==.5
        print('SOFTWARE_ONLY_KS_AND_CM_FIXTURES_PASSED');return
    import os
    assert a.output and not a.output.exists()
    sys.path.insert(0,str(a.source.resolve(strict=True)))
    if os.name!='nt' and os.environ.get('NCD_SKLEARN_ROOT'):
        sys.path.insert(1,os.environ['NCD_SKLEARN_ROOT'])
    import sklearn
    from ncd.section156_models import ReplayGBDTGate
    from ncd.calibrate_aacil_baselines import select_taus
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    xt=np.load(a.data/'train_x.npy',mmap_mode='r');yt=np.load(a.data/'train_y.npy')
    xe=np.load(a.data/'test_x.npy',mmap_mode='r');ye=np.load(a.data/'test_y.npy')
    counts=np.bincount(yt);n=len(counts)
    prior=np.log(counts/counts.sum()).astype(np.float32)
    assert np.all(prior[0]>prior[1:]) and n==5
    normal_ids=np.flatnonzero(yt==0)
    test_rows=np.arange(len(ye))
    if a.sample_per_class:
        assert a.sample_per_class>0
        test_rows=np.sort(np.concatenate([np.flatnonzero(ye==c)[:a.sample_per_class] for c in range(n)]))
    meta=read(a.data/'data_quality.json')
    truth=ye[test_rows];names=meta['feature_names']
    binding={key:dict(dtype=v.dtype.str,shape=list(v.shape),sha256=hashlib.sha256(memoryview(v).cast('B')).hexdigest())
             for key,v in dict(x_train=xt,y_train=yt,x_test=xe,y_test=ye).items()}
    a.output.mkdir(parents=True,exist_ok=False)
    failures=[]
    for method in a.methods:
        for seed in a.seeds:
            root=(a.stateless_root if method=='R3-A' else a.derpp_root)/f'seed{seed}'
            r=read(root/'cell/result.json');cp=r['checkpoints'][-1]
            assert r['identity']['data_binding']==binding and r['class_order']==meta['class_order']
            path=root/'cell'/cp['checkpoint_file']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==cp['checkpoint_sha256']
            state=torch.load(path,map_location='cpu',weights_only=True)
            assert state['identity']==r['identity'] and cp['seen_classes']==list(range(n))
            holdout=np.random.default_rng(seed*100003+17).permutation(normal_ids)[:max(5,math.ceil(len(normal_ids)*.1))]
            expected_hash=(cp['calibration']['0.01']['row_ids_sha256'] if method=='R3-A' else r['identity']['section26_holdout']['0']['validation_indices_sha256'])
            assert hashlib.sha256(holdout.tobytes()).hexdigest()==expected_hash
            if a.sample_per_class:holdout=holdout[:a.sample_per_class]
            if method=='R3-A':
                gate=ReplayGBDTGate(n)
                assert state['gate_state']['_extra_state']['estimator_pickle']==state['family_state']['joint._extra_state']['estimator_pickle']
                gate.load_state_dict(state['gate_state'],strict=True);gate.eval()
                evaluate=lambda v:gate.joint_logits(v)
            else:
                w=state['model_state']
                def evaluate(v):
                    v=F.relu(F.linear(v,w['features.0.weight'],w['features.0.bias']))
                    v=F.relu(F.linear(v,w['features.2.weight'],w['features.2.bias']))
                    return F.linear(v,w['head.weight'],w['head.bias'])
            batch=int(r['identity']['effective_config']['eval_batch_size'])
            def score(array,ids):
                pieces=[]
                with torch.inference_mode():
                    for start in range(0,len(ids),batch):
                        v=torch.from_numpy(np.array(array[ids[start:start+batch]],dtype=np.float32,copy=True))
                        pieces.append(evaluate(v).numpy())
                out=np.concatenate(pieces)
                assert np.isfinite(out).all()
                return out
            cal_logits=score(xt,holdout);test_logits=score(xe,test_rows)
            if method=='R3-A':
                scalar=lambda v:(torch.logsumexp(torch.from_numpy(v[:,1:]),dim=1)-torch.from_numpy(v[:,0])).numpy()
                cscore,tscore=scalar(cal_logits),scalar(test_logits)
                original_cal=cp['calibration']
                original_points=cp['operating_points']
                original_default=cp['default_decision']['confusion_matrix']
                decision=lambda v,tau:gate_predict(v,tau)
                thresholds={k:v['threshold'] for k,v in original_cal.items()}
                selected={}
                benign=np.sort(tscore[truth==0])
                for alpha in (.02,.05):
                    allowed=math.floor(alpha*len(benign));tau=float(benign[len(benign)-allowed-1])
                    selected[str(alpha)]=dict(threshold=tau,allowed_false_alarms=allowed)
                score_definition='logsumexp(joint attack logits)-joint Normal logit'
            else:
                scalar=lambda v:np.max((v[:,1:].astype(np.float64)-v[:,0,None].astype(np.float64))/(prior[0]-prior[1:]).astype(np.float64),axis=1)
                cscore,tscore=scalar(cal_logits),scalar(test_logits)
                c=read(root/'calibrated_stages.json')[-1]
                original_points=c['operating_points']
                original_default=read(root/'default_stages.json')[-1]['metrics']['confusion_matrix']
                thresholds={k:v['total_tau'] for k,v in c['calibration']['operating_points'].items()}
                decision=lambda v,tau:shifted_predict(v,prior,tau)
                _,points,_,_=select_taus(torch.from_numpy(test_logits[truth==0]),torch.from_numpy(prior),0)
                selected={k:dict(threshold=v['total_tau'],allowed_false_alarms=v['maximum_false_positives']) for k,v in points.items() if k in ('0.02','0.05')}
                score_definition='Maximum attack-versus-Normal prior-shift crossing tau; ideal real-arithmetic boundary, not cross-method comparable units. Decisions retain original float32 operations.'
            discrepancies=[]
            if not a.sample_per_class:
                if matrix(truth,test_logits.argmax(1),n).tolist()!=original_default:discrepancies.append('default')
                for key,tau in thresholds.items():
                    if matrix(truth,decision(test_logits,tau),n).tolist()!=original_points[key]['confusion_matrix']:discrepancies.append(key)
            key='0.01';alarm=decision(test_logits,thresholds[key])!=0
            alarm_ids=test_rows[(truth==0)&alarm]
            breakdown={}
            for field in ('service','flag'):
                prefix=f'categorical__{field}_';cols=[j for j,s in enumerate(names) if s.startswith(prefix)]
                block=np.asarray(xe[alarm_ids][:,cols])
                categories=[names[cols[j]][len(prefix):] if float(row.sum())>0 else 'UNKNOWN_TO_TASK0_ENCODER' for row,j in zip(block,block.argmax(1))]
                from collections import Counter
                breakdown[field]=dict(sorted(Counter(categories).items()))
            posthoc={}
            for key,selection in selected.items():
                cm=matrix(truth,decision(test_logits,selection['threshold']),n)
                assert int(cm[0].sum()-cm[0,0])<=selection['allowed_false_alarms']
                posthoc[key]=dict(selection,**metrics(cm),test_label_informed=True,registered_result=False,
                    target_role='At-most requested measured TEST benign FPR; ties may make actual FPR lower.')
            probs=np.array([0,.01,.05,.25,.5,.75,.9,.95,.98,.99,.995,1.])
            cfg=dict(r['identity']['effective_config']);cfg['problem_scout_data_root']='@@DATA_ROOT@@'
            record=dict(method=method,seed=seed,checkpoint_sha256=cp['checkpoint_sha256'],software_only=bool(a.sample_per_class),
                source_git_commit=r['identity']['git_commit'],effective_config=cfg,
                source_result_sha256=hashlib.sha256((root/'cell/result.json').read_bytes()).hexdigest(),
                diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                diagnostic_command=['python','-I','-B','diagnose-nsl-score-shift.py','--source','@@FROZEN_ROOT@@','--data','@@DATA_ROOT@@',
                    '--stateless-root','@@R3A_ROOT@@','--derpp-root','@@DERPP_ROOT@@','--output','@@OUTPUT_ROOT@@',
                    '--methods',*a.methods,'--seeds',*[str(s) for s in a.seeds]]+
                    (['--sample-per-class',str(a.sample_per_class)] if a.sample_per_class else [])+(['--require-exact'] if a.require_exact else []),
                log_file='@@OUTPUT_ROOT@@/diagnosis.stdout.log',
                runtime=dict(platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__),
                score_definition=score_definition,holdout_rows=len(holdout),test_normal_rows=int((truth==0).sum()),
                ks_statistic=ks_distance(cscore,tscore[truth==0]),quantile_probabilities=probs.tolist(),
                holdout_quantiles=np.quantile(cscore,probs).tolist(),test_benign_quantiles=np.quantile(tscore[truth==0],probs).tolist(),
                original_result_cm_discrepancies=discrepancies,original_holdout_threshold=thresholds['0.01'],
                original_onepct_test_fpr=float(len(alarm_ids)/(truth==0).sum()),alarm_test_row_ids=alarm_ids.tolist(),alarm_category_counts=breakdown,
                test_informed_operating_points=posthoc,limitation='Score-shift association is not a causal mechanism; raw records/features are not rewritten. No S1 test or original calibration is replaced.')
            stem=f'{method}-seed{seed}'
            (a.output/(stem+'.json')).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            np.savez_compressed(a.output/(stem+'.npz'),holdout_row_ids=holdout,test_row_ids=test_rows,test_labels=truth,
                holdout_logits=cal_logits,test_logits=test_logits,holdout_scores=cscore,test_scores=tscore)
            print(json.dumps(dict(event='NSL_SCORE_DIAGNOSIS',method=method,seed=seed,ks=record['ks_statistic'],software_only=record['software_only'],reference_cm_discrepancies=discrepancies)),flush=True)
            if discrepancies:failures.append(dict(method=method,seed=seed,views=discrepancies))
    if os.name=='nt':
        import psutil
        peak=psutil.Process().memory_info().peak_wset
    else:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    print(json.dumps(dict(event='NSL_DIAGNOSTIC_PROCESS_PEAK',bytes=peak)),flush=True)
    if a.require_exact and failures:raise RuntimeError(f'Original-runtime CM reconciliation failed: {failures}')

if __name__=='__main__':main()
