"""Read-only final scores for the eight NSL methods not covered by job493670.

TEST-informed operating points are descriptive, not registered evidence.
Never fits a model or changes the input checkpoint, result or threshold.
"""
import argparse,hashlib,importlib.util,json,math,os,platform,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cm(y,p,n):return np.bincount(y*n+p,minlength=n*n).reshape(n,n)
def metrics(m):return dict(confusion_matrix=m.tolist(),balanced_accuracy=float(np.mean(np.diag(m)/m.sum(1))),measured_test_fpr=float(1-m[0,0]/m[0].sum()),attack_recall=float(1-m[1:,0].sum()/m[1:].sum()))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--b-root',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--forward-helper',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=list(range(10,15)));p.add_argument('--methods',nargs='+',choices=('R3-L','ar','replay_bs','freq','wacb','icarl','ecbrs','bic'),default=['R3-L','ar','replay_bs','freq','wacb','icarl','ecbrs','bic']);p.add_argument('--sample-per-class',type=int,default=0);a=p.parse_args()
    assert not a.output.exists() and a.sample_per_class>=0
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(a.source.resolve(strict=True)))
    if os.name!='nt' and os.environ.get('NCD_SKLEARN_ROOT'):sys.path.insert(1,os.environ['NCD_SKLEARN_ROOT'])
    import sklearn
    from ncd.section156_models import ReplayGBDTGate
    from ncd.calibrate_aacil_baselines import select_taus
    assert sha(a.forward_helper)=='3fe63c8f404d5217a6c63108e7169132bab7b75751a03146d17038ecc856ca2a'
    spec=importlib.util.spec_from_file_location('same_score_forward',a.forward_helper);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    x,y,xt,yt=[np.load(a.data/(name+'.npy'),mmap_mode='r') for name in ('train_x','train_y','test_x','test_y')]
    binding={name:dict(dtype=v.dtype.str,shape=list(v.shape),sha256=hashlib.sha256(memoryview(v).cast('B')).hexdigest()) for name,v in zip(('x_train','y_train','x_test','y_test'),(x,y,xt,yt))}
    counts=np.bincount(y);prior=np.log(counts/counts.sum()).astype(np.float32);n=len(counts);assert n==5 and np.all(prior[0]>prior[1:])
    rows=np.arange(len(yt))
    if a.sample_per_class:rows=np.sort(np.concatenate([np.flatnonzero(yt==c)[:a.sample_per_class] for c in range(n)]))
    truth=yt[rows];a.output.mkdir(parents=True);failures=[];completed=0
    def mlp(v,w):return F.linear(helper.mlp_features(v,w),w['head.weight'],w['head.bias'])
    for method in a.methods:
        for seed in a.seeds:
            try:
                start=time.perf_counter();root=a.b_root/('confirmation/nsl-kdd/R3-L' if method=='R3-L' else 'nsl-kdd/'+method)/f'seed{seed}'
                cell=root/'cell' if (root/'cell/result.json').exists() else root;r=read(cell/'result.json');i=r['identity'];cp=r['checkpoints'][-1]
                assert r['completed'] and i['data_binding']==binding and i['seed']==seed and cp['seen_classes']==list(range(n))
                path=cell/cp['checkpoint_file'];assert sha(path)==cp['checkpoint_sha256'];state=torch.load(path,map_location='cpu',weights_only=True);assert state['identity']==i
                hold=np.random.default_rng(seed*100003+17).permutation(np.flatnonzero(y==0))[:max(5,math.ceil(int(counts[0])*.1))]
                if method=='R3-L':expected=cp['calibration']['0.01']['row_ids_sha256']
                else:expected=i.get('section26_holdout',i.get('holdout'))['0']['validation_indices_sha256']
                assert hashlib.sha256(hold.tobytes()).hexdigest()==expected
                if a.sample_per_class:hold=hold[:a.sample_per_class]
                if method=='R3-L':
                    gate=ReplayGBDTGate(n);gate.load_state_dict(state['gate_state'],strict=True);gate.eval()
                    assert state['gate_state']['_extra_state']['estimator_pickle']==state['family_state']['joint._extra_state']['estimator_pickle']
                    forward=lambda v:gate.joint_logits(v)
                elif method=='ar':
                    attacks=state['attack_class_ids'];assert attacks==list(range(1,n))
                    fw={k.removeprefix('student.'):v for k,v in state['family_model_state'].items() if k.startswith('student.')}
                    def forward(v):
                        g=mlp(v,state['gate_model_state']);f=mlp(v,fw)
                        return torch.cat((g,f),1)
                else:forward=helper.feature_logits(state,'cer-bic-common' if method=='bic' else method,list(range(n)),int(i['effective_config']['batch_size']))
                def logits(array,ids):
                    out=[]
                    with torch.inference_mode():
                        for off in range(0,len(ids),512):out.append(forward(torch.from_numpy(np.array(array[ids[off:off+512]],dtype=np.float32,copy=True))).numpy())
                    value=np.concatenate(out);assert np.isfinite(value).all();return value
                h=logits(x,hold);t=logits(xt,rows)
                gate_rule=method in ('R3-L','ar')
                if gate_rule:
                    if method=='ar':hs=h[:,1]-h[:,0];ts=t[:,1]-t[:,0];family=t[:,2:].argmax(1)+1;default=np.where(torch.from_numpy(t[:,:2]).softmax(1)[:,1].numpy()>.5,family,0)
                    else:hs=(torch.logsumexp(torch.from_numpy(h[:,1:]),1)-torch.from_numpy(h[:,0])).numpy();ts=(torch.logsumexp(torch.from_numpy(t[:,1:]),1)-torch.from_numpy(t[:,0])).numpy();family=t[:,1:].argmax(1)+1;default=t.argmax(1)
                    predict=lambda tau:np.where(ts>tau,family,0)
                    original=cp['operating_points'];thresholds={k:v['threshold'] for k,v in cp['calibration'].items()};selections={}
                    for alpha in (.02,.05):
                        benign=np.sort(ts[truth==0]);allowed=math.floor(alpha*len(benign));selections[str(alpha)]=dict(total_tau=float(benign[len(benign)-allowed-1]),maximum_false_positives=allowed)
                    definition='Original gate margin and unchanged family argmax; strict > threshold'
                else:
                    original_stage=(cp if 'operating_points' in cp else r.get('calibrated_stages',[None])[-1])
                    if original_stage is None:original_stage=read(root/'calibrated_stages.json')[-1]
                    original=original_stage['operating_points'];cal=original_stage.get('normal_holdout_calibration',original_stage.get('calibration'))
                    thresholds={k:v['total_tau'] for k,v in cal['operating_points'].items()}
                    hs=np.max((h[:,1:].astype(float)-h[:,0,None].astype(float))/(prior[0]-prior[1:]).astype(float),axis=1);ts=np.max((t[:,1:].astype(float)-t[:,0,None].astype(float))/(prior[0]-prior[1:]).astype(float),axis=1)
                    predict=lambda tau:(torch.from_numpy(t)+float(tau)*torch.from_numpy(prior)).argmax(1).numpy();default=t.argmax(1)
                    _,selected,_,_=select_taus(torch.from_numpy(t[truth==0]),torch.from_numpy(prior),0);selections={k:v for k,v in selected.items() if k in ('0.02','0.05')}
                    definition='Original float32 prior-shift decision; descriptive score is ideal crossing tau, not comparable absolute units across gate/prior methods'
                old_default=cp.get('default_decision',{}).get('confusion_matrix')
                if old_default is None:
                    defaults=r.get('default_stages')
                    if defaults is None:defaults=read(root/'default_stages.json')
                    old_default=defaults[-1].get('metrics',defaults[-1])['confusion_matrix']
                discrepancies=[]
                if not a.sample_per_class:
                    if cm(truth,default,n).tolist()!=old_default:discrepancies.append('default')
                    for k,tau in thresholds.items():
                        if cm(truth,predict(tau),n).tolist()!=original[k]['confusion_matrix']:discrepancies.append(k)
                    assert not discrepancies,discrepancies
                points={}
                for k,selection in selections.items():
                    m=cm(truth,predict(selection['total_tau']),n);assert int(m[0].sum()-m[0,0])<=selection['maximum_false_positives'];points[k]=dict(**metrics(m),threshold=selection['total_tau'],maximum_false_positives=selection['maximum_false_positives'],test_label_informed=True,registered_result=False)
                probs=[0,.01,.05,.25,.5,.75,.9,.95,.98,.99,.995,1.];benign=ts[truth==0];grid=np.unique(np.concatenate((hs,benign)));ks=float(np.max(np.abs(np.searchsorted(np.sort(hs),grid,side='right')/len(hs)-np.searchsorted(np.sort(benign),grid,side='right')/len(benign))))
                record=dict(method=method,seed=seed,software_only=bool(a.sample_per_class),source_result_sha256=sha(cell/'result.json'),checkpoint_sha256=cp['checkpoint_sha256'],source_git_commit=i['git_commit'],effective_config=i['effective_config'],diagnostic_source_sha256=sha(Path(__file__)),forward_helper_sha256=sha(a.forward_helper),runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__),data_binding=binding,permanent_holdout_indices_sha256=expected,score_definition=definition,original_cm_discrepancies=discrepancies,quantile_probabilities=probs,holdout_quantiles=np.quantile(hs,probs).tolist(),test_benign_quantiles=np.quantile(benign,probs).tolist(),ks_statistic=ks,test_informed_operating_points=points,elapsed_seconds=time.perf_counter()-start,command=sys.argv,log_file='@@OUTPUT_ROOT@@/diagnosis.stdout.log',scope='Posthoc TEST-label-informed diagnosis; unchanged model/original calibration/results. Not registered S1 or causal feature evidence.')
                stem=a.output/f'{method}-seed{seed}';stem.with_suffix('.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8');np.savez_compressed(stem.with_suffix('.npz'),holdout_score=hs,test_score=ts,holdout_row_ids=hold,test_row_ids=rows,test_labels=truth,holdout_logits=h,test_logits=t)
                completed+=1;print(json.dumps(dict(event='NSL_REMAINING_CELL_COMPLETE',method=method,seed=seed,software_only=bool(a.sample_per_class))),flush=True)
            except Exception as e:traceback.print_exc();failures.append(dict(method=method,seed=seed,error=repr(e)))
    print(json.dumps(dict(event='NSL_REMAINING_COMPLETE',completed=completed,failures=failures,software_only=bool(a.sample_per_class))),flush=True);raise SystemExit(bool(failures))
if __name__=='__main__':main()
