"""Read-only section344 calibration robustness check on saved checkpoints.

This is not training, registered-S1 replacement, or original result mutation.
Inputs use root/dataset/method/seedN/{result.json,checkpoints/*}.
"""
import argparse,hashlib,json,math,platform,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def mlp_features(x,w):
    return F.relu(F.linear(F.relu(F.linear(x,w['features.0.weight'],w['features.0.bias'])),w['features.2.weight'],w['features.2.bias']))

def bic_forward(x,w,heads):
    v=x.unsqueeze(1)
    for conv,bn in ((0,3),(4,7)):
        v=F.max_pool1d(F.relu(F.conv1d(v,w[f'encoder.{conv}.weight'],w[f'encoder.{conv}.bias'],padding=1)),2)
        v=F.batch_norm(v,w[f'encoder.{bn}.running_mean'],w[f'encoder.{bn}.running_var'],w[f'encoder.{bn}.weight'],w[f'encoder.{bn}.bias'],training=False,eps=1e-5)
    v=F.relu(F.linear(v.flatten(1),w['encoder.9.weight'],w['encoder.9.bias']))
    return torch.cat([w[f'scales.{j}']*F.linear(v,w[f'heads.{j}.weight'],w[f'heads.{j}.bias'])+w[f'offsets.{j}'] for j in range(heads)],1)

def metrics(y,p,n,seen):
    cm=np.bincount(y*n+p,minlength=n*n).reshape(n,n)
    supports=cm.sum(1);recall=np.divide(cm.diagonal(),supports,out=np.zeros(n,dtype=float),where=supports>0)
    return dict(confusion_matrix=cm.tolist(),balanced_accuracy=float(recall[seen].mean()),
                attack_recall=float(1-cm[1:,0].sum()/supports[1:].sum()),
                measured_test_fpr=float(1-cm[0,0]/supports[0]),per_class_recall=recall.tolist())

def threshold(scores,alpha):
    # Strict >, at most floor(alpha*n) false positives, including ties.
    k=math.floor(alpha*len(scores))
    tau=float(np.sort(scores)[len(scores)-k-1])
    fp=int((scores>tau).sum())
    assert fp<=k
    return dict(threshold=tau,holdout_false_positives=fp,holdout_count=len(scores),maximum_false_positives=k,comparison='strict >')

def feature_logits(state,method,seen,batch):
    w=state['model_state'];prototypes=None
    if method=='icarl':
        bx=torch.cat(state['buffer_x']);by=torch.cat(state['buffer_y']).numpy()
        with torch.inference_mode():
            embedded=torch.cat([mlp_features(bx[j:j+batch],w) for j in range(0,len(bx),batch)]).numpy()
        means=[]
        for label in seen:
            mean=embedded[by==label].mean(0);mean/=np.linalg.norm(mean)+1e-12
            means.append(torch.from_numpy(mean.astype(np.float32)))
        prototypes=F.normalize(torch.stack(means),dim=1)
    def forward(v):
        if method=='cer-bic-common':return bic_forward(v,w,state['task_index']+1)
        f=mlp_features(v,w)
        if prototypes is not None:
            f=F.normalize(f,dim=1)
            return -((f[:,None,:]-prototypes[None,:,:])**2).sum(2)
        return F.linear(f,w['head.weight'],w['head.bias'])[:,seen]
    return forward

def evaluate_cell(root,data,output,method,seed,sample):
    r=json.loads((root/'result.json').read_text(encoding='utf-8'));i=r['identity']
    assert r['completed'] and i['seed']==seed and i['class_order'][0]=='Normal'
    n=len(i['class_order']);x,y,xt,yt=data
    normal=np.flatnonzero(y==0)
    holdout=np.random.default_rng(seed*100003+17).permutation(normal)[:max(5,math.ceil(len(normal)*.1))]
    binding=i.get('section26_holdout',i.get('holdout'))
    assert hashlib.sha256(holdout.tobytes()).hexdigest()==binding['0']['validation_indices_sha256']
    if sample:holdout=holdout[:sample]
    assert len(holdout)*x.shape[1]*x.dtype.itemsize<1024**3
    hx=np.empty((len(holdout),x.shape[1]),dtype=x.dtype)
    order=np.argsort(holdout)
    for off in range(0,len(order),8192):
        slots=order[off:off+8192];hx[slots]=x[holdout[slots]]
    stages=[];start=time.perf_counter()
    for cp in r['checkpoints']:
        path=root/cp['checkpoint_file'];assert digest(path)==cp['checkpoint_sha256']
        s=torch.load(path,map_location='cpu',weights_only=True)
        assert s['identity']==i and s['task_index']==cp['task_index']
        seen=cp['seen_classes'];assert seen==list(range(len(seen)))
        batch=512 if method=='cer-bic-common' else int(i['effective_config']['eval_batch_size'])
        forward=feature_logits(s,method,seen,int(i['effective_config']['batch_size']))
        rows=np.flatnonzero(np.isin(yt,seen))
        if sample:rows=np.concatenate([rows[yt[rows]==c][:sample] for c in seen])
        def score(array,index):
            margins=[];families=[]
            with torch.inference_mode():
                for off in range(0,len(index),batch):
                    v=torch.from_numpy(np.array(array[index[off:off+batch]],dtype=np.float32,copy=True))
                    z=forward(v);assert z.shape[1]==len(seen) and torch.isfinite(z).all()
                    margins.append((torch.logsumexp(z[:,1:],1)-z[:,0]).numpy())
                    families.append((z[:,1:].argmax(1)+1).numpy())
            return np.concatenate(margins),np.concatenate(families)
        hs,_=score(hx,np.arange(len(hx)));ts,family=score(xt,rows)
        views={}
        for alpha in (.01,.02,.05):
            point=threshold(hs,alpha);pred=np.where(ts>point['threshold'],family,0)
            views[str(alpha)]=dict(calibration=point,metrics=metrics(yt[rows],pred,n,seen))
        stages.append(dict(task_index=cp['task_index'],seen_classes=seen,checkpoint_sha256=cp['checkpoint_sha256'],views=views))
        print(json.dumps(dict(event='SAME_SCORE_STAGE',method=method,seed=seed,task=cp['task_index'],rows=len(rows),software_only=bool(sample))),flush=True)
    final={}
    for view in ('0.01','0.02','0.05'):
        signed=[]
        for c in range(n):
            recalls=[t['views'][view]['metrics']['per_class_recall'][c] for t in stages if c in t['seen_classes']]
            if len(recalls)>1:signed.append(max(recalls[:-1])-recalls[-1])
        final[view]=dict(stages[-1]['views'][view]['metrics'],mean_signed_best_prior_forgetting=float(np.mean(signed)),forgetting_eligible_classes=len(signed))
    record=dict(dataset=i['dataset'],method=method,seed=seed,class_order=i['class_order'],tasks=i['tasks'],software_only=bool(sample),registered_primary_changed=False,
        score='logsumexp(seen attack logits)-Normal logit; NME negative squared distances for iCaRL',
        rule='attack iff score>holdout threshold; family=argmax attack logits',stages=stages,final=final,
        permanent_normal_holdout_indices_sha256=binding['0']['validation_indices_sha256'],
        reference_data_metadata_sha256=i['data_metadata_sha256'],reference_data_binding=i['data_binding'],
        original_prior_shift_stages=r.get('calibrated_stages') or json.loads((root/'calibrated_stages.json').read_text(encoding='utf-8')),source_result_sha256=digest(root/'result.json'),
        source_checkpoint_identity_commit=i['git_commit'],diagnostic_source_sha256=digest(Path(__file__)),
        runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,platform=platform.system()),
        elapsed_seconds=time.perf_counter()-start,command=['check-section178-same-score.py','--inputs','@@INPUTS@@','--data','@@DATA@@','--output','@@OUTPUT@@'],
        log_file='@@OUTPUT@@/diagnosis.stdout.log',scope='section344 post-hoc robustness, not registered S1 replacement; no training or original result mutation')
    output.mkdir(parents=True,exist_ok=True);dest=output/'same_score_calibration.json';assert not dest.exists()
    dest.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path);p.add_argument('--data',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--datasets',nargs='+',default=['unsw-nb15','cic-ids-2017','cic-ids-2018']);p.add_argument('--methods',nargs='+',default=['derpp','icarl','ecbrs-paper','cer-bic-common'])
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(10,15)));p.add_argument('--sample-per-class',type=int,default=0);p.add_argument('--self-test',action='store_true')
    a=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if a.self_test:
        for values in (np.arange(100,dtype=np.float32),np.zeros(100,dtype=np.float32)):
            for alpha in (.01,.02,.05):assert threshold(values,alpha)['holdout_false_positives']<=math.floor(len(values)*alpha)
        got=metrics(np.array([0,0,1,1]),np.array([0,1,1,0]),2,[0,1]);assert got['balanced_accuracy']==.5 and got['attack_recall']==.5
        print('SAME_SCORE_SOFTWARE_FIXTURES_OK');return
    failures=[];completed=0
    for ds in a.datasets:
        dp=a.data/ds;data=tuple(np.load(dp/(k+'.npy'),mmap_mode='r') for k in ('train_x','train_y','test_x','test_y'))
        binding={k:dict(dtype=v.dtype.str,shape=list(v.shape),sha256=hashlib.sha256(memoryview(v).cast('B')).hexdigest()) for k,v in zip(('x_train','y_train','x_test','y_test'),data)}
        for method in a.methods:
            for seed in a.seeds:
                try:
                    root=a.inputs/ds/method/f'seed{seed}';r=json.loads((root/'result.json').read_text(encoding='utf-8'))
                    assert r['identity']['data_binding']==binding
                    evaluate_cell(root,data,a.output/ds/method/f'seed{seed}',method,seed,a.sample_per_class);completed+=1
                except Exception as e:
                    import traceback
                    traceback.print_exc();failures.append(dict(dataset=ds,method=method,seed=seed,error=repr(e)))
    print(json.dumps(dict(event='SAME_SCORE_COMPLETE',completed=completed,failures=failures,software_only=bool(a.sample_per_class))),flush=True)
    raise SystemExit(bool(failures))
if __name__=='__main__':main()
