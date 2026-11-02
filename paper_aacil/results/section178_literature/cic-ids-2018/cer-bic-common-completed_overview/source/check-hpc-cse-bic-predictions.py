"""Read-only BiC forward reconstruction with independent tensor primitives."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

def forward(x,state,heads):
    v=x.unsqueeze(1)
    for conv,bn in ((0,3),(4,7)):
        v=F.max_pool1d(F.relu(F.conv1d(v,state[f'encoder.{conv}.weight'],state[f'encoder.{conv}.bias'],padding=1)),2)
        v=F.batch_norm(v,state[f'encoder.{bn}.running_mean'],state[f'encoder.{bn}.running_var'],
            state[f'encoder.{bn}.weight'],state[f'encoder.{bn}.bias'],training=False,eps=1e-5)
    v=F.relu(F.linear(v.flatten(1),state['encoder.9.weight'],state['encoder.9.bias']))
    return torch.cat([state[f'scales.{j}']*F.linear(v,state[f'heads.{j}.weight'],state[f'heads.{j}.bias'])+state[f'offsets.{j}'] for j in range(heads)],dim=1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path)
    p.add_argument('--data',type=Path)
    p.add_argument('--self-test-source',type=Path)
    a=p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if a.self_test_source:
        sys.path.insert(0,str(a.self_test_source.resolve(strict=True)))
        from ncd.cer_bic import FlowBiC
        torch.manual_seed(13)
        model=FlowBiC(78)
        for j,task in enumerate(((0,1),(2,3),(4,5),(6,))):
            model.expand(task)
            with torch.no_grad():
                model.scales[j].fill_(.7+j*.2)
                model.offsets[j].fill_(-.3+j*.1)
            model.eval()
            x=torch.randn(64,78)
            with torch.inference_mode():
                expected=model(x)
                actual=forward(x,model.state_dict(),j+1)
            assert torch.equal(expected,actual),float((expected-actual).abs().max())
        print(json.dumps(dict(event='BIC_FORWARD_COMPONENT_TEST',stages=4,rows=64,feature_count=78,bitwise_equal=True,scientific_result=False)))
        return
    assert a.directory and a.data
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    r=read(a.directory/'result.json')
    i=r['identity']
    assert r['completed'] and not i['smoke'] and i['dataset']=='CSE-CIC-IDS2018'
    assert i['method']=='cer-bic-flow-cnn-common' and i['seed'] in (13,14)
    x=np.load(a.data/'train_x.npy',mmap_mode='r')
    y=np.load(a.data/'train_y.npy')
    xt=np.load(a.data/'test_x.npy',mmap_mode='r')
    yt=np.load(a.data/'test_y.npy')
    counts=np.bincount(y,minlength=len(i['class_order']))
    assert counts.tolist()==i['original_train_counts']
    normal=i['class_order'].index('Normal')
    ids=np.flatnonzero(y==normal)
    holdout=np.random.default_rng(i['seed']*100003+normal*1009+17).permutation(ids)
    holdout=holdout[:max(5,int(np.ceil(len(ids)*.1)))]
    assert hashlib.sha256(holdout.tobytes()).hexdigest()==i['holdout'][str(normal)]['validation_indices_sha256']
    assert len(holdout)*x.shape[1]*x.dtype.itemsize<1024**3
    cache=np.empty((len(holdout),x.shape[1]),dtype=x.dtype)
    order=np.argsort(holdout)
    for offset in range(0,len(order),8192):
        slots=order[offset:offset+8192]
        cache[slots]=x[holdout[slots]]
    failures=[]
    for t,cp in enumerate(r['checkpoints']):
        path=a.directory/cp['checkpoint_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==cp['checkpoint_sha256']
        s=torch.load(path,map_location='cpu',weights_only=True)
        assert s['identity']==i and s['task_index']==t
        seen=cp['seen_classes']
        rows=np.flatnonzero(np.isin(yt,seen))
        def scores(array,index):
            pieces=[]
            with torch.inference_mode():
                for start in range(0,len(index),512):
                    batch=torch.from_numpy(np.array(array[index[start:start+512]],dtype=np.float32,copy=True))
                    pieces.append(forward(batch,s['model_state'],t+1))
            result=torch.cat(pieces)
            assert torch.isfinite(result).all()
            return result
        logits=scores(xt,rows)
        normal_logits=scores(cache,np.arange(len(cache)))
        n=len(i['class_order'])
        def cm(pred):return np.bincount(yt[rows]*n+np.asarray(seen)[pred],minlength=n*n).reshape(n,n).tolist()
        if cm(logits.argmax(1).numpy())!=r['default_stages'][t]['metrics']['confusion_matrix']:
            failures.append(dict(task=t,view='default'))
        prior=torch.from_numpy(np.log(counts/counts.sum()).astype(np.float32)[seen])
        cal=r['calibrated_stages'][t]
        for alpha,point in cal['calibration']['operating_points'].items():
            tau=point['total_tau']
            if cm((logits+tau*prior).argmax(1).numpy())!=cal['operating_points'][alpha]['confusion_matrix']:
                failures.append(dict(task=t,view=alpha))
            for label,value,expected in (('upper',tau,point['calibration_false_positives']),('lower',point['final_bisection_interval'][0],point['calibration_lower_false_positives'])):
                fp=int(((normal_logits+value*prior).argmax(1)!=seen.index(normal)).sum())
                if fp!=expected:failures.append(dict(task=t,view=alpha,bound=label,recorded_fp=expected,recomputed_fp=fp))
        print(json.dumps(dict(event='BIC_PREDICTIONS_CHECKED',seed=i['seed'],task=t,test_rows=len(rows),failures_so_far=len(failures),accepted=False)),flush=True)
    print(json.dumps(dict(event='BIC_PREDICTIONS_COMPLETE',seed=i['seed'],checkpoints=4,test_cm_views=16,calibration_bracket_counts=24,failures=failures,accepted=False)),flush=True)
    raise SystemExit(1 if failures else 0)

if __name__=='__main__':main()
