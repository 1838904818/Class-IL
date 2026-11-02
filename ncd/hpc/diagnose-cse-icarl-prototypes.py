"""Bounded retained-memory NME geometry diagnostic; no TEST scan, training, or threshold changes."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import torch
import torch.nn.functional as F

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
B=a.base.resolve(strict=True)
assert not a.output.exists(),'Do not overwrite diagnostics'
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
archive=B/'admin/cse-icarl-completed-publication-v3'
rows=[]
for seed in range(10,15):
    r=read(archive/f'icarl-scoped-npy_seed{seed}/result.json')
    root=B/('runs/section178/cic-ids-2018/icarl-scoped-npy/seed10' if seed==10 else f'validation/hpc-20261102T1345/a/icarl/seed{seed}')
    for cp in r['checkpoints']:
        path=root/'cell'/cp['checkpoint_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==cp['checkpoint_sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)
        w=state['model_state']; seen=cp['seen_classes']
        x=torch.cat(state['buffer_x']); y=torch.cat(state['buffer_y']).numpy()
        assert len(x)<=1400 and x.shape[1]==78
        def features(v):
            return F.relu(F.linear(F.relu(F.linear(v,w['features.0.weight'],w['features.0.bias'])),w['features.2.weight'],w['features.2.bias']))
        batch=int(r['identity']['effective_config']['batch_size'])
        with torch.inference_mode():
            z=torch.cat([features(x[j:j+batch]) for j in range(0,len(x),batch)]).numpy()
        means=np.stack([z[y==c].mean(axis=0) for c in seen])
        norms=np.linalg.norm(means,axis=1)
        means=means/(norms[:,None]+1e-12)
        protos=F.normalize(torch.from_numpy(means.astype(np.float32)),dim=1)
        normalized=F.normalize(torch.from_numpy(z),dim=1)
        dist=((normalized[:,None,:]-protos[None,:,:])**2).sum(2)
        pred=np.asarray(seen)[dist.argmin(1).numpy()]
        pair=((protos[:,None,:]-protos[None,:,:])**2).sum(2).numpy()
        exact=[(seen[j],seen[k]) for j in range(len(seen)) for k in range(j+1,len(seen)) if torch.equal(protos[j],protos[k])]
        cm=np.bincount(y*7+pred,minlength=49).reshape(7,7)
        item=dict(seed=seed,task=cp['task_index'],checkpoint_sha256=cp['checkpoint_sha256'],seen_classes=seen,
            memory_rows=len(x),class_mean_norm_before_normalization=norms.tolist(),prototype_squared_distances=pair.tolist(),
            exactly_identical_normalized_prototype_pairs=exact,zero_feature_rows_per_class=[int((np.linalg.norm(z[y==c],axis=1)==0).sum()) for c in seen],
            retained_memory_resubstitution_cm=cm.tolist(),retained_memory_resubstitution_recall=[float(cm[c,c]/(y==c).sum()) for c in seen])
        if 2 in seen:
            j=seen.index(2)
            competitors=[k for k in range(len(seen)) if k!=j]
            k=min(competitors,key=lambda k:pair[j,k])
            item['ddos']=dict(nearest_other_class=seen[k],squared_prototype_distance=float(pair[j,k]),
                memory_self_recall=float(cm[2,2]/(y==2).sum()),memory_predicted_counts=cm[2].tolist(),
                normalized_feature_unique_vectors=int(len(np.unique(normalized.numpy()[y==2],axis=0))))
            print(json.dumps(dict(event='DDOS_BUFFER_GEOMETRY',seed=seed,task=cp['task_index'],**item['ddos'],exact_pairs=exact)),flush=True)
        rows.append(item)
out=dict(scope='Existing retained TRAIN memory only, at most1400 rows per checkpoint; no TEST or holdout inference.',
    runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,platform=platform.platform()),
    interpretation='Resubstitution is optimistic and is not TEST performance. Exact equality/zero tests need no invented collapse cutoff; finite distances alone do not establish adequate separation.',
    causal_limit='No alternate exemplar budget, sub-mode labels, full TRAIN population or matched intervention was tested. Geometry cannot by itself establish why TEST recall is poor or the causal effect of memory capacity.',
    portability_limit='Diagnostic runs on Windows over fixed saved weights and memories; it does not replace Linux result matrices or claim bit-exact cross-runtime equivalence.',
    checkpoints=rows)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(event='PROTOTYPE_DIAGNOSTIC_COMPLETE',checkpoints=len(rows),exact_collapsed_pair_occurrences=sum(len(x['exactly_identical_normalized_prototype_pairs']) for x in rows),training=False,test_inference=False)),flush=True)
