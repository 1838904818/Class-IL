"""Fixed-embedding numerical diagnostics, not a replacement production scorer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def stable_distances(values,centroids):
    import numpy as np
    x=np.asarray(values,dtype=np.float64);c=np.asarray(centroids,dtype=np.float64)
    if x.ndim!=2 or c.ndim!=2 or x.shape[1]!=c.shape[1] or len(c)==0:raise ValueError('Invalid distance inputs')
    if not np.isfinite(x).all() or not np.isfinite(c).all():raise ValueError('Nonfinite distance inputs')
    out=np.full(len(x),np.inf)
    # Direct subtraction avoids cancellation in ||x||^2 + ||c||^2 - 2*x.c.
    # The diagnostic is labelled separately; production code is never changed.
    for start in range(0,len(c),16):
        delta=x[:,None,:]-c[None,start:start+16,:]
        ds=np.sum(delta*delta,axis=2,dtype=np.float64)
        out=np.minimum(out,ds.min(axis=1))
    return -np.sqrt(out)

def stable_router(values,states,classes):
    import numpy as np
    raw=np.column_stack([stable_distances(values,states[c].centroids) for c in classes])
    return ((raw-raw.mean(axis=1,keepdims=True))/(raw.std(axis=1,keepdims=True)+1e-8)).astype(np.float32)

def difference(a,b):
    import numpy as np
    error=np.abs(a-b)
    return {'max_abs_error':float(error.max()),'mean_abs_error':float(error.mean()),
            'changed_cells':int(np.count_nonzero(a!=b)),
            'outside_fixed_tolerance':int(np.count_nonzero(error>1e-4+1e-5*np.abs(b)))}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('runtime','inputs','data','protocols','registry','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
    import numpy as np
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if sha(a.registry)!='de43ff1d51f2993741d5a18807f129f1f80db8a97360c0289312d1944a63c58c':raise ValueError('Registry mismatch')
    registry={l.split(None,1)[1].removeprefix('./'):l.split(None,1)[0] for l in a.registry.read_text().splitlines()}
    protocol_file=a.protocols/'seed_1/protocol.json'
    if sha(protocol_file)!=registry['last_epoch/seed_1/protocol.json']:raise ValueError('Protocol mismatch')
    protocol=json.loads(protocol_file.read_text())
    for r in protocol['source']['files']:
        if sha(a.runtime/r['path'])!=r['sha256']:raise ValueError('Original source mismatch')
    sys.path.insert(0,str(a.runtime.resolve()))
    from streaming_full.monitoring import load_checkpoint
    from streaming_full.data import array_sha256,canonical_sha256
    from ofra_encoders import verify_ft_transformer_dependency
    upstream=verify_ft_transformer_dependency()
    probe_file=a.inputs/'probe_manifest.json'
    if sha(probe_file)!=registry['last_epoch/seed_1/monitoring_seed_1/probe_manifest.json']:raise ValueError('Probe mismatch')
    probe=json.loads(probe_file.read_text())
    if canonical_sha256({k:v for k,v in probe.items() if k!='canonical_sha256'})!=probe['canonical_sha256']:raise ValueError('Probe canonical mismatch')
    if sha(a.inputs/'streaming_manifest.json')!=probe['streaming_manifest_sha256']:raise ValueError('Data manifest mismatch')
    manifest=json.loads((a.inputs/'streaming_manifest.json').read_text());arrays={}
    for c in manifest['classes']:
        for j,r in enumerate(c['test']):
            if sha(a.data/r['path'])!=r['sha256']:raise ValueError('Test shard mismatch')
            arrays[c['id'],j]=np.load(a.data/r['path'],mmap_mode='r',allow_pickle=False)
    records=probe['official_test']['samples'];rows=[]
    for r in records:
        x=np.asarray(arrays[r['class_id'],r['shard_ordinal']][r['local_row']:r['local_row']+1],dtype=np.float32)
        if array_sha256(x)!=r['feature_row_sha256']:raise ValueError('Feature row mismatch')
        rows.append(x)
    raw=np.vstack(rows);classes=list(range(8));axis=np.asarray(classes)
    report={'scope':'frozen CPU embedding router arithmetic isolation; diagnostic only',
      'source_lf_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
      'protocol_sha256':sha(protocol_file),'probe_sha256':sha(probe_file),
      'original_runtime_files':protocol['source']['files'],
      'environment':{'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','threads':2,'upstream':upstream},
      'probe_rows':len(raw),'encoder_batch_size':32,'alternate_encoder_batch_size':len(raw),'router_batch_sizes':[1,32,128,len(raw)],
      'new_candidate_promoted':False,'calibration_fitted':False,'seed_results':[],
      'limitations':['CPU embedding differs from original CUDA; no saved original embedding is available in this checkpoint.',
       'High-precision direct subtraction is a diagnostic, not the registered production formula.',
       'Repeated seeds share the same probe rows; no accuracy, forgetting or explanation-governance claim follows.',
       'Fixed-embedding isolation concerns these probes and final checkpoints only.']}
    for seed in [1,2,3,4,42]:
        start=time.monotonic();base=a.inputs/f'seed_{seed}';prefix=f'last_epoch/seed_{seed}/monitoring_seed_{seed}/checkpoint_003/'
        for name in ('checkpoint_manifest.json','inference_state.npz','probe_scores.npz'):
            if sha(base/name)!=registry[prefix+name]:raise ValueError('Protected artifact mismatch')
        model=load_checkpoint(base/'checkpoint_manifest.json',device='cpu')
        if model.metadata['seed']!=seed or model.metadata['seen_classes']!=classes or model.metadata['probe_manifest_file_sha256']!=sha(probe_file):raise ValueError('Checkpoint identity')
        original=np.load(base/'probe_scores.npz',allow_pickle=False)
        if not np.array_equal(original['sample_id_sha256'],np.asarray([r['sample_id_sha256'].encode('ascii') for r in records],dtype='S64')):raise ValueError('Probe order')
        normalized=((raw.astype(np.float64)-model.mean)/model.scale).astype(np.float32)
        with torch.inference_mode():
            embedding=np.vstack([model.encoder(torch.from_numpy(normalized[i:i+32])).numpy() for i in range(0,len(raw),32)])
            tensor=torch.from_numpy(embedding)
            head=np.column_stack([torch.softmax(model.heads[c](tensor),dim=1)[:,1].numpy() for c in classes])
        routers={}
        for batch in report['router_batch_sizes']:
            routers[batch]=np.vstack([model.router.scores(embedding[i:i+batch],classes,'cap3000') for i in range(0,len(raw),batch)])
        fixed_reference=routers[len(raw)];reference_predictions=axis[(head+.5*fixed_reference).argmax(axis=1)]
        batch_checks=[]
        for batch,value in routers.items():
            predictions=axis[(head+.5*value).argmax(axis=1)]
            batch_checks.append({'batch':batch,'router_vs_same_embedding_full_batch':difference(value,fixed_reference),
              'predictions_vs_same_embedding_full_batch':int(np.count_nonzero(predictions!=reference_predictions)),
              'predictions_vs_saved_cuda':int(np.count_nonzero(predictions!=original['predicted_class_id']))})
        stable=stable_router(embedding,model.router.cap,classes)
        stable_chunks=np.vstack([stable_router(embedding[i:i+32],model.router.cap,classes) for i in range(0,len(raw),32)])
        stable_predictions=axis[(head+.5*stable).argmax(axis=1)]
        with torch.inference_mode():
            alternate_embedding=model.encoder(torch.from_numpy(normalized)).numpy()
            alternate_tensor=torch.from_numpy(alternate_embedding)
            alternate_head=np.column_stack([torch.softmax(model.heads[c](alternate_tensor),dim=1)[:,1].numpy() for c in classes])
        alternate_router=model.router.scores(alternate_embedding,classes,'cap3000')
        alternate_stable=stable_router(alternate_embedding,model.router.cap,classes)
        crossed=[]
        for head_name,h in [('encoder32',head),('encoder_full',alternate_head)]:
            for router_name,z in [('original_encoder32',fixed_reference),('original_encoder_full',alternate_router),('direct64_encoder32',stable),('direct64_encoder_full',alternate_stable)]:
                pred=axis[(h+.5*z).argmax(axis=1)]
                crossed.append({'head':head_name,'router':router_name,
                  'prediction_mismatches_vs_saved_cuda':int(np.count_nonzero(pred!=original['predicted_class_id']))})
        original_raw=model.router._distances(embedding,model.router.cap,classes)
        high_raw=np.column_stack([stable_distances(embedding,model.router.cap[c].centroids) for c in classes])
        std=original_raw.std(axis=1)
        result={'seed':seed,'embedding_sha256':array_sha256(embedding),
          'checkpoint_manifest_sha256':sha(base/'checkpoint_manifest.json'),
          'head_vs_saved':difference(head,original['head_scores']),
          'alternate_embedding_sha256':array_sha256(alternate_embedding),
          'embedding32_vs_full':difference(embedding,alternate_embedding),
          'head_embedding32_vs_full':difference(head,alternate_head),
          'router_embedding32_vs_full':difference(fixed_reference,alternate_router),
          'direct64_router_embedding32_vs_full':difference(stable,alternate_stable),
          'crossed_component_prediction_checks':crossed,
          'batch_checks':batch_checks,'original_raw_std_quantiles':np.quantile(std,[0,.01,.5,.99,1]).tolist(),
          'original_raw_vs_direct64':difference(original_raw,high_raw),
          'original_router_vs_direct64':difference(fixed_reference,stable),
          'direct64_router_batch32_vs_full':difference(stable_chunks,stable),
          'direct64_predictions_vs_original_full_batch':int(np.count_nonzero(stable_predictions!=reference_predictions)),
          'direct64_predictions_vs_saved_cuda':int(np.count_nonzero(stable_predictions!=original['predicted_class_id'])),
          'elapsed_seconds':time.monotonic()-start}
        original.close();del model
        report['seed_results'].append(result)
        a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
        print(json.dumps(result),flush=True)

if __name__=='__main__':main()
