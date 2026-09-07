"""Legacy scorer fidelity using protocol-bound encoder batching and whole probes.

No training, calibration fitting or full-test performance estimation is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))

def compare(actual,saved,atol=1e-4,rtol=1e-5):
    import numpy as np
    if not np.array_equal(actual['class_axis'],saved['class_axis']):raise ValueError('Class axis mismatch')
    errors={}
    for key in ('head_scores','router_z_scores','joint_scores'):
        x,y=actual[key],saved[key]
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid scores')
        e=np.abs(x-y);bad=e>atol+rtol*np.abs(y)
        errors[key]={'max_abs_error':float(e.max()),'mean_abs_error':float(e.mean()),'outside_tolerance_cells':int(bad.sum()),'bit_exact':bool(np.array_equal(x,y))}
    changed=np.flatnonzero(actual['predicted_class_id']!=saved['predicted_class_id'])
    return errors,changed

def canonical_score(model,raw,batch_size):
    import numpy as np
    import torch
    if batch_size<1:raise ValueError('Invalid encoder batch size')
    normalized=((np.asarray(raw,dtype=np.float64)-model.mean)/model.scale).astype(np.float32)
    if not np.isfinite(normalized).all():raise ValueError('Nonfinite normalized features')
    with torch.no_grad():
        model.encoder.eval()
        embedded=[]
        for start in range(0,len(normalized),batch_size):
            x=torch.from_numpy(np.ascontiguousarray(normalized[start:start+batch_size],dtype=np.float32)).to(model.device)
            embedded.append(model.encoder(x).cpu().numpy().astype(np.float32))
        embedding=np.vstack(embedded)
        tensor=torch.from_numpy(np.ascontiguousarray(embedding,dtype=np.float32)).to(model.device)
        seen=[int(x) for x in model.metadata['seen_classes']]
        head=np.empty((len(raw),len(seen)),dtype=np.float32)
        for col,c in enumerate(seen):
            model.heads[c].eval()
            head[:,col]=torch.softmax(model.heads[c](tensor),dim=1)[:,1].cpu().numpy()
        z=model.router.scores(embedding,seen,'cap3000')
        joint=head+np.float32(.5)*z;axis=np.asarray(seen,dtype=np.int64)
        return {'class_axis':axis,'head_scores':head,'router_z_scores':z,'joint_scores':joint,'predicted_class_id':axis[joint.argmax(axis=1)]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('runtime','inputs','data','protocols','registry','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import numpy as np
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if a.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    registry_sha='de43ff1d51f2993741d5a18807f129f1f80db8a97360c0289312d1944a63c58c'
    if sha(a.registry)!=registry_sha:raise ValueError('Registry mismatch')
    registry={l.split(None,1)[1].removeprefix('./'):l.split(None,1)[0] for l in a.registry.read_text().splitlines()}
    protocol_file=a.protocols/'seed_1/protocol.json'
    if sha(protocol_file)!=registry['last_epoch/seed_1/protocol.json']:raise ValueError('Protocol mismatch')
    protocol=load(protocol_file)
    for r in protocol['source']['files']:
        if sha(a.runtime/r['path'])!=r['sha256']:raise ValueError('Original runtime mismatch')
    sys.path.insert(0,str(a.runtime.resolve()))
    from streaming_full.monitoring import load_checkpoint
    from streaming_full.data import array_sha256,canonical_sha256
    from ofra_encoders import verify_ft_transformer_dependency
    upstream=verify_ft_transformer_dependency()
    probe_file=a.inputs/'probe_manifest.json'
    if sha(probe_file)!=registry['last_epoch/seed_1/monitoring_seed_1/probe_manifest.json']:raise ValueError('Probe mismatch')
    probe=load(probe_file)
    if canonical_sha256({k:v for k,v in probe.items() if k!='canonical_sha256'})!=probe['canonical_sha256']:raise ValueError('Probe canonical mismatch')
    mf=a.inputs/'streaming_manifest.json'
    if sha(mf)!=probe['streaming_manifest_sha256']:raise ValueError('Data manifest mismatch')
    manifest=load(mf);shards={};test_hashes={}
    for c in manifest['classes']:
        for j,r in enumerate(c['test']):
            if sha(a.data/r['path'])!=r['sha256']:raise ValueError('Test shard mismatch')
            shards[c['id'],j]=np.load(a.data/r['path'],mmap_mode='r',allow_pickle=False);test_hashes[r['path']]=r['sha256']
    records=probe['official_test']['samples'];rows=[]
    for r in records:
        x=np.asarray(shards[r['class_id'],r['shard_ordinal']][r['local_row']:r['local_row']+1],dtype=np.float32)
        if array_sha256(x)!=r['feature_row_sha256']:raise ValueError('Feature row mismatch')
        rows.append(x)
    all_raw=np.vstack(rows)
    report={'scope':'canonical legacy final-and-intermediate frozen-probe fidelity','source_lf_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
      'registry_sha256':registry_sha,'probe_manifest_sha256':sha(probe_file),'test_shard_hashes':test_hashes,
      'source_files':protocol['source']['files'],'original_environment':protocol['environment'],
      'environment':{'torch':torch.__version__,'numpy':np.__version__,'python':sys.version,'device':a.device,'upstream':upstream},
      'environment_matched':False,'atol':1e-4,'rtol':1e-5,'expected_checkpoint_count':20,'checkpoints':[],
      'batching':'encoder uses each original protocol eval_batch_size; Head and Router consume whole checkpoint probe matrix',
      'calibrator_fitted':False,'new_test_performance_computed':False,
      'limitations':['CPU results are cross-environment checks, not a matched historical A100 runtime.',
       'Even a tolerance pass is not bit-exact equality or new full-test performance evidence.',
       'Repeated checkpoints/seeds do not constitute independent probe rows.']}
    for seed in [1,2,3,4,42]:
        sp=a.protocols/f'seed_{seed}/protocol.json'
        if sha(sp)!=registry[f'last_epoch/seed_{seed}/protocol.json']:raise ValueError('Seed protocol mismatch')
        conf=load(sp)
        if conf['source']!=protocol['source'] or conf['config']['family_checkpoint_selection']!='last':raise ValueError('Source or checkpoint policy mismatch')
        batch=conf['config']['eval_batch_size']
        for cp in range(4):
            base=a.inputs/f'seed_{seed}'
            if cp!=3:base=base/f'checkpoint_{cp:03d}'
            prefix=f'last_epoch/seed_{seed}/monitoring_seed_{seed}/checkpoint_{cp:03d}/'
            hashes={}
            for name in ('checkpoint_manifest.json','inference_state.npz','probe_scores.npz'):
                hashes[name]=sha(base/name)
                if hashes[name]!=registry[prefix+name]:raise ValueError('Checkpoint pin mismatch')
            model=load_checkpoint(base/'checkpoint_manifest.json',device=a.device)
            seen=sorted(c for t in conf['tasks'][:cp+1] for c in t)
            if model.metadata['seed']!=seed or model.metadata['checkpoint']!=cp or model.metadata['seen_classes']!=seen:raise ValueError('Checkpoint identity')
            if model.metadata['probe_manifest_file_sha256']!=sha(probe_file):raise ValueError('Probe binding')
            indices=[i for i,r in enumerate(records) if r['class_id'] in seen]
            saved=np.load(base/'probe_scores.npz',allow_pickle=False)
            ids=np.asarray([records[i]['sample_id_sha256'].encode('ascii') for i in indices],dtype='S64')
            if not np.array_equal(ids,saved['sample_id_sha256']):raise ValueError('Probe row order')
            started=time.monotonic();actual=canonical_score(model,all_raw[indices],batch)
            if a.device=='cuda':torch.cuda.synchronize()
            elapsed=time.monotonic()-started
            errors,changed=compare(actual,saved)
            mismatch=[]
            for i in changed:
                old=np.sort(saved['joint_scores'][i]);new=np.sort(actual['joint_scores'][i])
                mismatch.append({'sample_id_sha256':ids[i].decode(),'saved_class':int(saved['predicted_class_id'][i]),'reconstructed_class':int(actual['predicted_class_id'][i]),'saved_top2_margin':float(old[-1]-old[-2]),'reconstructed_top2_margin':float(new[-1]-new[-2])})
            result={'seed':seed,'checkpoint':cp,'probe_rows':len(indices),'encoder_batch_size':batch,'artifact_sha256':hashes,
              'scores':errors,'mismatches':mismatch,'prediction_mismatch_count':len(changed),
              'saved_prediction_sha256':array_sha256(saved['predicted_class_id']),
              'reconstructed_prediction_sha256':array_sha256(actual['predicted_class_id']),
              'score_sha256':{k:array_sha256(actual[k]) for k in ('head_scores','router_z_scores','joint_scores')},
              'passed':not len(changed) and not any(v['outside_tolerance_cells'] for v in errors.values()),'elapsed_seconds':elapsed}
            saved.close();del model
            report['checkpoints'].append(result)
            report['coverage_complete']=len(report['checkpoints'])==20
            report['all_passed']=report['coverage_complete'] and all(r['passed'] for r in report['checkpoints'])
            a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
            print(json.dumps({'seed':seed,'checkpoint':cp,'passed':result['passed'],'mismatches':len(changed),'elapsed_seconds':elapsed}),flush=True)
    return 0 if report['all_passed'] else 2

if __name__=='__main__':raise SystemExit(main())
