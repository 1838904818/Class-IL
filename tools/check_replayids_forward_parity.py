"""Local CPU-only parity check on frozen probes; never fits a model or calibrator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','inputs','data','protocols','registry','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3,4,42])
    p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args()
    if a.batch_size<1:raise ValueError('Positive batch size required')
    # Fixed before any score comparison; failure is recorded, never relaxed here.
    atol,rtol=1e-4,1e-5
    os.environ['OMP_NUM_THREADS']='2'
    os.environ['OPENBLAS_NUM_THREADS']='2'
    os.environ['MKL_NUM_THREADS']='2'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    registry={line.split(None,1)[1].removeprefix('./'):line.split(None,1)[0] for line in a.registry.read_text().splitlines()}
    if sha(a.registry)!='de43ff1d51f2993741d5a18807f129f1f80db8a97360c0289312d1944a63c58c':raise ValueError('Registry pin mismatch')
    primary_protocol_path=a.protocols/'seed_1/protocol.json'
    if sha(primary_protocol_path)!=registry['last_epoch/seed_1/protocol.json']:raise ValueError('Primary protocol hash mismatch')
    protocol=json.loads(primary_protocol_path.read_text())
    for rec in protocol['source']['files']:
        if sha(a.runtime/rec['path'])!=rec['sha256']:raise ValueError('Original runtime mismatch: '+rec['path'])
    sys.path.insert(0,str(a.runtime.resolve()))
    from streaming_full.monitoring import load_checkpoint
    from streaming_full.data import array_sha256,canonical_sha256
    from ofra_encoders import verify_ft_transformer_dependency
    upstream=verify_ft_transformer_dependency()
    probe_path=a.inputs/'probe_manifest.json'
    if sha(probe_path)!=registry['last_epoch/seed_1/monitoring_seed_1/probe_manifest.json']:raise ValueError('Probe pin mismatch')
    probe=json.loads(probe_path.read_text())
    if canonical_sha256({k:v for k,v in probe.items() if k!='canonical_sha256'})!=probe['canonical_sha256']:raise ValueError('Probe canonical mismatch')
    manifest_path=a.inputs/'streaming_manifest.json'
    if sha(manifest_path)!=probe['streaming_manifest_sha256']:raise ValueError('Data manifest mismatch')
    manifest=json.loads(manifest_path.read_text())
    arrays={};test_hashes={}
    for c in manifest['classes']:
        for j,rec in enumerate(c['test']):
            path=a.data/rec['path']
            if sha(path)!=rec['sha256']:raise ValueError('Local test shard mismatch')
            arrays[c['id'],j]=np.load(path,mmap_mode='r',allow_pickle=False)
            test_hashes[rec['path']]=rec['sha256']
    records=probe['official_test']['samples'];rows=[]
    for rec in records:
        key=rec['class_id'],rec['shard_ordinal']
        parent=manifest['classes'][key[0]]['test'][key[1]]
        if parent['sha256']!=rec['parent_shard_sha256']:raise ValueError('Parent shard binding')
        raw=np.asarray(arrays[key][rec['local_row']:rec['local_row']+1],dtype=np.float32)
        if array_sha256(raw)!=rec['feature_row_sha256']:raise ValueError('Probe row mismatch')
        rows.append(raw)
    raw=np.vstack(rows)
    report={'scope':'final-checkpoint frozen-probe CPU reconstruction only',
      'audit_source_lf_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
      'atol':atol,'rtol':rtol,'batch_size':a.batch_size,'cpu_threads':2,
      'environment':{'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','upstream':upstream},
      'original_environment':protocol['environment'],
      'primary_protocol_sha256':sha(primary_protocol_path),
      'source_files':protocol['source']['files'],'test_shard_hashes':test_hashes,
      'probe_manifest_sha256':sha(probe_path),'probe_row_count':len(raw),'seeds':[],
      'calibrator_fitted':False,'new_test_metrics_computed':False,
      'limitations':['Numerical tolerance is not bit-exact equality.',
       'CPU and training CUDA environments differ.',
       'Only checkpoint 003 and pre-existing probes are checked, not full test performance or forgetting.',
       'Probe labels are used only to verify frozen identity, never to tune parameters.']}
    for seed in a.seeds:
        seed_protocol_path=a.protocols/f'seed_{seed}/protocol.json'
        if sha(seed_protocol_path)!=registry[f'last_epoch/seed_{seed}/protocol.json']:raise ValueError('Seed protocol hash mismatch')
        seed_protocol=json.loads(seed_protocol_path.read_text())
        if seed_protocol['source']!=protocol['source'] or seed_protocol['config']['family_checkpoint_selection']!='last':raise ValueError('Seed source or policy mismatch')
        base=a.inputs/f'seed_{seed}';prefix=f'last_epoch/seed_{seed}/monitoring_seed_{seed}/checkpoint_003/'
        for name in ('checkpoint_manifest.json','inference_state.npz','probe_scores.npz'):
            if sha(base/name)!=registry[prefix+name]:raise ValueError('Protected input hash mismatch')
        cp=json.loads((base/'checkpoint_manifest.json').read_text())
        if cp['seed']!=seed or cp['checkpoint']!=3 or cp['seen_classes']!=list(range(8)):raise ValueError('Checkpoint identity')
        if cp['probe_manifest_file_sha256']!=sha(probe_path):raise ValueError('Probe checkpoint binding')
        original=np.load(base/'probe_scores.npz',allow_pickle=False)
        ids=np.asarray([r['sample_id_sha256'].encode('ascii') for r in records],dtype='S64')
        if not np.array_equal(ids,original['sample_id_sha256']):raise ValueError('Sample order mismatch')
        if not np.array_equal([r['class_id'] for r in records],original['true_class_id']):raise ValueError('Sample labels mismatch')
        started=time.monotonic();model=load_checkpoint(base/'checkpoint_manifest.json',device='cpu')
        outputs=[]
        for start in range(0,len(raw),a.batch_size):
            outputs.append(model.score(raw[start:start+a.batch_size]))
        measured={k:np.concatenate([x[k] for x in outputs]) for k in ('head_scores','router_z_scores','joint_scores','predicted_class_id')}
        del model
        scores={}
        for key in ('head_scores','router_z_scores','joint_scores'):
            actual=measured[key];expected=original[key]
            error=np.abs(actual-expected)
            scores[key]={'max_abs_error':float(error.max()),'mean_abs_error':float(error.mean()),
              'allclose':bool(np.allclose(actual,expected,atol=atol,rtol=rtol)),
              'bit_exact':bool(np.array_equal(actual,expected)),
              'outside_tolerance_cells':int((error>atol+rtol*np.abs(expected)).sum())}
        mismatch_indices=np.flatnonzero(measured['predicted_class_id']!=original['predicted_class_id'])
        mismatches=len(mismatch_indices)
        mismatch_records=[]
        for i in mismatch_indices:
            saved=np.sort(original['joint_scores'][i])
            current=np.sort(measured['joint_scores'][i])
            mismatch_records.append({'sample_id_sha256':records[i]['sample_id_sha256'],
              'saved_prediction':int(original['predicted_class_id'][i]),
              'reconstructed_prediction':int(measured['predicted_class_id'][i]),
              'saved_top2_margin':float(saved[-1]-saved[-2]),
              'reconstructed_top2_margin':float(current[-1]-current[-2])})
        result={'seed':seed,'checkpoint':3,'scores':scores,'prediction_mismatches':mismatches,
          'prediction_mismatch_records':mismatch_records,
          'elapsed_seconds':time.monotonic()-started,'passed':mismatches==0 and all(x['allclose'] for x in scores.values()),
          'checkpoint_manifest_sha256':sha(base/'checkpoint_manifest.json'),'probe_scores_sha256':sha(base/'probe_scores.npz'),
          'inference_state_sha256':sha(base/'inference_state.npz')}
        original.close();report['seeds'].append(result)
        report['all_passed']=all(x['passed'] for x in report['seeds'])
        a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
        print(json.dumps(result),flush=True)
    return 0 if report['all_passed'] else 2

if __name__=='__main__':raise SystemExit(main())
