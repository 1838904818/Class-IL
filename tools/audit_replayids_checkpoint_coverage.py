"""Check complete primary checkpoint coverage locally, without running inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

REGISTRY_SHA='de43ff1d51f2993741d5a18807f129f1f80db8a97360c0289312d1944a63c58c'
PROBE_SHA='f6c820382926f9d30ea20cd474ed9a22a378551718edfe726e2d6b6687762c49'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))

def validate_scores(scores,cp,records):
    seen=cp['seen_classes'];axis=np.asarray(seen,dtype=np.int64)
    expected=[r for r in records if r['class_id'] in seen]
    n=len(expected)
    if cp['score_rows']!=n or not np.array_equal(scores['class_axis'],axis):raise ValueError('Class axis/row count mismatch')
    for key in ('head_scores','router_z_scores','joint_scores'):
        if scores[key].shape!=(n,len(seen)) or not np.isfinite(scores[key]).all():raise ValueError('Score shape or finiteness mismatch')
    mapping={'true_class_id':'class_id','shard_ordinal':'shard_ordinal','local_row':'local_row'}
    for key,field in mapping.items():
        if not np.array_equal(scores[key],np.asarray([r[field] for r in expected],dtype=np.int64)):raise ValueError('Probe coordinates mismatch')
    ids=np.asarray([r['sample_id_sha256'].encode('ascii') for r in expected],dtype='S64')
    if not np.array_equal(scores['sample_id_sha256'],ids):raise ValueError('Probe identities mismatch')
    joint=scores['head_scores']+np.float32(.5)*scores['router_z_scores']
    if not np.array_equal(joint,scores['joint_scores']):raise ValueError('Stored joint formula mismatch')
    if not np.array_equal(axis[joint.argmax(axis=1)],scores['predicted_class_id']):raise ValueError('Stored argmax mismatch')
    return n

def audit(root,registry_path):
    if sha(registry_path)!=REGISTRY_SHA:raise ValueError('Registry pin mismatch')
    registry={l.split(None,1)[1].removeprefix('./'):l.split(None,1)[0] for l in registry_path.read_text().splitlines()}
    probe_path=root/'probe_manifest.json'
    if sha(probe_path)!=PROBE_SHA:raise ValueError('Probe pin mismatch')
    probe=load(probe_path)
    if canonical({k:v for k,v in probe.items() if k!='canonical_sha256'})!=probe['canonical_sha256']:raise ValueError('Probe canonical mismatch')
    rows=[]
    for seed in [1,2,3,4,42]:
        for checkpoint in range(4):
            base=root/f'seed_{seed}'
            if checkpoint!=3:base=base/f'checkpoint_{checkpoint:03d}'
            prefix=f'last_epoch/seed_{seed}/monitoring_seed_{seed}/checkpoint_{checkpoint:03d}/'
            hashes={}
            for name in ('checkpoint_manifest.json','inference_state.npz','probe_scores.npz'):
                digest=sha(base/name)
                if digest!=registry[prefix+name]:raise ValueError('Protected checkpoint mismatch')
                hashes[name]=digest
            cp=load(base/'checkpoint_manifest.json')
            if cp['seed']!=seed or cp['checkpoint']!=checkpoint or cp['seen_classes']!=list(range(2*(checkpoint+1))):raise ValueError('Checkpoint identity/seen classes')
            if canonical({k:v for k,v in cp.items() if k!='canonical_sha256'})!=cp['canonical_sha256']:raise ValueError('Checkpoint canonical mismatch')
            if cp['probe_manifest_file_sha256']!=PROBE_SHA or cp['probe_contract_sha256']!=probe['canonical_sha256']:raise ValueError('Probe binding')
            if cp['inference_state_sha256']!=hashes['inference_state.npz'] or cp['probe_scores_sha256']!=hashes['probe_scores.npz']:raise ValueError('Internal artifact binding')
            if cp['router_standardization']!={'algorithm':'per_sample_population_zscore_v1','axis':'seen_class_axis','ddof':0,'epsilon':1e-8}:raise ValueError('Router standardization contract')
            schema=cp['state_schema'];keys=set(schema['encoder'].values())|set(schema['normalization'].values())
            for h in schema['heads'].values():keys.update(h.values())
            for r in schema['cap3000_router'].values():keys.update(r.values())
            with np.load(base/'inference_state.npz',allow_pickle=False) as state:
                if set(state.files)!=keys:raise ValueError('State schema mismatch')
                if any(not np.issubdtype(state[k].dtype,np.number) or not np.isfinite(state[k]).all() for k in keys):raise ValueError('Nonfinite state')
                if np.any(state[schema['normalization']['scale']]<=0):raise ValueError('Invalid scale')
            with np.load(base/'probe_scores.npz',allow_pickle=False) as scores:
                n=validate_scores(scores,cp,probe['official_test']['samples'])
            rows.append({'seed':seed,'checkpoint':checkpoint,'seen_classes':cp['seen_classes'],'probe_rows':n,
              'artifact_sha256':hashes,'state_bytes':(base/'inference_state.npz').stat().st_size,
              'state_tensor_count':len(keys),'saved_joint_and_argmax_consistent':True})
    return {'status':'complete_primary_checkpoint_input_coverage_verified','rows':rows,
      'source_lf_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
      'protected_registry_sha256':REGISTRY_SHA,'probe_manifest_sha256':PROBE_SHA,
      'checkpoint_count':len(rows),'artifact_count':3*len(rows),'state_bytes':sum(r['state_bytes'] for r in rows),
      'new_forward_inference_performed':False,'new_forgetting_computed':False,
      'limitations':['This is input integrity and stored-formula validation, not recomputed forward scores.',
       'Existing final-checkpoint CPU forward parity failed; no legacy score was replaced.',
       'No new calibration, test-performance, attribution or ETG result follows.']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('input','registry','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();report=audit(a.input,a.registry)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
