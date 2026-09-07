"""Local-only metadata, holdout-index and inference-state readiness audit.

Does not train, run model inference, fit calibration or read official-test rows.
Raw calibration rows and model weights must not be published with the report.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

PINS={'streaming_manifest.json':'6d6e467ed0687ff6eb2d171c9799c74ee16af3c5b65bea2e662fa569bc3d5e1b',
      'sampling_audit.json':'697750d599f448ba1120ba60decac98abb607204eec1a431073673cc3b124b9b',
      'BUILD_SUMMARY.json':'75f2c1d9fc26b60a2b18780c0bf8ddfb9356b151b8ff240c98eac3b27ce09f26',
      'DERIVED_SHA256SUMS.txt':'d855724b36712cc159ffdd4b0fbe062c5b93aa93650c5f08baa1cf432eb70d9e',
      'build_train_protocol_v2.py':'5762c7362dd6905e97e9407c02cf120e200f83c4f054351917e006deb388df75',
      'build_train_protocol.py':'e6bacab1659b1be09eee169f2b35be86639e82b7ccf5b031e7d7b3c323fad1c4'}
REGISTRY='de43ff1d51f2993741d5a18807f129f1f80db8a97360c0289312d1944a63c58c'
SEEDS=(1,2,3,4,42)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def indices(master,source,cid,n,fraction,cap):
    if n<1 or not 0<=fraction<1 or cap<1:raise ValueError('Invalid split parameters')
    tag=f'{master}|{source}|{cid}|normal_to_largest_attack|v2'
    seed=int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8],'little')
    ncal=min(n-1,max(1,math.floor(n*fraction))) if fraction>0 and n>1 else 0
    perm=np.random.default_rng(seed).permutation(n)
    return seed,np.sort(perm[:ncal]),np.sort(perm[ncal:ncal+cap])
def index_hash(values):return hashlib.sha256(json.dumps(values.tolist(),separators=(',',':')).encode()).hexdigest()
def checksum_registry(path):
    return {line.split(None,1)[1].removeprefix('./'):line.split(None,1)[0] for line in path.read_text().splitlines() if line.strip()}

def audit(root,protected_registry,protocol_root):
    for name,expected in PINS.items():
        if sha(root/name)!=expected:raise ValueError('Pinned input hash mismatch: '+name)
    if sha(protected_registry)!=REGISTRY:raise ValueError('Checkpoint registry mismatch')
    registry=checksum_registry(protected_registry)
    manifest,sampling,build=load(root/'streaming_manifest.json'),load(root/'sampling_audit.json'),load(root/'BUILD_SUMMARY.json')
    conf=sampling['configuration'];classes=sampling['classes'];normal=conf['normal_class_id']
    largest=max(r['fit_pool_rows'] for r in classes if r['id']!=normal)
    if largest!=conf['normal_fit_cap_rows']:raise ValueError('Adaptive normal cap mismatch')
    reports=[]
    for r,record in zip(classes,manifest['classes'],strict=True):
        cid=r['id'];n=r['source_train_rows'];ncal=r['calibration_rows']
        cap=largest if cid==normal else n
        seed,cal,fit=indices(conf['seed'],sampling['source_manifest']['sha256'],cid,n,conf['calibration_fraction'],cap)
        if seed!=r['seed'] or (len(cal),len(fit))!=(ncal,r['fit_rows']):raise ValueError('Split size/seed mismatch')
        if index_hash(cal)!=r['calibration_indices_sha256'] or index_hash(fit)!=r['fit_indices_sha256']:raise ValueError('Index digest mismatch')
        if np.intersect1d(cal,fit).size:raise ValueError('Fit/calibration index overlap')
        expected_train=[dict(path=r['fit']['path'],rows=r['fit_rows'],sha256=r['fit']['sha256'])]
        if record['train']!=expected_train:raise ValueError('Training manifest includes unexpected rows')
        cal_path=root/r['calibration']['path']
        if not cal_path.resolve().is_relative_to(root.resolve()) or sha(cal_path)!=r['calibration']['sha256']:raise ValueError('Calibration data hash mismatch')
        data=np.load(cal_path,mmap_mode='r',allow_pickle=False)
        if data.shape!=(ncal,manifest['feature_dim']) or not np.isfinite(data).all():raise ValueError('Calibration shape or nonfinite data')
        reports.append(dict(class_id=cid,class_name=r['name'],source_train_rows=n,fit_rows=len(fit),calibration_rows=len(cal),
            calibration_file_sha256=sha(cal_path),fit_file_sha256=r['fit']['sha256'],
            fit_indices_sha256=index_hash(fit),calibration_indices_sha256=index_hash(cal),index_overlap=0,
            manifest_training_is_fit_only=True,calibration_dtype=str(data.dtype)))
        del data
    if sum(r['fit_rows'] for r in reports)!=build['fit_rows'] or sum(r['calibration_rows'] for r in reports)!=build['calibration_rows']:raise ValueError('Total mismatch')
    states=[]
    for seed in SEEDS:
        base=root/f'seed_{seed}'; prefix=f'last_epoch/seed_{seed}/monitoring_seed_{seed}/checkpoint_003/'
        cp=load(base/'checkpoint_manifest.json')
        for name in ('checkpoint_manifest.json','inference_state.npz'):
            if sha(base/name)!=registry[prefix+name]:raise ValueError('Protected checkpoint hash mismatch')
        if (cp['seed'],cp['checkpoint'],cp['seen_classes'],cp['feature_dim'])!=(seed,3,list(range(8)),78):raise ValueError('Checkpoint identity')
        if cp['inference_state_sha256']!=sha(base/'inference_state.npz'):raise ValueError('State binding')
        protocol_path=protocol_root/f'seed_{seed}/protocol.json'
        if sha(protocol_path)!=registry[f'last_epoch/seed_{seed}/protocol.json']:raise ValueError('Protocol file hash')
        p=load(protocol_path)
        if p['config']['family_checkpoint_selection']!='last' or p['training_calibration']!={'enabled':False}:raise ValueError('Wrong primary calibration policy')
        if p['manifest_sha256']!=PINS['streaming_manifest.json']:raise ValueError('Training data manifest mismatch')
        train_records=[r for r in p['dataset_files'] if r['split']=='train']
        if len(train_records)!=8 or {r['sha256'] for r in train_records}!={r['fit_file_sha256'] for r in reports}:raise ValueError('Runtime training shard mismatch')
        schema=cp['state_schema'];keys=set(schema['encoder'].values())
        for head in schema['heads'].values():keys.update(head.values())
        keys.update(schema['normalization'].values())
        for router in schema['cap3000_router'].values():keys.update(router.values())
        with np.load(base/'inference_state.npz',allow_pickle=False) as arrays:
            if set(arrays.files)!=keys:raise ValueError('Inference-state schema membership')
            sizes={k:list(arrays[k].shape) for k in sorted(keys)}
            if not all(np.issubdtype(arrays[k].dtype,np.number) and np.isfinite(arrays[k]).all() for k in keys):raise ValueError('State nonnumeric/nonfinite')
            if np.any(arrays[schema['normalization']['scale']]<=0):raise ValueError('Invalid normalization scale')
        states.append(dict(seed=seed,checkpoint=3,checkpoint_manifest_sha256=sha(base/'checkpoint_manifest.json'),
            inference_state_sha256=sha(base/'inference_state.npz'),bytes=(base/'inference_state.npz').stat().st_size,
            tensor_count=len(keys),tensor_shapes=sizes,checkpoint_policy='last',training_calibration_enabled=False,
            runtime_train_shards_fit_only=True,reconstruction_scope=cp['reconstruction_scope'],
            reconstruction_verified_by_forward_parity=False,source_protocol_file_sha256=sha(protocol_path)))
    source_digest=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    return dict(status='verified_holdout_indices_and_state_integrity',audit_source_lf_sha256=source_digest,input_sha256=PINS,protected_registry_sha256=REGISTRY,
        numpy_version=np.__version__,classes=reports,states=states,
        totals=dict(fit_rows=build['fit_rows'],calibration_rows=build['calibration_rows'],official_test_rows=build['official_test_rows']),
        readiness=dict(final_checkpoint_states_verified=5,all_checkpoint_states_verified=False,
            calibration_rows_and_index_hashes_verified=True,training_manifest_fit_only=True,
            forward_reconstruction_parity_verified=False,new_score_tables_available=False,new_calibrator_fitted=False),
        limitations=['Disjoint source indices do not prove feature-content deduplication across partitions.',
            'This rechecks materialized calibration arrays, not every raw source or fit/test array byte.',
            'Heartbleed has one calibration row; per-class calibration/reliability cannot be established.',
            'Adaptive normal cap uses full offline attack-class counts; Task-0-only normalization is not a claim of wholly prospective stream construction.',
            'Only final-checkpoint inference states were retrieved here; a new forgetting calculation needs all checkpoints.',
            'State integrity is not forward-score parity. No training, calibration fitting, new test result or SHAP/ETG outcome was produced.'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--protected-registry',type=Path,required=True);p.add_argument('--protocol-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps(audit(a.input,a.protected_registry,a.protocol_root),indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
