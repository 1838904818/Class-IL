"""Deterministic extraction scheduling and exclusive, hash-verified records.

No cluster access, inference, fit, acceptance or evaluation is implemented here.
"""
import hashlib
import copy
import json
import math
import os
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     allow_nan=False).encode()).hexdigest()


def validate_item(item):
    require(type(item) is dict and set(item)=={'ordinal','target','checkpoint','repeat','seed','record_id'},'item keys')
    for key,limit in (('ordinal',1024),('checkpoint',2),('repeat',8)):
        require(type(item[key]) is int and 0<=item[key]<limit,'item '+key)
    require(type(item['seed']) is int and item['seed']==20270000+item['repeat'],'paired seed')
    t=item['target']
    require(type(t) is dict and set(t)=={'class_id','source_offset','row_id','calibration_ordinal',
        'parent_start','parent_rows','row_in_parent','native_shard_ordinal'},'target keys')
    for key in set(t)-{'row_id'}:
        require(type(t[key]) is int and t[key]>=0,'target '+key)
    require(t['class_id'] in (0,1) and t['native_shard_ordinal']==0,'target class/shard')
    require(1<=t['parent_rows']<=512 and t['parent_start']==t['calibration_ordinal']//512*512
        and t['row_in_parent']==t['calibration_ordinal']-t['parent_start']
        and t['row_in_parent']<t['parent_rows'],'native context')
    require(type(t['row_id']) is str and len(t['row_id'])==64 and
        all(x in '0123456789abcdef' for x in t['row_id']),'row identity')
    require(item['record_id']==digest({k:v for k,v in item.items() if k!='record_id'}),'record identity changed')


def all_targets(mapping):
    targets=[]
    for cid in (0,1):
        role=mapping[cid]['roles']['trigger']
        require(len(role['row_ids'])==len(role['offsets'])==len(role['ordinals'])==32,'32 triggers per old class')
        for offset,ordinal,row_id in zip(role['offsets'],role['ordinals'],role['row_ids']):
            ordinal=int(ordinal); start=ordinal//512*512
            size=min(512,len(mapping[cid]['calibration'])-start)
            require(0<=ordinal<len(mapping[cid]['calibration']),'out-of-range calibration row')
            targets.append(dict(class_id=cid,source_offset=int(offset),row_id=row_id,
                calibration_ordinal=ordinal,parent_start=start,parent_rows=size,
                row_in_parent=ordinal-start,native_shard_ordinal=0))
    require(len({t['row_id'] for t in targets})==64,'duplicate target identity')
    return targets


def schedule(targets):
    require(len(targets)==64 and len({t['row_id'] for t in targets})==64,'complete target set required')
    # Interleave classes, checkpoint, repeat to avoid a first chunk covering only Benign/cp0.
    by_class={c:[t for t in targets if t['class_id']==c] for c in (0,1)}
    require(all(len(v)==32 for v in by_class.values()),'two old classes')
    output=[]
    for position in [0,31]+list(range(1,31)):
        for cid in (0,1):
            for checkpoint in (0,1):
                for repeat in range(8):
                    target=copy.deepcopy(by_class[cid][position])
                    # Shared ordered permutation seeds make checkpoint/target comparisons paired.
                    # This fixed set excludes the feasibility seed and never depends on results.
                    seed=20270000+repeat
                    item=dict(ordinal=len(output),target=target,checkpoint=checkpoint,
                              repeat=repeat,seed=seed)
                    item['record_id']=digest(item)
                    validate_item(item)
                    output.append(item)
    return output


def check_record(result,item,science_hash):
    validate_item(item)
    fixed=dict(status='EXTRACTION_RECORD_COMPLETE',science_sha256=science_hash,record_id=item['record_id'],
        checkpoint=item['checkpoint'],class_id=item['target']['class_id'],
        row_id=item['target']['row_id'],seed=item['seed'],repeat=item['repeat'],
        max_evals=157,batch_size=1,repeat_count=1,shap_version='0.51.0',
        masker_invariance='exact-value-equality',shap_attributions_computed=True,
        etg_actions_fitted=False,efficacy_metrics_computed=False,statistical_certificate=False,
        scientific_result_generated=False,archived_score_parity_verified=False)
    for key,value in fixed.items():
        require(key in result and type(result[key]) is type(value) and result[key]==value,'record field '+key)
    a=result.get('attributions')
    require(type(a) is list and len(a)==78 and all(type(x) in (int,float) and math.isfinite(x) for x in a),'attribution contract')
    calls=result.get('native_calls')
    require(type(calls) is int and 0<calls<=5200,'call count')
    require(type(result.get('native_context_rows')) is int and
            result['native_context_rows']==calls*item['target']['parent_rows'],'context count')
    for key in ('base_value','native_target','additive_residual'):
        require(type(result.get(key)) in (int,float) and math.isfinite(result[key]),'finite '+key)
    residual=result['base_value']+sum(a)-result['native_target']
    require(abs(residual)<=1e-6+1e-6*abs(result['native_target']) and
            abs(residual-result['additive_residual'])<=1e-10,'reconstruction')


def sync_directory(path):
    if os.name=='posix':
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)


def safe_directory(directory):
    root=Path(directory).absolute()
    require(root.is_dir(),'missing record directory')
    for p in (root,*root.parents):
        require(not p.is_symlink() and not (getattr(p.lstat(),'st_file_attributes',0)&0x400),
                'linked record directory')
    return root


def write_record(directory,result,item,science_hash):
    check_record(result,item,science_hash)
    path=safe_directory(directory)/(item['record_id']+'.json')
    raw=(json.dumps(result,sort_keys=True,allow_nan=False,indent=2)+'\n').encode()
    # A partial exclusive file is retained after failure; never silently overwrite or skip it.
    with path.open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    sync_directory(path.parent)
    return dict(record_id=item['record_id'],sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def read_record(directory,entry,item,science_hash):
    validate_item(item)
    require(entry.get('record_id')==item['record_id'],'entry identity')
    path=safe_directory(directory)/(item['record_id']+'.json')
    require(path.is_file() and all(not p.is_symlink() for p in (path,*path.parents)) and
            path.stat().st_size<=1024*1024,'invalid record file')
    raw=path.read_bytes()
    require(len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],'record hash mismatch')
    value=json.loads(raw);check_record(value,item,science_hash)
    return value


def commit_record(directory,entry,item,science_hash):
    # Called by the supervisor only after the separate writer has exited successfully.
    read_record(directory,entry,item,science_hash)
    marker=dict(schema='etg-record-commit-v1',science_sha256=science_hash,**entry)
    raw=(json.dumps(marker,sort_keys=True,allow_nan=False)+'\n').encode()
    path=safe_directory(directory)/(item['record_id']+'.commit.json')
    with path.open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    sync_directory(path.parent)
    return dict(**entry,commit_sha256=hashlib.sha256(raw).hexdigest(),commit_bytes=len(raw))


def verified_resume(directory,entries,items,science_hash):
    for item in items:validate_item(item)
    lookup={i['record_id']:i for i in items};out={}
    require(len(lookup)==len(items),'duplicate scheduled IDs')
    root=safe_directory(directory)
    expected=set()
    for entry in entries:
        rid=entry.get('record_id')
        require(type(rid) is str and rid in lookup,'unknown resume ID')
        expected.update((rid+'.json',rid+'.commit.json'))
    require({p.name for p in root.iterdir()}==expected,
            'orphan, missing or unreviewed record file')
    for entry in entries:
        rid=entry['record_id']
        require(rid in lookup and rid not in out,'unknown/duplicate resume record')
        path=Path(directory)/(rid+'.commit.json')
        require(path.is_file() and all(not p.is_symlink() for p in (path,*path.parents)) and
                path.stat().st_size<=4096,'missing or invalid commit marker')
        raw=path.read_bytes()
        require(len(raw)==entry['commit_bytes'] and hashlib.sha256(raw).hexdigest()==entry['commit_sha256'],'commit hash mismatch')
        require(json.loads(raw)==dict(schema='etg-record-commit-v1',science_sha256=science_hash,
                **{k:entry[k] for k in ('record_id','bytes','sha256')}),'commit contract')
        out[rid]=read_record(directory,entry,lookup[rid],science_hash)
    return out


def completion_gate(directory,entries,items,science_hash):
    """No downstream ETG fitting on a partial Cartesian extraction set."""
    require(len(items)==1024,'full schedule required')
    for item in items:validate_item(item)
    require([i['ordinal'] for i in items]==list(range(1024)),'schedule order')
    descriptors={}
    for item in items:
        rid=item['target']['row_id']; target=item['target']
        require(rid not in descriptors or descriptors[rid]==target,'inconsistent target descriptor')
        descriptors[rid]=target
    require(all(sum(t['class_id']==cid for t in descriptors.values())==32 for cid in (0,1)),
            '32 distinct targets per old class')
    coordinates={(i['target']['row_id'],i['checkpoint'],i['repeat']) for i in items}
    require(len(coordinates)==1024,'duplicate extraction coordinates')
    targets={i['target']['row_id'] for i in items}
    require(len(targets)==64 and coordinates=={
        (rid,cp,rep) for rid in targets for cp in (0,1) for rep in range(8)},
        'incomplete Cartesian schedule')
    require(all(i['seed']==20270000+i['repeat'] for i in items),'unpaired seeds')
    require(len(entries)==1024,'extraction incomplete; ETG fitting forbidden')
    return verified_resume(directory,entries,items,science_hash)
