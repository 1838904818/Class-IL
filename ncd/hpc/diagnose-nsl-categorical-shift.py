"""Check NSL categorical novelty from original official split records and frozen prepared columns."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base',type=Path,required=True)
p.add_argument('--raw',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
B=a.base.resolve(strict=True)
assert not a.output.exists()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
data=B/'data/derived/section21_clean_v1/nsl-kdd'
meta=read(data/'data_quality.json')
splits={};hashes={}
for split,filename in (('train','KDDTrain+.txt'),('test','KDDTest+.txt')):
    path=a.raw/filename
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    receipt=next(x for x in meta['input_files'] if x['path'].endswith(filename))
    assert sha==receipt['sha256'] and path.stat().st_size==receipt['bytes']
    rows=[]
    with path.open(encoding='utf-8',newline='') as f:
        for r in csv.reader(f):
            assert len(r) in (42,43)
            rows.append((r[1],r[2],r[3],r[41].rstrip('.').lower()))
    y=np.load(data/f'{split}_y.npy')
    assert len(rows)==len(y)
    normal=np.array([r[3]=='normal' for r in rows])
    assert np.array_equal(normal,y==0),'Original split row order/Normal mapping differs'
    splits[split]=dict(rows=rows,normal=normal,y=y,x=np.load(data/f'{split}_x.npy',mmap_mode='r'))
    hashes[filename]=sha
findings={}
for j,field in enumerate(('protocol_type','service','flag')):
    train=splits['train'];test=splits['test']
    seen_all={r[j] for r in train['rows']}
    seen_normal={r[j] for r,n in zip(train['rows'],train['normal']) if n}
    task0=np.isin(train['y'],meta['tasks'][0])
    seen_task0={r[j] for r,n in zip(train['rows'],task0) if n}
    prefix=f'categorical__{field}_'
    columns=[i for i,s in enumerate(meta['feature_names']) if s.startswith(prefix)]
    represented={meta['feature_names'][i][len(prefix):] for i in columns}
    assert represented==seen_task0,'Task0 TRAIN-only encoder categories differ from original Task0 subset'
    normal_ids=np.flatnonzero(test['normal'])
    unseen_full=[int(i) for i in normal_ids if test['rows'][i][j] not in seen_all]
    unseen_encoder=[int(i) for i in normal_ids if test['rows'][i][j] not in represented]
    block=np.asarray(test['x'][:,columns])
    assert set(np.unique(block)).issubset({0.,1.})
    unknown=np.flatnonzero(block.sum(1)==0)
    expected=[i for i,r in enumerate(test['rows']) if r[j] not in represented]
    assert unknown.tolist()==expected
    findings[field]=dict(train_all_categories=sorted(seen_all),train_normal_categories=sorted(seen_normal),train_task0_categories=sorted(seen_task0),
        test_normal_categories=dict(sorted(Counter(test['rows'][i][j] for i in normal_ids).items())),
        test_normal_unseen_in_full_train_count=len(unseen_full),test_normal_unseen_in_full_train_row_ids=unseen_full,
        test_normal_unseen_in_task0_encoder_count=len(unseen_encoder),test_normal_unseen_in_task0_encoder_row_ids=unseen_encoder,
        all_test_unknown_encoder_rows=len(unknown),all_test_unknown_encoder_categories=dict(sorted(Counter(test['rows'][i][j] for i in unknown).items())))
train_labels={r[3] for r in splits['train']['rows']}
test_labels=Counter(r[3] for r in splits['test']['rows'])
novel={k:v for k,v in sorted(test_labels.items()) if k not in train_labels}
out=dict(dataset='NSL-KDD',source_sha256=hashes,metadata_sha256=hashlib.sha256((data/'data_quality.json').read_bytes()).hexdigest(),
    train_rows=len(splits['train']['rows']),test_rows=len(splits['test']['rows']),test_normal_rows=int(splits['test']['normal'].sum()),
    row_identity_check='Original official TXT order and Normal row mask exactly match prepared arrays for both complete splits.',
    preprocessing='One-hot fit on Task0 TRAIN (Normal plus DoS), not all TRAIN; unknowns map to all-zero within the category block.',
    categorical_findings=findings,test_attack_label_variants_absent_from_train=novel,
    variant_scope='Exact original attack-label strings, not new coarse family classes.',
    limitation='No model score or alarm list was computed. This can refute an unseen-category explanation when all TEST benign categories are already present, but cannot explain score shift within known categories or identify which rows alarm. Quantiles/KS and common TEST-informed FPR diagnostics remain separate.')
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(event='NSL_CATEGORY_NOVELTY_COMPLETE',test_normal_rows=out['test_normal_rows'],
    fields={k:dict(unseen_full_train=v['test_normal_unseen_in_full_train_count'],unseen_encoder=v['test_normal_unseen_in_task0_encoder_count']) for k,v in findings.items()},
    unseen_attack_variants=len(novel),unseen_attack_variant_rows=sum(novel.values()),model_inference=False)),flush=True)
