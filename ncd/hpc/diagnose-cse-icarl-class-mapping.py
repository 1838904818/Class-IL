"""Read-only class-mapping and operating-point decomposition from existing CSE iCaRL evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base',type=Path,required=True)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
B=a.base.resolve(strict=True)
archive=B/'admin/cse-icarl-completed-publication-v3'
data=B/'data/derived/section21_clean_v1/cic-ids-2018'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
meta=read(data/'data_quality.json')
names=['Normal','DoS','DDoS','Bruteforce','WebAttack','Infiltration','Botnet']
assert meta['class_order']==names
assert meta['tasks']==[[0,1],[2,3],[4,5],[6]]
y=np.load(data/'test_y.npy')
counts=np.bincount(y,minlength=7)
assert counts.tolist()==[meta['id_counts'][n] for n in names]
metrics=read(archive/'overview/per_seed_metrics.json')
result=[]
evidence=[]
for seed in range(10,15):
    r=read(archive/f'icarl-scoped-npy_seed{seed}/result.json')
    original=archive/f'icarl-scoped-npy_seed{seed}/result.json'
    evidence.append(dict(seed=seed,result_sha256=hashlib.sha256(original.read_bytes()).hexdigest()))
    i=r['identity']
    assert r['class_order']==i['class_order']==names and i['tasks']==meta['tasks']
    root=B/('runs/section178/cic-ids-2018/icarl-scoped-npy/seed10' if seed==10 else f'validation/hpc-20261102T1345/a/icarl/seed{seed}')
    for cp in r['checkpoints']:
        path=root/'cell'/cp['checkpoint_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==cp['checkpoint_sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['identity']['class_order']==names
        labels=torch.cat(state['buffer_y']).numpy()
        assert sorted(set(labels.tolist()))==cp['seen_classes']==list(range(len(cp['seen_classes'])))
        assert state['model_state']['head.weight'].shape[0]==len(names)
        assert all(int((labels==c).sum())==200 for c in cp['seen_classes'])
    views={}
    for view in ('default','0.01','0.02','0.05'):
        matrices=[np.array(d['metrics']['confusion_matrix'] if view=='default' else c['operating_points'][view]['confusion_matrix'],dtype=np.int64)
                  for d,c in zip(r['default_stages'],r['calibrated_stages'])]
        per_class=[]
        for label,name in enumerate(names):
            recalls=[]
            for cp,cm in zip(r['checkpoints'],matrices):
                assert cm.sum(axis=1).tolist()==[int(counts[c]) if c in cp['seen_classes'] else 0 for c in range(7)]
                if label in cp['seen_classes']:recalls.append(float(cm[label,label]/counts[label]))
            final=matrices[-1]
            per_class.append(dict(class_id=label,name=name,stage_recalls=recalls,first=recalls[0],peak=max(recalls),final=recalls[-1],
                final_to_normal=float(final[label,0]/counts[label]),final_predicted_counts=final[label].tolist()))
            expected=metrics[str(seed)][view]['per_class'][label]
            for key in ('first','peak','final'):assert abs(per_class[-1][key]-expected[key])<1e-12
        final_bacc=sum(v['final'] for v in per_class)/7
        assert abs(final_bacc-metrics[str(seed)][view]['bacc'])<1e-12
        views[view]=dict(final_bacc=final_bacc,classes=per_class)
    final_delta=100*(views['0.01']['final_bacc']-views['default']['final_bacc'])
    result.append(dict(seed=seed,views=views,calibrated_minus_default_final_bacc_pp=final_delta))
    dd={v:views[v]['classes'][2] for v in views}
    print(json.dumps(dict(event='ICARL_DDOS_PER_SEED',seed=seed,default_recalls=dd['default']['stage_recalls'],onepct_recalls=dd['0.01']['stage_recalls'],
        final_default_to_normal=dd['default']['final_to_normal'],final_onepct_to_normal=dd['0.01']['final_to_normal'],
        calibrated_minus_default_final_bacc_pp=final_delta)),flush=True)
model_source=(a.source/'ncd/model.py').read_text(encoding='utf-8')
assert 'seen = sorted(prototypes)' in model_source and 'scores.index_copy(1, self.seen_ids, seen_scores)' in model_source
diagnostic=dict(event='ICARL_CLASS_MAPPING_DIAGNOSTIC',class_order=names,test_supports=counts.tolist(),
    mapping_check='Prepared metadata, identity, checkpoint buffer labels and global output indices agree for20 checkpoints; not a new audit of original raw CSV labels.',
    inference_mapping_source='DecisionScore sorts prototype labels and index_copy assigns NME scores to those same global class columns.',
    inference_scope='Existing saved confusion matrices only; no new prediction or training.',
    causal_limit='An operating-point decomposition cannot prove exemplar-budget inadequacy or prototype collapse. No such causal conclusion is asserted.',
    input_evidence=evidence,metadata_sha256=hashlib.sha256((data/'data_quality.json').read_bytes()).hexdigest(),
    model_source_sha256=hashlib.sha256((a.source/'ncd/model.py').read_bytes()).hexdigest(),
    archive_source='Published icarl-scoped-npy five-seed archive; no original files modified.',
    seeds=result)
assert not a.output.exists(),'Do not overwrite prior diagnostics'
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(diagnostic,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(diagnostic,allow_nan=False),flush=True)
