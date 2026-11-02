"""Independent plain-list CM and original-source checks for a frozen export."""
import argparse,gzip,hashlib,json,re,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--base',type=Path,help='Optional protected original-log root; public CM verification does not require it');a=p.parse_args();D=a.archive;B=a.base
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def close(x,y):assert abs(x-y)<1e-12,(x,y)
rows={};source_dicts=[]
for seed in range(10,15):
    root=D/f'cer-bic-common-completed_seed{seed}';r=read(root/'result.json');stored=read(root/'metrics.json');i=r['identity'];source_dicts.append(i['source_sha256'])
    per={}
    for view in ('default','0.01','0.02','0.05'):
        history=[];stage=[]
        for t,cp in enumerate(r['checkpoints']):
            cm=(r['default_stages'][t]['metrics'] if view=='default' else r['calibrated_stages'][t]['operating_points'][view])['confusion_matrix']
            supports=[sum(row) for row in cm];cols=[sum(row[c] for row in cm) for c in range(7)];seen=cp['seen_classes']
            assert all(isinstance(v,int) and v>=0 for row in cm for v in row)
            recalls={c:cm[c][c]/supports[c] for c in seen};history.append(recalls)
            stage.append(dict(bacc=statistics.mean(recalls.values()),accuracy=sum(cm[c][c] for c in seen)/sum(supports),macro_f1=statistics.mean(2*cm[c][c]/(supports[c]+cols[c]) for c in seen),normal_fpr=1-cm[0][0]/supports[0],attack_recall=1-sum(cm[c][0] for c in range(1,7))/sum(supports[1:])))
        forgotten=[max(h[c] for h in history[:-1] if c in h)-history[-1][c] for c in seen if any(c in h for h in history[:-1])]
        point=dict(**stage[-1],incremental_bacc=statistics.mean(v['bacc'] for v in stage),forgetting=statistics.mean(forgotten))
        for key,value in point.items():close(value,stored[view][key])
        for t,expected in enumerate(stage):
            for key,value in expected.items():close(value,stored[view]['stages'][t][key])
        for cp in stored[view]['per_class']:
            c=cp['class_id'];values=[h[c] for h in history if c in h];prior=values[:-1]
            for key,value in dict(first=values[0],peak=max(values),best_prior=max(prior) if prior else None,final=values[-1],forgetting=max(prior)-values[-1] if prior else None).items():
                if value is None:assert cp[key] is None
                else:close(value,cp[key])
        per[view]=point
    rows[seed]=per
    costs=read(root/'costs.json');close(costs['backbone_seconds']+costs['bias_seconds'],costs['recorded_training_seconds_total'])
    close(costs['newly_recorded_training_seconds']+costs['prior_imported_training_seconds']+costs['resume_preexisting_epoch_seconds'],costs['recorded_training_seconds_total'])
    if seed==14:assert i['resume']['rng_restored'] is False and r['original_checkpoint_identity']['seed']==14
assert [len(v) for v in source_dicts]==[401,403,403,403,403]
assert source_dicts[1]==source_dicts[2]==source_dicts[3]==source_dicts[4]
assert all(source_dicts[1][k]==v for k,v in source_dicts[0].items())
aggregate=read(D/'overview/aggregate.json')
for view,metrics in aggregate.items():
    for key,expected in metrics.items():
        values=[rows[s][view][key] for s in range(10,15)];close(statistics.mean(values),expected['mean']);close(statistics.stdev(values),expected['sample_sd'])
if B is not None:
    for rec in read(D/'overview/log_artifacts.json'):
        original=B/rec['source_relative'];assert hashlib.sha256(original.read_bytes()).hexdigest()==rec['original_sha256']
        assert len(original.read_bytes())==rec['original_bytes']
files=[p for p in D.rglob('*') if p.is_file()]
for path in files:
    raw=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
    assert not re.search(rb'/home/user/[A-Za-z0-9]|/scr/user/[A-Za-z0-9]|(?<![A-Za-z])[CE]:[/\\]|ghp_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',raw),path
print(json.dumps(dict(event='CSE_BIC_EXPORT_INDEPENDENTLY_VERIFIED',artifacts=len(files),seeds=5,checkpoints=20,calibrations=60,exact_cm_integer_supports=True,metrics_and_cost_arithmetic=True,source401_subset403=True,all_four_source403_equal=True,protected_original_log_hashes_checked=B is not None,full_research_complete=False)),flush=True)
