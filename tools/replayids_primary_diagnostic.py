"""Read-only recomputation of registered-last-epoch ReplayIDS evidence.

Uses existing research result files; does not fit or select any model or rule.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_balanced_replay_replication import METRICS, LOWER, summary, exact_p, holm
import analyze_balanced_replay_replication as aggregate_tools

SEEDS=(1,2,3,4,42)
DATASET='cic-ids-2017-replayids-contract-normalcap-largest-attack'
REGISTRY_SHA='c46d3d25e4a9c839d0f653f4a45e58cfa5c7a5302df8b9c89e00d27c231edbe6'
REGISTRY_LF_SHA='a8572c7d52b79dfe12107f4519575668036ee8b5c051afd1b755c8a552ac0962'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return json.loads(path.read_text(encoding='utf-8'))
def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-12): raise ValueError('Metric mismatch')

def audit_cps(cps, scores, tasks):
    if len(cps)!=len(tasks): raise ValueError('Missing checkpoints')
    matrix=[]
    for ci,cp in enumerate(cps):
        seen=cp['seen_classes'];m=cp['metrics'];cm=m['confusion_matrix'];n=len(seen)
        if cp['checkpoint']!=ci or seen!=[c for t in tasks[:ci+1] for c in t]: raise ValueError('Class or checkpoint order')
        if len(cm)!=n or any(len(r)!=n for r in cm): raise ValueError('Confusion shape')
        if any(type(v)!=int or v<0 for row in cm for v in row): raise ValueError('Confusion values')
        total=sum(map(sum,cm))
        if total!=m['total_rows'] or len(m['per_class'])!=n: raise ValueError('Support mismatch')
        close(sum(cm[i][i] for i in range(n))/total,m['accuracy'])
        for i,row in enumerate(m['per_class']):
            support=sum(cm[i]);pred=sum(r[i] for r in cm);tp=cm[i][i]
            p=tp/pred if pred else 0;r=tp/support if support else 0;f=2*p*r/(p+r) if p+r else 0
            if (row['class_id'],row['support'])!=(seen[i],support): raise ValueError('Class identity')
            for key,v in [('precision',p),('recall',r),('f1',f)]: close(v,row[key])
        close(st.mean(r['recall'] for r in m['per_class']),m['balanced_accuracy'])
        close(st.mean(r['f1'] for r in m['per_class']),m['macro_f1'])
        bi=seen.index(0);benign=sum(cm[bi]);attacks=total-benign
        close(1-cm[bi][bi]/benign,m['binary_detection']['benign_false_positive_rate'])
        close(1-sum(row[bi] for i,row in enumerate(cm) if i!=bi)/attacks,m['binary_detection']['attack_detection_recall'])
        row=[]
        for ti,task in enumerate(tasks):
            ids=[seen.index(c) for c in task if c in seen]
            value=sum(cm[i][i] for i in ids)/sum(sum(cm[i]) for i in ids) if ids else None
            row.append(value)
            if value is not None: close(value,m['task_accuracy'][str(ti)])
        matrix.append(row)
    for row,expected in zip(matrix,scores['task_accuracy_matrix'],strict=True):
        for a,b in zip(row,expected,strict=True):
            if a is None:
                if b is not None: raise ValueError('Unevaluated task has score')
            else: close(a,b)
    close(st.mean(matrix[-1]),scores['average_task_accuracy'])
    f=[max(r[i] for r in matrix[:-1] if r[i] is not None)-matrix[-1][i] for i in range(len(tasks)-1)]
    close(st.mean(f),scores['average_forgetting'])
    final=cps[-1]['metrics']
    for key,src in [('final_overall_accuracy','accuracy'),('final_macro_f1','macro_f1'),('final_balanced_accuracy','balanced_accuracy')]: close(scores[key],final[src])
    for key in ('attack_detection_recall','benign_false_positive_rate'): close(scores['final_'+key],final['binary_detection'][key])

def compare_vectors(ofra,baseline):
    delta=[100*(o-b) for o,b in zip(ofra,baseline,strict=True)]
    return dict(ofra=summary([v*100 for v in ofra]),balanced_replay50=summary([v*100 for v in baseline]),
        ofra_minus_replay_pp=summary(delta),exact_sign_flip_p=exact_p(delta),exact_wilcoxon_p=exact_p(delta,True))

def analyze(repo, policy_path=None):
    source=repo/'results/replayids-d2-checkpoint-recall-guard-paired5'
    exported=repo/'results/balanced-replay-five-seed'
    registry=source/'job426307_independent_analysis.json'
    policy_path=policy_path or repo/'results/replayids-primary-diagnostic/POLICY_BINDINGS.json'
    policies={(p['role'],p['seed']):p for p in load(policy_path)}
    if set(policies)!={(r,s) for r in ('primary','guard') for s in SEEDS}: raise ValueError('Missing policy bindings')
    if hashlib.sha256(registry.read_text(encoding='utf-8').encode()).hexdigest()!=REGISTRY_LF_SHA:
        raise ValueError('Source registry LF hash mismatch')
    pins={(r['arm'],r['seed']):r['sha256'] for r in load(registry)['input_files']}
    hashes=load(exported/'EXPORT_SHA256.json')
    for name,h in hashes.items():
        if Path(name).name!=name or sha(exported/name)!=h: raise ValueError('Replay export hash mismatch')
    records={};input_hashes={'results/replayids-primary-diagnostic/POLICY_BINDINGS.json':sha(policy_path),
        'results/balanced-replay-five-seed/EXPORT_SHA256.json':sha(exported/'EXPORT_SHA256.json')};per_class=[]
    for seed in SEEDS:
        primary_path=source/f'baseline/seed_{seed}/result_seed_{seed}.json'
        guard_path=source/f'per_seed/seed_{seed}/result_seed_{seed}.json'
        for arm,path in [('baseline',primary_path),('candidate',guard_path)]:
            if sha(path)!=pins[(arm,seed)]: raise ValueError('OFRA source hash mismatch')
            input_hashes[path.relative_to(repo).as_posix()]=sha(path)
        primary,guard=load(primary_path),load(guard_path)
        for role,result in [('primary',primary),('guard',guard)]:
            binding=policies[(role,seed)]
            required='last' if role=='primary' else 'training_only_calibration_macro_f1_recall_fpr_guard'
            if binding['job_id']!=(425539 if role=='primary' else 426307): raise ValueError('Policy job mismatch')
            if binding['checkpoint_selection']!=required or binding['source_protocol_sha256']!=result['protocol_sha256']:
                raise ValueError('Checkpoint policy identity mismatch')
            if binding['result_sha256']!=pins[('baseline' if role=='primary' else 'candidate',seed)]: raise ValueError('Policy result hash mismatch')
        replay_path=exported/f'replayids_d2_seed{seed}_evaluation.json'
        replay=load(replay_path);protocol=load(exported/f'replayids_d2_seed{seed}_protocol.json')
        input_hashes[replay_path.relative_to(repo).as_posix()]=sha(replay_path)
        input_hashes[f'results/balanced-replay-five-seed/replayids_d2_seed{seed}_protocol.json']=sha(exported/f'replayids_d2_seed{seed}_protocol.json')
        if primary['seed']!=seed or primary['dataset']!=DATASET or guard['dataset']!=DATASET: raise ValueError('Dataset/seed identity')
        if primary['normalization']!=guard['normalization']: raise ValueError('Normalization mismatch')
        if primary['normalization']!=protocol['normalization']: raise ValueError('Replay normalization mismatch')
        tasks=protocol['tasks']
        for role in ('primary','guard'):
            if policies[(role,seed)]['tasks']!=tasks or policies[(role,seed)]['manifest_sha256']!=protocol['manifest_sha256']:
                raise ValueError('Policy data/task contract mismatch')
        for result in (primary,guard):
            for arm,scores in result['summary']['views']['official'].items():
                cps=[dict(checkpoint=c['checkpoint'],seen_classes=c['seen_classes'],metrics=c['views']['official']['arms'][arm]) for c in result['checkpoints']]
                audit_cps(cps,scores,tasks)
                if cps[-1]['metrics']['total_rows']!=227723: raise ValueError('Official test count')
                identity=lambda rows:[(r['class_id'],r['class_name'],r['support']) for r in rows]
                if identity(cps[-1]['metrics']['per_class'])!=identity(replay['balanced_replay50_checkpoints'][-1]['metrics']['per_class']): raise ValueError('Test class support mismatch')
        audit_cps(replay['balanced_replay50_checkpoints'],replay['balanced_replay50_summary'],tasks)
        records[seed]=dict(primary=primary['summary']['views']['official'],guard=guard['summary']['views']['official'],replay=replay['balanced_replay50_summary'])
        arms=primary['checkpoints'][-1]['views']['official']['arms']
        for i,row in enumerate(arms['joint_cap3000']['per_class']):
            counts={a:arms[a]['confusion_matrix'][i][0] for a in arms}
            replay_cm=replay['balanced_replay50_checkpoints'][-1]['metrics']['confusion_matrix']
            per_class.append(dict(seed=seed,class_id=row['class_id'],class_name=row['class_name'],support=row['support'],
                predicted_benign_counts=counts,balanced_replay50_predicted_benign=replay_cm[i][0],
                joint_minus_router_benign_count=counts['joint_cap3000']-counts['router_only_cap3000'],
                note='Aggregate net count, not identified per-row prediction flips. For true Benign, predicted Benign is correct, not a miss.'))
    cohorts={}
    for name,seeds in [('all_five_descriptive',SEEDS),('new_seeds_1_to_4_sensitivity',SEEDS[:4])]:
        arms={a:{k:summary([records[s]['primary'][a][k]*100 for s in seeds]) for k in METRICS} for a in records[1]['primary']}
        comparisons={}
        for policy in ('primary','guard'):
            rows={k:compare_vectors([records[s][policy]['joint_cap3000'][k] for s in seeds],[records[s]['replay'][k] for s in seeds]) for k in METRICS}
            for k,row in rows.items():
                d=row['ofra_minus_replay_pp']['values'];benefit=[-v if k in LOWER else v for v in d]
                row['ofra_wins']=sum(v>1e-12 for v in benefit);row['ofra_losses']=sum(v<-1e-12 for v in benefit)
            for test in ('sign_flip','wilcoxon'):
                for k,p in holm({k:v[f'exact_{test}_p'] for k,v in rows.items()}).items():rows[k][f'holm_{test}_p']=p
            comparisons[policy]=rows
        cohorts[name]=dict(seeds=list(seeds),primary_prediction_arms=arms,versus_replay=comparisons)
    return dict(schema_version=1,status='verified_retrospective_diagnostic',dataset=DATASET,
        checkpoint_policies={'primary':'Job 425539 last epoch, not superseded','guard':'Job 426307 guarded checkpoint, secondary; not promoted'},
        inference_primary='official/joint_cap3000',cohorts=cohorts,per_class_benign_count_accounting=per_class,
        input_sha256=input_hashes,analysis_source_lf_sha256=hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest(),
        input_lf_sha256={registry.relative_to(repo).as_posix():REGISTRY_LF_SHA},
        source_registry_protected_raw_sha256=REGISTRY_SHA,
        aggregation_helper_lf_sha256=hashlib.sha256(Path(aggregate_tools.__file__).read_text(encoding='utf-8').encode()).hexdigest(),
        limitations=['Retrospective policy/arm diagnostics, not test-set tuning or a newly selected winner.',
          'Aggregate confusion matrices cannot identify the same rows switching predictions between arms.',
          'In-training learned head quality and cross-head calibration are not isolated by these inference ablations.',
          'Original baseline selection used seed 42 and guarded OFRA comparator; last-epoch replay comparison is a retrospective scope correction, not a newly preregistered trial.',
          'Exact two-sided minimum p is .0625 for five nonzero pairs and .125 for four; no significant-superiority claim.',
          'Equal data/epoch/exemplar contracts do not match total compute, loss, negative exposure or router memory.',
          'No new training, SHAP, ETG or cloud verification occurred.'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--policy-bindings',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps(analyze(a.repo,a.policy_bindings),indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
