"""B2 AR-MLP: official CI/CII splits, fixed default gate and DER++/WA family.

The B2 adaptation uses the official balanced 10% replay budget, including benign
anchoring, instead of the B1 fixed-per-class buffer and external benign refresh.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('upstream', 'data', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--scenario', type=int, choices=(1, 2), required=True)
    p.add_argument('--seed', type=int, choices=(42, 1, 2, 3, 4), required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(a.upstream.resolve()), str(root)]
    os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    import numpy as np
    import torch
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from src.data.ci_builder import build_class_incremental_scenario
    import src
    src.__path__.append(str(root / 'src'))
    from ncd.model import MLP
    from ncd.aacil_compact import CompactFamily, train_compact_family
    from ncd.aacil_retention import apply_weight_alignment
    from ncd.run_aacil_retention import train_gate, logits_numpy
    from ncd.default_decision import predict_gate_family
    from ncd.train_cil import seed_everything
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = torch.device('cpu')
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    arrays = [np.load(a.data / (s+'.npy')) for s in ('train', 'val', 'test')]
    names = ['Benign', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest',
             'DoS slowloris', 'FTP-Patator', 'Heartbleed', 'SSH-Patator']
    stream = build_class_incremental_scenario(*arrays, all_classes=names,
                categorical_indices_file=str(a.data/'catfeaturelist.npy'),
                n_experiences=4, class_order=list(range(8)), scenario=a.scenario)
    del arrays
    cat = np.load(a.data/'catfeaturelist.npy').tolist()
    cat_scaler = StandardScaler().fit(stream[0].train_ds.features[:, cat])
    for exp in stream:
        for ds in (exp.train_ds, exp.val_ds, exp.test_ds):
            ds.features[:, cat] = cat_scaler.transform(ds.features[:, cat])
            ds.features = ds.features.astype(np.float32)
    config = dict(hidden=128, epochs_per_task=8, min_steps_per_task=2000,
                  batch_size=256, eval_batch_size=2048, learning_rate=.001,
                  derpp_alpha=1., derpp_beta=1., deadline_iso=None)
    provenance = dict(section=151, method='AR-MLP', seed=a.seed, scenario=a.scenario,
        git_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        upstream_commit=subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip(),
        config=config, memory_percent=10, data_split_seed=42,
        command=f'python -I -B ncd/run_section151_ar.py --upstream UPSTREAM --data DATA --output OUTPUT --scenario {a.scenario} --seed {a.seed}',
        log_path='stdout.log; stderr.log',
        adaptations=['Official per-experience continuous scaling; categorical scalar columns standardized using task zero TRAIN only.',
                     'Official balanced 10% replay quotas and seeded sampling from seen TRAIN; benign stays in replay.',
                     'No external 50000-row benign refresh in CI; CII uses the current benign partition.',
                     'DER targets cached at each official buffer rebuild after WA; old pools revisited as in official ER.',
                     'Default decision: binary gate argmax then seen-family argmax; no FPR calibration.'],
        limitations=['B2 descriptive adaptation, not B1 or an independent validation of RQ1/RQ3.',
                     'Intransigence requires the paired official oracle; left null until that finishes.'])
    (output/'run.json').write_text(json.dumps(provenance,indent=2))
    seed_everything(a.seed)
    gate = MLP(78,128,2)
    family = CompactFamily(78,128,7,'aacil_kd_wa')
    mapping = {c:c-1 for c in range(1,8)}
    memory, benign = {}, np.empty((0,78),np.float32)
    seen = set()
    matrix_acc, matrix_f1 = np.zeros((4,4)), np.zeros((4,4))
    records = []
    started = time.time()
    for t, exp in enumerate(stream):
        old_seen = seen.copy()
        seen.update(exp.class_ids)
        new = sorted(seen-old_seen-{0})
        family.set_seen([c-1 for c in seen if c])
        x = exp.train_ds.features
        y = exp.train_ds.labels.astype(np.int64)
        oldxs = [m['train_x'] for m in memory.values()]
        oldys = [np.full(len(m['train_x']),c,np.int64) for c,m in memory.items()]
        gx = np.concatenate([x, benign, *oldxs])
        gy = np.concatenate([y, np.zeros(len(benign),np.int64), *oldys])
        attack = gy != 0
        print(f'TASK_START AR scenario={a.scenario} seed={a.seed} task={t} rows={len(gy)}',flush=True)
        task_seed = a.seed*10000+t
        seed_everything(task_seed)
        _, gs = train_gate(gate,gx,(gy!=0).astype(np.int64),np.zeros((len(gy),2),np.float32),
                           np.zeros(len(gy),bool),config,task_seed,device,0.)
        _, fs = train_compact_family(family,gx[attack],gy[attack]-1,memory,mapping,config,task_seed,device)
        wa = apply_weight_alignment(family,[c-1 for c in old_seen if c],[c-1 for c in new])
        # Same quotas/order/random.sample as official build_replay_buffer, without
        # constructing a Python tensor object for every available training row.
        pools = {c:np.concatenate([e.train_ds.features[e.train_ds.labels==c]
                                  for e in stream[:t+1] if c in e.class_ids]) for c in sorted(seen)}
        quota = max(5,int(sum(len(v) for v in pools.values())*.1)//len(seen))
        rng = random.Random(a.seed)
        memory = {}
        for c,pool in pools.items():
            ids = np.asarray(rng.sample(range(len(pool)),min(quota,len(pool))),dtype=np.int64)
            selected = pool[ids].copy()
            if c == 0:
                benign = selected
                continue
            logits = logits_numpy(family.student,selected,device,2048)
            memory[c] = dict(train_ids=ids,train_x=selected,f_logits=logits,
                             f_target_mask=np.broadcast_to(family.seen_mask.numpy(),logits.shape).copy())
        del pools,gx,gy
        truths,preds = [],[]
        gate.eval(); family.eval()
        for j in range(t+1):
            ds = stream[j].test_ds
            parts=[]
            with torch.inference_mode():
                for start in range(0,len(ds),2048):
                    bx=torch.from_numpy(ds.features[start:start+2048])
                    parts.append(predict_gate_family(gate(bx),family(bx),list(range(1,8)),sorted(seen),0).numpy())
            pred=np.concatenate(parts); truth=ds.labels.astype(np.int64)
            matrix_acc[t,j]=accuracy_score(truth,pred)
            matrix_f1[t,j]=f1_score(truth,pred,labels=stream[j].class_ids,average='macro',zero_division=0)
            truths.append(truth);preds.append(pred)
        truth,pred=np.concatenate(truths),np.concatenate(preds)
        normal=truth==0
        record=dict(task=t,accuracy=accuracy_score(truth,pred),
                    macro_f1=f1_score(truth,pred,labels=sorted(seen),average='macro',zero_division=0),
                    argmax_fpr=float((pred[normal]!=0).mean()),confusion_matrix=confusion_matrix(truth,pred,labels=list(range(8))).tolist(),
                    gate_schedule=gs,family_schedule=fs,weight_alignment=wa,
                    replay_rows=len(benign)+sum(len(m['train_x']) for m in memory.values()))
        records.append(record)
        torch.save(dict(gate=gate.state_dict(),family=family.state_dict(),memory=memory,benign=benign,
                        task=t,config=config,seed=a.seed),output/f'checkpoint_{t}.pt')
        (output/f'task_{t}.json').write_text(json.dumps(record,indent=2))
        print('TASK_DONE '+json.dumps({k:record[k] for k in ('task','accuracy','macro_f1','argmax_fpr')}),flush=True)
    result=dict(provenance,completed=True,elapsed_seconds=time.time()-started,checkpoints=records,
                accuracy_matrix=matrix_acc.tolist(),f1_matrix=matrix_f1.tolist(),intransigence=None)
    for key,mat in [('accuracy',matrix_acc),('macro_f1',matrix_f1)]:
        result['final_forgetting_'+key]=float(np.mean([mat[j:3,j].max()-mat[3,j] for j in range(3)]))
    (output/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
