"""Reproduce Section 12.7/15.3 adjudication from registered raw results."""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'ncd/results/crr/section15/registered'
SEEDS = (1, 2, 3, 4, 42)
METHODS = ('replay','icarl','replay_lwf','kd_perturbed','rcl_tab','global_ir','oldpair_kd','crr')
def mean(x): return sum(x)/len(x)
def p_less(d):
    observed = mean(d)
    return sum(mean([s*v for s,v in zip(signs,d)]) <= observed+1e-15 for signs in itertools.product((-1,1),repeat=5))/32
def load(path): return json.loads(path.read_text(encoding='utf-8'))
def main():
    verdict = {'criteria':'Section 12.7 with 14.4 and 15.3 amendments','test':'exact one-sided paired sign-flip, 32 assignments, alpha=0.05, unadjusted; no pooled dataset test','datasets':{},'sources':{}}
    lines = ['# CRR Section 15 results — 2026-10-13','','## Decision','','CRR does not meet all registered criteria. The local final hijack gate failed, so Section 15.3 replaces criterion 1. This is a completed bounded experiment, not evidence of substantive innovation or publication readiness.','','## Provenance and scope','','Necessity: DICC 485405/485406, source `bfde946a44ea7838f07399cf14c7570a40df7dca`, 80 rows per dataset. Real sequence: first-complete local RTX 4060 Ti results, source `1560401b806371c117b080cbf5ca0bd4c30ab179`, 40 rows per dataset. Seeds are 1,2,3,4,42; lambda selection used only 5,6. Local real and HPC necessity use different devices: comparisons use within-run Replay-normalized training cost; absolute cross-device cost is not compared. HPC NSL 485498 is replication; HPC UNSW 485499 is still running and is not required for this adjudication.','','All raw per-cell JSON/CSV and PI drift rows are under `results/crr/section15/registered`. Necessity protected manifests were checked; selected transferred merged results and local files are hashed in `ARTIFACT_HASHES.json`. No data files or checkpoints are published.','','## Statistical rule','','Exact one-sided paired sign-flip test over five seeds (32 assignments), alpha 0.05, following the existing Section 12 implementation; p-values are unadjusted. Effect-size and numerical thresholds must also pass. k=3 is primary; k=2 is separately reported and retained in the registered k>=2 requirement. Means and sample standard deviations are descriptive, not confidence intervals.','','## Local final hijack gate','','The pre-existing `HIJACK_VERDICT.json` reports `passed=false`, threshold 0.15, and `fallback_section_15_3_active=true`. No hijack-prevention claim is supported.']
    for tag,dataset in [('nsl','NSL-KDD'),('unsw','UNSW-NB15')]:
        np=BASE/f'{tag}-necessity/necessity_result.json'; rp=BASE/f'{tag}-real/real_result.json'
        n=load(np); r=load(rp)
        assert len(n['rows'])==80 and len(r['rows'])==40 and not r['smoke']
        assert n['git_commit']=='bfde946a44ea7838f07399cf14c7570a40df7dca'
        assert r['git_commit']=={'nsl':'1560401b806371c117b080cbf5ca0bd4c30ab179','unsw':'e5817f70e84bb86e301580965fef3d633d30c6db'}[tag]
        nl={(x['method'],int(x['seed']),float(x['k'])):x for x in n['rows']}
        rl={(x['method'],int(x['seed'])):x for x in r['rows']}
        assert set(nl)=={(m,s,k) for m in METHODS for s in SEEDS for k in (2.,3.)}
        assert set(rl)=={(m,s) for m in METHODS for s in SEEDS}
        assert all(x['dataset']==dataset for x in n['rows']+r['rows'])
        assert all(x['effective_config']['epochs_per_task']==8 and not x['effective_config'].get('smoke_rows_per_class') for x in [load(p) for p in (BASE/f'{tag}-real/shards').glob('*/real_result.json')])
        def rv(m,f): return [float(rl[m,s][f]) for s in SEEDS]
        def nv(m,f,k): return [float(nl[m,s,k][f]) for s in SEEDS]
        drift={m:mean(rv(m,'old_pair_pi_drift')) for m in METHODS}
        f1={m:mean(rv(m,'macro_f1')) for m in METHODS}
        reduction=1-drift['crr']/drift['replay'] if drift['replay']>0 else None
        dp=p_less([a-b for a,b in zip(rv('crr','old_pair_pi_drift'),rv('replay','old_pair_pi_drift'))])
        c1= reduction is not None and reduction>=.30 and dp<.05 and drift['crr']<=1.2*drift['rcl_tab']
        necessity={}
        for k in (3.,2.):
            means={m:mean(nv(m,'cd_balanced_accuracy',k)) for m in METHODS}
            ps={m:p_less([b-a for a,b in zip(nv('crr','cd_balanced_accuracy',k),nv(m,'cd_balanced_accuracy',k))]) for m in ('rcl_tab','global_ir')}
            passed=all(means['crr']>means[m] and ps[m]<.05 for m in ps) and means['crr']>=means['replay']-.02
            necessity[str(int(k))]={'means':means,'p_crr_above':ps,'crr_minus_replay':means['crr']-means['replay'],'passed':passed}
        c2=all(x['passed'] for x in necessity.values())
        c3=f1['crr']>=f1['replay']-.02 and f1['crr']>=f1['rcl_tab']
        realcost=mean(rv('crr','cost_vs_replay'))
        ncost={str(int(k)):mean(nv('crr','cost_vs_replay',k)) for k in (2.,3.)}
        c4=realcost<=3 and all(v<=3 for v in ncost.values())
        out={'criterion1':{'passed':c1,'drift_means':drift,'reduction_vs_replay':reduction,'paired_p':dp,'ratio_vs_rcl':drift['crr']/drift['rcl_tab']},'criterion2':{'passed':c2,'by_k':necessity},'criterion3':{'passed':c3,'macro_f1_means':f1,'crr_minus_replay':f1['crr']-f1['replay'],'crr_minus_rcl':f1['crr']-f1['rcl_tab']},'criterion4':{'passed':c4,'mean_real_cost_vs_replay':realcost,'mean_necessity_cost_vs_replay':ncost},'overall_passed':all((c1,c2,c3,c4))}
        verdict['datasets'][dataset]=out
        lines += ['',f'## {dataset}','','| Method | Necessity BA k=3 | Necessity BA k=2 | Real Macro-F1 | Old-pair PI drift | Real training / Replay |','|---|---:|---:|---:|---:|---:|']
        for m in METHODS:
            lines += [f'| {m} | {necessity["3"]["means"][m]:.6f} | {necessity["2"]["means"][m]:.6f} | {f1[m]:.6f} | {drift[m]:.6f} | {mean(rv(m,"cost_vs_replay")):.4f} |']
        lines += ['',f'Criterion 1: {"PASS" if c1 else "FAIL"}; drift reduction {reduction:.2%}, paired p={dp:.5f}, CRR/RCL-tab={out["criterion1"]["ratio_vs_rcl"]:.4f}.',f'Criterion 2: {"PASS" if c2 else "FAIL"}.']
        for k,detail in necessity.items():
            lines += [f'- k={k}: {"PASS" if detail["passed"] else "FAIL"}; CRR minus Replay {100*detail["crr_minus_replay"]:.3f} percentage points; p above RCL-tab={detail["p_crr_above"]["rcl_tab"]:.5f}, Global-IR={detail["p_crr_above"]["global_ir"]:.5f}.']
        lines += ['',f'Criterion 3: {"PASS" if c3 else "FAIL"}; CRR minus Replay {100*(f1["crr"]-f1["replay"]):.3f} points, minus RCL-tab {100*(f1["crr"]-f1["rcl_tab"]):.3f} points.',f'Criterion 4: {"PASS" if c4 else "FAIL"}; real training ratio {realcost:.4f}, necessity k2/k3 ratios {ncost["2"]:.4f}/{ncost["3"]:.4f}.','',f'Dataset overall: {"PASS" if out["overall_passed"] else "FAIL"}.']
    verdict['overall_passed']=all(x['overall_passed'] for x in verdict['datasets'].values())
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(BASE.rglob('*')) if p.is_file() and p.name!='ARTIFACT_HASHES.json'}
    (BASE/'ARTIFACT_HASHES.json').write_text(json.dumps(hashes,indent=2,sort_keys=True),encoding='utf-8')
    verdict['sources']=hashes
    (ROOT/'ncd/CRR_SECTION15_ADJUDICATION_20261013.json').write_text(json.dumps(verdict,indent=2,sort_keys=True),encoding='utf-8')
    lines += ['','## Interpretation and next decision','','The registered evidence rejects the claim that CRR satisfies the combined stability, plasticity, predictive and cost criteria in these two settings. Retain the negative result and raw evidence. Further mechanism changes require a new explicit scientific plan; do not retune this registered comparison after inspecting the test results. The broader locked research questions remain open.']
    lines += ['','## Additional registered endpoints','','Training ratios above exclude PI evaluation; its cost is separately reported below. Forgetting uses the runner definition: mean historical-best minus final recall over all classes, including final-task classes.','','| Dataset | Method | Forgetting | Training seconds | PI evaluation seconds |','|---|---|---:|---:|---:|']
    for tag,dataset in [('nsl','NSL-KDD'),('unsw','UNSW-NB15')]:
        rows = load(BASE/f'{tag}-real/real_result.json')['rows']
        for m in METHODS:
            selected = [x for x in rows if x['method']==m]
            lines += [f'| {dataset} | {m} | {mean([x["forgetting"] for x in selected]):.6f} | {mean([x["training_seconds"] for x in selected]):.3f} | {mean([x["pi_evaluation_seconds"] for x in selected]):.3f} |']
    lines += ['','## Final hijack attempt raw outcomes','','| Dataset | Seed | rho | Target | Marker recall drop | Decorrelation recall drop |','|---|---:|---:|---|---:|---:|']
    hijack = load(ROOT/'ncd/results/crr/section15/local_hijack_final/hijack_result.json')
    for x in hijack['rows']:
        lines += [f'| {x["dataset"]} | {x["seed"]} | {x["rho"]} | {x["target_name"]} | {x["marker_recall_drop"]:.6f} | {x["marker_decorrelation_recall_drop"]:.6f} |']
    report = '\n'.join(lines).replace('source `1560401b806371c117b080cbf5ca0bd4c30ab179`, 40 rows per dataset','NSL source `1560401b806371c117b080cbf5ca0bd4c30ab179`, UNSW source `e5817f70e84bb86e301580965fef3d633d30c6db`, 40 rows per dataset (computational files identical between these commits)').replace('Means and sample standard deviations are descriptive, not confidence intervals.','Reported means are descriptive, not confidence intervals.')
    (ROOT/'ncd/CRR_SECTION15_REPORT_20261013.md').write_text(report+'\n',encoding='utf-8')
    print(json.dumps({d:{'criteria':[x[f'criterion{i}']['passed'] for i in range(1,5)],'overall':x['overall_passed']} for d,x in verdict['datasets'].items()}))
if __name__=='__main__': main()
