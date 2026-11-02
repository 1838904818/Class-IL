"""Freeze completed same-score results with independent list-CM and statistical checks."""
import argparse,copy,gzip,hashlib,json,math,re,statistics
from pathlib import Path
from scipy.special import stdtr

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def trajectory(cms,seen):
    histories=[];points=[]
    for cm,classes in zip(cms,seen):
        assert all(isinstance(v,int) and v>=0 for row in cm for v in row)
        support=[sum(row) for row in cm]
        rec={c:cm[c][c]/support[c] for c in classes};histories.append(rec)
        points.append(dict(balanced_accuracy=sum(rec.values())/len(rec),attack_recall=1-sum(row[0] for row in cm[1:])/sum(support[1:]),measured_test_fpr=1-cm[0][0]/support[0]))
    forgotten=[]
    for c in seen[-1]:
        values=[h[c] for h in histories if c in h]
        if len(values)>1:forgotten.append(max(values[:-1])-values[-1])
    return dict(**points[-1],incremental_balanced_accuracy=sum(p['balanced_accuracy'] for p in points)/len(points),signed_best_prior_forgetting=sum(forgotten)/len(forgotten))
def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,str):
        value=re.sub(r'[A-Za-z]:[/\\][^\s"\n]*','@@LOCAL_PATH@@',value)
        value=re.sub(r'/(?:scr/user|home)/[^\s"\n]*','@@OWNED_PATH@@',value)
    return value
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--references',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.base
    root=b/'validation/hpc-493707-20261102/protected-same-score-validation-20261102';aggregate=json.loads((b/'validation/section344-same-score-aggregate-20261102.json').read_text())
    assert not a.output.exists();payload={};raws={};verified=0;stages=0
    for path in sorted((root/'results').rglob('same_score_calibration.json')):
        ds,method,seed,_=path.relative_to(root/'results').parts;r=json.loads(path.read_text());assert not r['software_only']
        original=b/'validation/section344-inputs-20261102'/ds/method/seed/'result.json'
        if ds=='cic-ids-2018' and method=='cer-bic-common' and seed=='seed14':original=b/'validation/hpc-493652-bic14-20261102/seed14/result.json'
        assert sha(original)==r['source_result_sha256'];o=json.loads(original.read_text());identity=o['identity']
        assert r['diagnostic_source_sha256']=='3fe63c8f404d5217a6c63108e7169132bab7b75751a03146d17038ecc856ca2a'
        assert r['reference_data_binding']==identity['data_binding'] and r['seed']==int(seed[4:])
        assert r['class_order']==identity['class_order'] and r['tasks']==identity['tasks']
        assert [s['checkpoint_sha256'] for s in r['stages']]==[c['checkpoint_sha256'] for c in o['checkpoints']]
        seen=[s['seen_classes'] for s in r['stages']];stages+=len(seen)
        for alpha in ('0.01','0.02','0.05'):
            for s in r['stages']:
                c=s['views'][alpha]['calibration'];assert c['comparison']=='strict >' and c['holdout_false_positives']<=c['maximum_false_positives']==math.floor(float(alpha)*c['holdout_count'])
            for rule in ('same_score_holdout','original_prior_shift'):
                cms=[s['views'][alpha]['metrics']['confusion_matrix'] for s in r['stages']] if rule=='same_score_holdout' else [s['operating_points'][alpha]['confusion_matrix'] for s in r['original_prior_shift_stages']]
                calculated=trajectory(cms,seen);expected=aggregate['section344'][ds][alpha]['rules'][rule][method]['per_seed'][int(seed[4:])-10]
                for key,v in calculated.items():assert abs(v-expected[key])<1e-12,(ds,method,seed,alpha,key)
        public=copy.deepcopy(r);public['publication']=dict(raw_same_score_sha256=sha(path),job_id=493707,published_diagnostic_source_commit='e4dfd4ee12ddd49a0c7eadf74c228a847d905a43',source_effective_config=identity['effective_config'],source_checkpoint_identity_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest(),effective_config_scope='Source checkpoint configuration; diagnostic score and holdout rule above. Paths redacted, original identity hash precedes redaction.',runtime_scope='New Linux baseline inference; original prior-shift and R3-A references retain published historical runtime.',elapsed_scope='Per-cell evaluation excludes dataset hashing, archive extraction and holdout prefetch; do not add to job walltime.',canonical_alias_scope='CSE icarl_seedN consumes the same scoped-NPY cells, not extra experiments.')
        payload[f'paper_aacil/results/section178_literature/{ds}/{method}_{seed}/same_score_calibration.json']=clean(public);raws[path.relative_to(root).as_posix()]=sha(path);verified+=1
    assert verified==60 and stages==260
    for ds,alphas in aggregate['section344'].items():
        for alpha,report in alphas.items():
            for rule,methods in report['rules'].items():
                ps=[];eps=[]
                for method,row in methods.items():
                    values=[v['balanced_accuracy'] for v in row['per_seed']]
                    for key,stats in row['aggregate'].items():
                        vals=[v[key] for v in row['per_seed']];mean=sum(vals)/5;sd=math.sqrt(sum((v-mean)**2 for v in vals)/4)
                        assert abs(mean-stats['mean'])<1e-12 and abs(sd-stats['sample_sd'])<1e-12
                    t=row['final_bacc_comparison'];d=t['paired_differences'];mean=sum(d)/5;sd=math.sqrt(sum((x-mean)**2 for x in d)/4)
                    pv=float(stdtr(4,-mean/(sd/math.sqrt(5)))) if sd else (0. if mean>0 else 1.)
                    exact=sum(sum(x*(1 if mask&(1<<i) else -1) for i,x in enumerate(d))/5>=mean-1e-12 for mask in range(32))/32
                    assert abs(pv-t['paired_t_one_sided_p'])<1e-12 and exact==t['exact_signflip_one_sided_p'];ps.append((method,pv));eps.append((method,exact))
                for values,key in ((ps,'paired_t_holm_p'),(eps,'exact_signflip_holm_p')):
                    ceiling=0.
                    for rank,(method,v) in enumerate(sorted(values,key=lambda x:x[1])):
                        ceiling=max(ceiling,min(1.,v*(4-rank)));assert abs(ceiling-methods[method]['final_bacc_comparison'][key])<1e-12
    prefix='paper_aacil/results/section178_literature/same-score-robustness-493707'
    aggregate['publication_scope']='Lead section347 registered CSE prior-rule 4/4 PASS reproduced; same-score is posthoc robustness, not a replacement. Overall research and historical-runtime homogenization are not closed.'
    payload[prefix+'/aggregate.json']=aggregate
    payload[prefix+'/verification.json']=dict(completed_cells=60,stages=260,independent_list_cm_all_rules=True,independent_student_t_cdf=True,independent_bitmask_signflip=True,independent_holm=True,raw_sha256=raws,job=dict(id=493707,state='COMPLETED',exit_code='0:0',wall_seconds=1686,total_cpu_seconds=3213.365,same_score_step_peak_rss_kib=7231468),export_source_sha256=sha(Path(__file__)))
    a.output.mkdir(parents=True)
    for target,value in payload.items():
        dest=a.output/target;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    logs=list(root.glob('*.log'))+list((b/'validation/hpc-493707-20261102').glob('slurm-*'))
    for path in logs:
        raw=path.read_bytes();dest=a.output/prefix/'logs'/(path.name+'.gz');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(gzip.compress(clean(raw.decode()).encode(),mtime=0))
    lines=['# Saved-checkpoint common-score robustness (job493707)','', '60 cells /260 checkpoints; no training or mutation of original thresholds/results. Registered primary remains prior-shift; new score calibration is a posthoc robustness check. Normal holdout is permanent, prediction uses strict score > tau at1/2/5% targets. TEST FPR need not equal the holdout target. R3-A references remain as published, not newly re-evaluated with all baselines.','', '|Dataset|Method|Original BAcc %|Same-score BAcc %|Same-score TEST FPR %|paired-t Holm p|','|---|---|---:|---:|---:|---:|']
    for ds,alphas in aggregate['section344'].items():
        for method,row in alphas['0.01']['rules']['same_score_holdout'].items():
            old=alphas['0.01']['rules']['original_prior_shift'][method];lines.append(f"|{ds}|{method}|{100*old['aggregate']['balanced_accuracy']['mean']:.4f}|{100*row['aggregate']['balanced_accuracy']['mean']:.4f}|{100*row['aggregate']['measured_test_fpr']['mean']:.4f}|{row['final_bacc_comparison']['paired_t_holm_p']:.5g}|")
    lines+=['','All comparisons use five paired seeds against as-published R3-A. Exact signflip is sensitivity only (minimum one-sided p1/32 cannot pass Holm4). Changes are not uniformly beneficial; no feature/mechanism causal attribution follows from changing the operating-point rule. Full2/5% results, attack recall, signed best-prior forgetting (including Normal), sample SD and individual seeds are in aggregate.json. Original-runtime CSE registered prior-rule4/4 paired-t/Holm PASS is reproduced; this is not overall research completion.','', 'Costs: wall1686s, TotalCPU3213.365s; same-score step1445s, CPU2799.911s, MaxRSS7231468KiB. Original BiC14 validation occupied a separate sequential233s step. Per-cell elapsed excludes archive/data hashing and holdout prefetch. No hardware speedup claim. Complete job logs are sanitized; raw evidence hashes are retained locally. Each new JSON retains its pre-publication raw SHA; only publication metadata and path redaction are added.']
    (a.output/prefix/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('SECTION344_60_CELLS_260_CP_CM_STATISTICS_INDEPENDENTLY_VERIFIED_AND_EXPORTED')
if __name__=='__main__':main()
