"""Freeze eight-method NSL diagnostics and reference the already published two-method group."""
import argparse,copy,gzip,hashlib,json,math,re,shutil,statistics
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,str):
        v=re.sub(r'/home/user/[^/"\s]+','@@HPC_HOME@@',v);v=re.sub(r'/scr/user/[^/"\s]+','@@HPC_SCRATCH@@',v)
        v=re.sub(r'[A-Za-z]:[/\\][^\s"\n]*','@@LOCAL_PATH@@',v)
    return v
def summarize(values):return dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values),per_seed=values)
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.base
    assert not a.output.exists();source=b/'validation/hpc-493771-20261102';new=source/'protected-nsl-remaining-score-20261102/results';old=b/'admin/nsl-score-diagnosis-publication-v1'
    files={};allrows={};references={};counts=dict(new_cells=0,original_cm_exact=160,new_ks_checks=0,new_quantile_checks=0,new_posthoc_cm_checks=0);rowids=None
    for method in ('R3-A','R3-L','ar','replay_bs','freq','wacb','derpp','icarl','ecbrs','bic'):
        records=[];previous=method in ('R3-A','derpp')
        for seed in range(10,15):
            path=(old if previous else new)/f'{method}-seed{seed}.json';npz=path.with_suffix('.npz');r=read(path);z=np.load(npz);assert not r['software_only']
            assert not r['original_result_cm_discrepancies' if previous else 'original_cm_discrepancies']
            assert r['runtime']['torch']=='2.6.0+cu118' and r['runtime']['numpy']=='2.2.6' and r['runtime']['sklearn']=='1.7.2'
            ids=z['test_row_ids'];labels=z['test_labels'];assert len(labels)==22544 and sum(labels==0)==9711
            if rowids is None:rowids=(ids.copy(),labels.copy())
            else:assert np.array_equal(ids,rowids[0]) and np.array_equal(labels,rowids[1])
            h=z['holdout_scores' if previous else 'holdout_score'];s=z['test_scores' if previous else 'test_score'];assert len(h)==6735 and len(s)==22544
            assert abs(float(ks_2samp(h,s[labels==0]).statistic)-r['ks_statistic'])<1e-12
            for values,key in ((h,'holdout_quantiles'),(s[labels==0],'test_benign_quantiles')):
                ordered=sorted(float(x) for x in values)
                for q,value in zip(r['quantile_probabilities'],r[key]):
                    t=q*(len(ordered)-1);lo=math.floor(t);hi=math.ceil(t);actual=ordered[lo]+(ordered[hi]-ordered[lo])*(t-lo);assert abs(actual-value)<1e-10
                    if not previous:counts['new_quantile_checks']+=1
            if not previous:
                assert r['diagnostic_source_sha256']=='6cbb6eba765305e55b8fc0bd40b924f64625fbc66252d29935e99e7110c2d7a7'
                counts['new_cells']+=1;counts['new_ks_checks']+=1
            points={}
            for alpha,point in r['test_informed_operating_points'].items():
                m=point['confusion_matrix'];support=[sum(row) for row in m];fp=sum(m[0][1:]);bacc=sum(m[c][c]/support[c] for c in range(5))/5;fpr=fp/support[0];recall=1-sum(m[c][0] for c in range(1,5))/sum(support[1:])
                assert support==[int(sum(labels==c)) for c in range(5)] and fp<=math.floor(float(alpha)*support[0]);assert abs(bacc-point['balanced_accuracy'])<1e-12 and abs(fpr-point['measured_test_fpr'])<1e-12
                assert point['test_label_informed'] and not point['registered_result']
                points[alpha]=dict(balanced_accuracy=bacc,measured_test_fpr=fpr,attack_recall=recall,false_positives=fp,normal_support=support[0],confusion_matrix=m)
                if not previous:counts['new_posthoc_cm_checks']+=1
            records.append(dict(seed=seed,ks_statistic=r['ks_statistic'],holdout_quantiles=r['holdout_quantiles'],test_benign_quantiles=r['test_benign_quantiles'],posthoc=points))
            if previous:
                references[path.name]=dict(public_path='paper_aacil/results/section194_diagnostics/nsl-kdd/score_shift_r3a_derpp/'+path.name,json_sha256=sha(path),npz_sha256=sha(npz),recomputed_model=False)
            else:
                public=clean(copy.deepcopy(r));public['publication']=dict(original_raw_json_sha256=sha(path),original_npz_sha256=sha(npz),json_change_scope='Personal-path redaction and this publication receipt only; scientific numbers unchanged.',job_id=493771,diagnostic_published_source_commit='81ac672c4506adbf167ea6b05ca9f7d7c16eaa04')
                files[path.name]=(json.dumps(public,indent=2,allow_nan=False)+'\n').encode();files[npz.name]=npz.read_bytes()
        allrows[method]=dict(seeds=records,ks=summarize([v['ks_statistic'] for v in records]),posthoc={alpha:{metric:summarize([v['posthoc'][alpha][metric] for v in records]) for metric in ('balanced_accuracy','measured_test_fpr','attack_recall')} for alpha in ('0.02','0.05')})
    assert counts==dict(new_cells=40,original_cm_exact=160,new_ks_checks=40,new_quantile_checks=960,new_posthoc_cm_checks=80)
    for method,value in allrows.items():assert all(v['posthoc']['0.02']['false_positives']==194 for v in value['seeds'])
    comparisons={}
    for method,row in allrows.items():
        differences=[row['seeds'][j]['posthoc']['0.02']['balanced_accuracy']-allrows['R3-A']['seeds'][j]['posthoc']['0.02']['balanced_accuracy'] for j in range(5)]
        comparisons[method]=dict(bacc_difference_from_R3A=summarize(differences),meaning='Descriptive paired differences at identical194/9711 TEST benign false positives; TEST-label-informed, not registered inferential evidence.')
    assert allrows['ar']['posthoc']['0.02']['balanced_accuracy']['mean']>allrows['R3-A']['posthoc']['0.02']['balanced_accuracy']['mean']
    logs={}
    paths=[source/'protected-nsl-remaining-score-20261102/diagnosis.stdout.log',source/'protected-nsl-remaining-score-20261102/diagnosis.stderr.log',*sorted(source.glob('slurm-*'))]
    for stem in ('nsl-remaining-independent-20261102','nsl-remaining-software-sample-20261102','nsl-remaining-submit-20261102','nsl-remaining-static-preflight-20261102'):
        paths.extend(b/'logs'/f'{stem}.{stream}.log' for stream in ('stdout','stderr'))
    for path in paths:
        raw=path.read_bytes();redacted=clean(raw.decode()).encode();target='logs/'+path.name+'.gz';files[target]=gzip.compress(redacted,mtime=0);logs[target]=dict(original_sha256=sha(path),original_bytes=len(raw),redacted_sha256=hashlib.sha256(redacted).hexdigest())
    summary=dict(methods=allrows,descriptive_two_percent_comparisons=comparisons,counts=counts,previous_group_references=references,logs=logs,source_sha256={n:sha(b/'admin'/n) for n in ('diagnose-nsl-remaining-score-points.py','diagnose-nsl-remaining-score-points.sbatch','verify-nsl-remaining-score-points.py','export-nsl-remaining-completed.py')},job=dict(id=493771,state='COMPLETED',exit_code='0:0',wall_seconds=92,total_cpu_seconds=26.740,step_peak_rss_kib=1261872),scope='Unregistered TEST-label-informed operating-point diagnostic; no training, original threshold mutation or S1 replacement. Old493670 artifacts referenced, not republished as new results.',limits=['At2% all ten methods have exactly194/9711 false positives in each seed. Neural ar has higher mean BAcc than R3-A; R3-L is numerically close. No R3-A all-method clean win.', 'At5% R3-A actual mean4.6298% differs from the other methods4.9943%; do not call strictly matched actual FPR.', 'Original decision rules retained: gate-margin versus float32 prior shift. Absolute score units across methods are not comparable. KS does not prove feature causality.', 'No new inferential significance or registered S1 claim is made from TEST-exposed posthoc selection. Full training acceptance, WACB CSE score distributions, RQ1/RQ3, innovation, manuscript QA and backups remain separate.'])
    files['aggregate.json']=(json.dumps(summary,indent=2,allow_nan=False)+'\n').encode()
    lines=['# NSL remaining eight methods: final-checkpoint score diagnostics','', 'Share with caveats:40 newly evaluated final checkpoints /160 original default+1/2/5% confusion matrices matched exactly on Linux (job493771). No model training or original result/threshold mutation. Raw NPZ bytes are preserved; JSON scientific values are unchanged with personal paths redacted and raw SHA receipts added. The prior R3-A/DER++ group is referenced by immutable paths/hashes, not copied here or counted as new evaluation.','', '## Ten-method descriptive comparison','', 'Every method and seed at2% has194/9711=1.9977345279% TEST benign FPR. Selection uses TEST benign labels: this is posthoc, not registered S1. BAcc is a fraction in JSON and percent below. Sample SD is across five seeds.','', '|Method|2% BAcc mean ± sample SD (pp)|2% TEST FPR|5% BAcc mean|5% actual TEST FPR|','|---|---:|---:|---:|---:|']
    for method,row in allrows.items():
        two=row['posthoc']['0.02'];five=row['posthoc']['0.05'];lines.append(f"|{method}|{100*two['balanced_accuracy']['mean']:.4f} ± {100*two['balanced_accuracy']['sample_sd']:.4f}|{100*two['measured_test_fpr']['mean']:.7f}%|{100*five['balanced_accuracy']['mean']:.4f}|{100*five['measured_test_fpr']['mean']:.7f}%|")
    lines+=['', 'Neural ar51.2739% exceeds R3-A47.4398% at the identical2% actual FPR; R3-L47.4535% is numerically close to R3-A. Do not present R3-A as a matched-FPR all-method winner. R3-A has a lower actual FPR at the5% budget owing to ties; the5% comparison is not strictly FPR-matched. Default and original holdout-target results remain unchanged, including weak/negative results.','', 'The score rule remains the saved model\'s original gate margin or float32 prior-shift operation. Its numeric scale is not common across those rules. KS and within-model holdout/TEST quantile differences describe score transfer, not a causal feature mechanism. Raw arrays include logits/scores and rowIDs/labels, not input feature matrices.','', 'Independent validation: SciPy KS40, directly interpolated quantiles960, NumPy prediction + plain-list CM80 and budget/metric checks; completed verifier log included. Aggregate directly rechecks all50 JSON/NPZ receipts, all TEST rowIDs/labels and descriptive means/sample SD. This does not rerun prior two-method models. No inferential posthoc p-values are promoted to registered evidence.','', 'Cost: wall92s, TotalCPU26.740s, step MaxRSS1261872KiB. No hardware speedup claim. Full completed diagnostic/Slurm, software-sample, submission and static-check logs are redacted gzip with original byte/hash receipts. Export/publication streams created after this freeze stay local, not inside this archive. The effective config/source commit/seed/command and checkpoint/data bindings remain in each cell JSON.','', 'Primary scientific scope and remaining RQ1/RQ3/innovation/manuscript source visual QA/backup requirements are unchanged. C493535 pending is not a scientific result.']
    files['README.md']=('\n'.join(lines)+'\n').encode()
    for n,v in files.items():
        if n.endswith('.npz'):continue
        text=gzip.decompress(v) if n.endswith('.gz') else v
        assert not re.search(rb'/home/user/[A-Za-z0-9]|/scr/user/[A-Za-z0-9]|(?<![A-Za-z])[CE]:[/\\]|ghp_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY',text),n
    a.output.mkdir(parents=True)
    for n,v in files.items():dest=a.output/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(v)
    print(json.dumps(dict(event='NSL_REMAINING_ARCHIVE_COMPLETE',new_cells=40,ten_method_summary=True,artifact_count=len(files),old_group_republished=False)),flush=True)
if __name__=='__main__':main()
