"""Archive completed read-only NSL score diagnostics with independently checked statistics."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from statistics import mean,stdev
import numpy as np
from scipy.stats import ks_2samp

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
B=a.base.resolve(strict=True)
source=B/'validation/hpc-493670-20261102'
protected=source/'protected-nsl-score-diagnosis-20261102'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not a.output.exists()
records={}
novelty=read(B/'validation/nsl-categorical-shift-20261102.json')
for method in ('R3-A','derpp'):
    rows=[]
    for seed in range(10,15):
        stem=f'{method}-seed{seed}'
        r=read(protected/'results'/f'{stem}.json')
        z=np.load(protected/'results'/f'{stem}.npz')
        assert not r['software_only'] and not r['original_result_cm_discrepancies']
        assert r['runtime']['sklearn']=='1.7.2' and r['runtime']['torch']=='2.6.0+cu118'
        assert r['diagnostic_source_sha256']==sha(B/'admin/diagnose-nsl-score-shift.py')
        assert len(z['test_labels'])==22544 and int((z['test_labels']==0).sum())==9711 and len(z['holdout_row_ids'])==6735
        ks=ks_2samp(z['holdout_scores'],z['test_scores'][z['test_labels']==0]).statistic
        assert abs(ks-r['ks_statistic'])<1e-12
        for key,values in (('holdout',z['holdout_scores']),('test_benign',z['test_scores'][z['test_labels']==0])):
            ordered=np.sort(values.astype(np.float64))
            for probability,expected in zip(r['quantile_probabilities'],r[key+'_quantiles']):
                position=probability*(len(ordered)-1);lo=math.floor(position);hi=math.ceil(position)
                actual=float(ordered[lo]+(position-lo)*(ordered[hi]-ordered[lo]))
                assert abs(actual-expected)<1e-10
        for target,point in r['test_informed_operating_points'].items():
            cm=np.asarray(point['confusion_matrix'],dtype=np.int64)
            support=cm.sum(1)
            bacc=mean(int(cm[c,c])/int(support[c]) for c in range(5))
            fpr=(int(support[0])-int(cm[0,0]))/int(support[0])
            assert abs(bacc-point['balanced_accuracy'])<1e-12 and abs(fpr-point['measured_test_fpr'])<1e-12
            assert support[0]==9711 and int(support[0]-cm[0,0])<=math.floor(float(target)*9711)
            assert point['test_label_informed'] and not point['registered_result']
        alarm=set(r['alarm_test_row_ids'])
        assert len(alarm)==len(r['alarm_test_row_ids']) and abs(len(alarm)/9711-r['original_onepct_test_fpr'])<1e-12
        category_rates={}
        for field,counts in r['alarm_category_counts'].items():
            base=novelty['categorical_findings'][field]
            rates={}
            for category,number in counts.items():
                denominator=(base['test_normal_unseen_in_task0_encoder_count'] if category=='UNKNOWN_TO_TASK0_ENCODER'
                             else base['test_normal_categories'][category])
                assert 0<number<=denominator
                rates[category]=dict(false_alarms=number,test_normal_support=denominator,false_alarm_rate=number/denominator)
            assert sum(v['false_alarms'] for v in rates.values())==len(alarm)
            category_rates[field]=rates
        rawroot=B/'validation/hpc-20261102T1345/b/protected-b-v2'/(
            f'confirmation/nsl-kdd/R3-A/seed{seed}' if method=='R3-A' else f'nsl-kdd/derpp/seed{seed}')
        raw=read(rawroot/'cell/result.json')
        calibration=(raw['checkpoints'][-1]['calibration']['0.01'] if method=='R3-A' else
                     read(rawroot/'calibrated_stages.json')[-1]['calibration']['operating_points']['0.01'])
        holdout_fpr=calibration['actual_fpr'] if method=='R3-A' else calibration['calibration_fpr']
        assert r['source_result_sha256']==sha(rawroot/'cell/result.json')
        rows.append(dict(seed=seed,ks_statistic=float(ks),original_holdout_fpr=holdout_fpr,original_test_fpr=r['original_onepct_test_fpr'],
            original_target_tail_gap_pp=100*(r['original_onepct_test_fpr']-holdout_fpr),
            holdout_q99=r['holdout_quantiles'][9],test_benign_q99=r['test_benign_quantiles'][9],
            alarm_category_rates=category_rates,test_informed_operating_points=r['test_informed_operating_points']))
    records[method]=dict(seeds=rows,mean_ks=mean(r['ks_statistic'] for r in rows),
        mean_original_holdout_fpr=mean(r['original_holdout_fpr'] for r in rows),
        mean_original_test_fpr=mean(r['original_test_fpr'] for r in rows),
        posthoc={target:dict(bacc_mean=mean(r['test_informed_operating_points'][target]['balanced_accuracy'] for r in rows),
            bacc_sample_sd=stdev(r['test_informed_operating_points'][target]['balanced_accuracy'] for r in rows),
            measured_test_fpr_mean=mean(r['test_informed_operating_points'][target]['measured_test_fpr'] for r in rows)) for target in ('0.02','0.05')})
a.output.mkdir(parents=True,exist_ok=False)
for path in (protected/'results').iterdir():
    assert path.suffix in ('.json','.npz')
    shutil.copyfile(path,a.output/path.name)
summary=dict(job='493670',status='COMPLETED0:0',wall_seconds=21,total_cpu_seconds=11.499,slurm_step_maxrss_kib=1002552,
    methods=records,independent_ks_checks=10,independent_quantile_checks=240,posthoc_cm_checks=20,
    source_predictions_reconciled=dict(final_checkpoints=10,default_and_original_calibrated_cm=40,exact=True),
    limits=['Only R3-A and DER++ are covered; eight other NSL methods remain pending.',
        'Posthoc thresholds use TEST benign labels, are not registered, and replace no original thresholds/results.',
        'At-most2% and5% budgets: ties can make achieved FPR lower. R3-A5% mean actual4.6298% differs from DER++4.9943%; do not call these strictly equal.',
        'Score units differ between methods. Compare quantile shifts only within method; KS and tail gaps do not establish a causal mechanism.',
        'The full B training/memory/source acceptance and broader research completion are separate.'])
(a.output/'aggregate.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
text='''# NSL-KDD score-shift diagnostic: R3-A and DER++

Read-only job493670 completed in21s. All40 final-stage default/original1/2/5% confusion matrices matched the original Linux results exactly. The10 NPZ files preserve Normal holdout row IDs, TEST row IDs/labels and raw logits/scores. Original checkpoints and thresholds are unchanged. Source/config/runtime and checkpoint/source hashes are in each cell JSON. Source implementations are in ncd/hpc/diagnose-nsl-score-shift.py and its batch script.

This is an unregistered, post-hoc diagnostic, not a new S1 test. Only these two methods are covered; the other eight common-FPR curves remain pending. The2% and5% re-thresholding uses TEST benign labels. Ties mean an at-most budget, not guaranteed equality: at2% both methods achieve1.9977345% in each seed; at5% R3-A averages4.6298013% and DER++4.9943363%. Mean BAcc is47.4398322/42.6339204% at2%, and68.7948533/62.9648631% at the respective5% budgets. Do not present the latter as strictly matched FPR, or either comparison as a registered clean win over all methods.

Within each model, TEST benign upper-tail scores differ from the permanent TRAIN holdout. R3-A's99th percentile shifts from negative holdout margins to positive TEST margins in all five seeds; its mean test FPR at the original1% holdout target is6.0673463%, versus DER++1.4354855%. Mean KS statistics are0.1057207 and0.0877441, respectively, but R3-A is not higher in every seed. Score units differ (joint gate margin versus prior-shift crossing tau); absolute quantiles across methods are not comparable. These observations diagnose score-distribution transfer, not a proven causal feature mechanism.

Alarm category counts must be read with their benign denominators in aggregate.json, not as prevalence-adjusted risk by count alone. The separate categorical_novelty.json check shows no TEST benign protocol/service category unseen by the Task0 encoder and one unknown flag row; it does not prove that any particular row alarms. Raw CSV labels, model weights, data preprocessing and registered results were not changed.

Independent archive checks use SciPy's two-sample KS, direct sorted interpolation for240 quantiles, and independent per-class CM arithmetic for20 TEST-informed points. Original Linux score extraction, independent archive math, completed training acceptance and overall project completion remain distinct claims. Full completed diagnostic/Slurm stdout and stderr are retained as redacted gzip logs. Export/publish logs created after this archive stay local rather than being misrepresented as part of the frozen group.
'''
(a.output/'README.md').write_text(text,encoding='utf-8')
logroot=a.output/'logs';logroot.mkdir()
for path in [protected/'diagnosis.stdout.log',protected/'diagnosis.stderr.log',
             source/'slurm-nsl-score-shift-493670.out',source/'slurm-nsl-score-shift-493670.err']:
    raw=path.read_bytes().decode('utf-8')
    redacted=re.sub(r'/home/user/[^/"\s]+','@@HPC_HOME@@',raw)
    redacted=re.sub(r'/scr/user/[^/"\s]+','@@HPC_SCRATCH@@',redacted)
    (logroot/(path.name+'.gz')).write_bytes(gzip.compress(redacted.encode(),mtime=0))
for path in a.output.rglob('*'):
    if not path.is_file() or path.suffix=='.npz':continue
    raw=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
    assert not re.search(rb'/home/user/[A-Za-z0-9]|/scr/user/[A-Za-z0-9]|(?<![A-Za-z])[CE]:[/\\]|ghp_[A-Za-z0-9]{20}|-----BEGIN [A-Z ]*PRIVATE KEY',raw),path.name
print(json.dumps(dict(event='NSL_SCORE_ARCHIVE_COMPLETE',files=len([p for p in a.output.rglob('*') if p.is_file()]),methods=list(records),source_models_fitted=False)),flush=True)
