"""Freeze validated WACB score arrays as exact-byte parts and scoped science outputs."""
import argparse,copy,gzip,hashlib,json,re,statistics
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--owner',required=True);a=p.parse_args();b=a.base;assert not a.output.exists()
    root=b/'validation/hpc-493836-20261102';results=root/'protected-cse-wacb-score-guarded-20261102/results';verified=json.loads((b/'validation/cse-wacb-score-independent-20261102.json').read_text());assert verified['checks']['cells']==5 and verified['checks']['test_confusion_matrices']==20
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,str):
            v=re.sub(r'/home/user/[^/"\s]+','@@HPC_HOME@@',v);v=re.sub(r'/scr/user/[^/"\s]+','@@HPC_SCRATCH@@',v);v=re.sub(r'[A-Za-z]:[/\\][^\s"\n]*','@@LOCAL_PATH@@',v);v=v.replace(a.owner,'@@OWNER@@')
        return v
    a.output.mkdir(parents=True);raw_arrays={};new_records={};max_part=45*1024**2
    for path in sorted(results.glob('*.npz')):
        parts=[];whole=hashlib.sha256();size=0
        with path.open('rb') as f:
            index=0
            while True:
                data=f.read(max_part)
                if not data:break
                index+=1;relative='arrays/'+path.name+f'.part{index:03d}' if path.stat().st_size>max_part else path.name;dest=a.output/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);size+=len(data);whole.update(data);parts.append(dict(path=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        with np.load(path,allow_pickle=False) as z:headers={key:dict(shape=list(z[key].shape),dtype=z[key].dtype.str) for key in z.files}
        raw_arrays[path.name]=dict(bytes=size,sha256=whole.hexdigest(),parts=parts,arrays=headers)
        assert whole.hexdigest()==sha(path) and size==path.stat().st_size
        print(json.dumps(dict(event='WACB_RAW_ARRAY_PARTS_FROZEN',name=path.name,parts=len(parts),bytes=size)),flush=True)
    assert len(raw_arrays)==6
    for seed in range(10,15):
        path=results/f'wacb-seed{seed}.json';r=json.loads(path.read_text());v=verified['seeds'][seed-10];assert sha(path)==v['raw_json_sha256'] and raw_arrays[path.with_suffix('.npz').name]['sha256']==v['raw_npz_sha256']
        public=clean(copy.deepcopy(r));public['publication']=dict(original_raw_json_sha256=sha(path),raw_npz_sha256=v['raw_npz_sha256'],diagnostic_actual_forward_batch=4096,original_source_config_eval_batch=2048,guard_source_sha256='2f47907d5910366d49b3270ba7eb768b204055347fc0e3e7440da1c4cfa0c5f4',diagnostic_source_published_commit='b14ed8889c9a0f31596d2117b4279e2e606e1217',guard_published_commit='f1de16ae276a35744e88a0295d9ce99f41de2153',change_scope='Personal-path/account redaction and this metadata only; scientific numbers unchanged. CM/holdout-FP exactness is not original-logit bitexactness.')
        (a.output/path.name).write_text(json.dumps(public,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n');new_records[str(seed)]=public
    log_receipts={};paths=list((root/'protected-cse-wacb-score-guarded-20261102').glob('*.log'))+list(root.glob('slurm-*'))
    for stem in ('cse-wacb-guarded-software-sample-20261102','cse-wacb-guard-fixtures-20261102','cse-wacb-guarded-static-20261102','cse-wacb-guarded-submit-20261102','cse-wacb-score-independent-20261102'):paths.extend(b/'logs'/f'{stem}.{stream}.log' for stream in ('stdout','stderr'))
    assert len(paths)==14
    for path in paths:
        raw=path.read_bytes();redacted=clean(raw.decode()).encode();relative='logs/'+path.name+'.gz';dest=a.output/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(gzip.compress(redacted,mtime=0));log_receipts[relative]=dict(original_bytes=len(raw),original_sha256=hashlib.sha256(raw).hexdigest(),redacted_sha256=hashlib.sha256(redacted).hexdigest())
    summary=dict(independent=verified,raw_arrays=raw_arrays,log_receipts=log_receipts,source_sha256={name:sha(b/'admin'/name) for name in ('diagnose-cse-wacb-score-distributions.py','guard-cse-wacb-evaluation.py','verify-cse-wacb-score-distributions.py','restore-cse-wacb-score-arrays.py','export-cse-wacb-score-distributions.py')},job=dict(id=493836,state='COMPLETED',exit_code='0:0',wall_seconds=87,total_cpu_seconds=87.914,step_peak_rss_kib=6823588,stderr_scope='Nonempty expected import-time diagnostics; do not call stderr empty.'),prior_attempts=[dict(id=493796,state='CANCELLED by owner',wall_seconds=1031,total_cpu_seconds=2.513,scientific_cells=0),dict(id=493824,state='COMPLETED startup only',wall_seconds=35,total_cpu_seconds=9.427,scientific_cells=0)],workflow_hpc_wall_seconds=1153,workflow_total_cpu_seconds=99.854,scope='Read-only final-model score association. No training, original threshold/result changes, causal mechanism or research completion claim. Prior failed/startup evidence is referenced at wacb_startup_trace_493824, not republished.',parts_policy=dict(part_bytes_max=max_part,original_npz_bytes_unchanged=True,github_official_source='https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github',reason='Original NPZ exceeds100MiB; exact-byte parts stay below50MiB. Reassemble and verify before np.load.'))
    (a.output/'aggregate.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    lines=['# CSE2018 WACB final score distributions (job493836)','', 'Share with caveats. Five final checkpoints were read without training or original result/threshold changes. Independent raw-array checks reproduce20 TEST confusion matrices,15 holdout FP counts, all saved prior-crossing scores,5 KS statistics,540 quantiles, all CDF knots, per-class recall and Normal absorption. All recomputed holdout FP deltas are retained in aggregate.json.','', '|Seed|1% BAcc %|1% TEST FPR %|Benign holdout/TEST KS|Holdout q99|TEST benign q99|','|---|---:|---:|---:|---:|---:|']
    for row in verified['seeds']:v=row['original_views']['0.01'];lines.append(f"|{row['seed']}|{100*v['balanced_accuracy']:.5f}|{100*v['measured_test_fpr']:.7f}|{row['ks_statistic']:.8f}|{row['holdout_q99']:.7f}|{row['test_benign_q99']:.7f}|")
    lines+=['', 'Benign KS ranges0.00678361–0.01040315 while measured TEST FPR remains approximately1%. BAcc ranges21.40–72.65%, mean42.19362% with sampleSD22.33438pp. These observations do not support explaining the large spread as a large benign FPR change or a demonstrated bisection solver failure. Existing stored calibration searches were monotone and budget-bracketed; independent current holdout FP counts agree. Per-class score quantiles and Normal absorption describe different saved-model decision geometry, not its feature-level cause or a proven memory/training mechanism.','', 'Scores are ideal real-arithmetic prior-crossing tau values within each model. Original predictions still use float32 prior shift. Absolute score scale across independently trained seeds is not a common unit. Diagnostic forward batches4096 differ from original config2048: all required CMs/FP counts agree, but original per-row logits were not available for a bitexact comparison. The raw NPZ includes logits/scores/rowIDs/labels, not input features.','', '## Restore exact raw arrays','', 'Each original seed NPZ is134–136MB. [GitHub documents a100MiB regular-file block and50MiB warning](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Therefore raw bytes are split into45MiB parts without re-encoding. Each part and the full original SHA/size are in the scientific aggregate; shared row/labels are copied unchanged. Use the published restore tool with a fresh destination, for example:', '', '    python ncd/hpc/restore-cse-wacb-score-arrays.py --archive <this-directory> --output <fresh-directory>', '', 'The tool checks every part and full SHA, then loads every array with allow_pickle=False and verifies shapes/dtypes. Original NPZ files are retained unchanged in protected local storage. No checkpoint/data features are redistributed.','', '## Costs and remaining scope','', 'Successful job493836:87s wall,87.914s TotalCPU,step MaxRSS6823588KiB. The earlier cancelled attempt consumed1031s/2.513CPU seconds, and startup-only493824 consumed35s/9.427CPU seconds. Sequential HPC diagnostic workflow total1153s wall and99.854CPU seconds is reported separately from the successful run; do not call87s the whole cost or claim speedup. Old slowdown did not recur; its original cause remains unproved. Failed/startup streams are already published under ../wacb_startup_trace_493824, not counted as new model results.','', 'New completed logs include import-time stderr, which is not empty and must not be described as stderr0. Per-cell effective_config/source commit/seed/command/checkpoint/data/holdout bindings and all arrays remain inspectable. Later export/restore/publication validation streams stay local, not inside this frozen archive. Original registered comparisons, manuscript/RNG footnote/current-source visual QA, RQ1/RQ3, substantive innovation and backup requirements remain separate and open.']
    (a.output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    count=0
    for path in a.output.rglob('*'):
        if not path.is_file():continue
        count+=1;assert path.stat().st_size<=max_part
        if path.suffix not in ('.json','.md','.gz'):continue
        data=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes();assert not re.search(rb'/home/user/[A-Za-z0-9]|/scr/user/[A-Za-z0-9]|(?<![A-Za-z])[CE]:[/\\]|ghp_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY',data),path.name
    print(json.dumps(dict(event='WACB_SCORE_ARCHIVE_FROZEN',artifact_count=count,raw_npz_files=6,seed_npz_parts=15,source_models_evaluated=False)),flush=True)
if __name__=='__main__':main()
