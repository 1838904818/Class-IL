"""Export a completed B2 cell and redacted full logs; never run inference."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cell',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--redact-root',type=Path,required=True)
    a=p.parse_args()
    result=json.loads((a.cell/'result.json').read_text())
    if not result.get('completed') or len(result['accuracy_matrix'])!=4:
        raise ValueError('Only completed four-experience cells can be exported')
    key=('er' if result['method']=='ER-Bal-TabT' else 'ar')+f"_s{result['scenario']}_seed{result['seed']}"
    dest=a.destination/key
    dest.mkdir(parents=True,exist_ok=False)
    logs={}
    for stream in ('stdout','stderr'):
        source=a.cell.parent/(a.cell.name+'.'+stream+'.log')
        raw=source.read_bytes()
        text=raw.decode('utf-8',errors='replace')
        root=str(a.redact_root.resolve())
        text=text.replace(root,'WORKSPACE').replace(root.replace('\\','/'),'WORKSPACE')
        # Tracebacks can mention installed package paths outside the workspace.
        text=re.sub(r'[A-Za-z]:[/\\]Users[/\\][^\s"\n]+','LOCAL_PATH',text)
        if re.search(r'github_pat_|ghp_[A-Za-z0-9]+|-----BEGIN .*PRIVATE KEY|sk-[A-Za-z0-9]{20}',text):
            raise ValueError('Credential-like pattern in log; do not publish')
        name=stream+'.log.gz'
        (dest/name).write_bytes(gzip.compress(text.encode('utf-8'),mtime=0))
        logs[stream]=dict(path=name,raw_sha256=hashlib.sha256(raw).hexdigest())
    result['log_path']=logs
    result['intransigence_status']='paired oracle pending' if result['method']=='AR-MLP' else 'paired official oracle complete'
    if result['method']=='ER-Bal-TabT':
        # Official main makes four cumulative oracle evaluations after the
        # continual learner's final cumulative report.
        final=result['evaluations'][-5]
        if final['rows']!=227723 or len(result['oracle_acc'])!=4:
            raise ValueError('Official final cumulative evaluation not identified')
        result['final_metrics']=dict(accuracy=result['overall_accuracy_matrix'][-1],
                                    macro_f1=result['overall_f1_matrix'][-1],
                                    argmax_fpr=final['argmax_fpr'])
    else:
        final=result['checkpoints'][-1]
        result['final_metrics']={k:final[k] for k in ('accuracy','macro_f1','argmax_fpr')}
    # Keep source-level limitations and adaptations in every delivered result.
    (dest/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(str(dest))


if __name__=='__main__':
    main()
