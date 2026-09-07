"""Recover existing five-seed EG grids; no model inference or threshold fitting."""
import argparse
from pathlib import Path
import hashlib, json, math, statistics

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_grid(a):
    records=a['threshold_sensitivity']
    if len(records)!=64:
        raise ValueError('expected 64 registered settings')
    keys=set()
    for g in records:
        key=(g['k'],g['jaccard_threshold'],g['allowed_recall_drop'])
        if key in keys:
            raise ValueError('duplicate threshold setting')
        keys.add(key)
        rows=[r for r in a['transition_rows'] if r['delta_recall'] > -g['allowed_recall_drop']]
        events=sum(r[f"jaccard_top{g['k']}"] < g['jaccard_threshold'] for r in rows)
        if (events,len(rows))!=(g['events'],g['eligible_transitions']):
            raise ValueError('stored grid differs from transitions')
        rate=events/len(rows) if rows else None
        if rate is None:
            if g['rate'] is not None:
                raise ValueError('empty denominator must be null')
        elif not math.isclose(rate,g['rate'],rel_tol=0,abs_tol=1e-12):
            raise ValueError('rate mismatch')
    expected={(k,j,r) for k in [5,10,15,20] for j in [.5,.6,.7,.8] for r in [0,.02,.05,.1]}
    if keys!=expected:
        raise ValueError('registered grid changed')
    return records

def run(root,out):
    seeds=[1,2,3,4,42]
    exported=[]; bindings=[]; coverage=[]
    for seed in seeds:
        folder=root/f'seed-{seed}'
        checksum={}
        for line in (folder/'SHA256SUMS').read_text().splitlines():
            h,name=line.split(None,1); checksum[name.strip()]=h
        file=folder/'analysis.json'
        if digest(file)!=checksum['./expected-gradients-etg/analysis.json']:
            raise ValueError(f'seed {seed} input hash mismatch')
        a=json.loads(file.read_text())
        if a['seed']!=seed or a['dataset']!='malaya-network-gt':
            raise ValueError('wrong input identity')
        for g in verify_grid(a): exported.append({'seed':seed,**g})
        bindings.append({'seed':seed,'analysis_sha256':digest(file),'downloaded_checksum_registry_sha256':digest(folder/'SHA256SUMS')})
        coverage.append({'seed':seed,'transitions':len(a['transition_rows']),
             'finite_cosine':sum(isinstance(r.get('cosine_similarity'),(int,float)) and math.isfinite(r['cosine_similarity']) for r in a['transition_rows']),
             'finite_kendall_tau_b':sum(isinstance(r.get('kendall_tau_b'),(int,float)) and math.isfinite(r['kendall_tau_b']) for r in a['transition_rows'])})
    agg=[]
    for k,j,d in [(x['k'],x['jaccard_threshold'],x['allowed_recall_drop']) for x in exported if x['seed']==1]:
        rows=[r for r in exported if (r['k'],r['jaccard_threshold'],r['allowed_recall_drop'])==(k,j,d)]
        rates=[r['rate'] for r in rows if r['rate'] is not None]
        events=sum(r['events'] for r in rows); denom=sum(r['eligible_transitions'] for r in rows)
        agg.append({'k':k,'jaccard_threshold':j,'allowed_recall_drop':d,'events':events,'eligible_transitions':denom,
                    'mean_seed_rate':statistics.mean(rates) if rates else None,'valid_seed_rates':len(rates),
                    'pooled_rate':events/denom if denom else None})
    primary=next(r for r in agg if (r['k'],r['jaccard_threshold'],r['allowed_recall_drop'])==(15,.7,.05))
    if primary['events']!=61 or primary['eligible_transitions']!=82:
        raise ValueError('primary does not match published aggregate')
    result={'status':'existing_five_seed_eg_grid_recovered_and_recomputed','source_job':434747,'seeds':seeds,
        'method':'expected_gradients','settings_per_seed':64,'verified_rows':len(exported),
        'primary':primary,'grid':agg,'seed_rows':exported,'alternative_metric_coverage':coverage,'input_bindings':bindings,
        'scope':'Existing drift counts recomputed from saved transition values, not new attributions, full ETG state-grid replay, remote verification or independent raw-array validation',
        'replication_unit':'training seed on one fixed split; threshold rows are not independent experiments'}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({'status':result['status'],'verified_rows':len(exported),'primary':primary,'metric_coverage':coverage}))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)
