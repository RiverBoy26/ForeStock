"""Train 3-head v8 on REAL v7.4 out-of-fold predictions; no future labels.

Usage: python -m app.v8_train --rounds 180 --threads 8
Requires training/oof/7-fold files and frozen snapshot; does NOT require raw hakaton.zip.
"""
from __future__ import annotations
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from . import nonton_v7 as base, nonton_v74 as old, v8
from .snapshot import load_dataset
ROOT=Path(__file__).resolve().parents[1]
FOLDS=old.DEV_FOLDS
WEIGHTS=old.METRIC_WEIGHTS

def item_for_date(d,folder,date):
    path=folder/f'dev_predictions_{date}.csv.gz'
    if not path.is_file():raise FileNotFoundError(path)
    f=old.load_fold(d,path,date)
    frame=pd.read_csv(path,usecols=['hybrid_raw','v7_global_raw','m5lite_raw'])
    f['X']=v8.enrich(d,base.origin_idx(d.weeks,date),frame)
    if f['X'].isna().any().any() or not np.isfinite(f['X'].to_numpy()).all():
        raise ValueError('Nonfinite v8 features '+date)
    return f

def ref_report(d,folder):
    f=old.load_fold(d,folder/'holdout_predictions.csv.gz',old.HOLDOUT)
    frame=pd.read_csv(folder/'holdout_predictions.csv.gz',usecols=['hybrid_raw','v7_global_raw','m5lite_raw'])
    f['X']=v8.enrich(d,base.origin_idx(d.weeks,old.HOLDOUT),frame)
    return f

def metrics_dict(y,p):return {k:float(v) for k,v in base.metric(y,p).items()}

def score(allrows,dev_legacy):
    df=pd.DataFrame(allrows)
    df.to_csv(ROOT/'docs'/'v8_forward_validation.csv',index=False,float_format='%.8g')
    legacy=pd.read_csv(dev_legacy)
    legacy=legacy[legacy.candidate=='META_P0.550_V0.15_A7']
    if len(legacy)!=5:raise ValueError('v7.4 forward reference missing')
    reference={str(row.fold):row for row in legacy.itertuples()}
    out=[]
    for name,g in df.groupby('candidate',sort=False):
        if len(g)!=5:raise ValueError('incomplete candidate: '+name)
        rel=[]
        delta=[]
        for row in g.itertuples():
            r=reference[str(row.fold)]
            rel.append(sum(WEIGHTS[k]*float(getattr(row,k))/max(float(getattr(r,k)),1e-8) for k in old.METRICS))
            delta.append(float(row.SMAPE)-float(r.SMAPE))
        out.append({'candidate':name,'n_folds':len(g),
            'weighted_ratio_to_v74':float(np.mean(rel)),
            'stability_penalty':float(np.std(rel)*.03),
            'worst_smape_change_pp':float(max(delta)),
            'mean_smape_change_pp':float(np.mean(delta)),
            **{f'mean_{k}':float(g[k].mean()) for k in old.METRICS}})
    table=pd.DataFrame(out)
    table['selection_score']=table.weighted_ratio_to_v74+table.stability_penalty+np.maximum(0,table.worst_smape_change_pp-.75)*.01
    table=table.sort_values('selection_score').reset_index(drop=True)
    table.to_csv(ROOT/'docs'/'v8_candidate_ranking.csv',index=False,float_format='%.8g')
    return table

def train(threads=8,rounds=180):
    t0=time.time()
    d=load_dataset(ROOT/'training'/'history_snapshot.npz')
    folder=ROOT/'training'/'oof'
    items=[]
    for date in FOLDS:
        z=item_for_date(d,folder,date)
        items.append(z)
        print('LOADED',date,len(z['y']),len(z['X'].columns),'features',flush=True)
    hold=ref_report(d,folder)
    print('V8: forward tests only; holdout untouched until selection.',flush=True)
    configs=dict(v8.config_grid())
    allrows=[]
    for i in range(2,len(FOLDS)):
        date=FOLDS[i]
        print('FORWARD',date,'training folds',FOLDS[:i],flush=True)
        heads=v8.fit(items[:i],threads,rounds)
        output=v8.predict(heads,items[i])
        for name,cfg in configs.items():
            p=v8.apply(items[i],output,cfg)
            m=metrics_dict(items[i]['y'],p)
            allrows.append({'fold':date,'candidate':name,**m})
        del heads,output
        gc.collect()
    ranking=score(allrows, ROOT/'docs'/'v74_forward_validation.csv')
    chosen=ranking.iloc[0].to_dict()
    print('BEST v8',chosen['candidate'],'ratio to v7.4',chosen['weighted_ratio_to_v74'],flush=True)
    print('Refitting all 7 OOF folds',flush=True)
    models=v8.fit(items,threads,rounds)
    for name,model in zip(('v8_presence','v8_positive_quantity','v8_direct_tweedie'),models):
        model.save_model(str(ROOT/'models'/f'{name}.txt'))
    raw_np=np.load(ROOT/'training'/'final_raw_components.npz')
    frame=pd.DataFrame({k:raw_np[k] for k in ('hybrid_raw','v7_global_raw','m5lite_raw')})
    future={'X':v8.enrich(d,base.N_WEEKS,frame), 'raw':frame.hybrid_raw.to_numpy(dtype=np.float32)}
    foutputs=v8.predict(models,future)
    # All policies frozen BEFORE holdout examination.
    names=[str(chosen['candidate'])]
    # Pick distinct candidate designs using forward-only ranking.
    for name in ranking.candidate:
        if name=='V74_BASE':continue
        c=configs[name]
        if name not in names and (not names or c.get('kind')!=configs[names[0]].get('kind') or len(names)>1):
            names.append(name)
        if len(names)>=3:break
    if len(names)<3:
        names=list(dict.fromkeys(names+[n for n in configs if n!='V74_BASE']))[:3]
    variants={'v8_selected':names[0],'v8_alternative':names[1],'v8_experimental':names[2]}
    keyframe=pd.read_csv(ROOT/'training'/'keys.csv',dtype={'article':'str','warehouse_name':'str','week':'str'})
    if len(keyframe)!=79101 or keyframe.duplicated().any().any():raise ValueError('Bad frozen future keys')
    predictions=[]
    for alias,name in variants.items():
        p=v8.apply(future,foutputs,configs[name])
        out=keyframe.copy();out['quantity']=p
        old.validate_submission(out)
        out.to_csv(ROOT/'forecasts'/f'{alias}.csv',index=False,float_format='%.7g')
        predictions.append({'variant':alias,'candidate':name,'quantity_sum':float(p.sum()),'zero_pct':float((p==0).mean()*100)})
    # v7.4 selected reference frozen from previous run, for apples-to-apples production fallback.
    holdoutputs=v8.predict(models,hold)
    hrows=[]
    for name in ['V74_BASE',*names]:
        p=v8.apply(hold,holdoutputs,configs[name])
        hrows.append({'candidate':name,**metrics_dict(hold['y'],p)})
    pd.DataFrame(hrows).to_csv(ROOT/'docs'/'v8_holdout_report_only.csv',index=False)
    improve=float(chosen['weighted_ratio_to_v74'])<1
    summary={'architecture':'v8 hierarchical probabilistic 3-head + zero-aware decisions',
        'strict_cutoff':'2026-07-26','future_window':'2026-07-27 to 2026-09-27',
        'historical_folds':FOLDS,'forward_folds':FOLDS[2:],
        'chosen':chosen,'passes_v74_forward_metric_ratio':improve,
        'reference_v74_forward_metrics':{'SMAPE':60.6855248834,'WAPE':50.7632609974,'RMSE':2.6866081414,'Bias':76.9856148506,'RMSLE':.5212536214},
        'holdout_note':'May 25 holdout is repeatedly inspected. It is REPORT ONLY and did not tune/select variants.',
        'official_score_note':'Weighted relative five-metric loss is a local proxy, NOT official organizer rank score.',
        'training_rounds':rounds,'threads':threads,
        'candidates_compared':len(configs),'artifacts':predictions,
        'elapsed_seconds':round(time.time()-t0,1)}
    (ROOT/'docs'/'summary_v8.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'models'/'v8_schema.json').write_text(json.dumps({'feature_columns':list(future['X'].columns),
        'variants':variants,'configurations':{n:configs[n] for n in names},
        'n_train_folds':7,'cutoff':'2026-07-26'},indent=2,ensure_ascii=False),encoding='utf-8')
    print('COMPLETED',json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--rounds',type=int,default=180)
    a=p.parse_args()
    if a.threads<=0 or a.rounds<20:p.error('Invalid values')
    train(a.threads,a.rounds)
