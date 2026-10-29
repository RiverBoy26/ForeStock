#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nonton v7.4: full historical v7 ensemble + leak-safe rolling META + conditional quantity.

Command:
  python -u nonton_v74.py --raw-zip hakaton.zip --out results_v74 --threads 8

Important: May 25 holdout is report-only and was studied in earlier iterations.
No predictions/labels dated after 2026-07-26 are used in learning.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
try:
    from . import nonton_v7 as base
except ImportError:
    import nonton_v7 as base

# Nonoverlapping 9-week test horizons (we deliberately leave gaps between them).
DEV_FOLDS = [
    '2025-01-27', '2025-04-07', '2025-06-16', '2025-08-25',
    '2025-11-03', '2026-01-12', '2026-03-23',
]
HOLDOUT='2026-05-25'
TARGET_KEYS=['week','warehouse_name','article']
METRICS=['SMAPE','WAPE','RMSE','Bias','RMSLE']
METRIC_WEIGHTS={'SMAPE':.50,'WAPE':.20,'RMSE':.15,'Bias':.10,'RMSLE':.05}
PROB_THRESHOLDS=[.55,.575,.60,.625]
VOL_BLEND=[0.0,.15,.30,.45]
ACT_MIN=[0,7]

P_PARAMS={
    'objective':'binary','learning_rate':.04,'num_leaves':15,
    'min_data_in_leaf':300,'feature_fraction':.9,'lambda_l2':10,
    'verbosity':-1,'seed':42,'deterministic':True,'force_col_wise':True,
}
Q_PARAMS={
    'objective':'regression_l1','learning_rate':.04,'num_leaves':15,
    'min_data_in_leaf':250,'feature_fraction':.85,'lambda_l2':15,
    'verbosity':-1,'seed':43,'deterministic':True,'force_col_wise':True,
}


def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-zip',type=Path,default=Path('hakaton.zip'))
    p.add_argument('--out',type=Path,default=Path('results_v74'))
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--origins',type=int,default=0, help='0 = all valid rolling origins (maximum historical training); 36 = lower-memory')
    p.add_argument('--rounds',type=int,default=240)
    p.add_argument('--groups',type=int,default=24)
    p.add_argument('--meta-rounds',type=int,default=180)
    p.add_argument('--quantity-rounds',type=int,default=180)
    p.add_argument('--skip-base',action='store_true',help='Reuse complete results_v74/base if already trained')
    p.add_argument('--preflight',action='store_true',help='Check dataset and python environment only')
    a=p.parse_args()
    if min(a.threads,a.rounds,a.meta_rounds,a.quantity_rounds)<1 or a.groups<0 or a.origins<0:
        p.error('Invalid positive training limits')
    return a


def save_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def validate_temporal_folds():
    prev_end=None
    for s in DEV_FOLDS+[HOLDOUT]:
        date=pd.Timestamp(s)
        if date.dayofweek!=0:raise ValueError('Fold not Monday: '+s)
        if prev_end is not None and date<=prev_end:raise ValueError('Overlapping folds: '+s)
        prev_end=date+pd.Timedelta(weeks=9)-pd.Timedelta(days=1)
    if pd.Timestamp(HOLDOUT)+pd.Timedelta(weeks=9)-pd.Timedelta(days=1)>base.CUTOFF:
        raise ValueError('Holdout extends into restricted period')


def run_base(a):
    output=a.out/'base'
    required=[output/'summary.json',output/'final_raw_components.npz',
              output/'holdout_predictions.csv.gz',output/'submission_v7_selected.csv']
    required += [output/f'dev_predictions_{s}.csv.gz' for s in DEV_FOLDS]
    manifest=output/'experiment_config.json'
    intended={'folds':DEV_FOLDS,'holdout':HOLDOUT,'mode':'both','max_origins':a.origins,
              'rounds':a.rounds,'max_group_models':a.groups,'threads':a.threads}
    if manifest.is_file():
        previous=json.loads(manifest.read_text(encoding='utf-8'))
        if previous!=intended:raise RuntimeError('Different base training configuration found. Use NEW output folder. Existing: '+str(previous))
    if all(x.is_file() and x.stat().st_size>0 for x in required):
        old=json.loads((output/'summary.json').read_text(encoding='utf-8'))
        prev=old.get('arguments',{})
        expected={'max_origins':a.origins,'rounds':a.rounds,'max_group_models':a.groups,'threads':a.threads}
        mismatches={k:(prev.get(k),v) for k,v in expected.items() if prev.get(k)!=v}
        if old.get('folds')!=DEV_FOLDS or old.get('holdout')!=HOLDOUT or old.get('mode')!='both' or mismatches:
            raise RuntimeError('Existing base outputs were trained with DIFFERENT parameters. Use a NEW -Out directory, or matching arguments. Mismatches: '+str(mismatches))
        print('BASE: reusing complete training results with matching parameters',output,flush=True)
        return
    if a.skip_base:
        raise RuntimeError('Cannot --skip-base: missing required files: '+str([str(x) for x in required if not x.is_file()]))
    output.mkdir(parents=True,exist_ok=True)
    save_json(manifest,intended)
    cmd=[sys.executable,'-u',str(Path(__file__).with_name('nonton_v7.py')),
         '--raw-zip',str(a.raw_zip.resolve()),'--out',str(output.resolve()),
         '--mode','both','--max-origins',str(a.origins), '--rounds',str(a.rounds),
         '--max-group-models',str(a.groups),'--threads',str(a.threads),
         '--half-life','26','--folds',*DEV_FOLDS,'--holdout',HOLDOUT,
         '--scales','1.0','--thresholds','1.0','--save-models','--resume-folds']
    print('BASE TRAIN COMMAND:', ' '.join(cmd),flush=True)
    (a.out/'base_command.txt').write_text(' '.join(cmd),encoding='utf-8')
    proc=subprocess.run(cmd,check=False)
    if proc.returncode:raise RuntimeError(f'Base v7 FAILED: exit code {proc.returncode}; fix error before META')
    missing=[str(x) for x in required if not x.is_file()]
    if missing:raise RuntimeError('Base process exited 0 but did not create: '+str(missing))


def _features(d,oi,frame):
    """One training row per pair x horizon; all historical facts STRICTLY before origin oi."""
    n=d.n_pairs
    if oi<52 or oi>base.N_WEEKS:raise ValueError('Out-of-range origin')
    h=d.qty[:oi]
    rep=lambda x:np.repeat(np.asarray(x,dtype=np.float32),9)
    raw=frame['hybrid_raw'].to_numpy(dtype=np.float32)
    g=frame['v7_global_raw'].to_numpy(dtype=np.float32)
    m=frame['m5lite_raw'].to_numpy(dtype=np.float32)
    x={
        'raw_full':raw, 'raw_global':g,'raw_m5':m,
        'disagreement':np.abs(g-m),'global_minus_full':g-raw,
        'raw_log':np.log1p(np.maximum(raw,0)),
        'horizon':np.tile(np.arange(1,10,dtype=np.float32),n),
        'warehouse':rep(d.wh_ids),'category1':rep(d.cat1_ids),
        'category2':rep(d.cat2_ids),
        'last_stock':rep(np.nan_to_num(d.stock_last[oi-1],nan=-999)),
        'last_sale_days':rep(d.sale_days[oi-1]),
    }
    for w in (4,13,26,52):
        hist=h[-w:]
        x[f'sales_sum_{w}']=rep(hist.sum(axis=0))
        x[f'active_weeks_{w}']=rep((hist>0).sum(axis=0))
    x['last_qty_1']=rep(h[-1]);x['last_qty_4']=rep(h[-4])
    u=h[-4:].mean(axis=0)
    v=h[-13:].mean(axis=0)
    z=h[-26:].mean(axis=0)
    x['trend_4_13']=rep((u+.1)/(v+.1))
    x['trend_13_26']=rep((v+.1)/(z+.1))
    x['weeks_since_sale']=rep(np.where((h>0).any(axis=0),np.argmax((h>0)[::-1],axis=0),-1))
    # No future stock, sales, discounts or channel shares whatsoever.
    return pd.DataFrame(x,dtype='float32')


def load_fold(d,path,date):
    df=pd.read_csv(path)
    expected=d.n_pairs*9
    if len(df)!=expected or df[TARGET_KEYS].duplicated().any():
        raise ValueError('Invalid fold grid '+str(path))
    oi=base.N_WEEKS if date is None else base.origin_idx(d.weeks,date)
    exp_weeks=pd.date_range(date if date is not None else base.FINAL,periods=9,freq='7D').strftime('%Y-%m-%d').to_numpy()
    if not np.array_equal(df.week.astype(str).to_numpy(),np.tile(exp_weeks,d.n_pairs)):
        raise ValueError('Fold week/order mismatch '+str(path))
    if not np.array_equal(df.warehouse_name.astype(str).to_numpy(),np.repeat(d.names_wh,9)) or not np.array_equal(df.article.astype(str).to_numpy(),np.repeat(d.names_article,9)):
        raise ValueError('Fold pair/order mismatch '+str(path))
    if date is not None:
        y=d.qty[oi:oi+9].T.ravel().astype(np.float32)
        if not np.allclose(df.actual.to_numpy(dtype=np.float32),y,atol=.00001):
            raise ValueError('Fold actual inconsistent with original data '+date)
    else:y=None
    required={'hybrid_raw','v7_global_raw','m5lite_raw'}
    if not required.issubset(df.columns):raise ValueError('Missing raw components '+str(required-set(df.columns)))
    X=_features(d,oi,df)
    return {'X':X,'y':y,'raw':df['hybrid_raw'].to_numpy(dtype=np.float32)}


def fit_meta(items,threads,prounds,qrounds):
    X=pd.concat([z['X'] for z in items],ignore_index=True)
    y=np.concatenate([z['y'] for z in items]).astype(np.float32)
    if len(np.unique(y>0))!=2:raise ValueError('Both sales classes are required')
    p=lgb.train({**P_PARAMS,'num_threads':threads},lgb.Dataset(X,label=(y>0).astype(np.int8)),num_boost_round=prounds)
    positive=y>0
    q=lgb.train({**Q_PARAMS,'num_threads':threads},lgb.Dataset(X.loc[positive],label=np.log1p(y[positive])),num_boost_round=qrounds)
    del X,y
    gc.collect()
    return p,q


def predict_meta(pair,item):
    p,q=pair
    prob=np.clip(p.predict(item['X'],num_threads=1),0,1)
    conditional=np.maximum(np.expm1(q.predict(item['X'],num_threads=1)),0)
    return prob,conditional


def candidates():
    yield ('BASE_FULL', None)
    for threshold in PROB_THRESHOLDS:
        for alpha in VOL_BLEND:
            for amin in ACT_MIN:
                key=f'META_P{threshold:.3f}_V{alpha:.2f}_A{amin}'
                yield (key, (threshold,alpha,amin))


def apply(raw, X,prob,q,config):
    if config is None:
        return base.postprocess(raw,1,1)
    threshold,blend,amin=config
    original=np.maximum(np.asarray(raw,dtype=np.float64),0.)
    ratio=np.clip((q+.2)/(np.maximum(original,1.)+.2),.65,1.6)
    allowed=(X['active_weeks_13'].to_numpy()>=amin)
    adjust=np.where(allowed,1+blend*(ratio-1),1.)
    pred=np.where(prob>=threshold,np.maximum(original*adjust,1.),0.)
    return np.maximum(pred,0.)


def report(y,p):
    return {k:float(v) for k,v in base.metric(y,p).items()}


def select(dev_rows,folds):
    df=pd.DataFrame(dev_rows)
    df.to_csv(folds,index=False,float_format='%.9g')
    agg=[]
    for name,g in df.groupby('candidate',sort=False):
        worst_smape=float(g['delta_smape'].max())
        score=float(g['weighted_ratio'].mean())
        score+=max(0,worst_smape-.75)*.01  # penalize large SMAPE drawdown vs FULL
        score+=float(g['weighted_ratio'].std(ddof=0))*.03
        rec={'candidate':name,'score_to_minimize':score,
             'mean_relative_weighted_score':float(g['weighted_ratio'].mean()),
             'smape_worst_delta_pp':worst_smape,'n_forward_folds':len(g)}
        for k in METRICS:rec['mean_'+k]=float(g[k].mean())
        agg.append(rec)
    ranked=pd.DataFrame(agg).sort_values(['score_to_minimize','mean_SMAPE','candidate']).reset_index(drop=True)
    return ranked


def build_meta(a):
    d=base.load_raw(a.raw_zip)
    fold=[]
    bp=a.out/'base'
    for dt in DEV_FOLDS:
        f=load_fold(d,bp/f'dev_predictions_{dt}.csv.gz',dt)
        fold.append(f)
        print('READ DEV',dt,'rows',len(f['X']),'actual_sum',round(float(f['y'].sum()),2),flush=True)
    hi=load_fold(d,bp/'holdout_predictions.csv.gz',HOLDOUT)
    print('Holdout: REPORT ONLY. Not used for fitting, threshold selection, or model choice.',flush=True)
    forecast_raw=np.load(bp/'final_raw_components.npz')
    rawframe=pd.DataFrame({k:forecast_raw[k] for k in ('hybrid_raw','v7_global_raw','m5lite_raw')})
    future={'X':_features(d,base.N_WEEKS,rawframe), 'raw':rawframe.hybrid_raw.to_numpy(dtype=np.float32)}
    keys=pd.read_csv(bp/'submission_v7_selected.csv',usecols=TARGET_KEYS)
    if len(keys)!=d.n_pairs*9 or keys.duplicated().any():raise ValueError('Invalid final grid')
    del d,forecast_raw,rawframe
    gc.collect()
    # Forward validation: first two folds warm-up. Every evaluation trained ONLY on earlier OOFs.
    configs=dict(candidates())
    vals=[]
    for i in range(2,len(fold)):
        dt=DEV_FOLDS[i]
        print('META FORWARD',dt,'training prior folds',i,flush=True)
        pm=fit_meta(fold[:i],a.threads,a.meta_rounds,a.quantity_rounds)
        pp,qp=predict_meta(pm,fold[i])
        baseline=report(fold[i]['y'],apply(fold[i]['raw'],fold[i]['X'],pp,qp,None))
        for name,config in configs.items():
            pred=apply(fold[i]['raw'],fold[i]['X'],pp,qp,config)
            m=report(fold[i]['y'],pred)
            weighted=sum(METRIC_WEIGHTS[k]*min(m[k]/max(baseline[k],1e-6),10) for k in METRICS)
            vals.append({'fold':dt,'n_prior_folds':i,'candidate':name,
                         'weighted_ratio':float(weighted), 'delta_smape':m['SMAPE']-baseline['SMAPE'],
                         **{k:m[k] for k in METRICS},
                         **{'BASE_'+k:baseline[k] for k in METRICS},
                         'pred_sum':m['predicted_sum'], 'signed_bias':m['signed_global_bias']})
        del pm,pp,qp
        gc.collect()
    ranked=select(vals,a.out/'v74_forward_validation.csv')
    ranked.to_csv(a.out/'v74_dev_ranking.csv',index=False,float_format='%.9g')
    winner=str(ranked.iloc[0]['candidate'])
    selection=configs[winner]
    print('CHOSEN from 5 forward folds ONLY:',winner,'score',ranked.iloc[0]['score_to_minimize'],flush=True)
    # Final train on 7 dev windows; don't use holdout y anywhere in fitting.
    pm=fit_meta(fold,a.threads,a.meta_rounds,a.quantity_rounds)
    p_model,q_model=pm
    p_model.save_model(str(a.out/'meta_sale_probability.txt'))
    q_model.save_model(str(a.out/'meta_conditional_quantity.txt'))
    fprob,fqty=predict_meta(pm,future)
    hprob,hqty=predict_meta(pm,hi)
    # Holdout validation is a diagnostic, not an independent test (already inspected in past sessions).
    hout=[]
    for name in ['BASE_FULL',winner,'META_P0.575_V0.00_A0','META_P0.575_V0.30_A7',
                 'META_P0.600_V0.00_A0']:
        config=configs[name]
        m=report(hi['y'],apply(hi['raw'],hi['X'],hprob,hqty,config))
        hout.append({'candidate':name,**m})
    pd.DataFrame(hout).to_csv(a.out/'v74_holdout_report_only.csv',index=False,float_format='%.9g')
    # Best candidate always plus safe baseline, formatted like original competition grid.
    outputs=[]
    for name,config,filename in [('winner',selection,'submission_v74_selected.csv'),
                                 ('baseline',None,'submission_v7_FULL_control.csv')]:
        pred=apply(future['raw'],future['X'],fprob,fqty,config)
        out=keys.copy()
        out['quantity']=pred
        validate_submission(out)
        out.to_csv(a.out/filename,index=False,float_format='%.7g')
        meta={'name':name,'candidate':winner if name=='winner' else 'BASE_FULL',
              'filename':filename,'rows':len(out),'predicted_total':float(pred.sum()),
              'zero_pct':float(100*np.mean(pred==0))}
        outputs.append(meta)
        print('EXPORT',meta,flush=True)
    summary={'model':'Nonton v7.4 META GATE + CONDITIONAL QUANTITY',
             'cutoff':'2026-07-26','final_target':'2026-07-27 through 2026-09-27',
             'dev_folds':DEV_FOLDS,'dev_forward_folds':DEV_FOLDS[2:],
             'holdout':HOLDOUT,'holdout_used_for_selection':False,
             'base_training':{'max_origins':a.origins,'rounds':a.rounds,'groups':a.groups},
             'chosen':winner, 'selection':selection,
             'score_warning':'Local relative weighted metric ratios, NOT organizer rank-score',
             'holdout_warning':'May 25 holdout has repeatedly been inspected, it is REPORT ONLY not an untouched benchmark',
             'leaderboard_top5':ranked.head(5).to_dict('records'),
             'outputs':outputs,
             'no_hidden_labels_warning':'Hidden contest outcomes cannot be computed from supplied historical data'}
    save_json(a.out/'summary_v74.json',summary)
    print('DONE: ',(a.out/'submission_v74_selected.csv').resolve(),flush=True)


def validate_submission(df):
    if len(df)!=79_101:raise ValueError(f'Expected 79,101 rows; got {len(df)}')
    if df[TARGET_KEYS].duplicated().any():raise ValueError('Duplicate keys')
    weeks=pd.to_datetime(df.week,errors='raise')
    expected=list(pd.date_range('2026-07-27',periods=9,freq='7D'))
    if sorted(weeks.unique())!=expected:
        raise ValueError('Incorrect weeks')
    pred=pd.to_numeric(df.quantity,errors='coerce').to_numpy()
    if not np.isfinite(pred).all() or (pred<0).any():raise ValueError('Invalid quantities')
    if df.warehouse_name.nunique()!=11 or df.article.nunique()!=799:
        raise ValueError('Missing warehouse/article universe')
    if not df.groupby(['warehouse_name','article'],observed=True).size().eq(9).all():
        raise ValueError('Missing week(s) for some warehouse/article pairs')


def main():
    a=args()
    validate_temporal_folds()
    if not a.raw_zip.exists():raise FileNotFoundError(f'{a.raw_zip} missing: place original hakaton.zip alongside script')
    a.out.mkdir(parents=True,exist_ok=True)
    print('NONTON 7.4 | Python',sys.version.split()[0],'LightGBM',lgb.__version__,'threads',a.threads,flush=True)
    print('Seven NONOVERLAPPING historical development horizons:',DEV_FOLDS,flush=True)
    if a.preflight:
        d=base.load_raw(a.raw_zip)
        assert len(d.warehouses)==11 and len(d.articles)==799
        assert d.n_pairs==8789
        print('PREFLIGHT PASSED: original data accepted; future sales excluded; 8789 pairs',flush=True)
        return
    run_base(a)
    build_meta(a)

if __name__=='__main__':
    main()
