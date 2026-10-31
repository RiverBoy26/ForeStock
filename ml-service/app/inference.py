"""Reproducible inference of trained Nonton v7.4 WITHOUT retraining.

Replays 39 base LightGBM components and 2 META models from immutable historical snapshot.
Only the frozen contest horizon 2026-07-27 ... 2026-09-21 is supported.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import re
from functools import lru_cache
import numpy as np
import pandas as pd
import lightgbm as lgb
from . import nonton_v7 as base, nonton_v74 as meta
from .snapshot import load_dataset

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'forecasts'
VARIANTS={
    'selected':(.55,.15,7),
    'conservative':(.575,.0,0),
    'smape_experimental':(.55,.30,7),
    'baseline':None,
}

@lru_cache(maxsize=64)
def load_booster(path:str):
    return lgb.Booster(model_file=path)

def predict_components(d,model_dir:Path):
    files={p.stem:p for p in model_dir.glob('*.txt')}
    required={'global_tweedie','global_poisson','recursive_tweedie','recursive_poisson'}
    if not required.issubset(files):raise FileNotFoundError(f'Missing saved base models: {required-files.keys()}')
    X=base.make_features(d,base.N_WEEKS)
    expected=json.loads((model_dir/'schema.json').read_text(encoding='utf-8'))['features']
    if list(X.columns)!=expected:
        raise ValueError('Feature schema mismatch: model was trained on different columns')
    base_model=load_booster(str(files['global_tweedie']))
    g=base.pred_lgb(base_model,X)
    global_p=base.pred_lgb(load_booster(str(files['global_poisson'])),X)
    wh=g.copy()
    for name,p in files.items():
        if name.startswith('warehouse__'):
            wid=int(name.split('__')[1]); mask=X.warehouse_id.to_numpy()==wid
            if not mask.any():raise ValueError(f'Unrecognized warehouse model {name}')
            wh[mask]=base.pred_lgb(load_booster(str(p)),X.loc[mask])
    cat=wh.copy()
    for name,p in files.items():
        if name.startswith('whcat1__'):
            a,b=map(int,name.split('__')[1].split('_'))
            mask=(X.warehouse_id.to_numpy()==a)&(X.category1_id.to_numpy()==b)
            if not mask.any():raise ValueError(f'Unrecognized category model {name}')
            cat[mask]=base.pred_lgb(load_booster(str(p)),X.loc[mask])
    def recursive(kind):
        mdl=load_booster(str(files['recursive_'+kind]))
        hist=np.zeros((base.N_WEEKS+base.N_H,d.n_pairs),dtype=np.float32)
        hist[:base.N_WEEKS]=d.qty
        predicted=np.zeros((d.n_pairs,base.N_H),dtype=np.float32)
        for h in range(base.N_H):
            step=base.N_WEEKS+h
            x=base.make_features(d,step,qty_override=hist,cutoff_origin=base.N_WEEKS,recursive=True)
            p=base.pred_lgb(mdl,x)
            hist[step]=p
            predicted[:,h]=p
        return predicted.ravel()
    rt=recursive('tweedie');rp=recursive('poisson')
    m=np.mean(np.stack((g,global_p,wh,cat,rt,rp),axis=0),axis=0,dtype=np.float64).astype(np.float32)
    hybrid=(.5*g+.5*m).astype(np.float32)
    return {'v7_global_raw':g,'m5lite_raw':m,'hybrid_raw':hybrid}

def forecast_frame(d,pred):
    forecast_start=d.weeks[-1]+pd.Timedelta(days=7)
    weeks=(forecast_start+pd.to_timedelta(np.arange(9)*7,unit='D')).strftime('%Y-%m-%d').to_numpy()
    frame=pd.DataFrame({'week':np.tile(weeks,d.n_pairs),
            'warehouse_name':np.repeat(d.names_wh,9),
            'article':np.repeat(d.names_article,9),
            'quantity':np.asarray(pred,dtype=np.float64)})
    if len(frame)!=d.n_pairs*base.N_H or frame[['week','warehouse_name','article']].duplicated().any():
        raise ValueError('Invalid forecast grid')
    if not np.isfinite(frame.quantity).all() or (frame.quantity<0).any():
        raise ValueError('Invalid forecast quantities')
    return frame

def predict_dataset(d,model_dir:Path,meta_dir:Path,variant='selected'):
    if variant not in VARIANTS:raise ValueError(f'Unknown variant: {variant}')
    raw=predict_components(d,model_dir)
    item={'raw':raw['hybrid_raw'],'X':meta._features(d,base.N_WEEKS,pd.DataFrame(raw))}
    pmodel=load_booster(str(meta_dir/'meta_sale_probability.txt'))
    qmodel=load_booster(str(meta_dir/'meta_conditional_quantity.txt'))
    probability,conditional=meta.predict_meta((pmodel,qmodel),item)
    pred=meta.apply(item['raw'],item['X'],probability,conditional,VARIANTS[variant])
    return forecast_frame(d,pred)

def run(snapshot:Path,model_dir:Path,meta_dir:Path,output:Path,variants=tuple(VARIANTS)):
    d=load_dataset(snapshot)
    raw=predict_components(d,model_dir)
    item={'raw':raw['hybrid_raw'],'X':meta._features(d,base.N_WEEKS,pd.DataFrame(raw))}
    pmodel=load_booster(str(meta_dir/'meta_sale_probability.txt'))
    qmodel=load_booster(str(meta_dir/'meta_conditional_quantity.txt'))
    probability,conditional=meta.predict_meta((pmodel,qmodel),item)
    output.mkdir(parents=True,exist_ok=True)
    produced={}
    for variant in variants:
        if variant not in VARIANTS:raise ValueError(f'Unknown variant: {variant}')
        pred=meta.apply(item['raw'],item['X'],probability,conditional,VARIANTS[variant])
        frame=forecast_frame(d,pred)
        dest=output/f'{variant}.csv'
        frame.to_csv(dest,index=False,float_format='%.7g')
        produced[variant]={'path':str(dest),'rows':len(frame),'total_quantity':float(pred.sum()),'zero_percent':round(100*float((pred==0).mean()),4)}
        print('GENERATED',variant,produced[variant],flush=True)
    return produced

def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,default=ROOT/'snapshots'/'weekly_history_2026-07-26.npz')
    p.add_argument('--models',type=Path,default=ROOT/'models'/'base')
    p.add_argument('--meta-models',type=Path,default=ROOT/'models')
    p.add_argument('--out',type=Path,default=ROOT/'forecasts')
    p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    args=p.parse_args()
    print(json.dumps(run(args.snapshot,args.models,args.meta_models,args.out,args.variants),ensure_ascii=False,indent=2))

if __name__=='__main__':cli()
