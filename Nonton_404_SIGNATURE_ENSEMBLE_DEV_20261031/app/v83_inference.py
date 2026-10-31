"""Nonton v8.3: integer-count correction on v8.2 forecasts.
Preserves v8.1/v8.2 zero mask, 42 trained models untouched.
Usage: python -m app.v83_inference --variant nearest --verify
       python -m app.v83_inference --variant smape --out forecasts/custom.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from .catboost_inference import ROOT
from .v82_inference import predict as v82_predict

VARIANTS={
  'nearest': {'offset':0.50,'maximum':10.0,'filename':'v83_nearest10.csv'},
  'smape': {'offset':0.65,'maximum':10.0,'filename':'v83_smape10.csv'},
}

def snap_quantity(values,offset:float=0.50,maximum:float=10.0):
    values=np.asarray(values,dtype=np.float64)
    if not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Quantity must be nonnegative and finite')
    mask=(values>0)&(values<=maximum)
    result=values.copy()
    result[mask]=np.maximum(1.0,np.floor(values[mask]+offset))
    if not np.array_equal(result>0,values>0):
        raise AssertionError('Zero mask changed')
    return result

def run(variant:str='nearest',out=None,verify:bool=False,recompute_base:bool=False):
    if variant not in VARIANTS:raise ValueError('Unknown variant: '+str(variant))
    v82,baseline,alpha=v82_predict(recompute_base=recompute_base)
    cfg=VARIANTS[variant]
    result=snap_quantity(v82,cfg['offset'],cfg['maximum'])
    keys=pd.read_csv(ROOT/'training/keys.csv',dtype=str)
    if len(keys)!=79_101 or len(result)!=len(keys):raise AssertionError('Wrong target shape')
    if list(keys.columns)!=['week','warehouse_name','article']:raise AssertionError('Wrong key schema')
    if keys.duplicated().any() or len(keys.week.unique())!=9:raise AssertionError('Invalid keys')
    if not (result>=0).all() or not np.isfinite(result).all():raise AssertionError('Invalid quantities')
    if not np.array_equal(result>0,baseline>0):raise AssertionError('Official zero mask must be preserved')
    new=keys.copy();new['quantity']=result
    outfile=Path(out) if out is not None else ROOT/'forecasts'/cfg['filename']
    if verify:
       official=pd.read_csv(ROOT/'forecasts/v81_catboost.csv')
       if np.max(np.abs(official.quantity.to_numpy()-baseline))>1.e-3:raise AssertionError('Original not reproduced')
       if not outfile.exists():raise FileNotFoundError(outfile)
       saved=pd.read_csv(outfile,dtype={'week':str,'warehouse_name':str,'article':str})
       if not saved[['week','warehouse_name','article']].equals(new[['week','warehouse_name','article']]):raise AssertionError('Keys mismatch')
       max_error=float(np.max(np.abs(saved.quantity.to_numpy()-result)))
       if max_error>1.e-3:raise AssertionError(f'v8.3 reproduction mismatch {max_error}')
       print('PASS',variant,'rows',len(result),'total',round(result.sum(),2),'mask_ones',int((result>0).sum()),'max_abs_error',max_error)
       return new
    outfile.parent.mkdir(parents=True,exist_ok=True)
    new.to_csv(outfile,index=False,float_format='%.8g')
    print('SAVED',outfile,'rows',len(new),'total',round(result.sum(),2),'nonzero',int((result>0).sum()))
    return new

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=VARIANTS,default='nearest')
 p.add_argument('--out');p.add_argument('--verify',action='store_true');p.add_argument('--recompute-base',action='store_true')
 args=p.parse_args();run(args.variant,args.out,args.verify,args.recompute_base)
