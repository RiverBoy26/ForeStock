"""Conservative conditional-positive amount expert on frozen 39-LightGBM / 2-CatBoost forecast.
Reproduces unchanged original zero mask. Original models and v8.1 files untouched.
Usage: python -m app.v82_inference --verify
       python -m app.v82_inference --out forecasts/my_new_forecast.csv
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostRegressor,CatBoostClassifier
from .catboost_inference import ROOT,prepare_features,apply_catboost

PARAMS={'base_threshold':0.53,'alpha_scale':0.6,'confidence_width':0.45,'amount_reference':4.,'minimum_positive':1.}

def predict(recompute_base=False):
 d,raw,X=prepare_features(recompute_base)
 f=X.to_numpy(dtype='float32')
 original_prob=CatBoostClassifier();original_prob.load_model(str(ROOT/'models/catboost/catboost_prob.cbm'))
 original_q=CatBoostRegressor();original_q.load_model(str(ROOT/'models/catboost/catboost_logquantity.cbm'))
 experimental_q=CatBoostRegressor();experimental_q.load_model(str(ROOT/'models/experimental/conditional_positive_quantity.cbm'))
 prob=np.clip(original_prob.predict_proba(f)[:,1],0,1)
 old_amount=np.maximum(0,np.expm1(original_q.predict(f)))
 positive_amount=np.maximum(0,np.expm1(experimental_q.predict(f)))
 base=np.asarray(raw['hybrid_raw'],dtype='float64')
 confidence=np.clip((prob-PARAMS['base_threshold'])/PARAMS['confidence_width'],0,1)
 amount_factor=np.clip(base/(base+PARAMS['amount_reference']),0,1)
 alpha=PARAMS['alpha_scale']*confidence*amount_factor
 amount=(1-alpha)*old_amount+alpha*positive_amount
 p=apply_catboost(base,X,prob,amount)
 po=apply_catboost(base,X,prob,old_amount)
 assert np.array_equal(po==0,p==0),'Positive-quantity expert must not change the zero mask'
 return p,po,alpha

def run(path=None,verify=False,recompute_base=False):
 p,baseline,alpha=predict(recompute_base)
 keys=pd.read_csv(ROOT/'training/keys.csv',dtype={'week':str,'warehouse_name':str,'article':str})
 assert len(keys)==79101 and len(p)==len(keys)
 assert list(keys.columns)==['week','warehouse_name','article']
 assert not keys.duplicated().any() and np.isfinite(p).all() and (p>=0).all()
 result=keys.copy();result['quantity']=p
 target=Path(path) if path is not None else ROOT/'forecasts/v82_confidence_amount.csv'
 if verify:
  original=pd.read_csv(ROOT/'forecasts/v81_catboost.csv')
  delta=np.max(np.abs(original.quantity.to_numpy()-baseline))
  assert delta<.001,f'Original CatBoost replay mismatch: {delta}'
  if target.is_file():
   saved=pd.read_csv(target)
   assert saved[['week','warehouse_name','article']].equals(result[['week','warehouse_name','article']])
   reproducibility=np.max(np.abs(saved.quantity.to_numpy()-p))
   assert reproducibility<.001,f'Experimental forecast mismatch: {reproducibility}'
  print('VERIFIED original_max_abs',delta,'mask',int((p>0).sum()),'base_sum',baseline.sum(),'experimental_sum',p.sum())
  return
 target.parent.mkdir(parents=True,exist_ok=True)
 result.to_csv(target,index=False,float_format='%.8g')
 print('SAVED',target,'rows',len(result),'total',round(float(p.sum()),1),'nonzero',int((p>0).sum()))

if __name__=='__main__':
 arg=argparse.ArgumentParser();arg.add_argument('--out');arg.add_argument('--verify',action='store_true');arg.add_argument('--recompute-base',action='store_true');args=arg.parse_args()
 run(args.out,args.verify,args.recompute_base)
