"""Reproducible chronological forward test: original CatBoost vs conditional-positive expert.
Each fold uses only earlier, already completed OOF windows to train its heads.
Use: python -m app.v82_forward_test --threads 4 --out research_v82/reproduced_forward.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostClassifier,CatBoostRegressor
from . import v8_train,nonton_v74 as v74,nonton_v7 as base
from .catboost_inference import ROOT,apply_catboost
from .snapshot import load_dataset
from .v82_inference import PARAMS

def run(threads=4,iterations=350,out=None,max_folds=5):
 d=load_dataset(ROOT/'training/history_snapshot.npz')
 f=[v8_train.item_for_date(d,ROOT/'training/oof',dt) for dt in v74.DEV_FOLDS]
 rows=[]
 for i in range(2,min(7,2+max_folds)):
  X=np.concatenate([f[j]['X'].to_numpy(dtype='float32') for j in range(i)])
  y=np.concatenate([f[j]['y'] for j in range(i)])
  config=dict(iterations=iterations,depth=6,learning_rate=.045,l2_leaf_reg=9,thread_count=threads,verbose=False,allow_writing_files=False)
  pc=CatBoostClassifier(loss_function='Logloss',random_seed=403,**config).fit(X,(y>0).astype('int8'))
  q=CatBoostRegressor(loss_function='RMSE',random_seed=404,**config).fit(X,np.log1p(y))
  qpos=CatBoostRegressor(loss_function='RMSE',random_seed=141,**config).fit(X[y>0],np.log1p(y[y>0]))
  item=f[i];x=item['X'].to_numpy(dtype='float32')
  p=pc.predict_proba(x)[:,1]
  amount=np.maximum(0,np.expm1(q.predict(x)))
  positive=np.maximum(0,np.expm1(qpos.predict(x)))
  raw=item['raw']
  conf=np.clip((p-PARAMS['base_threshold'])/PARAMS['confidence_width'],0,1)
  factor=np.clip(raw/(raw+PARAMS['amount_reference']),0,1)
  weight=PARAMS['alpha_scale']*conf*factor
  pred=apply_catboost(raw,item['X'],p,(1-weight)*amount+weight*positive)
  reference=apply_catboost(raw,item['X'],p,amount)
  for kind,pr in [('original',reference),('confidence_amount',pred)]:
   rows.append({'fold':v74.DEV_FOLDS[i],'method':kind,**base.metric(item['y'],pr)})
  print('EVAL',v74.DEV_FOLDS[i],flush=True)
  table=pd.DataFrame(rows)
  output=Path(out) if out else ROOT/'research_v82/reproduced_forward.csv'
  output.parent.mkdir(parents=True,exist_ok=True)
  table.to_csv(output,index=False)
 print('\n',table.groupby('method')[['SMAPE','WAPE','RMSE','Bias','RMSLE']].mean())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--threads',type=int,default=4);p.add_argument('--iterations',type=int,default=350);p.add_argument('--out');p.add_argument('--max-folds',type=int,default=5);arg=p.parse_args()
 run(arg.threads,arg.iterations,arg.out,arg.max_folds)
