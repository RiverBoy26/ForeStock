"""Relearn one conditional-amount CatBoost from original seven frozen OOF folds.
Strictly supervised on existing positive labels, does not overwrite originals.
Usage: python -m app.v82_retrain_positive --out models/experimental/conditional_positive_quantity.cbm
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostRegressor
from .catboost_inference import ROOT
from .snapshot import load_dataset
from . import v8_train,nonton_v74 as v74

def train(iterations=350,threads=4,out=None):
 d=load_dataset(ROOT/'training/history_snapshot.npz')
 x=[];y=[]
 for dt in v74.DEV_FOLDS:
  item=v8_train.item_for_date(d,ROOT/'training/oof',dt)
  sel=item['y']>0
  x.append(item['X'].to_numpy(dtype='float32')[sel]);y.append(np.log1p(item['y'][sel]))
 x=np.vstack(x);y=np.concatenate(y)
 model=CatBoostRegressor(loss_function='RMSE',iterations=iterations,depth=6,learning_rate=.045,l2_leaf_reg=9,
      random_seed=141,thread_count=threads,verbose=False,allow_writing_files=False)
 model.fit(x,y)
 output=Path(out) if out else ROOT/'models/experimental/conditional_positive_quantity.cbm'
 output.parent.mkdir(parents=True,exist_ok=True);model.save_model(str(output))
 print('SAVED',output,'positive_examples',len(y),'features',x.shape[1])
 return model
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--iterations',type=int,default=350);p.add_argument('--threads',type=int,default=4);p.add_argument('--out');arg=p.parse_args()
 train(arg.iterations,arg.threads,arg.out)
