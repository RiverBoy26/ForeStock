"""Re-train ONLY the two CatBoost heads from seven OOF windows; never overwrites bundled models by default."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .snapshot import load_dataset
from . import v8_train
from .catboost_inference import ROOT
from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--iterations',type=int,default=350)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--out',type=Path,default=ROOT/'refitted_catboost')
    a=p.parse_args()
    if a.iterations<1 or a.threads<1: p.error('iterations and threads must be positive')
    d=load_dataset(ROOT/'training'/'history_snapshot.npz')
    items=[v8_train.item_for_date(d,ROOT/'training'/'oof',dt) for dt in v8_train.FOLDS]
    X=pd.concat([z['X'] for z in items],ignore_index=True).to_numpy(dtype='float32')
    y=np.concatenate([z['y'] for z in items]).astype('float32')
    common={'iterations':a.iterations,'depth':6,'learning_rate':.045,'l2_leaf_reg':9,
            'thread_count':a.threads,'verbose':False,'allow_writing_files':False}
    a.out.mkdir(parents=True,exist_ok=True)
    p_model=CatBoostClassifier(loss_function='Logloss',random_seed=403,**common)
    p_model.fit(X,(y>0).astype('int8'))
    q_model=CatBoostRegressor(loss_function='RMSE',random_seed=404,**common)
    q_model.fit(X,np.log1p(y))
    p_model.save_model(str(a.out/'catboost_prob.cbm'))
    q_model.save_model(str(a.out/'catboost_logquantity.cbm'))
    (a.out/'metadata.json').write_text(json.dumps({'iterations':a.iterations,'folds':v8_train.FOLDS,'rows':len(X)},indent=2),encoding='utf-8')
    print('SAVED',a.out,'records',len(X),'features',X.shape[1])
if __name__=='__main__':main()
