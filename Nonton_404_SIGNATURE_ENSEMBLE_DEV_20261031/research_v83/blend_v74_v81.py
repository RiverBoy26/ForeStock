"""New forward backtest: v7.4 independent META and v8.1 CatBoost, identical OOF origins.
Real historical forecasts. Meta for each origin trained on folds before that origin only.
Original 39 LightGBM predictions are frozen. No hidden labels.
"""
import os,sys,time,json
os.environ.setdefault('OMP_NUM_THREADS','3');os.environ.setdefault('OPENBLAS_NUM_THREADS','3')
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/mnt/data/Nonton_V82_EXPERIMENTAL');R=Path('/mnt/data/Nonton_V82_RESEARCH');OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from app import nonton_v74 as v74,nonton_v7 as base
from app.snapshot import load_dataset
from app.catboost_inference import apply_catboost
from app.v82_inference import PARAMS
F=v74.DEV_FOLDS;W={'SMAPE':.5,'WAPE':.2,'RMSE':.15,'Bias':.1,'RMSLE':.05}

def score(y,p):return {k:float(v) for k,v in base.metric(y,p).items()}

def run():
 t=time.time();d=load_dataset(ROOT/'training/history_snapshot.npz');folds=[]
 for dt in F:
  z=v74.load_fold(d,ROOT/'training/oof'/f'dev_predictions_{dt}.csv.gz',dt)
  folds.append(z)
 print('LOADED',len(folds),'t',round(time.time()-t,1),flush=True)
 rec=[]
 for i in range(2,7):
  model=v74.fit_meta(folds[:i],3,180,180)
  p74,q74=v74.predict_meta(model,folds[i])
  raw=folds[i]['raw'];X=folds[i]['X'];y=folds[i]['y']
  assert np.array_equal(y,np.load(R/f'results/predict_reference_fold{i}.npz')['actual'])
  z=np.load(R/f'results/predict_reference_fold{i}.npz');p81=z['prob'];a81=z['amount']
  pr81=apply_catboost(raw,X,p81,a81)
  pr74=v74.apply(raw,X,p74,q74,(.55,.15,7))
  p81gate=(pr81>0);p74gate=(pr74>0)
  candidates={'v81':pr81,'v74':pr74}
  # Distinct experts: retaining the original mask vs requiring consensus.
  candidates['and_mask81_amount']=np.where(p74gate,pr81,0)
  candidates['or_mask81_amount']=np.where(p74gate|p81gate,np.maximum(pr81,1),0)
  for g in [.65,.75,.85]:
   candidates[f'consensus_add_{g:.2f}']=np.where(p81gate|((p74>=g)&(p81>=.45)),np.maximum(pr81,1),0)
  for g in [.25,.35,.45]:
   candidates[f'consensus_remove_{g:.2f}']=np.where((p81gate)&((p74>=g)|(p81>=.8)),pr81,0)
  for b in [.1,.2,.3]:
   # Second estimator supplied as amount expert only; leave CatBoost zero gate untouched.
   candidates[f'v74_amount_blend_{b:.1f}']=np.where(p81gate,np.maximum(1.,(1-b)*pr81+b*np.maximum(pr74,1.)),0)
   candidates[f'ungated_blend_{b:.1f}']=(1-b)*pr81+b*pr74
  # Predicted agreement as classification feature, no target on evaluation fold.
  candidates['v74_strong_confirm']=np.where(p81gate|((p74>=.85)&(p81>=.4)),np.maximum(pr81,1),0)
  np.savez_compressed(OUT/f'v74_meta_fold{i}.npz',p74=p74.astype('float32'),q74=q74.astype('float32'),
                      prob81=p81.astype('float32'),amount81=a81.astype('float32'),raw=raw,
                      y=y,v74=pr74.astype('float32'),v81=pr81.astype('float32'))
  for name,p in candidates.items():rec.append({'fold':F[i],'method':name,**score(y,p)})
  print('FOLD',F[i],'v81',score(y,pr81)['SMAPE'],'v74',score(y,pr74)['SMAPE'],'t',round(time.time()-t,1),flush=True)
  pd.DataFrame(rec).to_csv(OUT/'blend_folds.csv',index=False)
 table=pd.DataFrame(rec)
 for minfold in (2,3,4,5):
  sub=table[table.fold.isin(F[minfold:])]; m=sub.groupby('method')[list(W)].mean()
  ref=m.loc['v81'];m['proxy']=sum(W[k]*m[k]/ref[k] for k in W);m['folds']=sub.groupby('method').size()
  m.sort_values('proxy').to_csv(OUT/f'blend_summary_from_fold{minfold}.csv')
  print('AGG',minfold,'\n',m.sort_values('proxy').head(10).round(5).to_string(),flush=True)
if __name__=='__main__':run()
