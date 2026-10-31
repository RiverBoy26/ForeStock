"""No model fitting. Predetermined integer-count forecasts, tested on five historic OOF windows.
Fine for diagnosing SMAPE (zero forecasts remain exactly zero)."""
import os,sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','3')
from pathlib import Path
import numpy as np,pandas as pd
R=Path('/mnt/data/Nonton_V82_RESEARCH');P=Path('/mnt/data/Nonton_V82_EXPERIMENTAL');O=Path(__file__).parent
sys.path.insert(0,str(P))
from app import nonton_v7 as base,nonton_v74 as v74
from app.catboost_inference import apply_catboost
W={'SMAPE':.5,'WAPE':.2,'RMSE':.15,'Bias':.1,'RMSLE':.05}
rows=[]
for i in range(2,7):
 z=np.load(R/f'results/predict_reference_fold{i}.npz');p=z['prob'];raw=z['raw'];amt=z['amount'];y=z['actual'];orig=apply_catboost(raw,pd.DataFrame({'active_weeks_13':np.zeros(len(p))}),p,amt)
 # exact baseline no meta changes: uses original 0.53 gate
 cand={'original':orig}
 for maxqty in (2,3,5,10,50,1000000):
  for offset in (.2,.35,.5,.65,.8):
   q=np.where((orig>0)&(orig<=maxqty),np.maximum(1.,np.floor(orig+offset)),orig)
   cand[f'round_max{maxqty}_offset{offset}']=q
 for name,q in cand.items():
  m=base.metric(y,q)
  rows.append({'fold':v74.DEV_FOLDS[i],'method':name,**{k:m[k] for k in W},'predsum':q.sum()})
 print('FOLD',v74.DEV_FOLDS[i],'orig smape',round(base.metric(y,orig)['SMAPE'],4),flush=True)
df=pd.DataFrame(rows);df.to_csv(O/'quantize_folds.csv',index=False)
ref=df.query('method=="original"')[list(W)].mean()
m=df.groupby('method')[list(W)].mean()
m['relative_proxy']=sum(W[k]*m[k]/ref[k] for k in W)
m.sort_values('relative_proxy').to_csv(O/'quantize_summary.csv')
print('BEST\n',m.sort_values('relative_proxy').head(18).round(5).to_string(),flush=True)
