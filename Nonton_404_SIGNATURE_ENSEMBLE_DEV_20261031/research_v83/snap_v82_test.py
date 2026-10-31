"""Integer-demand correction experiment, preserving v8.1 zero mask.
Uses five forward OOF windows; rolling selection uses ONLY earlier OOF windows.
Weighted ratio is a local proxy, NOT organizer Score.
"""
import os,sys,itertools,json
os.environ.setdefault('OPENBLAS_NUM_THREADS','3')
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/mnt/data/Nonton_V82_EXPERIMENTAL');R=Path('/mnt/data/Nonton_V82_RESEARCH');OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from app import nonton_v7 as base,nonton_v74 as v74,v8_train
from app.snapshot import load_dataset
from app.catboost_inference import apply_catboost
from app.v82_inference import PARAMS
F=v74.DEV_FOLDS; MET=['SMAPE','WAPE','RMSE','Bias','RMSLE'];WEIGHTS=np.array([.5,.2,.15,.1,.05])

def sm(y,p):return {k:float(v) for k,v in base.metric(y,p).items()}

def snap(v,offset,cap,weight=1.,guard=None):
 v=np.asarray(v);eligible=(v>0)&(v<=cap)
 if guard is not None:eligible &=guard
 r=np.where(eligible,np.maximum(1,np.floor(v+offset)),v)
 return (v+weight*(r-v)).astype('float64')

def run():
 d=load_dataset(ROOT/'training/history_snapshot.npz')
 rows=[]
 names=['baseline81','baseline82']
 definitions={}
 for version in ['v81','v82']:
  for cap in (3,5,10,100000):
   for off in (.35,.5,.65,.8):
    for w in (.5,1.):
     name=f'{version}_snap_cap{cap}_off{off}_w{w}'
     definitions[name]=(version,cap,off,w,0)
 # Additional predeclared confidence variants.
 for threshold in (.65,.8):
  for off in (.5,.65,.8):
   name=f'v82_guardp{threshold}_off{off}'
   definitions[name]=('v82',10,off,1.,threshold)
 for i in range(2,7):
  dt=F[i];z=np.load(R/f'results/predict_reference_fold{i}.npz');prob=z['prob'];raw=z['raw'];amt=z['amount'];y=z['actual']
  qpos=np.load(R/f'positive_results/positive_quantity_fold{i}.npz')['quantity']
  cf=np.clip((prob-.53)/.45,0,1)*np.clip(raw/(raw+4),0,1)
  b1=apply_catboost(raw,pd.DataFrame({'active_weeks_13':np.zeros(len(prob))}),prob,amt)
  b2=apply_catboost(raw,pd.DataFrame({'active_weeks_13':np.zeros(len(prob))}),prob,(1-.6*cf)*amt+.6*cf*qpos)
  b={'v81':b1,'v82':b2}
  for k,p in [('baseline81',b1),('baseline82',b2)]:rows.append(dict(fold=dt,method=k,**sm(y,p)))
  for key,(ver,cap,off,w,guard) in definitions.items():
   q=snap(b[ver],off,cap,w,prob>=guard if guard else None)
   rows.append(dict(fold=dt,method=key,**sm(y,q)))
  print('DONE',dt,'baseline',round(sm(y,b1)['SMAPE'],3),'baseline82',round(sm(y,b2)['SMAPE'],3),flush=True)
  pd.DataFrame(rows).to_csv(OUT/'snap_folds.csv',index=False)
 table=pd.DataFrame(rows)
 m=table.groupby('method')[MET].mean();a=m.loc['baseline81']
 m['proxy']=np.sum(m[MET].to_numpy()/a.to_numpy()*WEIGHTS[None,:],axis=1)
 m['smape_improve_pp']=a['SMAPE']-m['SMAPE']
 m.sort_values('proxy').to_csv(OUT/'snap_summary.csv')
 print('SUMMARY\n',m.sort_values('proxy').head(24).round(5).to_string(),flush=True)
 # Forward-only walk-forward select method based on earlier windows; exclude winning
 # segment dials and limit to 8 candidates that existed before inspecting holdout labels.
 allow=['baseline81','baseline82']+[f'{ver}_snap_cap{cap}_off{off}_w1.0' for ver in ['v81','v82'] for cap in (3,10) for off in (.5,.65)]
 choice=[]
 for i in range(3,7):
  prior=table[table.fold.isin(F[2:i])]
  prior_m=prior.pivot_table(index='method',values=MET,aggfunc='mean').loc[allow]
  rr=prior_m.loc['baseline81'];prior_m['relative']=np.sum(prior_m[MET].to_numpy()/rr.to_numpy()*WEIGHTS,axis=1)
  best=str(prior_m['relative'].idxmin())
  now=table[(table.fold==F[i])&(table.method==best)].iloc[0]
  ref=table[(table.fold==F[i])&(table.method=='baseline81')].iloc[0]
  choice.append({'fold':F[i],'selected_from_prior':best,**{k:now[k] for k in MET},**{'reference_'+k:ref[k] for k in MET}})
 pd.DataFrame(choice).to_csv(OUT/'snap_rolling_selection.csv',index=False)
 m2=pd.DataFrame(choice);print('ROLLING CHOICES\n',m2.to_string(index=False),flush=True)
 s=m2[MET].mean();r=m2[['reference_'+k for k in MET]].mean();relative=np.sum(np.array([s[k]/r['reference_'+k] for k in MET])*WEIGHTS)
 print('ROLLING avg',s.round(5).to_dict(),'baseline',r.round(5).to_dict(),'proxy',relative,flush=True)
if __name__=='__main__':run()
