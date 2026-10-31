"""Nonton v8.3 research: strictly chronological secondary experts on frozen v8.1 OOF.
No post-cutoff target data. Forward folds i=3..6 only; each head trains on already
completed earlier windows i=2..i-1. Original 39 LightGBM+2 CatBoost untouched.
"""
import os,sys,time,json,traceback
os.environ.setdefault('OMP_NUM_THREADS','3')
os.environ.setdefault('OPENBLAS_NUM_THREADS','3')
from pathlib import Path
import numpy as np,pandas as pd
from lightgbm import LGBMRegressor,LGBMClassifier
from catboost import CatBoostClassifier,CatBoostRegressor

SOURCE=Path('/mnt/data/Nonton_V82_EXPERIMENTAL')
RESEARCH=Path('/mnt/data/Nonton_V82_RESEARCH')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE))
from app import nonton_v7 as base,nonton_v74 as v74,v8_train
from app.snapshot import load_dataset
from app.catboost_inference import apply_catboost,prepare_features
from app.v82_inference import PARAMS
FOLDS=v74.DEV_FOLDS
H=9
W={'SMAPE':.5,'WAPE':.2,'RMSE':.15,'Bias':.1,'RMSLE':.05}

def metric(y,p):
 return {k:float(v) for k,v in base.metric(y,p).items()}

def features(item,z,qpos,macro=None,group=None):
 # Features available as of origin; p, q and base are out-of-fold predictions.
 x=item['X']
 ret=pd.DataFrame({
 'prob':z['prob'].astype('float32'),
 'amount':z['amount'].astype('float32'),
 'raw':z['raw'].astype('float32'),
 'positive_qty':np.asarray(qpos,dtype='float32'),
 'prob_distance':np.asarray(z['prob']-.53,dtype='float32'),
 'amount_disagreement':np.log1p(np.asarray(qpos,dtype='float32'))-np.log1p(z['amount']),
 },dtype='float32')
 for col in ['horizon','active_weeks_13','pair_occurrence_13','pair_occurrence_52',
             'persistence_4','persistence_13','difference_recent',
             'wh_growth_4_13','whcat_growth_4_13','whcat_growth_13_26',
             'wh_weekly_mean_4','whcat_weekly_mean_4','whcat_weekly_mean_13',
             'stock_last','stock_mean','warehouse_id','article_id','cat1_id',
             'forecast_month','forecast_weekofyear']:
  if col in x.columns:ret[col]=x[col].to_numpy(dtype='float32')
 if macro is not None and group is not None:
  # pair-major x horizon-minor; the macro model makes independent time-causal predictions.
  m=np.asarray(macro,dtype='float32')
  n=int(group.max())+1
  group_p=np.zeros((9,n),dtype='float64')
  baseline=apply_catboost(z['raw'],x,z['prob'],z['amount'])
  pr=baseline.reshape(-1,9)
  for h in range(9):group_p[h,:]=np.bincount(group,weights=pr[:,h],minlength=n)
  ratios=np.clip((m+.5)/(group_p+.5),.3,3.)
  ret['macro_group_week']=np.asarray(m[:,group].T.reshape(-1),dtype='float32')
  ret['macro_ratio']=np.asarray(ratios[:,group].T.reshape(-1),dtype='float32')
  ret['macro_group_base']=np.asarray(group_p[:,group].T.reshape(-1),dtype='float32')
 assert np.isfinite(ret.to_numpy()).all()
 return ret

def v82_prediction(z,x,qpos):
 prob=z['prob'];base_raw=z['raw'];old=z['amount']
 confidence=np.clip((prob-PARAMS['base_threshold'])/PARAMS['confidence_width'],0,1)
 amount_factor=np.clip(base_raw/(base_raw+PARAMS['amount_reference']),0,1)
 alpha=PARAMS['alpha_scale']*confidence*amount_factor
 out=apply_catboost(base_raw,x,prob,(1-alpha)*old+alpha*qpos)
 return out

def train_gate(X,y,z,cap=0,seed=300):
 # Per-example advantage of predicting the *original* positive quantity vs zero.
 positive=apply_catboost(z['raw'],pd.DataFrame({'active_weeks_13':np.zeros(len(y))}),np.ones(len(y)),z['amount'])
 gain=np.where(y>0,200.-200.*np.abs(y-positive)/np.maximum(y+positive,1e-9),-200.)
 model=LGBMRegressor(n_estimators=140,learning_rate=.045,num_leaves=11,min_child_samples=500,
     reg_lambda=35,colsample_bytree=.85,max_bin=127,verbosity=-1,n_jobs=3,random_state=seed)
 model.fit(X,gain)
 return model

def train_qty(X,y,seed=400):
 mask=y>0
 model=LGBMRegressor(n_estimators=150,learning_rate=.042,num_leaves=15,min_child_samples=260,
    reg_lambda=25,colsample_bytree=.88,max_bin=127,verbosity=-1,n_jobs=3,random_state=seed)
 model.fit(X.loc[mask],np.log1p(y[mask]))
 return model

def do():
 t0=time.time(); d=load_dataset(SOURCE/'training/history_snapshot.npz')
 group=pd.factorize(pd.MultiIndex.from_arrays([d.wh_ids,d.cat1_ids]),sort=True)[0]
 folds={}
 for i in range(2,7):
  f=v8_train.item_for_date(d,SOURCE/'training/oof',FOLDS[i])
  z=np.load(RESEARCH/f'results/predict_reference_fold{i}.npz')
  zz={key:z[key].copy() for key in z.files}
  qpos=np.load(RESEARCH/f'positive_results/positive_quantity_fold{i}.npz')['quantity']
  ma=np.load(RESEARCH/f'macro_results/macro_fold{i}.npz')['macro']
  assert np.array_equal(f['y'],zz['actual'])
  X=features(f,zz,qpos,ma,group)
  folds[i]={'item':f,'z':zz,'qpos':qpos,'X':X,'y':f['y']}
  print('LOADED',i,FOLDS[i],X.shape,'actual_sum',round(f['y'].sum()),'t',round(time.time()-t0,1),flush=True)
 data=[]; preds={}
 for i in range(2,7):
  e=folds[i]; z=e['z']; x=e['item']['X']; y=e['y']; p=z['prob']; amt=z['amount'];raw=z['raw'];qp=e['qpos']
  reference=apply_catboost(raw,x,p,amt)
  v82=v82_prediction(z,x,qp)
  candidates={'v81_reference':reference,'v82_positive':v82}
  if i>=3:
   train_idx=list(range(2,i))
   Xtrain=pd.concat([folds[j]['X'] for j in train_idx],ignore_index=True)
   ytrain=np.concatenate([folds[j]['y'] for j in train_idx])
   ztrain={k:np.concatenate([folds[j]['z'][k] for j in train_idx]) for k in ['prob','amount','raw']}
   print('TRAIN',i,'n',len(ytrain),'prior folds',train_idx,'t',round(time.time()-t0,1),flush=True)
   mg=train_gate(Xtrain,ytrain,ztrain)
   predict_gain=mg.predict(e['X'])
   oldgate=(p>=.53)
   # Predict positive only when estimated SMAPE improvement beats zero; optional guard.
   riskgate=(predict_gain>0)
   candidates['new_gate_unrestricted']=apply_catboost(raw,x,riskgate.astype(float),amt,policy={'threshold':.5,'blend':.5,'regime':0.})
   candidates['new_gate_safe']=apply_catboost(raw,x,np.where((p>=.45)&(p<=.65),riskgate.astype(float),oldgate.astype(float)),amt,policy={'threshold':.5,'blend':.5,'regime':0.})
   candidates['new_gate_conservative']=apply_catboost(raw,x,np.where((p>=.48)&(p<=.60),riskgate.astype(float),oldgate.astype(float)),amt,policy={'threshold':.5,'blend':.5,'regime':0.})
   mq=train_qty(Xtrain,ytrain)
   conditional=np.expm1(mq.predict(e['X'])).clip(min=0)
   # Keep original zero mask; two fixed shrinkages and a p-dependent quantity expert.
   candidates['new_qty_20']=apply_catboost(raw,x,p,.8*amt+.2*conditional)
   candidates['new_qty_40']=apply_catboost(raw,x,p,.6*amt+.4*conditional)
   cf=np.clip((p-.53)/.40,0,1)*np.clip(raw/(raw+4),0,1)
   candidates['new_qty_safe']=apply_catboost(raw,x,p,(1-.6*cf)*amt+.6*cf*conditional)
   # two independent enhancements: positive and risk-gate
   blend=(1-.6*cf)*amt+.6*cf*conditional
   candidates['new_qty_and_gate_safe']=apply_catboost(raw,x,np.where((p>=.45)&(p<=.65),riskgate.astype(float),oldgate.astype(float)),blend,policy={'threshold':.5,'blend':.5,'regime':0.})
   print('TRAINED',i,'flip_count',int((riskgate!=oldgate).sum()),'t',round(time.time()-t0,1),flush=True)
  for k,v in candidates.items():
   r={'fold':FOLDS[i],'method':k,'training_windows':max(0,i-2),'predicted_sum':float(v.sum()),'nonzero':int((v>0).sum()),**metric(y,v)}
   data.append(r)
   if k in ('v81_reference','v82_positive','new_gate_safe','new_qty_safe','new_qty_and_gate_safe'):
    preds.setdefault(k,{})[i]=v
  pd.DataFrame(data).to_csv(OUT/'folds_all.csv',index=False)
  print('TEST',i,'results',[(k,round(r['SMAPE'],3),round(r['WAPE'],3)) for r in data if r['fold']==FOLDS[i]],'time',round(time.time()-t0,1),flush=True)
 tab=pd.DataFrame(data)
 for minfold in [3,4,5]:
  t=tab[tab.fold.isin(FOLDS[minfold:7])]
  m=t.groupby('method')[list(W)].mean()
  ref=m.loc['v81_reference']
  m['proxy_vs_v81']=sum(W[k]*m[k]/ref[k] for k in W)
  m['folds']=t.groupby('method').size()
  m=m.sort_values('proxy_vs_v81')
  m.to_csv(OUT/f'comparison_from_fold{minfold}.csv')
  print('\nAGG FOLD',minfold,'\n',m.round(5).to_string(),flush=True)
 (OUT/'config.json').write_text(json.dumps({'folds':FOLDS,'train_from_fold':2,'threads':3,'lightgbm_version':__import__('lightgbm').__version__,'elapsed_s':round(time.time()-t0,2)},indent=2),encoding='utf-8')
if __name__=='__main__':
 do()
