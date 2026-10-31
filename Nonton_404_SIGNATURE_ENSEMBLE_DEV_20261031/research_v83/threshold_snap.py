"""Joint threshold / integer snapping, rolling time-origin validation.
Prespecified t grid 0.51..0.61 and snap offsets 0, .5, .65. This is exploratory
and optimizer selection is trained only on earlier evaluation windows.
"""
import os,sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','3')
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/mnt/data/Nonton_V82_EXPERIMENTAL');R=Path('/mnt/data/Nonton_V82_RESEARCH');OUT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from app import nonton_v7 as base,nonton_v74 as v74
F=v74.DEV_FOLDS;MET=['SMAPE','WAPE','RMSE','Bias','RMSLE'];W=np.array([.5,.2,.15,.1,.05])
rows=[]
for i in range(2,7):
 z=np.load(R/f'results/predict_reference_fold{i}.npz');p=z['prob'];raw=z['raw'];amt=z['amount'];y=z['actual'];qpos=np.load(R/f'positive_results/positive_quantity_fold{i}.npz')['quantity']
 conf=np.clip((p-.53)/.45,0,1)*np.clip(raw/(raw+4),0,1)
 pos=(1-.6*conf)*amt+.6*conf*qpos
 for version,aa in [('v81',amt),('v82',pos)]:
  ratio=np.clip((aa+.25)/(np.maximum(raw,.5)+.25),.6,1.8)
  unmasked=np.maximum(np.maximum(raw,.5)*(1+.5*(ratio-1)),1.)
  for t in (.51,.53,.55,.57,.59,.61):
   mask=p>=t
   for off in (0.,.5,.65):
    for cap in (3,10):
     name=f'{version}_t{t:.2f}_off{off}_cap{cap}'
     q=unmasked.copy()
     if off: q=np.where(q<=cap,np.maximum(1,np.floor(q+off)),q)
     q=np.where(mask,q,0)
     res=base.metric(y,q);rows.append({'fold':F[i],'method':name,'model':version,'threshold':t,'offset':off,'cap':cap,**{k:res[k] for k in MET}})
 print('FOLD',F[i],'done',flush=True)
df=pd.DataFrame(rows);df.to_csv(OUT/'threshold_snap_folds.csv',index=False)
m=df.groupby('method')[MET].mean();b=m.loc['v81_t0.53_off0.0_cap10'];m['proxy']=np.sum(m[MET].to_numpy()/b.to_numpy()*W,axis=1)
print('TOP ALL\n',m.sort_values('proxy').head(22).round(4).to_string(),flush=True)
m.sort_values('proxy').to_csv(OUT/'threshold_snap_summary.csv')
allowed=[a for a in m.index if '_cap10' in a and ('_off0.0_' in a or '_off0.5_' in a or '_off0.65_' in a)]
roll=[]
for i in range(3,7):
 prior=df[(df.fold.isin(F[2:i]))&(df.method.isin(allowed))]
 ma=prior.groupby('method')[MET].mean();br=ma.loc['v81_t0.53_off0.0_cap10']
 ma['p']=np.sum(ma[MET].to_numpy()/br.to_numpy()*W,axis=1)
 best=ma['p'].idxmin();curr=df[(df.fold==F[i])&(df.method==best)].iloc[0]
 ref=df[(df.fold==F[i])&(df.method=='v81_t0.53_off0.0_cap10')].iloc[0]
 roll.append({'fold':F[i],'selected_prior':best,**{k:curr[k] for k in MET},**{'reference_'+k:ref[k] for k in MET}})
rd=pd.DataFrame(roll);rd.to_csv(OUT/'threshold_snap_rolling.csv',index=False)
rv=rd[MET].mean();rr=rd[['reference_'+k for k in MET]].mean()
print('ROLLING',rd.to_string(index=False),'\nproxy',float(sum(W[j]*rv[k]/rr['reference_'+k] for j,k in enumerate(MET))),flush=True)
