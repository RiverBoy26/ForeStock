"""NONTON v8: hierarchical, probabilistic demand mixture over frozen v7.4 base.

Three heads: P(q>0), conditional log-quantity, and direct Tweedie expected quantity.
All hierarchy/recency features are computed from weeks strictly before each origin.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import nonton_v7 as base, nonton_v74 as v74

H=9
VERSION='8.0.0'


def _group_features(history: np.ndarray, ids: np.ndarray, win:int) -> tuple[np.ndarray,np.ndarray]:
    """Average item sales per week and active fraction within its group."""
    section=history[-win:]
    sums=section.sum(axis=0,dtype=np.float64)
    counts=np.bincount(ids,minlength=int(ids.max())+1).clip(min=1)
    totals=np.bincount(ids,weights=sums,minlength=len(counts))
    active=np.bincount(ids,weights=(section>0).sum(axis=0),minlength=len(counts))
    return (totals[ids]/(counts[ids]*win)).astype('float32'), (active[ids]/(counts[ids]*win)).astype('float32')


def enrich(d,oi:int,frame:pd.DataFrame)->pd.DataFrame:
    """Return strictly backward-looking features in PAIR-major/HORIZON-minor order."""
    X=v74._features(d,oi,frame)
    if oi<52 or oi>base.N_WEEKS:raise ValueError('Origin outside permitted history')
    h=d.qty[:oi]
    n=d.n_pairs
    rep=lambda a:np.repeat(np.asarray(a,dtype=np.float32),H)
    horizon=np.tile(np.arange(1,H+1,dtype=np.float32),n)
    group_cat=pd.factorize(pd.MultiIndex.from_arrays([d.wh_ids,d.cat1_ids]),sort=True)[0].astype('int32')
    group_wh=d.wh_ids.astype('int32')
    # Scale and activity of the aggregate group. No future weeks or target labels.
    for name, ids in [('wh',group_wh),('whcat',group_cat)]:
        means={};fractions={}
        for w in (4,13,26,52):
            means[w],fractions[w]=_group_features(h,ids,w)
            X[f'{name}_weekly_mean_{w}']=rep(means[w])
            X[f'{name}_active_share_{w}']=rep(fractions[w])
        g4=(means[4]+0.08)/(means[13]+0.08)
        g13=(means[13]+0.08)/(means[26]+0.08)
        g26=(means[26]+0.08)/(means[52]+0.08)
        for lab,g in [('4_13',g4),('13_26',g13),('26_52',g26)]:
            clipped=np.clip(g,.4,2.5)
            X[f'{name}_growth_{lab}']=rep(clipped)
            X[f'{name}_growth_{lab}_horizon']=rep(np.log(clipped))*horizon/9.
    # Forecast month/week is known in advance and safe.
    weeks= d.weeks[oi-1]+pd.to_timedelta(np.arange(1,H+1)*7,unit='D') if oi<len(d.weeks) else base.FINAL+pd.to_timedelta(np.arange(H)*7,unit='D')
    X['forecast_weekofyear']=np.tile(weeks.isocalendar().week.to_numpy(dtype='float32'),n)
    X['forecast_month']=np.tile(weeks.month.to_numpy(dtype='float32'),n)
    X['forecast_sin52']=np.sin(2*np.pi*X['forecast_weekofyear'].to_numpy()/52.18).astype('float32')
    X['forecast_cos52']=np.cos(2*np.pi*X['forecast_weekofyear'].to_numpy()/52.18).astype('float32')
    X['persistence_13']=rep(h[-13:].mean(axis=0))
    X['persistence_4']=rep(h[-4:].mean(axis=0))
    X['difference_recent']=rep(h[-4:].mean(axis=0)-h[-13:].mean(axis=0))
    X['pair_occurrence_13']=rep((h[-13:]>0).mean(axis=0))
    X['pair_occurrence_52']=rep((h[-52:]>0).mean(axis=0))
    return X.astype('float32')


def fit(items:list[dict],threads:int=8,rounds:int=210):
    import lightgbm as lgb
    X=pd.concat([it['X'] for it in items],ignore_index=True)
    y=np.concatenate([it['y'] for it in items]).astype(np.float32)
    P={**v74.P_PARAMS,'num_threads':threads,'seed':80,'min_data_in_leaf':420,'num_leaves':21}
    Q={**v74.Q_PARAMS,'num_threads':threads,'seed':81,'min_data_in_leaf':340,'num_leaves':21}
    D={'objective':'tweedie','tweedie_variance_power':1.4,'learning_rate':.038,
       'num_leaves':23,'min_data_in_leaf':360,'lambda_l2':17,'feature_fraction':.86,
       'verbosity':-1,'seed':82,'num_threads':threads,'deterministic':True,'force_col_wise':True}
    p=lgb.train(P,lgb.Dataset(X,label=(y>0).astype('int8')),num_boost_round=rounds)
    active=y>0
    q=lgb.train(Q,lgb.Dataset(X.loc[active],label=np.log1p(y[active])),num_boost_round=rounds)
    m=lgb.train(D,lgb.Dataset(X,label=y),num_boost_round=rounds)
    return (p,q,m)


def predict(models,item):
    p,q,m=models
    X=item['X']
    return (np.clip(p.predict(X,num_threads=1),0.,1.),
            np.maximum(np.expm1(q.predict(X,num_threads=1)),0.),
            np.maximum(m.predict(X,num_threads=1),0.))


def config_grid():
    """Finite predeclared policy grid; no use of holdout/contest labels."""
    yield 'V74_BASE',{'kind':'base'}
    for t in (.52,.55,.58,.61):
        for b in (0.,.2,.4):
            for c in (0.,.15):
                for macro in (0.,.15):
                    name=f'V8_P{t:.3f}_DIRECT{b:.2f}_POS{c:.2f}_MACRO{macro:.2f}'
                    yield name,{'kind':'v8','threshold':t,'direct':b,'positive':c,'macro':macro}
    # Two posterior regimes: extra caution on 0-2 recent active weeks, more permissive on 7+.
    for t in (.55,.58):
        for b in (.2,.4):
            for margin in (.035,.06):
                yield f'V8_REGIME_P{t:.3f}_DIRECT{b:.2f}_M{margin:.3f}',{'kind':'regime','threshold':t,'direct':b,'positive':0.,'macro':.15,'margin':margin}


def apply(item,outputs,config):
    original=np.maximum(item['raw'].astype(np.float64),0)
    if config['kind']=='base':return base.postprocess(original,1.,1.)
    prob,conditional,direct=outputs
    x=item['X']
    activity=x['active_weeks_13'].to_numpy()
    t=config['threshold']
    if config['kind']=='regime':
        t=t+np.where(activity<=2,config['margin'],np.where(activity>=7,-config['margin'],0.))
    # Direct objective influences magnitude, not just the probability of demand.
    # Clip to avoid explosive changes on rare high-volume articles.
    ratio=np.clip((direct+.2)/(np.maximum(original,.5)+.2),.7,1.45)
    raw=np.maximum(original,.5)* (1+config['direct']*(ratio-1))
    if config['positive']:
        pratio=np.clip((conditional+.2)/(np.maximum(raw,.5)+.2),.7,1.45)
        raw=raw*(1+config['positive']*(pratio-1))
    if config['macro']:
        # MULTILEVEL: adjust only active SKUs with smoothed, backward-looking warehouse-category slope.
        growth=x['whcat_growth_4_13'].to_numpy()
        whgrowth=x['wh_growth_4_13'].to_numpy()
        h=x['horizon'].to_numpy()
        slope=np.clip(.65*np.log(np.maximum(growth,.05))+.35*np.log(np.maximum(whgrowth,.05)),-.45,.45)
        factor=np.exp(config['macro']*slope*np.minimum(h,9)/9.)
        raw*=np.where(activity>=7,factor,1.)
    result=np.where(prob>=t,np.maximum(raw,1.),0.)
    return np.maximum(result,0.)
