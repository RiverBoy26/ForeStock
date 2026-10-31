"""Feature engineering as of historical origin, without target week leakage."""
import numpy as np
import pandas as pd
from data_utils import START,HORIZON

class FeatureBuilder:
    def __init__(self,panel):
        self.p=panel
        self.memo={}
    def _asof(self,o):
        assert 26<=o<=134
        if o in self.memo:return self.memo[o]
        p=self.p; s=p.S[:o];g=p.G[:o];aa=p.article_sums[:o];wh=p.wh_sums[:o]
        feats={}
        for w in (4,8,13,26,52):
            r=s[-w:];feats[f'mean{w}']=r.mean(0)
            feats[f'freq{w}']=(r>0).mean(0)
            feats[f'std{w}']=r.std(0)
        for w in (1,2,4,8,13,26,52):
            feats[f'lag{w}']=s[-w] if o>=w else np.zeros(p.N,dtype='float32')
        for w in (4,13,26):
            feats[f'group{w}']=g[-w:].mean(0)[p.grp]
            feats[f'article{w}']=aa[-w:].mean(0)[p.article]
            feats[f'warehouse{w}']=wh[-w:].mean(0)[p.wh]
        feats['trend_sku']=np.log((feats['mean4']+.3)/(feats['mean13']+.3))
        feats['trend_group']=np.log((feats['group4']+1)/(feats['group13']+1))
        feats['trend_article']=np.log((feats['article4']+.5)/(feats['article13']+.5))
        # First two-level structural mappings are metadata, not future sales values.
        feats['warehouse_id']=p.wh.astype('float32')
        feats['category_id']=p.cat.astype('float32')
        feats['article_id']=p.article.astype('float32')
        feats['company_last4']=np.full(p.N,s[-4:].sum(1).mean(),dtype='float32')
        for key,val in feats.items():
            v=np.asarray(val,dtype='float32'); assert v.shape==(p.N,),key
            feats[key]=np.clip(v,-15,1e6)
        out=np.column_stack(list(feats.values())).astype('float32')
        names=list(feats)
        self.memo[o]=(out,names)
        return self.memo[o]
    def sku_nn(self,o):
        return self._asof(o)[0]
    def sku_tabular(self,o):
        base,_=self._asof(o);n=self.p.N
        dates=pd.date_range(START+pd.Timedelta(weeks=o),periods=HORIZON,freq='7D')
        h=np.repeat(np.arange(1,HORIZON+1,dtype='float32'),n)
        doy=np.repeat(np.array([d.dayofyear for d in dates],dtype='float32'),n)
        year=np.repeat(np.array([d.year for d in dates],dtype='float32'),n)
        cal=np.column_stack((h, np.sin(2*np.pi*doy/365.25),np.cos(2*np.pi*doy/365.25),year-2024))
        return np.ascontiguousarray(np.hstack([np.tile(base,(HORIZON,1)),cal]).astype('float32'))
    def macro_tabular(self,o):
        p=self.p;g=p.G[:o];ng=p.NG
        hist={}
        for w in (4,8,13,26,52):
            r=g[-w:];hist[f'gmean{w}']=r.mean(0)
            hist[f'gfreq{w}']=(r>0).mean(0)
            hist[f'gstd{w}']=r.std(0)
        for w in (1,2,4,13,26,52):hist[f'lag{w}']=g[-w] if o>=w else np.zeros(ng,dtype='float32')
        hist['trend4_13']=np.log((hist['gmean4']+1)/(hist['gmean13']+1))
        hist['trend13_52']=np.log((hist['gmean13']+1)/(hist['gmean52']+1))
        hist['warehouse_id']=p.gwh.astype('float32')
        hist['category_id']=p.gcat.astype('float32')
        hist['group_id']=np.arange(ng,dtype='float32')
        hist['company_mean4']=np.full(ng,g[-4:].sum(1).mean(),dtype='float32')
        base=np.column_stack(list(hist.values())).astype('float32')
        dates=pd.date_range(START+pd.Timedelta(weeks=o),periods=HORIZON,freq='7D')
        h=np.repeat(np.arange(1,HORIZON+1,dtype='float32'),ng)
        doy=np.repeat(np.array([d.dayofyear for d in dates],dtype='float32'),ng)
        cal=np.column_stack((h,np.sin(2*np.pi*doy/365.25),np.cos(2*np.pi*doy/365.25)))
        return np.ascontiguousarray(np.hstack([np.tile(base,(HORIZON,1)),cal]).astype('float32'))
    def meta(self,o,base,legacy,prob,qty,macro):
        p=self.p;n=p.N
        assert base.shape==legacy.shape==prob.shape==qty.shape==(HORIZON,n)
        assert macro.shape==(HORIZON,p.NG)
        orig=self._asof(o)[0]
        nn=prob*qty
        mm=macro[:,p.grp]
        global_m=macro.sum(1)
        h=np.repeat(np.arange(1,HORIZON+1,dtype='float32'),n)
        dates=pd.date_range(START+pd.Timedelta(weeks=o),periods=HORIZON,freq='7D')
        doy=np.repeat(np.array([d.dayofyear for d in dates],dtype='float32'),n)
        cols=[np.log1p(base.ravel()),np.log1p(legacy.ravel()),
              prob.ravel(),np.log1p(qty.ravel()),np.log1p(nn.ravel()),
              np.log1p(qty.ravel())-np.log1p(base.ravel()),
              np.log1p(nn.ravel())-np.log1p(base.ravel()),
              np.log1p(mm.ravel()),
              np.log1p(np.repeat(global_m,n)),h,
              np.sin(2*np.pi*doy/365.25),np.cos(2*np.pi*doy/365.25)]
        X=np.column_stack(cols+[np.tile(orig,(HORIZON,1))[:,j] for j in range(orig.shape[1])]).astype('float32')
        return np.ascontiguousarray(np.clip(X,-50,1e7))
