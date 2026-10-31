"""Strict as-of Nonton weekly panels. Reconstructed, not the archived original."""
from __future__ import annotations
import json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

START = pd.Timestamp('2024-01-01')
CUTOFF = pd.Timestamp('2026-07-26')
HORIZON = 9
FOLDS = ['2025-01-27','2025-04-07','2025-07-28','2025-10-06',
         '2025-12-15','2026-03-02','2026-05-25']


def week_index(dates):
    dt = pd.to_datetime(dates)
    return ((dt - pd.to_timedelta(dt.dt.weekday, unit='D') - START).dt.days // 7).to_numpy(dtype=np.int32)


def stock_daily_sum(stock:pd.DataFrame) -> pd.DataFrame:
    """FAQ: ADD every repeated row, including apparently identical rows."""
    return stock.groupby(['date','warehouse_name','article'],as_index=False,sort=False,observed=True)['stock'].sum()


def load_raw(raw_zip:str, use_stock:bool=False):
    with zipfile.ZipFile(raw_zip) as archive:
        sales=pd.read_csv(archive.open('sales_dataset_hakaton_train.csv'),
            usecols=['date','warehouse_name','article','quantity','category_lvl_1','composite_division'],
            dtype={'warehouse_name':'string','article':'string','category_lvl_1':'string','composite_division':'string'},
            low_memory=False)
        sales['date']=pd.to_datetime(sales.date)
        rejected=int((sales.date>CUTOFF).sum())
        sales=sales.loc[(sales.date<=CUTOFF)&(sales.date>=START)].copy()
        if use_stock:
            stock=pd.read_csv(archive.open('stock_dataset_hakaton_train.csv'),
                usecols=['date','warehouse_name','article','stock'])
            stock['date']=pd.to_datetime(stock.date)
            stock=stock.loc[(stock.date<=CUTOFF)&(stock.date>=START)].copy()
            # Fully correct preprocessing; optional historical stock features are not
            # introduced in this faithful-to-v7 sales-only reconstruction.
            stock=stock_daily_sum(stock)
        else:stock=None
    return sales,stock,rejected


def build_panel(raw_zip:str, outdir:Path, use_stock=False):
    outdir.mkdir(parents=True,exist_ok=True)
    sales,stock,rejected=load_raw(raw_zip,use_stock)
    assert not sales.quantity.isna().any() and (sales.quantity>=0).all()
    weeks=pd.date_range(START,periods=134,freq='W-MON')
    assert weeks[-1]==pd.Timestamp('2026-07-20')
    warehouses=sorted(sales.warehouse_name.unique().tolist())
    articles=sorted(sales.article.unique().tolist())
    assert len(warehouses)==11 and len(articles)==799,(len(warehouses),len(articles))
    whmap={w:i for i,w in enumerate(warehouses)}
    artmap={a:i for i,a in enumerate(articles)}
    categories=sorted(sales.category_lvl_1.dropna().unique().tolist())
    catmap={c:i for i,c in enumerate(categories)}
    divisions=sorted(sales.composite_division.dropna().unique().tolist())
    depmap={d:i for i,d in enumerate(divisions)}
    # Only allowed history, static article -> category lookup by highest frequency.
    cat_by_article=(sales.groupby(['article','category_lvl_1'],observed=True).size()
        .reset_index(name='n').sort_values(['article','n'],ascending=[True,False])
        .drop_duplicates('article').set_index('article')['category_lvl_1'])
    article_cat=np.array([catmap[cat_by_article[a]] for a in articles],dtype=np.int16)
    widx=np.repeat(np.arange(len(warehouses),dtype=np.int16),len(articles))
    aidx=np.tile(np.arange(len(articles),dtype=np.int16),len(warehouses))
    cats=article_cat[aidx]
    grp=widx*len(categories)+cats
    nseries=len(widx); ng=len(warehouses)*len(categories)
    w=week_index(sales.date)
    si=sales.warehouse_name.map(whmap).to_numpy(dtype=np.int32)*len(articles)+sales.article.map(artmap).to_numpy(dtype=np.int32)
    S=np.zeros((134,nseries),dtype=np.float32)
    np.add.at(S,(w,si),sales.quantity.to_numpy(dtype=np.float32))
    G=np.zeros((134,ng),dtype=np.float32)
    for t in range(134):G[t]=np.bincount(grp,weights=S[t],minlength=ng)
    # Division mix is retained for downstream analysis, but not injected into the v7 router.
    D=np.zeros((134,ng,len(divisions)),dtype=np.float32)
    gindex=(sales.warehouse_name.map(whmap).to_numpy(dtype=np.int32)*len(categories)
        +sales.category_lvl_1.map(catmap).to_numpy(dtype=np.int32))
    didx=sales.composite_division.map(depmap).to_numpy(dtype=np.int32)
    np.add.at(D,(w,gindex,didx),sales.quantity.to_numpy(dtype=np.float32))
    assert np.isclose(S.sum(),sales.quantity.sum(),rtol=1e-6)
    assert np.allclose(G.sum(axis=1),S.sum(axis=1))
    assert np.allclose(D.sum(axis=(1,2)),S.sum(axis=1))
    outdir.joinpath('history.npz').write_bytes(b'') if False else None
    np.savez_compressed(outdir/'history.npz',S=S,G=G,D=D,wh=widx,article=aidx,cat=cats,grp=grp,
                        categories=np.array(categories),warehouses=np.array(warehouses),articles=np.array(articles))
    series=pd.DataFrame({'warehouse_name':np.array(warehouses)[widx],
       'article':np.array(articles)[aidx], 'category_lvl_1':np.array(categories)[cats],
       'wh_id':widx,'article_id':aidx,'cat_id':cats,'group_id':grp})
    series.to_csv(outdir/'series.csv',index=False)
    manifest={'cutoff':str(CUTOFF.date()),'train_weeks':134,'excluded_future_sales_rows':rejected,
       'warehouses':len(warehouses),'articles':len(articles),'categories':len(categories),
       'divisions':len(divisions),'group_count':ng,'series_count':nseries,
       'sum_quantity':float(S.sum()),'stock_loaded':bool(use_stock),
       'source':'sales only for current models; historical stock can be validated with --check-stock',
       'strict_asof_caveat':'Category codes are determined from entire allowed pre-cutoff history; early historical validation sees static mapping of categories not necessarily known at that time.'}
    (outdir/'dataset_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return outdir/'history.npz'


class Panel:
    def __init__(self,path):
        z=np.load(path,allow_pickle=False)
        for k in z.files:setattr(self,k,z[k])
        self.T,self.N=self.S.shape
        self.NG=self.G.shape[1]
        self.nart=int(self.article.max()+1)
        self.nwh=int(self.wh.max()+1)
        self.ncat=int(self.cat.max()+1)
        assert self.T==134 and self.N==8789
        self.article_sums=np.zeros((self.T,self.nart),dtype=np.float32)
        for t in range(self.T): self.article_sums[t]=np.bincount(self.article,weights=self.S[t],minlength=self.nart)
        self.wh_sums=np.zeros((self.T,self.nwh),dtype=np.float32)
        for t in range(self.T):self.wh_sums[t]=np.bincount(self.wh,weights=self.S[t],minlength=self.nwh)
        self.sku_keys=pd.DataFrame({'warehouse_name':self.warehouses[self.wh], 'article':self.articles[self.article]})
        self.gwh=np.repeat(np.arange(self.nwh),self.ncat)
        self.gcat=np.tile(np.arange(self.ncat),self.nwh)

    def fold_origins(self):
        return [int((pd.Timestamp(d)-START).days//7) for d in FOLDS]

    def target(self,origin,h=HORIZON):
        assert origin+h<=self.T, (origin,h,self.T)
        return self.S[origin:origin+h]


def trend_quality(y,p,group_id=None,ng=None):
    """Diagnose whether weekly demand *shape* and directions are learned.
    Business emphasis: trends rather than level-only extrapolation.
    Only meaningful on historical folds with known actual values.
    """
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    a=y.sum(1);b=p.sum(1)
    corr=float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-10 and np.std(b)>1e-10 else 0.
    act=np.diff(a);pred=np.diff(b)
    hit=float(np.mean(np.sign(act)==np.sign(pred)))
    out={'company_week_corr':round(corr,5),
         'company_trend_hit':round(100*hit,3),
         'company_first_last_actual_pct':round(float(100*(a[-1]/max(a[0],1e-6)-1)),3),
         'company_first_last_pred_pct':round(float(100*(b[-1]/max(b[0],1e-6)-1)),3)}
    if group_id is not None:
        gy=np.stack([np.bincount(group_id,weights=v,minlength=ng) for v in y])
        gp=np.stack([np.bincount(group_id,weights=v,minlength=ng) for v in p])
        delta_y=np.diff(gy,axis=0); delta_p=np.diff(gp,axis=0)
        valid=np.abs(delta_y)>=1.0
        out['group_week_direction_hit_pct']=round(float(100*(np.sign(delta_y[valid])==np.sign(delta_p[valid])).mean()),3) if valid.any() else None
    return out


def metrics(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    assert y.shape==p.shape and np.isfinite(p).all() and (p>=0).all()
    den=y+p
    smape=np.divide(200*np.abs(y-p),den,out=np.zeros_like(den),where=den>0).mean()
    wape=100*np.abs(y-p).sum()/max(y.sum(),1e-9)
    rmse=np.sqrt(np.mean((y-p)**2))
    sums_y=y.sum(axis=0);sums_p=p.sum(axis=0);ok=sums_y>0
    bias=(100*np.abs(sums_y[ok]-sums_p[ok])/sums_y[ok]).mean() if ok.any() else 0
    rmsle=np.sqrt(np.mean((np.log1p(y)-np.log1p(p))**2))
    return {'smape':round(float(smape),5),'wape':round(float(wape),5),
       'rmse':round(float(rmse),5),'bias':round(float(bias),5),'rmsle':round(float(rmsle),6),
       'sum_pred':round(float(p.sum()),2),'sum_actual':round(float(y.sum()),2)}


def validate_submission(df,reference=None):
    required=['week','warehouse_name','article','quantity']
    assert list(df.columns)==required
    assert len(df)==9*11*799 and not df.duplicated(required[:3]).any()
    assert np.isfinite(df.quantity.to_numpy()).all() and (df.quantity>=0).all()
    d=pd.to_datetime(df.week)
    assert d.min()==pd.Timestamp('2026-07-27')
    assert sorted(d.unique())==list(pd.date_range('2026-07-27',periods=9,freq='7D'))
    if reference is not None:
        ref=pd.read_csv(reference,usecols=required[:3]);k=required[:3]
        a=pd.MultiIndex.from_frame(ref[k]);b=pd.MultiIndex.from_frame(df[k]);assert len(a.symmetric_difference(b))==0
    return True
