#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nonton v7 / M5-lite: cutoff-safe weekly forecasts, channel-proxy feature engineering.

Do not edit original ZIP. This is an independent pipeline from RAW sales+stock.
`composite_division` is a DEPARTMENT, not a proven online/offline channel.
M5-lite is inspired by partial pooling and direct/recursive LightGBM forecasts;
not a replica of the original 220-model M5 solution.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb

START = pd.Timestamp('2024-01-01')
CUTOFF = pd.Timestamp('2026-07-26')
FINAL = pd.Timestamp('2026-07-27')
N_WEEKS = 134
N_H = 9
WINDOWS = (4, 13, 26, 52)
WEIGHTS = {'SMAPE':.50, 'WAPE':.20, 'RMSE':.15, 'Bias':.10, 'RMSLE':.05}
CATS = ['warehouse_id','article_id','category1_id','category2_id','characteristic_id']


def arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-zip',type=Path,required=True,help='Original hakaton(1).zip with raw sales and stock')
    p.add_argument('--out',type=Path,default=Path('results_v7'))
    p.add_argument('--mode',choices=['global','m5lite','both'],default='both')
    p.add_argument('--folds',nargs='+',default=['2025-10-27','2026-01-05','2026-03-16'])
    p.add_argument('--holdout',default='2026-05-25')
    p.add_argument('--max-origins',type=int,default=20,help='Most recent training origins; 0=all')
    p.add_argument('--rounds',type=int,default=200)
    p.add_argument('--threads',type=int,default=6)
    p.add_argument('--half-life',type=float,default=26.)
    p.add_argument('--top-departments',type=int,default=8)
    p.add_argument('--top-cities',type=int,default=5)
    p.add_argument('--min-group-pairs',type=int,default=12)
    p.add_argument('--min-group-train-rows',type=int,default=2500)
    p.add_argument('--max-group-models',type=int,default=24,help='Maximum warehouse-category models per fold; 0=all')
    p.add_argument('--enable-category2',action='store_true',help='Extra M5-like pool: warehouse x category_lvl_2')
    p.add_argument('--save-models',action='store_true',help='Save trained final refit models; may use significant disk space')
    p.add_argument('--ablate-department-city',action='store_true',help='Remove department and city history features for an otherwise matched GLOBAL control experiment')
    p.add_argument('--scales',nargs='+',type=float,default=[.85,1.0,1.15,1.30])
    p.add_argument('--thresholds',nargs='+',type=float,default=[0.,.4,.6,.8,1.])
    p.add_argument('--dev-only',action='store_true',help='Only development folds, no holdout, no final forecast')
    p.add_argument('--skip-final',action='store_true',help='Run folds+holdout without final fit')
    p.add_argument('--skip-holdout',action='store_true',help='Do not inspect holdout')
    p.add_argument('--resume-folds',action='store_true',help='Reuse complete per-fold raw predictions in the output directory')
    p.add_argument('--smoke',action='store_true',help='Fast synthetic/data wiring check: 2 origins, 10 trees, no groups, dev only')
    p.add_argument('--seed',type=int,default=42)
    a = p.parse_args()
    if a.ablate_department_city and a.mode != 'global':
        p.error('--ablate-department-city is currently supported only in --mode global')
    if a.smoke:
        a.rounds=10; a.max_origins=2; a.max_group_models=4; a.dev_only=True
        a.folds=a.folds[-1:]
        a.min_group_train_rows=12; a.min_group_pairs=2
    if a.threads<1 or a.rounds<1 or a.top_departments<0 or a.top_cities<0 or a.min_group_pairs<1:
        p.error('Invalid limits')
    if a.max_origins<0 or a.max_group_models<0 or a.half_life<=0: p.error('Invalid training controls')
    if a.folds!=sorted(set(a.folds)):p.error('Folds must be unique and increasing')
    if len(a.scales)*len(a.thresholds)>200:p.error('Calibration grid too large')
    if not all(v>0 for v in a.scales) or not all(v>=0 for v in a.thresholds):p.error('Bad scales/thresholds')
    return a


@dataclass
class Dataset:
    weeks: pd.DatetimeIndex
    warehouses: list[str]
    articles: list[str]
    names_wh: np.ndarray
    names_article: np.ndarray
    wh_ids: np.ndarray
    article_ids: np.ndarray
    cat1_ids: np.ndarray
    cat2_ids: np.ndarray
    char_ids: np.ndarray
    first_article_week: np.ndarray
    first_pair_week: np.ndarray
    qty: np.ndarray
    sale_days: np.ndarray
    stock_last: np.ndarray
    stock_mean: np.ndarray
    stock_days: np.ndarray
    department: np.ndarray
    city: np.ndarray
    dep_buckets: list[str]
    city_buckets: list[str]
    div_observation: np.ndarray
    discount_observation: np.ndarray
    log_sales_sum: np.ndarray
    log_sales_count: np.ndarray
    wh_cat1: np.ndarray
    wh_cat2: np.ndarray
    article_cat1: np.ndarray
    n_pairs: int


def _find_zip_member(z, suffix):
    members=[x for x in z.namelist() if x.endswith(suffix)]
    if len(members)!=1:
        raise ValueError('Expected exactly one member ending '+suffix+'; got '+str(members))
    return members[0]


def _stable_buckets(sales, name, max_top, weeks):
    """Pick feature buckets by first 26 historical weeks only (safe for all origins >=26)."""
    col=sales[name].astype('string').fillna('__MISSING__').astype(str)
    freq=(pd.DataFrame({'x':col[sales['_wi']<26], 'q':sales.loc[sales['_wi']<26,'quantity']})
          .groupby('x',observed=True).q.sum())
    # Deterministic tie handling.
    top=sorted(freq.index,key=lambda x:(-float(freq.loc[x]),str(x)))[:max_top]
    return top+['__OTHER__']


def _dense_buckets(sales, source, buckets, wcode, acode, weekcode, npairs, nw):
    # Never index unavailable classes using future values. OTHER includes groups not in top bucket list.
    mapping={x:i for i,x in enumerate(buckets[:-1])}
    tags=sales[source].astype('string').fillna('__MISSING__').astype(str).map(mapping).fillna(len(buckets)-1).to_numpy(dtype=np.int32)
    idx=(weekcode.astype(np.int64)*npairs+wcode.astype(np.int64)*nw+acode.astype(np.int64))
    out=np.zeros((len(buckets),N_WEEKS,npairs),dtype=np.float32)
    for i in range(len(buckets)):
        m=tags==i
        if m.any():out[i]=np.bincount(idx[m],weights=sales.quantity.to_numpy(dtype=np.float32)[m],minlength=N_WEEKS*npairs).reshape(N_WEEKS,npairs)
    return out


def load_raw(path:Path,top_departments=8,top_cities=5):
    t0=time.time()
    if not path.is_file():raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as z:
        sn=_find_zip_member(z,'sales_dataset_hakaton_train.csv')
        stn=_find_zip_member(z,'stock_dataset_hakaton_train.csv')
        with z.open(sn) as f:
            s=pd.read_csv(f,low_memory=False,dtype={'warehouse_name':'category','article':'category',
               'composite_division':'category','cities_name':'category','category_lvl_1':'category',
               'category_lvl_2':'category','characteristic':'category','quantity':'float32',
               'sales_sum':'float32','discount_sum':'float32'})
        needed={'date','warehouse_name','article','composite_division','cities_name','quantity','category_lvl_1','category_lvl_2','characteristic'}
        if not needed.issubset(s):raise ValueError('Missing raw sales columns: '+str(needed-set(s)))
        s['date']=pd.to_datetime(s.date,format='%Y-%m-%d',errors='raise')
        excluded=int((s.date>CUTOFF).sum())
        s=s[(s.date>=START)&(s.date<=CUTOFF)].copy()
        if not np.isfinite(s.quantity).all() or (s.quantity<0).any():raise ValueError('Invalid sales quantity')
        wh=sorted(map(str,s.warehouse_name.cat.categories))
        arts=sorted(map(str,s.article.cat.categories))
        if len(wh)!=11 or len(arts)!=799:
            print('WARNING: source has',len(wh),'warehouses,',len(arts),'articles; full 11x799 expected for competition')
        nwh=len(wh);na=len(arts);npairs=nwh*na
        wcode=pd.Categorical(s.warehouse_name.astype(str),categories=wh).codes.astype(np.int32)
        acode=pd.Categorical(s.article.astype(str),categories=arts).codes.astype(np.int32)
        s['_wi']=((s.date-START).dt.days//7).astype('int16')
        wi=s['_wi'].to_numpy(dtype=np.int32)
        pidx=wcode*na+acode
        k=wi.astype(np.int64)*npairs+pidx
        size=N_WEEKS*npairs
        qty=np.bincount(k,weights=s.quantity,minlength=size).astype('float32').reshape(N_WEEKS,npairs)
        day_code=(s.date-START).dt.days.to_numpy(dtype=np.int32)
        sale_day_unique=np.unique(day_code.astype(np.int64)*npairs+pidx)
        day_week_key=(sale_day_unique//npairs//7)*npairs+(sale_day_unique%npairs)
        sale_days=np.bincount(day_week_key,minlength=size).astype('float32').reshape(N_WEEKS,npairs)
        dep_b=_stable_buckets(s,'composite_division',top_departments,wi)
        city_b=_stable_buckets(s,'cities_name',top_cities,wi)
        department=_dense_buckets(s,'composite_division',dep_b,wcode,acode,wi, npairs,na)
        city=_dense_buckets(s,'cities_name',city_b,wcode,acode,wi,npairs,na)
        # Historical transformed sales and discount are proxies, not future-known promo info.
        def known_count(col):
            values=s[col].to_numpy(dtype=np.float64) if col in s else np.full(len(s),np.nan)
            good=np.isfinite(values)
            counts=np.bincount(k[good],minlength=size).astype('float32').reshape(N_WEEKS,npairs)
            total=np.bincount(k[good],weights=values[good],minlength=size).astype('float32').reshape(N_WEEKS,npairs)
            return total,counts
        logs,logcount=known_count('sales_sum')
        _,discountcount=known_count('discount_sum')
        # Division distinct row-count, aggregated week/pair; no actual channel identity inferred.
        div_observation=np.bincount(k,minlength=size).astype('float32').reshape(N_WEEKS,npairs)
        # Stable article metadata: first known sale only. Mask when origin <= first observed.
        first=s.sort_values('date',kind='stable').drop_duplicates('article',keep='first')
        first.index=first.article.astype(str)
        first=first.reindex(arts)
        mapping={};ids={}
        for src in ['category_lvl_1','category_lvl_2','characteristic']:
            labels=first[src].astype('string').fillna('__MISSING__').astype(str).tolist()
            choices=sorted(set(labels))
            mapping[src]={v:i for i,v in enumerate(choices)}
            ids[src]=np.array([mapping[src][v] for v in labels],dtype=np.int32)
        whids=np.repeat(np.arange(nwh,dtype=np.int32),na)
        artids=np.tile(np.arange(na,dtype=np.int32),nwh)
        cat1=np.tile(ids['category_lvl_1'],nwh)
        cat2=np.tile(ids['category_lvl_2'],nwh)
        char=np.tile(ids['characteristic'],nwh)
        first_art=(pd.to_datetime(first.date)-START).dt.days.to_numpy(dtype=np.float64)//7
        first_art=np.nan_to_num(first_art,nan=N_WEEKS).astype(np.int32)
        first_art_pair=first_art[artids]
        first_pair=np.full(npairs,N_WEEKS+10,dtype=np.int32)
        for week_idx in range(N_WEEKS):
            new=(sale_days[week_idx]>0)&(first_pair==N_WEEKS+10)
            first_pair[new]=week_idx
        del s,first,wcode,acode,wi,pidx,k,day_code,sale_day_unique,day_week_key
        gc.collect()
        print('RAW SALES: history cutoff',CUTOFF.date(),'excluded future rows',excluded,
              'pairs',npairs,'department buckets',dep_b,'city buckets',city_b,flush=True)
        with z.open(stn) as f:
            stock=pd.read_csv(f,low_memory=False,dtype={'warehouse_name':'category','article':'category','stock':'float32'})
    stock['date']=pd.to_datetime(stock.date,format='%Y-%m-%d',errors='raise')
    stock=stock[(stock.date>=START)&(stock.date<=CUTOFF)].copy()
    if 'article' not in stock and 'char_article' in stock:stock=stock.rename(columns={'char_article':'article'})
    if not np.isfinite(stock.stock).all():raise ValueError('Invalid stock')
    # All repeats must be SUMMED, including completely identical rows. Confirmed in current FAQ.
    daily=(stock.groupby(['date','warehouse_name','article'],observed=True,sort=False)
             .stock.sum().reset_index())
    del stock;gc.collect()
    daily=daily.sort_values('date',kind='stable')
    wsi=pd.Categorical(daily.warehouse_name.astype(str),categories=wh).codes.astype(np.int32)
    asi=pd.Categorical(daily.article.astype(str),categories=arts).codes.astype(np.int32)
    if (wsi<0).any() or (asi<0).any():raise ValueError('Stock has unknown pair outside sales universe')
    dwi=((daily.date-START).dt.days//7).to_numpy(dtype=np.int32)
    sid=dwi.astype(np.int64)*npairs+wsi*na+asi
    daily['sid']=sid
    # Last observed daily STOCK snapshot per week after SUM aggregation; not necessarily Sunday.
    weekly=(daily.groupby('sid',sort=False,observed=True).stock
             .agg(['mean','size','last']).reset_index())
    idx=weekly.sid.to_numpy(dtype=np.int64)
    st_last=np.full(size,np.nan,dtype=np.float32); st_last[idx]=weekly['last'].to_numpy(dtype=np.float32)
    st_mean=np.full(size,np.nan,dtype=np.float32); st_mean[idx]=weekly['mean'].to_numpy(dtype=np.float32)
    st_days=np.zeros(size,dtype=np.float32); st_days[idx]=weekly['size'].to_numpy(dtype=np.float32)
    st_last=st_last.reshape(N_WEEKS,npairs);st_mean=st_mean.reshape(N_WEEKS,npairs);st_days=st_days.reshape(N_WEEKS,npairs)
    names_wh=np.repeat(np.asarray(wh,dtype=object),na)
    names_article=np.tile(np.asarray(arts,dtype=object),nwh)
    names1=np.array([f'{x}:{y}' for x,y in zip(whids,cat1)],dtype=object)
    names2=np.array([f'{x}:{y}' for x,y in zip(whids,cat2)],dtype=object)
    print('RAW STOCK: summed duplicates, unique observed week-pairs',len(weekly),
          'elapsed seconds',round(time.time()-t0,1),flush=True)
    return Dataset(weeks=pd.date_range(START,periods=N_WEEKS,freq='7D'),warehouses=wh,articles=arts,
       names_wh=names_wh,names_article=names_article,wh_ids=whids,article_ids=artids,
       cat1_ids=cat1,cat2_ids=cat2,char_ids=char,
       first_article_week=first_art_pair,first_pair_week=first_pair,
       qty=qty,sale_days=sale_days,stock_last=st_last,stock_mean=st_mean,stock_days=st_days,
       department=department,city=city,dep_buckets=dep_b,city_buckets=city_b,
       div_observation=div_observation,discount_observation=discountcount,
       log_sales_sum=logs,log_sales_count=logcount,
       wh_cat1=names1,wh_cat2=names2,article_cat1=ids['category_lvl_1'],n_pairs=npairs)


def _rolling(arr,end,width):
    return arr[max(0,end-width):end]


def _bucket_features(prefix,values,origin,out,n_pairs):
    total=values.sum(axis=0)
    for w in [4,13,26]:
        d=values[:,max(0,origin-w):origin,:].sum(axis=1)
        s=d.sum(axis=0)
        shares=np.divide(d,s[None,:],out=np.zeros_like(d),where=s[None,:]>0)
        out[f'{prefix}_count_{w}w']=(d>0).sum(axis=0).astype(np.float32)
        out[f'{prefix}_topshare_{w}w']=shares.max(axis=0)
        out[f'{prefix}_entropy_{w}w']=-(np.where(shares>0,shares*np.log(np.maximum(shares,1.e-8)),0)).sum(axis=0)
        if w in [4,13]:
            for b in range(values.shape[0]):out[f'{prefix}_share_b{b}_{w}w']=shares[b].astype('float32')
        if w==4:share4=shares
        if w==13:
            out[f'{prefix}_mix_shift_4_13']=(np.abs(share4-shares).sum(axis=0)).astype('float32')


def make_features(d:Dataset,origin_idx:int, qty_override=None,cutoff_origin=None,recursive=False):
    """Direct: n_pairs*9, recursive: n_pairs*1, all features available as of cutoff_origin.
    qty_override extends the truth up to cutoff with model-generated predictions only.
    """
    end=origin_idx
    if cutoff_origin is None:cutoff_origin=origin_idx
    history = d.qty[:min(end,cutoff_origin)] if qty_override is None else qty_override[:end]
    assert len(history)==end and cutoff_origin<=len(d.weeks) and end>=26
    if qty_override is not None:
        assert end>=cutoff_origin
    n=d.n_pairs
    horizons=np.array([1],dtype=np.int32) if recursive else np.arange(1,10,dtype=np.int32)
    nh=len(horizons)
    origin_date=d.weeks[0]+pd.Timedelta(weeks=origin_idx)
    target_dates=pd.DatetimeIndex(origin_date+pd.to_timedelta((horizons-1)*7,unit='D')) if not recursive else pd.DatetimeIndex([origin_date])
    # for recursive h1, target calendar is the current step.
    pair={
      'warehouse_id':d.wh_ids.astype(np.float32),
      'article_id':d.article_ids.astype(np.float32),
      'category1_id':d.cat1_ids.astype(np.float32).copy(),
      'category2_id':d.cat2_ids.astype(np.float32).copy(),
      'characteristic_id':d.char_ids.astype(np.float32).copy(),
    }
    # Article metadata is unknown for an origin preceding first observation; keep IDs missing.
    unseen=d.first_article_week>=cutoff_origin
    for k in ['category1_id','category2_id','characteristic_id']:
        pair[k][unseen]=np.nan
    for lag in [1,2,3,4,8,13,26,52]:
        pair[f'qty_lag_{lag}']=history[-lag].astype(np.float32) if end>=lag else np.full(n,np.nan,dtype=np.float32)
    for w in WINDOWS:
        x=_rolling(history,end,w)
        pair[f'qty_mean_{w}w']=x.mean(axis=0)
        pair[f'qty_std_{w}w']=x.std(axis=0)
        pair[f'qty_nz_{w}w']=(x>0).sum(axis=0).astype(np.float32)
        pair[f'qty_max_{w}w']=x.max(axis=0)
    pair['qty_history_mean']=history.mean(axis=0)
    pair['qty_sum_history']=history.sum(axis=0)
    pair['qty_trend_4v13']=(history[-4:].mean(axis=0)/(history[-13:].mean(axis=0)+.1)).astype('float32')
    pair['qty_trend_13v26']=(history[-13:].mean(axis=0)/(history[-26:].mean(axis=0)+.1)).astype('float32')
    nz=(history>0)
    seen=nz.any(axis=0)
    pair['qty_weeks_since_sale']=np.where(seen,np.argmax(nz[::-1],axis=0),-1).astype(np.float32)
    pair['qty_weeks_since_first_sale']=np.where(seen,end-1-np.argmax(nz,axis=0),-1).astype(np.float32)
    pair['qty_seen']=seen.astype(np.float32)
    pair['qty_history_length']=np.full(n,end,dtype=np.float32)
    # Historical stock and departments must stop at the ORIGIN for recursive predictions.
    st=d.stock_last[:cutoff_origin]
    ss=d.stock_mean[:cutoff_origin]
    if len(st):
        pair['stock_last_lag_1']=st[-1]
        pair['stock_mean_lag_1']=ss[-1]
        latest=_rolling(st,cutoff_origin,4)
        latest_count=np.isfinite(latest).sum(axis=0)
        pair['stock_last_mean_4w']=np.divide(np.nansum(latest,axis=0),latest_count,
            out=np.full(n,np.nan,dtype=np.float32),where=latest_count>0)
        pair['stock_observed_13w']=np.isfinite(_rolling(st,cutoff_origin,13)).sum(axis=0).astype(np.float32)
        finite=np.isfinite(st); ever=finite.any(axis=0)
        last_idx=cutoff_origin-1-np.argmax(finite[::-1],axis=0)
        pair['stock_last_known']=np.where(ever,st[np.maximum(last_idx,0),np.arange(n)],np.nan)
        pair['stock_age_weeks']=np.where(ever,end-1-last_idx,-1).astype(np.float32)
    else:raise ValueError('Insufficient historical stock')
    sl=d.sale_days[:cutoff_origin]
    pair['sale_days_last_week']=sl[-1]
    pair['sale_days_8w']=_rolling(sl,cutoff_origin,8).sum(axis=0)
    # Group totals computed from real history or predicted future, not true future.
    gr4=history[-4:].mean(axis=0)
    gr13=history[-13:].mean(axis=0)
    for prefix,values,labels in [('warehouse',gr4,d.wh_ids),('category1',gr4,d.cat1_ids),
                                 ('wh_category1',gr4,d.wh_cat1)]:
        _,inv=np.unique(labels,return_inverse=True)
        ag=np.bincount(inv,weights=values)
        ct=np.bincount(inv)
        pair[f'{prefix}_total_4w']=ag[inv].astype(np.float32)
        pair[f'{prefix}_mean_4w']=(ag[inv]/np.maximum(ct[inv],1)).astype(np.float32)
    _,inv=np.unique(d.wh_ids,return_inverse=True)
    pair['warehouse_total_13w']=np.bincount(inv,weights=gr13)[inv].astype('float32')
    # Channel/department and city proxies: freeze at origin, no future group proportions.
    dept=d.department[:,:cutoff_origin,:]
    city=d.city[:,:cutoff_origin,:]
    _bucket_features('department',dept,cutoff_origin,pair,n)
    _bucket_features('city',city,cutoff_origin,pair,n)
    for field,mat in [('division_rows',d.div_observation),
                      ('discount_obs',d.discount_observation),('log_sales',d.log_sales_sum),
                      ('log_sales_cnt',d.log_sales_count)]:
        hist=mat[:cutoff_origin]
        pair[field+'_13w']=_rolling(hist,cutoff_origin,13).sum(axis=0)
    pair['discount_per_sale_row_13w']=(pair['discount_obs_13w']/(pair['division_rows_13w']+.01)).astype('float32')
    pair['log_sales_avg_13w']=(pair['log_sales_13w']/(pair['log_sales_cnt_13w']+.01)).astype('float32')
    # Future calendar values are legitimately known. Each origin emits horizon target rows.
    pair_data={k:np.repeat(np.asarray(v,dtype=np.float32),nh) for k,v in pair.items()}
    iso=np.asarray(target_dates.isocalendar().week,dtype=np.float64)
    pair_data.update({
       'horizon':np.tile(horizons,n).astype(np.float32),
       'target_week_index':np.tile(iso,n).astype(np.float32),
       'target_month':np.tile(np.asarray(target_dates.month),n).astype(np.float32),
       'target_year':np.tile(np.asarray(target_dates.year),n).astype(np.float32),
       'target_week_sin':np.tile(np.sin(2*np.pi*iso/52.1775),n).astype(np.float32),
       'target_week_cos':np.tile(np.cos(2*np.pi*iso/52.1775),n).astype(np.float32),
       'origin_index':np.full(n*nh,cutoff_origin,dtype=np.float32)
    })
    lastyear=np.full((n,nh),np.nan,dtype=np.float32)
    # IMPORTANT: year-back target week can be after real origin for far future? 52 weeks back is safe here.
    for h_i,h in enumerate(horizons):
        k=origin_idx+int(h)-1-52
        if 0<=k<cutoff_origin:lastyear[:,h_i]=d.qty[k]
    pair_data['qty_last_year_same_week']=lastyear.ravel()
    X=pd.DataFrame(pair_data,dtype=np.float32)
    return X


def observed_and_targets(d,origin_idx):
    assert origin_idx+N_H<=N_WEEKS
    return d.qty[origin_idx:origin_idx+N_H].T.ravel().astype(np.float32)


def train_matrix(d,origins):
    Xs=[];ys=[];inds=[]
    for origin in origins:
        X=make_features(d,int(origin))
        y=observed_and_targets(d,int(origin))
        # Avoid treating pre-launch all-zero dummy grid as a valid training segment.
        known=(d.first_pair_week<int(origin)) | (d.stock_days[:int(origin)]>0).any(axis=0)
        mask=np.repeat(known,N_H)
        Xs.append(X.iloc[mask].reset_index(drop=True)); ys.append(y[mask]);inds.append(np.full(mask.sum(),int(origin),dtype=np.int16))
    return pd.concat(Xs,ignore_index=True),np.concatenate(ys),np.concatenate(inds)


def configure_features(X):
    # LightGBM can directly handle numeric NaNs; categories must be fixed integer-coded.
    for c in CATS:
        if c in X: X[c]=X[c].astype('category')
    return X


def fit_lgb(X,y,weights,objective,rounds,threads,seed,recursive=False):
    assert len(X)==len(y)==len(weights)
    params={'objective':objective,'learning_rate':.045,'num_leaves':31,
       'min_data_in_leaf':120,'lambda_l2':5.,'feature_fraction':.95,
       'verbosity':-1,'num_threads':threads,'seed':seed,'deterministic':True,'force_col_wise':True}
    if objective=='tweedie':params['tweedie_variance_power']=1.3
    ds=lgb.Dataset(configure_features(X.copy()),label=y,weight=weights,
       categorical_feature=[c for c in CATS if c in X],free_raw_data=True)
    return lgb.train(params,ds,num_boost_round=rounds)


def pred_lgb(m,X):
    return np.maximum(np.nan_to_num(m.predict(configure_features(X.copy()),num_threads=1),nan=0.,posinf=0.,neginf=0.),0.).astype(np.float32)


def group_indexes(d,level):
    if level=='warehouse':return d.wh_ids
    if level=='whcat1':return d.wh_cat1
    if level=='whcat2':return d.wh_cat2
    raise ValueError(level)


def grouped_train_predict(d,Xtr,ytr,weights,Xval,kind,a,fallback,save_dir=None):
    """Group-wise partial pooling; train each group on only its observed training members.
    A group too small or outside model budget receives a prior coarser prediction.
    """
    output=fallback.copy()
    trwh=Xtr['warehouse_id'].to_numpy(dtype=np.float32)
    vawh=Xval['warehouse_id'].to_numpy(dtype=np.float32)
    trcat1=Xtr['category1_id'].to_numpy(dtype=np.float32)
    vacat1=Xval['category1_id'].to_numpy(dtype=np.float32)
    trcat2=Xtr['category2_id'].to_numpy(dtype=np.float32)
    vacat2=Xval['category2_id'].to_numpy(dtype=np.float32)
    if kind=='warehouse':
        keys=sorted(set(map(int,d.wh_ids)))
        matches=lambda i:(trwh==i,vawh==i)
    elif kind=='whcat1':
        keys=sorted(set(zip(d.wh_ids.tolist(),d.cat1_ids.tolist())))
        matches=lambda i:((trwh==i[0])&(trcat1==i[1]),(vawh==i[0])&(vacat1==i[1]))
    else:
        keys=sorted(set(zip(d.wh_ids.tolist(),d.cat2_ids.tolist())))
        matches=lambda i:((trwh==i[0])&(trcat2==i[1]),(vawh==i[0])&(vacat2==i[1]))
    viable=[]
    for key in keys:
        tm,vm=matches(key)
        count_pairs=(vm.sum()/N_H)
        if vm.any() and tm.sum()>=a.min_group_train_rows and count_pairs>=a.min_group_pairs:
            viable.append((int(tm.sum()),key))
    viable.sort(key=lambda x:(-x[0],str(x[1])))
    if kind!='warehouse' and a.max_group_models:viable=viable[:a.max_group_models]
    stats={'kind':kind,'total_keys':len(keys),'trained_groups':len(viable),'fallback_target_rows':0}
    covered=np.zeros(len(Xval),dtype=bool)
    for k,(count,key) in enumerate(viable):
        tm,vm=matches(key)
        model=fit_lgb(Xtr.loc[tm],ytr[tm],weights[tm],'tweedie',a.rounds,a.threads,a.seed)
        output[vm]=pred_lgb(model,Xval.loc[vm]);covered|=vm
        if save_dir is not None:
            name=(str(key).replace(' ','').replace('(','').replace(')','').replace(',','_')
                  .replace("'",'').replace(':','_'))
            model.save_model(str(save_dir/(kind+'__'+name+'.txt')))
        del model
        if k%20==0 or k==len(viable)-1:print(f'  {kind} group {k+1}/{len(viable)} ({count} rows)',flush=True)
    stats['fallback_target_rows']=int((~covered).sum())
    return output,stats


def recursive_forecast(d,Xtr,ytr,weights,eval_origin,objective,a,save_dir=None):
    # Train only horizon 1 labels at historical origins: different model from 9-horizon direct.
    step=(Xtr['horizon'].to_numpy()==1)
    model=fit_lgb(Xtr.loc[step],ytr[step],weights[step],objective,a.rounds,a.threads,a.seed,recursive=True)
    hist=np.zeros((eval_origin+N_H,d.n_pairs),dtype=np.float32)
    hist[:eval_origin]=d.qty[:eval_origin]
    out=np.zeros((d.n_pairs,N_H),dtype=np.float32)
    for h in range(N_H):
        step_origin=eval_origin+h
        x=make_features(d,step_origin,qty_override=hist,cutoff_origin=eval_origin,recursive=True)
        p=pred_lgb(model,x)
        hist[step_origin]=p
        out[:,h]=p
        del x
    if save_dir is not None:model.save_model(str(save_dir/('recursive_'+objective+'.txt')))
    return out.ravel(),model


def models_for_origin(d,train_origins,pred_origin,a,save_dir=None):
    tic=time.perf_counter()
    Xtr,ytr,idx=train_matrix(d,train_origins)
    Xval=make_features(d,pred_origin)
    if a.ablate_department_city:
        excluded=[c for c in Xtr.columns if c.startswith(('department_', 'city_'))]
        Xtr=Xtr.drop(columns=excluded)
        Xval=Xval.drop(columns=excluded)
        print('  ABLATION: removed',len(excluded),'department/city features',flush=True)
    weight=np.exp2(-(np.max(idx)-idx)/a.half_life).astype(np.float32)
    out={};details={}
    print(f'  TRAIN {len(Xtr):,} rows / {len(Xtr.columns)} features, origin {pred_origin}, {len(Xval):,} target rows',flush=True)
    if save_dir is not None:
        save_dir.mkdir(parents=True,exist_ok=True)
        (save_dir/'schema.json').write_text(json.dumps({'features':list(Xtr.columns),
           'categorical_features':CATS,'dep_buckets':d.dep_buckets,'city_buckets':d.city_buckets,
           'cutoff':str(CUTOFF.date())},ensure_ascii=False,indent=2),encoding='utf-8')
    global_model=fit_lgb(Xtr,ytr,weight,'tweedie',a.rounds,a.threads,a.seed)
    if save_dir is not None:global_model.save_model(str(save_dir/'global_tweedie.txt'))
    base=pred_lgb(global_model,Xval)
    out['v7_global_raw']=base
    # Persist full refit baseline model in memory only; caller may save it via return value.
    if a.mode in ('m5lite','both'):
        poisson=fit_lgb(Xtr,ytr,weight,'poisson',a.rounds,a.threads,a.seed)
        if save_dir is not None:poisson.save_model(str(save_dir/'global_poisson.txt'))
        ppoisson=pred_lgb(poisson,Xval)
        wh,st1=grouped_train_predict(d,Xtr,ytr,weight,Xval,'warehouse',a,base,save_dir)
        wc1,st2=grouped_train_predict(d,Xtr,ytr,weight,Xval,'whcat1',a,wh,save_dir)
        rec_t,_=recursive_forecast(d,Xtr,ytr,weight,pred_origin,'tweedie',a,save_dir)
        rec_p,_=recursive_forecast(d,Xtr,ytr,weight,pred_origin,'poisson',a,save_dir)
        components=[base,ppoisson,wh,wc1,rec_t,rec_p]
        details['m5_components']=['global_tweedie','global_poisson','warehouse_tweedie',
                                   'warehouse_category1_tweedie','recursive_tweedie','recursive_poisson']
        details['pooling']=[st1,st2]
        if a.enable_category2:
            wc2,st3=grouped_train_predict(d,Xtr,ytr,weight,Xval,'whcat2',a,wc1,save_dir)
            components.append(wc2)
            details['m5_components'].append('warehouse_category2_tweedie')
            details['pooling'].append(st3)
        out['m5lite_raw']=np.mean(np.stack(components,axis=0),axis=0,dtype=np.float64).astype('float32')
        out['hybrid_raw']=(.5*base+.5*out['m5lite_raw']).astype(np.float32)
        del poisson,wh,wc1,rec_t,rec_p,ppoisson
    print('  done, minutes',round((time.perf_counter()-tic)/60,2),flush=True)
    del Xtr,Xval,ytr,idx,weight,global_model
    gc.collect()
    return out,details


def metric(y,p):
    y=np.asarray(y,dtype=np.float64).reshape(-1,N_H)
    p=np.asarray(p,dtype=np.float64).reshape(-1,N_H)
    if not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid predictions')
    smape=np.divide(200*np.abs(y-p),y+p,out=np.zeros_like(y),where=(y+p)>0)
    good=y.sum(axis=1)>0
    actual_total=float(y.sum())
    # Official all-zero ground-truth edge case: WAPE = 0 for all-zero pred, else 100.
    wape=(100*np.abs(y-p).sum()/actual_total) if actual_total>0 else (0. if p.sum()==0 else 100.)
    signed=(100*(p.sum()-actual_total)/actual_total) if actual_total>0 else (0. if p.sum()==0 else 100.)
    return {'SMAPE':float(smape.mean()),
      'WAPE':float(wape),
      'RMSE':float(np.sqrt(np.square(y-p).mean())),
      'Bias':float(100*np.mean(np.abs(p.sum(axis=1)[good]-y.sum(axis=1)[good])/y.sum(axis=1)[good])) if good.any() else 0.,
      'RMSLE':float(np.sqrt(np.square(np.log1p(p)-np.log1p(y)).mean())),
      'signed_global_bias':float(signed),
      'zero_predictions_pct':float(100*(p==0).mean()),
      'actual_sum':float(y.sum()),'predicted_sum':float(p.sum())}


def postprocess(raw,scale,threshold):
    x=np.maximum(np.asarray(raw,dtype=np.float64)*scale,0.)
    x[x<threshold]=0.
    return x


def unnormalized_score(m,baseline):
    # Local surrogate only; organizer leaderboard currently uses ranks.
    return float(sum(WEIGHTS[k]*min(m[k]/baseline[k],1) if baseline[k]>0 else WEIGHTS[k]
                     for k in WEIGHTS))


def choose_on_dev(folds_data,a):
    all_preds=sorted({name for _,p in folds_data for name in p if not name.startswith('_')})
    rows=[]
    for name in all_preds:
        for scale in a.scales:
            for threshold in a.thresholds:
                scores=[];metrics=[]
                for actual,pred in folds_data:
                    current=metric(actual,postprocess(pred[name],scale,threshold))
                    # Compare to LAST HISTORICAL WEEK repeated for nine horizons.
                    y,pbaseline=actual,pred['_baseline']
                    base_metric=metric(y,pbaseline)
                    metrics.append(current);scores.append(unnormalized_score(current,base_metric))
                agg={k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
                rows.append({'candidate':name,'scale':scale,'threshold':threshold,
                   'surrogate_score':float(np.mean(scores)),
                   'surrogate_score_std':float(np.std(scores)),**agg})
    rows.sort(key=lambda x:(x['surrogate_score'],x['surrogate_score_std'],x['SMAPE']))
    return pd.DataFrame(rows)


def export_prediction(d,origin,raw,chosen,out):
    arr=postprocess(raw,chosen['scale'],chosen['threshold'])
    # pair-major, nine horizon predictions per pair
    weeks=(FINAL+pd.to_timedelta(np.arange(9)*7,unit='D')).strftime('%Y-%m-%d').to_numpy()
    frame=pd.DataFrame({'week':np.tile(weeks,d.n_pairs),
                        'warehouse_name':np.repeat(d.names_wh,N_H),
                        'article':np.repeat(d.names_article,N_H),'quantity':arr})
    frame=frame[['week','warehouse_name','article','quantity']]
    assert len(frame)==d.n_pairs*N_H and frame.week.nunique()==N_H
    assert not frame.duplicated(['week','warehouse_name','article']).any()
    assert frame.quantity.notna().all() and np.isfinite(frame.quantity).all() and (frame.quantity>=0).all()
    frame.to_csv(out,index=False,float_format='%.7g')
    return {'rows':len(frame),'total':float(frame.quantity.sum()),'zero_pct':float((frame.quantity==0).mean()*100),
            'all_zero_pairs':int((frame.quantity.to_numpy().reshape(-1,9)==0).all(axis=1).sum())}


def save_window_preds(d,origin,actual,preds,path):
    nh=N_H
    weeks=(START+pd.to_timedelta(origin*7,unit='D')+pd.to_timedelta(np.arange(nh)*7,unit='D')).strftime('%Y-%m-%d').to_numpy()
    frame=pd.DataFrame({'week':np.tile(weeks,d.n_pairs),
                        'warehouse_name':np.repeat(d.names_wh,nh),
                        'article':np.repeat(d.names_article,nh),'actual':actual})
    for k,p in preds.items():
        if k!='_baseline':frame[k]=p
    frame.to_csv(path,index=False,compression='gzip',float_format='%.7g')


def origin_idx(week,dt):
    t=pd.Timestamp(dt)
    if t not in week:raise ValueError(f'Fold {dt} not in known complete weeks')
    return int(week.get_loc(t))


def main():
    a=arg_parser()
    a.out.mkdir(parents=True,exist_ok=True)
    d=load_raw(a.raw_zip,a.top_departments,a.top_cities)
    for dt in a.folds+[a.holdout]:
        if not pd.Timestamp(dt).dayofweek==0:raise ValueError('Origin must be Monday')
    folds=[];res=[]
    for date in a.folds:
        oi=origin_idx(d.weeks,date)
        if oi<36 or oi+N_H>N_WEEKS:raise ValueError('Fold too early or too late '+date)
        safe_last=oi-N_H
        candidates=list(range(26,safe_last+1))
        if a.max_origins:candidates=candidates[-a.max_origins:]
        if not candidates:raise ValueError('No safe training origins')
        assert candidates[-1]+N_H-1<oi
        print('DEV',date,'train origins',len(candidates),'last train target',d.weeks[candidates[-1]+8].date(),flush=True)
        fp=a.out/('dev_predictions_'+date+'.csv.gz')
        if a.resume_folds and fp.is_file():
            saved=pd.read_csv(fp)
            actual=observed_and_targets(d,oi)
            if len(saved)!=d.n_pairs*N_H or not np.allclose(saved.actual.to_numpy(),actual,atol=.0001):
                raise ValueError('Cached fold mismatches current raw data '+date)
            expected_weeks=np.tile(pd.date_range(date,periods=N_H,freq='7D').strftime('%Y-%m-%d').to_numpy(),d.n_pairs)
            if not np.array_equal(saved.week.astype(str).to_numpy(),expected_weeks):
                raise ValueError('Cached fold keys mismatch '+date)
            if not np.array_equal(saved.warehouse_name.astype(str),np.repeat(d.names_wh,N_H)) or not np.array_equal(saved.article.astype(str),np.repeat(d.names_article,N_H)):
                raise ValueError('Cached fold warehouse/article mismatch '+date)
            out={k:saved[k].to_numpy(dtype=np.float32) for k in saved.columns if k.endswith('_raw')}
            required={'v7_global_raw'} if a.mode=='global' else {'v7_global_raw','m5lite_raw','hybrid_raw'}
            if not required.issubset(out):raise ValueError('Cached fold missing raw outputs '+date)
            print('  RESUMED cached completed fold',date,flush=True)
        else:
            out,detail=models_for_origin(d,candidates,oi,a)
            actual=observed_and_targets(d,oi)
            save_window_preds(d,oi,actual,out,fp)
        lastweek=d.qty[oi-1]
        out['_baseline']=np.repeat(lastweek,N_H)
        folds.append((actual,out))
        for k,p in out.items():
            if k=='_baseline':continue
            res.append({'fold':date,'model':k,**metric(actual,p)})
        gc.collect()
    pd.DataFrame(res).to_csv(a.out/'dev_raw_metrics.csv',index=False)
    leaderboard=choose_on_dev(folds,a)
    leaderboard.to_csv(a.out/'dev_calibration_grid.csv',index=False)
    best=leaderboard.iloc[0].to_dict()
    best_each={name:leaderboard[leaderboard.candidate==name].iloc[0].to_dict()
               for name in leaderboard.candidate.unique()}
    print('DEV BEST:',{k:best[k] for k in ['candidate','scale','threshold','surrogate_score','SMAPE','WAPE','Bias']},flush=True)
    report={'experiment':'Nonton v7 channel-proxy plus M5-lite','mode':a.mode,
       'official_score_warning':'Surrogate min-max vs previous-week baseline, NOT official rank-based leaderboard score',
       'division_warning':'composite_division is not verified sales_channel',
       'cutoff':str(CUTOFF.date()),'folds':a.folds,'holdout':a.holdout,
       'best_by_dev':best,'calibration_per_candidate':best_each,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
       'source_metadata':{'dep_buckets':d.dep_buckets,'city_buckets':d.city_buckets},
       'notes':['Each fold trains on origins whose last label target is strictly before its validation origin.',
                'Final refit uses only full targets ending <= 2026-07-26.',
                'Original ZIP is read-only. No unknown target-week sales or future stock is used.',
                'Top department/city buckets selected from first 26 historical weeks only.']}
    if best['candidate'].startswith('_'):
        raise AssertionError('Calibration selected an internal baseline instead of a trainable model')
    if a.dev_only:
        (a.out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        return
    if not a.skip_holdout:
        hi=origin_idx(d.weeks,a.holdout)
        if hi<=origin_idx(d.weeks,a.folds[-1])+N_H-1:
            raise ValueError('Holdout overlaps dev window')
        candidates=list(range(26,hi-N_H+1))
        if a.max_origins:candidates=candidates[-a.max_origins:]
        print('HOLDOUT',a.holdout,flush=True)
        hold_pred,_=models_for_origin(d,candidates,hi,a)
        actual=observed_and_targets(d,hi)
        save_window_preds(d,hi,actual,hold_pred,a.out/'holdout_predictions.csv.gz')
        report['holdout_raw']={k:metric(actual,p) for k,p in hold_pred.items()}
        report['holdout_best_by_dev']=metric(actual,postprocess(hold_pred[best['candidate']],best['scale'],best['threshold']))
        compare=[]
        for name,params in best_each.items():
            mt=metric(actual,postprocess(hold_pred[name],params['scale'],params['threshold']))
            compare.append({'candidate':name,'dev_surrogate_score':params['surrogate_score'],
                            'dev_scale':params['scale'],'dev_threshold':params['threshold'],**mt})
        pd.DataFrame(compare).to_csv(a.out/'holdout_candidate_comparison.csv',index=False)
        report['holdout_candidate_comparison']=compare
        print('HOLDOUT (no new tuning)',report['holdout_best_by_dev'],flush=True)
    if not a.skip_final:
        if d.n_pairs!=8789 or len(d.warehouses)!=11 or len(d.articles)!=799:
            raise ValueError('Real competition output requires 11 x 799 observed dimension universe; got '+str(d.n_pairs))
        # Final origin is 134, latest full training target is 2026-07-20, ending July 26.
        candidates=list(range(26,N_WEEKS-N_H+1))
        if a.max_origins:candidates=candidates[-a.max_origins:]
        assert candidates[-1]+N_H-1==N_WEEKS-1
        print('FINAL: train last target week',d.weeks[candidates[-1]+8].date(),flush=True)
        final_pred,details=models_for_origin(d,candidates,N_WEEKS,a, a.out/'models' if a.save_models else None)
        # v7.4 extension: retain UNCALIBRATED forecasts for independent meta-learning.
        np.savez_compressed(a.out/'final_raw_components.npz', **{
            k: np.asarray(v, dtype=np.float32) for k, v in final_pred.items()
        })
        meta=export_prediction(d,N_WEEKS,final_pred[best['candidate']],best,a.out/'submission_v7_selected.csv')
        report['submission']=meta
        report['final_model_components']=details
        # Candidates with their OWN frozen dev calibration; holdout never tunes these values.
        for k in ['v7_global_raw','m5lite_raw','hybrid_raw']:
            if k in final_pred:
                export_prediction(d,N_WEEKS,final_pred[k],best_each[k],a.out/(k+'_candidate.csv'))
        print('FINAL CSV written',a.out/'submission_v7_selected.csv',meta,flush=True)
    (a.out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMPLETED',a.out.resolve(),flush=True)

if __name__=='__main__':main()
