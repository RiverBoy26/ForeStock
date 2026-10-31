"""Read-only REST for frozen CatBoost contest predictions; DOES NOT accept live sales/stock files."""
from __future__ import annotations
import hmac, os
from functools import lru_cache
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, Query, Header, HTTPException, Depends
from fastapi.responses import FileResponse
from .catboost_inference import ROOT
api=FastAPI(title='Nonton v8.1 CatBoost API',version='8.1.0-catboost',redoc_url=None)
FILE=ROOT/'forecasts'/'v81_catboost.csv'
WEEKS=set(pd.date_range('2026-07-27',periods=9,freq='7D').strftime('%Y-%m-%d'))
def auth(x_api_key:str|None=Header(default=None)):
    expected=os.getenv('NONTON_API_KEY')
    if not expected:raise HTTPException(503,detail='API key not configured')
    if not x_api_key or not hmac.compare_digest(x_api_key,expected):raise HTTPException(401,detail='Unauthorized')
@lru_cache(maxsize=1)
def table():
    if not FILE.exists():raise HTTPException(503,detail='Forecast missing')
    df=pd.read_csv(FILE,dtype={'week':str,'warehouse_name':str,'article':str,'quantity':float})
    if len(df)!=79101 or df[['week','warehouse_name','article']].duplicated().any():
        raise HTTPException(503,detail='Invalid forecast')
    return df
@api.get('/healthz')
def health():return {'status':'ok'}
@api.get('/readyz')
def ready():return {'ready':True,'rows':len(table())}
@api.get('/v1/model',dependencies=[Depends(auth)])
def model():
    return {'version':'8.1.0-catboost','type':'fixed-horizon',
        'cutoff':'2026-07-26','weeks':sorted(WEEKS),'model_count':41,
        'dynamic_csv_inference':False}
@api.get('/v1/forecasts',dependencies=[Depends(auth)])
def forecasts(week:str|None=None,warehouse_name:str|None=None,article:str|None=None,
              offset:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=5000)):
    df=table()
    if week is not None:
        if week not in WEEKS:raise HTTPException(422,detail='Unsupported week')
        df=df[df.week==week]
    if warehouse_name is not None: df=df[df.warehouse_name==warehouse_name]
    if article is not None: df=df[df.article==article]
    return {'total':len(df),'items':df.iloc[offset:offset+limit].to_dict(orient='records'),
            'offset':offset,'limit':limit}
@api.get('/v1/export.csv',dependencies=[Depends(auth)])
def export():
    table()
    return FileResponse(FILE,filename='v81_catboost.csv',media_type='text/csv')
