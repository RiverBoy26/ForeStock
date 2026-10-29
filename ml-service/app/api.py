"""Forecast API backed by immutable models and dynamic uploaded observations."""
from __future__ import annotations
from contextlib import asynccontextmanager
import hashlib
import os
from pathlib import Path
from functools import lru_cache
import json
import tempfile
import threading
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Header, Depends, File, UploadFile
from fastapi.responses import FileResponse, Response
from .dynamic_dataset import build_dynamic_dataset
from .inference import ROOT, VARIANTS, predict_dataset

MAX_UPLOAD_BYTES=int(os.environ.get('NONTON_MAX_UPLOAD_BYTES',str(2*1024*1024*1024)))
INFERENCE_LOCK=threading.Lock()

@lru_cache(maxsize=1)
def verify_frozen_artifacts():
    manifest=json.loads((ROOT/'model_manifest.json').read_text(encoding='utf-8'))
    protected={name:sha for name,sha in manifest['artifact_sha256'].items()
               if name.startswith('models/') or name.startswith('snapshots/')}
    for name,expected in protected.items():
        path=ROOT/name
        if not path.is_file():raise RuntimeError(f'Missing frozen artifact: {name}')
        with path.open('rb') as stream:
            actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:raise RuntimeError(f'Frozen artifact hash mismatch: {name}')
    return manifest

@asynccontextmanager
async def lifespan(_app:FastAPI):
    verify_frozen_artifacts()
    yield

app=FastAPI(title='Nonton v7.4 Forecast API',version='2.0.0',description='Dynamic 9-week inference with immutable Nonton v7.4 model weights. No retraining.',docs_url='/docs',redoc_url=None,lifespan=lifespan)
BASE_PATH=ROOT/'forecasts'

def require_key(x_api_key: str | None=Header(default=None)):
    secret=os.environ.get('NONTON_API_KEY')
    if secret:
        import hmac
        if not x_api_key or not hmac.compare_digest(secret,x_api_key):
            raise HTTPException(status_code=401,detail='Missing or invalid API key')

def validate_variant(variant:str):
    if variant not in VARIANTS:
        raise HTTPException(status_code=422,detail=f'Unknown variant, choose {list(VARIANTS)}')
    return variant

@lru_cache(maxsize=3)
def get_frame(variant:str):
    variant=validate_variant(variant)
    file=BASE_PATH/f'{variant}.csv'
    if not file.is_file():raise HTTPException(status_code=503,detail=f'Forecast {variant} not generated')
    data=pd.read_csv(file,dtype={'week':str,'warehouse_name':str,'article':str,'quantity':'float64'})
    if data.shape[0]!=79101 or data[['week','warehouse_name','article']].duplicated().any():
        raise HTTPException(status_code=503,detail='Corrupted forecast')
    return data

@app.get('/healthz')
def health():return {'status':'ok'}

@app.get('/readyz')
def ready():
    try:
        verify_frozen_artifacts()
        f=get_frame('selected')
        return {'ready':True,'rows':len(f),'frozen_artifacts_verified':True}
    except (HTTPException,RuntimeError) as e:
        raise HTTPException(status_code=503,detail=str(getattr(e,'detail',e)))

@app.get('/v1/model',dependencies=[Depends(require_key)])
def metadata():
    return {'model':'Nonton v7.4 FULL+META','version':'7.4.0','cutoff':'2026-07-26',
        'target_week_starts':pd.date_range('2026-07-27',periods=9,freq='7D').strftime('%Y-%m-%d').tolist(),
        'pair_count':8789,'forecast_rows':79101,'variants':list(VARIANTS),
        'models_count':41,'frozen_snapshot':True,'dynamic_inference':True,
        'minimum_history_weeks':52,'new_catalog_values_supported':False,
        'warning':'Weights and categorical encodings are immutable. Uploaded observations only rebuild rolling features.'}

def save_upload(upload:UploadFile,target:Path):
    size=0
    with target.open('wb') as output:
        while True:
            chunk=upload.file.read(1024*1024)
            if not chunk:break
            size+=len(chunk)
            if size>MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413,detail=f'{upload.filename or "upload"} is too large')
            output.write(chunk)

@app.post('/v1/predict',dependencies=[Depends(require_key)])
def dynamic_forecast(
    sales:UploadFile=File(...),
    stock:UploadFile=File(...),
    variant:str='selected',
    warehouse_name:str|None=None,
    article:str|None=None,
):
    validate_variant(variant)
    verify_frozen_artifacts()
    with tempfile.TemporaryDirectory(prefix='nonton_dynamic_') as temporary_name:
        temporary=Path(temporary_name)
        sales_path=temporary/'sales.csv'
        stock_path=temporary/'stock.csv'
        save_upload(sales,sales_path)
        save_upload(stock,stock_path)
        try:
            with INFERENCE_LOCK:
                dataset=build_dynamic_dataset(
                    sales_path,
                    stock_path,
                    ROOT/'snapshots'/'weekly_history_2026-07-26.npz',
                )
                frame=predict_dataset(dataset,ROOT/'models'/'base',ROOT/'models',variant)
        except ValueError as exc:
            raise HTTPException(status_code=422,detail=str(exc)) from exc
        if warehouse_name is not None:frame=frame[frame.warehouse_name==warehouse_name]
        if article is not None:frame=frame[frame.article==article]
        payload=frame.to_csv(index=False,float_format='%.7g').encode('utf-8')
        forecast_start=str((dataset.weeks[-1]+pd.Timedelta(days=7)).date())
        filename=f'nonton_v74_{variant}_{forecast_start}.csv'
        return Response(
            content=payload,
            media_type='text/csv',
            headers={
                'Content-Disposition':f'attachment; filename="{filename}"',
                'X-Model-Version':'7.4.0',
                'X-Forecast-Start':forecast_start,
                'X-Forecast-Rows':str(len(frame)),
            },
        )

@app.get('/v1/forecasts',dependencies=[Depends(require_key)])
def forecasts(variant:str='selected',week:str|None=None,warehouse_name:str|None=None,
              article:str|None=None,limit:int=Query(100,ge=1,le=5000),offset:int=Query(0,ge=0)):
    data=get_frame(variant)
    if week is not None:
        weeks=pd.date_range('2026-07-27',periods=9,freq='7D').strftime('%Y-%m-%d').tolist()
        if week not in weeks:raise HTTPException(status_code=422,detail='Unsupported forecast week')
        data=data[data.week==week]
    if warehouse_name is not None:data=data[data.warehouse_name==warehouse_name]
    if article is not None:data=data[data.article==article]
    page=data.iloc[offset:offset+limit]
    return {'variant':variant,'total':int(len(data)),'offset':offset,'limit':limit,
       'items':page.to_dict(orient='records')}

@app.get('/v1/export/{variant}.csv',dependencies=[Depends(require_key)])
def download(variant:str):
    validate_variant(variant)
    get_frame(variant)
    return FileResponse(BASE_PATH/f'{variant}.csv',media_type='text/csv',filename=f'nonton_v74_{variant}.csv')
