import os
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient
from app.api import app, get_frame
from app.inference import ROOT, VARIANTS
from app.snapshot import load_dataset


def test_snapshot():
    d=load_dataset(ROOT/'snapshots/weekly_history_2026-07-26.npz')
    assert d.n_pairs==8789
    assert d.qty.shape==(134,8789)


def test_forecasts():
    for variant in VARIANTS:
        f=get_frame(variant)
        assert len(f)==79101
        assert f.quantity.ge(0).all()
        assert f[['week','warehouse_name','article']].duplicated().sum()==0


def test_api():
    os.environ['NONTON_API_KEY']='unit-secret'
    client=TestClient(app)
    assert client.get('/healthz').status_code==200
    assert client.get('/readyz').status_code==200
    assert client.get('/v1/model').status_code==401
    headers={'x-api-key':'unit-secret'}
    m=client.get('/v1/model',headers=headers)
    assert m.status_code==200 and m.json()['models_count']==41
    record=client.get('/v1/forecasts',headers=headers,params={'limit':2})
    assert record.status_code==200 and len(record.json()['items'])==2
    row=record.json()['items'][0]
    match=client.get('/v1/forecasts',headers=headers,params={'week':row['week'],'warehouse_name':row['warehouse_name'],'article':row['article']}).json()
    assert match['total']==1 and match['items'][0]==row
    assert client.get('/v1/forecasts',headers=headers,params={'variant':'X'}).status_code==422
    assert client.get('/v1/export/selected.csv',headers=headers).status_code==200
    os.environ.pop('NONTON_API_KEY',None)
