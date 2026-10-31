from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .inference import ROOT,VARIANTS
from .snapshot import load_dataset


def validate():
    metadata=json.loads((ROOT/'model_manifest.json').read_text(encoding='utf-8'))
    for name,sha in metadata['artifact_sha256'].items():
        target=ROOT/name
        if not target.is_file():raise FileNotFoundError(target)
        with target.open('rb') as f:
            actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=sha:raise ValueError(f'Corrupt model/snapshot: {name}')
    for name,sha in metadata['forecast_sha256'].items():
        target=ROOT/name
        with target.open('rb') as f:
            actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=sha:raise ValueError(f'Corrupt forecast: {name}')
    d=load_dataset(ROOT/'snapshots/weekly_history_2026-07-26.npz')
    assert d.n_pairs==8789
    expected_weeks=set(pd.date_range('2026-07-27',periods=9,freq='7D').strftime('%Y-%m-%d'))
    files={}
    for name in VARIANTS:
        df=pd.read_csv(ROOT/'forecasts'/f'{name}.csv',dtype={'week':str,'warehouse_name':str,'article':str})
        if list(df.columns)!=['week','warehouse_name','article','quantity']:raise ValueError('Invalid columns')
        if len(df)!=79101 or df[['week','warehouse_name','article']].duplicated().any():raise ValueError('Invalid forecast key grid')
        if set(df.week)!=expected_weeks or df.warehouse_name.nunique()!=11 or df.article.nunique()!=799:raise ValueError('Missing keys')
        if not np.isfinite(df.quantity).all() or (df.quantity<0).any():raise ValueError('Invalid quantity')
        files[name]={'rows':len(df),'total':float(df.quantity.sum()),'zero_percent':float(np.mean(df.quantity==0)*100)}
    print(json.dumps({'status':'PASS','models_count':len(list((ROOT/'models').rglob('*.txt'))),'forecasts':files},indent=2,ensure_ascii=False))
    return files

if __name__=='__main__':validate()
