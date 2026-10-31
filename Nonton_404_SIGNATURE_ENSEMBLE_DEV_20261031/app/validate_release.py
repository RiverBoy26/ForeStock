"""Validate immutable shipped release checksums and expected prediction schema."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd
from .catboost_inference import ROOT

def run():
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    for p,expected in manifest['sha256'].items():
        f=ROOT/p
        if not f.is_file():raise FileNotFoundError(f)
        digest=hashlib.sha256(f.read_bytes()).hexdigest()
        if digest!=expected:raise RuntimeError('SHA-256 mismatch '+p)
    df=pd.read_csv(ROOT/'forecasts'/'v81_catboost.csv')
    if len(df)!=79101 or df[['week','warehouse_name','article']].duplicated().any():
        raise RuntimeError('Wrong future keys')
    print('RELEASE OK; validated',len(manifest['sha256']),'files and',len(df),'prediction rows')

if __name__=='__main__':run()
