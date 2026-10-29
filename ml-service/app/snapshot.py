"""Immutable v7.4 weekly historical feature snapshot, without raw sales/stock files.

Contains commercially sensitive AGGREGATED business history. Protect at rest and in transit.
"""
from __future__ import annotations
from dataclasses import fields
from pathlib import Path
import json
import numpy as np
import pandas as pd
try:
    from . import nonton_v7 as base
except ImportError:
    import nonton_v7 as base

STRING_LISTS={'warehouses','articles','dep_buckets','city_buckets'}
DATE_LISTS={'weeks'}

def save_dataset(dataset:base.Dataset, target:Path):
    arrays={}
    for field in fields(base.Dataset):
        name=field.name
        if name=='n_pairs':
            arrays[name]=np.array(dataset.n_pairs,dtype='int32')
        elif name in DATE_LISTS:
            arrays[name]=np.array(dataset.weeks.strftime('%Y-%m-%d'),dtype='U16')
        elif name in STRING_LISTS:
            arrays[name]=np.asarray(getattr(dataset,name),dtype='U160')
        else:
            value=getattr(dataset,name)
            if isinstance(value,np.ndarray) and value.dtype.kind=='O':
                arrays[name]=value.astype('U160')
            else:
                arrays[name]=np.asarray(value)
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,**arrays)

def load_dataset(path:Path)->base.Dataset:
    with np.load(path,allow_pickle=False) as z:
        expected={f.name for f in fields(base.Dataset)}
        if set(z.files)!=expected:
            raise ValueError(f'Invalid snapshot: missing {expected-set(z.files)}, extra {set(z.files)-expected}')
        values={}
        for f in fields(base.Dataset):
            name=f.name
            if name=='n_pairs':values[name]=int(z[name])
            elif name in DATE_LISTS:values[name]=pd.DatetimeIndex(z[name].astype(str))
            elif name in STRING_LISTS:values[name]=list(z[name].astype(str))
            else:values[name]=z[name].copy()
    d=base.Dataset(**values)
    if len(d.weeks)!=base.N_WEEKS or d.n_pairs!=len(d.names_wh) or d.n_pairs!=8789:
        raise ValueError('Invalid dataset dimension')
    if d.weeks[0]!=base.START or d.weeks[-1]!=pd.Timestamp('2026-07-20'):
        raise ValueError('Unexpected history dates')
    if d.qty.shape!=(base.N_WEEKS,d.n_pairs):
        raise ValueError('Invalid quantity shape')
    return d

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--raw-zip',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    d=base.load_raw(a.raw_zip)
    save_dataset(d,a.out)
    r=load_dataset(a.out)
    print('SNAPSHOT READY',a.out,'pairs',r.n_pairs,'weeks',len(r.weeks))
