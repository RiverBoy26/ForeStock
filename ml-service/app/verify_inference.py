"""Slow golden regression: recompute all 4 CSVs from 41 trained LightGBM models.

Never uses labels from hidden competition period. Exit code !=0 on any mismatch.
"""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from .inference import ROOT,VARIANTS,run

def main():
    with tempfile.TemporaryDirectory(prefix='nonton_v74_verify_') as td:
        temporary=Path(td)
        run(ROOT/'snapshots/weekly_history_2026-07-26.npz',ROOT/'models/base',ROOT/'models',temporary)
        for name in VARIANTS:
            a=pd.read_csv(ROOT/'forecasts'/f'{name}.csv',dtype={'week':str,'warehouse_name':str,'article':str})
            b=pd.read_csv(temporary/f'{name}.csv',dtype={'week':str,'warehouse_name':str,'article':str})
            if not a[['week','warehouse_name','article']].equals(b[['week','warehouse_name','article']]):
                raise AssertionError(f'{name}: key mismatch')
            error=float(np.max(np.abs(a.quantity.to_numpy()-b.quantity.to_numpy())))
            print('GOLDEN',name,'max absolute difference',error)
            if error>1e-5:raise AssertionError(f'{name}: prediction deviation {error}')
    print('PASS: all four forecasts reproduced from trained models')
if __name__=='__main__':main()
