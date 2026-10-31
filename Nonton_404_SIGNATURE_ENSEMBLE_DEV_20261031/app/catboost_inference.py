"""v8.1 CatBoost frozen 9-week forecast. All row alignment is pair-major/horizon-minor."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from . import nonton_v7 as base, nonton_v74 as v74, v8
from .snapshot import load_dataset
from .inference import predict_components, forecast_frame
ROOT = Path(__file__).resolve().parents[1]
POLICY = {"threshold":0.53,"blend":0.50,"regime":0.0}

def apply_catboost(raw, X, prob, amount, policy=POLICY):
    raw = np.maximum(np.asarray(raw, dtype=np.float64), 0.0)
    act = X["active_weeks_13"].to_numpy()
    threshold = policy["threshold"] + policy["regime"] * np.where(act<=2, 1, np.where(act>=7,-1,0))
    ratio = np.clip((np.asarray(amount)+0.25)/(np.maximum(raw,0.5)+0.25),0.6,1.8)
    q = np.maximum(raw,0.5)*(1.0+policy["blend"]*(ratio-1.0))
    return np.where(np.asarray(prob)>=threshold, np.maximum(q,1.0),0.0).astype("float32")

def prepare_features(recompute_base=False):
    d=load_dataset(ROOT/'training'/'history_snapshot.npz')
    if recompute_base:
        raw=predict_components(d,ROOT/'models'/'base')
    else:
        # Exact base outputs from previously trained 39 LightGBM models.
        with np.load(ROOT/'training'/'final_raw_components.npz',allow_pickle=False) as z:
            raw={key:z[key].copy() for key in ('hybrid_raw','v7_global_raw','m5lite_raw')}
    X=v8.enrich(d,base.N_WEEKS,pd.DataFrame(raw))
    schema=json.loads((ROOT/'docs'/'model_schema.json').read_text(encoding='utf-8'))
    if X.columns.tolist()!=schema['feature_columns']:
        raise RuntimeError('Feature ordering changed; do not send incorrect predictions')
    if len(X)!=79101 or not np.isfinite(X.to_numpy()).all():
        raise RuntimeError('Invalid feature shape or nonfinite features')
    return d,raw,X

def run(out=None,verify=False,recompute_base=False):
    d,raw,X=prepare_features(recompute_base)
    p=CatBoostClassifier()
    p.load_model(str(ROOT/'models'/'catboost'/'catboost_prob.cbm'))
    q=CatBoostRegressor()
    q.load_model(str(ROOT/'models'/'catboost'/'catboost_logquantity.cbm'))
    features=X.to_numpy(dtype='float32')
    prob=np.clip(p.predict_proba(features)[:,1],0,1)
    amt=np.maximum(np.expm1(q.predict(features)),0)
    pred=apply_catboost(raw['hybrid_raw'],X,prob,amt)
    frame=forecast_frame(d,pred)
    path=Path(out) if out else ROOT/'forecasts'/'v81_catboost.csv'
    if verify:
        truth=pd.read_csv(ROOT/'forecasts'/'v81_catboost.csv',dtype={'week':str,'article':str,'warehouse_name':str})
        if not frame[['week','warehouse_name','article']].equals(truth[['week','warehouse_name','article']]):
            raise RuntimeError('Keys or ordering do not match shipped competition forecast')
        diff=np.max(np.abs(truth.quantity.to_numpy(dtype=float)-pred))
        if diff>1.e-3:raise RuntimeError(f'Prediction mismatch: max abs delta {diff}')
        print(f'PASS: {len(pred)} rows, max abs difference {diff:.8g}, quantity sum {pred.sum():.3f}')
        return {'rows':len(pred),'max_abs_delta':float(diff),'total_quantity':float(pred.sum())}
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,float_format='%.8g')
    print('CREATED',path,'rows',len(frame),'sum',round(float(pred.sum()),3))
    return {'rows':len(pred),'path':str(path),'total_quantity':float(pred.sum())}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify',action='store_true',help='Compare to original training-run CSV')
    p.add_argument('--recompute-base',action='store_true',help='Run all saved LightGBM models instead of frozen raw cache')
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    print(json.dumps(run(a.out,a.verify,a.recompute_base),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
