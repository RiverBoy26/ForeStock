"""End-to-end reconstruction of research Mixture-of-Experts for SpbTechRun Nonton.
This code is NEW and is not byte-equivalent to the lost original v7 project.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,json,hashlib,shutil,time
from pathlib import Path
import numpy as np,pandas as pd
from data_utils import build_panel,Panel,START,HORIZON,metrics,trend_quality,validate_submission,FOLDS
from features import FeatureBuilder
from models import (fit_experts,predict_experts,train_meta,predict_meta,
                    baseline_v6,macro_shape,train_origins,seeds)


def args_parser():
    ap=argparse.ArgumentParser(description='Reconstructed Nonton MOE research trainer')
    ap.add_argument('--raw-zip',required=True,help='Original hakaton.zip containing two source CSVs')
    ap.add_argument('--out',default='results_MOE_REBUILT')
    ap.add_argument('--reference',default=None,help='Optional official forecast_5 CSV for key validation and graph')
    ap.add_argument('--mode',choices=['smoke','quick','full'],default='full')
    ap.add_argument('--fold-count',type=int,default=None)
    ap.add_argument('--max-train-origins',type=int,default=36)
    ap.add_argument('--min-train-origin',type=int,default=26)
    ap.add_argument('--rounds',type=int,default=160)
    ap.add_argument('--macro-rounds',type=int,default=120)
    ap.add_argument('--meta-rounds',type=int,default=140)
    ap.add_argument('--nn-epochs',type=int,default=12)
    ap.add_argument('--nn-batch',type=int,default=2048)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto',help='PyTorch joint NN device; LightGBM remains CPU')
    ap.add_argument('--check-stock',action='store_true',help='sum stock duplicates correctly, for audit only; not model features')
    ap.add_argument('--no-final',action='store_true',help='OOF only, do not train and export final')
    ap.add_argument('--save-fold-models',action='store_true')
    return ap.parse_args()


def main():
    conf=args_parser()
    if conf.mode=='smoke':
        conf.rounds=min(conf.rounds,16);conf.macro_rounds=min(conf.macro_rounds,14)
        conf.meta_rounds=min(conf.meta_rounds,12)
        conf.nn_epochs=min(conf.nn_epochs,1);conf.max_train_origins=min(conf.max_train_origins,3)
        conf.threads=min(conf.threads,2)
        if conf.fold_count is None:conf.fold_count=2
    if conf.mode=='quick':
        conf.rounds=min(conf.rounds,75);conf.macro_rounds=min(conf.macro_rounds,65)
        conf.meta_rounds=min(conf.meta_rounds,65)
        conf.nn_epochs=min(conf.nn_epochs,5);conf.max_train_origins=min(conf.max_train_origins,15)
        if conf.fold_count is None:conf.fold_count=4
    if conf.fold_count is None:conf.fold_count=7
    assert 2<=conf.fold_count<=7
    seeds(42,conf.threads)
    out=Path(conf.out).resolve();out.mkdir(parents=True,exist_ok=True)
    data=out/'data'
    h=data/'history.npz'
    if not h.exists():
        print('Building legal weekly panel from source ZIP',flush=True)
        build_panel(conf.raw_zip,data,use_stock=conf.check_stock)
    panel=Panel(h);feat=FeatureBuilder(panel)
    foldids=panel.fold_origins()[:conf.fold_count]
    print('Data ready: weeks',panel.T,'series',panel.N,'macro groups',panel.NG,flush=True)
    print('Historic test origins:',[(i,FOLDS[i],x) for i,x in enumerate(foldids)],flush=True)
    oof=[];details=[]
    fold_dir=out/'oof';fold_dir.mkdir(exist_ok=True)
    fp_fields=['max_train_origins','min_train_origin','rounds','macro_rounds','nn_epochs','nn_batch','threads','device']
    fingerprint={k:getattr(conf,k) for k in fp_fields}
    fingerprint['dataset_sha256']=hashlib.sha256(h.read_bytes()).hexdigest()
    fingerprint_path=out/'oof_training_fingerprint.json'
    if fingerprint_path.exists():
        past=json.loads(fingerprint_path.read_text(encoding='utf-8'))
        if past!=fingerprint:
            raise RuntimeError('Different train parameters/data in existing --out. Choose a NEW --out directory; never mix smoke OOF with full OOF!')
    elif any(fold_dir.glob('fold_*.npz')):
        raise RuntimeError('Existing fold caches without version fingerprint. Use a new --out directory.')
    else:
        fingerprint_path.write_text(json.dumps(fingerprint,indent=2),encoding='utf-8')
    for i,o in enumerate(foldids):
        path=fold_dir/f'fold_{i}.npz'
        if path.exists():
            z=np.load(path);f={k:z[k] for k in z.files}
            assert int(f['origin'])==o
            print('Using saved OOF:',path,flush=True)
        else:
            model_dir=(out/'fold_models'/f'fold_{i}') if conf.save_fold_models else None
            trained=fit_experts(panel,feat,o,conf,save_dir=model_dir)
            pred=predict_experts(trained,panel,feat,o)
            f={'origin':np.array(o,dtype=np.int16),'truth':panel.target(o).astype('float32'),**pred}
            np.savez_compressed(path,**f)
            del trained
            print('Wrote OOF',path,flush=True)
        assert int(f['origin'])+8<134
        assert f['truth'].shape==(9,8789)
        oof.append(f)
        v6=baseline_v6(f)
        details.append({'fold':i,'origin':FOLDS[i],'method':'v6_reconstructed',**metrics(f['truth'],v6),**trend_quality(f['truth'],v6,panel.grp,panel.NG)})
    # Forward chaining: later meta labels CANNOT train predictors of earlier folds.
    for i in range(3,len(oof)):
        print('Training META from folds',list(range(i)),'=> evaluate fold',i,flush=True)
        router,corrector=train_meta(oof,feat,list(range(i)),conf.threads,conf.meta_rounds)
        preds=predict_meta(oof[i],panel,feat,foldids[i],router,corrector)
        for name,p in preds.items():
            details.append({'fold':i,'origin':FOLDS[i],'method':name,**metrics(oof[i]['truth'],p),**trend_quality(oof[i]['truth'],p,panel.grp,panel.NG)})
        print('Meta validation',i,[(k,metrics(oof[i]['truth'],p)['wape']) for k,p in preds.items()],flush=True)
    pd.DataFrame(details).to_csv(out/'historical_metrics.csv',index=False)
    config={k:v for k,v in vars(conf).items() if k!='raw_zip'}
    (out/'training_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    if conf.no_final:
        print('OOF / historical validation completed (no final requested).',flush=True)
        return
    print('FINAL REFIT as of 2026-07-27, history through 2026-07-26 only',flush=True)
    final_dir=out/'models_final'
    trained=fit_experts(panel,feat,134,conf,save_dir=final_dir)
    f=predict_experts(trained,panel,feat,134)
    np.savez_compressed(out/'final_expert_predictions.npz',**f)
    router,corrector=train_meta(oof,feat,list(range(len(oof))),conf.threads,conf.meta_rounds)
    router.booster_.save_model(str(final_dir/'meta_router.txt'))
    corrector.booster_.save_model(str(final_dir/'meta_quantity_corrector.txt'))
    preds=predict_meta(f,panel,feat,134,router,corrector)
    manifest={'status':'reconstruction, not original 2026-10-30 archive',
       'date_cutoff':'2026-07-26','official_score_known':False,
       'used_historical_folds':[FOLDS[i] for i in range(len(oof))],
       'historical_validation':'meta forward chain only, previously viewed origins, not untouched',
       'outputs':{},'models':{},'parameters':config}
    weeks=pd.date_range('2026-07-27',periods=9,freq='7D').strftime('%Y-%m-%d')
    for name,mat in preds.items():
        df=pd.DataFrame({'week':np.repeat(weeks,panel.N),
                        'warehouse_name':np.tile(panel.sku_keys.warehouse_name.to_numpy(),9),
                        'article':np.tile(panel.sku_keys.article.to_numpy(),9),
                        'quantity':mat.reshape(-1)})
        validate_submission(df,conf.reference)
        filename=f'{name}.csv'; target=out/filename
        df.to_csv(target,index=False,float_format='%.7f')
        manifest['outputs'][filename]={'total_quantity':float(df.quantity.sum()),
            'nonzero':int((df.quantity>0).sum()),
            'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'weekly_quantity':[round(float(s),2) for s in mat.sum(1)]}
        print('WROTE',target,'total',round(mat.sum(),1),flush=True)
    for p in final_dir.iterdir():
        manifest['models'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    if conf.reference:
        try:
            from compare import compare_files
            compare_files(str(conf.reference),str(out/'v7_hybrid_15.csv'),str(out/'comparison_v5_vs_MOE.png'),
                          str(out/'comparison_weekly.csv'))
        except Exception as e:print('Plot failed but training succeeded:',repr(e),flush=True)
    print('FINISHED. Original model is not claimed identical.',flush=True)

if __name__=='__main__':main()
