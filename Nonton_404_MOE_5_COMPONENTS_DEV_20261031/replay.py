"""Reproduce fixed-horizon Nonton reconstructed five-component MoE from saved weights.

No training or raw sales ZIP needed. This is a frozen research model for the
specific cutoff 2026-07-26, not a generic new-data forecasting API.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data_utils import Panel, validate_submission
from features import FeatureBuilder
from models import load_joint, predict_experts, predict_meta

H = 9


class BinaryBooster:
    def __init__(self, path):
        self.booster = lgb.Booster(model_file=str(path))

    def predict_proba(self, x):
        p = np.asarray(self.booster.predict(x, num_threads=4))
        return np.column_stack([1 - p, p])


class RegressorBooster:
    def __init__(self, path):
        self.booster = lgb.Booster(model_file=str(path))

    def predict(self, x):
        return self.booster.predict(x, num_threads=4)


def main():
    parser = argparse.ArgumentParser(description='Replay Nonton MoE frozen 9-week forecast from saved trained weights')
    parser.add_argument('--package', default='.', help='Release root (contains models_final and data)')
    parser.add_argument('--out', default='replayed_forecasts')
    parser.add_argument('--variant', default='all', choices=['all', 'v6_baseline', 'v7_routed', 'v7_hybrid_15', 'v7_hybrid_30'])
    parser.add_argument('--verify', action='store_true', help='Compare against published bundled CSVs')
    parser.add_argument('--recompute-experts', action='store_true', help='Recalculate base, NN and macro from six trained weight files')
    opt = parser.parse_args()
    root = Path(opt.package).resolve()
    out = Path(opt.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    panel = Panel(root / 'data' / 'history.npz')
    fbuilder = FeatureBuilder(panel)
    weights = root / 'models_final'
    with np.load(root / 'final_expert_predictions.npz') as z:
        cached = {k: z[k] for k in z.files}

    if opt.recompute_experts:
        expert_models = (
            BinaryBooster(weights / 'sku_occurrence.txt'),
            RegressorBooster(weights / 'sku_positive_quantity.txt'),
            RegressorBooster(weights / 'macro.txt'),
            load_joint(weights / 'joint.pt'),
        )
        raw = predict_experts(expert_models, panel, fbuilder, 134)
        for k in ('base', 'legacy', 'prob', 'qty', 'macro'):
            deviation = float(np.max(np.abs(raw[k].astype('float64') - cached[k].astype('float64'))))
            print(f'Weight replay {k}: max absolute error {deviation:.9g}')
            if deviation > 5e-5:
                raise RuntimeError(f'Saved expert weights do not reproduce cached {k} (diff {deviation})')
        print('All saved base expert weights reproduced successfully.')
    else:
        raw = cached

    router = BinaryBooster(weights / 'meta_router.txt')
    corrector = RegressorBooster(weights / 'meta_quantity_corrector.txt')
    forecasts = predict_meta(raw, panel, fbuilder, 134, router, corrector)
    weeks = pd.date_range('2026-07-27', periods=H, freq='7D').strftime('%Y-%m-%d')
    selected = list(forecasts) if opt.variant == 'all' else [opt.variant]
    summary = {}
    for key in selected:
        q = forecasts[key].reshape(-1)
        table = pd.DataFrame({
            'week': np.repeat(weeks, panel.N),
            'warehouse_name': np.tile(panel.sku_keys['warehouse_name'].to_numpy(), H),
            'article': np.tile(panel.sku_keys['article'].to_numpy(), H),
            'quantity': q,
        })
        validate_submission(table)
        output = out / (key + '.csv')
        table.to_csv(output, index=False, float_format='%.7f')
        record = {'sum': round(float(table.quantity.sum()), 4), 'nonzero': int((q > 0).sum()),
                  'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
        if opt.verify:
            ref_path = root / 'forecasts' / (key + '.csv')
            if not ref_path.exists(): ref_path = root / (key + '.csv')
            reference = pd.read_csv(ref_path)
            if not table[['week', 'warehouse_name', 'article']].equals(reference[['week', 'warehouse_name', 'article']]):
                raise AssertionError('Reference keys or order differ for ' + key)
            err = float(np.max(np.abs(table.quantity.to_numpy() - reference.quantity.to_numpy())))
            record['max_abs_diff_to_bundled_csv'] = err
            if err > 5e-5:
                raise AssertionError(f'Replay differs from bundled {key}: max {err}')
        print(f'{key}: {record}')
        summary[key] = record
    (out / 'replay_report.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Done. These are reconstructed experimental forecasts, not official leaderboard submissions.')


if __name__ == '__main__':
    main()
