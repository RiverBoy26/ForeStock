import unittest
from pathlib import Path
import sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.v82_inference import predict,ROOT as R
from app import nonton_v74 as v74,nonton_v7 as base
class TestV82(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.new,cls.old,cls.weights=predict()
 def test_keys(self):
  original=pd.read_csv(R/'forecasts/v81_catboost.csv')
  new=pd.read_csv(R/'forecasts/v82_confidence_amount.csv')
  self.assertTrue(original[['week','warehouse_name','article']].equals(new[['week','warehouse_name','article']]))
  self.assertEqual(len(new),79101)
  self.assertEqual(new.duplicated(['week','warehouse_name','article']).sum(),0)
  self.assertTrue(np.isfinite(new.quantity).all() and (new.quantity>=0).all())
 def test_original_replay(self):
  d=pd.read_csv(R/'forecasts/v81_catboost.csv')
  self.assertLess(np.max(np.abs(d.quantity.to_numpy()-self.old)),.001)
 def test_zero_mask(self):
  self.assertTrue(np.array_equal(self.new==0,self.old==0))
 def test_weight_bounds(self):
  self.assertGreaterEqual(self.weights.min(),0)
  self.assertLessEqual(self.weights.max(),.600001)
 def test_cutoff_and_fold_dates(self):
  from app.snapshot import load_dataset
  d=load_dataset(R/'training/history_snapshot.npz')
  self.assertEqual(str(d.weeks[-1].date()),'2026-07-20')
  for dt in v74.DEV_FOLDS:
   oi=base.origin_idx(d.weeks,dt)
   self.assertLessEqual(oi+9,len(d.qty))
 def test_csv_exact_replay(self):
  d=pd.read_csv(R/'forecasts/v82_confidence_amount.csv')
  self.assertLess(np.max(np.abs(d.quantity.to_numpy()-self.new)),.001)
if __name__=='__main__':unittest.main()
