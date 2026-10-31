from __future__ import annotations
import unittest
import numpy as np,pandas as pd
from app.v83_inference import snap_quantity,VARIANTS,run
class V83Tests(unittest.TestCase):
 def test_zeros(self):
  x=np.array([0.,1.,1.49,1.51,2.51,9.80,10.2]);v=snap_quantity(x)
  self.assertTrue(np.array_equal(x>0,v>0));self.assertTrue((v>=0).all())
 def test_nearest(self):
  x=np.array([1.49,1.5,1.51,2.45,2.65]);self.assertTrue(np.array_equal(snap_quantity(x),[1,2,2,2,3]))
 def test_upward_asymmetric(self):
  x=np.array([1.30,1.34,1.35,2.35]);self.assertTrue(np.array_equal(snap_quantity(x,.65),[1,1,2,3]))
 def test_cap(self):
  x=np.array([9.85,10.00001,11.65]);v=snap_quantity(x,.65,10.)
  self.assertAlmostEqual(v[0],10.);self.assertAlmostEqual(v[1],x[1]);self.assertAlmostEqual(v[2],x[2])
 def test_invalid(self):
  for arr in [[-1.,0.],[float('nan')],[float('inf')]]:
   with self.assertRaises(ValueError):snap_quantity(arr)
 def test_frozen_forecasts(self):
  for variant in VARIANTS:
   frame=run(variant,verify=True)
   self.assertEqual(len(frame),79101)
   self.assertTrue(frame.quantity.ge(0).all())
   self.assertEqual(frame.week.nunique(),9)
if __name__=='__main__':unittest.main()
