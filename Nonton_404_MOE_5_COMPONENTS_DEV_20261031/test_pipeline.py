"""Pure invariants + optional real-dataset tests: python -m unittest -v"""
import unittest
import tempfile
import numpy as np,pandas as pd
from data_utils import stock_daily_sum,metrics,trend_quality,Panel,validate_submission,HORIZON
from models import train_origins,macro_shape

class TestInvariants(unittest.TestCase):
    def test_stock_duplicates_are_summed_even_if_identical(self):
        x=pd.DataFrame({'date':['2026-01-02']*3,'warehouse_name':['w']*3,
                        'article':['a']*3,'stock':[2,2,-1]})
        self.assertEqual(stock_daily_sum(x).stock.iloc[0],3)
    def test_negative_stock_kept(self):
        x=pd.DataFrame({'date':['2026-01-02'],'warehouse_name':['w'],'article':['a'],'stock':[-5]})
        self.assertEqual(stock_daily_sum(x).stock.iloc[0],-5)
    def test_smape_both_zero(self):
        y=np.zeros((9,2));p=y.copy()
        self.assertEqual(metrics(y,p)['smape'],0)
    def test_smape_one_sided_zero(self):
        y=np.array([[1.]],dtype='float32');p=np.array([[0.]],dtype='float32')
        self.assertEqual(metrics(y,p)['smape'],200.)
    def test_train_window_cannot_overlap_truth(self):
        for o in [56,66,82,92,102,113,125,134]:
            ii=train_origins(o,36)
            self.assertTrue(all(t+8<o for t in ii))
            self.assertEqual(ii[-1]+8,o-1)
    def test_macro_shape_preserves_zeros(self):
        a=np.array([[2.,0.,0.],[4.,0.,0.]],dtype='float32')
        m=np.array([[3.],[6.]],dtype='float32')
        self.assertTrue(np.array_equal(macro_shape(a,m)==0,a==0))
    def test_group_direction(self):
        y=np.array([[1.,2.],[2.,3.],[3.,4.]],dtype='float32')
        p=y.copy()
        z=trend_quality(y,p,np.array([0,1]),2)
        self.assertEqual(z['group_week_direction_hit_pct'],100)
        self.assertEqual(z['company_trend_hit'],100)

if __name__=='__main__':unittest.main()
