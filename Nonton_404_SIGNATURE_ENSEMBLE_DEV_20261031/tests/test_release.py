"""python -m unittest discover -s tests -v"""
import os, json, unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from app.catboost_inference import ROOT, apply_catboost
from app.api import api

class Tests(unittest.TestCase):
    def test_schema(self):
        schema=json.loads((ROOT/'docs'/'model_schema.json').read_text(encoding='utf-8'))
        self.assertEqual(schema['feature_count'],62)
        self.assertEqual(len(list((ROOT/'models/base').glob('*.txt'))),39)
        self.assertEqual(len(list((ROOT/'models/catboost').glob('*.cbm'))),2)

    def test_forecast(self):
        df=pd.read_csv(ROOT/'forecasts/v81_catboost.csv')
        self.assertEqual(len(df),79101)
        self.assertEqual(df.columns.tolist(), ['week','warehouse_name','article','quantity'])
        self.assertEqual(df.week.nunique(),9)
        self.assertFalse(df[['week','warehouse_name','article']].duplicated().any())
        self.assertTrue(np.isfinite(df.quantity).all())
        self.assertGreaterEqual(df.quantity.min(),0)

    def test_policy(self):
        x=pd.DataFrame({'active_weeks_13':[0.,7.,7.]})
        pred=apply_catboost(np.array([0.,0.,10.]),x,np.array([.5,.7,.7]),np.array([10.,10.,10.]))
        self.assertEqual(pred[0],0)
        self.assertGreaterEqual(pred[1],1)

    def test_api(self):
        os.environ['NONTON_API_KEY']='test-abc-secret'
        with TestClient(api) as c:
            self.assertEqual(c.get('/healthz').status_code,200)
            self.assertEqual(c.get('/v1/forecasts').status_code,401)
            headers={'X-API-Key':'test-abc-secret'}
            r=c.get('/v1/forecasts?limit=2',headers=headers)
            self.assertEqual(r.status_code,200)
            self.assertEqual(len(r.json()['items']),2)
            self.assertEqual(c.get('/v1/export.csv',headers=headers).status_code,200)
