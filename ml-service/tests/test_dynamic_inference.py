from pathlib import Path
import os
import tempfile
import unittest

import pandas as pd
from fastapi.testclient import TestClient

from app.api import app
from app.dynamic_dataset import build_dynamic_dataset
from app.inference import ROOT, predict_dataset
from app.snapshot import load_dataset


def write_inputs(tmp_path: Path, *, warehouse: str, article: str):
    dates = pd.date_range("2025-09-07", "2026-10-25", freq="W-SUN")
    sales = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "warehouse_name": warehouse,
            "article": article,
            "composite_division": "dynamic-department",
            "cities_name": "dynamic-city",
            "quantity": 1.0,
            "category_lvl_1": "ignored-reference-encoding-is-used",
            "category_lvl_2": "ignored-reference-encoding-is-used",
            "characteristic": "ignored-reference-encoding-is-used",
            "sales_sum": 10.0,
            "discount_sum": 0.0,
        }
    )
    stock = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "warehouse_name": warehouse,
            "char_article": article,
            "stock": 5.0,
        }
    )
    sales_path = tmp_path / "sales.csv"
    stock_path = tmp_path / "stock.csv"
    sales.to_csv(sales_path, index=False)
    stock.to_csv(stock_path, index=False)
    return sales_path, stock_path


class DynamicInferenceTest(unittest.TestCase):

    def test_exact_52_complete_weeks_are_accepted(self):
        reference = load_dataset(ROOT / "snapshots" / "weekly_history_2026-07-26.npz")
        warehouse = reference.warehouses[0]
        article = reference.articles[0]
        dates = pd.date_range("2025-10-27", "2026-10-25", freq="D")

        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            sales_path = directory / "sales.csv"
            stock_path = directory / "stock.csv"
            pd.DataFrame(
                {
                    "date": dates.strftime("%Y-%m-%d"),
                    "warehouse_name": warehouse,
                    "article": article,
                    "composite_division": "department",
                    "cities_name": "city",
                    "quantity": 1.0,
                    "category_lvl_1": "category-1",
                    "category_lvl_2": "category-2",
                    "characteristic": "characteristic",
                    "sales_sum": 1.0,
                    "discount_sum": 0.0,
                }
            ).to_csv(sales_path, index=False)
            pd.DataFrame(
                {
                    "date": dates.strftime("%Y-%m-%d"),
                    "warehouse_name": warehouse,
                    "article": article,
                    "stock": 5.0,
                }
            ).to_csv(stock_path, index=False)

            dataset = build_dynamic_dataset(
                sales_path,
                stock_path,
                ROOT / "snapshots" / "weekly_history_2026-07-26.npz",
            )
            self.assertEqual(str(dataset.weeks[-1].date()), "2026-10-19")

    def test_dynamic_inference_uses_future_calendar(self):
        reference = load_dataset(ROOT / "snapshots" / "weekly_history_2026-07-26.npz")
        with tempfile.TemporaryDirectory() as name:
            sales_path, stock_path = write_inputs(
                Path(name),
                warehouse=reference.warehouses[0],
                article=reference.articles[0],
            )
            dataset = build_dynamic_dataset(
                sales_path,
                stock_path,
                ROOT / "snapshots" / "weekly_history_2026-07-26.npz",
            )
            self.assertEqual(str(dataset.weeks[-1].date()), "2026-10-19")
            frame = predict_dataset(
                dataset, ROOT / "models" / "base", ROOT / "models", "selected"
            )
            self.assertEqual(len(frame), 79_101)
            self.assertEqual(frame.week.min(), "2026-10-26")
            self.assertEqual(frame.week.max(), "2026-12-21")
            self.assertTrue(frame.quantity.ge(0).all())

    def test_dynamic_inference_rejects_unknown_catalog_value(self):
        reference = load_dataset(ROOT / "snapshots" / "weekly_history_2026-07-26.npz")
        with tempfile.TemporaryDirectory() as name:
            sales_path, stock_path = write_inputs(
                Path(name),
                warehouse="unknown-warehouse",
                article=reference.articles[0],
            )
            with self.assertRaisesRegex(ValueError, "unknown to frozen model"):
                build_dynamic_dataset(
                    sales_path,
                    stock_path,
                    ROOT / "snapshots" / "weekly_history_2026-07-26.npz",
                )

    def test_dynamic_http_endpoint(self):
        reference = load_dataset(ROOT / "snapshots" / "weekly_history_2026-07-26.npz")
        warehouse = reference.warehouses[0]
        article = reference.articles[0]
        with tempfile.TemporaryDirectory() as name:
            sales_path, stock_path = write_inputs(
                Path(name), warehouse=warehouse, article=article
            )
            os.environ["NONTON_API_KEY"] = "dynamic-test-secret"
            try:
                with TestClient(app) as client:
                    with sales_path.open("rb") as sales, stock_path.open("rb") as stock:
                        response = client.post(
                            "/v1/predict",
                            params={
                                "variant": "baseline",
                                "warehouse_name": warehouse,
                                "article": article,
                                "start_week": 2,
                                "end_week": 4,
                            },
                            headers={"X-API-Key": "dynamic-test-secret"},
                            files={
                                "sales": ("sales.csv", sales, "text/csv"),
                                "stock": ("stock.csv", stock, "text/csv"),
                            },
                        )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.headers["x-model-version"], "7.4.0")
                self.assertEqual(response.headers["x-forecast-start"], "2026-11-02")
                self.assertEqual(response.headers["x-forecast-end"], "2026-11-16")
                self.assertEqual(response.headers["x-forecast-interval"], "2-4")
                self.assertEqual(len(response.text.strip().splitlines()), 4)
            finally:
                os.environ.pop("NONTON_API_KEY", None)
