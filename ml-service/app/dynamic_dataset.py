"""Build a rolling feature snapshot from uploaded CSV files without training models.

The frozen reference snapshot owns all categorical encodings.  Uploaded data may
change the time-series values, but it may not redefine warehouses, articles, or
their category ids because the immutable LightGBM models were trained with those
exact encodings.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import nonton_v7 as base
from .snapshot import load_dataset


REQUIRED_SALES_COLUMNS = {
    "date",
    "warehouse_name",
    "article",
    "composite_division",
    "cities_name",
    "quantity",
    "category_lvl_1",
    "category_lvl_2",
    "characteristic",
    "sales_sum",
    "discount_sum",
}
REQUIRED_STOCK_COLUMNS = {"date", "warehouse_name", "stock"}
MIN_HISTORY_WEEKS = 52


def _read_csv(path: Path, name: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise ValueError(f"Cannot read {name} as CSV: {exc}") from exc
    if frame.empty:
        raise ValueError(f"{name} is empty")
    return frame


def _parse_date(frame: pd.DataFrame, name: str) -> None:
    try:
        frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    except Exception as exc:
        raise ValueError(f"{name}.date must use YYYY-MM-DD") from exc


def _numeric(frame: pd.DataFrame, column: str, name: str, *, nonnegative: bool = False) -> None:
    values = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(values).all():
        raise ValueError(f"{name}.{column} contains a non-numeric or non-finite value")
    if nonnegative and (values < 0).any():
        raise ValueError(f"{name}.{column} contains a negative value")
    frame[column] = values.astype("float32")


def _reject_unknown(frame: pd.DataFrame, column: str, allowed: set[str], name: str) -> None:
    actual = set(frame[column].astype(str).unique())
    unknown = sorted(actual - allowed)
    if unknown:
        preview = ", ".join(unknown[:5])
        suffix = " ..." if len(unknown) > 5 else ""
        raise ValueError(
            f"{name}.{column} contains {len(unknown)} value(s) unknown to frozen model: "
            f"{preview}{suffix}"
        )


def _bucket_tensor(
    sales: pd.DataFrame,
    source: str,
    buckets: list[str],
    week_code: np.ndarray,
    pair_code: np.ndarray,
    n_pairs: int,
) -> np.ndarray:
    mapping = {value: index for index, value in enumerate(buckets[:-1])}
    tags = (
        sales[source]
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
        .map(mapping)
        .fillna(len(buckets) - 1)
        .to_numpy(dtype=np.int32)
    )
    dense_index = week_code.astype(np.int64) * n_pairs + pair_code
    output = np.zeros((len(buckets), base.N_WEEKS, n_pairs), dtype=np.float32)
    quantities = sales["quantity"].to_numpy(dtype=np.float32)
    for bucket_index in range(len(buckets)):
        mask = tags == bucket_index
        if mask.any():
            output[bucket_index] = np.bincount(
                dense_index[mask],
                weights=quantities[mask],
                minlength=base.N_WEEKS * n_pairs,
            ).reshape(base.N_WEEKS, n_pairs)
    return output


def _complete_history_end(sales_max: pd.Timestamp, stock_max: pd.Timestamp) -> pd.Timestamp:
    """Return the latest Sunday covered by both inputs."""
    shared_max = min(sales_max.normalize(), stock_max.normalize())
    days_since_sunday = (shared_max.weekday() + 1) % 7
    return shared_max - pd.Timedelta(days=days_since_sunday)


def build_dynamic_dataset(
    sales_path: Path,
    stock_path: Path,
    reference_snapshot_path: Path,
) -> base.Dataset:
    reference = load_dataset(reference_snapshot_path)
    sales = _read_csv(sales_path, "sales")
    stock = _read_csv(stock_path, "stock")

    if "article" not in stock.columns and "char_article" in stock.columns:
        stock = stock.rename(columns={"char_article": "article"})

    missing_sales = REQUIRED_SALES_COLUMNS - set(sales.columns)
    missing_stock = (REQUIRED_STOCK_COLUMNS | {"article"}) - set(stock.columns)
    if missing_sales:
        raise ValueError(f"sales is missing columns: {sorted(missing_sales)}")
    if missing_stock:
        raise ValueError(f"stock is missing columns: {sorted(missing_stock)}")

    _parse_date(sales, "sales")
    _parse_date(stock, "stock")
    _numeric(sales, "quantity", "sales", nonnegative=True)
    _numeric(sales, "sales_sum", "sales")
    _numeric(sales, "discount_sum", "sales")
    _numeric(stock, "stock", "stock")

    warehouses = list(map(str, reference.warehouses))
    articles = list(map(str, reference.articles))
    _reject_unknown(sales, "warehouse_name", set(warehouses), "sales")
    _reject_unknown(sales, "article", set(articles), "sales")
    _reject_unknown(stock, "warehouse_name", set(warehouses), "stock")
    _reject_unknown(stock, "article", set(articles), "stock")

    complete_sunday = _complete_history_end(sales["date"].max(), stock["date"].max())
    forecast_start = complete_sunday + pd.Timedelta(days=1)
    window_start = forecast_start - pd.Timedelta(weeks=base.N_WEEKS)
    observed_span = complete_sunday - sales["date"].min().normalize()
    if observed_span < pd.Timedelta(weeks=MIN_HISTORY_WEEKS):
        raise ValueError(
            f"sales must cover at least {MIN_HISTORY_WEEKS} weeks through a complete Sunday"
        )

    sales = sales[(sales["date"] >= window_start) & (sales["date"] < forecast_start)].copy()
    stock = stock[(stock["date"] >= window_start) & (stock["date"] < forecast_start)].copy()
    if sales.empty or stock.empty:
        raise ValueError("No sales/stock rows remain in the rolling history window")

    n_wh = len(warehouses)
    n_articles = len(articles)
    n_pairs = n_wh * n_articles
    if n_pairs != reference.n_pairs:
        raise ValueError("Frozen reference catalog has inconsistent dimensions")
    wh_map = {value: index for index, value in enumerate(warehouses)}
    article_map = {value: index for index, value in enumerate(articles)}

    sales_wh = sales["warehouse_name"].astype(str).map(wh_map).to_numpy(dtype=np.int32)
    sales_article = sales["article"].astype(str).map(article_map).to_numpy(dtype=np.int32)
    sales_week = ((sales["date"] - window_start).dt.days // 7).to_numpy(dtype=np.int32)
    sales_pair = sales_wh * n_articles + sales_article
    dense_index = sales_week.astype(np.int64) * n_pairs + sales_pair
    dense_size = base.N_WEEKS * n_pairs

    qty = np.bincount(
        dense_index,
        weights=sales["quantity"].to_numpy(dtype=np.float32),
        minlength=dense_size,
    ).astype("float32").reshape(base.N_WEEKS, n_pairs)

    day_code = (sales["date"] - window_start).dt.days.to_numpy(dtype=np.int32)
    unique_sale_days = np.unique(day_code.astype(np.int64) * n_pairs + sales_pair)
    day_week_key = (unique_sale_days // n_pairs // 7) * n_pairs + unique_sale_days % n_pairs
    sale_days = np.bincount(day_week_key, minlength=dense_size).astype("float32").reshape(
        base.N_WEEKS, n_pairs
    )

    department = _bucket_tensor(
        sales,
        "composite_division",
        list(reference.dep_buckets),
        sales_week,
        sales_pair,
        n_pairs,
    )
    city = _bucket_tensor(
        sales,
        "cities_name",
        list(reference.city_buckets),
        sales_week,
        sales_pair,
        n_pairs,
    )

    def aggregate_known(column: str) -> tuple[np.ndarray, np.ndarray]:
        values = sales[column].to_numpy(dtype=np.float64)
        total = np.bincount(dense_index, weights=values, minlength=dense_size).astype("float32")
        count = np.bincount(dense_index, minlength=dense_size).astype("float32")
        return total.reshape(base.N_WEEKS, n_pairs), count.reshape(base.N_WEEKS, n_pairs)

    sales_sum, sales_sum_count = aggregate_known("sales_sum")
    _, discount_count = aggregate_known("discount_sum")
    division_rows = np.bincount(dense_index, minlength=dense_size).astype("float32").reshape(
        base.N_WEEKS, n_pairs
    )

    # Article metadata is part of the frozen reference catalog, so it remains
    # known even when an article has no sales in the uploaded rolling window.
    first_article_pair = np.zeros(n_pairs, dtype=np.int32)
    first_pair_week = np.full(n_pairs, base.N_WEEKS + 10, dtype=np.int32)
    for week_index in range(base.N_WEEKS):
        new_pair = (sale_days[week_index] > 0) & (first_pair_week == base.N_WEEKS + 10)
        first_pair_week[new_pair] = week_index

    daily = (
        stock.groupby(["date", "warehouse_name", "article"], observed=True, sort=False)["stock"]
        .sum()
        .reset_index()
        .sort_values("date", kind="stable")
    )
    stock_wh = daily["warehouse_name"].astype(str).map(wh_map).to_numpy(dtype=np.int32)
    stock_article = daily["article"].astype(str).map(article_map).to_numpy(dtype=np.int32)
    stock_week = ((daily["date"] - window_start).dt.days // 7).to_numpy(dtype=np.int32)
    stock_index = stock_week.astype(np.int64) * n_pairs + stock_wh * n_articles + stock_article
    daily["dense_index"] = stock_index
    weekly_stock = daily.groupby("dense_index", sort=False, observed=True)["stock"].agg(
        ["mean", "size", "last"]
    )
    stock_dense_index = weekly_stock.index.to_numpy(dtype=np.int64)
    stock_last = np.full(dense_size, np.nan, dtype=np.float32)
    stock_mean = np.full(dense_size, np.nan, dtype=np.float32)
    stock_days = np.zeros(dense_size, dtype=np.float32)
    stock_last[stock_dense_index] = weekly_stock["last"].to_numpy(dtype=np.float32)
    stock_mean[stock_dense_index] = weekly_stock["mean"].to_numpy(dtype=np.float32)
    stock_days[stock_dense_index] = weekly_stock["size"].to_numpy(dtype=np.float32)

    return base.Dataset(
        weeks=pd.date_range(window_start, periods=base.N_WEEKS, freq="7D"),
        warehouses=warehouses,
        articles=articles,
        names_wh=reference.names_wh.copy(),
        names_article=reference.names_article.copy(),
        wh_ids=reference.wh_ids.copy(),
        article_ids=reference.article_ids.copy(),
        cat1_ids=reference.cat1_ids.copy(),
        cat2_ids=reference.cat2_ids.copy(),
        char_ids=reference.char_ids.copy(),
        first_article_week=first_article_pair,
        first_pair_week=first_pair_week,
        qty=qty,
        sale_days=sale_days,
        stock_last=stock_last.reshape(base.N_WEEKS, n_pairs),
        stock_mean=stock_mean.reshape(base.N_WEEKS, n_pairs),
        stock_days=stock_days.reshape(base.N_WEEKS, n_pairs),
        department=department,
        city=city,
        dep_buckets=list(reference.dep_buckets),
        city_buckets=list(reference.city_buckets),
        div_observation=division_rows,
        discount_observation=discount_count,
        log_sales_sum=sales_sum,
        log_sales_count=sales_sum_count,
        wh_cat1=reference.wh_cat1.copy(),
        wh_cat2=reference.wh_cat2.copy(),
        article_cat1=reference.article_cat1.copy(),
        n_pairs=n_pairs,
    )
