import pandas as pd
import pytest

from app.dynamic_dataset import _resolve_cutoff


def _history(sales_end: str, stock_end: str):
    return (
        pd.DataFrame({"date": pd.to_datetime([sales_end])}),
        pd.DataFrame({"date": pd.to_datetime([stock_end])}),
    )


def test_inferred_cutoff_uses_last_shared_complete_sunday():
    sales, stock = _history("2026-07-31", "2026-07-30")
    assert _resolve_cutoff(sales, stock, None) == pd.Timestamp("2026-07-26")


def test_explicit_supported_cutoff_is_accepted():
    sales, stock = _history("2026-07-31", "2026-07-30")
    assert _resolve_cutoff(sales, stock, "2026-07-26") == pd.Timestamp("2026-07-26")


@pytest.mark.parametrize("cutoff", ["2026-07-27", "2026-08-02", "not-a-date"])
def test_unsupported_or_invalid_cutoff_is_rejected(cutoff):
    sales, stock = _history("2026-07-31", "2026-07-30")
    with pytest.raises(ValueError):
        _resolve_cutoff(sales, stock, cutoff)
