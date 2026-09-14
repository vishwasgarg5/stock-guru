import pandas as pd
import pytest

from stock_guru.pit import join_pit_fundamentals


def test_join_uses_latest_fundamental_available_on_or_before_price_date():
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-10", "2024-02-10", "2024-03-10"]),
        "symbol": ["A", "A", "A"],
        "close": [100.0, 101.0, 102.0],
    })
    fundamentals = pd.DataFrame({
        "symbol": ["A", "A"],
        "reported_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "available_date": pd.to_datetime(["2024-01-20", "2024-02-20"]),
        "roe": [10.0, 20.0],
    })

    out = join_pit_fundamentals(prices, fundamentals)

    assert pd.isna(out.loc[out["date"] == "2024-01-10", "roe"]).all()
    assert out.loc[out["date"] == "2024-02-10", "roe"].iloc[0] == 10.0
    assert out.loc[out["date"] == "2024-03-10", "roe"].iloc[0] == 20.0
    available = out["_pit_available_date"].dt.strftime("%Y-%m-%d")
    assert pd.isna(available.iloc[0])
    assert available.iloc[1:].tolist() == ["2024-01-20", "2024-02-20"]


def test_join_rejects_invalid_price_dates():
    prices = pd.DataFrame({"date": ["bad-date"], "symbol": ["A"]})
    fundamentals = pd.DataFrame({
        "symbol": ["A"],
        "reported_date": ["2024-01-01"],
        "available_date": ["2024-01-02"],
        "roe": [10.0],
    })

    with pytest.raises(ValueError, match="invalid dates"):
        join_pit_fundamentals(prices, fundamentals)


def test_join_preserves_all_price_rows():
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-10", "2024-01-11"]),
        "symbol": ["A", "B"],
        "close": [100.0, 200.0],
    })
    fundamentals = pd.DataFrame({
        "symbol": ["A"],
        "reported_date": ["2024-01-01"],
        "available_date": ["2024-01-01"],
        "roe": [10.0],
    })

    out = join_pit_fundamentals(prices, fundamentals)
    assert len(out) == len(prices)
    assert out.loc[out["symbol"] == "B", "roe"].isna().all()
