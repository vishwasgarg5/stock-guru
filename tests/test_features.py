import numpy as np
import pandas as pd

from stock_guru.features import build_features


def test_feature_pipeline_is_grouped_by_symbol():
    dates = pd.date_range("2024-01-01", periods=70, freq="D")
    rows = []
    for symbol, base in [("AAA", 100.0), ("BBB", 200.0)]:
        for i, date in enumerate(dates):
            close = base + i
            rows.append({
                "date": date,
                "symbol": symbol,
                "open": close - 1,
                "high": close + 2,
                "low": close - 2,
                "close": close,
                "volume": 1000 + i,
            })
    df = pd.DataFrame(rows)
    out, features = build_features(df)
    assert features
    assert len(out) == 140
    assert np.isfinite(out["ret_5d"].dropna()).all()
    assert out["target_close"].notna().sum() == 138


def test_feature_pipeline_accepts_point_in_time_fundamentals():
    dates = pd.date_range("2024-01-01", periods=25, freq="D")
    prices = pd.DataFrame({
        "date": dates,
        "symbol": "AAA",
        "open": 99.0 + np.arange(25),
        "high": 101.0 + np.arange(25),
        "low": 98.0 + np.arange(25),
        "close": 100.0 + np.arange(25),
        "volume": 1000.0,
    })
    fundamentals = pd.DataFrame({
        "symbol": ["AAA", "AAA"],
        "reported_date": ["2023-12-20", "2024-01-15"],
        "available_date": ["2023-12-25", "2024-01-20"],
        "roe": [10.0, 20.0],
    })

    out, features = build_features(prices, fundamentals)

    assert "roe" in features
    assert out.loc[out["date"] == "2024-01-10", "roe"].iloc[0] == 10.0
    assert out.loc[out["date"] == "2024-01-21", "roe"].iloc[0] == 20.0
