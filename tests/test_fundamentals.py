import pandas as pd
from stock_guru.fundamentals import asof_join


def test_asof_join_never_uses_future_fundamentals():
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-10", "2026-02-10"]),
        "symbol": ["ABC", "ABC"],
        "close": [100, 110],
    })
    fundamentals = pd.DataFrame({
        "symbol": ["ABC", "ABC"],
        "available_date": pd.to_datetime(["2026-01-01", "2026-02-15"]),
        "roe": [10.0, 20.0],
    })
    out = asof_join(prices, fundamentals)
    assert out.loc[out.date == pd.Timestamp("2026-01-10"), "roe"].iloc[0] == 10.0
    assert out.loc[out.date == pd.Timestamp("2026-02-10"), "roe"].iloc[0] == 10.0
