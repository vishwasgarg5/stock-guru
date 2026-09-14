import pandas as pd

from stock_guru.data import apply_universe_snapshots


def test_universe_snapshots_filter_members_by_observation_period():
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-10", "2026-01-10", "2026-02-10", "2026-02-10"]),
        "symbol": ["A", "B", "A", "B"],
        "close": [10, 20, 11, 21],
    })
    snapshots = pd.DataFrame({
        "as_of": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-02-01"]),
        "symbol": ["A", "A", "B"],
    })
    out = apply_universe_snapshots(prices, snapshots)
    assert set(map(tuple, out[["date", "symbol"]].itertuples(index=False, name=None))) == {
        (pd.Timestamp("2026-01-10"), "A"),
        (pd.Timestamp("2026-02-10"), "A"),
        (pd.Timestamp("2026-02-10"), "B"),
    }
