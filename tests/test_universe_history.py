import pandas as pd

from stock_guru.universe_history import (
    apply_point_in_time_universe,
    build_membership_intervals,
    universe_for_date,
)


def test_membership_intervals_change_at_next_snapshot():
    snapshots = pd.DataFrame({
        "as_of": pd.to_datetime(["2020-01-01", "2021-01-01", "2021-01-01"]),
        "symbol": ["A", "A", "B"],
    })
    intervals = build_membership_intervals(snapshots)
    assert universe_for_date(intervals, "2020-06-01") == {"A"}
    assert universe_for_date(intervals, "2021-01-01") == {"A", "B"}


def test_apply_point_in_time_universe_excludes_pre_membership_rows():
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2019-01-01", "2020-06-01", "2021-06-01"]),
        "symbol": ["A", "A", "B"],
        "close": [10, 11, 20],
    })
    snapshots = pd.DataFrame({
        "as_of": pd.to_datetime(["2020-01-01", "2021-01-01"]),
        "symbol": ["A", "B"],
    })
    intervals = build_membership_intervals(snapshots)
    out = apply_point_in_time_universe(prices, intervals)
    assert set(map(tuple, out[["date", "symbol"]].itertuples(index=False, name=None))) == {
        (pd.Timestamp("2020-06-01"), "A"),
        (pd.Timestamp("2021-06-01"), "B"),
    }


def test_no_snapshot_history_is_not_fabricated():
    intervals = build_membership_intervals(pd.DataFrame({
        "as_of": pd.to_datetime(["2020-01-01"]),
        "symbol": ["A"],
    }))
    assert universe_for_date(intervals, "2019-12-31") == set()
