import pandas as pd
import pytest

from stock_guru.universe_history import apply_point_in_time_universe, validate_membership_intervals


def test_interval_validation_rejects_overlap():
    intervals = pd.DataFrame({
        "symbol": ["A", "A"],
        "start_date": pd.to_datetime(["2024-01-01", "2024-03-01"]),
        "end_date": pd.to_datetime(["2024-04-01", "2024-05-01"]),
    })
    with pytest.raises(ValueError, match="Overlapping"):
        validate_membership_intervals(intervals)


def test_interval_validation_allows_adjacent_membership():
    intervals = pd.DataFrame({
        "symbol": ["A", "A"],
        "start_date": pd.to_datetime(["2024-01-01", "2024-04-01"]),
        "end_date": pd.to_datetime(["2024-04-01", "2024-05-01"]),
    })
    out = validate_membership_intervals(intervals)
    assert len(out) == 2


def test_apply_point_in_time_universe_normalizes_symbols():
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-15", "2024-02-15"]),
        "symbol": ["a", "b"],
        "close": [10.0, 20.0],
    })
    intervals = pd.DataFrame({
        "symbol": ["A"],
        "start_date": pd.to_datetime(["2024-01-01"]),
        "end_date": [pd.NaT],
    })
    out = apply_point_in_time_universe(prices, intervals)
    assert out["symbol"].tolist() == ["A"]
