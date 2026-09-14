import pandas as pd
import pytest

from stock_guru.pipeline import Pipeline
from stock_guru.universe_source import snapshot_gap_diagnostics


def test_pipeline_fails_closed_when_universe_has_no_eligible_rows():
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-10"]),
        "symbol": ["A"],
        "open": [10.0], "high": [11.0], "low": [9.0], "close": [10.5], "volume": [1000],
    })
    intervals = pd.DataFrame({
        "symbol": ["A"],
        "start_date": pd.to_datetime(["2024-02-01"]),
        "end_date": [pd.NaT],
    })
    with pytest.raises(ValueError, match="no eligible market rows"):
        Pipeline().train(prices, universe_intervals=intervals)


def test_snapshot_gap_diagnostics_reports_large_calendar_gap():
    snapshots = pd.DataFrame({
        "as_of": pd.to_datetime(["2024-01-01", "2024-08-01", "2024-09-01"]),
        "symbol": ["A", "A", "A"],
    })
    report = snapshot_gap_diagnostics(snapshots, threshold_days=180)
    assert report["snapshot_gap_count"] == 1
    assert report["max_gap_days"] == 213
    assert report["gaps"][0]["from"] == "2024-01-01"
    assert report["gaps"][0]["to"] == "2024-08-01"
