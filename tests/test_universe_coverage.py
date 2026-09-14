import pandas as pd
import pytest

from stock_guru.universe_coverage import build_coverage_report, summarize_events, summarize_snapshots


def _snapshots():
    return pd.DataFrame({
        "as_of": pd.to_datetime(["2020-01-01", "2021-01-01", "2021-01-01"]),
        "symbol": ["A", "A", "B"],
    })


def _events():
    return pd.DataFrame({
        "effective_date": pd.to_datetime(["2021-01-01", "2022-01-01"]),
        "symbol": ["B", "A"],
        "action": ["include", "exclude"],
        "source": ["official", "official"],
        "source_id": ["inc-1", "exc-1"],
    })


def test_snapshot_summary_reports_supported_span_without_claiming_completeness():
    report = build_coverage_report(_snapshots(), _events())
    assert report["earliest_date"] == "2020-01-01"
    assert report["latest_date"] == "2021-01-01"
    assert report["snapshot_dates"] == 2
    assert report["snapshot_rows"] == 3
    assert report["unique_symbols"] == 2
    assert report["inclusion_events"] == 1
    assert report["exclusion_events"] == 1
    assert report["historical_completeness"] == "unknown"


def test_snapshot_summary_rejects_duplicates():
    snapshots = pd.concat([_snapshots(), pd.DataFrame({"as_of": [pd.Timestamp("2020-01-01")], "symbol": ["A"]})])
    with pytest.raises(ValueError, match="duplicate"):
        summarize_snapshots(snapshots)


def test_event_summary_rejects_unsupported_action():
    events = _events().copy()
    events.loc[0, "action"] = "replace"
    with pytest.raises(ValueError, match="unsupported actions"):
        summarize_events(events)


def test_snapshot_summary_requires_provenance_independent_fields():
    with pytest.raises(ValueError, match="Missing snapshot columns"):
        summarize_snapshots(pd.DataFrame({"as_of": ["2020-01-01"]}))
