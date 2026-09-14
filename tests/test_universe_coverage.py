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
    assert report["event_count"] == 2
    assert report["event_earliest_date"] == "2021-01-01"
    assert report["event_latest_date"] == "2022-01-01"
    assert report["event_unique_symbols"] == 2
    assert report["event_provenance_complete"] is True
    assert report["historical_completeness"] == "unknown"


def test_snapshot_summary_rejects_duplicates():
    snapshots = pd.concat([_snapshots(), pd.DataFrame({"as_of": [pd.Timestamp("2020-01-01")], "symbol": ["A"]})])
    with pytest.raises(ValueError, match="duplicate"):
        summarize_snapshots(snapshots)


def test_snapshot_summary_rejects_blank_symbol():
    snapshots = pd.DataFrame({"as_of": ["2020-01-01"], "symbol": ["  "]})
    with pytest.raises(ValueError, match="invalid dates or blank symbols"):
        summarize_snapshots(snapshots)


def test_snapshot_summary_rejects_invalid_date():
    snapshots = pd.DataFrame({"as_of": ["not-a-date"], "symbol": ["A"]})
    with pytest.raises(ValueError, match="invalid dates or blank symbols"):
        summarize_snapshots(snapshots)


def test_event_summary_rejects_unsupported_action():
    events = _events().copy()
    events.loc[0, "action"] = "replace"
    with pytest.raises(ValueError, match="unsupported actions"):
        summarize_events(events)


def test_event_summary_rejects_blank_provenance():
    events = _events().copy()
    events.loc[0, "source_id"] = "  "
    with pytest.raises(ValueError, match="blank symbol or provenance"):
        summarize_events(events)


def test_event_summary_rejects_invalid_date():
    events = _events().copy()
    events.loc[0, "effective_date"] = "bad-date"
    with pytest.raises(ValueError, match="invalid effective dates"):
        summarize_events(events)


def test_event_summary_rejects_duplicate_symbol_date():
    events = pd.concat([_events(), _events().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate effective_date/symbol"):
        summarize_events(events)


def test_snapshot_summary_requires_provenance_independent_fields():
    with pytest.raises(ValueError, match="Missing snapshot columns"):
        summarize_snapshots(pd.DataFrame({"as_of": ["2020-01-01"]}))


def test_event_summary_requires_provenance_columns():
    with pytest.raises(ValueError, match="Missing event columns"):
        summarize_events(pd.DataFrame({"effective_date": ["2020-01-01"], "symbol": ["A"], "action": ["include"]}))
