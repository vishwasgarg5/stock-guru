import pandas as pd
import pytest

from stock_guru.universe_ingest import (
    validate_universe_events,
    validate_universe_snapshots,
)


def test_snapshot_validation_normalizes_symbols_and_sorts():
    frame = pd.DataFrame({
        "as_of": ["2026-02-01", "2026-01-01"],
        "symbol": [" reliance ", "tcs"],
        "source": ["s", "s"],
        "source_id": ["2", "1"],
    })
    out = validate_universe_snapshots(frame)
    assert out[["as_of", "symbol"]].to_dict("records") == [
        {"as_of": pd.Timestamp("2026-01-01"), "symbol": "TCS"},
        {"as_of": pd.Timestamp("2026-02-01"), "symbol": "RELIANCE"},
    ]


def test_snapshot_validation_rejects_duplicate_membership():
    frame = pd.DataFrame({
        "as_of": ["2026-01-01", "2026-01-01"],
        "symbol": ["A", "A"],
    })
    with pytest.raises(ValueError, match="duplicate"):
        validate_universe_snapshots(frame)


def test_event_validation_requires_provenance_and_supported_action():
    frame = pd.DataFrame({
        "effective_date": ["2026-01-01"],
        "symbol": ["a"],
        "action": ["replace"],
        "source": ["source"],
        "source_id": ["id"],
    })
    with pytest.raises(ValueError, match="unsupported actions"):
        validate_universe_events(frame)


def test_event_validation_normalizes_action_and_symbol():
    frame = pd.DataFrame({
        "effective_date": ["2026-01-02", "2026-01-01"],
        "symbol": [" tcs ", "reliance"],
        "action": ["EXCLUDE", " INCLUDE "],
        "source": ["s", "s"],
        "source_id": ["2", "1"],
    })
    out = validate_universe_events(frame)
    assert out["symbol"].tolist() == ["RELIANCE", "TCS"]
    assert out["action"].tolist() == ["include", "exclude"]
