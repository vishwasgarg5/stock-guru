import numpy as np
import pandas as pd
import pytest

from stock_guru.fundamentals_ingest import validate_pit_fundamentals


def test_pit_fundamentals_are_sorted_and_normalized():
    frame = pd.DataFrame({"symbol": [" TCS ", "TCS"], "reported_date": ["2026-01-10", "2026-01-01"], "available_date": ["2026-01-11", "2026-01-05"], "eps": [3.0, 2.0]})
    out = validate_pit_fundamentals(frame)
    assert out["symbol"].tolist() == ["TCS", "TCS"]
    assert out["available_date"].tolist() == [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-11")]


def test_pit_fundamentals_reject_availability_before_report():
    frame = pd.DataFrame({"symbol": ["TCS"], "reported_date": ["2026-01-10"], "available_date": ["2026-01-09"]})
    with pytest.raises(ValueError, match="available_date"):
        validate_pit_fundamentals(frame)


def test_pit_fundamentals_reject_duplicate_asof_observations():
    frame = pd.DataFrame({"symbol": ["TCS", "TCS"], "reported_date": ["2026-01-10", "2026-01-10"], "available_date": ["2026-01-11", "2026-01-11"]})
    with pytest.raises(ValueError, match="Duplicate"):
        validate_pit_fundamentals(frame)


def test_pit_fundamentals_reject_nonnumeric_or_nonfinite_values():
    for value in ["not-a-number", np.inf]:
        frame = pd.DataFrame({"symbol": ["TCS"], "reported_date": ["2026-01-10"], "available_date": ["2026-01-11"], "eps": [value]})
        with pytest.raises(ValueError, match="nonnumeric or nonfinite"):
            validate_pit_fundamentals(frame)


def test_pit_fundamentals_require_complete_provenance_pair():
    frame = pd.DataFrame({"symbol": ["TCS"], "reported_date": ["2026-01-10"], "available_date": ["2026-01-11"], "source": ["NSE"]})
    with pytest.raises(ValueError, match="both source and source_id"):
        validate_pit_fundamentals(frame)


def test_pit_fundamentals_reject_blank_provenance():
    frame = pd.DataFrame({"symbol": ["TCS"], "reported_date": ["2026-01-10"], "available_date": ["2026-01-11"], "source": [""], "source_id": ["1"]})
    with pytest.raises(ValueError, match="provenance"):
        validate_pit_fundamentals(frame)
