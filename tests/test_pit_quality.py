import pandas as pd
import pytest

from stock_guru.pit_quality import validate_pit_join_output


def test_pit_quality_accepts_valid_output():
    frame = pd.DataFrame({"symbol": ["ABC"], "date": ["2026-09-10"], "_pit_available_date": ["2026-09-01"]})
    report = validate_pit_join_output(frame)
    assert report["leakage_free"] is True
    assert report["pit_available_date_checked"] is True


def test_pit_quality_rejects_future_fundamental():
    frame = pd.DataFrame({"symbol": ["ABC"], "date": ["2026-09-10"], "_pit_available_date": ["2026-09-11"]})
    with pytest.raises(ValueError, match="available after"):
        validate_pit_join_output(frame)


def test_pit_quality_rejects_duplicate_key():
    frame = pd.DataFrame({"symbol": ["ABC", "ABC"], "date": ["2026-09-10", "2026-09-10"]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_pit_join_output(frame)
