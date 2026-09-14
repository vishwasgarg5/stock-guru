import pandas as pd
import pytest
from stock_guru.scheduling import next_session_date, validate_trading_calendar


def test_next_session_is_strictly_future():
    assert next_session_date("2026-09-14", ["2026-09-14", "2026-09-15"]) == pd.Timestamp("2026-09-15")


def test_empty_calendar_rejected():
    with pytest.raises(ValueError):
        validate_trading_calendar([])


def test_duplicate_calendar_rejected():
    with pytest.raises(ValueError):
        validate_trading_calendar(["2026-09-14", "2026-09-14"])
