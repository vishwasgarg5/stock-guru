from pathlib import Path

import pandas as pd
import pytest

from stock_guru.calendar_io import load_trading_calendar


def test_load_trading_calendar(tmp_path: Path):
    path = tmp_path / "calendar.csv"
    pd.DataFrame({"date": ["2026-01-02", "2026-01-01"]}).to_csv(path, index=False)
    assert load_trading_calendar(path).tolist() == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")]


def test_load_calendar_requires_date_column(tmp_path: Path):
    path = tmp_path / "calendar.csv"
    pd.DataFrame({"session": ["2026-01-01"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="date column"):
        load_trading_calendar(path)
