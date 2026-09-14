import pandas as pd
import pytest

from stock_guru.features import add_technical_features


def _row(date, symbol="AAA"):
    return {"date": date, "symbol": symbol, "open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0, "volume": 1000.0}


def test_technical_features_reject_duplicate_symbol_date():
    frame = pd.DataFrame([_row("2026-01-01"), _row("2026-01-01")])
    with pytest.raises(ValueError, match="duplicate date/symbol"):
        add_technical_features(frame)


def test_technical_features_reject_missing_market_column():
    frame = pd.DataFrame([_row("2026-01-01")]).drop(columns="volume")
    with pytest.raises(ValueError, match="Missing market columns"):
        add_technical_features(frame)
