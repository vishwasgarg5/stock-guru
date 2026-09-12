import pandas as pd

from stock_guru.ohlc import OHLCForecaster


def test_ohlc_constraints_make_valid_candles():
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01"]),
        "symbol": ["AAA"],
        "close": [100.0],
        "pred_open": [105.0],
        "pred_high": [102.0],
        "pred_low": [106.0],
        "pred_close": [104.0],
    })
    out = OHLCForecaster.enforce_ohlc_constraints(raw)
    assert out.loc[0, "pred_high"] == 105.0
    assert out.loc[0, "pred_low"] == 104.0
