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


def test_regime_weights_modestly_emphasize_adverse_markets():
    train = pd.DataFrame({
        "market_ret_20d": [-0.04, -0.04, 0.04, 0.0],
        "market_volatility_20": [0.01, 0.03, 0.01, 0.01],
        "market_breadth": [0.40, 0.40, 0.70, 0.50],
    })
    weights = OHLCForecaster.regime_weights(train)
    assert weights.tolist() == [1.25, 1.5, 1.0, 1.0]
    assert weights.max() < 2.0
