import numpy as np
import pandas as pd

from stock_guru.ohlc import OHLCForecaster, TARGETS


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


def test_regime_adjustment_uses_training_residuals_only():
    forecaster = OHLCForecaster(params={"n_estimators": 5, "max_depth": 2})
    forecaster.features = ["feature"]
    train = pd.DataFrame({
        "feature": np.arange(20, dtype=float),
        "market_ret_20d": [-0.04] * 10 + [0.04] * 10,
        "market_volatility_20": [0.01] * 20,
        "market_breadth": [0.40] * 10 + [0.70] * 10,
    })
    for target in TARGETS:
        train[target] = 0.01
    forecaster.models = {target: _ConstantModel(0.0) for target in TARGETS}
    for target in TARGETS:
        train.loc[:9, target] = 0.03
    forecaster._fit_regime_adjustments(train)
    assert set(forecaster.regime_adjustments) == {"bear"}
    assert all(value == 0.03 for value in forecaster.regime_adjustments["bear"].values())


class _ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)
