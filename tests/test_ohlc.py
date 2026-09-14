import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

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


def test_regime_adjustment_uses_oof_residuals_with_shrinkage():
    forecaster = OHLCForecaster()
    forecaster.features = ["feature"]
    train = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=40, freq="D"),
        "feature": np.arange(40, dtype=float),
        "market_ret_20d": [0.04] * 20 + [-0.04] * 20,
        "market_volatility_20": [0.01] * 40,
        "market_breadth": [0.70] * 20 + [0.40] * 20,
    })
    for target in TARGETS:
        train[target] = 0.01
    forecaster.models = {target: _ConstantModel(0.0) for target in TARGETS}
    for target in TARGETS:
        train.loc[20:, target] = 0.03

    forecaster._fit_regime_adjustments(train)

    assert set(forecaster.regime_adjustments) == {"bear"}
    assert all(
        value == pytest.approx(0.015)
        for value in forecaster.regime_adjustments["bear"].values()
    )


def test_regime_adjustment_skips_small_regimes():
    forecaster = OHLCForecaster()
    forecaster.features = ["feature"]
    train = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=19, freq="D"),
        "feature": np.arange(19, dtype=float),
        "market_ret_20d": [-0.04] * 9 + [0.04] * 10,
        "market_volatility_20": [0.01] * 19,
        "market_breadth": [0.40] * 9 + [0.70] * 10,
    })
    for target in TARGETS:
        train[target] = 0.01
    forecaster.models = {target: _ConstantModel(0.0) for target in TARGETS}
    forecaster._fit_regime_adjustments(train)
    assert "bear" not in forecaster.regime_adjustments


def test_adverse_regime_specialist_blends_at_25_percent():
    forecaster = OHLCForecaster()
    forecaster.features = ["feature"]
    forecaster.models = {target: _ConstantModel(0.04) for target in TARGETS}
    forecaster.regime_models = {
        "bear": {target: _ConstantModel(0.08) for target in TARGETS}
    }
    frame = pd.DataFrame({
        "date": [pd.Timestamp("2026-01-01")],
        "symbol": ["AAA"],
        "close": [100.0],
        "feature": [1.0],
        "market_ret_20d": [-0.04],
        "market_volatility_20": [0.01],
        "market_breadth": [0.40],
    })

    out = forecaster.predict(frame)

    expected = 0.04 * 0.75 + 0.08 * 0.25
    for col in ["pred_open", "pred_high", "pred_low", "pred_close"]:
        assert out.loc[0, col] == pytest.approx(100.0 * (1.0 + expected))


def test_specialist_is_not_used_for_neutral_regime():
    forecaster = OHLCForecaster()
    forecaster.features = ["feature"]
    forecaster.models = {target: _ConstantModel(0.04) for target in TARGETS}
    forecaster.regime_models = {
        "bear": {target: _ConstantModel(0.08) for target in TARGETS}
    }
    frame = pd.DataFrame({
        "date": [pd.Timestamp("2026-01-01")],
        "symbol": ["AAA"],
        "close": [100.0],
        "feature": [1.0],
        "market_ret_20d": [0.0],
        "market_volatility_20": [0.01],
        "market_breadth": [0.50],
    })

    out = forecaster.predict(frame)

    for col in ["pred_open", "pred_high", "pred_low", "pred_close"]:
        assert out.loc[0, col] == pytest.approx(104.0)


class _ConstantModel(BaseEstimator):
    def __init__(self, value=0.0):
        self.value = value

    def fit(self, frame, target):
        return self

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)
