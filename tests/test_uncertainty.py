import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from stock_guru.evaluation import evaluate
from stock_guru.ohlc import OHLCForecaster, TARGETS
from stock_guru.risk import RiskConfig, final_trade_decision


class _ConstantModel(BaseEstimator):
    def __init__(self, value=0.0):
        self.value = value

    def fit(self, frame, target):
        return self

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


def test_forecaster_exposes_close_uncertainty_and_confidence():
    forecaster = OHLCForecaster()
    forecaster.features = ["feature"]
    forecaster.models = {target: _ConstantModel(0.01) for target in TARGETS}
    forecaster.global_uncertainty = {target: 0.02 for target in TARGETS}
    frame = pd.DataFrame({
        "date": [pd.Timestamp("2026-01-01")], "symbol": ["AAA"],
        "close": [100.0], "feature": [1.0],
        "market_ret_20d": [0.0], "market_volatility_20": [0.01],
        "market_breadth": [0.5],
    })
    out = forecaster.predict(frame)
    assert out.loc[0, "pred_close_uncertainty_pct"] == pytest.approx(0.02)
    assert out.loc[0, "pred_ohlc_uncertainty_pct"] == pytest.approx(0.02)
    assert out.loc[0, "forecast_confidence"] == pytest.approx(1.0 / 1.2)


def test_evaluate_reports_uncertainty_interval_coverage():
    df = pd.DataFrame({
        "base_close": [100.0, 100.0], "pred_close": [102.0, 102.0],
        "actual_close": [103.0, 106.0], "pred_close_uncertainty_pct": [0.02, 0.02],
        "pred_open": [101.0, 101.0], "pred_high": [103.0, 103.0], "pred_low": [99.0, 99.0],
        "actual_open": [101.0, 101.0], "actual_high": [103.0, 103.0], "actual_low": [99.0, 99.0],
    })
    metrics = evaluate(df)
    assert metrics["uncertainty_samples"] == 2
    assert metrics["close_interval_coverage"] == pytest.approx(0.5)


def test_adverse_regime_can_require_return_relative_to_uncertainty():
    frame = pd.DataFrame([{
        "symbol": "AAA", "close": 100.0, "pred_close": 102.0,
        "rank_confidence": 1.0, "forecast_confidence": 0.9,
        "pred_close_uncertainty_pct": 0.05,
        "atr_pct_14": 0.02, "volatility_20": 0.03,
        "downside_volatility_20": 0.02, "volume_ratio_20": 1.0,
        "market_regime": "bear",
    }])
    out = final_trade_decision(frame, RiskConfig(max_position_pct=1.0))
    assert out.iloc[0]["decision"] == "NO_TRADE"
    assert "uncertainty_return" in out.iloc[0]["risk_reason"]
