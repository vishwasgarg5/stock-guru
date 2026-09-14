import pandas as pd

from stock_guru.calibration import confidence_calibration_table, uncertainty_scale_for_coverage, apply_uncertainty_scale
from stock_guru.monitoring import model_health_summary
from stock_guru.risk import RiskConfig, size_positions


def _eligible(symbols):
    n = len(symbols)
    return pd.DataFrame({
        "symbol": symbols,
        "sector": ["IT"] * n,
        "ai_score": list(range(n, 0, -1)),
        "rank_confidence": [0.9] * n,
        "forecast_confidence": [0.9] * n,
        "pred_close_uncertainty_pct": [0.01] * n,
        "atr_pct_14": [0.02] * n,
        "risk_pass": [True] * n,
        "market_regime": ["neutral"] * n,
    })


def test_confidence_calibration_table_reports_direction_accuracy():
    frame = pd.DataFrame({
        "forecast_confidence": [0.55, 0.55, 0.85, 0.85],
        "actual_close": [101, 99, 101, 101],
        "pred_close": [101, 101, 101, 99],
        "base_close": [100, 100, 100, 100],
    })
    table = confidence_calibration_table(frame)
    assert table
    assert any(row["samples"] == 2 for row in table.values())


def test_uncertainty_scale_targets_empirical_coverage():
    frame = pd.DataFrame({
        "actual_close": [101, 103, 99, 100],
        "pred_close": [100, 100, 100, 100],
        "base_close": [100, 100, 100, 100],
        "pred_close_uncertainty_pct": [0.01] * 4,
    })
    scale = uncertainty_scale_for_coverage(frame, 0.75)
    assert scale > 0
    out = apply_uncertainty_scale(frame, scale)
    assert (out["pred_close_uncertainty_pct"] > 0.01).all()


def test_correlation_cap_removes_lower_signal_position():
    frame = _eligible(["AAA", "BBB"])
    matrix = pd.DataFrame([[1.0, 0.95], [0.95, 1.0]], index=["AAA", "BBB"], columns=["AAA", "BBB"])
    out = size_positions(frame, RiskConfig(max_pairwise_correlation=0.85), correlation_matrix=matrix)
    assert (out["position_weight"] > 0).sum() == 1


def test_model_health_summary_alerts_on_drift_and_weak_feedback():
    out = model_health_summary(
        {"features": 10, "drifted_features": 2},
        {"uncertainty_p90": 0.08},
        {"recent_feedback_direction_accuracy": 0.40},
    )
    assert out["status"] == "degraded"
    assert set(out["alerts"]) == {"input_drift", "low_recent_direction_accuracy", "high_forecast_uncertainty"}
