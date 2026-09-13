import pandas as pd
import pytest

from stock_guru.retrainer import _training_history, should_accept
from stock_guru.model_selection import should_promote, summarize, summarize_feedback
from stock_guru.walk_forward import FoldResult


def test_retraining_accepts_only_validation_improvement():
    decision = should_accept({"pred_close_rmse": 2.0}, {"pred_close_rmse": 1.9})
    assert decision.accepted
    assert not should_accept({"pred_close_rmse": 2.0}, {"pred_close_rmse": 2.0}).accepted


def test_promotion_rejects_ranking_regression():
    old = {"pred_close_rmse": 2.0, "close_direction_accuracy": 0.60, "top_k_excess_return": 0.004, "precision_at_k": 0.55}
    new = {"pred_close_rmse": 1.9, "close_direction_accuracy": 0.61, "top_k_excess_return": 0.003, "precision_at_k": 0.56}
    assert not should_promote(old, new)


def test_promotion_accepts_forecast_and_ranking_improvement():
    old = {"pred_close_rmse": 2.0, "close_direction_accuracy": 0.60, "top_k_excess_return": 0.004, "precision_at_k": 0.55}
    new = {"pred_close_rmse": 1.9, "close_direction_accuracy": 0.61, "top_k_excess_return": 0.005, "precision_at_k": 0.57}
    assert should_promote(old, new)


def test_promotion_respects_accumulated_feedback_direction_accuracy():
    old = {"pred_close_rmse": 2.0, "close_direction_accuracy": 0.60}
    new = {"pred_close_rmse": 1.9, "close_direction_accuracy": 0.61}
    feedback = {"feedback_rows": 25, "feedback_direction_accuracy": 0.62, "feedback_return_mae": 0.01}
    assert not should_promote(old, new, feedback=feedback)


def test_promotion_rejects_insufficient_validation_folds():
    old = {"pred_close_rmse": 2.0, "close_direction_accuracy": 0.60}
    new = {"pred_close_rmse": 1.9, "close_direction_accuracy": 0.61, "validation_folds": 19}
    assert not should_promote(old, new)


def test_promotion_rejects_weak_bear_regime_direction_accuracy():
    old = {"pred_close_rmse": 2.0, "close_direction_accuracy": 0.50}
    new = {
        "pred_close_rmse": 1.9,
        "close_direction_accuracy": 0.51,
        "validation_folds": 20,
        "regime_metrics": {"bear": {"samples": 50, "close_direction_accuracy": 0.44}},
    }
    assert not should_promote(old, new)


def test_promotion_accepts_robust_adverse_regime_candidate():
    old = {"pred_close_rmse": 2.0, "close_direction_accuracy": 0.50, "top_k_excess_return": 0.004, "precision_at_k": 0.55}
    new = {
        "pred_close_rmse": 1.9,
        "close_direction_accuracy": 0.51,
        "top_k_excess_return": 0.005,
        "precision_at_k": 0.56,
        "validation_folds": 20,
        "regime_metrics": {
            "bear": {"samples": 50, "close_direction_accuracy": 0.45},
            "high_vol_bear": {"samples": 10, "close_direction_accuracy": 0.50},
        },
    }
    assert should_promote(old, new)


def test_promotion_keeps_legacy_metric_dicts_compatible():
    old = {"pred_close_rmse": 2.0, "close_direction_accuracy": 0.60}
    new = {"pred_close_rmse": 1.9, "close_direction_accuracy": 0.61}
    assert should_promote(old, new)


def test_feedback_summary_requires_minimum_history():
    feedback = pd.DataFrame({"direction_correct": [True] * 19, "return_error": [0.01] * 19})
    assert summarize_feedback(feedback) is None


def test_feedback_summary_uses_accumulated_rows():
    feedback = pd.DataFrame({"direction_correct": [True, False, True, True] * 5, "return_error": [0.01, -0.02, 0.03, -0.01] * 5})
    summary = summarize_feedback(feedback, min_rows=20)
    assert summary["feedback_rows"] == 20
    assert summary["feedback_direction_accuracy"] == 0.75
    assert summary["feedback_return_mae"] == 0.0175


def test_validation_summary_aggregates_regime_metrics_without_breaking_scalar_metrics():
    results = [
        FoldResult("2025-01-01", "2025-01-02", {"pred_close_rmse": 0.10}, {"bull": {"samples": 2, "pred_close_rmse": 0.08}}),
        FoldResult("2025-01-02", "2025-01-03", {"pred_close_rmse": 0.20}, {"bull": {"samples": 1, "pred_close_rmse": 0.14}, "bear": {"samples": 1, "pred_close_rmse": 0.30}}),
    ]
    summary = summarize(results)
    assert summary["pred_close_rmse"] == pytest.approx(0.15)
    assert summary["validation_folds"] == 2
    assert summary["regime_metrics"]["bull"]["samples"] == 3
    assert summary["regime_metrics"]["bull"]["pred_close_rmse"] == pytest.approx((0.08 * 2 + 0.14) / 3)
    assert summary["regime_metrics"]["bear"]["pred_close_rmse"] == pytest.approx(0.30)


def test_training_history_excludes_prediction_date_and_future_sessions():
    raw = pd.DataFrame({"date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]), "symbol": ["AAA"] * 4, "close": [100.0, 101.0, 102.0, 103.0]})
    history = _training_history(raw, prediction_date="2025-01-06")
    assert history["date"].tolist() == list(pd.to_datetime(["2025-01-02", "2025-01-03"]))
    assert (history["date"] < pd.Timestamp("2025-01-06")).all()


def test_training_history_without_cutoff_preserves_all_history():
    raw = pd.DataFrame({"date": pd.to_datetime(["2025-01-02", "2025-01-03"]), "symbol": ["BBB", "AAA"], "close": [200.0, 100.0]})
    history = _training_history(raw)
    assert history["date"].tolist() == list(pd.to_datetime(["2025-01-02", "2025-01-03"]))
    assert history["symbol"].tolist() == ["BBB", "AAA"]
