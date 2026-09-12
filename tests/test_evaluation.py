import pandas as pd

from stock_guru.evaluation import ranking_metrics, evaluate


def test_ranking_metrics_reward_good_top_k():
    df = pd.DataFrame({
        "prediction_date": ["2026-01-02"] * 4,
        "symbol": ["A", "B", "C", "D"],
        "rank_score": [4.0, 3.0, 2.0, 1.0],
        "base_close": [100.0] * 4,
        "actual_close": [110.0, 105.0, 99.0, 95.0],
    })
    metrics = ranking_metrics(df, k=2)
    assert metrics["precision_at_k"] == 1.0
    assert metrics["top_k_excess_return"] > 0
    assert metrics["ndcg_at_k"] > 0.99


def test_evaluate_reports_regime_metrics():
    df = pd.DataFrame({
        "market_regime": ["bull", "bull", "bear", "bear"],
        "base_close": [100.0] * 4,
        "pred_open": [101.0] * 4,
        "pred_high": [103.0] * 4,
        "pred_low": [99.0] * 4,
        "pred_close": [102.0, 101.0, 98.0, 99.0],
        "actual_open": [101.0] * 4,
        "actual_high": [103.0] * 4,
        "actual_low": [99.0] * 4,
        "actual_close": [103.0, 100.0, 97.0, 100.0],
        "symbol": ["A", "B", "C", "D"],
        "rank_score": [4.0, 3.0, 2.0, 1.0],
    })
    metrics = evaluate(df)
    assert set(metrics["regime_metrics"]) == {"bull", "bear"}
    assert metrics["regime_metrics"]["bull"]["samples"] == 2
    assert metrics["regime_metrics"]["bear"]["samples"] == 2
    assert 0.0 <= metrics["regime_metrics"]["bull"]["close_direction_accuracy"] <= 1.0
