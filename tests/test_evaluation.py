import pandas as pd

from stock_guru.evaluation import ranking_metrics


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
