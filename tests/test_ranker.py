import pandas as pd
import pytest

from stock_guru.ranker import StockRanker


def test_relevance_stays_within_xgboost_ndcg_range_and_preserves_order():
    values = pd.Series([0.30, -0.10, 0.20, 0.05, 0.90] + [float(i) for i in range(5, 500)])
    relevance = StockRanker.relevance(values)
    assert relevance.dtype.kind in "iu"
    assert relevance.min() == 0
    assert relevance.max() == 31
    assert (relevance >= 0).all()
    assert (relevance <= 31).all()
    assert relevance.loc[values.idxmin()] == 0
    assert relevance.loc[values.idxmax()] == 31


def test_ranking_target_prefers_return_per_unit_of_downside_risk():
    df = pd.DataFrame({
        "target_return": [0.04, 0.06],
        "downside_volatility_20": [0.01, 0.03],
    })
    target = StockRanker.build_ranking_target(df)
    assert target.iloc[0] == pytest.approx(4.0)
    assert target.iloc[1] == pytest.approx(2.0)
    assert target.iloc[0] > target.iloc[1]


def test_ranking_target_falls_back_to_return_without_risk_data():
    df = pd.DataFrame({"target_return": [0.01, -0.02]})
    target = StockRanker.build_ranking_target(df)
    assert target.tolist() == [0.01, -0.02]
