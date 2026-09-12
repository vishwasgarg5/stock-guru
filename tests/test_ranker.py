import pandas as pd

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
