import pandas as pd

from stock_guru.regime import confidence_from_rank, regime_label


def test_confidence_is_relative_and_ordered():
    scores = pd.Series([0.1, 0.5, 0.9])
    confidence = confidence_from_rank(scores)
    assert list(confidence) == [1 / 3, 2 / 3, 1.0]


def test_bull_regime():
    row = pd.Series({
        "market_ret_20d": 0.05,
        "market_volatility_20": 0.01,
        "market_breadth": 0.70,
    })
    assert regime_label(row) == "bull"


def test_high_vol_bear_regime():
    row = pd.Series({
        "market_ret_20d": -0.05,
        "market_volatility_20": 0.03,
        "market_breadth": 0.35,
    })
    assert regime_label(row) == "high_vol_bear"
