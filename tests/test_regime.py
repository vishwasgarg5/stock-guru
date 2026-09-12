import pandas as pd

from stock_guru.regime import add_market_regime_features, confidence_from_rank, regime_label


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


def test_market_returns_are_equal_weighted_not_price_weighted():
    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    raw = pd.DataFrame({
        "date": [dates[0], dates[0], dates[1], dates[1]],
        "symbol": ["LOW", "HIGH", "LOW", "HIGH"],
        "open": [10, 1000, 10, 1000],
        "high": [11, 1100, 11, 1100],
        "low": [9, 900, 9, 900],
        "close": [10, 1000, 11, 1050],
        "volume": [100, 100, 100, 100],
    })
    result = add_market_regime_features(raw)
    day_two = result[result["date"] == dates[1]]
    expected = ((11 / 10 - 1) + (1050 / 1000 - 1)) / 2
    assert day_two["market_ret_1d"].iloc[0] == expected
