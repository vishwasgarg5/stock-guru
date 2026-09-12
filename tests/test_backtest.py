import pandas as pd

from stock_guru.backtest import backtest


def test_backtest_calculates_return_and_turnover():
    predictions = pd.DataFrame([
        {"prediction_date": "2025-01-02", "symbol": "AAA", "position_weight": 0.5, "trade": True, "base_close": 100, "actual_close": 102},
        {"prediction_date": "2025-01-02", "symbol": "BBB", "position_weight": 0.5, "trade": True, "base_close": 100, "actual_close": 99},
        {"prediction_date": "2025-01-03", "symbol": "AAA", "position_weight": 0.0, "trade": False, "base_close": 102, "actual_close": 101},
        {"prediction_date": "2025-01-03", "symbol": "BBB", "position_weight": 1.0, "trade": True, "base_close": 99, "actual_close": 100},
    ])
    result = backtest(predictions, transaction_cost_bps=0)
    assert result["days"] == 2
    assert result["total_return"] > 0
    assert result["average_turnover"] > 0
    assert result["max_drawdown"] <= 0


def test_backtest_reports_regime_metrics():
    predictions = pd.DataFrame([
        {"prediction_date": "2025-01-02", "symbol": "AAA", "position_weight": 1.0, "trade": True, "base_close": 100, "actual_close": 102, "market_regime": "bull"},
        {"prediction_date": "2025-01-03", "symbol": "AAA", "position_weight": 1.0, "trade": True, "base_close": 102, "actual_close": 99, "market_regime": "bear"},
    ])
    result = backtest(predictions, transaction_cost_bps=0)
    assert set(result["regime_metrics"]) == {"bull", "bear"}
    assert result["regime_metrics"]["bull"]["days"] == 1
    assert result["regime_metrics"]["bear"]["days"] == 1
    assert result["regime_metrics"]["bull"]["total_return"] > 0
    assert result["regime_metrics"]["bear"]["total_return"] < 0


def test_backtest_rejects_missing_columns():
    try:
        backtest(pd.DataFrame([{"symbol": "AAA"}]))
    except ValueError as exc:
        assert "Missing backtest columns" in str(exc)
    else:
        raise AssertionError("Expected missing-column validation")
