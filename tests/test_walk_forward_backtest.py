import pandas as pd

from stock_guru.walk_forward_backtest import run_strategy_walk_forward


def test_strategy_walk_forward_rejects_short_history():
    raw = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=5),
        "symbol": ["AAA"] * 5,
        "open": [100] * 5,
        "high": [101] * 5,
        "low": [99] * 5,
        "close": [100, 101, 102, 103, 104],
        "volume": [1000] * 5,
    })
    try:
        run_strategy_walk_forward(raw, min_train_days=10)
    except ValueError as exc:
        assert "Not enough dates" in str(exc)
    else:
        raise AssertionError("Expected short-history validation")


def test_strategy_walk_forward_reports_missing_pit_context(monkeypatch):
    import stock_guru.walk_forward_backtest as module

    monkeypatch.setattr(module, "run_walk_forward_with_predictions", lambda *args, **kwargs: ([], []))
    raw = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"]), "symbol": ["AAA"]})
    result = run_strategy_walk_forward(raw)
    assert result["pit_context"]["fundamentals_supplied"] is False
    assert result["pit_context"]["universe_intervals_supplied"] is False
    assert result["pit_context"]["research_warning"]
