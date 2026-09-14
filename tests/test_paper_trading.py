import pandas as pd
import pytest

from stock_guru.paper_trading import PaperConfig, execute_signals


def test_execute_signals_uses_next_session_and_applies_costs():
    signals = pd.DataFrame({
        "prediction_date": pd.to_datetime(["2026-01-01"]),
        "symbol": ["A"],
        "position_weight": [0.50],
    })
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "symbol": ["A", "A"],
        "open": [100.0, 110.0],
        "close": [105.0, 120.0],
    })
    out = execute_signals(signals, prices, 1_000_000, PaperConfig(max_position_pct=0.10, commission_bps=10, slippage_bps=5))
    row = out.iloc[0]
    assert row["entry_date"] == pd.Timestamp("2026-01-02")
    assert row["entry_price"] > 110
    assert row["quantity"] > 0
    assert row["net_pnl"] > 0


def test_execute_signals_caps_position_weight():
    signals = pd.DataFrame({
        "prediction_date": ["2026-01-01"], "symbol": ["A"], "position_weight": [0.50]
    })
    prices = pd.DataFrame({
        "date": ["2026-01-02"], "symbol": ["A"], "open": [100.0], "close": [100.0]
    })
    out = execute_signals(signals, prices, 1_000_000, PaperConfig(max_position_pct=0.10))
    assert out.iloc[0]["allocated_capital"] == 100_000


def test_execute_signals_shares_cash_budget_across_same_day_signals():
    signals = pd.DataFrame({
        "prediction_date": ["2026-01-01", "2026-01-01"],
        "symbol": ["A", "B"],
        "position_weight": [0.10, 0.10],
    })
    prices = pd.DataFrame({
        "date": ["2026-01-02", "2026-01-02"],
        "symbol": ["A", "B"],
        "open": [100.0, 100.0],
        "close": [100.0, 100.0],
    })
    out = execute_signals(signals, prices, 1_000.0, PaperConfig(commission_bps=0, slippage_bps=0))
    assert out["allocated_capital"].sum() == 200.0
    assert (out["cash_after"] == 1_000.0).all()
    assert (out["equity_after"] == 1_000.0).all()


def test_execute_signals_rejects_invalid_inputs():
    signals = pd.DataFrame({
        "prediction_date": ["2026-01-01"], "symbol": ["A"], "position_weight": [-0.1]
    })
    prices = pd.DataFrame({
        "date": ["2026-01-02"], "symbol": ["A"], "open": [100.0], "close": [100.0]
    })
    with pytest.raises(ValueError, match="non-negative"):
        execute_signals(signals, prices)
