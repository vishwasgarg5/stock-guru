import pandas as pd
import pytest

from stock_guru.paper_reconciliation import reconcile_trades


def test_reconcile_trades_reports_consistent_equity():
    trades = pd.DataFrame({
        "prediction_date": ["2024-01-02", "2024-01-03"],
        "symbol": ["A", "B"],
        "quantity": [10, 5],
        "net_pnl": [100.0, -20.0],
        "cash_after": [1000100.0, 1000080.0],
        "equity_after": [1000100.0, 1000080.0],
    })
    report = reconcile_trades(trades, initial_capital=1_000_000)
    assert report["net_pnl"] == 80.0
    assert report["final_equity"] == 1_000_080.0
    assert report["reconciled"] is True


def test_reconcile_trades_rejects_missing_fields():
    with pytest.raises(ValueError, match="Missing trade columns"):
        reconcile_trades(pd.DataFrame({"symbol": ["A"]}), initial_capital=1000)
