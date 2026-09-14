from __future__ import annotations

import pandas as pd


def reconcile_trades(trades: pd.DataFrame, *, initial_capital: float) -> dict[str, object]:
    """Reconcile persisted paper trades against the starting capital."""
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    required = {"prediction_date", "symbol", "quantity", "net_pnl", "cash_after", "equity_after"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"Missing trade columns: {sorted(missing)}")
    if trades.empty:
        return {"trade_rows": 0, "prediction_dates": 0, "final_cash": initial_capital, "final_equity": initial_capital, "net_pnl": 0.0, "reconciled": True}
    df = trades.copy()
    df["prediction_date"] = pd.to_datetime(df["prediction_date"], errors="coerce").dt.normalize()
    if df["prediction_date"].isna().any():
        raise ValueError("Trades contain invalid prediction dates")
    for col in ["quantity", "net_pnl", "cash_after", "equity_after"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["quantity", "net_pnl", "cash_after", "equity_after"]].isna().any().any():
        raise ValueError("Trade reconciliation fields contain non-numeric values")
    if (df["quantity"] < 0).any():
        raise ValueError("Trade quantities cannot be negative")
    final = df.sort_values(["prediction_date", "symbol"]).iloc[-1]
    net_pnl = float(df["net_pnl"].sum())
    final_equity = float(final["equity_after"])
    tolerance = max(1e-8, abs(initial_capital) * 1e-10)
    return {
        "trade_rows": int(len(df)),
        "prediction_dates": int(df["prediction_date"].nunique()),
        "final_cash": float(final["cash_after"]),
        "final_equity": final_equity,
        "net_pnl": net_pnl,
        "equity_minus_initial_minus_pnl": final_equity - initial_capital - net_pnl,
        "reconciled": abs(final_equity - initial_capital - net_pnl) <= tolerance,
    }
