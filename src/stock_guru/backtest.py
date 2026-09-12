from __future__ import annotations

import numpy as np
import pandas as pd


def backtest(predictions: pd.DataFrame, transaction_cost_bps: float = 10.0,
             benchmark: pd.DataFrame | None = None, initial_capital: float = 1.0) -> dict:
    """Backtest daily risk-approved positions using next-session close returns.

    Predictions must represent decisions made before the next trading session and
    contain prediction_date, symbol, position_weight, trade and actual_close.
    """
    required = {"prediction_date", "symbol", "position_weight", "trade", "base_close", "actual_close"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing backtest columns: {sorted(missing)}")

    p = predictions.copy()
    p["prediction_date"] = pd.to_datetime(p["prediction_date"]).dt.normalize()
    p["gross_return"] = p["actual_close"] / p["base_close"] - 1.0
    p["position_weight"] = p["position_weight"].fillna(0.0).clip(lower=0.0)
    p.loc[~p["trade"].astype(bool), "position_weight"] = 0.0

    daily = []
    previous = {}
    cost = transaction_cost_bps / 10000.0
    for date, g in p.groupby("prediction_date", sort=True):
        current = dict(zip(g["symbol"], g["position_weight"]))
        symbols = set(previous) | set(current)
        turnover = sum(abs(current.get(s, 0.0) - previous.get(s, 0.0)) for s in symbols)
        gross = float((g["position_weight"] * g["gross_return"]).sum())
        net = gross - turnover * cost
        daily.append({"date": date, "gross_return": gross, "turnover": turnover,
                      "transaction_cost": turnover * cost, "net_return": net})
        previous = current

    if not daily:
        return {"days": 0}
    curve = pd.DataFrame(daily).sort_values("date")
    curve["equity"] = initial_capital * (1.0 + curve["net_return"]).cumprod()
    running_max = curve["equity"].cummax()
    drawdown = curve["equity"] / running_max - 1.0
    returns = curve["net_return"]
    annualized_return = float(curve["equity"].iloc[-1] ** (252.0 / len(curve)) - 1.0)
    annualized_vol = float(returns.std(ddof=1) * np.sqrt(252)) if len(curve) > 1 else 0.0
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)) if len(curve) > 1 and returns.std(ddof=1) > 0 else 0.0
    winning = returns[returns > 0]

    result = {
        "days": int(len(curve)),
        "final_equity": float(curve["equity"].iloc[-1]),
        "total_return": float(curve["equity"].iloc[-1] / initial_capital - 1.0),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "average_daily_return": float(returns.mean()),
        "win_rate": float((returns > 0).mean()),
        "average_turnover": float(curve["turnover"].mean()),
        "total_transaction_cost": float(curve["transaction_cost"].sum()),
        "trading_days_with_positive_return": int((returns > 0).sum()),
    }

    if benchmark is not None and {"date", "close"}.issubset(benchmark.columns):
        b = benchmark.copy()
        b["date"] = pd.to_datetime(b["date"]).dt.normalize()
        b = b.sort_values("date").drop_duplicates("date")
        b = b[b["date"].isin(curve["date"])]
        if len(b) >= 2:
            bret = b["close"].pct_change().dropna()
            result["benchmark_total_return"] = float((1.0 + bret).prod() - 1.0)
            result["benchmark_sharpe"] = float(bret.mean() / bret.std(ddof=1) * np.sqrt(252)) if bret.std(ddof=1) > 0 else 0.0
            result["benchmark_max_drawdown"] = float(((1.0 + bret).cumprod() / (1.0 + bret).cumprod().cummax() - 1.0).min())

    return result
