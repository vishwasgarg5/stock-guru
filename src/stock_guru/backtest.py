from __future__ import annotations

import math
import numpy as np
import pandas as pd


def _regime_portfolio_metrics(curve: pd.DataFrame) -> dict:
    if "market_regime" not in curve.columns:
        return {}
    result = {}
    for regime, group in curve.groupby("market_regime", dropna=False):
        label = "unknown" if pd.isna(regime) else str(regime)
        returns = group["net_return"].astype(float)
        if returns.empty:
            continue
        std = returns.std(ddof=1)
        result[label] = {
            "days": int(len(group)),
            "total_return": float((1.0 + returns).prod() - 1.0),
            "average_daily_return": float(returns.mean()),
            "annualized_volatility": float(std * np.sqrt(252)) if len(group) > 1 else 0.0,
            "sharpe": float(returns.mean() / std * np.sqrt(252)) if len(group) > 1 and std > 0 else 0.0,
            "win_rate": float((returns > 0).mean()),
            "max_drawdown": float(((1.0 + returns).cumprod() / (1.0 + returns).cumprod().cummax() - 1.0).min()),
        }
    return result


def backtest(predictions: pd.DataFrame, transaction_cost_bps: float = 10.0,
             benchmark: pd.DataFrame | None = None, initial_capital: float = 1.0,
             slippage_bps: float = 0.0) -> dict:
    """Backtest daily risk-approved positions using next-session close returns.

    ``transaction_cost_bps`` models explicit fees/taxes. ``slippage_bps`` is a
    separate execution-friction assumption applied to turnover, allowing the
    backtest to stress realistic fills without changing the fee assumption.
    """
    required = {"prediction_date", "symbol", "position_weight", "trade", "base_close", "actual_close"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing backtest columns: {sorted(missing)}")
    if transaction_cost_bps < 0 or slippage_bps < 0:
        raise ValueError("transaction_cost_bps and slippage_bps must be non-negative")

    p = predictions.copy()
    p["prediction_date"] = pd.to_datetime(p["prediction_date"]).dt.normalize()
    p["gross_return"] = p["actual_close"] / p["base_close"] - 1.0
    p["position_weight"] = p["position_weight"].fillna(0.0).clip(lower=0.0)
    p.loc[~p["trade"].astype(bool), "position_weight"] = 0.0

    daily = []
    previous = {}
    fee_rate = transaction_cost_bps / 10000.0
    slippage_rate = slippage_bps / 10000.0
    for date, g in p.groupby("prediction_date", sort=True):
        current = dict(zip(g["symbol"], g["position_weight"]))
        symbols = set(previous) | set(current)
        turnover = sum(abs(current.get(s, 0.0) - previous.get(s, 0.0)) for s in symbols)
        gross = float((g["position_weight"] * g["gross_return"]).sum())
        transaction_cost = turnover * fee_rate
        slippage_cost = turnover * slippage_rate
        net = gross - transaction_cost - slippage_cost
        regime = "unknown"
        if "market_regime" in g.columns and g["market_regime"].notna().any():
            regime = str(g["market_regime"].dropna().iloc[0])
        daily.append({"date": date, "gross_return": gross, "turnover": turnover,
                      "transaction_cost": transaction_cost, "slippage_cost": slippage_cost,
                      "execution_cost": transaction_cost + slippage_cost, "net_return": net,
                      "market_regime": regime})
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
        "total_slippage_cost": float(curve["slippage_cost"].sum()),
        "total_execution_cost": float(curve["execution_cost"].sum()),
        "trading_days_with_positive_return": int((returns > 0).sum()),
        "regime_metrics": _regime_portfolio_metrics(curve),
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


def cost_sensitivity(predictions: pd.DataFrame, cost_scenarios_bps=(0.0, 10.0, 25.0, 50.0),
                     slippage_bps: float = 0.0) -> dict:
    """Evaluate the same OOS trades under multiple fee assumptions and fixed slippage."""
    result = {}
    for bps in cost_scenarios_bps:
        result[str(float(bps)).rstrip("0").rstrip(".")] = backtest(
            predictions, transaction_cost_bps=float(bps), slippage_bps=slippage_bps
        )
    return result
