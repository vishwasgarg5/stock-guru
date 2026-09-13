from __future__ import annotations

import pandas as pd

from .backtest import backtest, cost_sensitivity
from .walk_forward import run_walk_forward_with_predictions
from .risk import final_trade_decision


def run_strategy_walk_forward(raw: pd.DataFrame, min_train_days: int = 252,
                              step_days: int = 20, top_k: int = 10,
                              transaction_cost_bps: float = 10.0) -> dict:
    """Run leakage-safe walk-forward predictions and backtest selected trades.

    Each fold is fitted exactly once. Forecasts are labeled with the following
    session's actual close, risk-filtered, and passed to the portfolio backtester.
    """
    fold_results, fold_predictions = run_walk_forward_with_predictions(
        raw, min_train_days=min_train_days, step_days=step_days, top_k=top_k
    )

    all_predictions = [final_trade_decision(pred.copy()) for pred in fold_predictions]
    all_predictions = [pred for pred in all_predictions if not pred.empty]

    if not all_predictions:
        return {
            "folds": len(fold_results),
            "prediction_rows": 0,
            "portfolio": {"days": 0},
            "fold_results": fold_results,
        }

    predictions = pd.concat(all_predictions, ignore_index=True)
    portfolio = backtest(predictions, transaction_cost_bps=transaction_cost_bps)
    sensitivity = cost_sensitivity(predictions)
    return {
        "folds": len(fold_results),
        "prediction_rows": len(predictions),
        "portfolio": portfolio,
        "cost_sensitivity": sensitivity,
        "fold_results": fold_results,
        "predictions": predictions,
    }
