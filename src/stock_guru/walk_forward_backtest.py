from __future__ import annotations

import pandas as pd

from .backtest import backtest
from .walk_forward import run_walk_forward
from .regime import regime_label


def run_strategy_walk_forward(raw: pd.DataFrame, min_train_days: int = 252,
                              step_days: int = 20, top_k: int = 10,
                              transaction_cost_bps: float = 10.0) -> dict:
    """Run leakage-safe walk-forward predictions and backtest selected trades.

    Each fold is trained only on dates before the prediction date. The resulting
    predictions are labeled with the following session's actual close and then
    passed to the portfolio backtester.
    """
    dates = sorted(pd.to_datetime(raw["date"]).dt.normalize().unique())
    if len(dates) <= min_train_days + 1:
        raise ValueError("Not enough dates for strategy walk-forward validation")

    all_predictions = []
    fold_results = run_walk_forward(raw, min_train_days=min_train_days,
                                    step_days=step_days, top_k=top_k)

    from .features import build_features
    from .pipeline import Pipeline
    from .feedback import label_predictions
    from .risk import final_trade_decision

    features, _ = build_features(raw)
    normalized = pd.to_datetime(features["date"]).dt.normalize()
    raw_dates = pd.to_datetime(raw["date"]).dt.normalize()

    for idx in range(min_train_days, len(dates) - 1, step_days):
        train_end = dates[idx - 1]
        prediction_date = dates[idx]
        train = raw[raw_dates <= train_end].copy()
        pipe = Pipeline(top_k=top_k).train(train)
        day = features[normalized == prediction_date].copy().dropna(subset=pipe.features)
        if day.empty:
            continue
        ranked = pipe.ranker.score(day).head(top_k)
        pred = pipe.forecaster.predict(ranked)
        if pred.empty:
            continue
        if "market_regime" in day.columns:
            regime_map = day[["symbol", "market_ret_20d", "market_volatility_20", "market_breadth"]].drop_duplicates("symbol").set_index("symbol")
            pred = pred.join(regime_map, on="symbol")
            pred["market_regime"] = pred.apply(regime_label, axis=1)
        else:
            pred["market_regime"] = "unknown"
        labeled = label_predictions(pred, raw)
        labeled = labeled[labeled["prediction_date"] == prediction_date].copy()
        if labeled.empty:
            continue
        labeled = final_trade_decision(labeled)
        all_predictions.append(labeled)

    if not all_predictions:
        return {"folds": len(fold_results), "prediction_rows": 0, "portfolio": {"days": 0}, "fold_results": fold_results}

    predictions = pd.concat(all_predictions, ignore_index=True)
    portfolio = backtest(predictions, transaction_cost_bps=transaction_cost_bps)
    return {
        "folds": len(fold_results),
        "prediction_rows": len(predictions),
        "portfolio": portfolio,
        "fold_results": fold_results,
        "predictions": predictions,
    }
