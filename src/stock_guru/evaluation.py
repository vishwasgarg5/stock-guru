from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ACTUAL = ["actual_open", "actual_high", "actual_low", "actual_close"]
PRED = ["pred_open", "pred_high", "pred_low", "pred_close"]


def _ndcg(scores: np.ndarray, relevance: np.ndarray, k: int) -> float:
    order = np.argsort(scores)[::-1][:k]
    ideal = np.sort(relevance)[::-1][:k]
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float(np.sum(relevance[order] * discounts))
    idcg = float(np.sum(ideal * discounts))
    return dcg / idcg if idcg > 0 else 0.0


def ranking_metrics(predictions: pd.DataFrame, k: int = 10) -> dict:
    """Measure cross-sectional stock-selection quality for each prediction date."""
    required = {"symbol", "actual_close", "base_close"}
    if not required.issubset(predictions.columns):
        return {}
    score_col = "rank_score" if "rank_score" in predictions.columns else None
    if score_col is None and "rank" not in predictions.columns:
        return {}

    p = predictions.copy()
    p["actual_return"] = p["actual_close"] / p["base_close"] - 1.0
    groups = p.groupby("prediction_date", sort=False) if "prediction_date" in p.columns else [("all", p)]

    precisions, top_returns, universe_returns, excess, ndcgs = [], [], [], [], []
    for _, g in groups:
        g = g.dropna(subset=["actual_return"])
        if g.empty:
            continue
        ordered = g.sort_values(score_col, ascending=False) if score_col else g.sort_values("rank")
        top = ordered.head(k)
        universe_mean = float(g["actual_return"].mean())
        top_mean = float(top["actual_return"].mean())
        precisions.append(float((top["actual_return"] > 0).mean()))
        top_returns.append(top_mean)
        universe_returns.append(universe_mean)
        excess.append(top_mean - universe_mean)
        relevance = g["actual_return"].to_numpy()
        relevance = relevance - relevance.min() + 1e-12
        scores = ordered[score_col].to_numpy() if score_col else -ordered["rank"].to_numpy()
        ndcgs.append(_ndcg(scores, relevance, k))

    if not precisions:
        return {}
    return {
        "precision_at_k": float(np.mean(precisions)),
        "top_k_mean_return": float(np.mean(top_returns)),
        "universe_mean_return": float(np.mean(universe_returns)),
        "top_k_excess_return": float(np.mean(excess)),
        "ndcg_at_k": float(np.mean(ndcgs)),
    }


def _grouped_forecast_metrics(predictions: pd.DataFrame) -> dict:
    """Compute forecasting metrics by market regime without using future features."""
    if "market_regime" not in predictions.columns:
        return {}
    rows = {}
    for regime, group in predictions.groupby("market_regime", dropna=False):
        label = "unknown" if pd.isna(regime) else str(regime)
        if group.empty:
            continue
        metrics = {}
        for a, p in zip(ACTUAL, PRED):
            y = group[a].astype(float)
            yh = group[p].astype(float)
            metrics[f"{p}_mae"] = float(mean_absolute_error(y, yh))
            metrics[f"{p}_rmse"] = float(np.sqrt(mean_squared_error(y, yh)))
        metrics["close_direction_accuracy"] = float(
            np.mean(np.sign(group["pred_close"] - group["base_close"]) ==
                    np.sign(group["actual_close"] - group["base_close"]))
        )
        metrics["samples"] = int(len(group))
        rows[label] = metrics
    return rows


def evaluate(predictions: pd.DataFrame) -> dict:
    result = {}
    for a, p in zip(ACTUAL, PRED):
        y = predictions[a].astype(float)
        yh = predictions[p].astype(float)
        result[f"{p}_mae"] = float(mean_absolute_error(y, yh))
        result[f"{p}_rmse"] = float(np.sqrt(mean_squared_error(y, yh)))
        result[f"{p}_mape_pct"] = float(np.mean(np.abs((y - yh) / y.replace(0, np.nan))) * 100)
    result["close_direction_accuracy"] = float(
        np.mean(np.sign(predictions["pred_close"] - predictions["base_close"]) ==
                np.sign(predictions["actual_close"] - predictions["base_close"]))
    )
    result.update(ranking_metrics(predictions))
    regime_metrics = _grouped_forecast_metrics(predictions)
    if regime_metrics:
        result["regime_metrics"] = regime_metrics
    return result


def attach_actuals(pred: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    actual = market[["date", "symbol", "open", "high", "low", "close"]].copy()
    actual = actual.rename(columns={"open":"actual_open", "high":"actual_high", "low":"actual_low", "close":"actual_close"})
    out = pred.copy().rename(columns={"close":"base_close"})
    return out.merge(actual, on=["date", "symbol"], how="inner")
