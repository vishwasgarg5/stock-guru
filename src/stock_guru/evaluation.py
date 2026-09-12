from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ACTUAL = ["actual_open", "actual_high", "actual_low", "actual_close"]
PRED = ["pred_open", "pred_high", "pred_low", "pred_close"]


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
    return result


def ranking_metrics(predictions: pd.DataFrame, k: int = 10) -> dict:
    """Evaluate whether the model's top-ranked stocks outperform the universe."""
    p = predictions.copy()
    if "rank_score" in p.columns:
        score = "rank_score"
    elif "rank" in p.columns:
        score = None
    else:
        return {}

    p["actual_return"] = p["actual_close"] / p["base_close"] - 1
    if score:
        p = p.sort_values(score, ascending=False)
    else:
        p = p.sort_values("rank")
    top = p.groupby("prediction_date", group_keys=False).head(k) if "prediction_date" in p else p.head(k)
    universe_mean = float(p["actual_return"].mean())
    top_mean = float(top["actual_return"].mean()) if not top.empty else float("nan")
    result = {
        "precision_at_k": float((top["actual_return"] > 0).mean()) if not top.empty else float("nan"),
        "top_k_mean_return": top_mean,
        "universe_mean_return": universe_mean,
        "top_k_excess_return": top_mean - universe_mean if not top.empty else float("nan"),
    }
    # NDCG-style score using non-negative shifted next-day returns as relevance.
    relevance = p["actual_return"] - p["actual_return"].min() + 1e-12
    discounts = 1.0 / np.log2(np.arange(2, len(p) + 2))
    dcg = float(np.sum(relevance.to_numpy() * discounts))
    ideal = np.sort(relevance.to_numpy())[::-1]
    idcg = float(np.sum(ideal * discounts))
    result["ndcg"] = dcg / idcg if idcg > 0 else 0.0
    return result


def attach_actuals(pred: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    actual = market[["date", "symbol", "open", "high", "low", "close"]].copy()
    actual = actual.rename(columns={"open":"actual_open", "high":"actual_high", "low":"actual_low", "close":"actual_close"})
    out = pred.copy().rename(columns={"close":"base_close"})
    return out.merge(actual, on=["date", "symbol"], how="inner")
