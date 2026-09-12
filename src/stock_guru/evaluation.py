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
    return result


def attach_actuals(pred: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    actual = market[["date", "symbol", "open", "high", "low", "close"]].copy()
    actual = actual.rename(columns={"open":"actual_open", "high":"actual_high", "low":"actual_low", "close":"actual_close"})
    out = pred.copy().rename(columns={"close":"base_close"})
    return out.merge(actual, on=["date", "symbol"], how="inner")
