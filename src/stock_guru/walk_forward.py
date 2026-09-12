from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from .evaluation import attach_actuals, evaluate
from .pipeline import Pipeline


@dataclass
class FoldResult:
    train_end: str
    prediction_date: str
    metrics: dict


def run_walk_forward(raw: pd.DataFrame, min_train_days: int = 252, step_days: int = 20, top_k: int = 10) -> list[FoldResult]:
    """Run expanding-window validation without allowing future rows into training."""
    dates = sorted(pd.to_datetime(raw["date"]).dt.normalize().unique())
    if len(dates) <= min_train_days:
        raise ValueError("Not enough dates for walk-forward validation")
    results: list[FoldResult] = []
    for idx in range(min_train_days, len(dates), step_days):
        train_end = dates[idx - 1]
        prediction_date = dates[idx]
        train = raw[pd.to_datetime(raw["date"]).dt.normalize() <= train_end].copy()
        pipe = Pipeline(top_k=top_k).train(train)
        pred = pipe.predict_date(train, str(prediction_date.date()))
        if pred.empty:
            continue
        actual_market = raw[pd.to_datetime(raw["date"]).dt.normalize() == prediction_date].copy()
        labeled = attach_actuals(pred, actual_market)
        if labeled.empty:
            continue
        results.append(FoldResult(str(train_end.date()), str(prediction_date.date()), evaluate(labeled)))
    return results
