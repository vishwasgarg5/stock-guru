from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from .evaluation import attach_actuals, evaluate
from .features import build_features
from .pipeline import Pipeline


@dataclass
class FoldResult:
    train_end: str
    prediction_date: str
    metrics: dict


def run_walk_forward(raw: pd.DataFrame, min_train_days: int = 252, step_days: int = 20, top_k: int = 10) -> list[FoldResult]:
    """Run expanding-window validation without allowing future labels into training."""
    dates = sorted(pd.to_datetime(raw["date"]).dt.normalize().unique())
    if len(dates) <= min_train_days:
        raise ValueError("Not enough dates for walk-forward validation")
    results: list[FoldResult] = []
    full_features, _ = build_features(raw)
    for idx in range(min_train_days, len(dates), step_days):
        train_end = dates[idx - 1]
        prediction_date = dates[idx]
        train = raw[pd.to_datetime(raw["date"]).dt.normalize() <= train_end].copy()
        pipe = Pipeline(top_k=top_k).train(train)
        day = full_features[full_features["date"].astype(str).str[:10] == str(prediction_date.date())].copy()
        day = day.dropna(subset=pipe.features)
        ranked = pipe.ranker.score(day).head(top_k)
        pred = pipe.forecaster.predict(ranked)
        if pred.empty:
            continue
        actual_market = raw[pd.to_datetime(raw["date"]).dt.normalize() == prediction_date].copy()
        labeled = attach_actuals(pred, actual_market)
        if labeled.empty:
            continue
        results.append(FoldResult(str(train_end.date()), str(prediction_date.date()), evaluate(labeled)))
    return results
