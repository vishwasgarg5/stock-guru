from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from .feedback import label_predictions, score_labeled
from .features import build_features
from .pipeline import Pipeline


@dataclass
class FoldResult:
    train_end: str
    prediction_date: str
    metrics: dict


def run_walk_forward(raw: pd.DataFrame, min_train_days: int = 252, step_days: int = 20, top_k: int = 10) -> list[FoldResult]:
    """Evaluate next-session forecasts with expanding, leakage-safe training windows."""
    dates = sorted(pd.to_datetime(raw["date"]).dt.normalize().unique())
    if len(dates) <= min_train_days + 1:
        raise ValueError("Not enough dates for walk-forward validation")

    results: list[FoldResult] = []
    full_features, _ = build_features(raw)
    normalized = pd.to_datetime(full_features["date"]).dt.normalize()
    raw_dates = pd.to_datetime(raw["date"]).dt.normalize()

    for idx in range(min_train_days, len(dates) - 1, step_days):
        train_end = dates[idx - 1]
        prediction_date = dates[idx]
        train = raw[raw_dates <= train_end].copy()
        pipe = Pipeline(top_k=top_k).train(train)

        day = full_features[normalized == prediction_date].copy()
        day = day.dropna(subset=pipe.features)
        ranked = pipe.ranker.score(day).head(top_k)
        pred = pipe.forecaster.predict(ranked)
        if pred.empty:
            continue

        # A forecast made on D must be scored against the next available session.
        labeled = label_predictions(pred, raw)
        labeled = labeled[labeled["prediction_date"] == prediction_date]
        if labeled.empty:
            continue

        results.append(FoldResult(str(train_end.date()), str(prediction_date.date()), score_labeled(labeled)))

    return results
