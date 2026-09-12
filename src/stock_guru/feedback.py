from __future__ import annotations

from pathlib import Path
import pandas as pd
from .evaluation import evaluate
from .pipeline import Pipeline


def label_predictions(predictions: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Join a prediction made for date D to the next available OHLC for that symbol."""
    p = predictions.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    m = market.copy()
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()
    m = m.sort_values(["symbol", "date"])
    actual = m[["date", "symbol", "open", "high", "low", "close"]].copy()
    actual["prediction_date"] = actual.groupby("symbol")["date"].shift(1)
    actual = actual.dropna(subset=["prediction_date"]).rename(columns={
        "open": "actual_open", "high": "actual_high",
        "low": "actual_low", "close": "actual_close",
    })
    return p.rename(columns={"date": "prediction_date"}).merge(
        actual.drop(columns=["date"]), on=["prediction_date", "symbol"], how="inner"
    )


def score_labeled(labeled: pd.DataFrame) -> dict:
    return evaluate(labeled)


def append_feedback(store: str, labeled: pd.DataFrame) -> None:
    path = Path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(path, mode="a" if path.exists() else "w", header=not path.exists(), index=False)


def retrain_candidate(raw: pd.DataFrame, validation_dates: int = 20, top_k: int = 10) -> tuple[Pipeline, dict]:
    """Train on the pre-holdout period and evaluate predictions on untouched dates."""
    dates = sorted(pd.to_datetime(raw["date"]).dt.normalize().unique())
    if len(dates) <= validation_dates + 252:
        raise ValueError("Need more history before adaptive retraining")
    train_dates = dates[:-validation_dates]
    valid_dates = dates[-validation_dates:]
    train = raw[pd.to_datetime(raw["date"]).dt.normalize().isin(train_dates)].copy()
    pipe = Pipeline(top_k=top_k).train(train)
    predictions = [pipe.predict_date(raw, str(day.date())) for day in valid_dates]
    predictions = [p for p in predictions if not p.empty]
    if not predictions:
        return pipe, {}
    labeled = label_predictions(pd.concat(predictions, ignore_index=True), raw)
    return pipe, score_labeled(labeled) if not labeled.empty else {}
