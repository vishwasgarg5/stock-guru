from __future__ import annotations

from pathlib import Path
import pandas as pd
from .evaluation import attach_actuals, evaluate
from .pipeline import Pipeline

PREDICTION_COLUMNS = [
    "prediction_date", "symbol", "rank", "model_version",
    "pred_open", "pred_high", "pred_low", "pred_close", "base_close",
]


def label_predictions(predictions: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Join a prediction made for date D to the actual OHLC observed on D+1."""
    p = predictions.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    m = market.copy()
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()
    # The model's prediction date is D; actuals are the next available market row per symbol.
    m = m.sort_values(["symbol", "date"])
    actual = m[["date", "symbol", "open", "high", "low", "close"]].copy()
    actual["prediction_date"] = actual.groupby("symbol")["date"].shift(1)
    actual = actual.dropna(subset=["prediction_date"])
    actual = actual.rename(columns={
        "open": "actual_open", "high": "actual_high",
        "low": "actual_low", "close": "actual_close",
    })
    out = p.merge(actual.drop(columns=["date"]), on=["prediction_date", "symbol"], how="inner")
    return out


def score_labeled(labeled: pd.DataFrame) -> dict:
    return evaluate(labeled.rename(columns={"base_close": "base_close"}))


def append_feedback(store: str, labeled: pd.DataFrame) -> None:
    path = Path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(path, mode="a" if path.exists() else "w", header=not path.exists(), index=False)


def retrain_candidate(raw: pd.DataFrame, validation_dates: int = 20, top_k: int = 10) -> tuple[Pipeline, dict]:
    """Train a candidate while leaving the final validation window untouched."""
    dates = sorted(pd.to_datetime(raw["date"]).dt.normalize().unique())
    if len(dates) <= validation_dates + 252:
        raise ValueError("Need more history before adaptive retraining")
    train_dates = dates[:-validation_dates]
    valid_dates = dates[-validation_dates:]
    train = raw[pd.to_datetime(raw["date"]).dt.normalize().isin(train_dates)].copy()
    valid = raw[pd.to_datetime(raw["date"]).dt.normalize().isin(valid_dates)].copy()
    pipe = Pipeline(top_k=top_k).train(train)
    predictions = []
    for day in valid_dates:
        pred = pipe.predict_date(train, str(day.date())) if day in train_dates else None
        if pred is not None and not pred.empty:
            predictions.append(pred)
    if not predictions:
        # Validate using a compact holdout prediction generated from all pre-validation history.
        day = str(valid_dates[0].date())
        data_pipe = pipe.predict_date(pd.concat([train, valid.head(0)]), day)
        predictions = [data_pipe]
    labeled = label_predictions(pd.concat(predictions, ignore_index=True), raw)
    return pipe, score_labeled(labeled) if not labeled.empty else {}
