from __future__ import annotations

from pathlib import Path
import pandas as pd
from .evaluation import evaluate
from .pipeline import Pipeline


FEEDBACK_KEY_COLUMNS = ["prediction_date", "symbol", "model_version"]


def label_predictions(predictions: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Join a prediction made for date D to the next available OHLC for that symbol."""
    p = predictions.copy()
    if "date" in p.columns:
        p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    elif "prediction_date" in p.columns:
        p["prediction_date"] = pd.to_datetime(p["prediction_date"]).dt.normalize()
    else:
        raise ValueError("Predictions must contain date or prediction_date")

    m = market.copy()
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()
    m = m.sort_values(["symbol", "date"])
    actual = m[["date", "symbol", "open", "high", "low", "close"]].copy()
    actual["prediction_date"] = actual.groupby("symbol")["date"].shift(1)
    actual = actual.dropna(subset=["prediction_date"]).rename(columns={
        "open": "actual_open", "high": "actual_high",
        "low": "actual_low", "close": "actual_close",
    })

    if "date" in p.columns:
        p = p.rename(columns={"date": "prediction_date"})
    labeled = p.merge(
        actual.drop(columns=["date"]),
        on=["prediction_date", "symbol"], how="inner"
    )
    if "base_close" not in labeled.columns and "close" in labeled.columns:
        labeled["base_close"] = labeled["close"]
    labeled["prediction_date"] = pd.to_datetime(labeled["prediction_date"]).dt.normalize()
    labeled["actual_return"] = labeled["actual_close"] / labeled["base_close"] - 1.0
    labeled["predicted_return"] = labeled["pred_close"] / labeled["base_close"] - 1.0
    labeled["return_error"] = labeled["actual_return"] - labeled["predicted_return"]
    labeled["direction_correct"] = (
        labeled["actual_return"].ge(0) == labeled["predicted_return"].ge(0)
    )
    return labeled


def score_labeled(labeled: pd.DataFrame) -> dict:
    return evaluate(labeled)


def _feedback_keys(df: pd.DataFrame) -> pd.Series:
    """Build a stable deduplication key for prediction feedback rows."""
    key = pd.to_datetime(df["prediction_date"]).dt.normalize().astype(str) + "|" + df["symbol"].astype(str)
    if "model_version" in df.columns:
        key = key + "|" + df["model_version"].fillna("").astype(str)
    return key


def append_feedback(store: str, labeled: pd.DataFrame) -> None:
    """Append only new prediction outcomes, deduplicated by date/symbol/model."""
    if labeled.empty:
        return

    path = Path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = labeled.copy()
    incoming["prediction_date"] = pd.to_datetime(incoming["prediction_date"]).dt.normalize()

    if path.exists():
        existing = pd.read_csv(path)
        if not existing.empty:
            existing["prediction_date"] = pd.to_datetime(existing["prediction_date"]).dt.normalize()
            combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
        else:
            combined = incoming
    else:
        combined = incoming

    combined = combined.drop_duplicates(subset=FEEDBACK_KEY_COLUMNS if all(
        col in combined.columns for col in FEEDBACK_KEY_COLUMNS
    ) else ["prediction_date", "symbol"], keep="last")
    combined = combined.sort_values(["prediction_date", "symbol"])
    combined.to_csv(path, index=False)


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
