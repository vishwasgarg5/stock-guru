from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from .evaluation import evaluate
from .pipeline import Pipeline
from .ledger import load_pending
from .retrainer import RetrainingDecision, adaptive_retrain

FEEDBACK_KEY_COLUMNS = ["prediction_date", "symbol", "model_version"]


@dataclass(frozen=True)
class FeedbackCycleResult:
    """Outcome of one idempotent settlement and retraining cycle."""

    settled_rows: int
    feedback_rows: int
    retraining: RetrainingDecision | None


def label_predictions(predictions: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Join each prediction to the next available session for its symbol."""
    p = predictions.copy()
    if "date" in p.columns:
        p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    elif "prediction_date" in p.columns:
        p["prediction_date"] = pd.to_datetime(p["prediction_date"]).dt.normalize()
    else:
        raise ValueError("Predictions must contain date or prediction_date")
    if "symbol" not in p.columns:
        raise ValueError("Predictions must contain symbol")
    m = market.copy()
    required = {"date", "symbol", "open", "high", "low", "close"}
    if not required.issubset(m.columns):
        raise ValueError(f"Market data missing columns: {sorted(required - set(m.columns))}")
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()
    m = m.sort_values(["symbol", "date"])
    actual = m[["date", "symbol", "open", "high", "low", "close"]].copy()
    actual["prediction_date"] = actual.groupby("symbol")["date"].shift(1)
    actual = actual.dropna(subset=["prediction_date"]).rename(columns={
        "open": "actual_open", "high": "actual_high", "low": "actual_low", "close": "actual_close",
    })
    if "date" in p.columns:
        p = p.rename(columns={"date": "prediction_date"})
    labeled = p.merge(actual.drop(columns=["date"]), on=["prediction_date", "symbol"], how="inner")
    if "base_close" not in labeled.columns and "close" in labeled.columns:
        labeled["base_close"] = labeled["close"]
    if "base_close" not in labeled.columns:
        raise ValueError("Predictions must contain base_close or close")
    if "pred_close" not in labeled.columns:
        raise ValueError("Predictions must contain pred_close")
    labeled["prediction_date"] = pd.to_datetime(labeled["prediction_date"]).dt.normalize()
    labeled["actual_return"] = labeled["actual_close"] / labeled["base_close"] - 1.0
    labeled["predicted_return"] = labeled["pred_close"] / labeled["base_close"] - 1.0
    labeled["return_error"] = labeled["actual_return"] - labeled["predicted_return"]
    labeled["direction_correct"] = labeled["actual_return"].ge(0) == labeled["predicted_return"].ge(0)
    return labeled


def score_labeled(labeled: pd.DataFrame) -> dict:
    return evaluate(labeled)


def _feedback_keys(df: pd.DataFrame) -> pd.Series:
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
    if "model_version" not in incoming.columns:
        incoming["model_version"] = "unknown"
    if path.exists():
        existing = pd.read_csv(path)
        if not existing.empty:
            existing["prediction_date"] = pd.to_datetime(existing["prediction_date"]).dt.normalize()
            if "model_version" not in existing.columns:
                existing["model_version"] = "unknown"
            incoming = pd.concat([existing, incoming], ignore_index=True, sort=False)
    incoming = incoming.drop_duplicates(subset=FEEDBACK_KEY_COLUMNS, keep="last")
    incoming = incoming.sort_values(["prediction_date", "symbol"])
    incoming.to_csv(path, index=False)


def settle_prediction_feedback(prediction_store: str, feedback_store: str, market: pd.DataFrame,
                                as_of: str | None = None) -> pd.DataFrame:
    """Label pending ledger rows whose next market session is now available."""
    pending = load_pending(prediction_store, as_of=as_of, settled_store=feedback_store)
    if pending.empty:
        return pending
    labeled = label_predictions(pending, market)
    append_feedback(feedback_store, labeled)
    return labeled


def retrain_candidate(raw: pd.DataFrame, validation_dates: int = 20, top_k: int = 10) -> tuple[Pipeline, dict]:
    """Train on the pre-holdout period and evaluate on untouched dates.

    This function deliberately returns a candidate and metrics only; promotion is
    left to the existing champion/challenger gate so retraining cannot silently
    replace the production model.
    """
    if validation_dates <= 0:
        raise ValueError("validation_dates must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
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


def run_feedback_cycle(
    prediction_store: str,
    feedback_store: str,
    raw: pd.DataFrame,
    market: pd.DataFrame,
    model_dir: str = "artifacts",
    as_of: str | None = None,
    min_feedback_rows: int = 20,
    min_train_days: int = 252,
    step_days: int = 20,
    top_k: int = 10,
    rmse_tolerance: float = 0.0,
    ranking_tolerance: float = 0.0,
    prediction_date: str | None = None,
) -> FeedbackCycleResult:
    """Settle outcomes, then retrain only when accumulated feedback is sufficient.

    Settlement is idempotent through the prediction ledger. Retraining remains
    guarded by walk-forward, regime, ranking, feedback, and artifact-promotion
    gates in ``adaptive_retrain``.
    """
    if min_feedback_rows <= 0:
        raise ValueError("min_feedback_rows must be positive")
    settled = settle_prediction_feedback(prediction_store, feedback_store, market, as_of=as_of)
    feedback_path = Path(feedback_store)
    if not feedback_path.exists():
        return FeedbackCycleResult(len(settled), 0, None)
    feedback = pd.read_csv(feedback_path)
    if len(feedback) < min_feedback_rows:
        return FeedbackCycleResult(len(settled), len(feedback), None)
    decision = adaptive_retrain(
        raw,
        model_dir=model_dir,
        min_train_days=min_train_days,
        step_days=step_days,
        top_k=top_k,
        rmse_tolerance=rmse_tolerance,
        ranking_tolerance=ranking_tolerance,
        prediction_date=prediction_date or as_of,
        feedback_store=feedback_store,
    )
    return FeedbackCycleResult(len(settled), len(feedback), decision)
