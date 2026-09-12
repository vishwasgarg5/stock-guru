from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd

from .model_selection import evaluate_candidate, should_promote
from .pipeline import Pipeline


@dataclass(frozen=True)
class RetrainingDecision:
    accepted: bool
    reason: str
    old_metrics: dict | None
    new_metrics: dict


def decide_retraining(old_metrics: dict | None, new_metrics: dict,
                      rmse_tolerance: float = 0.0,
                      ranking_tolerance: float = 0.0) -> RetrainingDecision:
    accepted = should_promote(old_metrics, new_metrics,
                              rmse_tolerance=rmse_tolerance,
                              ranking_tolerance=ranking_tolerance)
    if old_metrics is None:
        reason = "no incumbent validation metrics; candidate accepted as baseline"
    elif accepted:
        reason = "candidate improves forecasting without regressing stock-selection metrics"
    else:
        reason = "candidate rejected: incumbent remains better or equal on required validation metrics"
    return RetrainingDecision(accepted, reason, old_metrics, new_metrics)


def adaptive_retrain(raw: pd.DataFrame, model_dir: str = "artifacts",
                     min_train_days: int = 252, step_days: int = 20,
                     top_k: int = 10, rmse_tolerance: float = 0.0,
                     ranking_tolerance: float = 0.0) -> RetrainingDecision:
    """Evaluate a candidate with walk-forward validation before promotion."""
    model_path = Path(model_dir)
    metrics_path = model_path / "walk_forward_metrics.json"
    old_metrics = None
    if metrics_path.exists():
        old_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    new_metrics = evaluate_candidate(raw, min_train_days=min_train_days,
                                     step_days=step_days, top_k=top_k)
    decision = decide_retraining(old_metrics, new_metrics,
                                 rmse_tolerance=rmse_tolerance,
                                 ranking_tolerance=ranking_tolerance)
    if decision.accepted:
        candidate = Pipeline(top_k=top_k).train(raw)
        model_path.mkdir(parents=True, exist_ok=True)
        candidate.ranker.save(str(model_path / "ranker.joblib"))
        candidate.forecaster.save(str(model_path / "ohlc.joblib"))
        pd.Series(candidate.features).to_csv(model_path / "features.csv", index=False, header=False)
        metrics_path.write_text(json.dumps(new_metrics, indent=2), encoding="utf-8")
    return decision


def append_labeled_predictions(store: str, predictions: pd.DataFrame) -> None:
    """Append actual-vs-predicted rows after the next session closes."""
    path = Path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    predictions.to_csv(store, mode=mode, header=(mode == "w"), index=False)
