from __future__ import annotations

from pathlib import Path
import json
import math
import pandas as pd
from .walk_forward import run_walk_forward


def summarize(results) -> dict:
    if not results:
        raise ValueError("No walk-forward validation results")
    keys = sorted({key for result in results for key in result.metrics})
    out = {}
    for key in keys:
        values = [float(result.metrics[key]) for result in results]
        finite = [value for value in values if math.isfinite(value)]
        if finite:
            out[key] = float(sum(finite) / len(finite))
    out["validation_folds"] = len(results)
    return out


def evaluate_candidate(raw: pd.DataFrame, min_train_days: int = 252, step_days: int = 20, top_k: int = 10) -> dict:
    return summarize(run_walk_forward(raw, min_train_days=min_train_days, step_days=step_days, top_k=top_k))


def summarize_feedback(feedback: pd.DataFrame | None, min_rows: int = 20) -> dict | None:
    """Summarize realized live-model performance for promotion gating."""
    if feedback is None or feedback.empty:
        return None
    required = {"direction_correct", "return_error"}
    if not required.issubset(feedback.columns):
        return None
    data = feedback.dropna(subset=["direction_correct", "return_error"])
    if len(data) < min_rows:
        return None
    return {
        "feedback_rows": int(len(data)),
        "feedback_direction_accuracy": float(data["direction_correct"].astype(float).mean()),
        "feedback_return_mae": float(data["return_error"].abs().mean()),
    }


def _metric(metrics: dict, key: str, default: float) -> float:
    value = metrics.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def should_promote(old: dict | None, new: dict, rmse_tolerance: float = 0.0,
                   ranking_tolerance: float = 0.0, feedback: dict | None = None) -> bool:
    """Promote only after validation improvement and live-feedback sanity checks."""
    old_rmse = _metric(old, "pred_close_rmse", float("inf")) if old else float("inf")
    old_direction = _metric(old, "close_direction_accuracy", 0.0) if old else 0.0
    new_rmse = _metric(new, "pred_close_rmse", float("inf"))
    new_direction = _metric(new, "close_direction_accuracy", 0.0)

    required_direction = old_direction
    if feedback is not None:
        required_direction = max(required_direction, _metric(feedback, "feedback_direction_accuracy", 0.0))

    if not (new_rmse < old_rmse - rmse_tolerance and new_direction >= required_direction):
        return False
    if old is None:
        return True

    for key in ["top_k_excess_return", "precision_at_k"]:
        if key in old and key in new:
            if _metric(new, key, float("-inf")) < _metric(old, key, float("-inf")) - ranking_tolerance:
                return False
    return True


def save_metrics(model_dir: str | Path, metrics: dict) -> None:
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "walk_forward_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
