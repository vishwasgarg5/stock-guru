from __future__ import annotations

from pathlib import Path
import json
import math
import pandas as pd
from .walk_forward import run_walk_forward


def _aggregate_regime_metrics(results) -> dict:
    """Aggregate fold regime metrics using forecast-sample weighting."""
    buckets: dict[str, list[dict]] = {}
    for result in results:
        for regime, metrics in (getattr(result, "regime_metrics", None) or {}).items():
            buckets.setdefault(str(regime), []).append(metrics)

    aggregated = {}
    for regime, rows in buckets.items():
        total = sum(int(row.get("samples", 0)) for row in rows)
        if total <= 0:
            continue
        summary = {"samples": total}
        keys = {key for row in rows for key in row if key != "samples"}
        for key in sorted(keys):
            weighted = [
                (float(row[key]), int(row.get("samples", 0)))
                for row in rows
                if key in row
            ]
            denominator = sum(weight for _, weight in weighted)
            if denominator:
                summary[key] = float(sum(value * weight for value, weight in weighted) / denominator)
        aggregated[regime] = summary
    return aggregated


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
    regime_metrics = _aggregate_regime_metrics(results)
    if regime_metrics:
        out["regime_metrics"] = regime_metrics
    return out


def evaluate_candidate(raw: pd.DataFrame, min_train_days: int = 252, step_days: int = 20, top_k: int = 10) -> dict:
    return summarize(run_walk_forward(raw, min_train_days=min_train_days, step_days=step_days, top_k=top_k))


def summarize_feedback(feedback: pd.DataFrame | None, min_rows: int = 20, recent_rows: int = 20) -> dict | None:
    """Summarize cumulative and recent realized live-model performance."""
    if feedback is None or feedback.empty:
        return None
    required = {"direction_correct", "return_error"}
    if not required.issubset(feedback.columns):
        return None
    data = feedback.copy()
    data["prediction_date"] = pd.to_datetime(data["prediction_date"], errors="coerce") if "prediction_date" in data.columns else pd.NaT
    data = data.dropna(subset=["direction_correct", "return_error"])
    if len(data) < min_rows:
        return None
    data = data.sort_values("prediction_date") if "prediction_date" in data.columns else data
    recent = data.tail(max(1, recent_rows))
    result = {
        "feedback_rows": int(len(data)),
        "feedback_direction_accuracy": round(float(data["direction_correct"].astype(float).mean()), 12),
        "feedback_return_mae": round(float(data["return_error"].abs().mean()), 12),
        "recent_feedback_rows": int(len(recent)),
        "recent_feedback_direction_accuracy": round(float(recent["direction_correct"].astype(float).mean()), 12),
        "recent_feedback_return_mae": round(float(recent["return_error"].abs().mean()), 12),
    }
    return result


def _metric(metrics: dict, key: str, default: float) -> float:
    value = metrics.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def should_promote(old: dict | None, new: dict, rmse_tolerance: float = 0.0,
                   ranking_tolerance: float = 0.0, feedback: dict | None = None,
                   min_validation_folds: int = 20, min_regime_samples: int = 10,
                   min_adverse_regime_direction: float = 0.45) -> bool:
    """Promote only after robust OOS improvement and live/regime sanity checks.

    Legacy metric dictionaries without validation_folds/regime_metrics remain
    supported for compatibility tests and older persisted artifacts. New
    walk-forward summaries are held to minimum validation coverage and an
    adverse-regime directional-accuracy floor when enough samples exist.
    """
    if "validation_folds" in new and _metric(new, "validation_folds", 0.0) < min_validation_folds:
        return False

    regime_metrics = new.get("regime_metrics") or {}
    for regime in ("bear", "high_vol_bear"):
        metrics = regime_metrics.get(regime) or {}
        samples = _metric(metrics, "samples", 0.0)
        if samples >= min_regime_samples:
            direction = _metric(metrics, "close_direction_accuracy", float("nan"))
            if not math.isfinite(direction) or direction < min_adverse_regime_direction:
                return False

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
