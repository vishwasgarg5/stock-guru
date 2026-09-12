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


def evaluate_candidate(
    raw: pd.DataFrame,
    min_train_days: int = 252,
    step_days: int = 20,
    top_k: int = 10,
) -> dict:
    return summarize(
        run_walk_forward(
            raw,
            min_train_days=min_train_days,
            step_days=step_days,
            top_k=top_k,
        )
    )


def _metric(metrics: dict, key: str, default: float) -> float:
    value = metrics.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def should_promote(
    old: dict | None,
    new: dict,
    rmse_tolerance: float = 0.0,
    ranking_tolerance: float = 0.0,
) -> bool:
    """Promote only when the candidate strictly improves forecasting and does not regress.

    Missing ranking metrics are ignored for backward compatibility with older
    metric files. Once ranking metrics exist, the candidate must not regress
    on top-K excess return or positive-return precision.
    """
    if old is None:
        return True

    old_rmse = _metric(old, "pred_close_rmse", float("inf"))
    new_rmse = _metric(new, "pred_close_rmse", float("inf"))
    old_direction = _metric(old, "close_direction_accuracy", 0.0)
    new_direction = _metric(new, "close_direction_accuracy", 0.0)

    # Equality is not an improvement. A positive tolerance additionally
    # requires the candidate to beat the incumbent by more than that margin.
    forecast_ok = new_rmse < old_rmse - rmse_tolerance and new_direction >= old_direction
    if not forecast_ok:
        return False

    ranking_keys = ["top_k_excess_return", "precision_at_k"]
    for key in ranking_keys:
        if key in old and key in new:
            old_value = _metric(old, key, float("-inf"))
            new_value = _metric(new, key, float("-inf"))
            if new_value < old_value - ranking_tolerance:
                return False

    return True


def save_metrics(model_dir: str | Path, metrics: dict) -> None:
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "walk_forward_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
