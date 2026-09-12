from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
from .walk_forward import run_walk_forward


def summarize(results) -> dict:
    if not results:
        raise ValueError("No walk-forward validation results")
    keys = results[0].metrics.keys()
    out = {k: float(sum(r.metrics[k] for r in results) / len(results)) for k in keys}
    out["validation_folds"] = len(results)
    return out


def evaluate_candidate(raw: pd.DataFrame, min_train_days: int = 252, step_days: int = 20, top_k: int = 10) -> dict:
    return summarize(run_walk_forward(raw, min_train_days=min_train_days, step_days=step_days, top_k=top_k))


def should_promote(old: dict | None, new: dict) -> bool:
    if old is None:
        return True
    return float(new.get("pred_close_rmse", float("inf"))) <= float(old.get("pred_close_rmse", float("inf"))) and float(new.get("close_direction_accuracy", 0.0)) >= float(old.get("close_direction_accuracy", 0.0))


def save_metrics(model_dir: str | Path, metrics: dict) -> None:
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "walk_forward_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
