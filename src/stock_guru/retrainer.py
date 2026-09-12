from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from .evaluation import evaluate

@dataclass
class RetrainingDecision:
    accepted: bool
    reason: str
    old_rmse: float
    new_rmse: float


def should_accept(old_metrics: dict, new_metrics: dict, key: str = "pred_close_rmse", tolerance: float = 0.0) -> RetrainingDecision:
    old_rmse = float(old_metrics[key]); new_rmse = float(new_metrics[key])
    accepted = new_rmse < old_rmse - tolerance
    return RetrainingDecision(accepted, "new model improves validation RMSE" if accepted else "new model rejected: no validation improvement", old_rmse, new_rmse)


def append_labeled_predictions(store: str, predictions: pd.DataFrame) -> None:
    """Append actual-vs-predicted rows after the next session closes."""
    mode = "a" if __import__("pathlib").Path(store).exists() else "w"
    predictions.to_csv(store, mode=mode, header=(mode == "w"), index=False)
