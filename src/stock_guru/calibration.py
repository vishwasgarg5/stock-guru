from __future__ import annotations

import math
import pandas as pd


def confidence_calibration_table(predictions: pd.DataFrame, bins: list[float] | None = None) -> dict:
    """Measure empirical directional accuracy by forecast-confidence bucket."""
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        return {}
    required = {"forecast_confidence", "actual_close", "pred_close", "base_close"}
    if not required.issubset(predictions.columns):
        return {}
    frame = predictions[list(required)].copy()
    for col in ["forecast_confidence", "actual_close", "pred_close", "base_close"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna()
    frame = frame[frame["base_close"] != 0]
    if frame.empty:
        return {}
    frame["direction_correct"] = (
        (frame["actual_close"] - frame["base_close"]) * (frame["pred_close"] - frame["base_close"]) >= 0
    )
    edges = bins or [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001]
    if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("confidence bins must be strictly increasing")
    frame["bucket"] = pd.cut(frame["forecast_confidence"], bins=edges, right=False, include_lowest=True)
    result = {}
    for bucket, group in frame.groupby("bucket", observed=True):
        if group.empty:
            continue
        result[str(bucket)] = {
            "samples": int(len(group)),
            "mean_confidence": float(group["forecast_confidence"].mean()),
            "observed_direction_accuracy": float(group["direction_correct"].mean()),
        }
    return result


def uncertainty_scale_for_coverage(predictions: pd.DataFrame, target_coverage: float = 0.80) -> float:
    """Return an empirical multiplicative scale for uncertainty intervals."""
    if not 0 < target_coverage < 1:
        raise ValueError("target_coverage must be between 0 and 1")
    required = {"actual_close", "pred_close", "base_close", "pred_close_uncertainty_pct"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Missing calibration columns: {sorted(required - set(predictions.columns))}")
    frame = predictions[list(required)].apply(pd.to_numeric, errors="coerce").dropna()
    frame = frame[(frame["base_close"] != 0) & (frame["pred_close_uncertainty_pct"] > 0)]
    if frame.empty:
        raise ValueError("No valid observations for interval calibration")
    error_pct = (frame["actual_close"] - frame["pred_close"]).abs() / frame["base_close"].abs()
    normalized = error_pct / frame["pred_close_uncertainty_pct"]
    scale = float(normalized.quantile(target_coverage))
    return max(scale, math.ulp(1.0))


def apply_uncertainty_scale(predictions: pd.DataFrame, scale: float) -> pd.DataFrame:
    if not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError("scale must be a positive finite number")
    out = predictions.copy()
    if "pred_close_uncertainty_pct" not in out:
        raise ValueError("pred_close_uncertainty_pct is required")
    out["pred_close_uncertainty_pct"] = pd.to_numeric(out["pred_close_uncertainty_pct"], errors="coerce") * float(scale)
    return out
