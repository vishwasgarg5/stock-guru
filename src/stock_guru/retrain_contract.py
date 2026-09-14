from __future__ import annotations


def validate_retrain_gate(*, feedback_rows: int, min_feedback_rows: int, candidate_metrics: dict, champion_metrics: dict | None = None) -> dict:
    if min_feedback_rows <= 0:
        raise ValueError("min_feedback_rows must be positive")
    if feedback_rows < min_feedback_rows:
        return {"eligible": False, "reason": "insufficient_feedback"}
    if not candidate_metrics:
        return {"eligible": False, "reason": "missing_candidate_metrics"}
    if champion_metrics is None:
        return {"eligible": True, "reason": "no_champion_metrics"}
    return {"eligible": True, "reason": "metrics_available"}
