from __future__ import annotations


def model_health_summary(drift: dict | None = None, confidence: dict | None = None,
                         feedback: dict | None = None) -> dict:
    """Combine drift, confidence and realized-feedback signals into one health snapshot."""
    drift = drift or {}
    confidence = confidence or {}
    feedback = feedback or {}
    drifted = int(drift.get("drifted_features", 0) or 0)
    features = int(drift.get("features", 0) or 0)
    direction = feedback.get("recent_feedback_direction_accuracy")
    uncertainty = confidence.get("uncertainty_p90")

    alerts = []
    if drifted > 0:
        alerts.append("input_drift")
    if direction is not None:
        try:
            if float(direction) < 0.45:
                alerts.append("low_recent_direction_accuracy")
        except (TypeError, ValueError):
            pass
    if uncertainty is not None:
        try:
            if float(uncertainty) > 0.06:
                alerts.append("high_forecast_uncertainty")
        except (TypeError, ValueError):
            pass

    if alerts:
        status = "degraded"
    elif features == 0 and not feedback and not confidence:
        status = "unknown"
    else:
        status = "healthy"
    return {
        "status": status,
        "alerts": alerts,
        "drifted_features": drifted,
        "features": features,
        "recent_direction_accuracy": direction,
        "uncertainty_p90": uncertainty,
    }
