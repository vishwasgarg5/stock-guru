from stock_guru.monitoring import model_health_summary


def test_model_health_flags_drift_and_low_realized_accuracy():
    result = model_health_summary(
        {"features": 10, "drifted_features": 2},
        {"uncertainty_p90": 0.03},
        {"recent_feedback_direction_accuracy": 0.40},
    )
    assert result["status"] == "degraded"
    assert "input_drift" in result["alerts"]
    assert "low_recent_direction_accuracy" in result["alerts"]


def test_model_health_is_unknown_without_observations():
    assert model_health_summary() == {
        "status": "unknown", "alerts": [], "drifted_features": 0,
        "features": 0, "recent_direction_accuracy": None, "uncertainty_p90": None,
    }


def test_model_health_can_be_healthy():
    result = model_health_summary({"features": 5, "drifted_features": 0}, {"uncertainty_p90": 0.02}, {})
    assert result["status"] == "healthy"
    assert result["alerts"] == []
