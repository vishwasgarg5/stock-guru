from stock_guru.readiness_report import build_readiness_report


def test_readiness_requires_real_historical_data():
    report = build_readiness_report(model_exists=True, pit_universe_validated=True, fundamentals_validated=True, monitoring_configured=True, tests_green=True, historical_data_available=False)
    assert report["status"] == "blocked"
    assert report["historical_performance_claims_allowed"] is False


def test_readiness_can_be_ready_when_all_gates_pass():
    report = build_readiness_report(model_exists=True, pit_universe_validated=True, fundamentals_validated=True, monitoring_configured=True, tests_green=True, historical_data_available=True)
    assert report["status"] == "ready"
