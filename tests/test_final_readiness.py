from stock_guru.final_readiness import build_final_readiness_report


def _kwargs():
    return dict(
        model_exists=True,
        pit_universe_validated=True,
        fundamentals_validated=True,
        monitoring_configured=True,
        tests_green=True,
        historical_data_available=True,
        historical_data_verified=True,
        backtest_completed=True,
        walk_forward_completed=True,
        paper_trading_validated=True,
        feedback_cycle_validated=True,
        artifact_reproducible=True,
    )


def test_final_readiness_requires_real_evidence():
    args = _kwargs()
    args["historical_data_verified"] = False
    out = build_final_readiness_report(**args)
    assert out["status"] == "blocked"
    assert "historical_data_verified" in out["blockers"]


def test_final_readiness_ready_when_all_evidence_exists():
    out = build_final_readiness_report(**_kwargs())
    assert out["status"] == "ready"
    assert out["blockers"] == []
