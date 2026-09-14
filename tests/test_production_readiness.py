from stock_guru.production_readiness import evaluate_readiness


def test_readiness_blocks_without_pit_evidence():
    out = evaluate_readiness(model_exists=True, pit_universe_validated=False,
                             fundamentals_validated=False, monitoring_configured=True,
                             tests_green=True)
    assert out["status"] == "blocked"
    assert any(not gate["passed"] for gate in out["gates"])


def test_readiness_passes_when_all_gates_are_green():
    out = evaluate_readiness(model_exists=True, pit_universe_validated=True,
                             fundamentals_validated=True, monitoring_configured=True,
                             tests_green=True)
    assert out["status"] == "ready"
