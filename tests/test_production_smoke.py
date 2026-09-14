from stock_guru.production_smoke import run_production_smoke


def test_production_smoke_passes_fail_closed_checks():
    report = run_production_smoke()
    assert report["status"] == "PASS"
    assert {check["name"] for check in report["checks"]} == {"package_imports", "readiness_fails_closed"}


def test_production_smoke_does_not_claim_ready_without_evidence():
    report = run_production_smoke()
    readiness = next(check for check in report["checks"] if check["name"] == "readiness_fails_closed")
    assert readiness["status"] == "blocked"
