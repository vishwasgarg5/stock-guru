from __future__ import annotations

from pathlib import Path
from typing import Any


def run_production_smoke(*, model_dir: str | Path | None = None,
                         event_manifest: str | Path | None = None) -> dict[str, Any]:
    """Run lightweight production-path safety checks without fabricating data.

    The smoke test verifies imports, the readiness gate's fail-closed behavior,
    optional artifact integrity, and optional event-chain auditability. It never
    turns missing historical evidence into a passing production decision.
    """
    checks: list[dict[str, Any]] = []

    from .final_readiness import build_final_readiness_report

    checks.append({"name": "package_imports", "passed": True})

    blocked = build_final_readiness_report(
        model_exists=True,
        pit_universe_validated=True,
        fundamentals_validated=True,
        monitoring_configured=True,
        tests_green=True,
        historical_data_available=False,
        historical_data_verified=False,
        backtest_completed=False,
        walk_forward_completed=False,
        paper_trading_validated=False,
        feedback_cycle_validated=False,
        artifact_reproducible=False,
    )
    checks.append({
        "name": "readiness_fails_closed",
        "passed": blocked["status"] == "blocked",
        "status": blocked["status"],
    })

    if model_dir is not None:
        from .artifact_validation import validate_model_artifact
        try:
            artifact = validate_model_artifact(model_dir)
            checks.append({"name": "model_artifact", "passed": True, "model_version": artifact["model_version"]})
        except (OSError, ValueError) as exc:
            checks.append({"name": "model_artifact", "passed": False, "reason": str(exc)})

    if event_manifest is not None:
        from .event_audit import audit_event_chain
        try:
            audit = audit_event_chain(event_manifest)
            checks.append({"name": "event_chain_audit", "passed": audit["status"] == "PASS", "status": audit["status"]})
        except (OSError, ValueError) as exc:
            checks.append({"name": "event_chain_audit", "passed": False, "reason": str(exc)})

    passed = all(check["passed"] for check in checks)
    return {"status": "PASS" if passed else "BLOCKED", "checks": checks}
