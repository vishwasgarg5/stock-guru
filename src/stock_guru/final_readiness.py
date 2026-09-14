from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .production_readiness import evaluate_readiness


REPORT_SCHEMA_VERSION = "1.0"


def build_final_readiness_report(
    *,
    model_exists: bool,
    pit_universe_validated: bool,
    fundamentals_validated: bool,
    monitoring_configured: bool,
    tests_green: bool,
    historical_data_available: bool,
    historical_data_verified: bool,
    backtest_completed: bool,
    walk_forward_completed: bool,
    paper_trading_validated: bool,
    feedback_cycle_validated: bool,
    artifact_reproducible: bool,
) -> dict[str, Any]:
    """Build a conservative, auditable final-readiness/certification report.

    This function deliberately cannot manufacture evidence. A production-ready
    result requires both the baseline gates and real historical-data/evaluation
    evidence. Missing evidence produces ``blocked``.
    """
    baseline = evaluate_readiness(
        model_exists=model_exists,
        pit_universe_validated=pit_universe_validated,
        fundamentals_validated=fundamentals_validated,
        monitoring_configured=monitoring_configured,
        tests_green=tests_green,
        historical_data_available=historical_data_available,
    )
    evidence = {
        "historical_data_verified": bool(historical_data_verified),
        "backtest_completed": bool(backtest_completed),
        "walk_forward_completed": bool(walk_forward_completed),
        "paper_trading_validated": bool(paper_trading_validated),
        "feedback_cycle_validated": bool(feedback_cycle_validated),
        "artifact_reproducible": bool(artifact_reproducible),
    }
    evidence_ok = all(evidence.values())
    status = "ready" if baseline["status"] == "ready" and evidence_ok else "blocked"
    blockers = [gate["reason"] for gate in baseline["gates"] if not gate["passed"]]
    blockers.extend(name for name, passed in evidence.items() if not passed)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "certification": "READY" if status == "ready" else "BLOCKED",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline,
        "evidence": evidence,
        "blockers": blockers,
        "decision": (
            "Production readiness demonstrated by validated PIT data and completed evaluation evidence."
            if status == "ready"
            else "Production readiness blocked: real PIT data and/or required evaluation evidence is missing."
        ),
    }
