from __future__ import annotations

from typing import Any

from .production_readiness import evaluate_readiness


def build_readiness_report(*, model_exists: bool, pit_universe_validated: bool,
                           fundamentals_validated: bool, monitoring_configured: bool,
                           tests_green: bool, historical_data_available: bool) -> dict[str, Any]:
    """Produce a conservative production-readiness report without inventing data."""
    gates = evaluate_readiness(
        model_exists=model_exists,
        pit_universe_validated=pit_universe_validated,
        fundamentals_validated=fundamentals_validated,
        monitoring_configured=monitoring_configured,
        tests_green=tests_green,
        historical_data_available=historical_data_available,
    )
    gates["historical_data_available"] = bool(historical_data_available)
    gates["status"] = "ready" if gates["status"] == "ready" else "blocked"
    gates["historical_performance_claims_allowed"] = bool(historical_data_available and pit_universe_validated and fundamentals_validated)
    return gates
