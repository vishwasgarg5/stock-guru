from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ReadinessGate:
    name: str
    passed: bool
    reason: str


def evaluate_readiness(*, model_exists: bool, pit_universe_validated: bool,
                       fundamentals_validated: bool, monitoring_configured: bool,
                       tests_green: bool, historical_data_available: bool = False) -> dict[str, Any]:
    gates = [
        ReadinessGate("model_artifact", bool(model_exists), "model artifact available" if model_exists else "model artifact missing"),
        ReadinessGate("pit_universe", bool(pit_universe_validated), "PIT universe evidence validated" if pit_universe_validated else "PIT universe evidence not validated"),
        ReadinessGate("pit_fundamentals", bool(fundamentals_validated), "PIT fundamentals validated" if fundamentals_validated else "PIT fundamentals not validated"),
        ReadinessGate("monitoring", bool(monitoring_configured), "monitoring configured" if monitoring_configured else "monitoring not configured"),
        ReadinessGate("ci", bool(tests_green), "CI/tests green" if tests_green else "CI/tests not green"),
        ReadinessGate("historical_data", bool(historical_data_available), "verified historical data available" if historical_data_available else "verified historical data unavailable"),
    ]
    return {"status": "ready" if all(g.passed for g in gates) else "blocked", "gates": [asdict(g) for g in gates]}
