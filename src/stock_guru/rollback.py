from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .artifact_validation import validate_model_artifact


@dataclass(frozen=True)
class RecoveryDecision:
    action: str
    reason: str
    target: str | None


def decide_recovery(*, health_status: str, current_artifact: str | Path | None,
                    last_known_good: str | Path | None) -> dict[str, Any]:
    """Choose a conservative recovery action without mutating production state.

    DEGRADED/UNKNOWN health rolls back only to a separately validated last-known-good
    artifact. HEALTHY keeps the current artifact. Missing or invalid rollback targets
    produce BLOCKED rather than an unsafe promotion or fabricated recovery.
    """
    status = str(health_status).strip().upper()
    if status not in {"HEALTHY", "DEGRADED", "UNKNOWN"}:
        raise ValueError("health_status must be HEALTHY, DEGRADED, or UNKNOWN")

    current = str(current_artifact) if current_artifact is not None else None
    good = str(last_known_good) if last_known_good is not None else None

    if status == "HEALTHY":
        return asdict(RecoveryDecision("KEEP", "current health is HEALTHY", current))

    if not good:
        return asdict(RecoveryDecision("BLOCK", "no last-known-good artifact is available", None))

    try:
        validation = validate_model_artifact(good, require_pit_context=True)
    except (OSError, ValueError) as exc:
        return asdict(RecoveryDecision("BLOCK", f"last-known-good artifact is invalid: {exc}", None))

    return asdict(RecoveryDecision(
        "ROLLBACK",
        f"health is {status}; validated last-known-good artifact {validation['model_version']} selected",
        good,
    ))


def validate_rollback_target(path: str | Path) -> dict[str, Any]:
    """Validate a rollback target before any external deployment system uses it."""
    return validate_model_artifact(path, require_pit_context=True)
