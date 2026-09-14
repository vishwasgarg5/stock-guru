from __future__ import annotations

from typing import Any


def validate_monitoring_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Validate the minimum production monitoring contract."""
    required = {"status", "model_version", "timestamp"}
    missing = required - set(snapshot)
    if missing:
        raise ValueError(f"Monitoring snapshot missing fields: {sorted(missing)}")
    if snapshot["status"] not in {"HEALTHY", "DEGRADED", "UNKNOWN"}:
        raise ValueError("Monitoring status must be HEALTHY, DEGRADED, or UNKNOWN")
    if not str(snapshot["model_version"]).strip():
        raise ValueError("Monitoring model_version must be nonblank")
    if not str(snapshot["timestamp"]).strip():
        raise ValueError("Monitoring timestamp must be nonblank")
    return dict(snapshot)
