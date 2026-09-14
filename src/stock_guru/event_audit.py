from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from .event_chain import load_event_rows


REQUIRED_COLUMNS = {"source_id", "effective_date", "evidence_file", "status"}
VALID_STATUSES = {"verified", "correction-preserved-in-file"}


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not REQUIRED_COLUMNS <= set(rows[0]):
        raise ValueError("Event-chain manifest is missing required columns")
    return rows


def audit_event_chain(manifest_path: str | Path) -> dict:
    manifest = load_manifest(manifest_path)
    root = Path(manifest_path).resolve().parents[1]
    ids = [row["source_id"] for row in manifest]
    duplicate_manifest_ids = sorted({x for x in ids if ids.count(x) > 1})
    invalid_statuses = sorted({row["status"] for row in manifest if row["status"] not in VALID_STATUSES})
    evidence = [root / row["evidence_file"] for row in manifest]
    missing = [str(path) for path in evidence if not path.is_file()]
    if duplicate_manifest_ids or invalid_statuses or missing:
        return {"status": "BLOCKED", "reason": "manifest validation failed", "duplicate_manifest_source_ids": duplicate_manifest_ids, "invalid_statuses": invalid_statuses, "missing_files": missing}
    rows = load_event_rows(evidence)
    manifest_effective_dates = {row["source_id"]: row["effective_date"] for row in manifest}
    observed_effective_dates = {source_id: sorted({row["effective_date"] for row in rows if row["source_id"] == source_id}) for source_id in ids}
    date_mismatches = {source_id: {"manifest": manifest_effective_dates[source_id], "observed": observed_effective_dates[source_id]} for source_id in ids if observed_effective_dates[source_id] != [manifest_effective_dates[source_id]]}
    unknown_sources = sorted(set(row["source_id"] for row in rows) - set(ids))
    if date_mismatches or unknown_sources:
        return {"status": "BLOCKED", "reason": "event provenance reconciliation failed", "date_mismatches": date_mismatches, "unknown_sources": unknown_sources}
    by_source = Counter(row["source_id"] for row in rows)
    dates = sorted({row["effective_date"] for row in rows})
    return {
        "status": "PASS",
        "source_count": len(manifest),
        "event_count": len(rows),
        "effective_dates": dates,
        "same_date_event_source_count": sum(1 for date in dates if len({r["source_id"] for r in rows if r["effective_date"] == date}) > 1),
        "events_by_source": dict(sorted(by_source.items())),
        "duplicate_manifest_source_ids": [],
        "invalid_statuses": [],
        "date_mismatches": {},
        "unknown_sources": [],
        "missing_files": [],
    }


def save_event_audit(manifest_path: str | Path, output: str | Path) -> Path:
    import json
    report = audit_event_chain(manifest_path)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
