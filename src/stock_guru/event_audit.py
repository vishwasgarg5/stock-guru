from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from .event_chain import load_event_rows


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"source_id", "effective_date", "evidence_file", "status"}
    if not rows or not required <= set(rows[0]):
        raise ValueError("Event-chain manifest is missing required columns")
    return rows


def audit_event_chain(manifest_path: str | Path) -> dict:
    manifest = load_manifest(manifest_path)
    root = Path(manifest_path).resolve().parents[1]
    evidence = [root / row["evidence_file"] for row in manifest]
    missing = [str(path) for path in evidence if not path.is_file()]
    if missing:
        return {"status": "BLOCKED", "reason": "missing evidence files", "missing_files": missing}
    rows = load_event_rows(evidence)
    by_source = Counter(row["source_id"] for row in rows)
    manifest_ids = [row["source_id"] for row in manifest]
    duplicate_manifest_ids = sorted({x for x in manifest_ids if manifest_ids.count(x) > 1})
    dates = sorted({row["effective_date"] for row in rows})
    return {
        "status": "PASS",
        "source_count": len(manifest),
        "event_count": len(rows),
        "effective_dates": dates,
        "same_date_event_source_count": sum(1 for date in dates if len({r["source_id"] for r in rows if r["effective_date"] == date}) > 1),
        "events_by_source": dict(sorted(by_source.items())),
        "duplicate_manifest_source_ids": duplicate_manifest_ids,
        "missing_files": [],
    }


def save_event_audit(manifest_path: str | Path, output: str | Path) -> Path:
    import json
    report = audit_event_chain(manifest_path)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
