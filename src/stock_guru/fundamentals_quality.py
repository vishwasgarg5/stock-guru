from __future__ import annotations

from pathlib import Path
import hashlib
import json
import pandas as pd

from .fundamentals_ingest import validate_pit_fundamentals

REQUIRED_MANIFEST = {"dataset", "source_name", "source_url", "retrieved_at", "license_or_terms"}


def validate_fundamentals_manifest(manifest: dict) -> dict:
    missing = REQUIRED_MANIFEST - set(manifest)
    if missing:
        raise ValueError(f"Missing fundamentals manifest fields: {sorted(missing)}")
    if manifest["dataset"] != "pit_fundamentals":
        raise ValueError("Fundamentals manifest dataset must be pit_fundamentals")
    for key in REQUIRED_MANIFEST:
        if not str(manifest[key]).strip():
            raise ValueError(f"Fundamentals manifest field {key!r} must be nonblank")
    return dict(manifest)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_pit_fundamentals(path: str | Path, manifest: dict) -> dict:
    manifest = validate_fundamentals_manifest(manifest)
    frame = pd.read_csv(path)
    clean = validate_pit_fundamentals(frame)
    return {
        "dataset": "pit_fundamentals",
        "source_name": manifest["source_name"],
        "source_url": manifest["source_url"],
        "retrieved_at": manifest["retrieved_at"],
        "rows": int(len(clean)),
        "unique_symbols": int(clean["symbol"].nunique()),
        "min_reported_date": clean["reported_date"].min().date().isoformat(),
        "max_reported_date": clean["reported_date"].max().date().isoformat(),
        "min_available_date": clean["available_date"].min().date().isoformat(),
        "max_available_date": clean["available_date"].max().date().isoformat(),
        "sha256": file_sha256(path),
        "provenance_validated": True,
    }


def save_audit_report(report: dict, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
