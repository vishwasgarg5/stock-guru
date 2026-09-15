"""Fail-closed audit for certification-grade PIT fundamentals evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from stock_guru.fundamentals_ingest import validate_pit_fundamentals

CERTIFICATION_COLUMNS = {
    "symbol", "reported_date", "available_date", "available_timestamp",
    "source", "source_id",
}


def audit(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        return {"status": "blocked", "reason": f"fundamentals file not found: {source}", "rows": 0}
    raw = pd.read_csv(source)
    missing = sorted(CERTIFICATION_COLUMNS - set(raw.columns))
    if missing:
        return {"status": "blocked", "reason": f"certification fundamentals missing columns: {missing}", "rows": int(len(raw))}
    try:
        clean = validate_pit_fundamentals(raw)
    except (ValueError, TypeError) as exc:
        return {"status": "blocked", "reason": str(exc), "rows": int(len(raw))}
    if clean.empty:
        return {"status": "blocked", "reason": "fundamentals dataset is empty", "rows": 0}
    timestamps = pd.to_datetime(clean["available_timestamp"], errors="coerce", utc=True)
    if timestamps.isna().any():
        return {"status": "blocked", "reason": "invalid availability timestamps", "rows": int(len(clean))}
    return {
        "status": "validated",
        "rows": int(len(clean)),
        "symbols": int(clean["symbol"].nunique()),
        "reported_start": clean["reported_date"].min().date().isoformat(),
        "reported_end": clean["reported_date"].max().date().isoformat(),
        "available_start": clean["available_date"].min().date().isoformat(),
        "available_end": clean["available_date"].max().date().isoformat(),
        "source_ids": sorted(clean["source_id"].astype(str).unique().tolist()),
        "file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "validated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
