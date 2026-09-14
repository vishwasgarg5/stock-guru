from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

REQUIRED = {"symbol", "reported_date", "available_date"}

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def validate_pit_fundamentals(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the canonical point-in-time fundamentals contract.

    Certification-grade rows may carry publication timestamps, immutable source
    identifiers and explicit filing versions.  The validator preserves those
    fields and refuses ambiguous provenance/version collisions rather than
    silently collapsing them.
    """
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing PIT fundamental columns: {sorted(missing)}")
    out = frame.copy()
    for col in ("reported_date", "available_date"):
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    if out["symbol"].eq("").any() or out[["reported_date", "available_date"]].isna().any().any():
        raise ValueError("PIT fundamentals contain invalid symbols or dates")
    if (out["available_date"] < out["reported_date"]).any():
        raise ValueError("available_date cannot precede reported_date")

    if "available_timestamp" in out.columns:
        ts = pd.to_datetime(out["available_timestamp"], errors="coerce", utc=True)
        if ts.isna().any():
            raise ValueError("available_timestamp contains invalid timestamps")
        out["available_timestamp"] = ts
        if (out["available_timestamp"].dt.normalize().dt.tz_localize(None) < out["available_date"]).any():
            raise ValueError("available_timestamp cannot precede available_date")

    provenance_columns = {"source", "source_id"}
    if provenance_columns.intersection(out.columns):
        if not provenance_columns.issubset(out.columns):
            raise ValueError("Fundamentals provenance requires both source and source_id")
        if out["source"].astype(str).str.strip().eq("").any() or out["source_id"].astype(str).str.strip().eq("").any():
            raise ValueError("Fundamentals provenance fields cannot be blank")

    if "source_url" in out.columns and out["source_url"].astype(str).str.strip().eq("").any():
        raise ValueError("source_url cannot be blank when supplied")
    if "source_sha256" in out.columns:
        hashes = out["source_sha256"].astype(str).str.strip()
        if hashes.eq("").any() or ~hashes.str.match(_SHA256).all():
            raise ValueError("source_sha256 must be a 64-character SHA-256 when supplied")

    if "version" in out.columns:
        if out["version"].astype(str).str.strip().eq("").any():
            raise ValueError("fundamental version cannot be blank when supplied")
        key = ["symbol", "reported_date", "available_date", "source_id" if "source_id" in out.columns else "symbol", "version"]
    elif "source_id" in out.columns:
        key = ["symbol", "reported_date", "available_date", "source_id"]
    else:
        key = ["symbol", "available_date"]
    if out.duplicated(key).any():
        raise ValueError(f"Duplicate PIT observations for key {key}")

    value_exclusions = REQUIRED | {"source", "source_id", "source_url", "source_sha256", "available_timestamp", "version", "statement_type", "filing_type", "audited_status", "currency", "units"}
    value_columns = [c for c in out.columns if c not in value_exclusions]
    for column in value_columns:
        converted = pd.to_numeric(out[column], errors="coerce")
        invalid = out[column].notna() & (converted.isna() | ~np.isfinite(converted))
        if invalid.any():
            raise ValueError(f"PIT fundamental column {column!r} contains nonnumeric or nonfinite values")
        out[column] = converted
    return out.sort_values(["symbol", "available_date"] + (["available_timestamp"] if "available_timestamp" in out.columns else [])).reset_index(drop=True)


def ingest_pit_fundamentals(input_path: str | Path, output_path: str | Path) -> Path:
    """Validate and normalize an externally collected filing-derived dataset."""
    raw = pd.read_csv(input_path)
    clean = validate_pit_fundamentals(raw)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output, index=False)
    return output
