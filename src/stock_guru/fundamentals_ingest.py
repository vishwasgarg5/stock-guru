from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {"symbol", "reported_date", "available_date"}


def validate_pit_fundamentals(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the canonical PIT fundamentals contract."""
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
    if out.duplicated(["symbol", "available_date"]).any():
        raise ValueError("Duplicate symbol/available_date PIT observations")
    if {"source", "source_id"}.intersection(out.columns):
        if not {"source", "source_id"}.issubset(out.columns):
            raise ValueError("Fundamentals provenance requires both source and source_id")
        if out["source"].astype(str).str.strip().eq("").any() or out["source_id"].astype(str).str.strip().eq("").any():
            raise ValueError("Fundamentals provenance fields cannot be blank")

    value_columns = [c for c in out.columns if c not in REQUIRED]
    for column in value_columns:
        if column in {"source", "source_id"}:
            continue
        converted = pd.to_numeric(out[column], errors="coerce")
        invalid = out[column].notna() & (converted.isna() | ~np.isfinite(converted))
        if invalid.any():
            raise ValueError(f"PIT fundamental column {column!r} contains nonnumeric or nonfinite values")
        out[column] = converted
    return out.sort_values(["symbol", "available_date"]).reset_index(drop=True)


def ingest_pit_fundamentals(input_path: str | Path, output_path: str | Path) -> Path:
    """Validate and normalize an externally collected filing-derived dataset."""
    raw = pd.read_csv(input_path)
    clean = validate_pit_fundamentals(raw)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output, index=False)
    return output
