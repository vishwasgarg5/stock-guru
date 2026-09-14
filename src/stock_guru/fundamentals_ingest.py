from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED = {"symbol", "reported_date", "available_date"}


def validate_pit_fundamentals(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the canonical PIT fundamentals contract.

    ``available_date`` is the first market date on which the value may be used;
    it must never precede the report date. Duplicate symbol/available_date rows
    are rejected so as-of joins remain deterministic.
    """
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing PIT fundamental columns: {sorted(missing)}")
    out = frame.copy()
    for col in ("reported_date", "available_date"):
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.strip()
    if out["symbol"].eq("").any() or out[["reported_date", "available_date"]].isna().any().any():
        raise ValueError("PIT fundamentals contain invalid symbols or dates")
    if (out["available_date"] < out["reported_date"]).any():
        raise ValueError("available_date cannot precede reported_date")
    if out.duplicated(["symbol", "available_date"]).any():
        raise ValueError("Duplicate symbol/available_date PIT observations")
    return out.sort_values(["symbol", "available_date"]).reset_index(drop=True)


def ingest_pit_fundamentals(input_path: str | Path, output_path: str | Path) -> Path:
    """Validate and normalize an externally collected filing-derived dataset."""
    raw = pd.read_csv(input_path)
    clean = validate_pit_fundamentals(raw)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output, index=False)
    return output
