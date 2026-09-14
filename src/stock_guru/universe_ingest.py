from __future__ import annotations

from pathlib import Path

import pandas as pd

SNAPSHOT_REQUIRED = {"as_of", "symbol"}
EVENT_REQUIRED = {"effective_date", "symbol", "action", "source", "source_id"}


def validate_universe_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate dated universe membership observations."""
    missing = SNAPSHOT_REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing snapshot columns: {sorted(missing)}")

    out = frame.copy()
    out["as_of"] = pd.to_datetime(out["as_of"], errors="coerce").dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    if out["as_of"].isna().any():
        raise ValueError("Snapshots contain invalid dates")
    if out["symbol"].eq("").any():
        raise ValueError("Snapshots contain blank symbols")
    if out.duplicated(["as_of", "symbol"]).any():
        raise ValueError("Snapshots contain duplicate as_of/symbol rows")
    return out.sort_values(["as_of", "symbol"]).reset_index(drop=True)


def validate_universe_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate provenance-bearing membership events."""
    missing = EVENT_REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")

    out = frame.copy()
    out["effective_date"] = pd.to_datetime(
        out["effective_date"], errors="coerce"
    ).dt.normalize()
    for column in ("symbol", "action", "source", "source_id"):
        out[column] = out[column].astype(str).str.strip()
    out["symbol"] = out["symbol"].str.upper()
    out["action"] = out["action"].str.lower()

    if out["effective_date"].isna().any():
        raise ValueError("Events contain invalid effective dates")
    if out[["symbol", "source", "source_id"]].eq("").any().any():
        raise ValueError("Events contain blank symbol or provenance fields")
    if (~out["action"].isin({"include", "exclude"})).any():
        raise ValueError("Events contain unsupported actions")
    if out.duplicated(["effective_date", "symbol"]).any():
        raise ValueError("Events contain duplicate effective_date/symbol rows")
    return out.sort_values(["effective_date", "symbol"]).reset_index(drop=True)


def ingest_universe_snapshots(input_path: str | Path, output_path: str | Path) -> Path:
    """Validate a CSV snapshot file and write its normalized representation."""
    frame = pd.read_csv(input_path)
    normalized = validate_universe_snapshots(frame)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output, index=False)
    return output


def ingest_universe_events(input_path: str | Path, output_path: str | Path) -> Path:
    """Validate a CSV event file and write its normalized representation."""
    frame = pd.read_csv(input_path)
    normalized = validate_universe_events(frame)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output, index=False)
    return output
