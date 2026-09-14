from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from .universe_history import load_snapshots


REQUIRED_MANIFEST = {
    "dataset",
    "source_name",
    "source_url",
    "retrieved_at",
    "license_or_terms",
}


def validate_source_manifest(manifest: dict) -> dict:
    """Validate provenance metadata before historical universe data is accepted."""
    missing = REQUIRED_MANIFEST - set(manifest)
    if missing:
        raise ValueError(f"Missing source manifest fields: {sorted(missing)}")
    if manifest["dataset"] != "nifty500_membership":
        raise ValueError("Source manifest dataset must be nifty500_membership")
    for key in REQUIRED_MANIFEST:
        if not str(manifest[key]).strip():
            raise ValueError(f"Source manifest field {key} must not be blank")
    return manifest


def load_source_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Source manifest must be a JSON object")
    return validate_source_manifest(manifest)


def validate_historical_snapshots(
    snapshots: pd.DataFrame,
    *,
    expected_constituents: int | None = None,
    require_full_snapshot_size: bool = False,
) -> pd.DataFrame:
    """Validate supplied historical snapshots without filling missing history."""
    clean = snapshots.copy()
    required = {"as_of", "symbol"}
    missing = required - set(clean.columns)
    if missing:
        raise ValueError(f"Missing snapshot columns: {sorted(missing)}")
    clean["as_of"] = pd.to_datetime(clean["as_of"], errors="coerce").dt.normalize()
    clean["symbol"] = clean["symbol"].astype(str).str.strip().str.upper()
    if clean["as_of"].isna().any():
        raise ValueError("Historical snapshots contain invalid dates")
    if clean["symbol"].eq("").any():
        raise ValueError("Historical snapshots contain blank symbols")
    if clean.duplicated(["as_of", "symbol"]).any():
        raise ValueError("Historical snapshots contain duplicate as_of/symbol rows")
    if require_full_snapshot_size and expected_constituents is None:
        raise ValueError("expected_constituents is required when full snapshot size is enforced")
    if expected_constituents is not None:
        counts = clean.groupby("as_of")["symbol"].nunique()
        if require_full_snapshot_size and (counts != expected_constituents).any():
            bad = {str(k.date()): int(v) for k, v in counts.items() if v != expected_constituents}
            raise ValueError(f"Historical snapshots are not full sets: {bad}")
    return clean.sort_values(["as_of", "symbol"]).reset_index(drop=True)


def validate_historical_snapshot_file(
    snapshot_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_constituents: int | None = None,
    require_full_snapshot_size: bool = False,
) -> pd.DataFrame:
    """Validate a snapshot CSV only when its provenance manifest is present."""
    load_source_manifest(manifest_path)
    snapshots = load_snapshots(snapshot_path)
    return validate_historical_snapshots(
        snapshots,
        expected_constituents=expected_constituents,
        require_full_snapshot_size=require_full_snapshot_size,
    )


def validate_source_bundle(
    snapshot_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_constituents: int | None = None,
    require_full_snapshot_size: bool = False,
) -> dict[str, object]:
    """Validate a snapshot plus its provenance and return an audit summary."""
    manifest = load_source_manifest(manifest_path)
    snapshots = validate_historical_snapshot_file(
        snapshot_path,
        manifest_path,
        expected_constituents=expected_constituents,
        require_full_snapshot_size=require_full_snapshot_size,
    )
    counts = snapshots.groupby("as_of")["symbol"].nunique()
    return {
        "dataset": manifest["dataset"],
        "source_name": manifest["source_name"],
        "source_url": manifest["source_url"],
        "retrieved_at": manifest["retrieved_at"],
        "snapshot_dates": int(len(counts)),
        "earliest_date": str(snapshots["as_of"].min().date()),
        "latest_date": str(snapshots["as_of"].max().date()),
        "snapshot_rows": int(len(snapshots)),
        "unique_symbols": int(snapshots["symbol"].nunique()),
        "min_constituents": int(counts.min()),
        "max_constituents": int(counts.max()),
        "provenance_validated": True,
    }
