from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pandas as pd

from .universe_history import load_snapshots


REQUIRED_MANIFEST = {"dataset", "source_name", "source_url", "retrieved_at", "license_or_terms"}


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
    if "source_sha256" in manifest and not str(manifest["source_sha256"]).strip():
        raise ValueError("Source manifest source_sha256 must be nonblank when supplied")
    return manifest


def load_source_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Source manifest must be a JSON object")
    return validate_source_manifest(manifest)


def file_sha256(path: str | Path) -> str:
    """Return a stable SHA-256 fingerprint for an ingested source file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_historical_snapshots(snapshots: pd.DataFrame, *, expected_constituents: int | None = None, require_full_snapshot_size: bool = False) -> pd.DataFrame:
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


def snapshot_gap_diagnostics(snapshots: pd.DataFrame, *, threshold_days: int | None = None) -> dict[str, object]:
    """Report calendar gaps between supplied snapshot dates without inferring missing history."""
    dates = pd.Series(pd.to_datetime(snapshots["as_of"], errors="coerce").dt.normalize().drop_duplicates().sort_values().tolist())
    if len(dates) < 2:
        return {"snapshot_gap_count": 0, "max_gap_days": 0, "gaps": [], "threshold_days": threshold_days}
    gaps = dates.diff().dt.days.iloc[1:]
    records = []
    for idx, gap in gaps.items():
        if threshold_days is None or int(gap) > threshold_days:
            records.append({"from": str(dates.iloc[idx - 1].date()), "to": str(dates.iloc[idx].date()), "gap_days": int(gap)})
    return {"snapshot_gap_count": len(records), "max_gap_days": int(gaps.max()), "gaps": records, "threshold_days": threshold_days}


def validate_historical_snapshot_file(snapshot_path: str | Path, manifest_path: str | Path, *, expected_constituents: int | None = None, require_full_snapshot_size: bool = False) -> pd.DataFrame:
    """Validate a snapshot CSV only when its provenance manifest is present."""
    load_source_manifest(manifest_path)
    snapshots = load_snapshots(snapshot_path)
    return validate_historical_snapshots(snapshots, expected_constituents=expected_constituents, require_full_snapshot_size=require_full_snapshot_size)


def validate_source_bundle(snapshot_path: str | Path, manifest_path: str | Path, *, expected_constituents: int | None = None, require_full_snapshot_size: bool = False, gap_threshold_days: int | None = None) -> dict[str, object]:
    """Validate a snapshot plus provenance and return an audit summary."""
    manifest = load_source_manifest(manifest_path)
    snapshots = validate_historical_snapshot_file(snapshot_path, manifest_path, expected_constituents=expected_constituents, require_full_snapshot_size=require_full_snapshot_size)
    actual_sha256 = file_sha256(snapshot_path)
    expected_sha256 = manifest.get("source_sha256")
    if expected_sha256 is not None and actual_sha256 != str(expected_sha256).strip():
        raise ValueError("Universe source fingerprint does not match manifest")
    counts = snapshots.groupby("as_of")["symbol"].nunique()
    return {"dataset": manifest["dataset"], "source_name": manifest["source_name"], "source_url": manifest["source_url"], "retrieved_at": manifest["retrieved_at"], "snapshot_dates": int(len(counts)), "earliest_date": str(snapshots["as_of"].min().date()), "latest_date": str(snapshots["as_of"].max().date()), "snapshot_rows": int(len(snapshots)), "unique_symbols": int(snapshots["symbol"].nunique()), "min_constituents": int(counts.min()), "max_constituents": int(counts.max()), "source_sha256": actual_sha256, "provenance_validated": True, "gap_diagnostics": snapshot_gap_diagnostics(snapshots, threshold_days=gap_threshold_days)}
