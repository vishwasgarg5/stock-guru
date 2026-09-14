from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import pandas as pd


@dataclass(frozen=True)
class UniverseCoverage:
    """Coverage and integrity summary for supplied PIT universe evidence."""

    earliest_date: str
    latest_date: str
    snapshot_dates: int
    snapshot_rows: int
    unique_symbols: int
    inclusion_events: int = 0
    exclusion_events: int = 0


def summarize_snapshots(snapshots: pd.DataFrame) -> UniverseCoverage:
    """Summarize supplied snapshots without assuming unsupported history."""
    required = {"as_of", "symbol"}
    if not required.issubset(snapshots.columns):
        raise ValueError(f"Missing snapshot columns: {sorted(required - set(snapshots.columns))}")
    df = snapshots[["as_of", "symbol"]].copy()
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce").dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.strip()
    if df["as_of"].isna().any() or df["symbol"].eq("").any():
        raise ValueError("Snapshots contain invalid dates or blank symbols")
    if df.duplicated(["as_of", "symbol"]).any():
        raise ValueError("Snapshots contain duplicate as_of/symbol rows")
    return UniverseCoverage(
        earliest_date=str(df["as_of"].min().date()),
        latest_date=str(df["as_of"].max().date()),
        snapshot_dates=int(df["as_of"].nunique()),
        snapshot_rows=int(len(df)),
        unique_symbols=int(df["symbol"].nunique()),
    )


def _validate_events(events: pd.DataFrame) -> pd.DataFrame:
    required = {"effective_date", "symbol", "action", "source", "source_id"}
    if not required.issubset(events.columns):
        raise ValueError(f"Missing event columns: {sorted(required - set(events.columns))}")
    df = events[list(required)].copy()
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce").dt.normalize()
    for column in ("symbol", "action", "source", "source_id"):
        df[column] = df[column].astype(str).str.strip()
    df["action"] = df["action"].str.lower()
    if df["effective_date"].isna().any():
        raise ValueError("Events contain invalid effective dates")
    if df[["symbol", "source", "source_id"]].eq("").any().any():
        raise ValueError("Events contain blank symbol or provenance fields")
    if (~df["action"].isin({"include", "exclude"})).any():
        raise ValueError("Events contain unsupported actions")
    if df.duplicated(["effective_date", "symbol"]).any():
        raise ValueError("Events contain duplicate effective_date/symbol rows")
    return df.sort_values(["effective_date", "symbol"]).reset_index(drop=True)


def summarize_events(events: pd.DataFrame) -> dict[str, object]:
    """Validate and summarize provenance-bearing inclusion/exclusion events."""
    df = _validate_events(events)
    return {
        "event_count": int(len(df)),
        "inclusion_events": int(df["action"].eq("include").sum()),
        "exclusion_events": int(df["action"].eq("exclude").sum()),
        "event_earliest_date": str(df["effective_date"].min().date()) if not df.empty else None,
        "event_latest_date": str(df["effective_date"].max().date()) if not df.empty else None,
        "event_unique_symbols": int(df["symbol"].nunique()),
        "event_provenance_complete": True,
    }


def build_coverage_report(
    snapshots: pd.DataFrame,
    events: pd.DataFrame | None = None,
    *,
    source_manifest: dict | None = None,
) -> dict:
    """Build a machine-readable report; no completeness is inferred."""
    coverage = summarize_snapshots(snapshots)
    report = asdict(coverage)
    if events is not None:
        report.update(summarize_events(events))
    if source_manifest is not None:
        required = {"dataset", "source_name", "source_url", "retrieved_at", "license_or_terms"}
        missing = required - set(source_manifest)
        if missing:
            raise ValueError(f"Missing source manifest fields: {sorted(missing)}")
        report["source"] = {
            "dataset": source_manifest["dataset"],
            "source_name": source_manifest["source_name"],
            "source_url": source_manifest["source_url"],
            "retrieved_at": source_manifest["retrieved_at"],
        }
        report["provenance_validated"] = True
    else:
        report["provenance_validated"] = False
    report["historical_completeness"] = "unknown"
    report["provenance_required"] = True
    return report


def save_coverage_report(
    snapshots: pd.DataFrame,
    output_path: str | Path,
    events: pd.DataFrame | None = None,
    *,
    source_manifest: dict | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_coverage_report(snapshots, events, source_manifest=source_manifest), indent=2),
        encoding="utf-8",
    )
    return output
