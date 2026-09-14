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


def summarize_events(events: pd.DataFrame) -> dict[str, int]:
    """Count validated inclusion/exclusion events by action."""
    required = {"effective_date", "symbol", "action", "source", "source_id"}
    if not required.issubset(events.columns):
        raise ValueError(f"Missing event columns: {sorted(required - set(events.columns))}")
    actions = events["action"].astype(str).str.strip().str.lower()
    if (~actions.isin({"include", "exclude"})).any():
        raise ValueError("Events contain unsupported actions")
    return {
        "inclusion_events": int(actions.eq("include").sum()),
        "exclusion_events": int(actions.eq("exclude").sum()),
    }


def build_coverage_report(snapshots: pd.DataFrame, events: pd.DataFrame | None = None) -> dict:
    """Build a machine-readable report; no completeness is inferred."""
    coverage = summarize_snapshots(snapshots)
    report = asdict(coverage)
    if events is not None:
        report.update(summarize_events(events))
    report["historical_completeness"] = "unknown"
    report["provenance_required"] = True
    return report


def save_coverage_report(snapshots: pd.DataFrame, output_path: str | Path,
                         events: pd.DataFrame | None = None) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_coverage_report(snapshots, events), indent=2), encoding="utf-8")
    return output
